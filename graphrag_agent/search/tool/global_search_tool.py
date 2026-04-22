"""
Global Search Tool —— 基于 Leiden 社区摘要的跨文档聚合检索

适合"枚举/聚合"类查询（"哪些论文用了 X" / "有哪些主流方法"），
传统 Naive/Agentic top-K chunk 检索在这类问题上结构性失效。

Map-Reduce 架构:
  1. 拉 top-N 大社区（按摘要长度降序过滤）
  2. Map 阶段并行：每个社区单独判断是否和 query 相关，抽取相关信息
  3. Reduce 阶段：LLM 整合所有中间结果生成最终答案
"""
import time
import concurrent.futures
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool

from graphrag_agent.config.prompts import (
    MAP_SYSTEM_PROMPT,
    REDUCE_SYSTEM_PROMPT,
    GLOBAL_SEARCH_MAP_PROMPT,
    GLOBAL_SEARCH_REDUCE_PROMPT,
)
from graphrag_agent.config.settings import response_type
from graphrag_agent.search.tool.base import BaseSearchTool
from graphrag_agent.utils.logger import get_logger
from graphrag_agent.utils.resilience import retry_sync
from graphrag_agent.utils.langfuse_client import get_langfuse_handler

logger = get_logger(__name__)


class GlobalSearchTool(BaseSearchTool):
    """基于社区摘要的全局检索工具"""

    def __init__(self,
                 community_level: int = 0,
                 max_communities: int = 30,
                 map_workers: int = 10):
        """
        参数:
            community_level: 使用哪层社区（0 最细，层级越大越粗）
            max_communities: 最多处理多少个社区（按摘要长度降序）
            map_workers: Map 阶段并行度
        """
        super().__init__(cache_dir="./cache/global_search")
        self.community_level = community_level
        self.max_communities = max_communities
        self.map_workers = map_workers
        self._setup_chains()

    def _setup_chains(self):
        map_prompt = ChatPromptTemplate.from_messages([
            ("system", MAP_SYSTEM_PROMPT),
            ("human", GLOBAL_SEARCH_MAP_PROMPT),
        ])
        self.map_chain = map_prompt | self.llm | StrOutputParser()

        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", REDUCE_SYSTEM_PROMPT),
            ("human", GLOBAL_SEARCH_REDUCE_PROMPT),
        ])
        self.reduce_chain = reduce_prompt | self.llm | StrOutputParser()

    # 宽度信号：命中则用更高 level（粗粒度，社区少、更快）
    _BROAD_SCOPE_KEYWORDS = [
        "summarize entire", "summarize all", "across all", "across these",
        "main themes", "main approaches", "main areas", "overall landscape",
        "big picture", "overview of the", "trends in",
        "整个语料库", "所有论文", "全局", "总览", "大主题",
    ]

    @classmethod
    def _detect_level(cls, query: str, default_level: int, max_level: int = 4) -> int:
        """按 query 语义动态决定使用哪层。越宽的问题用越高的 level。"""
        low = (query or "").lower()
        for kw in cls._BROAD_SCOPE_KEYWORDS:
            if kw in low:
                return max_level   # 最粗粒度
        return default_level

    # Hybrid 排序系数（和 GraphPath 保持一致）：
    # score = entity_hits × α + vec_sim × β + log(weight+1) × γ
    _HYBRID_ALPHA = 3.0
    _HYBRID_BETA = 2.0
    _HYBRID_GAMMA = 0.1

    def _get_top_communities(self, level: int = None, query: str = "",
                             entities: List[str] = None) -> List[Dict[str, Any]]:
        """
        取指定 level 的 top-N 社区，Hybrid 融合排序：
          score = entity_hits × α + vec_sim × β + log(weight+1) × γ

        参数：
          query:    用户 query，用于 embed 算 vec_sim。空字符串时退化到 entity + weight
          entities: query 抽出的实体 list，用于 entity_hits 维度。None 等价空列表
          level:    社区层级（None 用默认 self.community_level）

        前置条件：需先跑 scripts/maintenance/backfill_community_hybrid.py
                  给社区补 c.embedding 和 c.entities_kwd 两个字段。
        """
        use_level = self.community_level if level is None else level
        entities = entities or []

        # Embed query
        q_vec = None
        if query:
            try:
                from graphrag_agent.models.get_models import get_embeddings_model
                emb = get_embeddings_model()
                q_vec = emb.embed_query(query)
            except Exception as e:
                logger.warning(f"[global] query embed failed: {e}")

        # Hybrid Cypher（需要 c.embedding 和 c.entities_kwd 字段都存在）
        if q_vec is not None:
            try:
                rows = self.graph.query(
                    """
                    MATCH (c:__Community__)
                    WHERE c.level = $level
                      AND c.embedding IS NOT NULL
                      AND c.full_content IS NOT NULL
                      AND size(c.full_content) > 200
                    WITH c,
                         CASE WHEN size($entities) > 0 AND c.entities_kwd IS NOT NULL
                              THEN size([e IN c.entities_kwd WHERE e IN $entities])
                              ELSE 0 END AS entity_hits,
                         vector.similarity.cosine(c.embedding, $q_vec) AS vec_sim,
                         log(toFloat(coalesce(c.weight, 1)) + 1.0) AS weight_log
                    WITH c, entity_hits, vec_sim, weight_log,
                         entity_hits * $alpha + vec_sim * $beta + weight_log * $gamma AS score
                    ORDER BY score DESC
                    LIMIT $limit
                    RETURN c.id AS community_id,
                           c.full_content AS content,
                           entity_hits, vec_sim, score
                    """,
                    params={
                        "level": use_level,
                        "entities": entities,
                        "q_vec": q_vec,
                        "alpha": self._HYBRID_ALPHA,
                        "beta": self._HYBRID_BETA,
                        "gamma": self._HYBRID_GAMMA,
                        "limit": self.max_communities,
                    },
                )
                if rows:
                    top = [f"{r['score']:.2f}(e={r['entity_hits']},v={r['vec_sim']:.2f})" for r in rows[:3]]
                    logger.info(f"[global] hybrid level={use_level} got {len(rows)}, top: {', '.join(top)}")
                    return rows
            except Exception as e:
                logger.warning(f"[global] hybrid Cypher failed: {e}, fallback to legacy")

        # Fallback：没 embedding / hybrid 失败 → 退化到老的 content_size 排序
        logger.info(f"[global] fallback: content_size ordering at level={use_level}")
        rows = self.graph.query(
            """
            MATCH (c:__Community__)
            WHERE c.level = $level AND c.full_content IS NOT NULL
              AND size(c.full_content) > 200
            RETURN c.id AS community_id,
                   c.full_content AS content,
                   size(c.full_content) AS content_size
            ORDER BY content_size DESC
            LIMIT $limit
            """,
            params={"level": use_level, "limit": self.max_communities},
        )
        if not rows and use_level > 0:
            logger.info(f"[global] level={use_level} 无摘要，fallback 到 level 0")
            rows = self.graph.query(
                """
                MATCH (c:__Community__)
                WHERE c.level = 0 AND c.full_content IS NOT NULL
                  AND size(c.full_content) > 200
                RETURN c.id AS community_id,
                       c.full_content AS content,
                       size(c.full_content) AS content_size
                ORDER BY content_size DESC
                LIMIT $limit
                """,
                params={"limit": self.max_communities},
            )
        return rows

    def _map_one(self, query: str, community: Dict, lf_config: Dict) -> Dict:
        """处理一个社区（并行执行）"""
        try:
            content = retry_sync(max_retries=2, base_delay=1.0)(
                self.map_chain.invoke
            )(
                {"question": query, "context_data": community["content"][:4000]},
                config=lf_config if lf_config else None,
            )
            return {"community_id": community["community_id"], "response": content}
        except Exception as e:
            logger.warning(f"Map 失败: {e}", extra={"error": str(e)})
            return {"community_id": community["community_id"], "response": ""}

    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        return {"low_level": [], "high_level": []}

    def search(self, query_input: Any, session_id: Optional[str] = None,
               parent_config: Optional[Dict] = None) -> str:
        """
        Community 检索 —— RAGFlow 模式重构版。

        之前：map-reduce，每个社区一次 LLM (30+1 calls, ~30min)。
        现在：Cypher 取 top-N 社区（按 weight + level 预计算字段，0 LLM call）
              + 1 次 LLM synthesize（reduce，保留原有 prompt）。
        总 LLM calls: 31 → 1 (降 97%)，延迟 min → sec。

        保持 search(query, session_id, parent_config) -> str 接口不变，
        下游 GraphAgent / hybrid_agent / deep_research_tool 无感知。
        """
        overall_start = time.time()

        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
        else:
            query = str(query_input)

        cache_key = f"global:{query}"
        cached = self.cache_manager.get(cache_key)
        if cached:
            return cached

        # parent_config 优先（嵌套进 Router/Agent 的 trace），否则独立 trace
        if parent_config:
            lf_config = dict(parent_config)
        else:
            lf_config = {}
            handler = get_langfuse_handler()
            if handler is not None:
                lf_config = {
                    "callbacks": [handler],
                    "run_name": "global_search",
                    "metadata": {
                        "langfuse_session_id": session_id or "default",
                        "langfuse_tags": ["global_search", "cypher_retrieval"],
                    },
                }

        try:
            # Step 1: 决定社区 level（宽域 query 用高 level 粗粒度）
            chosen_level = self._detect_level(query, default_level=self.community_level)
            # 带 query 触发 Hybrid 排序（entity_hits + vec_sim + weight）。
            # 这里不抽实体——GlobalSearchTool 的历史用户没传实体，
            # 传空列表等价于"纯 vec_sim + weight 排序"，仍然 query-relevant。
            communities = self._get_top_communities(level=chosen_level, query=query)
            logger.info(
                f"[global] Cypher 取 {len(communities)} 个社区 (level={chosen_level}, "
                f"default was {self.community_level})",
                extra={"component": "global_search", "level": chosen_level},
            )
            if not communities:
                return "未找到任何社区数据。"

            # Step 2: 直接拼社区 content（0 LLM）。每条取前 4000 字符避免单社区过长挤压预算
            retrieve_time = time.time() - overall_start
            report_data = "\n\n---\n\n".join(
                f"[Community {c['community_id']}]\n{(c.get('content') or '')[:4000]}"
                for c in communities[:20]  # 只给 reduce 看前 20 个，避免 context 爆
            )

            # Step 3: 单次 LLM synthesize（reduce prompt 复用，不改语义）
            reduce_start = time.time()
            answer = retry_sync(max_retries=2, base_delay=1.0)(
                self.reduce_chain.invoke
            )(
                {"report_data": report_data, "question": query, "response_type": response_type},
                config=lf_config if lf_config else None,
            )
            reduce_time = time.time() - reduce_start
            total_time = time.time() - overall_start
            logger.info(
                f"[global] 完成 Cypher {retrieve_time:.2f}s + Reduce {reduce_time:.1f}s, "
                f"使用 {len(communities[:20])}/{len(communities)} 社区",
                extra={"component": "global_search", "elapsed": round(total_time, 2)},
            )

            self.cache_manager.set(cache_key, answer)
            return answer

        except Exception as e:
            logger.error(f"[global] 失败: {e}", extra={"error": str(e)})
            return f"Global search 出错: {e}"

    def get_tool(self) -> BaseTool:
        class GlobalRetrievalTool(BaseTool):
            name: str = "global_search"
            description: str = "跨文档聚合查询，适合'哪些文档...' '有多少 X'这类需要汇总多篇信息的问题"

            def _run(self_tool, query: Any) -> str:
                return self.search(query)

            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError

        return GlobalRetrievalTool()
