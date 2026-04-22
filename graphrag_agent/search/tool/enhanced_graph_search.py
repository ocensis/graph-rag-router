"""
增强版图检索：Graph 定位实体 → 关联 chunk → BM25 + 向量混合排序

相比原始 local_search：
1. 最终返回原始 chunk 文本（不只是实体描述）
2. 对关联 chunk 做 BM25 + 向量混合排序
3. 找不到实体时降级到全量 chunk 混合检索（类似 LinearRAG 的降级策略）
"""
import re
import math
import time
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graphrag_agent.config.prompts import LC_SYSTEM_PROMPT, LOCAL_SEARCH_CONTEXT_PROMPT
from graphrag_agent.config.settings import response_type, lc_description, LOCAL_SEARCH_SETTINGS
from graphrag_agent.search.tool.base import BaseSearchTool
from graphrag_agent.search.utils import VectorUtils
from graphrag_agent.utils.logger import get_logger
from graphrag_agent.utils.resilience import retry_sync, retry_async, async_timeout
from graphrag_agent.utils.llm_output_schemas import SufficiencyCheck
from graphrag_agent.utils.langfuse_client import get_langfuse_handler

logger = get_logger(__name__)


class EnhancedGraphSearchTool(BaseSearchTool):
    """增强版图检索工具"""

    def __init__(self):
        super().__init__(cache_dir="./cache/enhanced_graph_search")

        self.top_entities = LOCAL_SEARCH_SETTINGS.get("top_entities", 10)
        self.top_k = 5  # 最终返回的 chunk 数

        self._setup_chains()

    def _setup_chains(self):
        self.query_prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", LOCAL_SEARCH_CONTEXT_PROMPT),
        ])
        self.query_chain = self.query_prompt | self.llm | StrOutputParser()

        # Sufficiency check 也走 chain 模式，不要直接 llm.invoke——
        # 直接 invoke 从普通 Python 里调是 root run，跟 @observe context 联动不上，
        # 结果变成独立 trace（ThinkingCompatChatOpenAI）。chain.invoke 通过
        # RunnableConfig 系统传播 parent_run_id，能正确 nest。
        self.sufficiency_prompt_template = ChatPromptTemplate.from_messages([
            ("human", self._sufficiency_prompt),
        ])
        self.sufficiency_chain = self.sufficiency_prompt_template | self.llm | StrOutputParser()

    def _bm25_score(self, query: str, documents: List[Dict]) -> List[Dict]:
        """轻量 BM25 打分"""
        k1, b = 1.5, 0.75
        query_tokens = re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]', query.lower())
        if not query_tokens or not documents:
            return documents

        doc_tokens_list = [re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]', d.get("text", "").lower()) for d in documents]
        avg_dl = sum(len(dt) for dt in doc_tokens_list) / max(len(doc_tokens_list), 1)

        n_docs = len(documents)
        df = Counter()
        for dt in doc_tokens_list:
            for token in set(dt):
                df[token] += 1

        idf = {t: math.log((n_docs - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1) for t in query_tokens}

        for i, doc in enumerate(documents):
            dl = len(doc_tokens_list[i])
            tf = Counter(doc_tokens_list[i])
            score = sum(
                idf.get(t, 0) * tf.get(t, 0) * (k1 + 1) / (tf.get(t, 0) + k1 * (1 - b + b * dl / max(avg_dl, 1)) + 1e-8)
                for t in query_tokens
            )
            doc["bm25_score"] = score

        return documents

    def _hybrid_rank(self, query: str, query_embedding, chunks: List[Dict], top_k: int) -> List[Dict]:
        """RRF (Reciprocal Rank Fusion) 混合排序"""
        if not chunks:
            return []

        K = 60  # RRF 常数

        # 向量排序
        vec_scored = VectorUtils.rank_by_similarity(query_embedding, chunks, "embedding", len(chunks))
        vec_rank = {c["id"]: rank + 1 for rank, c in enumerate(vec_scored)}

        # BM25 排序
        bm25_scored = self._bm25_score(query, chunks)
        bm25_sorted = sorted(bm25_scored, key=lambda x: x.get("bm25_score", 0), reverse=True)
        bm25_rank = {c["id"]: rank + 1 for rank, c in enumerate(bm25_sorted)}

        # RRF 融合：score = 1/(K+vec_rank) + 1/(K+bm25_rank)
        for chunk in chunks:
            cid = chunk["id"]
            chunk["hybrid_score"] = 1.0 / (K + vec_rank.get(cid, len(chunks))) + 1.0 / (K + bm25_rank.get(cid, len(chunks)))

        return sorted(chunks, key=lambda x: x.get("hybrid_score", 0), reverse=True)[:top_k]

    def _retrieve_chunks(self, query: str, query_embedding, exclude_ids: set = None) -> List[Dict]:
        """单轮检索：图结构 + 降级到向量检索"""
        exclude_ids = exclude_ids or set()

        # 图结构检索
        try:
            scored_entities = self.graph.query(
                """CALL db.index.vector.queryNodes('vector', $k, $vec)
                YIELD node, score
                RETURN node.id AS id, node.description AS desc, score""",
                {"k": self.top_entities, "vec": query_embedding}
            )
        except Exception:
            scored_entities = []

        chunks = []
        if scored_entities:
            entity_ids = [e["id"] for e in scored_entities]
            related_chunks = self.graph.query("""
                MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
                WHERE e.id IN $entity_ids AND c.embedding IS NOT NULL
                RETURN DISTINCT c.id AS id, coalesce(c.original_text, c.text) AS text, c.embedding AS embedding
            """, {"entity_ids": entity_ids})

            # 过滤已检索过的
            chunks = [c for c in related_chunks if c["id"] not in exclude_ids]

        # 降级到向量检索
        if len(chunks) < self.top_k:
            try:
                vector_chunks = self.graph.query(
                    """CALL db.index.vector.queryNodes('chunk_vector', $k, $vec)
                    YIELD node, score
                    RETURN node.id AS id, coalesce(node.original_text, node.text) AS text, score""",
                    {"k": self.top_k * 3, "vec": query_embedding}
                )
                existing_ids = {c["id"] for c in chunks}
                for c in vector_chunks:
                    if c["id"] not in existing_ids and c["id"] not in exclude_ids:
                        chunks.append(c)
            except Exception:
                pass

        if chunks:
            return self._hybrid_rank(query, query_embedding, chunks, self.top_k)
        return []

    _sufficiency_prompt = """Based on the information collected so far, determine if we can answer the question.

Question: {query}

Information collected:
{info_summary}

Respond in this exact format:
SUFFICIENT: yes or no
MISSING: what information is still needed (if not sufficient)
SUBQUERY: a search query to find the missing information (if not sufficient)"""

    def _check_sufficiency(self, query: str, info_summary: str, lf_config: dict = None) -> Dict:
        """LLM 判断当前信息是否足够回答问题（Pydantic 校验输出）。
        走 sufficiency_chain（RunnableSequence）而不是 self.llm.invoke，
        让 callback 和 @observe context 正确 nest。"""
        try:
            content = retry_sync(max_retries=2, base_delay=1.0)(
                self.sufficiency_chain.invoke
            )(
                {"query": query, "info_summary": info_summary},
                config=lf_config if lf_config else None,
            )
            parsed = SufficiencyCheck.parse_llm_output(content)
            return parsed.model_dump()
        except Exception as e:
            logger.warning("充分性判断失败，默认 sufficient", extra={"error": str(e)})
            return {"sufficient": True, "missing": "", "subquery": ""}

    def search(self, query_input: Any, session_id: str = None,
               parent_config: dict = None) -> str:
        overall_start = time.time()

        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
        else:
            query = str(query_input)

        # 缓存
        cache_key = f"enhanced_graph:{query}"
        cached = self.cache_manager.get(cache_key)
        if cached:
            return cached

        # Langfuse config: parent_config 优先（嵌套进上游 trace），否则独立 trace
        if parent_config:
            lf_config = dict(parent_config)
        else:
            lf_config = {}
            handler = get_langfuse_handler()
            if handler is not None:
                lf_config = {
                    "callbacks": [handler],
                    "run_name": "enhanced_graph_search",
                    "metadata": {
                        "langfuse_session_id": session_id or "default",
                        "langfuse_tags": ["enhanced_graph_search"],
                    },
                }

        try:
            max_rounds = 3
            all_context_chunks = []
            used_chunk_ids = set()
            info_summary = ""

            for round_num in range(max_rounds):
                # 决定检索用的 query
                if round_num == 0:
                    search_query = query
                else:
                    # 用 LLM 判断缺什么信息，生成子查询
                    check = self._check_sufficiency(query, info_summary, lf_config=lf_config)
                    if check["sufficient"]:
                        break
                    search_query = check["subquery"] if check["subquery"] else query

                # 检索
                search_embedding = self.embeddings.embed_query(search_query)
                new_chunks = self._retrieve_chunks(search_query, search_embedding, used_chunk_ids)

                if not new_chunks:
                    break

                # 累积
                for c in new_chunks:
                    used_chunk_ids.add(c["id"])
                    all_context_chunks.append(c)

                # 更新信息摘要
                new_text = "\n".join([c.get("text", "") for c in new_chunks])
                info_summary += f"\n--- Round {round_num + 1} ---\n{new_text}"

                # 第一轮后检查充分性
                if round_num == 0:
                    check = self._check_sufficiency(query, info_summary, lf_config=lf_config)
                    if check["sufficient"]:
                        break

            # 生成最终回答
            if all_context_chunks:
                # 获取实体关系信息作为补充
                try:
                    scored_entities = self.graph.query(
                        """CALL db.index.vector.queryNodes('vector', $k, $vec)
                        YIELD node, score
                        RETURN node.id AS id, node.description AS desc, score""",
                        {"k": 5, "vec": self.embeddings.embed_query(query)}
                    )
                except Exception:
                    scored_entities = []

                context = self._build_context(query, scored_entities, all_context_chunks)
                answer = self._generate_answer(query, context, lf_config=lf_config)
            else:
                answer = "未找到相关信息。"

            self.cache_manager.set(cache_key, answer)
            return answer

        except Exception as e:
            logger.error("Graph 搜索失败", extra={"error": str(e), "query": query[:50]})
            return f"搜索出错: {e}"

    def _build_context(self, query: str, entities: List[Dict], chunks: List[Dict]) -> str:
        """构建上下文：原始 chunk 文本为主，实体信息为辅"""
        parts = []

        # 主体：chunk 原文
        parts.append("### 相关文档段落")
        for c in chunks:
            parts.append(c.get("text", ""))

        # 补充：实体关系信息（简短）
        if entities:
            entity_ids = [e["id"] for e in entities[:5]]
            relations = self.graph.query("""
                MATCH (e1:__Entity__)-[r]-(e2:__Entity__)
                WHERE e1.id IN $ids AND e2.id IN $ids AND e1.id < e2.id
                RETURN e1.id AS src, type(r) AS rel, e2.id AS tgt,
                       CASE WHEN r.description IS NULL THEN '' ELSE r.description END AS desc
                LIMIT 10
            """, {"ids": entity_ids})

            if relations:
                parts.append("\n### 实体关系")
                for r in relations:
                    parts.append(f"- {r['src']} --{r['rel']}--> {r['tgt']}: {r['desc']}")

        return "\n\n".join(parts)

    def _generate_answer(self, query: str, context: str, lf_config: dict = None) -> str:
        return retry_sync(max_retries=2, base_delay=1.0)(
            self.query_chain.invoke
        )(
            {"context": context, "input": query, "response_type": response_type},
            config=lf_config if lf_config else None,
        )

    # ==================== 异步版本 ====================

    async def _async_retrieve_chunks(self, query: str, query_embedding, exclude_ids: set = None) -> List[Dict]:
        """异步检索：实体 HNSW 和 chunk 向量 HNSW 两路并行"""
        exclude_ids = exclude_ids or set()

        # 两路并行召回
        entity_task = self.async_db_query(
            """CALL db.index.vector.queryNodes('vector', $k, $vec)
            YIELD node, score
            RETURN node.id AS id, node.description AS desc, score""",
            {"k": self.top_entities, "vec": query_embedding}
        )
        chunk_vector_task = self.async_db_query(
            """CALL db.index.vector.queryNodes('chunk_vector', $k, $vec)
            YIELD node, score
            RETURN node.id AS id, coalesce(node.original_text, node.text) AS text, score""",
            {"k": self.top_k * 3, "vec": query_embedding}
        )

        # 并行等待两路结果
        try:
            scored_entities, fallback_chunks = await asyncio.gather(
                entity_task, chunk_vector_task, return_exceptions=True
            )
        except Exception:
            scored_entities, fallback_chunks = [], []

        if isinstance(scored_entities, Exception):
            scored_entities = []
        if isinstance(fallback_chunks, Exception):
            fallback_chunks = []

        # 实体 → 关联 chunk
        chunks = []
        if scored_entities:
            entity_ids = [e["id"] for e in scored_entities]
            related_chunks = await self.async_db_query(
                """MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
                WHERE e.id IN $entity_ids AND c.embedding IS NOT NULL
                RETURN DISTINCT c.id AS id, coalesce(c.original_text, c.text) AS text, c.embedding AS embedding""",
                {"entity_ids": entity_ids}
            )
            chunks = [c for c in related_chunks if c["id"] not in exclude_ids]

        # 合并 chunk 向量召回的结果
        if len(chunks) < self.top_k and fallback_chunks:
            existing_ids = {c["id"] for c in chunks}
            for c in fallback_chunks:
                if c["id"] not in existing_ids and c["id"] not in exclude_ids:
                    chunks.append(c)

        if chunks:
            return self._hybrid_rank(query, query_embedding, chunks, self.top_k)
        return []

    async def _async_check_sufficiency(self, query: str, info_summary: str) -> Dict:
        """异步版充分性判断（Pydantic 校验输出 + timeout）"""
        prompt = self._sufficiency_prompt.format(query=query, info_summary=info_summary)
        try:
            result = await async_timeout(self.llm.ainvoke(prompt), timeout_seconds=30)
            if result is None:
                logger.warning("充分性判断超时，默认 sufficient")
                return {"sufficient": True, "missing": "", "subquery": ""}
            content = result.content if hasattr(result, 'content') else str(result)
            parsed = SufficiencyCheck.parse_llm_output(content)
            return parsed.model_dump()
        except Exception as e:
            logger.warning("充分性判断失败，默认 sufficient", extra={"error": str(e)})
            return {"sufficient": True, "missing": "", "subquery": ""}

    async def async_search(self, query_input: Any) -> str:
        """
        异步版增强图检索
        - embedding / Neo4j / LLM 全部走 async
        - 实体 HNSW 和 chunk 向量 HNSW 两路并行召回
        """
        overall_start = time.time()

        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
        else:
            query = str(query_input)

        cache_key = f"enhanced_graph:{query}"
        cached = self.cache_manager.get(cache_key)
        if cached:
            return cached

        try:
            max_rounds = 3
            all_context_chunks = []
            used_chunk_ids = set()
            info_summary = ""

            for round_num in range(max_rounds):
                if round_num == 0:
                    search_query = query
                else:
                    check = await self._async_check_sufficiency(query, info_summary)
                    if check["sufficient"]:
                        break
                    search_query = check["subquery"] if check["subquery"] else query

                # 异步 embedding（带 timeout，超时则跳过本轮）
                search_embedding = await async_timeout(
                    self.embeddings.aembed_query(search_query), timeout_seconds=30
                )
                if search_embedding is None:
                    logger.warning("embedding 超时，跳过本轮检索", extra={"round": round_num})
                    break
                # 异步两路并行检索
                new_chunks = await self._async_retrieve_chunks(search_query, search_embedding, used_chunk_ids)

                if not new_chunks:
                    break

                for c in new_chunks:
                    used_chunk_ids.add(c["id"])
                    all_context_chunks.append(c)

                new_text = "\n".join([c.get("text", "") for c in new_chunks])
                info_summary += f"\n--- Round {round_num + 1} ---\n{new_text}"

                if round_num == 0:
                    check = await self._async_check_sufficiency(query, info_summary)
                    if check["sufficient"]:
                        break

            if all_context_chunks:
                # 异步获取实体关系信息
                try:
                    final_embedding = await async_timeout(
                        self.embeddings.aembed_query(query), timeout_seconds=30
                    )
                    if final_embedding is None:
                        raise TimeoutError("final embedding 超时")
                    scored_entities = await self.async_db_query(
                        """CALL db.index.vector.queryNodes('vector', $k, $vec)
                        YIELD node, score
                        RETURN node.id AS id, node.description AS desc, score""",
                        {"k": 5, "vec": final_embedding}
                    )
                except Exception:
                    scored_entities = []

                context = self._build_context(query, scored_entities, all_context_chunks)
                # 异步 LLM 生成（带 retry + timeout）
                @retry_async(max_retries=2, base_delay=1.0)
                async def _gen():
                    return await async_timeout(
                        self.query_chain.ainvoke({
                            "context": context, "input": query, "response_type": response_type,
                        }),
                        timeout_seconds=60,
                        default="LLM 生成超时，请重试。"
                    )
                answer = await _gen()
            else:
                answer = "未找到相关信息。"

            self.cache_manager.set(cache_key, answer)
            return answer

        except Exception as e:
            logger.error("Graph 异步搜索失败", extra={"error": str(e), "query": query[:50]})
            return f"搜索出错: {e}"

    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        return {"low_level": [], "high_level": []}

    def get_tool(self) -> BaseTool:
        class EnhancedGraphTool(BaseTool):
            name: str = "enhanced_graph_retriever"
            description: str = lc_description

            def _run(self_tool, query: Any) -> str:
                return self.search(query)

            async def _arun(self_tool, query: Any) -> str:
                return await self.async_search(query)

        return EnhancedGraphTool()
