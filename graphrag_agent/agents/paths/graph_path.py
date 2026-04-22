"""
Graph Path —— 实体关系明确 / 因果链 / 全局汇总场景，覆盖 15-20%

对应 GraphRAG 的两类查询：
  - Local  (实体邻域)：问题锁定到 1-3 个实体，要邻居/关系
  - Global (社区级)：问题是"整个语料库"的主题/趋势

路径内部再做一次轻判分（local vs global）：
  - 含 "summarize/overall/across all/trends/common/main approaches" → global
  - 其他 → local（默认）

本条 path 产出答案基于：graph_lookup (邻域) + hybrid grounding chunk。
Global 用 GlobalSearchTool 的 map-reduce。
"""
from __future__ import annotations

import re
from typing import Optional, Dict, TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graphrag_agent.models.get_models import get_llm_model
from graphrag_agent.search.tool.primitives import (
    HybridSearchTool,
    GraphLookupTool,
    _get_graph,
    _get_embeddings,
)
# GlobalSearchTool 已弃用：原 map-reduce 架构改为统一 Cypher 社区检索
from graphrag_agent.agents.paths.context_packer import pack_context
from graphrag_agent.utils.logger import get_logger
from graphrag_agent.utils.resilience import retry_sync

logger = get_logger(__name__)


# ==================== State ====================

class GraphPathState(TypedDict, total=False):
    query: str
    scope: str           # "narrow" | "broad"，决定 community level 和 top_k
    entities: list       # extracted named entities
    graph_ctx: str       # from graph_lookup
    chunk_ctx: str       # from hybrid_search
    community_ctx: str   # from community summary lookup (entity-pivoted + weight fallback)
    answer: str


# ==================== Prompts ====================

ENTITY_EXTRACT_PROMPT = """Extract up to 3 named entities (methods, models, datasets, papers) from the question.

Question: {query}

Output: comma-separated names only, no explanation.
Examples:
- "What does UniDoc-RL use for retrieval?" → UniDoc-RL
- "How do VISOR and MM-Doc-R1 differ?" → VISOR, MM-Doc-R1
- "Which papers use GRPO?" → GRPO

Entities:"""


GRAPH_ANSWER_PROMPT = """You are answering a question about RAG research papers using THREE sources of evidence from the knowledge graph:

**IMPORTANT: Respond in English.**

## Question
{query}

## Context (community summaries + entity graph + text chunks)
{packed_ctx}

## Rules
- Start with the direct answer in the first sentence. No "### Overview" intro.
- Prefer concrete facts from CHUNKS; use GRAPH relations to verify and enrich; use COMMUNITY summaries for high-level themes.
- For relational/multi-entity questions, cite the specific relation types (e.g. USES, EVALUATED_ON).
- Be concise. Plain paragraphs.
- If context is insufficient, say "I don't know".
"""


# 判 scope 宽度的触发词——命中说明是跨文档聚合类问题，
# community 检索用更粗粒度（level 4）+ 实体不足时按 weight fallback
_BROAD_SCOPE_TRIGGERS = [
    "summarize", "summary of", "overall", "overview of",
    "across all", "across the", "across these", "across papers",
    "trends", "common approach", "common technique", "main approach", "main theme",
    "主要", "整体", "所有论文", "纵观", "总结",
]


def _detect_scope(query: str) -> str:
    """
    判断 query 的 scope 宽度。
      - "narrow": 实体锁定型问题，用 level 0 社区（细粒度，实体邻域为主）
      - "broad":  跨文档聚合问题，用 level 4 社区（粗粒度 + weight fallback）
    注意：从原来的 local/global 两条路径合并为一条——都走 local pipeline，只是参数不同。
         这对齐 RAGFlow 工程做法：社区检索就是一次 Cypher DB 查询，不再 map-reduce。
    """
    low = (query or "").lower()
    for k in _BROAD_SCOPE_TRIGGERS:
        if k in low:
            return "broad"
    return "narrow"


# scope → (community_level, community_top_n, chunk_top_k) 参数映射
_SCOPE_PARAMS = {
    "narrow": {"level": 0, "community_top_n": 3, "chunk_top_k": 5},
    "broad":  {"level": 4, "community_top_n": 8, "chunk_top_k": 8},
}


# ==================== Path ====================

class GraphPath:
    """Entity-relation / global-aggregation retrieval path."""

    def __init__(self):
        self.llm = get_llm_model()
        self.hybrid_tool = HybridSearchTool()
        self.graph_tool = GraphLookupTool()
        # 注意：已移除 GlobalSearchTool 依赖。
        # 原 Global Map-Reduce 架构（每 query 31 次 LLM call、~30min 延迟）是 MS GraphRAG
        # 研究型做法；这里改成 RAGFlow 工业做法——community 检索就是 Cypher 查询
        # （entity-pivoted + weight fallback），0 次 LLM call，延迟毫秒级。

        self.entity_extract_chain = (
            ChatPromptTemplate.from_messages([("human", ENTITY_EXTRACT_PROMPT)])
            | self.llm | StrOutputParser()
        )
        self.answer_chain = (
            ChatPromptTemplate.from_messages([("human", GRAPH_ANSWER_PROMPT)])
            | self.llm | StrOutputParser()
        )

        self._graph = self._build_graph()

    # ------- helpers -------

    @staticmethod
    def _scoped(config: Optional[Dict], name: str) -> Optional[Dict]:
        if not config:
            return config
        c = dict(config)
        c["run_name"] = name
        return c

    # ------- nodes -------

    def _node_detect_scope(self, state: GraphPathState) -> GraphPathState:
        """
        判 scope 决定 community level：
          - narrow: 实体锁定型 → level 0 细粒度
          - broad:  跨文档聚合型 → level 4 粗粒度 + weight fallback
        """
        scope = _detect_scope(state["query"])
        logger.info(f"[graph_path] scope={scope}", extra={"query": state["query"][:60]})
        return {"scope": scope}

    def _node_extract_entities(self, state: GraphPathState, config: Dict = None) -> GraphPathState:
        """
        抽 query 里的实体。broad scope 下可能抽不到——后续 community 检索会
        自动 fallback 到"按 weight 取 top-N"，不靠实体。
        """
        try:
            raw = retry_sync(max_retries=2, base_delay=0.5)(
                self.entity_extract_chain.invoke
            )(
                {"query": state["query"]},
                config=self._scoped(config, "extract_entities"),
            )
            ents = [e.strip() for e in raw.split(",") if e.strip()][:3]
        except Exception as e:
            logger.warning(f"[graph_path] entity extract failed: {e}")
            ents = []
        return {"entities": ents}

    def _node_retrieve(self, state: GraphPathState, config: Dict = None) -> GraphPathState:
        """
        三路证据并收（local/broad 统一 pipeline）：
          1. entity graph_lookup      —— 实体邻居 + 关系（broad 无实体时跳过）
          2. hybrid chunks            —— 原文 grounding（top_k 随 scope）
          3. community summaries      —— 实体命中优先 + weight fallback，**纯 Cypher，0 LLM**

        这替代了原来的 GlobalSearchTool map-reduce（31 次 LLM/query），对齐 RAGFlow。
        """
        scope = state.get("scope", "narrow")
        params = _SCOPE_PARAMS.get(scope, _SCOPE_PARAMS["narrow"])
        entities = state.get("entities", [])

        # Path 1: entity neighborhood
        snippets = []
        for ent in entities:
            try:
                snippets.append(
                    self.graph_tool.invoke(
                        {"entity_name": ent, "max_neighbors": 8},
                        config=self._scoped(config, f"graph_lookup({ent[:30]})"),
                    )
                )
            except Exception as e:
                logger.warning(f"[graph_path] lookup({ent}) failed: {e}")
        graph_ctx = "\n\n".join(snippets) if snippets else "(no entity neighborhoods)"

        # Path 2: hybrid chunks
        try:
            chunk_ctx = self.hybrid_tool.invoke(
                {"query": state["query"], "top_k": params["chunk_top_k"]},
                config=self._scoped(config, "hybrid_grounding"),
            )
        except Exception as e:
            chunk_ctx = f"(hybrid failed: {e})"

        # Path 3: community summaries (hybrid ranking)
        community_ctx = self._fetch_community_summaries(
            state["query"],
            entities,
            top_n=params["community_top_n"],
            level=params["level"],
        )

        return {"graph_ctx": graph_ctx, "chunk_ctx": chunk_ctx, "community_ctx": community_ctx}

    # 混合排序的权重系数。写成类属性方便后面调参 / 写进 .env
    # score = entity_hits * α + vec_sim * β + log(weight+1) * γ
    _HYBRID_ALPHA = 3.0   # entity 命中权重（强信号）
    _HYBRID_BETA = 2.0    # 向量相似度权重
    _HYBRID_GAMMA = 0.1   # 社区 weight 轻微 bias（tiebreaker）

    def _fetch_community_summaries(self, query: str, entities: List[str],
                                   top_n: int = 3, level: int = 0) -> str:
        """
        Hybrid community retrieval（替代之前的 entity-pivoted / weight-fallback 两分支）。

        三维融合排序：
          score = entity_hits × α + vec_sim × β + log(weight+1) × γ

        - entity_hits: query 实体命中社区 entities_kwd 的数量（narrow 场景主导）
        - vec_sim:    query embedding 和社区 embedding 的 cosine（broad / 实体抽偏时兜底）
        - weight_log: 社区大小的对数（平手时偏向信息多的社区）

        无论 narrow/broad，都走同一条 Cypher。没有 embedding 或 entities_kwd 的社区
        自动被 WHERE 过滤掉（需要先跑 scripts/maintenance/backfill_community_hybrid.py）。
        """
        g = _get_graph()

        # Embed query（1 次轻量 call，~1ms）
        try:
            emb = _get_embeddings()
            q_vec = emb.embed_query(query)
        except Exception as e:
            logger.warning(f"[graph_path] query embed failed: {e}")
            q_vec = None

        # 没 embedding 就降级到纯 entity-pivoted（兜底）
        if q_vec is None:
            return self._fetch_community_entity_only(entities, top_n, level)

        try:
            rows = g.query(
                """
                MATCH (c:__Community__)
                WHERE c.level = $level
                  AND c.embedding IS NOT NULL
                  AND (c.full_content IS NOT NULL OR c.summary IS NOT NULL)
                WITH c,
                     CASE WHEN size($entities) > 0 AND c.entities_kwd IS NOT NULL
                          THEN size([e IN c.entities_kwd WHERE e IN $entities])
                          ELSE 0 END AS entity_hits,
                     vector.similarity.cosine(c.embedding, $q_vec) AS vec_sim,
                     log(toFloat(coalesce(c.weight, 1)) + 1.0) AS weight_log
                WITH c, entity_hits, vec_sim, weight_log,
                     entity_hits * $alpha + vec_sim * $beta + weight_log * $gamma AS score
                ORDER BY score DESC
                LIMIT $top_n
                RETURN c.id AS id,
                       coalesce(c.full_content, c.summary) AS content,
                       entity_hits, vec_sim, score
                """,
                {
                    "level": level,
                    "entities": entities,
                    "q_vec": q_vec,
                    "alpha": self._HYBRID_ALPHA,
                    "beta": self._HYBRID_BETA,
                    "gamma": self._HYBRID_GAMMA,
                    "top_n": top_n,
                },
            )
        except Exception as e:
            logger.warning(f"[graph_path] hybrid community lookup failed: {e}, fallback entity-only")
            return self._fetch_community_entity_only(entities, top_n, level)

        if rows:
            top_scores = [f"{r['score']:.2f}(e={r['entity_hits']}, v={r['vec_sim']:.2f})" for r in rows[:3]]
            logger.info(f"[graph_path] hybrid retrieve {len(rows)} communities, top scores: {', '.join(top_scores)}")

        return self._format_community_rows(rows)

    def _fetch_community_entity_only(self, entities: List[str], top_n: int, level: int) -> str:
        """Fallback：embedding 不可用时退化到老的 entity-pivoted 逻辑。"""
        if not entities:
            return ""
        g = _get_graph()
        try:
            rows = g.query(
                """
                UNWIND $names AS name
                MATCH (e:__Entity__)-[:IN_COMMUNITY]->(c:__Community__)
                WHERE (toLower(e.id) = toLower(name) OR toLower(e.id) CONTAINS toLower(name))
                  AND c.level = $level
                  AND (c.full_content IS NOT NULL OR c.summary IS NOT NULL)
                WITH c, count(DISTINCT e) AS entity_hits
                ORDER BY entity_hits DESC, c.weight DESC
                LIMIT $top_n
                RETURN c.id AS id,
                       coalesce(c.full_content, c.summary) AS content
                """,
                {"names": entities, "level": level, "top_n": top_n},
            )
            return self._format_community_rows(rows)
        except Exception as e:
            logger.warning(f"[graph_path] entity-only fallback failed: {e}")
            return ""

    @staticmethod
    def _format_community_rows(rows: List[Dict]) -> str:
        if not rows:
            return ""
        return "\n\n".join(f"[Community {r['id']}]\n{r['content']}" for r in rows)

    def _node_compose(self, state: GraphPathState, config: Dict = None) -> GraphPathState:
        """
        Token 预算管理：参考 MS GraphRAG LocalSearch 比例
          community : graph : chunks = 0.25 : 0.25 : 0.50
        broad query 下 community 信息更重要，但这里先不动比例——避免一次改动两个变量。
        """
        packed = pack_context(
            parts={
                "community": state.get("community_ctx") or "",
                "graph":     state.get("graph_ctx") or "",
                "chunks":    state.get("chunk_ctx") or "",
            },
            budget=6000,
            ratios={"community": 0.25, "graph": 0.25, "chunks": 0.50},
        )
        try:
            ans = retry_sync(max_retries=2, base_delay=0.5)(
                self.answer_chain.invoke
            )(
                {"query": state["query"], "packed_ctx": packed},
                config=self._scoped(config, "graph_compose"),
            )
        except Exception as e:
            logger.error(f"[graph_path] compose failed: {e}")
            ans = f"Graph path compose failed: {e}"
        return {"answer": ans}

    # ------- build -------

    def _build_graph(self):
        """
        合并后的单一线性 pipeline：
            detect_scope → extract_entities → retrieve → compose → END
        （原 local/global 条件分支已移除——scope 只影响参数，不再走不同路径）
        """
        g = StateGraph(GraphPathState)
        g.add_node("detect_scope", self._node_detect_scope)
        g.add_node("extract_entities", self._node_extract_entities)
        g.add_node("retrieve", self._node_retrieve)
        g.add_node("compose", self._node_compose)

        g.set_entry_point("detect_scope")
        g.add_edge("detect_scope", "extract_entities")
        g.add_edge("extract_entities", "retrieve")
        g.add_edge("retrieve", "compose")
        g.add_edge("compose", END)
        return g.compile()

    # ------- public -------

    def run(self, query: str,
            session_id: Optional[str] = None,
            parent_config: Optional[Dict] = None) -> str:
        init: GraphPathState = {"query": query}
        try:
            final = self._graph.invoke(init, config=parent_config)
            return final.get("answer", "Graph path produced no answer")
        except Exception as e:
            logger.error(f"[graph_path] graph invoke failed: {e}")
            return f"Graph path failed: {e}"

    def close(self):
        # 已无外部 tool 需要显式 close（hybrid_tool / graph_tool 是 BaseTool 无资源）
        pass
