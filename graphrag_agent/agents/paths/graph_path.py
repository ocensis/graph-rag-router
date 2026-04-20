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
)
from graphrag_agent.search.tool.global_search_tool import GlobalSearchTool
from graphrag_agent.agents.paths.context_packer import pack_context
from graphrag_agent.utils.logger import get_logger
from graphrag_agent.utils.resilience import retry_sync

logger = get_logger(__name__)


# ==================== State ====================

class GraphPathState(TypedDict, total=False):
    query: str
    mode: str            # "local" | "global"
    entities: list       # extracted named entities
    graph_ctx: str       # from graph_lookup
    chunk_ctx: str       # from hybrid_search
    community_ctx: str   # from community summary lookup (borrowed from MS GraphRAG)
    global_answer: str   # from GlobalSearchTool
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


# 判 global 的触发词
_GLOBAL_TRIGGERS = [
    "summarize", "summary of", "overall", "overview of",
    "across all", "across the", "across these", "across papers",
    "trends", "common approach", "common technique", "main approach", "main theme",
    "主要", "整体", "所有论文", "纵观", "总结",
]


def _detect_mode(query: str) -> str:
    low = query.lower()
    for k in _GLOBAL_TRIGGERS:
        if k in low:
            return "global"
    return "local"


# ==================== Path ====================

class GraphPath:
    """Entity-relation / global-aggregation retrieval path."""

    def __init__(self):
        self.llm = get_llm_model()
        self.hybrid_tool = HybridSearchTool()
        self.graph_tool = GraphLookupTool()
        self.global_tool = GlobalSearchTool()

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

    def _node_detect_mode(self, state: GraphPathState) -> GraphPathState:
        mode = _detect_mode(state["query"])
        logger.info(f"[graph_path] mode={mode}", extra={"query": state["query"][:60]})
        return {"mode": mode}

    def _node_extract_entities(self, state: GraphPathState, config: Dict = None) -> GraphPathState:
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

    def _node_local_retrieve(self, state: GraphPathState, config: Dict = None) -> GraphPathState:
        """
        三路证据并收：
          1. entity graph_lookup（每个实体的邻居 + 关系）
          2. hybrid chunks（原文段落 grounding）
          3. community summaries（从实体 IN_COMMUNITY 出去找高 rank 社区的摘要）
        """
        # Path 1: entity neighborhood
        snippets = []
        for ent in state.get("entities", []):
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
                {"query": state["query"], "top_k": 5},
                config=self._scoped(config, "hybrid_grounding"),
            )
        except Exception as e:
            chunk_ctx = f"(hybrid failed: {e})"

        # Path 3: community summaries (借鉴 MS GraphRAG LocalSearch 的 community_prop)
        community_ctx = self._fetch_community_summaries(state.get("entities", []))

        return {"graph_ctx": graph_ctx, "chunk_ctx": chunk_ctx, "community_ctx": community_ctx}

    def _fetch_community_summaries(self, entities: List[str], top_n: int = 3,
                                   level: int = 0) -> str:
        """
        找 entities 所属的社区，按 community_rank + weight 取 top_n，返回内容拼接
        默认用 level=0；对齐微软 GraphRAG Local Search 读 full_content（更丰富的上下文）。
        full_content 不可用时 fallback 到 summary。
        """
        if not entities:
            return ""
        try:
            g = _get_graph()
            rows = g.query(
                """
                UNWIND $names AS name
                MATCH (e:__Entity__)-[:IN_COMMUNITY]->(c:__Community__)
                WHERE (toLower(e.id) = toLower(name) OR toLower(e.id) CONTAINS toLower(name))
                  AND c.level = $level
                  AND (c.full_content IS NOT NULL OR c.summary IS NOT NULL)
                WITH c, count(DISTINCT e) AS entity_hits
                ORDER BY entity_hits DESC, c.community_rank DESC, c.weight DESC
                LIMIT $top_n
                RETURN c.id AS id,
                       coalesce(c.full_content, c.summary) AS content
                """,
                {"names": entities, "level": level, "top_n": top_n},
            )
        except Exception as e:
            logger.warning(f"[graph_path] community lookup failed: {e}")
            return ""

        if not rows:
            return ""
        out = []
        for r in rows:
            out.append(f"[Community {r['id']}]\n{r['content']}")
        return "\n\n".join(out)

    def _node_local_compose(self, state: GraphPathState, config: Dict = None) -> GraphPathState:
        # Token 预算管理：参考 MS GraphRAG LocalSearch 比例
        #   community : graph : chunks = 0.25 : 0.25 : 0.50
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

    def _node_global(self, state: GraphPathState, config: Dict = None) -> GraphPathState:
        try:
            ans = self.global_tool.search(
                state["query"],
                session_id=None,
                parent_config=self._scoped(config, "global_map_reduce"),
            )
        except Exception as e:
            logger.error(f"[graph_path] global failed: {e}")
            ans = f"Global search failed: {e}"
        return {"answer": ans}

    def _route(self, state: GraphPathState) -> str:
        return state.get("mode", "local")

    # ------- build -------

    def _build_graph(self):
        g = StateGraph(GraphPathState)
        g.add_node("detect_mode", self._node_detect_mode)
        g.add_node("extract_entities", self._node_extract_entities)
        g.add_node("local_retrieve", self._node_local_retrieve)
        g.add_node("local_compose", self._node_local_compose)
        g.add_node("global_search", self._node_global)

        g.set_entry_point("detect_mode")
        g.add_conditional_edges(
            "detect_mode",
            self._route,
            {"local": "extract_entities", "global": "global_search"},
        )
        g.add_edge("extract_entities", "local_retrieve")
        g.add_edge("local_retrieve", "local_compose")
        g.add_edge("local_compose", END)
        g.add_edge("global_search", END)
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
        for t in (self.global_tool,):
            if hasattr(t, "close"):
                try:
                    t.close()
                except Exception:
                    pass
