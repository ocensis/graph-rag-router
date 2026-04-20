"""
Retrieval Primitives —— 5 个可被 ReAct agent 直接调用的工具原语

为什么拆成独立文件：
- Router cascade 的每一档（naive / local_graph / agentic）都要复用这些原语
- ReAct agent 需要每个工具有独立 BaseTool + args_schema，LLM 才能做 tool_calls
- 和 Langfuse 搭配：每次 tool 调用是一个 span，可视化调用树清楚

5 个原语:
  1. hybrid_search(query, top_k)      —— vector + BM25 + RRF
  2. graph_lookup(entity_name, ...)   —— 实体邻域 + 关系展开
  3. path_search(entity_a, entity_b)  —— 两实体间最短路径
  4. fetch_document(file_name)        —— 按 fileName 取整篇论文的 chunks
  5. entity_search(name, top_k)       —— 模糊匹配 + 向量重排的实体消歧候选
"""
from __future__ import annotations

import threading
from typing import List, Optional, Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.models.get_models import get_embeddings_model
from graphrag_agent.utils.logger import get_logger

logger = get_logger(__name__)


# ==================== Shared singletons ====================
# 原语被多个 agent 共用，用 module 级单例避免重复初始化 Neo4j 驱动/embeddings

_lock = threading.Lock()
_graph = None
_embeddings = None


def _get_graph():
    global _graph
    if _graph is None:
        with _lock:
            if _graph is None:
                _graph = get_db_manager().get_graph()
    return _graph


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        with _lock:
            if _embeddings is None:
                _embeddings = get_embeddings_model()
    return _embeddings


# ==================== 1. hybrid_search ====================

class HybridSearchArgs(BaseModel):
    query: str = Field(..., description="The text query to search for")
    top_k: int = Field(5, description="Number of top chunks to return (default 5)")


class HybridSearchTool(BaseTool):
    """Vector + BM25 双路召回 + RRF 融合。最常用的文本事实检索工具。"""
    name: str = "hybrid_search"
    description: str = (
        "Search text chunks by semantic similarity (vector) AND keyword matching (BM25), "
        "merged by Reciprocal Rank Fusion. Returns top-k chunks with text and source paper. "
        "Use this for factual questions about a specific concept, method, or paper."
    )
    args_schema: Type[BaseModel] = HybridSearchArgs

    def _run(self, query: str, top_k: int = 5) -> str:
        graph = _get_graph()
        emb = _get_embeddings()
        try:
            vec = emb.embed_query(query)
        except Exception as e:
            return f"[hybrid_search] embedding failed: {e}"

        K = 60  # RRF constant
        per_path = max(50, top_k * 5)

        # 向量召回
        try:
            vec_rows = graph.query(
                """CALL db.index.vector.queryNodes('chunk_vector', $k, $vec)
                YIELD node, score
                RETURN node.id AS id,
                       coalesce(node.original_text, node.text) AS text,
                       node.fileName AS fileName,
                       score""",
                {"k": per_path, "vec": vec},
            )
        except Exception as e:
            logger.warning(f"[hybrid_search] vector recall failed: {e}")
            vec_rows = []

        # BM25 召回
        try:
            bm25_rows = graph.query(
                """CALL db.index.fulltext.queryNodes('chunk_fulltext', $q)
                YIELD node, score
                RETURN node.id AS id,
                       coalesce(node.original_text, node.text) AS text,
                       node.fileName AS fileName,
                       score
                LIMIT $k""",
                {"k": per_path, "q": query},
            )
        except Exception as e:
            logger.warning(f"[hybrid_search] bm25 recall failed: {e}")
            bm25_rows = []

        # RRF
        rrf: dict = {}
        for i, r in enumerate(vec_rows):
            cid = r["id"]
            rrf[cid] = rrf.get(cid, 0) + 1.0 / (K + i + 1)
        for i, r in enumerate(bm25_rows):
            cid = r["id"]
            rrf[cid] = rrf.get(cid, 0) + 1.0 / (K + i + 1)

        by_id = {r["id"]: r for r in vec_rows + bm25_rows}
        ranked = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]

        if not ranked:
            return "[hybrid_search] no results"

        out = [f"Found {len(ranked)} chunks (rrf-fused):\n"]
        for i, (cid, s) in enumerate(ranked, 1):
            row = by_id[cid]
            paper = (row.get("fileName") or "").replace(".pdf", "")
            text = (row.get("text") or "")[:500]
            out.append(f"[{i}] {paper} (chunk={cid[:8]}, rrf={s:.4f})\n{text}\n")
        return "\n".join(out)


# ==================== 2. graph_lookup ====================

class GraphLookupArgs(BaseModel):
    entity_name: str = Field(..., description="Name of the entity to look up (e.g., 'GRPO', 'UniDoc-RL')")
    max_neighbors: int = Field(10, description="Max neighboring entities to return (default 10)")
    rel_types: Optional[List[str]] = Field(
        None,
        description="Optional relation types to filter (e.g., ['USES', 'EVALUATED_ON']). "
                    "Omit to include all non-MENTIONS relations.",
    )


class GraphLookupTool(BaseTool):
    """从实体出发做 1-hop 邻域扩展，返回相关实体 + 关系类型 + description。"""
    name: str = "graph_lookup"
    description: str = (
        "Look up an entity in the knowledge graph and return its direct neighbors and "
        "the relationships connecting them. Use this for questions like 'what does X use?', "
        "'what is X compared with?', 'what papers use method Y?'."
    )
    args_schema: Type[BaseModel] = GraphLookupArgs

    def _run(self, entity_name: str, max_neighbors: int = 10,
             rel_types: Optional[List[str]] = None) -> str:
        graph = _get_graph()

        rel_filter = ""
        params = {"name": entity_name, "k": max_neighbors}
        if rel_types:
            rel_filter = "AND type(r) IN $rel_types"
            params["rel_types"] = rel_types

        # 精确匹配优先于 CONTAINS 匹配（避免 "VISOR" 查询先命中 "VISOR Agent Loop"）
        cypher = f"""
        MATCH (e:__Entity__)
        WHERE toLower(e.id) = toLower($name)
           OR toLower(e.id) CONTAINS toLower($name)
        WITH e,
             CASE WHEN toLower(e.id) = toLower($name) THEN 2
                  WHEN toLower(e.id) STARTS WITH toLower($name) THEN 1
                  ELSE 0 END AS match_priority
        ORDER BY match_priority DESC
        LIMIT 1
        MATCH (e)-[r]-(n:__Entity__)
        WHERE type(r) <> 'MENTIONS' AND type(r) <> 'IN_COMMUNITY' {rel_filter}
        RETURN e.id AS center,
               coalesce(e.description, '') AS center_desc,
               type(r) AS rel,
               n.id AS neighbor,
               coalesce(n.description, '') AS neighbor_desc,
               labels(n) AS neighbor_labels
        LIMIT $k
        """
        try:
            rows = graph.query(cypher, params)
        except Exception as e:
            return f"[graph_lookup] query failed: {e}"

        if not rows:
            return f"[graph_lookup] entity '{entity_name}' not found or has no neighbors"

        center = rows[0]["center"]
        center_desc = (rows[0]["center_desc"] or "")[:200]
        out = [f"Entity: {center}"]
        if center_desc:
            out.append(f"Description: {center_desc}")
        out.append(f"\n{len(rows)} neighbors:")
        for r in rows:
            labels = [l for l in (r["neighbor_labels"] or []) if l != "__Entity__"]
            label_str = f"[{labels[0]}] " if labels else ""
            desc = (r["neighbor_desc"] or "")[:120]
            out.append(f"  -[{r['rel']}]-> {label_str}{r['neighbor']}")
            if desc:
                out.append(f"       {desc}")
        return "\n".join(out)


# ==================== 3. path_search ====================

class PathSearchArgs(BaseModel):
    entity_a: str = Field(..., description="Source entity name")
    entity_b: str = Field(..., description="Target entity name")
    max_hops: int = Field(3, description="Max path length (default 3)")


class PathSearchTool(BaseTool):
    """两实体间最短路径（排除 MENTIONS/IN_COMMUNITY 元边）。用于多跳推理。"""
    name: str = "path_search"
    description: str = (
        "Find the shortest relational path between two entities in the knowledge graph. "
        "Returns the chain of relationships connecting them, if any. "
        "Use this for multi-hop questions like 'how is method A related to method B?'."
    )
    args_schema: Type[BaseModel] = PathSearchArgs

    def _run(self, entity_a: str, entity_b: str, max_hops: int = 3) -> str:
        graph = _get_graph()
        try:
            rows = graph.query(
                """
                MATCH (a:__Entity__), (b:__Entity__)
                WHERE (toLower(a.id) = toLower($a) OR toLower(a.id) CONTAINS toLower($a))
                  AND (toLower(b.id) = toLower($b) OR toLower(b.id) CONTAINS toLower($b))
                WITH a, b LIMIT 1
                MATCH p = shortestPath((a)-[*..""" + str(max_hops) + """]-(b))
                WHERE all(r IN relationships(p) WHERE type(r) <> 'MENTIONS' AND type(r) <> 'IN_COMMUNITY')
                RETURN [n IN nodes(p) | n.id] AS nodes,
                       [r IN relationships(p) | type(r)] AS rels,
                       length(p) AS hops
                """,
                {"a": entity_a, "b": entity_b},
            )
        except Exception as e:
            return f"[path_search] query failed: {e}"

        if not rows:
            return f"[path_search] no path found between '{entity_a}' and '{entity_b}' within {max_hops} hops"

        row = rows[0]
        nodes = row["nodes"]
        rels = row["rels"]
        hops = row["hops"]
        chain = nodes[0]
        for i, rel in enumerate(rels):
            chain += f" -[{rel}]-> {nodes[i + 1]}"
        return f"Path ({hops} hops):\n  {chain}"


# ==================== 4. fetch_document ====================

class FetchDocumentArgs(BaseModel):
    file_name: str = Field(..., description="Paper fileName or arxiv id (fuzzy match)")
    max_chunks: int = Field(20, description="Max chunks of the doc to return (default 20)")


class FetchDocumentTool(BaseTool):
    """按 fileName 取整篇论文的 chunks（按 position 排序）。chunk 不够时用。"""
    name: str = "fetch_document"
    description: str = (
        "Fetch chunks of a specific paper by its file name or arxiv id. "
        "Use this when hybrid_search returns a relevant paper but you need more context from it."
    )
    args_schema: Type[BaseModel] = FetchDocumentArgs

    def _run(self, file_name: str, max_chunks: int = 20) -> str:
        graph = _get_graph()
        try:
            rows = graph.query(
                """
                MATCH (d:__Document__)
                WHERE toLower(d.fileName) CONTAINS toLower($q)
                WITH d LIMIT 1
                MATCH (c:__Chunk__)-[:PART_OF]->(d)
                RETURN d.fileName AS paper,
                       c.id AS id,
                       coalesce(c.original_text, c.text) AS text,
                       c.position AS pos
                ORDER BY c.position
                LIMIT $k
                """,
                {"q": file_name, "k": max_chunks},
            )
        except Exception as e:
            return f"[fetch_document] query failed: {e}"

        if not rows:
            return f"[fetch_document] paper '{file_name}' not found"

        paper = (rows[0]["paper"] or "").replace(".pdf", "")
        out = [f"Paper: {paper} ({len(rows)} chunks)\n"]
        for r in rows:
            text = (r.get("text") or "")[:400]
            out.append(f"[pos={r.get('pos', '?')}] {text}\n")
        return "\n".join(out)


# ==================== 5. entity_search ====================

class EntitySearchArgs(BaseModel):
    name: str = Field(..., description="Entity name (possibly ambiguous or partial)")
    top_k: int = Field(5, description="Number of candidates to return (default 5)")


class EntitySearchTool(BaseTool):
    """实体消歧 —— 文本模糊 + 向量语义双路，返回候选实体列表供 LLM 选择。"""
    name: str = "entity_search"
    description: str = (
        "Disambiguate an entity name by finding all matching canonical entities. "
        "Returns candidates ranked by name similarity and semantic similarity. "
        "Use when you're unsure which entity a name refers to (e.g., 'GPT' could be many things)."
    )
    args_schema: Type[BaseModel] = EntitySearchArgs

    def _run(self, name: str, top_k: int = 5) -> str:
        graph = _get_graph()
        emb = _get_embeddings()
        # 字符模糊匹配
        try:
            fuzzy = graph.query(
                """
                MATCH (e:__Entity__)
                WHERE toLower(e.id) CONTAINS toLower($name)
                RETURN e.id AS id,
                       coalesce(e.description, '') AS desc,
                       labels(e) AS labels
                LIMIT $k
                """,
                {"name": name, "k": top_k * 2},
            )
        except Exception as e:
            fuzzy = []
            logger.warning(f"[entity_search] fuzzy failed: {e}")

        # 向量语义匹配
        vec_rows = []
        try:
            vec = emb.embed_query(name)
            vec_rows = graph.query(
                """CALL db.index.vector.queryNodes('vector', $k, $vec)
                YIELD node, score
                RETURN node.id AS id,
                       coalesce(node.description, '') AS desc,
                       labels(node) AS labels,
                       score""",
                {"k": top_k, "vec": vec},
            )
        except Exception as e:
            logger.warning(f"[entity_search] vector failed: {e}")

        # 合并去重（以 id 为 key），字符匹配优先
        seen = set()
        candidates = []
        for r in fuzzy:
            if r["id"] not in seen:
                seen.add(r["id"])
                candidates.append({"source": "fuzzy", **r})
        for r in vec_rows:
            if r["id"] not in seen:
                seen.add(r["id"])
                candidates.append({"source": "vector", **r})
        candidates = candidates[:top_k]

        if not candidates:
            return f"[entity_search] no candidates for '{name}'"

        out = [f"{len(candidates)} candidates for '{name}':"]
        for i, c in enumerate(candidates, 1):
            labels = [l for l in (c.get("labels") or []) if l != "__Entity__"]
            label_str = f"[{labels[0]}] " if labels else ""
            desc = (c.get("desc") or "")[:150]
            out.append(f"  {i}. {label_str}{c['id']}  ({c['source']})")
            if desc:
                out.append(f"       {desc}")
        return "\n".join(out)


# ==================== 工具集合导出 ====================

def get_all_primitive_tools() -> List[BaseTool]:
    """返回 5 个原语工具实例，可直接绑定到 ReAct agent"""
    return [
        HybridSearchTool(),
        GraphLookupTool(),
        PathSearchTool(),
        FetchDocumentTool(),
        EntitySearchTool(),
    ]


__all__ = [
    "HybridSearchTool",
    "GraphLookupTool",
    "PathSearchTool",
    "FetchDocumentTool",
    "EntitySearchTool",
    "get_all_primitive_tools",
]
