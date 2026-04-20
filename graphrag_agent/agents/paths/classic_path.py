"""
Classic Path —— 默认路径，覆盖 70-90% 企业 KB 查询

定位：找文档片段 + 给引用
流程：hybrid retrieval (vector + BM25 + RRF) → optional rerank → LLM compose w/ citations

实现：直接复用 NaiveSearchTool（已经做了 hybrid + 可选 cross-encoder rerank + answer w/ citations）
只包一层：统一 parent_config 透传，确保 Langfuse 下一个独立 span
"""
from __future__ import annotations

from typing import Optional, Dict

from graphrag_agent.search.tool.naive_search_tool import NaiveSearchTool
from graphrag_agent.utils.logger import get_logger

logger = get_logger(__name__)


class ClassicPath:
    """Classic RAG path = hybrid search + rerank + cite."""

    def __init__(self, reranker_mode: str = "none"):
        # reranker_mode="cross_encoder" 可启用 Qwen3-Reranker
        self.tool = NaiveSearchTool(reranker_mode=reranker_mode)
        self.cache_manager = self.tool.cache_manager

    def run(self, query: str,
            session_id: Optional[str] = None,
            parent_config: Optional[Dict] = None) -> str:
        try:
            return self.tool.search(query, session_id=session_id, parent_config=parent_config)
        except Exception as e:
            logger.error(f"[classic_path] failed: {e}")
            return f"Classic path failed: {e}"

    def close(self):
        if hasattr(self.tool, "close"):
            try:
                self.tool.close()
            except Exception:
                pass
