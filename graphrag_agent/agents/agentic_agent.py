"""
AgenticAgent —— standalone agentic path（绕过 router 分类器，直接跑 plan-execute-aggregate）

用途: 前端 agent 对照组；绝大多数生产场景应该用 RouterAgent，由它动态决定是否走 agentic。
"""
from __future__ import annotations

import time
import uuid

from graphrag_agent.agents.paths import AgenticPath, ClassicPath, GraphPath
from graphrag_agent.utils.langfuse_client import get_langfuse_handler
from graphrag_agent.utils.logger import get_logger

logger = get_logger(__name__)


class AgenticAgent:
    """绕过 router，直接调 AgenticPath。planner → ReAct (classic/graph sub-tools) → aggregate"""

    def __init__(self):
        self.classic = ClassicPath(reranker_mode="none")
        self.graph = GraphPath()
        self.agentic = AgenticPath(classic_path=self.classic, graph_path=self.graph)
        self.cache_manager = self.classic.cache_manager

    def ask(self, query: str, skip_cache: bool = False, **kwargs) -> str:
        start = time.time()
        session_id = kwargs.get("session_id") or f"agentic_{int(start)}_{uuid.uuid4().hex[:6]}"

        config = {}
        handler = get_langfuse_handler()
        if handler is not None:
            config = {
                "callbacks": [handler],
                "run_name": "agentic_agent_standalone",
                "metadata": {
                    "langfuse_session_id": session_id,
                    "langfuse_tags": ["agentic", "standalone"],
                    "query": query[:100],
                },
            }

        try:
            answer = self.agentic.run(query, session_id=session_id, parent_config=config)
            elapsed = round(time.time() - start, 2)
            logger.info(f"[agentic_standalone] done ({elapsed}s)")
            return answer
        except Exception as e:
            logger.error(f"[agentic_standalone] failed: {e}")
            return f"Agentic agent failed: {e}"

    def check_fast_cache(self, query: str, session_id: str = "default"):
        """chat_service 兼容接口，不做 fast cache"""
        return None

    def ask_with_trace(self, query: str, thread_id: str = None, **kwargs) -> dict:
        """供前端 debug 面板用。简单包装 ask() 加一行 trace"""
        import time as _t
        t0 = _t.time()
        answer = self.ask(query, session_id=thread_id or "default")
        return {
            "answer": answer,
            "execution_log": [{
                "node": "agentic_standalone",
                "input": query[:100],
                "output": "planner → ReAct (with classic/graph sub-tools) → aggregate",
                "latency": round(_t.time() - t0, 2),
            }],
            "kg_data": {"nodes": [], "links": []},
            "reference": {},
        }

    def close(self):
        for t in (self.classic, self.graph):
            if hasattr(t, "close"):
                try:
                    t.close()
                except Exception:
                    pass
