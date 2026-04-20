"""
AgenticReActTool —— 真正的 tool-calling ReAct agent

用 LangGraph 的 prebuilt `create_react_agent` 绑定 5 个检索原语：
  - hybrid_search       文本事实查询
  - graph_lookup        实体邻域/关系展开
  - path_search         两实体间多跳路径
  - fetch_document      整篇论文取回
  - entity_search       实体消歧

每轮由 LLM 自主决定调哪个工具、什么参数，比之前"固定 retrieval + 改写 query"更灵活。

Langfuse：
  - 每个 tool call 是一个 span (name = 工具名)
  - LLM 的 tool_calls 消息是 generation span
  - 整体是一棵树，不是平铺
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from graphrag_agent.models.get_models import get_llm_model
from graphrag_agent.search.tool.base import BaseSearchTool
from graphrag_agent.search.tool.primitives import get_all_primitive_tools
from graphrag_agent.utils.langfuse_client import get_langfuse_handler
from graphrag_agent.utils.logger import get_logger

logger = get_logger(__name__)


AGENTIC_SYSTEM_PROMPT = """You are a research assistant that answers questions about RAG academic papers. You have 5 tools:

1. **hybrid_search(query, top_k)** — Semantic + keyword search over text chunks. Use for factual questions about a specific concept or method.
2. **graph_lookup(entity_name, max_neighbors, rel_types)** — Get an entity's neighbors and relationships. Use for "what does X use?", "what is X compared with?".
3. **path_search(entity_a, entity_b, max_hops)** — Find shortest path between two entities. Use for multi-hop questions like "how are A and B related?".
4. **fetch_document(file_name, max_chunks)** — Get all chunks of a specific paper. Use when you've located the right paper but need more context from it.
5. **entity_search(name, top_k)** — Disambiguate an entity name. Use when you're unsure which entity a name refers to.

## Strategy
- Start with **hybrid_search** for most queries
- If results mention entities you need more info about → **graph_lookup**
- For comparison/relationship questions → **graph_lookup** or **path_search** on both entities
- For enumeration (e.g. "which papers use X?") → **graph_lookup** on X to find all `USES` or `EVALUATED_ON` neighbors
- Only call **fetch_document** if hybrid_search gave you a relevant paper but you need more of it
- Don't make more than 6 tool calls. Stop when you have enough to answer.

## Output
After gathering evidence, produce a final answer that:
- **Starts directly with the key fact.** No "### Overview" preamble.
- Responds in **English**.
- Cites specific paper names when giving facts.
- If evidence is insufficient, say so plainly — don't guess.
"""


class AgenticReActTool(BaseSearchTool):
    """Tool-calling ReAct agent built on LangGraph prebuilt."""

    def __init__(self, recursion_limit: int = 12):
        super().__init__(cache_dir="./cache/agentic_react")
        self.tools = get_all_primitive_tools()
        self.react_llm = get_llm_model()
        self.recursion_limit = recursion_limit

        # Build the ReAct agent graph. create_react_agent returns a compiled LangGraph.
        self._agent = create_react_agent(
            model=self.react_llm,
            tools=self.tools,
            prompt=AGENTIC_SYSTEM_PROMPT,
        )

    def _setup_chains(self):
        """BaseSearchTool abstract — ReAct 用 create_react_agent，无需手搭 chain"""
        pass

    def search(self, query_input: Any,
               session_id: Optional[str] = None,
               parent_config: Optional[Dict] = None) -> str:
        """
        Compatible with BaseSearchTool.search() interface.
        Returns final answer string.
        """
        overall_start = time.time()
        query = query_input["query"] if isinstance(query_input, dict) else str(query_input)

        cache_key = f"agentic_react:{query}"
        cached = self.cache_manager.get(cache_key)
        if cached:
            return cached

        # Langfuse config: parent 优先，否则独立 run
        if parent_config:
            config = dict(parent_config)
        else:
            config = {}
            handler = get_langfuse_handler()
            if handler is not None:
                config = {
                    "callbacks": [handler],
                    "run_name": "agentic_react",
                    "metadata": {
                        "langfuse_session_id": session_id or f"react_{uuid.uuid4().hex[:8]}",
                        "langfuse_tags": ["agentic", "react"],
                        "query": query[:100],
                    },
                }

        # recursion_limit 控制最大 step 数（每步 = 1 tool call 或 1 LLM response）
        config["recursion_limit"] = self.recursion_limit

        messages = [HumanMessage(content=query)]

        try:
            final_state = self._agent.invoke({"messages": messages}, config=config)
            msgs = final_state.get("messages", [])
            # 最后一个 AI message 就是答案
            answer = ""
            for m in reversed(msgs):
                if hasattr(m, "content") and m.content and getattr(m, "type", "") == "ai":
                    answer = m.content
                    break
            if not answer:
                answer = "No answer produced"

            self.cache_manager.set(cache_key, answer)
            elapsed = time.time() - overall_start
            n_tool_calls = sum(
                1 for m in msgs if getattr(m, "type", "") == "tool"
            )
            logger.info(
                f"[react] done ({elapsed:.1f}s, {n_tool_calls} tool calls)",
                extra={"elapsed": elapsed, "tool_calls": n_tool_calls},
            )
            return answer
        except Exception as e:
            logger.error(f"[react] graph failed: {e}")
            return f"ReAct agent failed: {e}"

    def extract_keywords(self, query: str) -> Dict[str, list]:
        return {"low_level": [], "high_level": []}

    def get_tool(self):
        """Return a BaseTool wrapper (for BaseAgent interop)."""
        from langchain_core.tools import BaseTool as _BT

        outer = self

        class _ReActToolWrapper(_BT):
            name: str = "agentic_react"
            description: str = "ReAct agent with 5 retrieval primitives for complex multi-step queries"

            def _run(self_t, query: Any) -> str:
                return outer.search(query)

            def _arun(self_t, query: Any) -> str:
                raise NotImplementedError

        return _ReActToolWrapper()
