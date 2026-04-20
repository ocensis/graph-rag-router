"""
Agentic Path —— 复杂查询规划，覆盖 5-10% 长尾

关键设计：**Classic 和 Graph 是 agentic 的 sub-tools**，而不是被 agentic 绕过。
这样 agentic 在路径级别做规划，不用在原语层硬搓。

流程:
  1. Planner  —— LLM 把 query 拆成 2-4 个子问题
  2. Execute  —— ReAct 循环，对每个子问题选 classic_search 或 graph_search (或其他工具)
  3. Aggregate —— LLM 聚合所有子问题结果成最终答案

工具集 (5 个)：
  - classic_search(query) —— 调 Classic path（hybrid + rerank + cite）
  - graph_search(query)   —— 调 Graph path（entity / global）
  - fetch_document(file_name) —— 取整篇论文
  - entity_search(name)   —— 实体消歧
  - path_search(a, b)     —— 两实体间最短路径

Langfuse：每个 planner / execute / aggregate 是独立 span，子工具调用嵌套其下
"""
from __future__ import annotations

import re
from typing import Optional, Dict, List, TypedDict, Type, Any

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

from graphrag_agent.models.get_models import get_llm_model
from graphrag_agent.search.tool.primitives import (
    FetchDocumentTool,
    EntitySearchTool,
    PathSearchTool,
)
from graphrag_agent.agents.paths.context_packer import truncate_to_tokens, count_tokens
from graphrag_agent.utils.langfuse_client import get_langfuse_handler
from graphrag_agent.utils.logger import get_logger
from graphrag_agent.utils.resilience import retry_sync

logger = get_logger(__name__)


# ==================== State ====================

class AgenticPathState(TypedDict, total=False):
    query: str
    subqueries: List[str]
    evidence: str          # 聚合的子问题答案
    answer: str


# ==================== Prompts ====================

PLANNER_PROMPT = """You are a query planner for a research assistant. Decompose the user question into 2-4 focused sub-questions that can each be answered independently by a retriever.

## Rules
- If the original question is already atomic (single fact), output ONE sub-question that rephrases it.
- For comparison questions: one sub-question per entity being compared, plus one "comparison axis" if useful.
- For enumeration/aggregation: one sub-question per dimension or criterion.
- Each sub-question should be self-contained and specific.

## Examples

Q: "How do UniDoc-RL and VISOR differ in RL training?"
Subqueries:
- What RL training approach does UniDoc-RL use?
- What RL training approach does VISOR use?
- What specific mechanism or reward design differs between them?

Q: "Which papers use GRPO?"
Subqueries:
- Which RAG papers mention or apply GRPO as their training algorithm?
- What variants of GRPO are used across these papers?

Q: "What is the contribution of MM-Doc-R1?"
Subqueries:
- What is the main contribution of MM-Doc-R1?

## User question
{query}

## Output
Return ONLY the sub-questions, one per line starting with "- ". No other text."""


AGGREGATE_PROMPT = """You are composing a final answer for the user by synthesizing findings from multiple sub-question retrievals.

**IMPORTANT: Respond in English.**

## Original question
{query}

## Sub-question findings
{evidence}

## Rules
- Start the answer with the direct key fact. No "### Overview" preamble.
- Merge the findings into a coherent answer.
- For comparison questions: cover both sides side-by-side with specifics.
- For enumeration: list all items found, grouped if helpful.
- Cite concrete paper names when relevant.
- If findings disagree or are insufficient, say so plainly.
- Plain paragraphs, minimal markdown.
"""


# ==================== Sub-path tools (wrap paths as BaseTool) ====================

class ClassicSearchArgs(BaseModel):
    query: str = Field(..., description="Focused sub-question to retrieve doc chunks for")


class GraphSearchArgs(BaseModel):
    query: str = Field(..., description="Sub-question about entity relations or corpus-wide patterns")


def make_classic_tool(classic_path) -> BaseTool:
    """Wrap ClassicPath as a BaseTool so the ReAct LLM can call it."""
    class ClassicSearchTool(BaseTool):
        name: str = "classic_search"
        description: str = (
            "Search text chunks via hybrid retrieval (vector + BM25 + RRF) and synthesize an "
            "answer with citations. Use this for factual questions where the answer is in doc text."
        )
        args_schema: Type[BaseModel] = ClassicSearchArgs

        def _run(self_t, query: str) -> str:
            return classic_path.run(query)

    return ClassicSearchTool()


def make_graph_tool(graph_path) -> BaseTool:
    """Wrap GraphPath as a BaseTool."""
    class GraphSearchTool(BaseTool):
        name: str = "graph_search"
        description: str = (
            "Search the knowledge graph for entity relationships or aggregate across the corpus. "
            "Use this for 'how is A related to B?', 'what does X use?', or 'summarize the main "
            "approaches across all papers'."
        )
        args_schema: Type[BaseModel] = GraphSearchArgs

        def _run(self_t, query: str) -> str:
            return graph_path.run(query)

    return GraphSearchTool()


# ==================== Path ====================

class AgenticPath:
    """Planner → ReAct (w/ classic & graph as tools) → Aggregate."""

    def __init__(self, classic_path, graph_path, recursion_limit: int = 15):
        self.llm = get_llm_model()
        self.recursion_limit = recursion_limit

        # Planner & aggregator
        self.planner_chain = (
            ChatPromptTemplate.from_messages([("human", PLANNER_PROMPT)])
            | self.llm | StrOutputParser()
        )
        self.aggregate_chain = (
            ChatPromptTemplate.from_messages([("human", AGGREGATE_PROMPT)])
            | self.llm | StrOutputParser()
        )

        # ReAct tools
        self.tools = [
            make_classic_tool(classic_path),
            make_graph_tool(graph_path),
            FetchDocumentTool(),
            EntitySearchTool(),
            PathSearchTool(),
        ]
        self._react_system = (
            "You are a research assistant answering a sub-question about RAG academic papers. "
            "You can call the following tools to gather evidence:\n\n"
            "1. **classic_search(query)** — hybrid retrieval + cited answer. Use first for factual queries.\n"
            "2. **graph_search(query)** — entity relationships or corpus-wide aggregation.\n"
            "3. **fetch_document(file_name)** — get full chunks of a specific paper.\n"
            "4. **entity_search(name)** — disambiguate an entity name.\n"
            "5. **path_search(entity_a, entity_b)** — shortest relational path between two entities.\n\n"
            "Strategy: start with classic_search unless the sub-question is explicitly relational "
            "or corpus-wide. Call at most 2-3 tools. Produce a concise final answer in English."
        )
        self._react = create_react_agent(
            model=self.llm, tools=self.tools, prompt=self._react_system
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

    @staticmethod
    def _parse_subqueries(raw: str) -> List[str]:
        subs = []
        for line in raw.strip().splitlines():
            m = re.match(r"^\s*[-*]\s+(.+)$", line)
            if m:
                subs.append(m.group(1).strip())
        # fallback: take non-empty lines if no bullet format
        if not subs:
            subs = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        return subs[:4]

    # ------- nodes -------

    def _node_plan(self, state: AgenticPathState, config: Dict = None) -> AgenticPathState:
        try:
            raw = retry_sync(max_retries=2, base_delay=0.5)(self.planner_chain.invoke)(
                {"query": state["query"]},
                config=self._scoped(config, "planner"),
            )
            subs = self._parse_subqueries(raw)
        except Exception as e:
            logger.warning(f"[agentic_path] plan failed: {e}, fallback single subquery")
            subs = [state["query"]]
        if not subs:
            subs = [state["query"]]
        logger.info(f"[agentic_path] planned {len(subs)} subqueries")
        return {"subqueries": subs}

    def _node_execute(self, state: AgenticPathState, config: Dict = None) -> AgenticPathState:
        subs = state.get("subqueries", [])
        findings = []
        for i, sq in enumerate(subs, 1):
            try:
                cfg = self._scoped(config, f"subquery_{i}")
                if cfg:
                    cfg["recursion_limit"] = self.recursion_limit
                final = self._react.invoke(
                    {"messages": [HumanMessage(content=sq)]},
                    config=cfg,
                )
                # 取最后一个 AI message 作为答案
                ai_answer = ""
                for m in reversed(final.get("messages", [])):
                    if getattr(m, "type", "") == "ai" and getattr(m, "content", ""):
                        ai_answer = m.content
                        break
                findings.append(f"### Sub-Q{i}: {sq}\n{ai_answer}")
            except Exception as e:
                logger.error(f"[agentic_path] subquery {i} failed: {e}")
                findings.append(f"### Sub-Q{i}: {sq}\n(failed: {e})")

        evidence = "\n\n".join(findings)
        return {"evidence": evidence}

    def _node_aggregate(self, state: AgenticPathState, config: Dict = None) -> AgenticPathState:
        # Token 预算：给 evidence 留 6k token（子问题答案可能很长，合起来容易爆）
        evidence = truncate_to_tokens(state.get("evidence", ""), max_tokens=6000)
        try:
            ans = retry_sync(max_retries=2, base_delay=0.5)(self.aggregate_chain.invoke)(
                {"query": state["query"], "evidence": evidence},
                config=self._scoped(config, "aggregator"),
            )
        except Exception as e:
            logger.error(f"[agentic_path] aggregate failed: {e}")
            ans = state.get("evidence", "") or f"Agentic aggregate failed: {e}"
        return {"answer": ans}

    # ------- build -------

    def _build_graph(self):
        g = StateGraph(AgenticPathState)
        g.add_node("plan", self._node_plan)
        g.add_node("execute", self._node_execute)
        g.add_node("aggregate", self._node_aggregate)
        g.set_entry_point("plan")
        g.add_edge("plan", "execute")
        g.add_edge("execute", "aggregate")
        g.add_edge("aggregate", END)
        return g.compile()

    # ------- public -------

    def run(self, query: str,
            session_id: Optional[str] = None,
            parent_config: Optional[Dict] = None) -> str:
        init: AgenticPathState = {"query": query}
        try:
            final = self._graph.invoke(init, config=parent_config)
            return final.get("answer", "Agentic path produced no answer")
        except Exception as e:
            logger.error(f"[agentic_path] graph invoke failed: {e}")
            return f"Agentic path failed: {e}"
