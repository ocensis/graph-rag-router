"""
3 条 retrieval path (classic / graph / agentic)，各自独立 LangGraph subgraph
供 RouterAgent 顶层分派调用
"""
from graphrag_agent.agents.paths.classic_path import ClassicPath
from graphrag_agent.agents.paths.graph_path import GraphPath
from graphrag_agent.agents.paths.agentic_path import AgenticPath

__all__ = ["ClassicPath", "GraphPath", "AgenticPath"]
