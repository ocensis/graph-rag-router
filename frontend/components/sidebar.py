import streamlit as st
from utils.api import clear_chat
from frontend_config.settings import examples


AGENT_OPTIONS = [
    "router_agent",
    "naive_rag_agent",
    "graph_agent",
    "agentic_agent",
]

AGENT_HELP = {
    "router_agent":    "🎯 **推荐** · 3-way Classifier Router · 动态分派 Naive / Graph / Agentic",
    "naive_rag_agent": "Baseline: vector + BM25 + RRF (Router 的 Classic 路径)",
    "graph_agent":     "Entity 邻域 + Community 摘要 (Router 的 Graph 路径)",
    "agentic_agent":   "Planner + ReAct (Classic/Graph 作 sub-tool) (Router 的 Agentic 路径)",
}


def display_sidebar():
    """显示应用侧边栏（重构版，对齐当前架构）"""
    with st.sidebar:
        st.title("🎯 GraphRAG")
        st.caption("Multi-path Agentic RAG  |  RAG Papers KG")
        st.markdown("---")

        # ---- Agent 选择 ----
        st.subheader("Agent 策略")
        current = st.session_state.agent_type if st.session_state.agent_type in AGENT_OPTIONS else "router_agent"
        agent_type = st.radio(
            "选择检索策略",
            AGENT_OPTIONS,
            index=AGENT_OPTIONS.index(current),
            format_func=lambda x: {
                "router_agent": "🎯 Router (推荐)",
                "naive_rag_agent": "Naive",
                "graph_agent": "Graph",
                "agentic_agent": "Agentic",
            }[x],
            key="sidebar_agent_type",
            label_visibility="collapsed",
        )
        st.session_state.agent_type = agent_type
        st.caption(AGENT_HELP.get(agent_type, ""))
        st.session_state.show_thinking = False

        st.markdown("---")

        # ---- 系统设置 ----
        st.subheader("系统设置")
        debug_mode = st.toggle(
            "调试模式",
            value=st.session_state.debug_mode,
            key="sidebar_debug_mode",
            help="展示执行 trace / KG 可视化 / 性能指标",
        )
        previous_debug_mode = st.session_state.debug_mode
        if debug_mode != previous_debug_mode and debug_mode:
            st.session_state.use_stream = False
        st.session_state.debug_mode = debug_mode

        if not debug_mode:
            use_stream = st.toggle(
                "流式响应",
                value=st.session_state.get("use_stream", True),
                key="sidebar_use_stream",
            )
            st.session_state.use_stream = use_stream
        else:
            st.caption("调试模式下流式响应自动关闭")

        st.markdown("---")

        # ---- 示例问题 ----
        st.subheader("示例问题")
        st.caption("点击复制到输入框")
        for q in examples:
            if st.button(q, key=f"eg_{hash(q)}", use_container_width=True):
                # 用 session_state 传递到 chat 输入框
                st.session_state.example_question = q
                st.rerun()

        st.markdown("---")

        # ---- 项目信息 ----
        with st.expander("关于本项目"):
            st.markdown(
                """
**架构**: LangGraph 3-way classifier router
- **Classic** (~40%): hybrid vector+BM25+RRF
- **Graph** (~48%): entity + community summary + chunks
- **Agentic** (~12%): planner + ReAct (classic/graph 作 sub-tools)

**实测 eng200 benchmark**
- LLM-Acc **0.57** (vs Agentic 0.55)
- 延迟 ↓50% · 成本 ↓57%

**KG**: 51 papers / 11K entities / 65K rels
"""
            )

        if st.button("🗑️ 清除对话历史", use_container_width=True, key="clear_chat_btn"):
            clear_chat()
