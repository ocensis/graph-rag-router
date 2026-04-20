"""
Streamlit Page: Trace Viewer
从 Langfuse 拉最近的 trace，渲染单条 trace 的完整调用树
"""
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st

_frontend_root = Path(__file__).parent.parent
if str(_frontend_root) not in sys.path:
    sys.path.insert(0, str(_frontend_root))

from utils.langfuse_helpers import (
    get_client,
    fetch_traces_by_name,
    fetch_observations,
    analyze_trace,
    build_observation_tree,
)

st.set_page_config(page_title="Trace Viewer", page_icon="🔍", layout="wide")


# ==================== Sidebar ====================

with st.sidebar:
    st.title("🎯 GraphRAG")
    st.caption("Multi-path Agentic RAG  |  RAG Papers KG")
    st.markdown("---")
    st.info("🔍 你在 **Trace Viewer** 调用链页")
    st.caption("单条 trace 的完整 span 树")
    if st.button("← 回到对话", use_container_width=True):
        st.switch_page("app.py")
    st.markdown("---")
    with st.expander("Span 类型图例"):
        st.markdown(
            """
- 🧠 **GENERATION** — LLM 调用（input/output/model/token）
- 🔧 **TOOL** — 工具执行
- 📦 **SPAN** — 通用 scope（LangGraph node 多是这种）
- 🎯 **EVENT** — 瞬时事件（无持续时间）
"""
        )


# ==================== Data ====================

@st.cache_data(ttl=60, show_spinner="拉 trace 列表...")
def load_recent_traces(trace_name: str, hours: int, limit: int):
    client = get_client()
    if client is None:
        return []
    traces = fetch_traces_by_name(client, trace_name, hours, limit)
    # 取需要的字段，避免 cache 序列化整个 Pydantic 对象
    return [{
        "id": t.id,
        "name": t.name,
        "timestamp": t.timestamp,
        "latency": t.latency or 0.0,
        "input": t.input,
        "session_id": getattr(t, "session_id", None),
    } for t in traces]


@st.cache_data(ttl=300, show_spinner="拉 trace observations...")
def load_trace_obs(trace_id: str):
    client = get_client()
    if client is None:
        return None
    return fetch_observations(client, trace_id)


# ==================== UI ====================

st.title("🔍 Trace Viewer")
st.caption("单条 trace 的完整调用链（Router → Path → Tool → LLM）")

client = get_client()
if client is None:
    st.error("Langfuse 未配置。检查 .env 的 LANGFUSE_* 变量。")
    st.stop()

# ---- Select trace ----
col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    trace_name = st.selectbox("Trace name", ["router_agent", "agentic_agent_standalone"], index=0)
with col2:
    hours = st.selectbox("时间范围", [1, 6, 24, 168],
                        format_func=lambda h: {1: "1h", 6: "6h", 24: "24h", 168: "7d"}[h],
                        index=2)

traces = load_recent_traces(trace_name, hours, 50)
if not traces:
    st.warning(f"近 {hours}h 没有 `{trace_name}` trace")
    st.stop()

# ---- Trace selector ----
def _fmt(t):
    query = ""
    if isinstance(t["input"], dict):
        query = t["input"].get("query", "")
    elif isinstance(t["input"], list) and t["input"]:
        first = t["input"][0]
        if isinstance(first, dict):
            query = first.get("content", "")
    query = (query or "")[:80]
    ts = t["timestamp"].strftime("%H:%M:%S") if t["timestamp"] else "?"
    return f"[{ts}] {t['latency']:.1f}s · {query}..."

with col3:
    idx = st.selectbox("选择一条 trace", range(len(traces)),
                      format_func=lambda i: _fmt(traces[i]))

selected = traces[idx]

# ---- Trace overview ----
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Latency", f"{selected['latency']:.2f}s")

obs_list = load_trace_obs(selected["id"])
if obs_list is None:
    st.error("拉 observations 失败（Langfuse 后端可能慢）。稍后重试。")
    st.stop()

summary = analyze_trace(
    type("T", (), {"id": selected["id"], "input": selected["input"],
                   "latency": selected["latency"], "timestamp": selected["timestamp"]})(),
    obs_list,
)
col2.metric("Route", summary["route"] or "—")
col3.metric("LLM calls", summary["llm_calls"])
col4.metric("Tool calls", summary["tool_calls"])

if summary["cost"] > 0:
    st.caption(f"Total cost: **${summary['cost']:.6f}**  ·  Models: {', '.join(summary['models']) or '—'}")

# ---- Query & session info ----
query = ""
if isinstance(selected["input"], dict):
    query = selected["input"].get("query", "")
elif isinstance(selected["input"], list) and selected["input"]:
    first = selected["input"][0]
    if isinstance(first, dict):
        query = first.get("content", "")

with st.expander(f"📝 Query: {query[:120]}...", expanded=False):
    st.code(query, language="text")
    st.caption(f"trace_id: `{selected['id']}`  ·  session: `{selected['session_id'] or '—'}`")

# ---- Observation tree ----
st.markdown("---")
st.subheader(f"调用链 ({len(obs_list)} 个 span)")

tree = build_observation_tree(obs_list)

TYPE_ICON = {"GENERATION": "🧠", "TOOL": "🔧", "SPAN": "📦", "EVENT": "🎯"}
TYPE_COLOR = {"GENERATION": "#4b9bff", "TOOL": "#fbbc04", "SPAN": "#9aa0a6", "EVENT": "#a142f4"}


def flatten_tree(tree, depth=0, out=None):
    """DFS 展平树到 list of (depth, node)，保证父在前子在后"""
    if out is None:
        out = []
    for node in tree:
        out.append((depth, node))
        flatten_tree(node["children"], depth + 1, out)
    return out


def render_flat_node(depth, node, max_latency):
    """扁平渲染：用缩进线 + 彩色边框容器代替 expander（Streamlit 不支持 expander 嵌套）"""
    o = node["obs"]
    otype = getattr(o, "type", "") or "SPAN"
    icon = TYPE_ICON.get(otype, "•")
    color = TYPE_COLOR.get(otype, "#6b7280")
    name = o.name or "(unnamed)"
    latency = getattr(o, "latency", 0) or 0
    # bar 长度按耗时占比
    bar_pct = min(100, int(latency / max(max_latency, 0.001) * 100)) if max_latency > 0 else 0

    # 缩进线：每层 │ + 分支 ├/└
    indent_bar = "│ " * max(0, depth - 1) + ("├─ " if depth > 0 else "")

    header_html = f"""
    <div style="border-left: 3px solid {color}; padding: 4px 10px;
                margin-left: {depth * 20}px; margin-bottom: 2px;
                background: rgba(155, 155, 155, 0.05);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-family: monospace; font-size: 13px;">
                {icon} <b>{name}</b>
                <span style="color: #9aa0a6; font-size: 11px;">· {otype}</span>
            </span>
            <span style="font-family: monospace; font-size: 12px; color: {color};">
                {latency:.2f}s
            </span>
        </div>
        <div style="height: 3px; background: {color}; opacity: 0.6;
                    width: {bar_pct}%; margin-top: 2px; border-radius: 2px;"></div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    # 详情放 expander 里（扁平结构，不嵌套）—— 只对有详情的 GENERATION / TOOL 显示
    details_key = f"details_{o.id}"
    if otype in ("GENERATION", "TOOL"):
        with st.expander(f"{'  ' * depth}🔍 view details: {name[:40]}", expanded=False):
            if otype == "GENERATION":
                col1, col2, col3 = st.columns(3)
                col1.caption(f"Model: `{getattr(o, 'model', '—')}`")
                usage = getattr(o, "usage", None)
                if usage:
                    inp = getattr(usage, "input", 0) or 0
                    out = getattr(usage, "output", 0) or 0
                    col2.caption(f"Tokens: {inp} in / {out} out")
                if hasattr(o, "calculated_total_cost") and o.calculated_total_cost:
                    col3.caption(f"Cost: ${float(o.calculated_total_cost):.6f}")

            inp_text = getattr(o, "input", None)
            if inp_text:
                st.markdown("**📥 Input**")
                if isinstance(inp_text, (dict, list)):
                    st.json(inp_text)
                else:
                    st.code(str(inp_text)[:3000])
            out_text = getattr(o, "output", None)
            if out_text:
                st.markdown("**📤 Output**")
                if isinstance(out_text, (dict, list)):
                    st.json(out_text)
                else:
                    st.code(str(out_text)[:3000])


flat_nodes = flatten_tree(tree)
max_lat = max((getattr(n["obs"], "latency", 0) or 0 for _, n in flat_nodes), default=0)

for depth, node in flat_nodes:
    render_flat_node(depth, node, max_lat)
