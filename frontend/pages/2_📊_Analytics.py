"""
Streamlit Page: Analytics Dashboard
从 Langfuse 拉最近的 router_agent traces，聚合 route 分布 / 延迟 / 成本 / LLM 调用
"""
import sys
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd
import plotly.express as px

_frontend_root = Path(__file__).parent.parent
if str(_frontend_root) not in sys.path:
    sys.path.insert(0, str(_frontend_root))

from utils.langfuse_helpers import (
    get_client,
    fetch_traces_by_name,
    fetch_observations,
    analyze_trace,
)

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")


# ==================== Sidebar ====================

with st.sidebar:
    st.title("🎯 GraphRAG")
    st.caption("Multi-path Agentic RAG  |  RAG Papers KG")
    st.markdown("---")
    st.info("📊 你在 **Analytics** 聚合页")
    st.caption("Agent 切换 / 调试开关在 **Chat** 主页")
    if st.button("← 回到对话", use_container_width=True):
        st.switch_page("app.py")
    st.markdown("---")
    with st.expander("数据来源"):
        st.markdown(
            """
- Langfuse SDK 拉 `router_agent` trace
- 按 `parent_path_span` 判 route (classic / graph / agentic)
- 聚合 latency / cost / LLM calls
- 5 分钟缓存
"""
        )


# ==================== Data layer (cached) ====================

@st.cache_data(ttl=300, show_spinner="拉 Langfuse traces...")
def load_aggregate(trace_name: str, hours: int, limit: int, deep: bool = False) -> pd.DataFrame:
    """
    两档模式:
      - shallow (deep=False): 只拉 trace 列表（1 次 paginated API call），
        仅得到 latency/query/timestamp 这些 trace 层级字段，无 route/cost（快，秒级）
      - deep  (deep=True):  每条 trace 额外调 fetch_observations 做深度分析
        用 ThreadPool 并发 20 条，200 trace 大约 20-40s
    """
    client = get_client()
    if client is None:
        return pd.DataFrame()

    traces = fetch_traces_by_name(client, trace_name, hours, limit)
    if not traces:
        return pd.DataFrame()

    if not deep:
        # shallow: 只用 trace 本身的字段 + 从 tags 读 route
        rows = []
        for t in traces:
            query = ""
            if isinstance(t.input, dict):
                query = t.input.get("query", "")
            elif isinstance(t.input, list) and t.input:
                first = t.input[0]
                if isinstance(first, dict):
                    query = first.get("content", "")
            # 从 trace tags 里读 route (格式: "route:classic" / "route:graph" / "route:agentic")
            route = None
            tags = getattr(t, "tags", None) or []
            for tag in tags:
                if tag.startswith("route:"):
                    route = tag.split(":", 1)[1]
                    break
            rows.append({
                "trace_id": t.id,
                "query": query,
                "latency": t.latency or 0.0,
                "timestamp": getattr(t, "timestamp", None),
                "route": route,
                "llm_calls": 0,
                "tool_calls": 0,
                "tools_used": {},
                "subqueries": 0,
                "cost": 0.0,
                "models": [],
            })
        return pd.DataFrame(rows)

    # deep: 逐 trace 拉 observations
    # 关键: max_workers=5 而非 20。Langfuse ClickHouse 后端并发能力有限，
    # 20 并发大量 timeout 反而更慢（还 burn CPU 在重试上）
    rows = []
    progress = st.progress(0.0)
    total = len(traces)
    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_observations, client, t.id): t for t in traces}
        for f in as_completed(futures):
            t = futures[f]
            obs = f.result()
            if obs is not None:
                rows.append(analyze_trace(t, obs))
            else:
                failed += 1
            done += 1
            progress.progress(done / total)
    progress.empty()
    if failed:
        st.warning(f"有 {failed}/{total} 条 trace 拉 observations 失败（Langfuse 后端超时）")
    return pd.DataFrame(rows)


# ==================== UI ====================

st.title("📊 Analytics Dashboard")
st.caption("从 Langfuse 聚合的 Router 运行时数据（route 分布 / 延迟 / 成本 / 工具调用）")

client = get_client()
if client is None:
    st.error("Langfuse 未配置。检查 .env 的 LANGFUSE_* 变量。")
    st.stop()

# ---- Filter bar ----
col1, col2, col3, col4, col5 = st.columns([1.2, 1, 1, 1.2, 1.2])
with col1:
    trace_name = st.selectbox(
        "Trace name",
        ["router_agent", "agentic_agent_standalone", "naive_rag_search"],
        index=0,
    )
with col2:
    hours = st.selectbox("时间范围", [6, 24, 48, 168, 720],
                        format_func=lambda h: {6: "6h", 24: "24h", 48: "2d", 168: "7d", 720: "30d"}[h],
                        index=1)
with col3:
    limit = st.selectbox("最多 trace 数", [30, 50, 100, 200, 500], index=1)
with col4:
    deep_analysis = st.checkbox(
        "深度分析 (route/cost)", value=False,
        help="需要对每条 trace 额外拉 observations（慢 10x，按 trace 数 × ~100ms 算）",
    )
with col5:
    load_clicked = st.button("▶️ 加载数据", type="primary", use_container_width=True)

# 状态按钮控制：不点就不拉
if "analytics_loaded" not in st.session_state:
    st.session_state.analytics_loaded = False
if load_clicked:
    st.cache_data.clear()   # 强制重拉
    st.session_state.analytics_loaded = True
    st.session_state.analytics_deep = deep_analysis
    st.session_state.analytics_params = (trace_name, hours, limit)

if not st.session_state.analytics_loaded:
    st.info("👈 调整过滤条件后点 **▶️ 加载数据**。"
            "默认只拉 trace 层级的基本指标（1 次 API 调用）；勾选"
            "「深度分析」才会逐条拉 observations 聚合 route / cost（慢很多）。")
    st.stop()

trace_name, hours, limit = st.session_state.analytics_params
deep = st.session_state.analytics_deep

df = load_aggregate(trace_name, hours, limit, deep=deep)

if df.empty:
    st.warning(f"没有拉到 `{trace_name}` 的 trace（近 {hours}h）。可能：1) 该时段没运行 2) Langfuse 后端不可用")
    st.stop()

st.success(f"分析 {len(df)} 条 trace")

# ---- Top metrics ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total traces", f"{len(df):,}")
valid_route = df[df["route"].notna()]
col2.metric("带 route 标记", f"{len(valid_route):,}",
            help=f"{len(valid_route)/len(df)*100:.0f}% traces 解析到 route 标签")
col3.metric("平均 latency", f"{df['latency'].mean():.2f}s")
if df['cost'].sum() > 0:
    col4.metric("总 cost", f"${df['cost'].sum():.4f}",
                help=f"平均 ${df['cost'].mean():.6f}/query")
else:
    col4.metric("平均 LLM 调用", f"{df['llm_calls'].mean():.1f}")

st.markdown("---")

# ---- Charts row 1: route distribution + latency by route ----
if not valid_route.empty:
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Route 分布")
        route_counts = valid_route["route"].value_counts().reset_index()
        route_counts.columns = ["route", "count"]
        fig = px.pie(
            route_counts, values="count", names="route",
            color="route",
            color_discrete_map={"classic": "#9aa0a6", "graph": "#34a853", "agentic": "#a142f4"},
            hole=0.4,
        )
        fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("各 route 的延迟分布")
        lat_df = valid_route.groupby("route").agg(
            mean=("latency", "mean"),
            p50=("latency", lambda x: x.quantile(0.5)),
            p95=("latency", lambda x: x.quantile(0.95)),
            n=("latency", "count"),
        ).reset_index()
        lat_long = lat_df.melt(
            id_vars=["route", "n"], value_vars=["mean", "p50", "p95"],
            var_name="metric", value_name="latency (s)",
        )
        fig = px.bar(
            lat_long, x="route", y="latency (s)", color="metric", barmode="group",
            color_discrete_map={"mean": "#4b9bff", "p50": "#34a853", "p95": "#ea4335"},
        )
        fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

# ---- Charts row 2: cost / LLM calls ----
if not valid_route.empty:
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("各 route 的平均成本 / 调用")
        agg = valid_route.groupby("route").agg(
            avg_cost=("cost", "mean"),
            avg_llm_calls=("llm_calls", "mean"),
            avg_tool_calls=("tool_calls", "mean"),
            n=("latency", "count"),
        ).reset_index()
        # 双 y 轴 bar
        agg_long = agg.melt(
            id_vars=["route", "n"], value_vars=["avg_llm_calls", "avg_tool_calls"],
            var_name="kind", value_name="calls per query",
        )
        fig = px.bar(
            agg_long, x="route", y="calls per query", color="kind", barmode="group",
            color_discrete_map={"avg_llm_calls": "#4b9bff", "avg_tool_calls": "#fbbc04"},
        )
        fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        # Cost 数据单列
        if agg["avg_cost"].sum() > 0:
            cost_table = agg[["route", "avg_cost", "n"]].copy()
            cost_table["avg_cost"] = cost_table["avg_cost"].apply(lambda x: f"${x:.6f}")
            st.caption("平均成本 per query")
            st.dataframe(cost_table, use_container_width=True, hide_index=True)

    with col_r:
        st.subheader("Agentic 内部工具使用")
        agentic_rows = valid_route[valid_route["route"] == "agentic"]
        if agentic_rows.empty:
            st.info("没有 agentic trace")
        else:
            tool_totals = Counter()
            for tools in agentic_rows["tools_used"]:
                if isinstance(tools, dict):
                    for t, c in tools.items():
                        tool_totals[t] += c
            if tool_totals:
                tool_df = pd.DataFrame(
                    [{"tool": t, "count": c} for t, c in tool_totals.most_common()]
                )
                fig = px.bar(tool_df, x="count", y="tool", orientation="h",
                            color_discrete_sequence=["#a142f4"])
                fig.update_layout(height=340, yaxis=dict(autorange="reversed"),
                                margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("该时段 agentic 没调工具")

st.markdown("---")

# ---- Detail table ----
st.subheader("Trace 明细 (按时间降序)")
display_df = df.copy()
if "timestamp" in display_df.columns:
    display_df = display_df.sort_values("timestamp", ascending=False)
# drop noisy columns
display_df = display_df[["timestamp", "query", "route", "latency", "llm_calls", "tool_calls", "cost", "trace_id"]].copy()
display_df["latency"] = display_df["latency"].apply(lambda x: f"{x:.2f}s")
display_df["cost"] = display_df["cost"].apply(lambda x: f"${x:.6f}" if x > 0 else "—")
display_df["query"] = display_df["query"].astype(str).str.slice(0, 80) + "..."
st.dataframe(display_df, use_container_width=True, hide_index=True, height=300)
