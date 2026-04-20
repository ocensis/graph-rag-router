"""
Streamlit Page: KG Health Dashboard
直接跑 Cypher 查 Neo4j，把 KG 结构/覆盖率/重复/社区可视化
（不依赖 Langfuse，数据从 graphrag_agent 直接读）
"""
import sys
import re
import time
from pathlib import Path
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.express as px

# 让 pages 子目录能 import frontend 本地模块（Streamlit multi-page 场景）
_frontend_root = Path(__file__).parent.parent
if str(_frontend_root) not in sys.path:
    sys.path.insert(0, str(_frontend_root))

st.set_page_config(page_title="KG Health", page_icon="🗺️", layout="wide")


# ==================== Shared sidebar (保证全站一致) ====================

with st.sidebar:
    st.title("🎯 GraphRAG")
    st.caption("Multi-path Agentic RAG  |  RAG Papers KG")
    st.markdown("---")
    st.info("📊 你在 **KG Health** 监控页")
    st.caption("Agent 切换 / 调试开关在 **Chat** 主页")
    if st.button("← 回到对话", use_container_width=True):
        st.switch_page("app.py")
    st.markdown("---")
    with st.expander("关于本项目"):
        st.markdown(
            """
**架构**: LangGraph 3-way classifier router
- Classic (~40%) · hybrid vector+BM25+RRF
- Graph (~48%) · entity + community summary
- Agentic (~12%) · planner + ReAct

**实测 eng200**
- LLM-Acc **0.57** (vs Agentic 0.55)
- 延迟 ↓50% · 成本 ↓57%
"""
        )


# ==================== Data layer ====================

@st.cache_resource
def get_graph():
    from graphrag_agent.config.neo4jdb import get_db_manager
    return get_db_manager().get_graph()


@st.cache_data(ttl=300)
def fetch_scale():
    g = get_graph()
    out = {}
    for key, cypher in [
        ("documents", "MATCH (d:__Document__) RETURN count(d) AS n"),
        ("chunks",    "MATCH (c:__Chunk__) RETURN count(c) AS n"),
        ("entities",  "MATCH (e:__Entity__) RETURN count(e) AS n"),
        ("relationships", "MATCH ()-[r]->() RETURN count(r) AS n"),
    ]:
        out[key] = g.query(cypher)[0]["n"]
    # Contextual Retrieval coverage
    try:
        r = g.query(
            "MATCH (c:__Chunk__) RETURN count(c.original_text) AS orig, count(c.context) AS ctx, count(c) AS total"
        )[0]
        out["chunks_with_context"] = r["ctx"]
        out["chunks_with_original_text"] = r["orig"]
    except Exception:
        out["chunks_with_context"] = 0
    return out


@st.cache_data(ttl=300)
def fetch_entity_labels():
    rows = get_graph().query(
        """
        MATCH (e:__Entity__)
        UNWIND labels(e) AS l
        WITH l WHERE l <> '__Entity__'
        RETURN l AS label, count(*) AS count
        ORDER BY count DESC
        """
    )
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def fetch_rel_types():
    rows = get_graph().query(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
    )
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def fetch_completeness():
    g = get_graph()
    out = {}
    r = g.query(
        """MATCH (e:__Entity__)
           RETURN count(e) AS total,
                  count(CASE WHEN e.description IS NOT NULL AND e.description <> '' THEN 1 END) AS with_desc"""
    )[0]
    out["entity_total"] = r["total"]
    out["with_description"] = r["with_desc"]
    out["description_coverage"] = r["with_desc"] / r["total"] if r["total"] else 0

    r = g.query(
        """MATCH (e:__Entity__)
           WHERE NOT EXISTS { MATCH (e)-[r]-(:__Entity__) }
           RETURN count(e) AS n"""
    )[0]
    out["orphan_entities"] = r["n"]
    out["orphan_rate"] = r["n"] / out["entity_total"] if out["entity_total"] else 0

    r = g.query(
        """MATCH (c:__Chunk__)
           RETURN count(c) AS total,
                  count(CASE WHEN EXISTS { MATCH (c)-[:MENTIONS]->(:__Entity__) } THEN 1 END) AS with_mentions"""
    )[0]
    out["chunk_total"] = r["total"]
    out["chunks_with_mentions"] = r["with_mentions"]
    out["mention_coverage"] = r["with_mentions"] / r["total"] if r["total"] else 0
    return out


@st.cache_data(ttl=300)
def fetch_community_coverage():
    rows = get_graph().query(
        """MATCH (c:__Community__)
           RETURN c.level AS level, count(c) AS total, count(c.summary) AS with_summary
           ORDER BY level"""
    )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["coverage"] = df["with_summary"] / df["total"]
    return df


@st.cache_data(ttl=300)
def fetch_top_entities(limit=20):
    rows = get_graph().query(
        f"""MATCH (e:__Entity__)
            OPTIONAL MATCH (e)-[r]-(:__Entity__)
            WHERE type(r) <> 'MENTIONS' AND type(r) <> 'IN_COMMUNITY'
            WITH e, count(r) AS degree
            ORDER BY degree DESC
            LIMIT {limit}
            RETURN e.id AS entity,
                   [l IN labels(e) WHERE l <> '__Entity__'][0] AS label,
                   degree"""
    )
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def fetch_dup_suspects(threshold=0.8, max_check=2000):
    """名字 3-gram Jaccard ≥ threshold 的实体对"""
    ids = [r["id"] for r in get_graph().query(
        f"MATCH (e:__Entity__) RETURN e.id AS id ORDER BY e.id LIMIT {max_check}"
    ) if r.get("id")]

    def grams(s):
        s = s.lower().strip()
        if len(s) < 3:
            return {s}
        return {s[i : i + 3] for i in range(len(s) - 2)}

    def jaccard(a, b):
        ga, gb = grams(a), grams(b)
        if not ga or not gb:
            return 0.0
        return len(ga & gb) / len(ga | gb)

    # bucket by first-2-char to cut O(N^2)
    buckets = {}
    for i in ids:
        buckets.setdefault(i[:2].lower() if len(i) >= 2 else i.lower(), []).append(i)

    pairs = []
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                s = jaccard(bucket[i], bucket[j])
                if s >= threshold:
                    pairs.append((bucket[i], bucket[j], round(s, 3)))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:30]


# ==================== UI ====================

st.title("🗺️ Knowledge Graph Health")
st.caption("Neo4j KG 规模 / 完整性 / 连通性 / 社区覆盖 / 重复疑点")

if st.button("🔄 刷新数据 (清缓存)"):
    st.cache_data.clear()
    st.rerun()

try:
    scale = fetch_scale()
except Exception as e:
    st.error(f"连接 Neo4j 失败: {e}")
    st.stop()

# ---- Top-line metrics ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Documents", f"{scale['documents']:,}")
col2.metric("Chunks", f"{scale['chunks']:,}",
            help=f"Contextual Retrieval 覆盖: {scale.get('chunks_with_context',0)}/{scale['chunks']}")
col3.metric("Entities", f"{scale['entities']:,}")
col4.metric("Relationships", f"{scale['relationships']:,}")

# Contextual Retrieval indicator
if scale['chunks'] > 0:
    ctx_pct = scale.get('chunks_with_context', 0) / scale['chunks'] * 100
    if ctx_pct >= 90:
        st.success(f"✓ Contextual Retrieval 覆盖 {ctx_pct:.0f}%")
    elif ctx_pct > 0:
        st.warning(f"⚠ Contextual Retrieval 仅覆盖 {ctx_pct:.0f}%（跑 contextualize_chunks.py 补齐）")
    else:
        st.error("✗ Contextual Retrieval 未运行")

st.markdown("---")

# ---- Completeness + Community coverage side by side ----
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("完整性")
    comp = fetch_completeness()
    c1, c2, c3 = st.columns(3)
    c1.metric("Description 覆盖率", f"{comp['description_coverage']*100:.1f}%")
    c2.metric("Orphan 率", f"{comp['orphan_rate']*100:.1f}%",
              delta=f"-{comp['orphan_entities']}", delta_color="inverse")
    c3.metric("Chunk→Entity 覆盖", f"{comp['mention_coverage']*100:.1f}%")
    if comp['orphan_rate'] > 0.15:
        st.caption("⚠ orphan 率 >15%，多数是 Author 只有单条 AUTHORED_BY 边")

with col_right:
    st.subheader("Community 摘要覆盖")
    cov_df = fetch_community_coverage()
    if not cov_df.empty:
        fig = px.bar(
            cov_df, x="level", y="coverage",
            labels={"coverage": "Summary 覆盖率"},
            text=[f"{v*100:.0f}%" for v in cov_df["coverage"]],
        )
        fig.update_layout(yaxis_tickformat=".0%", height=260,
                          margin=dict(t=10, b=10, l=10, r=10))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("没有 __Community__ 节点")

st.markdown("---")

# ---- Entity labels + Relationship types side by side ----
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("实体类型分布 (Top 10)")
    ent_df = fetch_entity_labels().head(10)
    if not ent_df.empty:
        fig = px.bar(ent_df, x="count", y="label", orientation="h")
        fig.update_layout(height=340, yaxis=dict(autorange="reversed"),
                          margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader("关系类型分布 (Top 10)")
    rel_df = fetch_rel_types().head(10)
    if not rel_df.empty:
        # highlight OTHER
        rel_df["color"] = rel_df["type"].apply(lambda t: "OTHER" if t == "OTHER" else "normal")
        fig = px.bar(rel_df, x="count", y="type", orientation="h",
                     color="color",
                     color_discrete_map={"normal": "#4b9bff", "OTHER": "#ea4335"})
        fig.update_layout(height=340, yaxis=dict(autorange="reversed"),
                          showlegend=False,
                          margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        other_cnt = rel_df[rel_df["type"] == "OTHER"]["count"].sum()
        total = rel_df["count"].sum()
        if total > 0 and other_cnt / total > 0.05:
            st.caption(f"⚠ OTHER 占 {other_cnt/total*100:.1f}%，schema 对齐差")

st.markdown("---")

# ---- Top entities by degree ----
st.subheader("Top 20 实体（按度数）")
top_ent = fetch_top_entities(20)
if not top_ent.empty:
    st.dataframe(
        top_ent,
        use_container_width=True,
        hide_index=True,
        column_config={
            "entity": "Entity",
            "label": "Type",
            "degree": st.column_config.ProgressColumn(
                "Degree",
                min_value=0,
                max_value=int(top_ent["degree"].max()),
            ),
        },
    )

st.markdown("---")

# ---- Duplicate suspects ----
st.subheader("疑似重复实体")
threshold = st.slider("Jaccard 阈值", 0.5, 1.0, 0.8, 0.05)
with st.spinner("扫描..."):
    dup_pairs = fetch_dup_suspects(threshold=threshold)

if dup_pairs:
    st.caption(f"找到 {len(dup_pairs)} 对（显示 Top 30）")
    dup_df = pd.DataFrame(dup_pairs, columns=["A", "B", "Jaccard"])
    st.dataframe(dup_df, use_container_width=True, hide_index=True)
    st.caption("💡 可跑 `python _dedup_case_variants.py` 自动合并 case-variant（大小写重复）")
else:
    st.success(f"没有 Jaccard ≥ {threshold} 的可疑实体对")
