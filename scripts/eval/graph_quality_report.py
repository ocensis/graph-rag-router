"""
KG 质量报告 —— 跑一遍 Cypher + 采样 LLM judge，
输出结构化指标，回答"你的图质量怎么样"这个面试问题。

用法:
  python _graph_quality_report.py                  # 基础指标，不跑 LLM judge
  python _graph_quality_report.py --sample 50      # 随机采 50 条边让 LLM 判对错
  python _graph_quality_report.py --out report.md  # 输出 Markdown

指标（参考企业级 KG 健康度清单）:
  - 规模:          节点/边总数、类型分布
  - 完整性:        description 覆盖率、孤儿节点率
  - 连通性:        平均度、最大连通分量占比
  - 去重疑点:      名字 Jaccard > 0.8 的实体对
  - 关系语义:      OTHER 占比（过高=schema 抽取没对齐）
  - 精度采样:      LLM judge 随机采样边的事实性
"""
import os
import sys
import json
import time
import argparse
import random
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()


def query(graph, cypher, params=None):
    """统一执行，失败返回 []"""
    try:
        return graph.query(cypher, params or {})
    except Exception as e:
        print(f"  [!] Cypher 失败: {e}", file=sys.stderr)
        return []


def jaccard(a: str, b: str) -> float:
    """字符级 3-gram Jaccard，找拼写相似的实体对"""
    def grams(s):
        s = s.lower().strip()
        if len(s) < 3:
            return {s}
        return {s[i : i + 3] for i in range(len(s) - 2)}

    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


# ==================== 指标采集 ====================

def collect_scale(graph):
    """规模"""
    out = {}
    r = query(graph, "MATCH (e:__Entity__) RETURN count(e) AS n")
    out["entity_total"] = r[0]["n"] if r else 0
    r = query(graph, "MATCH (c:__Chunk__) RETURN count(c) AS n")
    out["chunk_total"] = r[0]["n"] if r else 0
    r = query(graph, "MATCH (d:__Document__) RETURN count(d) AS n")
    out["document_total"] = r[0]["n"] if r else 0
    r = query(graph, "MATCH ()-[r]->() RETURN count(r) AS n")
    out["relationship_total"] = r[0]["n"] if r else 0
    return out


def collect_type_distribution(graph):
    """实体 / 关系类型分布"""
    out = {}
    # 关系类型
    r = query(graph, "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n ORDER BY n DESC")
    out["rel_types"] = [{"type": x["t"], "count": x["n"]} for x in r]
    # 实体 secondary label（除 __Entity__ 外）
    r = query(
        graph,
        """
        MATCH (e:__Entity__)
        UNWIND labels(e) AS l
        WITH l WHERE l <> '__Entity__'
        RETURN l AS label, count(*) AS n ORDER BY n DESC
        """,
    )
    out["entity_labels"] = [{"label": x["label"], "count": x["n"]} for x in r]
    return out


def collect_completeness(graph):
    """完整性：description 覆盖率、孤儿节点"""
    out = {}
    r = query(
        graph,
        """
        MATCH (e:__Entity__)
        RETURN count(e) AS total,
               count(CASE WHEN e.description IS NOT NULL AND e.description <> '' THEN 1 END) AS with_desc
        """,
    )
    if r:
        out["entity_total"] = r[0]["total"]
        out["with_description"] = r[0]["with_desc"]
        out["description_coverage"] = (
            r[0]["with_desc"] / r[0]["total"] if r[0]["total"] else 0
        )

    # 孤儿实体：没有任何关系边（不含 MENTIONS 之类的元关系）
    r = query(
        graph,
        """
        MATCH (e:__Entity__)
        WHERE NOT EXISTS { MATCH (e)-[r]-(:__Entity__) }
        RETURN count(e) AS n
        """,
    )
    out["orphan_entities"] = r[0]["n"] if r else 0
    out["orphan_rate"] = (
        out["orphan_entities"] / out["entity_total"] if out.get("entity_total") else 0
    )

    # MENTIONS 覆盖：有多少 chunk mentions 至少一个 entity
    r = query(
        graph,
        """
        MATCH (c:__Chunk__)
        RETURN count(c) AS total,
               count(CASE WHEN EXISTS { MATCH (c)-[:MENTIONS]->(:__Entity__) } THEN 1 END) AS with_mentions
        """,
    )
    if r:
        out["chunk_total"] = r[0]["total"]
        out["chunks_with_mentions"] = r[0]["with_mentions"]
        out["mention_coverage"] = (
            r[0]["with_mentions"] / r[0]["total"] if r[0]["total"] else 0
        )
    return out


def collect_connectivity(graph):
    """连通性：平均度、度分布"""
    out = {}
    r = query(
        graph,
        """
        MATCH (e:__Entity__)
        OPTIONAL MATCH (e)-[r]-(:__Entity__)
        WITH e, count(r) AS deg
        RETURN avg(deg) AS avg_deg,
               percentileCont(toFloat(deg), 0.5) AS p50,
               percentileCont(toFloat(deg), 0.95) AS p95,
               max(deg) AS max_deg,
               count(CASE WHEN deg = 0 THEN 1 END) AS isolates
        """,
    )
    if r:
        out["avg_degree"] = round(r[0]["avg_deg"] or 0, 2)
        out["degree_p50"] = round(r[0]["p50"] or 0, 2)
        out["degree_p95"] = round(r[0]["p95"] or 0, 2)
        out["degree_max"] = r[0]["max_deg"] or 0
        out["degree_zero_nodes"] = r[0]["isolates"] or 0

    # 社区模块度（如果 community 属性存在）
    r = query(
        graph,
        """
        MATCH (e:__Entity__)
        WHERE e.community IS NOT NULL
        RETURN count(DISTINCT e.community) AS n_communities, count(e) AS n_with_comm
        """,
    )
    if r:
        out["community_count"] = r[0]["n_communities"]
        out["entities_with_community"] = r[0]["n_with_comm"]
    return out


def collect_dup_suspects(graph, threshold=0.8, max_check=2000):
    """
    去重疑点：名字 Jaccard > threshold 的实体对
    为避免 O(N²) 爆炸，只在前 max_check 个实体里找
    """
    r = query(
        graph,
        f"MATCH (e:__Entity__) RETURN e.id AS id ORDER BY e.id LIMIT {max_check}",
    )
    ids = [x["id"] for x in r if x.get("id")]

    if len(ids) < 2:
        return {"checked": len(ids), "suspect_pairs": 0, "samples": []}

    pairs = []
    # 按首字符分桶加速，只比较同首字母的对
    buckets = {}
    for i in ids:
        key = i[:2].lower() if len(i) >= 2 else i.lower()
        buckets.setdefault(key, []).append(i)

    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                s = jaccard(bucket[i], bucket[j])
                if s >= threshold:
                    pairs.append((bucket[i], bucket[j], round(s, 3)))
    # 只保留 top 20 样例
    pairs.sort(key=lambda x: -x[2])
    return {
        "checked": len(ids),
        "threshold": threshold,
        "suspect_pairs": len(pairs),
        "samples": [{"a": a, "b": b, "jaccard": s} for a, b, s in pairs[:20]],
    }


def collect_other_relation(graph):
    """OTHER 占比：过高说明 schema 抽取对齐差"""
    r = query(
        graph,
        """
        MATCH ()-[r]->()
        RETURN sum(CASE WHEN type(r) = 'OTHER' THEN 1 ELSE 0 END) AS other,
               count(r) AS total
        """,
    )
    if not r:
        return {}
    other = r[0]["other"] or 0
    total = r[0]["total"] or 1
    return {
        "other_count": other,
        "total": total,
        "other_ratio": round(other / total, 4),
    }


# ==================== 采样 LLM judge ====================

def judge_precision(graph, n_sample=30):
    """随机采 n 条边，让 LLM 判断是否事实合理"""
    try:
        from graphrag_agent.models.get_models import get_llm_model
    except Exception as e:
        print(f"  [!] LLM 不可用，跳过精度采样: {e}", file=sys.stderr)
        return None

    llm = get_llm_model()

    # 拉随机样本（用 rand() 排序）
    rows = query(
        graph,
        f"""
        MATCH (a:__Entity__)-[r]->(b:__Entity__)
        WHERE type(r) <> 'MENTIONS'
        RETURN a.id AS a_id,
               coalesce(a.description, '') AS a_desc,
               type(r) AS rel,
               b.id AS b_id,
               coalesce(b.description, '') AS b_desc,
               rand() AS rnd
        ORDER BY rnd
        LIMIT {n_sample}
        """,
    )

    if not rows:
        return {"sampled": 0, "note": "no relationships found"}

    def judge_one(row):
        prompt = f"""You are a strict fact-checker evaluating a relationship edge in a knowledge graph about academic RAG papers.

Edge:
  ({row['a_id']}) --[{row['rel']}]--> ({row['b_id']})

Entity descriptions:
  A: {(row['a_desc'] or '(no description)')[:300]}
  B: {(row['b_desc'] or '(no description)')[:300]}

Is this relationship plausibly correct given the descriptions and general domain knowledge?

Rules:
- "correct": the relation type makes sense AND both entities are real/recognizable concepts
- "incorrect": relation contradicts descriptions, or entities are garbage/hallucinated, or mismatch
- "unclear": insufficient info to judge (don't mark too many as this)

Respond with ONE word: correct / incorrect / unclear
Response:"""
        try:
            resp = llm.invoke(prompt)
            content = resp.content.strip().lower() if resp.content else ""
            if "incorrect" in content:
                return "incorrect"
            if "correct" in content:
                return "correct"
            return "unclear"
        except Exception:
            return "unclear"

    # 并发判
    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(judge_one, r): r for r in rows}
        for f in as_completed(futures):
            results.append({"row": futures[f], "verdict": f.result()})

    verdicts = Counter(r["verdict"] for r in results)
    total = len(results)
    correct = verdicts["correct"]
    incorrect = verdicts["incorrect"]
    unclear = verdicts["unclear"]
    judged = correct + incorrect
    precision = correct / judged if judged else 0

    # 留几个样例作诊断
    samples = []
    for r in results[:5]:
        samples.append(
            {
                "edge": f"({r['row']['a_id']}) -[{r['row']['rel']}]-> ({r['row']['b_id']})",
                "verdict": r["verdict"],
            }
        )

    return {
        "sampled": total,
        "correct": correct,
        "incorrect": incorrect,
        "unclear": unclear,
        "precision_on_judged": round(precision, 3),
        "samples": samples,
    }


# ==================== 输出 ====================

def format_markdown(report: dict) -> str:
    """把 dict 渲染成 Markdown 报告"""
    lines = []
    lines.append(f"# KG Quality Report")
    lines.append(f"")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append(f"")

    s = report["scale"]
    lines.append("## 1. Scale")
    lines.append(f"- Entities:        **{s.get('entity_total', 0):,}**")
    lines.append(f"- Chunks:          **{s.get('chunk_total', 0):,}**")
    lines.append(f"- Documents:       **{s.get('document_total', 0):,}**")
    lines.append(f"- Relationships:   **{s.get('relationship_total', 0):,}**")
    lines.append("")

    t = report.get("types", {})
    lines.append("## 2. Type Distribution")
    lines.append("### Relationship types (top 10)")
    lines.append("| Type | Count |")
    lines.append("|---|---:|")
    for x in (t.get("rel_types") or [])[:10]:
        lines.append(f"| {x['type']} | {x['count']:,} |")
    lines.append("")
    lines.append("### Entity labels (top 10)")
    lines.append("| Label | Count |")
    lines.append("|---|---:|")
    for x in (t.get("entity_labels") or [])[:10]:
        lines.append(f"| {x['label']} | {x['count']:,} |")
    lines.append("")

    c = report["completeness"]
    lines.append("## 3. Completeness")
    lines.append(f"- Description coverage:  **{c.get('description_coverage', 0)*100:.1f}%** ({c.get('with_description',0):,}/{c.get('entity_total',0):,})")
    lines.append(f"- Orphan entities:       **{c.get('orphan_entities', 0):,}** ({c.get('orphan_rate',0)*100:.1f}%)")
    lines.append(f"- Chunks with MENTIONS:  **{c.get('mention_coverage', 0)*100:.1f}%** ({c.get('chunks_with_mentions',0):,}/{c.get('chunk_total',0):,})")
    lines.append("")

    co = report["connectivity"]
    lines.append("## 4. Connectivity")
    lines.append(f"- Avg degree:       **{co.get('avg_degree', 0)}**")
    lines.append(f"- Degree p50/p95/max: {co.get('degree_p50', 0)} / {co.get('degree_p95', 0)} / {co.get('degree_max', 0)}")
    lines.append(f"- Isolate nodes:    {co.get('degree_zero_nodes', 0):,}")
    if co.get("community_count"):
        lines.append(f"- Communities:      {co['community_count']:,}  ({co.get('entities_with_community',0):,} entities assigned)")
    lines.append("")

    o = report.get("other_rel", {})
    lines.append("## 5. Relationship Semantic Health")
    lines.append(f"- `OTHER` relation ratio: **{o.get('other_ratio', 0)*100:.2f}%** ({o.get('other_count',0):,}/{o.get('total',0):,})")
    lines.append(f"  - Interpretation: high OTHER% means LLM extraction wasn't aligned to schema well")
    lines.append("")

    d = report["dup_suspects"]
    lines.append("## 6. Duplicate Suspects (name Jaccard ≥ 0.8)")
    lines.append(f"- Checked first {d.get('checked', 0):,} entities → **{d.get('suspect_pairs', 0)} suspect pairs**")
    if d.get("samples"):
        lines.append("")
        lines.append("| A | B | Jaccard |")
        lines.append("|---|---|---:|")
        for s in d["samples"][:10]:
            lines.append(f"| {s['a']} | {s['b']} | {s['jaccard']} |")
    lines.append("")

    p = report.get("precision")
    if p:
        lines.append("## 7. Relationship Precision (LLM-sampled)")
        lines.append(f"- Sampled: **{p.get('sampled', 0)}** non-MENTIONS edges")
        lines.append(f"- Correct: {p.get('correct', 0)} | Incorrect: {p.get('incorrect', 0)} | Unclear: {p.get('unclear', 0)}")
        lines.append(f"- **Precision on judged: {p.get('precision_on_judged', 0)*100:.1f}%**")
        if p.get("samples"):
            lines.append("")
            lines.append("Examples:")
            for sx in p["samples"]:
                lines.append(f"- `{sx['edge']}` → *{sx['verdict']}*")
        lines.append("")

    return "\n".join(lines)


# ==================== Main ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="LLM-judge 采样多少条边（0=跳过，默认 0）")
    ap.add_argument("--jaccard-threshold", type=float, default=0.8)
    ap.add_argument("--out", type=str, default=None, help="输出 Markdown 到文件（同时会生成同名 .json）")
    args = ap.parse_args()

    # 连接 Neo4j：复用项目的 LocalSearchTool 里的 graph wrapper (langchain Neo4jGraph)
    print("Connecting to Neo4j...")
    from graphrag_agent.config.neo4jdb import get_db_manager
    dbm = get_db_manager()

    # Neo4jGraph（langchain 里的包装），search/tool/base 也用的是它
    from langchain_neo4j import Neo4jGraph
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )

    report = {"generated_at": datetime.now().isoformat(timespec="seconds")}

    print("[1/6] Scale...")
    report["scale"] = collect_scale(graph)

    print("[2/6] Type distribution...")
    report["types"] = collect_type_distribution(graph)

    print("[3/6] Completeness...")
    report["completeness"] = collect_completeness(graph)

    print("[4/6] Connectivity...")
    report["connectivity"] = collect_connectivity(graph)

    print("[5/6] OTHER relation share...")
    report["other_rel"] = collect_other_relation(graph)

    print(f"[6/6] Duplicate suspects (Jaccard ≥ {args.jaccard_threshold})...")
    report["dup_suspects"] = collect_dup_suspects(graph, threshold=args.jaccard_threshold)

    if args.sample > 0:
        print(f"[7/6] LLM-judge precision on {args.sample} sampled edges...")
        report["precision"] = judge_precision(graph, n_sample=args.sample)

    # 打印摘要
    md = format_markdown(report)
    print()
    print(md)

    # 保存
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        json_path = args.out.replace(".md", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n已保存: {args.out} + {json_path}")


if __name__ == "__main__":
    main()
