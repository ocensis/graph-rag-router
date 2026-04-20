"""
行为分析 —— 从 Langfuse trace 里还原 Router 和 Agentic 的实际执行路径

两个问题:
  1. Router: 每类问题实际走了哪条 path？每条 path 平均多少步？
  2. Agentic (独跑): 每道题做了几轮 retrieve？改写发生率？

用法:
  python _behavior_analysis.py                 # 最近 24h router + agentic
  python _behavior_analysis.py --hours 6       # 只看最近 6h
  python _behavior_analysis.py --router-only   # 只分析 router
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse


# ==================== Langfuse helpers ====================

def _client():
    return Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_BASE_URL"),
    )


def fetch_traces_by_name(client, name, hours, limit):
    import time as _time
    from_ts = datetime.utcnow() - timedelta(hours=hours)
    traces, page = [], 1
    while len(traces) < limit:
        resp = None
        # 重试 3 次应对 ClickHouse 偶发 500
        for attempt in range(3):
            try:
                resp = client.fetch_traces(from_timestamp=from_ts, limit=100, page=page, name=name)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [!] fetch_traces page={page} 最终失败: {e}")
                    return traces[:limit]
                _time.sleep(1.0 * (attempt + 1))
        if not resp or not resp.data:
            break
        traces.extend(resp.data)
        if len(resp.data) < 100:
            break
        page += 1
    return traces[:limit]


def fetch_obs(client, trace_id, retries=3):
    import time
    for i in range(retries):
        try:
            return client.fetch_observations(trace_id=trace_id).data
        except Exception:
            if i < retries - 1:
                time.sleep(0.5 * (i + 1))
    return None


# ==================== Router behavior ====================

ROUTER_PATH_NAMES = {"classic_path", "graph_path", "agentic_path"}
ROUTER_SUB_NAMES = {
    "classifier", "classic_path", "graph_path", "agentic_path",
    "detect_mode", "extract_entities", "local_retrieve", "local_compose",
    "global_search", "global_map_reduce", "hybrid_grounding", "graph_compose",
    "planner", "aggregator", "self_check", "check_naive", "check_graph",
}


def analyze_router_trace(trace, obs_list):
    """从 trace 观测出路径选择 + 行为指标"""
    route = None
    path_latency = 0.0
    n_tool_calls = 0             # ReAct 里的真工具调用
    n_subqueries = 0             # agentic path 的子问题数
    n_llm_calls = 0              # GENERATION spans
    cost = 0.0
    tools_used = Counter()
    models = set()

    # 先按名字过一遍找 route
    for o in obs_list or []:
        name = getattr(o, "name", "") or ""
        if name in ROUTER_PATH_NAMES and route is None:
            route = name.replace("_path", "")
        # 统计 GENERATION 调用
        if getattr(o, "type", "") == "GENERATION":
            n_llm_calls += 1
            if hasattr(o, "calculated_total_cost") and o.calculated_total_cost:
                cost += float(o.calculated_total_cost)
            elif hasattr(o, "total_cost") and o.total_cost:
                cost += float(o.total_cost)
            if hasattr(o, "model") and o.model:
                models.add(o.model)
        # Agentic 子问题个数 = span 里名字叫 subquery_N 的个数
        if name.startswith("subquery_"):
            n_subqueries += 1
        # 工具调用：primitives 里的工具名
        for tool_n in ("hybrid_search", "graph_lookup", "path_search",
                       "fetch_document", "entity_search",
                       "classic_search", "graph_search"):
            if name == tool_n or name.startswith(f"{tool_n}("):
                n_tool_calls += 1
                tools_used[tool_n] += 1
                break

    return {
        "trace_id": trace.id,
        "query": (trace.input or {}).get("query", "") if isinstance(trace.input, dict) else "",
        "latency": trace.latency or 0.0,
        "route": route,
        "n_llm_calls": n_llm_calls,
        "n_tool_calls": n_tool_calls,
        "n_subqueries": n_subqueries,
        "cost": cost,
        "tools_used": dict(tools_used),
        "models": list(models),
    }


# ==================== Agentic (standalone bench) behavior ====================

def analyze_agentic_trace(trace, obs_list):
    """独跑 agentic 的 trace：数 retrieve 次数 = 轮次"""
    n_retrieve = 0
    n_rewrite = 0
    n_llm_calls = 0
    cost = 0.0
    for o in obs_list or []:
        name = getattr(o, "name", "") or ""
        if name == "retrieve":
            n_retrieve += 1
        elif name == "rewrite":
            n_rewrite += 1
        if getattr(o, "type", "") == "GENERATION":
            n_llm_calls += 1
            if hasattr(o, "calculated_total_cost") and o.calculated_total_cost:
                cost += float(o.calculated_total_cost)
    return {
        "trace_id": trace.id,
        "latency": trace.latency or 0.0,
        "rounds": max(1, n_retrieve),
        "rewrites": n_rewrite,
        "n_llm_calls": n_llm_calls,
        "cost": cost,
    }


# ==================== Cross-ref with bench result ====================

def load_bench_question_types(bench_file):
    """query → question_type 映射"""
    if not os.path.exists(bench_file):
        return {}
    try:
        d = json.load(open(bench_file, encoding="utf-8"))
        return {r["question"]: r["type"] for r in d.get("details", [])}
    except Exception as e:
        print(f"  [warn] load {bench_file}: {e}")
        return {}


# ==================== Reports ====================

def percentile(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(int(len(xs) * p), len(xs) - 1)]


def print_router_report(stats, q_type_map):
    valid = [s for s in stats if s["route"]]
    if not valid:
        print("无 router 数据")
        return

    print(f"\n{'='*72}")
    print(f"ROUTER 行为分析  (n={len(valid)} traces)")
    print(f"{'='*72}")

    # 1) Route 分布
    route_counts = Counter(s["route"] for s in valid)
    total = sum(route_counts.values())
    print("\n── Route 分布 ──")
    for r in ("classic", "graph", "agentic"):
        n = route_counts.get(r, 0)
        pct = n / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {r:8s}  {n:4d}  {pct:5.1f}%  {bar}")

    # 2) 每条 route 的行为（latency / LLM calls / tools）
    print("\n── 每条 route 的行为指标 ──")
    print(f"  {'route':8s} {'n':>4s} {'lat_avg':>8s} {'lat_p50':>8s} {'lat_p95':>8s} {'LLM/q':>7s} {'tools/q':>8s} {'subq/q':>7s} {'$/q':>10s}")
    for r in ("classic", "graph", "agentic"):
        sub = [s for s in valid if s["route"] == r]
        if not sub:
            continue
        lats = [s["latency"] for s in sub]
        llm = [s["n_llm_calls"] for s in sub]
        tools = [s["n_tool_calls"] for s in sub]
        subq = [s["n_subqueries"] for s in sub]
        costs = [s["cost"] for s in sub]
        print(f"  {r:8s} {len(sub):>4d} {sum(lats)/len(lats):>7.2f}s "
              f"{percentile(lats, 0.5):>7.2f}s {percentile(lats, 0.95):>7.2f}s "
              f"{sum(llm)/len(llm):>7.2f} {sum(tools)/len(tools):>8.2f} "
              f"{sum(subq)/len(subq):>7.2f} ${sum(costs)/len(costs):>8.6f}")

    # 3) Agentic path 里的工具用法
    agentic_stats = [s for s in valid if s["route"] == "agentic"]
    if agentic_stats:
        tool_totals = Counter()
        for s in agentic_stats:
            for t, c in s.get("tools_used", {}).items():
                tool_totals[t] += c
        print("\n── Agentic path 内部工具调用分布 ──")
        total_calls = sum(tool_totals.values()) or 1
        for t, c in tool_totals.most_common():
            print(f"  {t:20s}  {c:4d}  ({c/total_calls*100:5.1f}%)")

    # 4) Route × question_type 交叉表
    if q_type_map:
        print("\n── Route × Question Type (上行=route, 列=type) ──")
        cross = defaultdict(lambda: Counter())
        matched = 0
        for s in valid:
            t = q_type_map.get(s["query"])
            if t:
                matched += 1
                cross[t][s["route"]] += 1
        if cross:
            print(f"  (matched {matched}/{len(valid)} traces to bench questions)")
            # 列宽对齐
            print(f"  {'type':40s} {'classic':>8s} {'graph':>8s} {'agentic':>8s}")
            for t in sorted(cross.keys()):
                c = cross[t]
                tot = sum(c.values()) or 1
                print(f"  {t:40s} "
                      f"{c['classic']:>3d}({c['classic']/tot*100:>3.0f}%) "
                      f"{c['graph']:>3d}({c['graph']/tot*100:>3.0f}%) "
                      f"{c['agentic']:>3d}({c['agentic']/tot*100:>3.0f}%)")


def print_agentic_report(stats):
    if not stats:
        print("无 agentic 独跑数据")
        return
    print(f"\n{'='*72}")
    print(f"AGENTIC (独跑) 行为分析  (n={len(stats)} traces)")
    print(f"{'='*72}")

    # 轮次分布
    round_counts = Counter(s["rounds"] for s in stats)
    print("\n── 轮次分布 ──")
    total = len(stats)
    for r in sorted(round_counts):
        n = round_counts[r]
        pct = n / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {r} 轮  {n:4d}  {pct:5.1f}%  {bar}")

    # 改写触发率
    with_rewrite = sum(1 for s in stats if s["rewrites"] > 0)
    print(f"\n触发改写（≥1 rewrite）: {with_rewrite}/{total} = {with_rewrite/total*100:.1f}%")

    # 按轮次的延迟 & 成本
    print("\n── 按轮次分组的延迟 / 成本 ──")
    print(f"  {'rounds':>6s} {'n':>4s} {'lat_avg':>8s} {'$/q':>10s}")
    by_r = defaultdict(list)
    for s in stats:
        by_r[s["rounds"]].append(s)
    for r in sorted(by_r):
        sub = by_r[r]
        lats = [s["latency"] for s in sub]
        costs = [s["cost"] for s in sub]
        print(f"  {r:>6d} {len(sub):>4d} {sum(lats)/len(lats):>7.2f}s ${sum(costs)/len(costs):>8.6f}")


# ==================== Main ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--router-only", action="store_true")
    ap.add_argument("--agentic-only", action="store_true")
    ap.add_argument("--router-bench",
                    default="bench_results/hotpot_router_eng200_3way_v2_eval.json",
                    help="Router bench result 用于 query→type 交叉")
    args = ap.parse_args()

    client = _client()
    q_type_map = load_bench_question_types(args.router_bench)
    print(f"question_type 映射加载 {len(q_type_map)} 条")

    # ====== Router ======
    if not args.agentic_only:
        print(f"\n拉 router_agent traces (≤{args.hours}h, limit {args.limit})...")
        router_traces = fetch_traces_by_name(client, "router_agent", args.hours, args.limit)
        print(f"  得 {len(router_traces)} 条")

        if router_traces:
            print("  并行拉 observations...")
            router_stats = []
            failed = 0
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(fetch_obs, client, t.id): t for t in router_traces}
                for i, f in enumerate(as_completed(futs), 1):
                    t = futs[f]
                    obs = f.result()
                    if obs is None:
                        failed += 1
                        continue
                    router_stats.append(analyze_router_trace(t, obs))
                    if i % 50 == 0:
                        print(f"    {i}/{len(router_traces)}  (失败 {failed})")
            if failed:
                print(f"  ⚠ 失败 {failed} 条")
            print_router_report(router_stats, q_type_map)

    # ====== Agentic standalone ======
    if not args.router_only:
        print(f"\n拉 agentic_hotpot_bench traces...")
        ag_traces = fetch_traces_by_name(client, "agentic_hotpot_bench", args.hours, args.limit)
        print(f"  得 {len(ag_traces)} 条")

        if ag_traces:
            print("  并行拉 observations...")
            ag_stats = []
            failed = 0
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(fetch_obs, client, t.id): t for t in ag_traces}
                for i, f in enumerate(as_completed(futs), 1):
                    t = futs[f]
                    obs = f.result()
                    if obs is None:
                        failed += 1
                        continue
                    ag_stats.append(analyze_agentic_trace(t, obs))
                    if i % 50 == 0:
                        print(f"    {i}/{len(ag_traces)}  (失败 {failed})")
            if failed:
                print(f"  ⚠ 失败 {failed} 条")
            print_agentic_report(ag_stats)


if __name__ == "__main__":
    main()
