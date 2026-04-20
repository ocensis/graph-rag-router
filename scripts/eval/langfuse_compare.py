"""
用 Langfuse 数据对比 4 个 agent（Naive / Graph / Agentic / Router）的
成本-延迟-质量权衡，支撑 "Router 保留 X% 质量、成本降 Y%" 的简历数据。

用法:
  python _langfuse_compare.py                     # 默认最近 24h，各抓 200 条
  python _langfuse_compare.py --hours 6 --limit 300

匹配策略（按 trace name）：
  naive    ← naive_rag_search
  graph    ← graph_agent          (fallback: 含 local_search)
  agentic  ← agentic_graph_search / agentic_hotpot_bench
  router   ← router_agent
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse


# 每个 agent 匹配的 trace name（按优先级尝试）
AGENT_NAMES = {
    "naive":   ["naive_rag_search"],
    "graph":   ["graph_agent", "local_search"],
    "agentic": ["agentic_hotpot_bench", "agentic_graph_search"],
    "router":  ["router_agent"],
}


def fetch_traces_by_name(client, name, from_ts, limit):
    """分页拉指定 name 的 trace"""
    traces = []
    page = 1
    per_page = 100
    while len(traces) < limit:
        resp = client.fetch_traces(
            from_timestamp=from_ts,
            limit=per_page,
            page=page,
            name=name,
        )
        if not resp.data:
            break
        traces.extend(resp.data)
        if len(resp.data) < per_page:
            break
        page += 1
    return traces[:limit]


def fetch_observations(client, trace_id, max_retries=3):
    """拉 trace 下的所有 observations（backend 偶发 500，加重试）"""
    import time
    for attempt in range(max_retries):
        try:
            return client.fetch_observations(trace_id=trace_id).data
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            return None  # 用 None 标记失败，不等于 0 observations
    return None


def analyze_trace(trace, obs_list):
    """从 observations 抽取 LLM 调用数、token、cost"""
    llm_calls = 0
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0
    models = set()
    for o in obs_list:
        if o.type == "GENERATION":
            llm_calls += 1
            if hasattr(o, "usage") and o.usage:
                input_tokens += getattr(o.usage, "input", 0) or 0
                output_tokens += getattr(o.usage, "output", 0) or 0
            if hasattr(o, "calculated_total_cost") and o.calculated_total_cost:
                cost_usd += float(o.calculated_total_cost)
            elif hasattr(o, "total_cost") and o.total_cost:
                cost_usd += float(o.total_cost)
            if hasattr(o, "model") and o.model:
                models.add(o.model)
    return {
        "latency": trace.latency or 0.0,
        "llm_calls": llm_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "models": models,
    }


def percentile(sorted_list, p):
    if not sorted_list:
        return 0
    idx = min(int(len(sorted_list) * p), len(sorted_list) - 1)
    return sorted_list[idx]


def summarize(agent, stats):
    """聚合单 agent 的统计"""
    if not stats:
        return None
    n = len(stats)
    lats = sorted(s["latency"] for s in stats)
    llm_calls = [s["llm_calls"] for s in stats]
    in_toks = [s["input_tokens"] for s in stats]
    out_toks = [s["output_tokens"] for s in stats]
    costs = [s["cost_usd"] for s in stats]
    models = set()
    for s in stats:
        models.update(s["models"])

    return {
        "agent": agent,
        "n": n,
        "lat_avg": sum(lats) / n,
        "lat_p50": percentile(lats, 0.50),
        "lat_p95": percentile(lats, 0.95),
        "llm_calls_avg": sum(llm_calls) / n,
        "llm_calls_total": sum(llm_calls),
        "in_tok_avg": sum(in_toks) / n,
        "out_tok_avg": sum(out_toks) / n,
        "cost_avg": sum(costs) / n,
        "cost_total": sum(costs),
        "models": sorted(models),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24,
                        help="只看最近 N 小时的 trace（默认 24）")
    parser.add_argument("--limit", type=int, default=250,
                        help="每个 agent 最多拉多少条 trace（默认 250）")
    parser.add_argument("--workers", type=int, default=30,
                        help="拉 observations 的并发数")
    parser.add_argument("--agents", type=str, default="naive,agentic,router",
                        help="逗号分隔的 agent 列表（graph 未 instrument Langfuse，默认跳过）")
    args = parser.parse_args()

    client = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_BASE_URL"),
    )

    from_ts = datetime.utcnow() - timedelta(hours=args.hours)
    agents = [a.strip() for a in args.agents.split(",")]

    # ===== 1. 拉每个 agent 的 traces =====
    traces_by_agent = {}
    for agent in agents:
        candidate_names = AGENT_NAMES.get(agent, [agent])
        matched = []
        for name in candidate_names:
            traces = fetch_traces_by_name(client, name, from_ts, args.limit - len(matched))
            if traces:
                matched.extend(traces)
                print(f"[{agent:8s}] name='{name}': +{len(traces)} (累计 {len(matched)})")
            if len(matched) >= args.limit:
                break
        traces_by_agent[agent] = matched[:args.limit]
        if not matched:
            print(f"[{agent:8s}] ⚠ 没拉到 trace")

    # ===== 2. 并行拉 observations 算成本/tokens =====
    summaries = {}
    for agent, traces in traces_by_agent.items():
        if not traces:
            continue
        print(f"\n[{agent}] 分析 {len(traces)} 条 trace 的 observations...")
        stats = []
        failed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as exe:
            futures = {exe.submit(fetch_observations, client, t.id): t for t in traces}
            for i, f in enumerate(as_completed(futures), 1):
                t = futures[f]
                obs = f.result()
                if obs is None:
                    failed += 1
                    continue  # 跳过失败的 trace，不计入统计
                stats.append(analyze_trace(t, obs))
                if i % 50 == 0:
                    print(f"  {i}/{len(traces)}  (失败 {failed})")
        if failed:
            print(f"  ⚠ 有 {failed}/{len(traces)} 条 trace 抓 observations 失败，已跳过")
        summaries[agent] = summarize(agent, stats)

    # ===== 3. 打印对比表 =====
    valid = {a: s for a, s in summaries.items() if s}
    if not valid:
        print("\n所有 agent 都没拉到数据，可能原因：")
        print("  1. trace name 不对，看上面 [xxx] 是否 +0")
        print("  2. --hours 设小了，扩大范围")
        return

    print("\n" + "=" * 92)
    print("AGENT 成本-延迟对比")
    print("=" * 92)
    header = f"{'Agent':10s} {'n':>5s} {'lat_avg':>8s} {'p50':>7s} {'p95':>7s} {'LLM/q':>7s} {'in_tok':>8s} {'out_tok':>8s} {'$/query':>10s}"
    print(header)
    print("-" * 92)
    for agent in agents:
        s = valid.get(agent)
        if not s:
            print(f"{agent:10s} (无数据)")
            continue
        print(f"{s['agent']:10s} {s['n']:>5d} "
              f"{s['lat_avg']:>7.2f}s {s['lat_p50']:>6.2f}s {s['lat_p95']:>6.2f}s "
              f"{s['llm_calls_avg']:>7.2f} "
              f"{s['in_tok_avg']:>8.0f} {s['out_tok_avg']:>8.0f} "
              f"${s['cost_avg']:>8.6f}")

    # ===== 4. 相对对比（以 agentic 为基准） =====
    base = valid.get("agentic")
    if base:
        print("\n" + "=" * 60)
        print("相对 Agentic（baseline）")
        print("=" * 60)
        print(f"{'Agent':10s} {'lat%':>8s} {'cost%':>8s} {'LLM%':>8s}")
        print("-" * 60)
        for agent in agents:
            s = valid.get(agent)
            if not s:
                continue
            lat_ratio = s["lat_avg"] / base["lat_avg"] * 100 if base["lat_avg"] else 0
            cost_ratio = s["cost_avg"] / base["cost_avg"] * 100 if base["cost_avg"] else 0
            llm_ratio = s["llm_calls_avg"] / base["llm_calls_avg"] * 100 if base["llm_calls_avg"] else 0
            print(f"{agent:10s} {lat_ratio:>7.1f}% {cost_ratio:>7.1f}% {llm_ratio:>7.1f}%")

    # ===== 5. 叠加质量数据（从 bench_results 读） =====
    import json
    quality_files = {
        "naive":   "bench_results/hotpot_naive_eng200_en_eval.json",
        "graph":   "bench_results/hotpot_graph_eng200_en_eval.json",
        "agentic": "bench_results/hotpot_agentic_eng200_en_n200_eval.json",
        "router":  "bench_results/hotpot_router_eng200_final_eval.json",
    }
    print("\n" + "=" * 60)
    print("完整权衡表（成本 vs 质量）")
    print("=" * 60)
    print(f"{'Agent':10s} {'LLM-Acc':>8s} {'lat':>8s} {'$/q':>10s} {'LLM/q':>7s}")
    print("-" * 60)
    for agent in agents:
        s = valid.get(agent)
        if not s:
            continue
        acc = 0.0
        qf = quality_files.get(agent)
        if qf and os.path.exists(qf):
            d = json.load(open(qf, encoding="utf-8"))
            if "overall" in d and isinstance(d["overall"], dict):
                acc = d["overall"].get("llm_accuracy", 0)
            else:
                # fallback: 从 details 算
                ds = d.get("details", [])
                if ds:
                    acc = sum(r.get("llm_accuracy", 0) for r in ds) / len(ds)
        print(f"{agent:10s} {acc:>7.3f} "
              f"{s['lat_avg']:>7.2f}s "
              f"${s['cost_avg']:>8.6f} "
              f"{s['llm_calls_avg']:>7.2f}")

    # Router vs Agentic 核心卖点
    if valid.get("router") and valid.get("agentic"):
        r, a = valid["router"], valid["agentic"]
        r_acc = json.load(open(quality_files["router"], encoding="utf-8"))["overall"]["llm_accuracy"]
        a_acc = json.load(open(quality_files["agentic"], encoding="utf-8"))["overall"]["llm_accuracy"]
        print("\n" + "=" * 60)
        print("ROUTER vs AGENTIC 简历卖点")
        print("=" * 60)
        print(f"  质量保留:    {r_acc/a_acc*100:.1f}% ({r_acc:.3f} vs {a_acc:.3f})")
        print(f"  延迟降低:    {(1 - r['lat_avg']/a['lat_avg'])*100:.1f}%")
        print(f"  成本降低:    {(1 - r['cost_avg']/a['cost_avg'])*100:.1f}%")
        print(f"  LLM 调用降低: {(1 - r['llm_calls_avg']/a['llm_calls_avg'])*100:.1f}%")


if __name__ == "__main__":
    main()
