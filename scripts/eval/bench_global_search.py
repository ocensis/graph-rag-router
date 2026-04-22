"""
直接跑 GlobalSearchTool.search() 的 A/B bench，绕过 GraphAgent/Router
——因为他们不触发 global_tool，拿不到 trace。

用法:
    # 1. 当前 HEAD = cypher 版
    python scripts/eval/bench_global_search.py --tag cypher --n 3

    # 2. git checkout HEAD~2 -- graphrag_agent/search/tool/global_search_tool.py
    #    切回 map-reduce 版
    python scripts/eval/bench_global_search.py --tag mapreduce --n 3

    # 3. git checkout HEAD -- graphrag_agent/search/tool/global_search_tool.py
    #    恢复 cypher 版

Langfuse 里过滤 tag=global_bench:cypher / global_bench:mapreduce 就能看两组对比。
"""
import sys
import json
import time
import argparse

sys.stdout.reconfigure(encoding="utf-8")

from graphrag_agent.search.tool.global_search_tool import GlobalSearchTool
from graphrag_agent.utils.langfuse_client import get_langfuse_handler, flush_langfuse


# 这些 question type 会触发 broad / cross-doc aggregation——global search 的目标场景
BROAD_TYPES = {
    "enumeration",
    "statistics",
    "topical_grouping",
    "shared_problem_different_solutions",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="eg. 'cypher' or 'mapreduce'")
    ap.add_argument("--n", type=int, default=3, help="要跑几题")
    ap.add_argument(
        "--questions",
        default="benchmarks/eng_cross_doc_questions.json",
        help="问题文件路径",
    )
    args = ap.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        all_qs = json.load(f)

    broad = [q for q in all_qs if q.get("type") in BROAD_TYPES][: args.n]
    print(f"tag={args.tag}  n={len(broad)}  (broad types: {BROAD_TYPES})")

    tool = GlobalSearchTool()
    handler = get_langfuse_handler()
    if handler is None:
        print("⚠ Langfuse handler 为空，trace 不会上传")
        return

    overall_start = time.time()
    per_query = []

    for i, q in enumerate(broad, 1):
        cfg = {
            "callbacks": [handler],
            "run_name": f"global_bench_{args.tag}",
            "metadata": {
                "langfuse_tags": [f"global_bench:{args.tag}", "ab_compare"],
                "langfuse_session_id": f"global_bench_{args.tag}",
                "query_type": q.get("type", "?"),
            },
        }
        qtext = q["question"]
        print(f"\n[{i}/{len(broad)}] ({q.get('type')}) {qtext[:80]}")
        t0 = time.time()
        try:
            ans = tool.search(qtext, parent_config=cfg)
            elapsed = time.time() - t0
            per_query.append({
                "question": qtext[:80],
                "type": q.get("type"),
                "elapsed": elapsed,
                "answer_len": len(ans) if ans else 0,
            })
            print(f"    [{elapsed:.1f}s]  ans_len={len(ans) if ans else 0}")
        except Exception as e:
            print(f"    [FAIL] {e}")
            per_query.append({
                "question": qtext[:80],
                "type": q.get("type"),
                "elapsed": time.time() - t0,
                "error": str(e)[:200],
            })

    total = time.time() - overall_start
    print(f"\n======== summary (tag={args.tag}) ========")
    print(f"Total: {total:.1f}s  (avg {total/len(broad):.1f}s/q)")
    for r in per_query:
        print(f"  {r.get('elapsed', 0):6.1f}s  {r.get('type'):<30}  {r.get('question')[:60]}")

    print("\nflushing langfuse...")
    flush_langfuse()
    print(f"[done] Langfuse UI 过滤 tag=global_bench:{args.tag} 看 trace")


if __name__ == "__main__":
    main()
