"""
重跑社区摘要 —— 分层版（借鉴微软 GraphRAG）

在现有 KG 上重跑，不需要删数据。会覆盖所有 __Community__ 节点的 summary 字段。

用法:
  python _rebuild_hierarchical_summaries.py --dry-run           # 只统计预期调用数，不真跑
  python _rebuild_hierarchical_summaries.py                     # 摘 level 0/2/4（默认 skip 1/3）
  python _rebuild_hierarchical_summaries.py --all-levels        # 摘全部 0/1/2/3/4
  python _rebuild_hierarchical_summaries.py --max-level 2       # 只到 level 2
  python _rebuild_hierarchical_summaries.py --min-entities 3    # 只摘 ≥3 实体的社区
  python _rebuild_hierarchical_summaries.py --workers 12
"""
import os
import sys
import argparse
import time

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all-levels", action="store_true", help="摘所有层（默认跳 1/3）")
    ap.add_argument("--max-level", type=int, default=4)
    ap.add_argument("--min-entities", type=int, default=2)
    ap.add_argument("--workers", type=int, default=None,
                    help="并行 LLM 调用数，默认用 settings.MAX_WORKERS")
    ap.add_argument("--only-levels", type=str, default=None,
                    help="只跑指定层级（逗号分隔，如 '2,4'），跳过其他。用于已补齐 level 0 后只补 2/4")
    args = ap.parse_args()

    only_levels = None
    if args.only_levels:
        only_levels = {int(x.strip()) for x in args.only_levels.split(",") if x.strip()}

    from graphrag_agent.config.neo4jdb import get_db_manager
    g = get_db_manager().get_graph()

    # 预统计：每层有多少社区要摘
    print("预估调用量 ...")
    for level in range(args.max_level + 1):
        if level == 0:
            n = g.query(
                """
                MATCH (c:__Community__ {level: 0})<-[:IN_COMMUNITY]-(e:__Entity__)
                WITH c, count(e) AS deg WHERE deg >= $min
                RETURN count(c) AS n
                """,
                {"min": args.min_entities},
            )[0]["n"]
        else:
            n = g.query(
                """
                MATCH (c:__Community__ {level: $lv})
                MATCH (child:__Community__ {level: $cl})-[:IN_COMMUNITY]->(c)
                WITH c, count(child) AS deg WHERE deg >= 1
                RETURN count(c) AS n
                """,
                {"lv": level, "cl": level - 1},
            )[0]["n"]
        should_skip = (not args.all_levels) and level in (1, 3)
        tag = "SKIP" if should_skip else "SUMMARIZE"
        print(f"  level {level}: {n:>5} 个社区  [{tag}]")

    # 估算成本：假设 gpt-4o 每次调用 0.005 USD（粗略）
    total_calls = 0
    for level in range(args.max_level + 1):
        if (not args.all_levels) and level in (1, 3):
            continue
        if level == 0:
            n = g.query(
                """
                MATCH (c:__Community__ {level: 0})<-[:IN_COMMUNITY]-(e:__Entity__)
                WITH c, count(e) AS deg WHERE deg >= $min
                RETURN count(c) AS n
                """,
                {"min": args.min_entities},
            )[0]["n"]
        else:
            n = g.query(
                """
                MATCH (c:__Community__ {level: $lv})
                MATCH (child:__Community__ {level: $cl})-[:IN_COMMUNITY]->(c)
                WITH c, count(child) AS deg WHERE deg >= 1
                RETURN count(c) AS n
                """,
                {"lv": level, "cl": level - 1},
            )[0]["n"]
        total_calls += n
    est_cost_lo = total_calls * 0.0005  # cheap model
    est_cost_hi = total_calls * 0.005   # gpt-4o
    print(f"\n预计总 LLM 调用: {total_calls}")
    print(f"预计 API 成本: ${est_cost_lo:.2f} (cheap model) ~ ${est_cost_hi:.2f} (gpt-4o)")

    if args.dry_run:
        print("\n--dry-run 模式，不实际执行")
        return

    print("\n开始分层摘要...\n")
    from graphrag_agent.community.summary.hierarchical import HierarchicalCommunitySummarizer
    summarizer = HierarchicalCommunitySummarizer(max_workers=args.workers)

    start = time.time()
    if only_levels is not None:
        # 只跑指定层级
        stats = {}
        for level in sorted(only_levels):
            if level < 0 or level > args.max_level:
                continue
            print(f"\n{'='*60}\nLevel {level} (only-levels 模式)\n{'='*60}")
            if level == 0:
                stats[f"level_{level}"] = summarizer.summarize_level0(min_entities=args.min_entities)
            else:
                stats[f"level_{level}"] = summarizer.summarize_higher_level(level)
    else:
        stats = summarizer.summarize_all_levels(
            max_level=args.max_level,
            skip_odd_levels=not args.all_levels,
            min_entities=args.min_entities,
        )
    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"全部完成，耗时 {elapsed/60:.1f} 分钟")
    print(f"产出摘要数: {stats}")
    print("=" * 60)

    # 验证
    print("\n验证各层 summary 覆盖率:")
    for level in range(args.max_level + 1):
        r = g.query(
            """
            MATCH (c:__Community__ {level: $lv})
            RETURN count(c) AS total, count(c.summary) AS with_summary
            """,
            {"lv": level},
        )[0]
        total = r["total"]
        ws = r["with_summary"]
        pct = ws / total * 100 if total else 0
        print(f"  level {level}: {ws:>4}/{total:<5} ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
