"""
KG Case-Variant Dedup —— 合并 id.lower().strip() 碰撞的实体节点

问题：KG 质量报告发现 112 对 Jaccard=1.0 重复实体（"A-MEM / A-Mem / A-mem"、"ACC / Acc"），
     entity_disambiguation 基于 WCC 共享邻居去重，case-variant 如果邻居不重合就漏掉。

修复：按 toLower(trim(id)) 分组 → 选度数最高的作 canonical → 用 APOC mergeNodes 合并其余节点
     （APOC 自动转移所有边 + 属性，冲突时保留主节点的）

用法:
  python _dedup_case_variants.py --dry-run    # 只统计，不修改
  python _dedup_case_variants.py              # 真跑
  python _dedup_case_variants.py --min-size 2 # 合并 size >= 2 的组（默认）
"""
import os
import sys
import argparse
import time

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from langchain_neo4j import Neo4jGraph


def connect():
    return Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )


def check_apoc(graph):
    """确认 APOC 可用"""
    try:
        r = graph.query("RETURN apoc.version() AS v")
        return r[0]["v"] if r else None
    except Exception:
        return None


def find_case_groups(graph, min_size: int = 2):
    """找所有 case-variant 组"""
    return graph.query(
        """
        MATCH (e:__Entity__)
        WHERE e.id IS NOT NULL
        WITH toLower(trim(e.id)) AS norm_id, collect(e.id) AS variants, count(*) AS n
        WHERE n >= $min_size
        RETURN norm_id, variants, n
        ORDER BY n DESC
        """,
        {"min_size": min_size},
    )


def merge_group(graph, variants: list) -> dict:
    """合并一组 case-variant，返回 {canonical, merged_count}"""
    # 取度数最高的作 canonical
    pick = graph.query(
        """
        UNWIND $variants AS v
        MATCH (e:__Entity__ {id: v})
        WITH e, COUNT { (e)--() } AS deg
        ORDER BY deg DESC
        LIMIT 1
        RETURN e.id AS canonical
        """,
        {"variants": variants},
    )
    if not pick:
        return {"canonical": None, "merged_count": 0}

    canonical = pick[0]["canonical"]
    others = [v for v in variants if v != canonical]

    if not others:
        return {"canonical": canonical, "merged_count": 0}

    # 用 APOC 一次合并（保留 canonical 的属性，转移所有边）
    # properties='discard' 意思：冲突时丢其他节点的值，保留 canonical 的
    result = graph.query(
        """
        MATCH (canonical:__Entity__ {id: $canonical})
        UNWIND $others AS oid
        MATCH (other:__Entity__ {id: oid})
        WITH canonical, collect(other) AS others
        CALL apoc.refactor.mergeNodes(
            [canonical] + others,
            {properties: 'discard', mergeRels: true}
        ) YIELD node
        RETURN count(*) AS merged, node.id AS final_id
        """,
        {"canonical": canonical, "others": others},
    )
    if result:
        return {
            "canonical": result[0]["final_id"],
            "merged_count": len(others),
        }
    return {"canonical": canonical, "merged_count": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="只统计，不执行合并")
    ap.add_argument("--min-size", type=int, default=2,
                    help="组内最少节点数才合并（默认 2）")
    ap.add_argument("--limit", type=int, default=0,
                    help="最多处理多少组（0 = 全部）")
    args = ap.parse_args()

    graph = connect()
    apoc_version = check_apoc(graph)
    if not apoc_version:
        print("✗ APOC 不可用，脚本需要 APOC 的 apoc.refactor.mergeNodes")
        return
    print(f"Connected to Neo4j (APOC {apoc_version})")

    # Before 统计
    before = graph.query(
        "MATCH (e:__Entity__) RETURN count(e) AS total_entities"
    )
    total_before = before[0]["total_entities"] if before else 0
    print(f"Before: {total_before:,} entities")

    # 找 case-variant 组
    groups = find_case_groups(graph, min_size=args.min_size)
    print(f"\n找到 {len(groups)} 组 case-variant（min_size={args.min_size}）")
    total_duplicates = sum(g["n"] - 1 for g in groups)
    print(f"  涉及 {sum(g['n'] for g in groups):,} 个节点")
    print(f"  预计可合并 {total_duplicates:,} 个冗余节点")

    # 预览前 10 组
    print("\n前 10 组样例:")
    for g in groups[:10]:
        print(f"  [{g['n']}] {g['variants']}")

    if args.dry_run:
        print("\n--dry-run 模式，不执行合并")
        return

    if args.limit > 0:
        groups = groups[: args.limit]
        print(f"\n--limit {args.limit}，只合并前 {len(groups)} 组")

    # 执行合并
    print(f"\n开始合并 {len(groups)} 组...")
    start = time.time()
    total_merged = 0
    failed = 0
    for i, g in enumerate(groups, 1):
        try:
            res = merge_group(graph, g["variants"])
            total_merged += res["merged_count"]
            if i % 20 == 0 or i == len(groups):
                print(f"  进度 {i}/{len(groups)}  已合并 {total_merged} 个节点  "
                      f"({time.time()-start:.1f}s)")
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  [!] 组 {i} 合并失败: {e}")

    elapsed = time.time() - start
    print(f"\n完成: 合并 {total_merged:,} 个冗余节点，失败 {failed} 组，耗时 {elapsed:.1f}s")

    # After 统计
    after = graph.query(
        "MATCH (e:__Entity__) RETURN count(e) AS total_entities"
    )
    total_after = after[0]["total_entities"] if after else 0
    print(f"\nAfter:  {total_after:,} entities  (-{total_before - total_after:,})")

    # 再次检查重复
    new_groups = find_case_groups(graph, min_size=args.min_size)
    print(f"剩余 case-variant 组: {len(new_groups)}")


if __name__ == "__main__":
    main()
