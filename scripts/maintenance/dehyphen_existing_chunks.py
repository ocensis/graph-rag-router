"""
Option B: 原地更新 Neo4j 中已有 chunk 的 text / original_text
         消除 LaTeX 软连字符折行（"It-\\nerative" → "Iterative"）

不重建 KG，不重跑 embedding / 实体抽取。
影响：
  + text 字段立刻生效（BM25 chunk_fulltext 自动 reindex）
  + original_text 生效（LLM compose 看到的是干净文本）
  - embedding 是旧的（基于带折行文本），与新 text 语义有轻微不一致
    但 LaTeX 折行对 embedding 影响其实很小，实测影响可忽略

用法:
  python _dehyphen_existing_chunks.py --dry-run    # 统计不写
  python _dehyphen_existing_chunks.py              # 真写
"""
import os
import sys
import re
import argparse
import time

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

# 复用 file_reader 里的正则
from graphrag_agent.pipelines.ingestion.file_reader import _dehyphenate, _HYPHEN_BREAK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只统计，不更新")
    ap.add_argument("--batch", type=int, default=200, help="每批更新多少条")
    args = ap.parse_args()

    from graphrag_agent.config.neo4jdb import get_db_manager
    graph = get_db_manager().get_graph()

    # 拉全量 chunks（text 和 original_text）
    print("拉取所有 chunks ...")
    chunks = graph.query(
        """
        MATCH (c:__Chunk__)
        RETURN c.id AS id,
               c.text AS text,
               c.original_text AS original_text
        """
    )
    print(f"  总计 {len(chunks)} 条")

    # 统计 + 构造更新批
    updates = []
    break_count_before = 0
    affected = 0

    for c in chunks:
        old_text = c.get("text") or ""
        old_orig = c.get("original_text") or ""
        new_text = _dehyphenate(old_text)
        new_orig = _dehyphenate(old_orig)

        # 统计折行消除数
        breaks_before = len(_HYPHEN_BREAK.findall(old_text)) + len(_HYPHEN_BREAK.findall(old_orig))
        break_count_before += breaks_before

        if new_text != old_text or new_orig != old_orig:
            affected += 1
            updates.append({
                "id": c["id"],
                "text": new_text,
                "original_text": new_orig,
            })

    print(f"\n统计:")
    print(f"  影响 chunk 数:    {affected}/{len(chunks)} = {affected/len(chunks)*100:.1f}%")
    print(f"  总折行数（消除前）: {break_count_before:,}")
    print(f"  平均每 chunk:     {break_count_before/max(len(chunks),1):.1f} 个折行")

    if args.dry_run:
        print("\n--dry-run，不写入")
        # 展示前 3 个样例
        print("\n前 3 个修改样例:")
        shown = 0
        for c in chunks:
            old = c.get("text") or ""
            new = _dehyphenate(old)
            if old != new:
                # 找第一个差异处
                m = _HYPHEN_BREAK.search(old)
                if m:
                    s, e = m.start(), m.end()
                    print(f"  chunk {c['id'][:12]}: "
                          f"...{old[max(0,s-30):e+20]!r} → {new[max(0,s-30):e+19]!r}...")
                shown += 1
                if shown >= 3:
                    break
        return

    if not updates:
        print("没有需要更新的 chunk")
        return

    # 批量 SET 回 Neo4j
    print(f"\n批量更新 {len(updates)} 条 chunk（batch={args.batch}）...")
    start = time.time()
    total_updated = 0
    for i in range(0, len(updates), args.batch):
        batch = updates[i : i + args.batch]
        graph.query(
            """
            UNWIND $batch AS row
            MATCH (c:__Chunk__ {id: row.id})
            SET c.text = row.text,
                c.original_text = row.original_text,
                c.dehyphenated_at = datetime()
            """,
            {"batch": batch},
        )
        total_updated += len(batch)
        print(f"  {total_updated}/{len(updates)}  ({time.time()-start:.1f}s)")

    print(f"\n✓ 更新完成，共 {total_updated} 条 chunk，耗时 {time.time()-start:.1f}s")

    # 验证
    print("\n验证：再次扫描剩余折行数 ...")
    after = graph.query(
        """
        MATCH (c:__Chunk__)
        WITH c.text AS t, c.original_text AS o
        RETURN sum(
          size(split(t + o, '')) - size(replace(t + o, '\\n', ''))
        ) AS placeholder
        LIMIT 1
        """
    )
    # 上面那个 Cypher 算的是换行数，不是折行 —— 改用 Python 侧验证
    recheck = graph.query(
        "MATCH (c:__Chunk__) RETURN c.text AS t, c.original_text AS o"
    )
    remaining = sum(
        len(_HYPHEN_BREAK.findall((r.get("t") or ""))) +
        len(_HYPHEN_BREAK.findall((r.get("o") or "")))
        for r in recheck
    )
    print(f"  剩余折行数: {remaining}  (应该 = 0)")

    if remaining == 0:
        print("\n✓ 全部清理完成。chunk_fulltext BM25 索引会自动重建。")
        print("  Embedding 保留（基于旧文本），但实测 LaTeX 折行对向量影响很小。")
        print("  下一步：跑 bench 看收益：")
        print("    python run_hotpotqa_bench.py --agent router "
              "--questions bench_results/eng_cross_doc_questions.json "
              "--tag eng200_dehyphen --workers 8")


if __name__ == "__main__":
    main()
