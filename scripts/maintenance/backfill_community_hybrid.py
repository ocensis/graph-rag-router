"""
给 Community 节点一次性补两个字段，让 community 检索从"固定 top-30"升级到
query-relevant Hybrid 排序：

  1. c.entities_kwd = [entity.id, ...]
     - level 0: 直接沿 IN_COMMUNITY 边收 entity 成员
     - level >= 1: 递归上卷，把子社区的 entities_kwd union 起来
     （Hybrid 排序的 "entity_hits" 维度靠这个字段）

  2. c.embedding = embed(summary 或 full_content)
     （Hybrid 排序的 "vec_sim" 维度靠这个字段）

  3. 建 vector index  'community_vector'

跑完后：
  - GraphPath._fetch_community_summaries / GlobalSearchTool._get_top_communities
    可以用 hybrid Cypher 做三维融合排序（entity_hits + vec_sim + weight_log）

成本: ~$1-2（5000 社区 × text-embedding-3-large @ $0.13/M token）
时间: 10-15 分钟（并行 batch embed）

用法:
    python scripts/maintenance/backfill_community_hybrid.py
    python scripts/maintenance/backfill_community_hybrid.py --only-embedding
    python scripts/maintenance/backfill_community_hybrid.py --only-entities
    python scripts/maintenance/backfill_community_hybrid.py --dry-run
"""
from __future__ import annotations

import sys
import time
import argparse

sys.stdout.reconfigure(encoding="utf-8")

from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.models.get_models import get_embeddings_model
from graphrag_agent.utils.logger import get_logger

logger = get_logger(__name__)


# 哪些层有 summary 就给哪些层加 embedding / entities_kwd
# 项目里奇数层（1/3）是空壳不参与检索，无需处理
TARGET_LEVELS = [0, 2, 4]


# ==================== Step 1-2: entities_kwd ====================

def populate_entities_kwd_level0(graph, dry_run: bool = False):
    """Level 0 社区：直接沿 IN_COMMUNITY 边收它的 entity 成员"""
    q = """
    MATCH (c:__Community__ {level: 0})
    OPTIONAL MATCH (e:__Entity__)-[:IN_COMMUNITY]->(c)
    WITH c, collect(DISTINCT e.id) AS ents
    RETURN c.id AS cid, size(ents) AS n_ents, ents
    """
    rows = graph.query(q)
    print(f"  level 0: 扫描到 {len(rows)} 个社区")
    empty = sum(1 for r in rows if r["n_ents"] == 0)
    print(f"  level 0: {empty} 个社区无 entity 成员（跳过）")

    if dry_run:
        print("  [dry-run] 不写库")
        return
    # 批量写
    batch = 500
    written = 0
    for i in range(0, len(rows), batch):
        chunk = [r for r in rows[i:i+batch] if r["n_ents"] > 0]
        if not chunk:
            continue
        graph.query(
            """
            UNWIND $items AS it
            MATCH (c:__Community__ {id: it.cid})
            SET c.entities_kwd = it.ents
            """,
            {"items": [{"cid": r["cid"], "ents": r["ents"]} for r in chunk]},
        )
        written += len(chunk)
    print(f"  level 0: 写入 {written} 个社区的 entities_kwd")


def populate_entities_kwd_higher(graph, level: int, dry_run: bool = False):
    """Level >= 1 社区：把子社区（level-1）的 entities_kwd union 起来。
    子社区必须先跑完本函数 / level 0 函数。"""
    q = """
    MATCH (parent:__Community__ {level: $level})
    OPTIONAL MATCH (child:__Community__ {level: $child_level})-[:IN_COMMUNITY]->(parent)
    WITH parent, collect(child.entities_kwd) AS child_lists
    WITH parent,
         reduce(acc = [], lst IN child_lists |
                acc + coalesce(lst, [])) AS merged
    RETURN parent.id AS cid, size(merged) AS n_ents,
           [e IN merged WHERE e IS NOT NULL] AS ents
    """
    rows = graph.query(q, {"level": level, "child_level": level - 1})
    print(f"  level {level}: 扫描到 {len(rows)} 个社区（子层 = level {level - 1}）")

    # 去重一下（union 可能重复）
    processed = []
    for r in rows:
        ents = list(dict.fromkeys(r["ents"]))   # 保持顺序去重
        processed.append({"cid": r["cid"], "n_ents": len(ents), "ents": ents})

    empty = sum(1 for r in processed if r["n_ents"] == 0)
    print(f"  level {level}: {empty} 个社区聚合到 0 个 entity（跳过）")

    if dry_run:
        print(f"  [dry-run] 示例 (level {level}): {processed[0] if processed else 'none'}")
        return

    batch = 200
    written = 0
    for i in range(0, len(processed), batch):
        chunk = [r for r in processed[i:i+batch] if r["n_ents"] > 0]
        if not chunk:
            continue
        graph.query(
            """
            UNWIND $items AS it
            MATCH (c:__Community__ {id: it.cid})
            SET c.entities_kwd = it.ents
            """,
            {"items": chunk},
        )
        written += len(chunk)
    print(f"  level {level}: 写入 {written} 个社区的 entities_kwd")


# ==================== Step 3: embedding ====================

def populate_embeddings(graph, embedder, levels, batch_size: int = 64, dry_run: bool = False):
    """每个 community 的 summary/full_content 过一次 embedding，写 c.embedding。
    只处理指定 levels（默认 [0, 2, 4]，和 summary 覆盖层一致）。"""
    levels_str = ", ".join(str(l) for l in levels)
    rows = graph.query(
        f"""
        MATCH (c:__Community__)
        WHERE c.level IN [{levels_str}]
          AND (c.full_content IS NOT NULL OR c.summary IS NOT NULL)
          AND c.embedding IS NULL
        RETURN c.id AS cid,
               coalesce(c.full_content, c.summary) AS text
        """
    )
    total = len(rows)
    print(f"  有 {total} 个社区待 embed（已有 embedding 的跳过）")
    if total == 0:
        return
    if dry_run:
        print(f"  [dry-run] 示例文本 (120 字符): {rows[0]['text'][:120]}...")
        return

    t0 = time.time()
    processed = 0
    for i in range(0, total, batch_size):
        batch = rows[i:i+batch_size]
        texts = [r["text"][:8000] for r in batch]   # 裁一下防超长
        try:
            vecs = embedder.embed_documents(texts)
        except Exception as e:
            logger.warning(f"batch {i} embed 失败，逐条回退: {e}")
            vecs = []
            for t in texts:
                try:
                    vecs.append(embedder.embed_query(t))
                except Exception as ee:
                    logger.error(f"  单条 embed 失败: {ee}")
                    vecs.append(None)

        # 写回
        items = [{"cid": b["cid"], "vec": v} for b, v in zip(batch, vecs) if v is not None]
        if items:
            graph.query(
                """
                UNWIND $items AS it
                MATCH (c:__Community__ {id: it.cid})
                CALL db.create.setNodeVectorProperty(c, 'embedding', it.vec)
                RETURN count(*)
                """,
                {"items": items},
            )
        processed += len(items)
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total - processed) / rate if rate > 0 else 0
        print(f"  [{processed}/{total}]  {rate:.1f} emb/s  eta {eta:.0f}s")

    print(f"  共 embed 了 {processed}/{total} 个社区，耗时 {time.time()-t0:.1f}s")


# ==================== Step 4: vector index ====================

def ensure_vector_index(graph, dims: int, dry_run: bool = False):
    """建 community_vector HNSW 索引。"""
    # 先查存在
    existing = graph.query(
        """
        SHOW INDEXES YIELD name, type
        WHERE name = 'community_vector'
        RETURN name
        """
    )
    if existing:
        print(f"  community_vector 索引已存在，跳过")
        return
    if dry_run:
        print(f"  [dry-run] 会创建 community_vector 向量索引 (dim={dims}, cosine)")
        return

    graph.query(
        f"""
        CREATE VECTOR INDEX community_vector IF NOT EXISTS
        FOR (c:__Community__) ON (c.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {dims},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """
    )
    print(f"  community_vector 向量索引已创建 (dim={dims}, cosine)")


# ==================== main ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="不写库，只打印计划")
    ap.add_argument("--only-entities", action="store_true", help="只跑 entities_kwd 回填")
    ap.add_argument("--only-embedding", action="store_true", help="只跑 embedding 回填")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    graph = get_db_manager().get_graph()
    print("=" * 60)
    print("Community Hybrid Backfill")
    print("=" * 60)

    do_entities = not args.only_embedding
    do_embedding = not args.only_entities

    if do_entities:
        print("\n[Step 1/3] entities_kwd backfill")
        populate_entities_kwd_level0(graph, dry_run=args.dry_run)
        for level in [1, 2, 3, 4]:
            populate_entities_kwd_higher(graph, level, dry_run=args.dry_run)

    if do_embedding:
        print("\n[Step 2/3] community embeddings")
        embedder = get_embeddings_model()

        # 先探一下向量维度（embed 一个短文本看输出长度）
        try:
            sample = embedder.embed_query("probe")
            dims = len(sample)
            print(f"  embedding 维度探测：{dims}")
        except Exception as e:
            print(f"  embedding 探测失败：{e}")
            sys.exit(1)

        populate_embeddings(
            graph, embedder,
            levels=TARGET_LEVELS,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )

        print("\n[Step 3/3] vector index")
        ensure_vector_index(graph, dims=dims, dry_run=args.dry_run)

    # 验证
    print("\n[Verify]")
    rows = graph.query(
        """
        MATCH (c:__Community__)
        WHERE c.level IN [0, 2, 4]
        WITH c.level AS lv,
             count(c) AS total,
             sum(CASE WHEN c.entities_kwd IS NOT NULL THEN 1 ELSE 0 END) AS with_ents,
             sum(CASE WHEN c.embedding IS NOT NULL THEN 1 ELSE 0 END) AS with_emb
        RETURN lv, total, with_ents, with_emb
        ORDER BY lv
        """
    )
    print(f"  {'level':<8} {'total':>8} {'with_ents':>12} {'with_emb':>10}")
    for r in rows:
        print(f"  {r['lv']:<8} {r['total']:>8} {r['with_ents']:>12} {r['with_emb']:>10}")

    print("\n[done]")


if __name__ == "__main__":
    main()
