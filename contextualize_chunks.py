"""
Contextual Retrieval（Anthropic 论文方法）—— 给每个 chunk 加上情境化上下文后重建索引

流程:
1. 拉 Neo4j 所有 chunk（skip 已有 context 的，支持断点续跑）
2. 每个 chunk 用 LLM 生成 50-80 词 contextual summary
   （输入: paper abstract + chunk → 输出: 包含 paper / method / dataset 锚点的一句话）
3. 备份原文到 c.original_text
4. 覆盖 c.text = context + "\n\n" + original_text
5. 重新计算 c.embedding（基于新 text）
6. HNSW / fulltext 索引会自动 reindex

成本: ~$1（gpt-4o-mini 生成 + text-embedding-3-large 重 embed）
耗时: ~5-10 分钟（20 并发）

用法:
  python contextualize_chunks.py               # 全量
  python contextualize_chunks.py --limit 50    # 小样本先试
  python contextualize_chunks.py --dry-run     # 只生成不写库
"""
import sys
import json
import time
import argparse
import concurrent.futures
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.models.get_models import get_llm_model, get_embeddings_model
from graphrag_agent.utils.logger import get_logger

logger = get_logger(__name__)


CONTEXT_PROMPT = """You are given a research paper's metadata and a text chunk from that paper.
Generate a single short contextual summary (40-80 words) that helps situate this chunk within its paper.

Your summary MUST mention:
1. The paper's method name (if this chunk relates to the method)
2. Any dataset, benchmark, or metric names present
3. What aspect this chunk discusses (introduction / method / experiment / result / limitation)

Paper title: {title}
Paper abstract: {abstract}

Chunk text:
{chunk_text}

Contextual summary (single paragraph, 40-80 words):"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个 chunk（测试用）")
    parser.add_argument("--workers", type=int, default=20, help="LLM 并发数")
    parser.add_argument("--dry-run", action="store_true", help="不写回 Neo4j")
    parser.add_argument("--redo", action="store_true", help="强制重新处理已有 context 的 chunk")
    args = parser.parse_args()

    # ---- Load papers metadata ----
    metadata_path = Path("files/rag_papers/_metadata.json")
    papers = json.load(open(metadata_path, "r", encoding="utf-8"))
    # key 用 fileName（PDF 名），匹配 Neo4j 的 c.fileName
    paper_by_filename = {}
    for p in papers:
        fname = Path(p["pdf_path"]).name  # e.g. "2604_14967v1__UniDoc-RL_...pdf"
        paper_by_filename[fname] = p
    print(f"加载 {len(paper_by_filename)} 篇论文元数据")

    # ---- Fetch chunks ----
    graph = get_db_manager().graph
    llm = get_llm_model()
    embeddings = get_embeddings_model()

    where_clause = "c.text IS NOT NULL"
    if not args.redo:
        where_clause += " AND (c.context IS NULL OR c.context = '')"

    chunks = graph.query(f"""
        MATCH (c:__Chunk__)
        WHERE {where_clause}
        RETURN c.id AS id, c.text AS text, c.fileName AS fname
        ORDER BY c.id
    """)
    if args.limit:
        chunks = chunks[: args.limit]

    print(f"待处理 chunks: {len(chunks)}")
    if not chunks:
        print("没有 chunks 需要处理（已全部有 context，用 --redo 强制重跑）")
        return

    # ---- Generate contexts in parallel ----
    def _generate_one(chunk):
        paper = paper_by_filename.get(chunk["fname"], {})
        title = paper.get("title", chunk["fname"])
        abstract = paper.get("summary", "")[:1500]
        prompt = CONTEXT_PROMPT.format(
            title=title,
            abstract=abstract,
            chunk_text=chunk["text"][:2000],
        )
        try:
            resp = llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            return chunk["id"], content.strip(), None
        except Exception as e:
            return chunk["id"], "", str(e)

    print(f"\n生成 contextual summaries ({args.workers} 并发)...")
    t0 = time.time()
    contexts = {}
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = [exe.submit(_generate_one, c) for c in chunks]
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            cid, ctx, err = fut.result()
            contexts[cid] = ctx
            if err:
                errors += 1
            done += 1
            if done % 50 == 0 or done == len(chunks):
                elapsed = time.time() - t0
                qps = done / elapsed
                print(f"  [{done}/{len(chunks)}] {elapsed:.0f}s, {qps:.1f}/s, errors={errors}")

    # Filter empty (failed) contexts
    good = [(c["id"], contexts[c["id"]], c["text"])
            for c in chunks if contexts.get(c["id"])]
    print(f"\n成功生成: {len(good)}/{len(chunks)}")

    # 展示 1 个样本
    if good:
        sample_id, sample_ctx, sample_text = good[0]
        print(f"\n--- 样本 ---")
        print(f"Chunk ID: {sample_id}")
        print(f"Original text[:150]: {sample_text[:150]}...")
        print(f"Generated context: {sample_ctx}")
        print(f"--- 样本结束 ---\n")

    if args.dry_run:
        print("--dry-run，不写回 Neo4j")
        return

    # 先把 contexts 存盘，避免 embedding 挂了白跑 LLM
    ctx_cache = Path("cache/_contexts_backup.json")
    ctx_cache.parent.mkdir(parents=True, exist_ok=True)
    with open(ctx_cache, "w", encoding="utf-8") as f:
        json.dump([{"id": cid, "context": ctx, "text": txt} for cid, ctx, txt in good],
                  f, ensure_ascii=False)
    print(f"  contexts 已备份到 {ctx_cache}")

    # ---- 分批 re-embed ----
    print(f"分批 re-embedding (batch=64)...")
    t0 = time.time()
    new_texts = [f"{ctx}\n\n{txt}" for _, ctx, txt in good]
    BATCH = 64
    new_embeddings = []
    failed_batches = []
    for i in range(0, len(new_texts), BATCH):
        batch = new_texts[i: i + BATCH]
        try:
            emb = embeddings.embed_documents(batch)
            new_embeddings.extend(emb)
        except Exception as e:
            logger.warning(f"batch {i} embed 失败: {str(e)[:100]}")
            # 逐条 fallback
            for t in batch:
                try:
                    single = embeddings.embed_query(t[:6000])  # 截断防 token 超限
                    new_embeddings.append(single)
                except Exception:
                    new_embeddings.append(None)
                    failed_batches.append(i)
        if (i + BATCH) % 320 == 0 or i + BATCH >= len(new_texts):
            print(f"  [{min(i+BATCH, len(new_texts))}/{len(new_texts)}] {time.time()-t0:.0f}s")
    print(f"  完成 ({time.time()-t0:.1f}s), 失败 {sum(1 for e in new_embeddings if e is None)}")

    # ---- Write back to Neo4j in batches ----
    print(f"写回 Neo4j ...")
    t0 = time.time()
    batch_size = 100
    updated = 0
    for i in range(0, len(good), batch_size):
        batch = good[i: i + batch_size]
        batch_emb = new_embeddings[i: i + batch_size]
        params = [
            {
                "id": cid,
                "context": ctx,
                "new_text": f"{ctx}\n\n{txt}",
                "embedding": emb,
            }
            for (cid, ctx, txt), emb in zip(batch, batch_emb)
            if emb is not None  # 跳过 embed 失败的
        ]
        if not params:
            continue
        graph.query(
            """
            UNWIND $params AS p
            MATCH (c:__Chunk__ {id: p.id})
            SET c.original_text = coalesce(c.original_text, c.text),
                c.context = p.context,
                c.text = p.new_text,
                c.embedding = p.embedding
            """,
            {"params": params},
        )
        updated += len(batch)
        print(f"  [{updated}/{len(good)}] 写入中...")

    print(f"  写回完成 ({time.time()-t0:.1f}s)")

    # ---- 验证 ----
    r = graph.query("""
        MATCH (c:__Chunk__) WHERE c.context IS NOT NULL AND c.context <> ''
        RETURN count(c) AS with_ctx
    """)
    total_r = graph.query("MATCH (c:__Chunk__) RETURN count(c) AS total")
    print(f"\n最终状态:")
    print(f"  有 context 的 chunks: {r[0]['with_ctx']} / {total_r[0]['total']}")

    print(f"\n✓ 完成。HNSW 和 fulltext 索引会自动反映新内容。")
    print(f"  下一步: 清缓存 + 重跑 naive benchmark 对比效果")


if __name__ == "__main__":
    main()
