"""
从 arXiv 下载 RAG 相关论文

用法:
  python download_rag_papers.py                           # 默认 50 篇 RAG 论文
  python download_rag_papers.py --n 100
  python download_rag_papers.py --query "graph RAG"       # 自定义关键词
  python download_rag_papers.py --n 20 --dir files/rag_papers
"""
import os
import sys
import json
import time
import argparse
import re
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str,
                        default='abs:"retrieval augmented generation" OR abs:"retrieval-augmented generation" OR abs:"RAG"',
                        help="arXiv 查询语法，支持 ti: / abs: / cat: 等字段")
    parser.add_argument("--n", type=int, default=50, help="下载数量")
    parser.add_argument("--dir", type=str, default="files/rag_papers", help="保存目录")
    parser.add_argument("--cat", type=str, default="cs.CL,cs.IR,cs.AI",
                        help="限定 arXiv 分类，逗号分隔")
    parser.add_argument("--skip-pdf", action="store_true", help="只下元数据不下 PDF")
    parser.add_argument("--from-ids", type=str, default=None,
                        help="从 JSON 文件（[{arxiv_id, title}, ...]）按 ID 下载，保证可复现")
    args = parser.parse_args()

    try:
        import arxiv
    except ImportError:
        print("请先安装: pip install arxiv")
        sys.exit(1)

    # 组合查询：关键词 + 分类
    cat_filter = " OR ".join(f"cat:{c.strip()}" for c in args.cat.split(","))
    full_query = f"({args.query}) AND ({cat_filter})"
    print(f"查询: {full_query}")

    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_file = out_dir / "_metadata.json"

    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)

    # --from-ids 模式：按指定 arxiv_id 列表下载，保证项目可复现
    # 用法: python download_rag_papers.py --from-ids benchmarks/paper_ids.json
    if args.from_ids:
        id_list = json.load(open(args.from_ids, encoding="utf-8"))
        raw_ids = [p["arxiv_id"].replace("_", ".") for p in id_list]  # 2604_14967v1 -> 2604.14967v1
        print(f"按 paper_ids.json 下载 {len(raw_ids)} 篇")
        search = arxiv.Search(id_list=raw_ids)
    else:
        search = arxiv.Search(
            query=full_query,
            max_results=args.n,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

    results = []
    downloaded = 0
    skipped = 0
    errors = 0

    print(f"开始拉取（最多 {args.n} 篇，按提交时间降序）...\n")

    for i, result in enumerate(client.results(search), 1):
        arxiv_id = result.entry_id.split("/")[-1].replace(".", "_")
        title_safe = re.sub(r'[^\w\s-]', '', result.title)[:80].strip().replace(' ', '_')
        pdf_name = f"{arxiv_id}__{title_safe}.pdf"
        pdf_path = out_dir / pdf_name

        meta = {
            "arxiv_id": arxiv_id,
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "published": result.published.isoformat() if result.published else None,
            "updated": result.updated.isoformat() if result.updated else None,
            "categories": result.categories,
            "primary_category": result.primary_category,
            "summary": result.summary,
            "pdf_url": result.pdf_url,
            "entry_id": result.entry_id,
            "pdf_path": str(pdf_path),
        }
        results.append(meta)

        print(f"[{i}] {result.title[:80]}")
        print(f"    {result.published.date()} | {result.primary_category} | {arxiv_id}")

        if args.skip_pdf:
            skipped += 1
            continue

        if pdf_path.exists():
            print(f"    已存在，跳过下载")
            skipped += 1
            continue

        try:
            result.download_pdf(dirpath=str(out_dir), filename=pdf_name)
            downloaded += 1
            print(f"    ✓ 下载成功")
        except Exception as e:
            errors += 1
            print(f"    ✗ 下载失败: {e}")

        # 避免打太快被 arxiv 限流
        time.sleep(0.5)

    # 保存元数据
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"完成!")
    print(f"  找到:      {len(results)} 篇")
    print(f"  下载:      {downloaded} 个 PDF")
    print(f"  已跳过:    {skipped}")
    print(f"  失败:      {errors}")
    print(f"  元数据:    {meta_file}")
    print(f"  PDF 目录:  {out_dir}/")


if __name__ == "__main__":
    main()
