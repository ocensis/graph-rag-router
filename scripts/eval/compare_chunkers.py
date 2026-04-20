"""
对比 jieba-based ChineseTextChunker vs LangChain RecursiveCharacterTextSplitter
在同一篇 arxiv 英文论文上的分块差异。

结论指标:
  - chunk 数量
  - chunk 长度分布
  - 句边界对齐率（chunk 结尾是否在句号/问号等处）
  - 空白/垃圾 chunk 占比
"""
import os
import sys
import argparse

sys.stdout.reconfigure(encoding="utf-8")

from graphrag_agent.pipelines.ingestion.file_reader import FileReader
from graphrag_agent.pipelines.ingestion.text_chunker import ChineseTextChunker

from langchain_text_splitters import RecursiveCharacterTextSplitter


def stats(chunks_text, name):
    n = len(chunks_text)
    if n == 0:
        print(f"{name}: 0 chunks")
        return
    lens_char = [len(c) for c in chunks_text]
    lens_word = [len(c.split()) for c in chunks_text]
    # 句边界对齐: chunk 最后一个非空字符是句点、问号、感叹号、分号
    ends_sent = sum(1 for c in chunks_text if c.strip() and c.strip()[-1] in ".!?。！？;")
    # 开头从空白/碎片开始（首字符是小写，说明被切到句子中间）
    starts_mid_sent = sum(
        1 for c in chunks_text
        if c.strip() and c.strip()[0].islower()
    )
    short = sum(1 for l in lens_char if l < 100)

    print(f"\n=== {name} ===")
    print(f"  chunks 数量:           {n}")
    print(f"  平均长度 (chars):      {sum(lens_char)/n:.0f}")
    print(f"  平均长度 (words):      {sum(lens_word)/n:.0f}")
    print(f"  min / max (chars):     {min(lens_char)} / {max(lens_char)}")
    print(f"  句号结尾率:            {ends_sent}/{n} = {ends_sent/n*100:.1f}%")
    print(f"  切到句中起始率:         {starts_mid_sent}/{n} = {starts_mid_sent/n*100:.1f}%")
    print(f"  短 chunk (<100 chars): {short}/{n}")


def show_boundaries(chunks_text, name, n=2):
    """展示前 n 个 chunk 结尾和下一个 chunk 开头，看对齐效果"""
    print(f"\n--- {name} 前 {n} 个 chunk 的边界 ---")
    for i in range(min(n, len(chunks_text) - 1)):
        end = chunks_text[i][-80:].replace("\n", " ")
        start = chunks_text[i+1][:80].replace("\n", " ")
        print(f"  Chunk [{i}] ... {end!r}")
        print(f"  Chunk [{i+1}] {start!r} ...")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="files/rag_papers/2604_09508v1__VISOR_Agentic_Visual_Retrieval-Augmented_Generation_via_Iterative_Search_and_Ove.pdf")
    args = ap.parse_args()

    # 1. 读 PDF
    print(f"读取: {os.path.basename(args.pdf)}")
    fr = FileReader(directory_path=os.path.dirname(args.pdf))
    text = fr._read_pdf(args.pdf)
    print(f"全文字符: {len(text)}, 词数: {len(text.split())}")

    # 2. jieba chunker (current project)
    jieba_chunker = ChineseTextChunker(chunk_size=500, overlap=100)
    jieba_chunks = jieba_chunker.chunk_text(text)
    # jieba 返回 token list → 合并回文本
    jieba_chunks_text = ["".join(toks) for toks in jieba_chunks]

    # 3. LangChain RecursiveCharacterTextSplitter (target size = 500 tokens ≈ 2500 chars for English)
    lc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=500,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )
    lc_chunks_text = lc_splitter.split_text(text)

    stats(jieba_chunks_text, "jieba (CHUNK_SIZE=500 tokens)")
    stats(lc_chunks_text, "LangChain Recursive (chunk_size=2500 chars)")

    show_boundaries(jieba_chunks_text, "jieba")
    show_boundaries(lc_chunks_text, "LangChain")


if __name__ == "__main__":
    main()
