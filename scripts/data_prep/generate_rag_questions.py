"""
为 RAG 论文生成评测问题

对每篇论文，基于 abstract 用 LLM 生成 2 个带参考答案的问题：
  1. 方法/贡献类：论文提出了什么新方法
  2. 实验/结果类：在哪些数据集上验证，相比 baseline 提升多少

输出格式对齐 HotpotQA（兼容现有评测脚本）:
  {"id": ..., "question": ..., "answer": ..., "question_type": ...}

用法:
  python generate_rag_questions.py                    # 默认读 files/rag_papers/_metadata.json
  python generate_rag_questions.py --workers 10
"""
import os
import sys
import json
import time
import argparse
import concurrent.futures
from pathlib import Path
from typing import List, Optional

sys.stdout.reconfigure(encoding='utf-8')

from pydantic import BaseModel, Field, ValidationError


# ==================== LLM 输出结构 ====================

class QAPair(BaseModel):
    """一组 Q&A"""
    question: str = Field(..., description="基于论文内容的具体问题")
    answer: str = Field(..., description="简洁答案（一句话或一个实体）")
    question_type: str = Field(..., description="问题类型：method / result / comparison")


class PaperQuestions(BaseModel):
    """每篇论文产出的 2 个问题"""
    method_question: QAPair
    result_question: QAPair


# ==================== Prompt ====================

QA_GEN_PROMPT = """You are building an evaluation set for a RAG system over academic papers.
Based on the paper abstract below, generate 2 evaluation questions with concise reference answers.

Requirements:
1. **method_question**: Ask what novel method/technique/approach this paper proposes. Answer should be a specific method name or a short phrase (≤20 words).
2. **result_question**: Ask about a specific experimental result, dataset, or improvement quantified in the abstract. Answer should be a dataset name, metric name, or a concrete number with unit.
3. Questions must be answerable from the abstract alone (no guessing).
4. Keep answers short — a reference answer for exact match evaluation.
5. Avoid yes/no questions.

Paper title: {title}
Paper abstract: {abstract}

Return a JSON object with this exact structure:
{{
  "method_question": {{
    "question": "...",
    "answer": "...",
    "question_type": "method"
  }},
  "result_question": {{
    "question": "...",
    "answer": "...",
    "question_type": "result"
  }}
}}

Respond with ONLY the JSON, no markdown, no explanation."""


# ==================== 生成逻辑 ====================

def generate_qa_for_paper(paper: dict, llm) -> Optional[PaperQuestions]:
    """用 LLM 给单篇论文生成 2 个 QA pair"""
    prompt = QA_GEN_PROMPT.format(
        title=paper["title"],
        abstract=paper["summary"][:3000],  # 截断防超长
    )

    for attempt in range(3):
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # 剥掉可能的 markdown 代码块
            import re
            cleaned = re.sub(r'^```(?:json)?\s*', '', content.strip())
            cleaned = re.sub(r'\s*```$', '', cleaned)

            data = json.loads(cleaned)
            return PaperQuestions.model_validate(data)

        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == 2:
                print(f"  [{paper['arxiv_id']}] 解析失败 (已重试 {attempt+1} 次): {e}")
                return None
            time.sleep(1)
        except Exception as e:
            print(f"  [{paper['arxiv_id']}] LLM 调用失败: {e}")
            return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="files/rag_papers/_metadata.json")
    parser.add_argument("--out", type=str, default="bench_results/rag_papers_questions.json")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 篇（调试用）")
    args = parser.parse_args()

    # 读论文元数据
    with open(args.meta, "r", encoding="utf-8") as f:
        papers = json.load(f)

    if args.limit:
        papers = papers[:args.limit]

    print(f"加载 {len(papers)} 篇论文")

    # 初始化 LLM
    from graphrag_agent.models.get_models import get_llm_model
    llm = get_llm_model()
    print(f"使用模型: {os.getenv('OPENAI_LLM_MODEL')}")

    # 并发生成
    print(f"\n开始生成（{args.workers} 并发）...\n")
    results = [None] * len(papers)
    completed = 0
    start = time.time()

    def process_one(idx, paper):
        qa = generate_qa_for_paper(paper, llm)
        return idx, qa

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, i, p): i for i, p in enumerate(papers)}

        for future in concurrent.futures.as_completed(futures):
            try:
                idx, qa = future.result(timeout=120)
                results[idx] = qa
            except Exception as e:
                idx = futures[future]
                print(f"  [{papers[idx]['arxiv_id']}] 异常: {e}")
                results[idx] = None

            completed += 1
            if completed % 10 == 0 or completed == len(papers):
                elapsed = time.time() - start
                print(f"  [{completed}/{len(papers)}] {elapsed:.1f}s, "
                      f"{completed/elapsed:.2f} paper/s")

    elapsed = time.time() - start

    # 组装 HotpotQA 格式
    questions = []
    success = 0
    for paper, qa in zip(papers, results):
        if qa is None:
            continue
        success += 1
        for qpair, qtype in [(qa.method_question, "method"),
                             (qa.result_question, "result")]:
            questions.append({
                "id": f"{paper['arxiv_id']}_{qtype}",
                "question": qpair.question,
                "answer": qpair.answer,
                "question_type": qtype,
                "source_paper": {
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                },
            })

    # 保存
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"完成! 耗时 {elapsed:.1f}s")
    print(f"  成功生成:      {success}/{len(papers)} 篇论文")
    print(f"  总问题数:      {len(questions)}")
    print(f"  method 类型:   {sum(1 for q in questions if q['question_type'] == 'method')}")
    print(f"  result 类型:   {sum(1 for q in questions if q['question_type'] == 'result')}")
    print(f"  输出文件:      {args.out}")

    # 展示 3 个样本
    print(f"\n样本预览:")
    for q in questions[:3]:
        print(f"\n  Q ({q['question_type']}): {q['question']}")
        print(f"  A: {q['answer']}")
        print(f"  Source: {q['source_paper']['title'][:60]}...")


if __name__ == "__main__":
    main()
