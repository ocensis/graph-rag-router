"""
AgenticGraphSearchTool 在 HotpotQA 上的评测

相比 run_hotpotqa_bench.py：
- 用 LangGraph StateGraph 版本的 agentic 检索
- 额外统计多轮迭代指标：平均轮次、触发改写比例
- Langfuse session ID 带 benchmark 批次，方便在 UI 里聚合查看

用法:
  python run_agentic_hotpot_bench.py --n 50 --workers 10
  python run_agentic_hotpot_bench.py --n 1000 --workers 20
"""
import sys
import json
import time
import re
import string
import argparse
import os
import collections
import concurrent.futures

sys.stdout.reconfigure(encoding='utf-8')

BENCH_DIR = os.path.join(os.path.dirname(__file__), 'bench_results')
QUESTIONS_FILE = os.path.join(BENCH_DIR, 'hotpot_questions.json')


# ========== SQuAD 标准 EM/F1（和 run_hotpotqa_bench.py 保持一致）==========

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def extract_short_answer(full_answer):
    if not full_answer or full_answer.startswith("Error"):
        return ""
    bold_matches = re.findall(r'\*\*(.+?)\*\*', full_answer)
    if bold_matches:
        return bold_matches[0]
    first_line = full_answer.strip().split('\n')[0]
    return re.sub(r'^#+\s*', '', first_line)[:200]


# ========== 运行 Agentic 检索 ==========

def run_agentic(questions, max_workers=10, session_prefix="bench"):
    """
    用 AgenticGraphSearchTool 批量回答，同时捕获每条 query 走了几轮
    """
    from graphrag_agent.search.tool.agentic_graph_search import AgenticGraphSearchTool

    print(f"\n加载 AgenticGraphSearchTool ...")
    tool = AgenticGraphSearchTool(max_rounds=3)
    tool.cache_manager.clear() if hasattr(tool.cache_manager, 'clear') else None

    results = [None] * len(questions)
    rounds_taken = [0] * len(questions)
    num_chunks = [0] * len(questions)
    completed = 0

    def answer_one(idx, q):
        """直接调用 graph.invoke 拿到完整 final_state，顺便统计轮次"""
        try:
            from graphrag_agent.search.tool.agentic_graph_search import _reset_chunk_store
            _reset_chunk_store()  # 每个 query 独立的 chunk store

            initial_state = {
                "original_query": q["question"],
                "current_query": q["question"],
                "round_num": 0,
                "max_rounds": 3,
                "used_chunk_ids": [],
                "info_summary": "",
                "sufficient": False,
                "subquery": "",
                "final_answer": "",
            }
            from graphrag_agent.utils.langfuse_client import get_langfuse_handler
            config = {}
            handler = get_langfuse_handler()
            if handler is not None:
                config = {
                    "callbacks": [handler],
                    "run_name": "agentic_hotpot_bench",
                    "metadata": {
                        "langfuse_session_id": f"{session_prefix}_{idx}",
                        "langfuse_tags": ["benchmark", "hotpotqa", "agentic"],
                        "query_id": q.get("id", str(idx)),
                    },
                }
            final_state = tool._graph.invoke(initial_state, config=config)
            answer = final_state.get("final_answer", "") or ""
            rounds = final_state.get("round_num", 0) + 1
            chunks_count = len(final_state.get("used_chunk_ids", []))
            return idx, str(answer), rounds, chunks_count, None
        except Exception as e:
            return idx, f"Error: {e}", 0, 0, str(e)

    print(f"开始回答 {len(questions)} 个问题（{max_workers} 并发）...")
    start = time.time()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    futures = {executor.submit(answer_one, i, q): i for i, q in enumerate(questions)}

    try:
        for future in concurrent.futures.as_completed(futures, timeout=3600):
            try:
                idx, answer, rounds, chunks_count, err = future.result(timeout=300)
            except Exception as e:
                idx = futures[future]
                answer, rounds, chunks_count, err = f"Error: {e}", 0, 0, str(e)
            results[idx] = answer
            rounds_taken[idx] = rounds
            num_chunks[idx] = chunks_count
            completed += 1
            status = "ok" if not answer.startswith("Error") else "FAIL"
            if completed % 10 == 0 or completed == len(questions):
                elapsed = time.time() - start
                qps = completed / elapsed
                print(f"  [{completed}/{len(questions)}] {status} rounds={rounds} "
                      f"chunks={chunks_count} | {qps:.2f} q/s")
    except concurrent.futures.TimeoutError:
        print(f"\n超时！已完成 {completed}/{len(questions)}")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    for i in range(len(results)):
        if results[i] is None:
            results[i] = "Error: timeout"

    elapsed = time.time() - start
    errors = sum(1 for r in results if r.startswith("Error"))
    print(f"\n完成! 耗时 {elapsed:.1f}s, 成功 {len(results)-errors}/{len(results)}")

    # Flush Langfuse
    from graphrag_agent.utils.langfuse_client import flush_langfuse
    flush_langfuse()

    return results, rounds_taken, num_chunks, elapsed


# ========== 主流程 ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="测试题数（默认 50）")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--session", type=str, default=f"bench_{int(time.time())}",
                        help="Langfuse session 前缀，便于 UI 筛选")
    parser.add_argument("--questions", type=str, default=QUESTIONS_FILE,
                        help="问题 JSON 路径（默认 HotpotQA）")
    parser.add_argument("--tag", type=str, default="",
                        help="输出文件名后缀（区分数据集，如 rag_papers）")
    args = parser.parse_args()

    os.makedirs(BENCH_DIR, exist_ok=True)

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)[:args.n]
    print(f"加载 {len(questions)} 个问题 from {args.questions}")
    print(f"Langfuse session 前缀: {args.session}")

    # 跑 Agentic
    answers, rounds_taken, num_chunks, wall_time = run_agentic(
        questions, max_workers=args.workers, session_prefix=args.session
    )

    # ===== LLM-Eval（和 run_hotpotqa_bench.py 一致）=====
    from graphrag_agent.models.get_models import get_llm_model
    eval_llm = get_llm_model()

    def llm_eval_one(question, gold, prediction):
        try:
            prompt = f"""Please evaluate if the generated answer is correct by comparing it with the gold answer.
Generated answer: {prediction}
Gold answer: {gold}

The generated answer should be considered correct if it:
1. Contains the key information from the gold answer
2. Is factually accurate and consistent with the gold answer
3. Does not contain any contradicting information

Respond with ONLY 'correct' or 'incorrect'.
Response:"""
            response = eval_llm.invoke(prompt)
            content = response.content.strip().lower() if response.content else ""
            if "incorrect" in content:
                return 0.0
            elif "correct" in content:
                return 1.0
            return 0.0
        except Exception:
            return 0.0

    print("\n计算 LLM-Eval Accuracy（并行）...")
    llm_eval_results = [0.0] * len(questions)
    eval_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(os.getenv("EVAL_MAX_WORKERS", "10")))
    )
    futures = {
        eval_executor.submit(llm_eval_one, q["question"], q["answer"], a): i
        for i, (q, a) in enumerate(zip(questions, answers))
    }
    done_count = 0
    try:
        for future in concurrent.futures.as_completed(futures, timeout=600):
            try:
                idx = futures[future]
                llm_eval_results[idx] = future.result(timeout=60)
            except Exception:
                llm_eval_results[futures[future]] = 0.0
            done_count += 1
            if done_count % 20 == 0:
                print(f"  LLM-Eval: {done_count}/{len(questions)}")
    except concurrent.futures.TimeoutError:
        print(f"  LLM-Eval 超时，已完成 {done_count}/{len(questions)}")
    finally:
        eval_executor.shutdown(wait=False, cancel_futures=True)

    # ===== 汇总指标 =====
    em_scores, f1_scores, str_scores = [], [], []
    results_detail = []
    for i, (q, full_ans) in enumerate(zip(questions, answers)):
        gold = q["answer"]
        pred = extract_short_answer(full_ans)
        em = 1.0 if exact_match_score(pred, gold) else 0.0
        f1 = f1_score(pred, gold)
        str_acc = 1.0 if normalize_answer(gold) in normalize_answer(full_ans) else 0.0
        em_scores.append(em)
        f1_scores.append(f1)
        str_scores.append(str_acc)

        results_detail.append({
            "id": q.get("id", i),
            "question": q["question"],
            "type": q.get("question_type", "unknown"),
            "gold_answer": gold,
            "predicted_answer": pred,
            "full_answer": full_ans,
            "em": em,
            "f1": f1,
            "string_accuracy": str_acc,
            "llm_accuracy": llm_eval_results[i],
            "rounds": rounds_taken[i],
            "num_chunks": num_chunks[i],
        })

    n = len(questions)
    avg_em = sum(em_scores) / n
    avg_f1 = sum(f1_scores) / n
    avg_str = sum(str_scores) / n
    avg_llm = sum(llm_eval_results) / n

    # Agentic 专属指标
    valid_rounds = [r for r in rounds_taken if r > 0]
    avg_rounds = sum(valid_rounds) / len(valid_rounds) if valid_rounds else 0
    multi_round_pct = sum(1 for r in rounds_taken if r > 1) / n * 100
    avg_chunks = sum(num_chunks) / n

    print(f"\n{'='*60}")
    print(f"HotpotQA Agentic 评测结果  (n={n})")
    print(f"{'='*60}")
    print(f"准确率:")
    print(f"  String-Acc:  {avg_str:.4f}")
    print(f"  LLM-Acc:     {avg_llm:.4f}")
    print(f"  EM:          {avg_em:.4f}")
    print(f"  F1:          {avg_f1:.4f}")
    print(f"\nAgentic 迭代统计:")
    print(f"  平均轮次:        {avg_rounds:.2f}")
    print(f"  触发改写比例:    {multi_round_pct:.1f}%  (走 2+ 轮的 query)")
    print(f"  平均 chunks 数:  {avg_chunks:.1f}")
    print(f"\n耗时: {wall_time:.1f}s  ({n/wall_time:.2f} q/s)")

    # 按类型统计
    by_type = {}
    for r in results_detail:
        qt = r["type"]
        by_type.setdefault(qt, []).append(r)
    print(f"\n按类型:")
    for qt, rs in by_type.items():
        qt_str = sum(r["string_accuracy"] for r in rs) / len(rs)
        qt_llm = sum(r["llm_accuracy"] for r in rs) / len(rs)
        qt_rounds = sum(r["rounds"] for r in rs) / len(rs)
        print(f"  {qt}:  Str={qt_str:.3f}  LLM={qt_llm:.3f}  avg_rounds={qt_rounds:.2f}  (n={len(rs)})")

    # 保存
    output = {
        "method": "agentic_graph_search",
        "n": n,
        "wall_time": wall_time,
        "langfuse_session_prefix": args.session,
        "overall": {
            "em": avg_em, "f1": avg_f1, "string_accuracy": avg_str, "llm_accuracy": avg_llm,
            "avg_rounds": avg_rounds, "multi_round_pct": multi_round_pct,
            "avg_chunks": avg_chunks,
        },
        "details": results_detail,
    }
    tag_suffix = f"_{args.tag}" if args.tag else ""
    out_file = os.path.join(BENCH_DIR, f'hotpot_agentic{tag_suffix}_n{n}_eval.json')
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_file}")

    if os.getenv("LANGFUSE_SECRET_KEY"):
        print(f"\nLangfuse 查看 trace: http://localhost:3000")
        print(f"  在 Sessions 页面搜索前缀: {args.session}")


if __name__ == "__main__":
    main()
