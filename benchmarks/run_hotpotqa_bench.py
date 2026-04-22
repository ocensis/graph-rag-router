"""
HotpotQA 评测脚本

用 Agent 回答问题，计算 Answer EM 和 F1（SQuAD 标准）

用法：
  python run_hotpotqa_bench.py --agent naive --workers 20
  python run_hotpotqa_bench.py --agent graph --workers 20
  python run_hotpotqa_bench.py --agent hybrid --workers 20
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


# ========== SQuAD 标准 EM/F1 评测（和 HotpotQA 官方一致）==========

def normalize_answer(s):
    """标准化答案文本"""
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
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def run_agent(agent_name, questions, max_workers=20):
    """用 Agent 回答所有问题"""
    from graphrag_agent.evaluation.utils.eval_utils import load_agent

    print(f"\n加载 {agent_name} Agent...")
    agent = load_agent(agent_name)
    if agent is None:
        print(f"加载 {agent_name} 失败!")
        return []

    # 强制清空内存缓存，确保用新的检索逻辑
    if hasattr(agent, 'cache_manager'):
        agent.cache_manager.clear() if hasattr(agent.cache_manager, 'clear') else None
    if hasattr(agent, 'global_cache_manager'):
        agent.global_cache_manager.clear() if hasattr(agent.global_cache_manager, 'clear') else None
    print(f"  已清空 {agent_name} 的内存缓存")

    results = [None] * len(questions)
    completed = 0

    def answer_one(idx, q):
        try:
            answer = agent.ask(q['question'], skip_cache=True)
            if isinstance(answer, dict):
                answer = answer.get('answer', answer.get('content', str(answer)))
            return idx, str(answer) if answer else ""
        except Exception as e:
            return idx, f"Error: {e}"

    print(f"开始回答 {len(questions)} 个问题（{max_workers} 并发）...")
    start = time.time()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    futures = {executor.submit(answer_one, i, q): i for i, q in enumerate(questions)}

    try:
        for future in concurrent.futures.as_completed(futures, timeout=1800):
            try:
                idx, answer = future.result(timeout=120)
            except Exception as e:
                idx = futures[future]
                answer = f"Error: {e}"
            results[idx] = answer
            completed += 1
            status = "ok" if not str(answer).startswith("Error") else "FAIL"
            print(f"  [{completed}/{len(questions)}] {status} | {questions[idx]['question'][:50]}...")
    except concurrent.futures.TimeoutError:
        print(f"\n超时！已完成 {completed}/{len(questions)}，跳过剩余任务")
    finally:
        # 强制关闭，不等卡住的线程
        executor.shutdown(wait=False, cancel_futures=True)

    for i in range(len(results)):
        if results[i] is None:
            results[i] = "Error: timeout"

    elapsed = time.time() - start
    errors = sum(1 for r in results if str(r).startswith("Error"))
    print(f"\n完成! 耗时 {elapsed:.1f}s, 成功 {len(results)-errors}/{len(results)}")
    return results


def extract_short_answer(full_answer):
    """从 Agent 的长回答中提取简短答案"""
    if not full_answer or full_answer.startswith("Error"):
        return ""

    # 尝试提取 **加粗** 的内容（Agent 通常会加粗关键答案）
    bold_matches = re.findall(r'\*\*(.+?)\*\*', full_answer)
    if bold_matches:
        return bold_matches[0]

    # 取第一句话
    first_line = full_answer.strip().split('\n')[0]
    # 去掉 markdown 标题
    first_line = re.sub(r'^#+\s*', '', first_line)
    return first_line[:200]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="naive")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--questions", type=str, default=QUESTIONS_FILE,
                        help="评测问题 JSON 文件路径（默认 HotpotQA）")
    parser.add_argument("--tag", type=str, default=None,
                        help="输出文件名后缀，区分不同数据集 (如 rag_papers)")
    parser.add_argument("--n", type=int, default=None, help="只测前 N 条")
    args = parser.parse_args()

    os.makedirs(BENCH_DIR, exist_ok=True)

    # 加载问题
    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if args.n:
        questions = questions[:args.n]
    print(f"加载 {len(questions)} 个问题 from {args.questions}")

    # 跑 Agent
    answers = run_agent(args.agent, questions, args.workers)

    # 计算 EM、F1 和 LLM-Eval Accuracy
    em_scores = []
    f1_scores = []
    llm_acc_scores = []
    results_detail = []

    # LLM-Eval: 用 LLM 判断回答是否正确（和 LinearRAG/LogicRAG 一致）
    from graphrag_agent.models.get_models import get_llm_model
    eval_llm = get_llm_model()

    def llm_eval_one(question, gold, prediction):
        """LLM 判断 - 使用 LinearRAG/LogicRAG 论文的标准 prompt"""
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
            # 注意：不能用 "correct" in content，因为 "incorrect" 也包含 "correct"
            if "incorrect" in content:
                return 0.0
            elif "correct" in content:
                return 1.0
            return 0.0
        except Exception:
            return 0.0

    print("\n计算 LLM-Eval Accuracy（并行）...")
    llm_eval_results = [0.0] * len(questions)
    eval_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(os.getenv("EVAL_MAX_WORKERS", "10"))))
    futures = {}
    for i, (q, full_ans) in enumerate(zip(questions, answers)):
        futures[eval_executor.submit(llm_eval_one, q["question"], q["answer"], full_ans)] = i
    done_count = 0
    try:
        for future in concurrent.futures.as_completed(futures, timeout=300):
            try:
                idx = futures[future]
                llm_eval_results[idx] = future.result(timeout=60)
            except Exception:
                idx = futures[future]
                llm_eval_results[idx] = 0.0
            done_count += 1
            if done_count % 10 == 0:
                print(f"  LLM-Eval: {done_count}/{len(questions)}")
    except concurrent.futures.TimeoutError:
        print(f"\n  LLM-Eval 超时，已完成 {done_count}/{len(questions)}")
    finally:
        eval_executor.shutdown(wait=False, cancel_futures=True)

    for i, (q, full_ans) in enumerate(zip(questions, answers)):
        gold = q["answer"]
        pred = extract_short_answer(full_ans)

        em = 1.0 if exact_match_score(pred, gold) else 0.0
        f1 = f1_score(pred, gold)
        llm_acc = llm_eval_results[i]
        # String-Acc: gold answer 是否包含在 normalized prediction 中（和论文一致）
        string_acc = 1.0 if normalize_answer(gold) in normalize_answer(full_ans) else 0.0
        em_scores.append(em)
        f1_scores.append(f1)
        llm_acc_scores.append(llm_acc)

        results_detail.append({
            "id": q["id"],
            "question": q["question"],
            "type": q["question_type"],
            "gold_answer": gold,
            "predicted_answer": pred,
            "full_answer": full_ans,
            "em": em,
            "f1": f1,
            "string_accuracy": string_acc,
            "llm_accuracy": llm_acc,
        })

    # 按类型统计
    by_type = {}
    for r in results_detail:
        qt = r["type"]
        by_type.setdefault(qt, {"em": [], "f1": [], "string_accuracy": [], "llm_accuracy": []})
        by_type[qt]["em"].append(r["em"])
        by_type[qt]["f1"].append(r["f1"])
        by_type[qt]["string_accuracy"].append(r["string_accuracy"])
        by_type[qt]["llm_accuracy"].append(r["llm_accuracy"])

    # 打印结果
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    avg_str = sum(r["string_accuracy"] for r in results_detail) / len(results_detail) if results_detail else 0
    avg_llm = sum(llm_acc_scores) / len(llm_acc_scores) if llm_acc_scores else 0

    print(f"\n{'='*50}")
    print(f"HotpotQA 评测结果 - {args.agent}")
    print(f"{'='*50}")
    print(f"Overall:  String-Acc={avg_str:.4f}  LLM-Acc={avg_llm:.4f}  EM={avg_em:.4f}  F1={avg_f1:.4f}")
    for qt, scores in by_type.items():
        qt_em = sum(scores["em"]) / len(scores["em"])
        qt_f1 = sum(scores["f1"]) / len(scores["f1"])
        qt_str = sum(scores["string_accuracy"]) / len(scores["string_accuracy"])
        qt_llm = sum(scores["llm_accuracy"]) / len(scores["llm_accuracy"])
        print(f"  {qt}:  String-Acc={qt_str:.4f}  LLM-Acc={qt_llm:.4f}  EM={qt_em:.4f}  F1={qt_f1:.4f}  (n={len(scores['em'])})")

    # 保存
    output = {
        "agent": args.agent,
        "overall": {"em": avg_em, "f1": avg_f1, "string_accuracy": avg_str, "llm_accuracy": avg_llm, "count": len(questions)},
        "by_type": {qt: {"em": sum(s["em"])/len(s["em"]), "f1": sum(s["f1"])/len(s["f1"]), "string_accuracy": sum(s["string_accuracy"])/len(s["string_accuracy"]), "llm_accuracy": sum(s["llm_accuracy"])/len(s["llm_accuracy"]), "count": len(s["em"])} for qt, s in by_type.items()},
        "details": results_detail,
    }

    tag = f"_{args.tag}" if args.tag else ""
    output_file = os.path.join(BENCH_DIR, f'hotpot_{args.agent}{tag}_eval.json')
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_file}")

    # Langfuse callback 是 async buffered——process 退出前必须 flush，否则 trace 全丢
    try:
        from graphrag_agent.utils.langfuse_client import flush_langfuse
        flush_langfuse()
        print("Langfuse trace 已 flush")
    except Exception as e:
        print(f"Langfuse flush 失败（不影响 bench 结果）: {e}")


if __name__ == '__main__':
    main()
