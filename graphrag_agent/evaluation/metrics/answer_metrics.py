import re
from typing import Dict, List, Tuple
from graphrag_agent.evaluation.core.base_metric import BaseMetric
from graphrag_agent.evaluation.core.evaluation_data import AnswerEvaluationData
from graphrag_agent.evaluation.utils.text_utils import normalize_answer

class ExactMatch(BaseMetric):
    """精确匹配评估指标"""
    
    metric_name = "em"

    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)
    
    def calculate_em(self, prediction: str, golden_answer: str) -> float:
        """
        计算单个预测的精确匹配得分
        
        Args:
            prediction: 预测答案
            golden_answer: 标准答案
            
        Returns:
            float: 得分（1.0表示匹配，0.0表示不匹配）
        """
        if not prediction or not golden_answer:
            return 0.0
            
        normalized_prediction = normalize_answer(prediction)
        normalized_golden = normalize_answer(golden_answer)
        
        # 完全匹配
        if normalized_prediction == normalized_golden:
            return 1.0
        return 0.0
    
    def calculate_metric(self, data: AnswerEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算精确匹配指标 - 使用规则匹配和LLM回退混合评分
        
        Args:
            data: 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        self.log("======== ExactMatch 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
        
        golden_answers = data.golden_answers
        system_answers = data.system_answers
        
        # 第一步：规则评分 + 收集需要 LLM 的样本
        rule_scores = []
        llm_needed = []  # (idx, prompt, default_score)

        for idx, (pred, golden) in enumerate(zip(system_answers, golden_answers)):
            cleaned_pred = re.sub(r'^###.*?\n+', '', pred, flags=re.MULTILINE)
            cleaned_pred = re.sub(r'\n\s*\n', '\n', cleaned_pred).strip()
            normalized_pred = normalize_answer(cleaned_pred)
            normalized_golden = normalize_answer(golden)

            if normalized_pred == normalized_golden:
                rule_scores.append(1.0)
            else:
                similarity_score = self._calculate_content_similarity(cleaned_pred, golden)
                if similarity_score >= 0.7:
                    rule_scores.append(0.7 + (similarity_score - 0.7))
                elif self.llm:
                    rule_scores.append(None)  # 占位，等 LLM 结果
                    llm_needed.append((idx, f"""请比较下面两个答案，评估它们在内容上的等价性，给出0到1之间的分数。
0表示完全不同，1表示内容上完全等价。
请只考虑实质内容，忽略格式、表达方式和顺序的差异。

标准答案:
{golden}

系统答案:
{cleaned_pred}

只返回一个0到1之间的数字表示分数，不要有任何其他文字。""", similarity_score))
                else:
                    rule_scores.append(similarity_score)

        # 第二步：并行 LLM 评分
        if llm_needed:
            prompts = [item[1] for item in llm_needed]
            defaults = [item[2] for item in llm_needed]
            self.log(f"并行 LLM 评分 {len(prompts)} 个样本...")
            llm_scores = self.parallel_llm_score(prompts, default_score=0.5)
            for i, (idx, _, default) in enumerate(llm_needed):
                rule_scores[idx] = max(llm_scores[i], default)

        metric_score_list = rule_scores
        em_score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        self.log(f"精确匹配平均得分: {em_score:.4f}")
        self.log("======== ExactMatch 计算结束 ========\n")

        return {"em": em_score}, metric_score_list
    
    def _calculate_content_similarity(self, pred: str, golden: str) -> float:
        """
        计算两个文本的内容相似度
        
        Args:
            pred: 预测答案
            golden: 标准答案
            
        Returns:
            float: 内容相似度分数 (0-1)
        """
        # 标准化处理
        pred_norm = normalize_answer(pred).split()
        golden_norm = normalize_answer(golden).split()
        
        if not pred_norm or not golden_norm:
            return 0.0
            
        # 计算共有词的数量
        common_words = set(pred_norm) & set(golden_norm)
        
        # 计算Jaccard相似度
        union_words = set(pred_norm) | set(golden_norm)
        if union_words:
            jaccard = len(common_words) / len(union_words)
        else:
            jaccard = 0.0
            
        # 计算词覆盖率
        pred_coverage = len(common_words) / len(set(pred_norm)) if pred_norm else 0
        golden_coverage = len(common_words) / len(set(golden_norm)) if golden_norm else 0
        
        # 综合得分 - Jaccard占40%，两个覆盖率各占30%
        similarity = 0.4 * jaccard + 0.3 * pred_coverage + 0.3 * golden_coverage
        
        return similarity

class F1Score(BaseMetric):
    """F1分数评估指标"""
    
    metric_name = "f1"

    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)
    
    def calculate_metric(self, data: AnswerEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算F1分数 - 使用规则匹配和LLM回退混合评分
        
        Args:
            data: 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        self.log("\n======== F1Score 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
        
        golden_answers = data.golden_answers
        system_answers = data.system_answers
        
        import jieba

        # 第一步：规则 F1 + 收集需要 LLM 的样本
        rule_f1_list = []
        llm_needed = []  # (idx, prompt, rule_f1)

        for idx, (pred, golden) in enumerate(zip(system_answers, golden_answers)):
            cleaned_pred = re.sub(r'^###.*?\n+', '', pred, flags=re.MULTILINE)
            cleaned_pred = re.sub(r'\n\s*\n', '\n', cleaned_pred).strip()
            pred_text = normalize_answer(cleaned_pred)
            golden_text = normalize_answer(golden)

            try:
                pred_tokens = list(jieba.cut(pred_text))
                golden_tokens = list(jieba.cut(golden_text))
                stopwords = {'的', '了', '和', '在', '是', '为', '以', '与', '或', '且'}
                pred_tokens = [t for t in pred_tokens if len(t) > 1 and t not in stopwords]
                golden_tokens = [t for t in golden_tokens if len(t) > 1 and t not in stopwords]

                if not pred_tokens or not golden_tokens:
                    rule_f1 = 1.0 if (not pred_tokens and not golden_tokens) else 0.0
                else:
                    common = set(pred_tokens) & set(golden_tokens)
                    p = len(common) / len(pred_tokens)
                    r = len(common) / len(golden_tokens)
                    rule_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            except Exception:
                rule_f1 = 0.0

            rule_f1_list.append(rule_f1)

            if self.llm:
                llm_needed.append((idx, f"""请比较下面两个答案的内容相似度，评估它们包含的信息重叠程度，并给出0到1之间的分数。
0表示完全不同信息，1表示信息完全重叠。
请考虑实质内容的相似性，而不仅是表面文字的匹配。在评估时，请特别关注关键信息点是否一致。

标准答案:
{golden}

系统答案:
{cleaned_pred}

只返回一个0到1之间的数字表示分数，不要有任何其他文字。""", rule_f1))

        # 第二步：并行 LLM 评分
        f1_scores = list(rule_f1_list)
        if llm_needed:
            prompts = [item[1] for item in llm_needed]
            self.log(f"并行 LLM 评分 {len(prompts)} 个样本...")
            llm_scores = self.parallel_llm_score(prompts, default_score=0.5)
            for i, (idx, _, rule_f1) in enumerate(llm_needed):
                f1_scores[idx] = max(llm_scores[i], rule_f1)

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        self.log(f"F1平均得分: {avg_f1:.4f}")
        self.log("======== F1Score 计算结束 ========\n")

        return {"f1": avg_f1}, f1_scores