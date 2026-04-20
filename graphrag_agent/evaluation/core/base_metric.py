from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

class BaseMetric(ABC):
    """所有评估指标的基类"""
    
    # 指标名称，子类必须重写
    metric_name = "base"
    
    def __init__(self, config):
        """
        初始化评估指标基类
        
        Args:
            config: 评估配置
        """
        # 支持字典或EvaluatorConfig对象
        if isinstance(config, dict):
            from graphrag_agent.evaluation.evaluator_config.evaluatorConfig import EvaluatorConfig
            self.config = EvaluatorConfig(config)
        else:
            self.config = config
            
        self.dataset_name = self.config.get('dataset_name', 'default')
        self.debug = self.config.get('debug', False)
        # 获取LLM模型，用于回退评估
        self.llm = self.config.get('llm', None)
    
    @abstractmethod
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List]:
        """
        计算评估指标
        
        Args:
            data: 评估数据对象
            
        Returns:
            Tuple[Dict, List]: 评估结果和每个样本的评分
        """
        return {}, []
    
    def log(self, message, *args, **kwargs):
        """
        输出调试日志
        
        Args:
            message: 日志消息
            *args, **kwargs: 额外参数
        """
        from graphrag_agent.evaluation import debug_print
        if self.debug:
            debug_print(f"[{self.__class__.__name__}] {message}", *args, **kwargs)
            
    def parallel_llm_score(self, prompts: List[str], default_score: float = 0.5) -> List[float]:
        """
        并行调用 LLM 对多个 prompt 打分，带重试机制（应对 503 等临时错误）
        """
        import os
        import re
        import time as _time
        import concurrent.futures

        if not self.llm:
            return [default_score] * len(prompts)

        max_workers = int(os.getenv("EVAL_MAX_WORKERS", "5"))
        max_retries = 3
        scores = [default_score] * len(prompts)

        def _score_one(idx, prompt):
            for attempt in range(max_retries):
                try:
                    response = self.llm.invoke(prompt)
                    text = response.content if hasattr(response, 'content') else str(response)
                    match = re.search(r'(\d+(\.\d+)?)', text)
                    if match:
                        s = float(match.group(1))
                        return idx, max(0.0, min(1.0, s))
                    return idx, default_score
                except Exception as e:
                    err_str = str(e).lower()
                    if '503' in err_str or 'unavailable' in err_str or 'overloaded' in err_str:
                        _time.sleep(2 * (attempt + 1))
                        continue
                    return idx, default_score
            return idx, default_score

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_score_one, i, p): i for i, p in enumerate(prompts)}
            for future in concurrent.futures.as_completed(futures):
                idx, score = future.result()
                scores[idx] = score

        return scores

    def get_llm_fallback_score(self, prompt: str, default_score: float = 0.5) -> float:
        """
        使用LLM进行回退评分
        
        Args:
            prompt: 提示文本
            default_score: 默认分数，当LLM评分失败时返回
            
        Returns:
            float: LLM评分结果或默认分数
        """
        # 如果没有LLM，直接返回默认分数
        if not self.llm:
            self.log(f"  LLM不可用，使用默认分数: {default_score:.4f}")
            return default_score
            
        import re
        import time as _time
        for attempt in range(3):
            try:
                response = self.llm.invoke(prompt)
                score_text = response.content if hasattr(response, 'content') else response
                score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                if score_match:
                    return max(0.0, min(1.0, float(score_match.group(1))))
                return default_score
            except Exception as e:
                err_str = str(e).lower()
                if '503' in err_str or 'unavailable' in err_str or 'overloaded' in err_str:
                    _time.sleep(2 * (attempt + 1))
                    continue
                self.log(f"  LLM评分出错: {e}")
                return default_score
        return default_score