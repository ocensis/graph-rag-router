"""
LLM 输出的 Pydantic 校验模型

面试点：LLM 输出不确定性怎么处理？
→ Pydantic 校验 JSON schema，不合规就 retry
"""
import re
from pydantic import BaseModel, field_validator
from typing import Optional


class SufficiencyCheck(BaseModel):
    """充分性判断的 LLM 输出结构"""
    sufficient: bool
    missing: str = ""
    subquery: str = ""

    @classmethod
    def parse_llm_output(cls, content: str) -> "SufficiencyCheck":
        """
        从 LLM 的自由文本输出中解析结构化结果
        如果解析失败，返回默认 sufficient=True（保守降级）
        """
        if not content or not isinstance(content, str):
            return cls(sufficient=True)

        content_lower = content.lower()

        # 解析 sufficient
        sufficient = (
            "sufficient: yes" in content_lower
            or "sufficient:yes" in content_lower.replace(" ", "")
        )

        # 解析 missing 和 subquery
        missing = ""
        subquery = ""
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("MISSING:"):
                missing = stripped.split(":", 1)[1].strip()
            elif stripped.upper().startswith("SUBQUERY:"):
                subquery = stripped.split(":", 1)[1].strip()

        return cls(sufficient=sufficient, missing=missing, subquery=subquery)


class RerankerScore(BaseModel):
    """Reranker LLM 评分输出"""
    score: int

    @field_validator("score")
    @classmethod
    def clamp_score(cls, v):
        return max(0, min(10, v))

    @classmethod
    def parse_llm_output(cls, content: str) -> "RerankerScore":
        """从 LLM 输出中提取 0-10 的整数分数"""
        if not content:
            return cls(score=0)
        match = re.search(r'\d+', content)
        if match:
            return cls(score=int(match.group()))
        return cls(score=0)
