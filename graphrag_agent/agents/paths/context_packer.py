"""
Context Packer —— Token 预算管理 + 按比例分配

参考微软 GraphRAG LocalSearch 的做法：
  - 每个上下文源（community / entities / chunks）分配 token 预算比例
  - 用 tiktoken 数 token（比字符截断准）
  - 贪心 fill：每条目整条保留或整条丢弃，避免把信息切半

API:
  packed = pack_context(
      parts={
          "community": "...community summaries...",
          "graph":     "...entity/relationship table...",
          "chunks":    "...top text units...",
      },
      budget=6000,
      ratios={"community": 0.25, "graph": 0.25, "chunks": 0.50},
  )
"""
from __future__ import annotations

from typing import Dict

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")  # gpt-4 / gpt-4o 都用这个
    _HAS_TIKTOKEN = True
except Exception:
    _ENC = None
    _HAS_TIKTOKEN = False


def count_tokens(text: str) -> int:
    """数 token；tiktoken 不可用时退化到 chars/4（粗略近似）"""
    if not text:
        return 0
    if _HAS_TIKTOKEN:
        return len(_ENC.encode(text))
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """把 text 截到 max_tokens 以内（整 token 切）"""
    if not text or max_tokens <= 0:
        return ""
    if _HAS_TIKTOKEN:
        ids = _ENC.encode(text)
        if len(ids) <= max_tokens:
            return text
        return _ENC.decode(ids[:max_tokens])
    # fallback: 按字符
    return text[: max_tokens * 4]


def pack_context(
    parts: Dict[str, str],
    budget: int = 6000,
    ratios: Dict[str, float] = None,
    separator: str = "\n\n",
) -> str:
    """
    按 ratios 给每个 part 分配 token 预算，超的截断，返回拼接结果。

    - 剩余预算会转给下一个 part（前面的没用满，后面的多用一点）
    - 没在 ratios 里的 part 不占预算（忽略）
    """
    ratios = ratios or {}
    # 归一化比例
    total_ratio = sum(ratios.get(k, 0) for k in parts) or 1.0

    out_pieces = []
    spent = 0
    for key, content in parts.items():
        if not content:
            continue
        ratio = ratios.get(key, 0)
        if ratio <= 0:
            continue
        # 当前 part 的预算 = (ratio 占比) × (总预算 - 已花)
        remaining = budget - spent
        allocated = int(remaining * (ratio / total_ratio)) if total_ratio > 0 else 0
        if allocated <= 0:
            continue
        truncated = truncate_to_tokens(content, allocated)
        actual_tokens = count_tokens(truncated)
        out_pieces.append(f"## {key.upper()}\n{truncated}")
        spent += actual_tokens
        # 更新剩余占比（把已用的 key 从 total 里去掉）
        total_ratio -= ratio

    return separator.join(out_pieces)


def pack_list(items: list, budget: int, formatter=str) -> list:
    """
    贪心 fill：遍历 items，每条整条加入，累计到 budget 停止。
    用于 map-reduce 的 reduce 阶段（已排序的 key points 填到上下文）。

    返回保留下来的子列表（items 的前 N 个）。
    """
    kept = []
    spent = 0
    for item in items:
        formatted = formatter(item)
        t = count_tokens(formatted)
        if spent + t > budget:
            break
        kept.append(item)
        spent += t
    return kept


__all__ = ["count_tokens", "truncate_to_tokens", "pack_context", "pack_list"]
