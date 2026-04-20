"""
Langfuse SDK 前端共享 helper

- fetch_traces_by_name: 分页拉 traces（带重试，应对 ClickHouse 偶发 500）
- fetch_observations: 单 trace 所有 span（带重试）
- analyze_trace: 从 spans 判 route / 数 LLM 调用 / 汇总 cost
- build_observation_tree: 把 flat spans 建成父子树（Trace Viewer 用）
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from collections import Counter
from typing import List, Dict, Optional

from langfuse import Langfuse


# ==================== Client ====================

_client = None


def get_client() -> Optional[Langfuse]:
    """单例 Langfuse client，key 缺失就 None。
    关键: timeout=15 避免 ClickHouse 慢响应把整个前端卡死（默认 120s 太长）"""
    global _client
    if _client is None:
        sk = os.getenv("LANGFUSE_SECRET_KEY")
        pk = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_BASE_URL")
        if not (sk and pk and host):
            return None
        _client = Langfuse(secret_key=sk, public_key=pk, host=host, timeout=15)
    return _client


# ==================== Fetch with retry ====================

def fetch_traces_by_name(client: Langfuse, name: str, hours: int, limit: int,
                         retries: int = 3) -> List:
    """按 name 分页拉 trace，带重试"""
    from_ts = datetime.utcnow() - timedelta(hours=hours)
    traces: List = []
    page = 1
    while len(traces) < limit:
        resp = None
        for attempt in range(retries):
            try:
                resp = client.fetch_traces(
                    from_timestamp=from_ts, limit=min(100, limit - len(traces)),
                    page=page, name=name,
                )
                break
            except Exception:
                if attempt < retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return traces[:limit]
        if not resp or not resp.data:
            break
        traces.extend(resp.data)
        if len(resp.data) < 100:
            break
        page += 1
    return traces[:limit]


def fetch_observations(client: Langfuse, trace_id: str, retries: int = 3):
    """拉单 trace 的所有 observation，带重试"""
    for attempt in range(retries):
        try:
            return client.fetch_observations(trace_id=trace_id).data
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
    return None


# ==================== Trace analysis ====================

ROUTE_PATH_NAMES = {"classic_path", "graph_path", "agentic_path"}
TOOL_NAMES = {
    "hybrid_search", "graph_lookup", "path_search", "fetch_document", "entity_search",
    "classic_search", "graph_search",
}


def analyze_trace(trace, obs_list) -> Dict:
    """从 observations 提取 route / LLM calls / tool usage / cost"""
    route = None
    llm_calls = 0
    tool_calls = 0
    tools_used = Counter()
    models = set()
    cost = 0.0
    subqueries = 0

    for o in obs_list or []:
        name = getattr(o, "name", "") or ""

        if name in ROUTE_PATH_NAMES and route is None:
            route = name.replace("_path", "")

        if getattr(o, "type", "") == "GENERATION":
            llm_calls += 1
            if hasattr(o, "calculated_total_cost") and o.calculated_total_cost:
                cost += float(o.calculated_total_cost)
            elif hasattr(o, "total_cost") and o.total_cost:
                cost += float(o.total_cost)
            if hasattr(o, "model") and o.model:
                models.add(o.model)

        if name.startswith("subquery_"):
            subqueries += 1

        for tn in TOOL_NAMES:
            if name == tn or name.startswith(f"{tn}("):
                tool_calls += 1
                tools_used[tn] += 1
                break

    return {
        "trace_id": trace.id,
        "query": (trace.input or {}).get("query", "") if isinstance(trace.input, dict) else "",
        "latency": trace.latency or 0.0,
        "route": route,
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "tools_used": dict(tools_used),
        "subqueries": subqueries,
        "cost": cost,
        "models": list(models),
        "timestamp": trace.timestamp if hasattr(trace, "timestamp") else None,
    }


# ==================== Observation tree (Trace Viewer 用) ====================

def build_observation_tree(obs_list) -> List[Dict]:
    """
    把 flat observations 按 parent_observation_id 建成父子树
    返回 root list，每个 node = {
        "obs": <raw observation>,
        "children": [...]
    }
    """
    if not obs_list:
        return []

    by_id = {o.id: {"obs": o, "children": []} for o in obs_list}
    roots = []

    for o in obs_list:
        node = by_id[o.id]
        parent_id = getattr(o, "parent_observation_id", None)
        if parent_id and parent_id in by_id:
            by_id[parent_id]["children"].append(node)
        else:
            roots.append(node)

    # 每层按 startTime 排序（让时序正确）
    def _sort(nodes):
        nodes.sort(key=lambda n: getattr(n["obs"], "start_time", None) or 0)
        for n in nodes:
            _sort(n["children"])

    _sort(roots)
    return roots
