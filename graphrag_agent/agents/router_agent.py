"""
RouterAgent —— Classifier + 3-way dispatch (Classic / Graph / Agentic)

关键设计:
  - 不是 cascade，是 classifier 一次性分派（降低延迟 + 避免中间档答不好污染最终）
  - Classic 是默认路径（覆盖 70-90% 企业 KB 查询）
  - Graph 只处理实体关系 / 因果链 / 全局汇总
  - Agentic 是复杂规划，内部把 Classic 和 Graph 都当 sub-tool 调用

路由规则（对齐企业场景）:
  - 默认 → Classic
  - 含明确实体 + 关系信号 → Graph (local)
  - 含 "summarize / across all / overall" → Graph (global)
  - 含多约束 / 多步 / 多源 → Agentic

Langfuse: 顶层 router trace，3 条 path 各自独立 LangGraph subgraph，span 嵌套清晰
"""
from __future__ import annotations

import time
import uuid
from typing import TypedDict, Optional, Dict

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graphrag_agent.models.get_models import get_llm_model
from graphrag_agent.utils.logger import get_logger
from graphrag_agent.utils.resilience import retry_sync
from graphrag_agent.utils.langfuse_client import get_langfuse_handler

from graphrag_agent.agents.paths import ClassicPath, GraphPath, AgenticPath

logger = get_logger(__name__)


# ==================== State ====================

class RouterState(TypedDict, total=False):
    query: str             # 原始 user query（记录用）
    rewritten_query: str   # 经 history-aware rewriter 改写后的自包含 query，下游所有节点都吃这个
    session_id: str
    route: str       # "classic" | "graph" | "agentic"
    confidence: str  # "high" | "medium" | "low"
    answer: str
    # 兜底机制相关
    verify_ok: bool        # verifier 判定：True=答得上 False=需要升级
    verify_reason: str     # verifier 给的理由（debug 用）
    escalated: bool        # 是否触发了升级到 agentic（观测用，避免无限循环）


# ==================== Rewriter Prompt ====================
#
# 多轮对话里用户经常省略主语 / 用指代词 / 只说增量：
#   Turn 1: "What dataset does UniDoc-RL use?"
#   Turn 2: "What about VISOR?"        ← 必须结合 Turn 1 才能理解
#   Turn 3: "And their accuracy?"      ← 需要追溯到 Turn 1/2 的主题
#
# Rewriter 在 classify 之前把 Turn 2/3 改写成自包含形式，
# 让下游 classifier / retrieval 看到的是完整问题。
# 空 history → 直接透传（不浪费 LLM 调用）
#
# 1 次 LLM call 成本，~$0.0001，延迟 ~300-500ms。

REWRITER_PROMPT = """You rewrite the user's latest question so it stands on its own without needing the chat history.

## Rules
- Resolve pronouns / references ("it", "that paper", "their accuracy") to the concrete entity they refer to
- Fill in implicit subject / scope when the question is an elliptical follow-up ("And X?", "How about Y?")
- Preserve the exact original intent; do NOT answer, explain, or expand scope
- If the question is already self-contained, return it unchanged
- Return ONLY the rewritten question, no preamble, no quotes

## Examples

History:
- user: What dataset does UniDoc-RL evaluate on?
Current: What about VISOR?
Rewritten: What dataset does VISOR evaluate on?

History:
- user: What are the main RL training frameworks across these RAG papers?
- user: Which one does MM-Doc-R1 use?
Current: And its reward design?
Rewritten: What is the reward design of the RL training framework that MM-Doc-R1 uses?

History:
- user: Compare UniDoc-RL and VISOR on retrieval accuracy.
Current: What benchmarks did you use to compare them?
Rewritten: What benchmarks are used to compare UniDoc-RL and VISOR on retrieval accuracy?

History: (empty)
Current: What is GRPO?
Rewritten: What is GRPO?

## Input

{history_block}
Current: {query}
Rewritten:"""


# ==================== Verifier Prompt ====================
#
# 在 classic / graph path 跑完后判"这个答案回答到点上了吗"。
# 关键是区分两类"失败"：
#   - 明显 hedge / 拒绝 ("I don't know", "Not enough context") → 直接 fail，不用 LLM
#   - 看起来像答案但其实没答到点 → 需要 LLM 判
# Pass → 直接返回
# Fail → 升级到 agentic 重跑（agentic 会做 plan + 多轮 retrieval，兜住大部分 classic/graph 漏的 case）

VERIFIER_PROMPT = """You are a quality judge for a retrieval-augmented system. Decide whether the ANSWER actually addresses the QUESTION.

## Pass criteria (ok=true)
- Answer contains specific, concrete information directly responsive to the question
- For factual questions: names a concrete method/dataset/number/fact
- For comparison questions: covers both/all sides with specifics

## Fail criteria (ok=false)
- Answer is a refusal or hedge ("I don't know", "insufficient context", "cannot find")
- Answer restates the question without answering
- Answer is tangential / off-topic / too vague
- Answer contradicts itself or mentions the wrong entity

## Output (JSON on ONE line)
{{"ok": true|false, "reason": "<10 words max>"}}

## Examples

Q: "What dataset does UniDoc-RL evaluate on?"
A: "UniDoc-RL evaluates on MMLongBench-Doc and DocVQA benchmarks."
Output: {{"ok": true, "reason": "names specific datasets"}}

Q: "What dataset does UniDoc-RL evaluate on?"
A: "I don't have sufficient information about UniDoc-RL's evaluation."
Output: {{"ok": false, "reason": "explicit hedge"}}

Q: "Compare GRPO and SPO reward design."
A: "Both GRPO and SPO are reinforcement learning methods used in RAG papers."
Output: {{"ok": false, "reason": "generic, no comparison specifics"}}

Q: "What is the main contribution of MM-Doc-R1?"
A: "MM-Doc-R1 introduces a multi-modal reasoning framework based on SPO reward."
Output: {{"ok": true, "reason": "states concrete contribution"}}

## Input
Question: {query}
Answer: {answer}
Output:"""


# 明显 hedge 短路词：命中任一就直接 fail 不调 LLM 省钱
_HEDGE_MARKERS = (
    "i don't know",
    "i do not know",
    "insufficient context",
    "insufficient information",
    "not enough information",
    "cannot find",
    "could not find",
    "no relevant information",
    "unable to answer",
    "unable to determine",
    "我无法",
    "信息不足",
    "没有足够",
    "无法找到",
)


# ==================== Classifier Prompt ====================
#
# 规则设计参考企业 GraphRAG 场景：
# - 默认 classic，除非有明确 graph/agentic 信号
# - graph 信号：命名实体 + 关系词 / 因果链 / 全局汇总
# - agentic 信号：多约束 / 多步 / 多源 / 不确定用哪个 retriever

CLASSIFIER_PROMPT = """You are a query router for a RAG system over academic papers. Classify the query into one of 3 paths.

## Paths

**classic** — default, covers 70-90% of questions. Use for:
- Factual questions about a concept, method, dataset, or number
- Answer is likely in specific doc chunks
- Single-entity questions where the fact-of-interest is named directly

**graph** — specialized, use when:
- Question mentions clear entities AND their relationship (who uses what, who depends on whom)
- Question asks about causal / process / dependency chains across documents
- Question asks for corpus-wide summary/trends/themes ("across all papers", "main approaches", "common techniques")

**agentic** — complex planning, use when:
- Question has MULTIPLE distinct constraints that need separate retrieval steps
- Question needs multi-step decomposition before answering
- Enumeration across many papers with qualifier filters
- Multi-way comparison (3+ entities or axes)

## Examples

Q: "What dataset does UniDoc-RL evaluate on?"                         → classic (high)
Q: "What is the SPO reward formula in MM-Doc-R1?"                     → classic (high)
Q: "How do UniDoc-RL and VISOR differ in RL training?"                → classic (medium; 2-way compare often in chunks)
Q: "What is the relationship between GRPO and SPO across papers?"     → graph (high; relational)
Q: "Summarize the main approaches to multi-hop reasoning in these papers." → graph (high; corpus-wide)
Q: "Which papers use GRPO AND evaluate on MMLongbench?"               → agentic (high; multi-constraint)
Q: "Which papers that use reinforcement learning also address security?" → agentic (high; multi-constraint)
Q: "Compare methods from UniDoc-RL, MM-Doc-R1, and Doc-V* on training efficiency." → agentic (medium; 3-way)

## Output

Respond in JSON on ONE line:
{{"route": "classic" | "graph" | "agentic", "confidence": "high" | "medium" | "low"}}

- high: signals clearly match one path
- medium: primary path chosen but could overlap with another
- low: signals unclear → low confidence means caller will fallback to classic

Question: {query}
Output:"""


# ==================== Router ====================

HISTORY_WINDOW = 5  # rewriter 看最近 N 轮；大了成本涨，小了丢长依赖上下文


class RouterAgent:
    """Classifier + 3-way dispatch router，带 history-aware query rewriter 支持多轮对话。"""

    def __init__(self):
        # 3 条 path
        self.classic = ClassicPath(reranker_mode="none")
        self.graph = GraphPath()
        self.agentic = AgenticPath(classic_path=self.classic, graph_path=self.graph)

        self.llm = get_llm_model()
        self.classifier_chain = (
            ChatPromptTemplate.from_messages([("human", CLASSIFIER_PROMPT)])
            | self.llm | StrOutputParser()
        )
        self.rewriter_chain = (
            ChatPromptTemplate.from_messages([("human", REWRITER_PROMPT)])
            | self.llm | StrOutputParser()
        )
        self.verifier_chain = (
            ChatPromptTemplate.from_messages([("human", VERIFIER_PROMPT)])
            | self.llm | StrOutputParser()
        )

        # 简单缓存（同一 query 分类结果复用）
        self._classify_cache: Dict[str, Dict] = {}
        self.cache_manager = self.classic.cache_manager
        self.route_counts = {"classic": 0, "graph": 0, "agentic": 0, "error": 0}
        self.verify_counts = {"pass": 0, "fail_hedge": 0, "fail_llm": 0, "escalated": 0}

        # session_id → [(user_query, answer), ...]  最新轮在末尾
        # 进程内存即可；Redis 那种分布式 session 不是本项目范围
        self._session_history: Dict[str, list] = {}

        self._graph = self._build_graph()

    # ==================== Session history ====================

    def _get_history_block(self, session_id: Optional[str]) -> str:
        """把 session 的历史轮拼成一块给 rewriter 吃；空就返回 '(empty)'。"""
        if not session_id:
            return "History: (empty)"
        hist = self._session_history.get(session_id, [])
        if not hist:
            return "History: (empty)"
        # 只取最近 HISTORY_WINDOW 轮，每轮记一行 user query（答案不进 rewriter 上下文——
        # rewriter 只需要知道"上文问过什么"来解析指代，答案细节反而会让它 over-rewrite）
        recent = hist[-HISTORY_WINDOW:]
        lines = ["History:"] + [f"- user: {q}" for q, _ in recent]
        return "\n".join(lines)

    def _append_history(self, session_id: Optional[str], query: str, answer: str):
        if not session_id:
            return
        # 只保留答案前 200 字符，避免内存无上限增长
        trimmed = (answer or "")[:200]
        self._session_history.setdefault(session_id, []).append((query, trimmed))
        # 硬上限防异常 session 撑爆
        if len(self._session_history[session_id]) > HISTORY_WINDOW * 4:
            self._session_history[session_id] = self._session_history[session_id][-HISTORY_WINDOW * 2:]

    # ==================== Helpers ====================

    @staticmethod
    def _parse_classifier(raw: str) -> Dict[str, str]:
        import json as _json
        import re as _re
        raw = raw.strip()
        m = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, _re.DOTALL)
        if m:
            raw = m.group(1)
        m = _re.search(r"\{.*?\}", raw, _re.DOTALL)
        if m:
            try:
                obj = _json.loads(m.group(0))
                return {
                    "route": str(obj.get("route", "")).lower(),
                    "confidence": str(obj.get("confidence", "medium")).lower(),
                }
            except _json.JSONDecodeError:
                pass
        # Fallback: keyword extraction
        low = raw.lower()
        for r in ("agentic", "graph", "classic"):
            if r in low:
                return {"route": r, "confidence": "low"}
        return {"route": "classic", "confidence": "low"}

    @staticmethod
    def _scoped(config: Optional[Dict], name: str) -> Optional[Dict]:
        if not config:
            return config
        c = dict(config)
        c["run_name"] = name
        return c

    # ==================== Nodes ====================

    def _node_rewrite(self, state: RouterState, config: Dict = None) -> RouterState:
        """
        History-aware query rewriter。
        - 空 history → 直接透传（跳过 LLM call 省钱）
        - 有 history → LLM 改写成自包含 query
        - LLM 失败 → fallback 原 query（不让 rewriter 故障阻塞整条链路）
        """
        original = state["query"]
        session_id = state.get("session_id")
        history_block = self._get_history_block(session_id)

        # 空 history: pass-through
        if history_block == "History: (empty)":
            logger.info(f"[rewrite] no history, passthrough | {original[:60]}")
            return {"rewritten_query": original}

        try:
            rewritten = retry_sync(max_retries=2, base_delay=0.5)(
                self.rewriter_chain.invoke
            )(
                {"history_block": history_block, "query": original},
                config=self._scoped(config, "rewriter"),
            )
            rewritten = (rewritten or "").strip().strip('"').strip("'")
            # 防御：LLM 输出为空或巨长（说明跑飞了）就回退原 query
            if not rewritten or len(rewritten) > len(original) * 6:
                logger.warning(f"[rewrite] bad output, fallback | {rewritten[:60]}")
                rewritten = original
            else:
                logger.info(
                    f"[rewrite] '{original[:40]}' -> '{rewritten[:60]}'",
                    extra={"original": original, "rewritten": rewritten},
                )
        except Exception as e:
            logger.error(f"[rewrite] failed: {e} → fallback original")
            rewritten = original

        return {"rewritten_query": rewritten}

    def _node_classify(self, state: RouterState, config: Dict = None) -> RouterState:
        # classify 看的是改写后的 query（完整版本）
        query = state.get("rewritten_query") or state["query"]
        if query in self._classify_cache:
            cached = self._classify_cache[query]
            return {"route": cached["route"], "confidence": cached["confidence"]}

        route = "classic"
        confidence = "low"
        try:
            raw = retry_sync(max_retries=2, base_delay=0.5)(self.classifier_chain.invoke)(
                {"query": query},
                config=self._scoped(config, "classifier"),
            )
            parsed = self._parse_classifier(raw)
            route = parsed["route"] if parsed["route"] in ("classic", "graph", "agentic") else "classic"
            confidence = parsed["confidence"]

            # 低置信度 → fallback classic (default safe)
            if confidence == "low" and route != "classic":
                logger.info(f"[classifier] low confidence, fallback classic "
                            f"(was {route}) | {query[:60]}")
                route = "classic"
        except Exception as e:
            logger.error(f"[classifier] failed: {e} → fallback classic")
            route = "classic"
            confidence = "low"

        self._classify_cache[query] = {"route": route, "confidence": confidence}
        logger.info(f"[router] {route} (conf={confidence}) ← {query[:60]}",
                    extra={"route": route, "confidence": confidence})
        return {"route": route, "confidence": confidence}

    def _node_classic(self, state: RouterState, config: Dict = None) -> RouterState:
        q = state.get("rewritten_query") or state["query"]
        try:
            ans = self.classic.run(
                q,
                session_id=state.get("session_id"),
                parent_config=self._scoped(config, "classic_path"),
            )
            self.route_counts["classic"] += 1
        except Exception as e:
            logger.error(f"[classic] failed: {e}")
            self.route_counts["error"] += 1
            ans = f"Classic path failed: {e}"
        return {"answer": ans}

    def _node_graph(self, state: RouterState, config: Dict = None) -> RouterState:
        q = state.get("rewritten_query") or state["query"]
        try:
            ans = self.graph.run(
                q,
                session_id=state.get("session_id"),
                parent_config=self._scoped(config, "graph_path"),
            )
            self.route_counts["graph"] += 1
        except Exception as e:
            logger.error(f"[graph] failed: {e}")
            self.route_counts["error"] += 1
            ans = f"Graph path failed: {e}"
        return {"answer": ans}

    def _node_agentic(self, state: RouterState, config: Dict = None) -> RouterState:
        q = state.get("rewritten_query") or state["query"]
        try:
            ans = self.agentic.run(
                q,
                session_id=state.get("session_id"),
                parent_config=self._scoped(config, "agentic_path"),
            )
            self.route_counts["agentic"] += 1
        except Exception as e:
            logger.error(f"[agentic] failed: {e}")
            self.route_counts["error"] += 1
            ans = f"Agentic path failed: {e}"
        return {"answer": ans}

    # ==================== Verify & Escalate ====================

    @staticmethod
    def _parse_verifier(raw: str) -> Dict:
        """解析 verifier JSON 输出。失败就当 ok（保守策略——不确定就不升级）。"""
        import json as _json
        import re as _re
        raw = (raw or "").strip()
        m = _re.search(r"\{.*?\}", raw, _re.DOTALL)
        if not m:
            return {"ok": True, "reason": "verifier parse failed, default pass"}
        try:
            obj = _json.loads(m.group(0))
            return {
                "ok": bool(obj.get("ok", True)),
                "reason": str(obj.get("reason", ""))[:80],
            }
        except _json.JSONDecodeError:
            return {"ok": True, "reason": "verifier json decode failed, default pass"}

    def _node_verify(self, state: RouterState, config: Dict = None) -> RouterState:
        """
        对 classic/graph 的答案做质量检查。
        两级筛：
          Level 1（免费）：命中 hedge 关键词直接 fail
          Level 2（1 次 LLM call）：让 LLM 判"答到点上没"
        Agentic 不走这里——它是最强 path，再失败也没地方升级。
        """
        query = state.get("rewritten_query") or state["query"]
        answer = state.get("answer") or ""

        # Level 1: 纯字符串短路
        low = answer.lower()
        if any(m in low for m in _HEDGE_MARKERS):
            self.verify_counts["fail_hedge"] += 1
            logger.info(f"[verify] fail (hedge marker) | {query[:60]}")
            return {"verify_ok": False, "verify_reason": "hedge marker detected"}

        # 极短答案（< 30 chars 且不含数字）也可疑
        if len(answer.strip()) < 30 and not any(c.isdigit() for c in answer):
            self.verify_counts["fail_hedge"] += 1
            logger.info(f"[verify] fail (too short) | ans={answer[:40]}")
            return {"verify_ok": False, "verify_reason": "answer too short"}

        # Level 2: LLM 判
        # 答案太长就截——判"质量"只看前 1500 字符足够，完整上下文会让 LLM 注意力涣散
        answer_excerpt = answer[:1500]
        try:
            raw = retry_sync(max_retries=2, base_delay=0.5)(self.verifier_chain.invoke)(
                {"query": query, "answer": answer_excerpt},
                config=self._scoped(config, "verifier"),
            )
            parsed = self._parse_verifier(raw)
            if parsed["ok"]:
                self.verify_counts["pass"] += 1
                logger.info(f"[verify] pass | {query[:60]}")
            else:
                self.verify_counts["fail_llm"] += 1
                logger.info(f"[verify] fail (llm) reason='{parsed['reason']}' | {query[:60]}")
            return {"verify_ok": parsed["ok"], "verify_reason": parsed["reason"]}
        except Exception as e:
            # Verifier 挂了走保守策略：默认 pass，不无谓升级
            logger.warning(f"[verify] failed: {e} → default pass")
            self.verify_counts["pass"] += 1
            return {"verify_ok": True, "verify_reason": f"verifier error: {e}"}

    def _node_escalate(self, state: RouterState, config: Dict = None) -> RouterState:
        """
        Verify 失败 → 升级到 agentic path 重跑。
        Agentic 内部会做 planner → multi-round ReAct → aggregate，大概率能兜住
        classic/graph 漏答的 case。代价：~3-5x 成本，只在 ~10-15% 被触发。
        """
        q = state.get("rewritten_query") or state["query"]
        prev_route = state.get("route", "?")
        prev_reason = state.get("verify_reason", "")
        logger.info(
            f"[escalate] {prev_route} → agentic | reason='{prev_reason}' | {q[:60]}"
        )
        self.verify_counts["escalated"] += 1
        try:
            ans = self.agentic.run(
                q,
                session_id=state.get("session_id"),
                parent_config=self._scoped(config, "agentic_escalate"),
            )
            # 注意：不再覆盖 route 为 agentic——保留原 route 便于 analytics 看"谁被升级了"
            # 但标记 escalated=True，前端 / benchmark 可以据此区分
            return {"answer": ans, "escalated": True}
        except Exception as e:
            logger.error(f"[escalate] failed: {e}")
            # 升级也挂了就把原答案保留 + 加个错误标记
            orig = state.get("answer", "")
            return {
                "answer": orig + f"\n\n[Note: escalation also failed: {e}]",
                "escalated": True,
            }

    # ==================== Routing ====================

    def _route_after_classify(self, state: RouterState) -> str:
        return state.get("route", "classic")

    def _route_after_verify(self, state: RouterState) -> str:
        """verify 通过 → END；失败 → escalate"""
        return "ok" if state.get("verify_ok", True) else "fail"

    # ==================== Build ====================

    def _build_graph(self):
        g = StateGraph(RouterState)
        g.add_node("rewrite", self._node_rewrite)
        g.add_node("classify", self._node_classify)
        g.add_node("classic_path", self._node_classic)
        g.add_node("graph_path", self._node_graph)
        g.add_node("agentic_path", self._node_agentic)
        g.add_node("verify", self._node_verify)
        g.add_node("escalate", self._node_escalate)

        g.set_entry_point("rewrite")
        g.add_edge("rewrite", "classify")
        g.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {"classic": "classic_path", "graph": "graph_path", "agentic": "agentic_path"},
        )
        # classic / graph 都走 verify；verify 结果决定 END 还是 escalate
        g.add_edge("classic_path", "verify")
        g.add_edge("graph_path", "verify")
        g.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {"ok": END, "fail": "escalate"},
        )
        g.add_edge("escalate", END)
        # Agentic 本身不走 verify——它是最强 path，再错也没地方升级了
        g.add_edge("agentic_path", END)
        return g.compile()

    # ==================== 对外接口 ====================

    def ask(self, query: str, skip_cache: bool = False, **kwargs) -> str:
        overall_start = time.time()
        session_id = kwargs.get("session_id") or f"router_{int(overall_start)}_{uuid.uuid4().hex[:6]}"

        config = {}
        handler = get_langfuse_handler()
        if handler is not None:
            config = {
                "callbacks": [handler],
                "run_name": "router_agent",
                "metadata": {
                    "langfuse_session_id": session_id,
                    "langfuse_tags": ["router", "3way_dispatch"],
                    "query": query[:100],
                },
            }

        initial_state: RouterState = {"query": query, "session_id": session_id}

        try:
            final_state = self._graph.invoke(initial_state, config=config)
            answer = final_state.get("answer", "No answer produced")
            elapsed = round(time.time() - overall_start, 2)
            route = final_state.get("route")
            logger.info(
                f"[router] done route={route} ({elapsed}s)",
                extra={"route": route, "elapsed": elapsed},
            )

            # 记录本轮到 session history（下一轮 rewriter 消费）
            # 存原始 query 而不是改写后的——rewriter 的示例里 "History: - user: ..." 也是原 query 风格
            self._append_history(session_id, query, answer)

            # 把 route 写进 Langfuse trace tag
            # 这样 Analytics 前端只需 1 次 fetch_traces API 就能拿到 route 分布，
            # 不用逐 trace 调 fetch_observations（那个慢 10-100x 且易 timeout）
            if handler is not None and route:
                try:
                    trace_id = getattr(handler, "trace_id", None) \
                               or getattr(handler, "last_trace_id", None)
                    if trace_id:
                        from graphrag_agent.utils.langfuse_client import get_langfuse_client
                        lf = get_langfuse_client()
                        if lf is not None:
                            lf.trace(id=trace_id, tags=["router", "3way_dispatch", f"route:{route}"])
                except Exception as _e:
                    logger.debug(f"[router] failed to update trace tag: {_e}")

            return answer
        except Exception as e:
            logger.error(f"[router] graph failed: {e}")
            return f"Router failed: {e}"

    # ==================== Debug 接口 (前端 debug panel 消费) ====================

    def ask_with_trace(self, query: str, thread_id: str = None, **kwargs) -> Dict:
        """
        带结构化 trace 的 ask，供前端 debug 面板用。
        返回:
          {
            "answer":         "...",
            "execution_log":  [{"node":..., "input":..., "output":..., "latency":...}],
            "kg_data":        {"nodes": [...], "links": [...]},
            "reference":      {"Chunks": [...]},
          }
        """
        import re as _re
        session_id = thread_id or f"router_debug_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        trace_steps = []

        # Step 0: rewrite (history-aware)
        t_rw = time.time()
        rewrite_result = self._node_rewrite({"query": query, "session_id": session_id})
        q_effective = rewrite_result.get("rewritten_query", query)
        trace_steps.append({
            "node": "rewriter",
            "input": query[:120],
            "output": q_effective[:120] if q_effective != query else "(unchanged, no history)",
            "latency": round(time.time() - t_rw, 2),
        })

        # Step 1: classifier (吃改写后的 query)
        t0 = time.time()
        classify_result = self._node_classify({"query": query, "rewritten_query": q_effective})
        route = classify_result.get("route", "classic")
        trace_steps.append({
            "node": "classifier",
            "input": q_effective[:120],
            "output": f"route={route}  confidence={classify_result.get('confidence', '?')}",
            "latency": round(time.time() - t0, 2),
        })

        # Step 2: execute path (吃改写后的 query)
        t_path = time.time()
        answer = ""
        try:
            if route == "classic":
                answer = self.classic.run(q_effective, session_id=session_id)
                trace_steps.append({
                    "node": "classic_path",
                    "input": q_effective[:80],
                    "output": "hybrid (vector+BM25+RRF) → generate answer with citations",
                    "latency": round(time.time() - t_path, 2),
                })
                self.route_counts["classic"] += 1
            elif route == "graph":
                answer = self.graph.run(q_effective, session_id=session_id)
                trace_steps.append({
                    "node": "graph_path",
                    "input": q_effective[:80],
                    "output": "detect_mode → extract_entities → graph_lookup + community + hybrid chunks → compose",
                    "latency": round(time.time() - t_path, 2),
                })
                self.route_counts["graph"] += 1
            else:  # agentic
                answer = self.agentic.run(q_effective, session_id=session_id)
                trace_steps.append({
                    "node": "agentic_path",
                    "input": q_effective[:80],
                    "output": "planner → execute (ReAct w/ classic/graph sub-tools) → aggregate",
                    "latency": round(time.time() - t_path, 2),
                })
                self.route_counts["agentic"] += 1
        except Exception as e:
            logger.error(f"[ask_with_trace] path {route} failed: {e}")
            answer = f"Router path failed: {e}"

        # Step 2.5: verify + escalate（仅对 classic/graph 生效，agentic 跳过）
        if route in ("classic", "graph") and not answer.startswith("Router path failed"):
            t_v = time.time()
            v_result = self._node_verify({
                "query": query, "rewritten_query": q_effective, "answer": answer,
            })
            v_ok = v_result.get("verify_ok", True)
            trace_steps.append({
                "node": "verify",
                "input": f"route={route}, answer_len={len(answer)}",
                "output": f"ok={v_ok} reason={v_result.get('verify_reason', '')}",
                "latency": round(time.time() - t_v, 2),
            })
            if not v_ok:
                t_e = time.time()
                escalate_result = self._node_escalate({
                    "query": query, "rewritten_query": q_effective,
                    "answer": answer, "route": route,
                    "verify_reason": v_result.get("verify_reason", ""),
                    "session_id": session_id,
                })
                answer = escalate_result.get("answer", answer)
                trace_steps.append({
                    "node": "escalate",
                    "input": f"{route} failed verify",
                    "output": "re-ran via agentic path",
                    "latency": round(time.time() - t_e, 2),
                })

        # Step 3: 构建 KG 可视化数据（从改写后的 query 抽实体——这样指代已解析）
        kg_data = self._build_kg_for_query(q_effective)

        # Step 4: 从 answer 里解析 References（Classic / Graph 返回的答案带 ### References 块）
        reference = self._parse_references(answer)

        # 记录到 session history，debug 面板多轮也能享受 rewriter
        self._append_history(session_id, query, answer)

        return {
            "answer": answer,
            "execution_log": trace_steps,
            "kg_data": kg_data,
            "reference": reference,
        }

    def _build_kg_for_query(self, query: str, max_entities: int = 3,
                            max_neighbors_per: int = 6) -> Dict:
        """
        为前端 KG 可视化构造当次查询的实体子图。
        复用 graph_path 的 entity_extract_chain + graph_lookup 原语。
        """
        try:
            raw = self.graph.entity_extract_chain.invoke({"query": query})
            entities = [e.strip() for e in raw.split(",") if e.strip()][:max_entities]
        except Exception as e:
            logger.debug(f"[kg_data] entity extract failed: {e}")
            return {"nodes": [], "links": []}

        if not entities:
            return {"nodes": [], "links": []}

        from graphrag_agent.search.tool.primitives import _get_graph
        g = _get_graph()

        nodes_by_id: Dict[str, Dict] = {}
        links = []
        for ent in entities:
            try:
                rows = g.query(
                    """
                    MATCH (e:__Entity__)
                    WHERE toLower(e.id) = toLower($name)
                       OR toLower(e.id) CONTAINS toLower($name)
                    WITH e LIMIT 1
                    MATCH (e)-[r]-(n:__Entity__)
                    WHERE type(r) <> 'MENTIONS' AND type(r) <> 'IN_COMMUNITY'
                    RETURN e.id AS center,
                           [l IN labels(e) WHERE l <> '__Entity__'][0] AS center_type,
                           type(r) AS rel,
                           n.id AS neighbor,
                           [l IN labels(n) WHERE l <> '__Entity__'][0] AS neighbor_type
                    LIMIT $limit
                    """,
                    {"name": ent, "limit": max_neighbors_per},
                )
            except Exception as _e:
                logger.debug(f"[kg_data] lookup({ent}) failed: {_e}")
                continue

            for row in rows:
                c_id = row["center"]
                n_id = row["neighbor"]
                if c_id not in nodes_by_id:
                    nodes_by_id[c_id] = {
                        "id": c_id, "label": c_id,
                        "group": row["center_type"] or "Entity",
                        "is_query_entity": True,
                    }
                if n_id not in nodes_by_id:
                    nodes_by_id[n_id] = {
                        "id": n_id, "label": n_id,
                        "group": row["neighbor_type"] or "Entity",
                        "is_query_entity": False,
                    }
                links.append({
                    "source": c_id,
                    "target": n_id,
                    "label": row["rel"],
                    "relation": row["rel"],
                })

        return {"nodes": list(nodes_by_id.values()), "links": links}

    @staticmethod
    def _parse_references(answer: str) -> Dict:
        """从 NaiveSearchTool / GraphPath 答案末尾的 References 块抽 chunk id"""
        import re as _re
        m = _re.search(r"'Chunks'\s*:\s*\[([^\]]*)\]", answer or "")
        if not m:
            return {}
        ids = _re.findall(r"'([a-f0-9]{8,})'", m.group(1))
        return {"Chunks": ids}

    def check_fast_cache(self, message: str, session_id: str):
        """chat_service 需要这个接口；Router 不做特殊 fast cache（直接返回 None 走完整流程）"""
        return None

    def get_route_stats(self):
        return {
            "routes": dict(self.route_counts),
            "verify": dict(self.verify_counts),
        }

    def visualize(self) -> str:
        try:
            return self._graph.get_graph().draw_mermaid()
        except Exception:
            return "(mermaid dump failed)"

    def close(self):
        for t in (self.classic, self.graph):
            if hasattr(t, "close"):
                try:
                    t.close()
                except Exception:
                    pass
