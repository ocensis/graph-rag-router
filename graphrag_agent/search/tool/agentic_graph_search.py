"""
Agentic Graph Search —— 基于 LangGraph StateGraph 的可观测多轮迭代检索

和 enhanced_graph_search.py 的区别：
1. 用 StateGraph 代替 for 循环，每个节点职责单一，路由逻辑集中在 conditional_edges
2. 集成 Langfuse CallbackHandler，每个 LLM 调用自动上报 trace
3. 状态通过 State TypedDict 显式传递，而不是局部变量

节点流转：
    START → retrieve → accumulate → check_sufficiency
                                      ├─ 足够/达上限 → generate → END
                                      └─ 不够 → rewrite → retrieve (loop)
"""
import time
import threading
from typing import TypedDict, List, Dict, Any, Set, Optional

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool

from graphrag_agent.config.prompts import LC_SYSTEM_PROMPT, LOCAL_SEARCH_CONTEXT_PROMPT
from graphrag_agent.config.settings import response_type, lc_description, LOCAL_SEARCH_SETTINGS
from graphrag_agent.search.tool.base import BaseSearchTool
from graphrag_agent.search.utils import VectorUtils
from graphrag_agent.search.tool.enhanced_graph_search import EnhancedGraphSearchTool
from graphrag_agent.utils.logger import get_logger
from graphrag_agent.utils.resilience import retry_sync
from graphrag_agent.utils.llm_output_schemas import SufficiencyCheck
from graphrag_agent.utils.langfuse_client import get_langfuse_handler, flush_langfuse

logger = get_logger(__name__)


# ==================== State ====================

class AgenticState(TypedDict, total=False):
    """
    LangGraph 图状态，贯穿所有节点

    注意: 为了避免 Langfuse 1MB/event 限制触发截断，state 里不再直接存 chunk 全文。
    完整 chunks 存在 thread-local 的 _chunk_store 里，state 只携带轻量 id 引用。
    """
    original_query: str              # 用户原始问题
    current_query: str               # 本轮检索用的 query（可能是改写后的）
    round_num: int                   # 当前轮次 (0-based)
    max_rounds: int                  # 最大轮次限制

    # 检索结果累积（轻量版）
    used_chunk_ids: List[str]        # 去重用 + 提供给 generate 节点取回 full chunks
    info_summary: str                # 本字段也会被截断（只保留最近 ~4000 字符）

    # 路由决策
    sufficient: bool
    subquery: str                    # rewrite 节点产出

    # 最终结果
    final_answer: str


# ==================== Thread-local chunk store ====================
# LangGraph 把整个 state 序列化给 Langfuse，大对象会超过 1MB 限制。
# 解决方案：chunk 全文放 thread-local，state 只保留 id 引用。
# 为什么用 thread-local？benchmark 用 ThreadPoolExecutor 多线程并发跑，
# 不同线程的 chunk store 互相隔离，避免串扰。
_thread_local = threading.local()


def _get_chunk_store() -> Dict[str, Dict]:
    if not hasattr(_thread_local, "store"):
        _thread_local.store = {}
    return _thread_local.store


def _reset_chunk_store() -> None:
    _thread_local.store = {}


# ==================== Tool ====================

class AgenticGraphSearchTool(BaseSearchTool):
    """用 LangGraph 重写的 agentic 检索工具"""

    # 充分性判断 prompt
    _SUFFICIENCY_PROMPT = """Based on the information collected so far, determine if we can answer the question.

Question: {query}

Information collected:
{info_summary}

Respond in this exact format:
SUFFICIENT: yes or no
MISSING: what information is still needed (if not sufficient)
SUBQUERY: a search query to find the missing information (if not sufficient)"""

    def __init__(self, max_rounds: int = 3, top_k: int = 5, top_entities: int = 10):
        super().__init__(cache_dir="./cache/agentic_graph_search")
        self.max_rounds = max_rounds
        self.top_k = top_k
        self.top_entities = LOCAL_SEARCH_SETTINGS.get("top_entities", top_entities)

        # 复用 enhanced_graph_search 的检索 + 混合排序逻辑
        self._helper = EnhancedGraphSearchTool()
        self._helper.top_k = top_k
        self._helper.top_entities = self.top_entities

        self._setup_chains()
        self._graph = self._build_graph()

    def _setup_chains(self):
        """生成答案的 chain"""
        self.query_prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", LOCAL_SEARCH_CONTEXT_PROMPT),
        ])
        self.query_chain = self.query_prompt | self.llm | StrOutputParser()

    # ==================== 节点实现 ====================

    # 单个节点 input/output 的最大字符数（给 Langfuse 留 buffer，避免 1MB 限制）
    _MAX_SUMMARY_CHARS = 4000

    def _node_retrieve(self, state: AgenticState) -> AgenticState:
        """
        节点：用 current_query 召回 chunks 并累积到 thread-local store
        （chunk 全文不进 state，避免 Langfuse 超过 1MB 被截断）
        """
        query = state["current_query"]
        round_num = state["round_num"]
        used_ids = set(state.get("used_chunk_ids", []))
        info_summary = state.get("info_summary", "")
        chunk_store = _get_chunk_store()

        logger.info(
            f"[retrieve] Round {round_num}",
            extra={"round": round_num, "query": query[:60], "component": "agentic"}
        )

        try:
            query_embedding = self.embeddings.embed_query(query)
            new_chunks = self._helper._retrieve_chunks(query, query_embedding, used_ids)
        except Exception as e:
            logger.warning(f"[retrieve] 失败: {e}", extra={"error": str(e), "component": "agentic"})
            new_chunks = []

        added_texts = []
        for c in new_chunks:
            if c["id"] not in used_ids:
                used_ids.add(c["id"])
                chunk_store[c["id"]] = c          # 完整 chunk 存 thread-local
                added_texts.append(c.get("text", ""))

        if added_texts:
            new_section = f"\n--- Round {round_num + 1} ---\n" + "\n".join(added_texts)
            info_summary += new_section
            # 截断：只保留最近 ~4000 字符（足够 LLM 判断充分性）
            if len(info_summary) > self._MAX_SUMMARY_CHARS:
                info_summary = "...[earlier truncated]...\n" + info_summary[-self._MAX_SUMMARY_CHARS:]

        logger.info(
            f"[retrieve] 本轮新增 {len(added_texts)}，累计 {len(used_ids)} ids",
            extra={"round": round_num, "num_results": len(added_texts), "component": "agentic"}
        )

        return {
            "used_chunk_ids": list(used_ids),
            "info_summary": info_summary,
        }

    def _node_check_sufficiency(self, state: AgenticState) -> AgenticState:
        """节点：LLM 判断信息是否足够回答问题（Pydantic 校验输出）"""
        query = state["original_query"]
        info_summary = state["info_summary"]
        round_num = state["round_num"]

        # 没有检索到任何内容 → 不够，但也没必要继续问 LLM
        if not state.get("used_chunk_ids"):
            logger.info("[check_sufficiency] 无内容，直接判定不够",
                        extra={"round": round_num, "component": "agentic"})
            return {"sufficient": False, "subquery": ""}

        prompt = self._SUFFICIENCY_PROMPT.format(query=query, info_summary=info_summary)

        try:
            result = retry_sync(max_retries=2, base_delay=1.0)(self.llm.invoke)(prompt)
            content = result.content if hasattr(result, 'content') else str(result)
            parsed = SufficiencyCheck.parse_llm_output(content)
            logger.info(
                f"[check_sufficiency] sufficient={parsed.sufficient}",
                extra={"round": round_num, "component": "agentic"}
            )
            return {"sufficient": parsed.sufficient, "subquery": parsed.subquery}
        except Exception as e:
            logger.warning(f"[check_sufficiency] 失败，默认足够: {e}",
                           extra={"error": str(e), "component": "agentic"})
            return {"sufficient": True, "subquery": ""}

    def _node_rewrite(self, state: AgenticState) -> AgenticState:
        """节点：将 subquery 设为下一轮的 current_query，round_num +1"""
        subquery = state.get("subquery", "") or state["original_query"]
        round_num = state["round_num"] + 1

        logger.info(
            f"[rewrite] 进入 Round {round_num}，subquery={subquery[:60]}",
            extra={"round": round_num, "component": "agentic"}
        )

        return {
            "current_query": subquery,
            "round_num": round_num,
        }

    def _node_generate(self, state: AgenticState) -> AgenticState:
        """节点：从 thread-local store 取回 full chunks 并生成最终回答"""
        query = state["original_query"]
        used_ids = state.get("used_chunk_ids", [])
        round_num = state["round_num"]

        # 从 thread-local 取回完整 chunks
        chunk_store = _get_chunk_store()
        chunks = [chunk_store[cid] for cid in used_ids if cid in chunk_store]

        if not chunks:
            logger.info("[generate] 无 chunks，返回默认回答",
                        extra={"component": "agentic"})
            return {"final_answer": "未找到相关信息。"}

        # 获取补充的实体关系
        try:
            final_embedding = self.embeddings.embed_query(query)
            scored_entities = self.graph.query(
                """CALL db.index.vector.queryNodes('vector', $k, $vec)
                YIELD node, score
                RETURN node.id AS id, node.description AS desc, score""",
                {"k": 5, "vec": final_embedding}
            )
        except Exception:
            scored_entities = []

        context = self._helper._build_context(query, scored_entities, chunks)
        try:
            answer = retry_sync(max_retries=2, base_delay=1.0)(self.query_chain.invoke)({
                "context": context,
                "input": query,
                "response_type": response_type,
            })
        except Exception as e:
            logger.error(f"[generate] 失败: {e}", extra={"error": str(e)})
            answer = f"生成答案时出错: {e}"

        logger.info(
            f"[generate] 回答长度 {len(answer)}，总轮次 {round_num + 1}",
            extra={"round": round_num, "component": "agentic"}
        )

        return {"final_answer": answer}

    # ==================== 路由 ====================

    def _route_after_check(self, state: AgenticState) -> str:
        """条件路由：check_sufficiency 之后去哪？"""
        if state.get("sufficient"):
            return "generate"
        if state.get("round_num", 0) + 1 >= self.max_rounds:
            # 达到最大轮次，强制生成
            logger.info("[route] 达到最大轮次，强制生成", extra={"component": "agentic"})
            return "generate"
        if not state.get("subquery"):
            # LLM 没给出 subquery，避免死循环，直接生成
            logger.info("[route] 无 subquery，直接生成", extra={"component": "agentic"})
            return "generate"
        return "rewrite"

    # ==================== 构建图 ====================

    def _build_graph(self):
        g = StateGraph(AgenticState)

        g.add_node("retrieve", self._node_retrieve)
        g.add_node("check_sufficiency", self._node_check_sufficiency)
        g.add_node("rewrite", self._node_rewrite)
        g.add_node("generate", self._node_generate)

        g.set_entry_point("retrieve")

        g.add_edge("retrieve", "check_sufficiency")
        g.add_conditional_edges(
            "check_sufficiency",
            self._route_after_check,
            {"generate": "generate", "rewrite": "rewrite"},
        )
        g.add_edge("rewrite", "retrieve")  # 循环回到检索
        g.add_edge("generate", END)

        return g.compile()

    # ==================== 对外接口 ====================

    def search(self, query_input: Any, session_id: Optional[str] = None,
               parent_config: Optional[Dict] = None) -> str:
        """
        执行 agentic 检索

        参数:
            query_input: 查询字符串或 {"query": ...} 字典
            session_id: 可选，Langfuse 会按 session 聚合同一会话的多次调用
            parent_config: 父 RunnableConfig（来自 Router 等上层），有则复用以嵌套 trace
        """
        overall_start = time.time()

        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
        else:
            query = str(query_input)

        cache_key = f"agentic_graph:{query}"
        cached = self.cache_manager.get(cache_key)
        if cached:
            logger.debug("缓存命中", extra={"query": query[:30], "component": "agentic"})
            return cached

        # 清空当前线程的 chunk store（不同 query 互不干扰）
        _reset_chunk_store()

        initial_state: AgenticState = {
            "original_query": query,
            "current_query": query,
            "round_num": 0,
            "max_rounds": self.max_rounds,
            "used_chunk_ids": [],
            "info_summary": "",
            "sufficient": False,
            "subquery": "",
            "final_answer": "",
        }

        # 构建 Langfuse config（如有 parent_config 则复用，使本次 invoke 嵌套到父 trace）
        if parent_config:
            config = parent_config
        else:
            config = {}
            handler = get_langfuse_handler()
            if handler is not None:
                config = {
                    "callbacks": [handler],
                    "run_name": "agentic_graph_search",
                    "metadata": {
                        "langfuse_session_id": session_id or "default",
                        "langfuse_tags": ["agentic", "graph_rag"],
                        "query": query[:100],
                    },
                }

        try:
            final_state = self._graph.invoke(initial_state, config=config)
            answer = final_state.get("final_answer", "未找到相关信息。")
            self.cache_manager.set(cache_key, answer)

            elapsed = round(time.time() - overall_start, 2)
            logger.info(
                f"agentic 检索完成",
                extra={
                    "elapsed": elapsed,
                    "round": final_state.get("round_num", 0) + 1,
                    "num_results": len(final_state.get("used_chunk_ids", [])),
                    "component": "agentic",
                }
            )
            return answer
        except Exception as e:
            logger.error(f"agentic 检索失败: {e}",
                         extra={"error": str(e), "query": query[:50], "component": "agentic"})
            return f"检索出错: {e}"

    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        return {"low_level": [], "high_level": []}

    def get_tool(self) -> BaseTool:
        class AgenticRetrievalTool(BaseTool):
            name: str = "agentic_graph_retriever"
            description: str = lc_description

            def _run(self_tool, query: Any) -> str:
                return self.search(query)

            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("当前仅提供同步版本")

        return AgenticRetrievalTool()

    def visualize(self) -> str:
        """导出 Mermaid 图（调试用，方便放到 README 或 PPT）"""
        try:
            return self._graph.get_graph().draw_mermaid()
        except Exception:
            return "(mermaid 导出失败，可能缺少依赖)"

    def close(self):
        flush_langfuse()
        super().close()
