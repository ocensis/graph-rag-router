"""
Text2Cypher Tool —— 跨文档枚举/聚合查询的终极补丁

为什么存在：Naive/Graph/Agentic/Global 都无法处理 "哪些论文用了 X" 这类枚举查询，
因为 top-K chunk retrieval 架构天生无法全库枚举。

解法：LLM 把自然语言 query 翻译成 Cypher，直接在图上查询，再用 LLM 把结果翻成自然语言。

安全：只允许 MATCH / RETURN，禁用 CREATE / DELETE / MERGE / SET / DROP。
"""
import re
import time
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END

from graphrag_agent.search.tool.base import BaseSearchTool
from graphrag_agent.utils.logger import get_logger
from graphrag_agent.utils.resilience import retry_sync
from graphrag_agent.utils.langfuse_client import get_langfuse_handler

logger = get_logger(__name__)


class Text2CypherState(TypedDict, total=False):
    """Text2Cypher LangGraph 状态"""
    query: str
    cypher: Optional[str]        # LLM 生成的 Cypher；CANNOT_CYPHER 表示不适合
    results: List[Dict]          # Neo4j 执行结果
    answer: str


# ==================== Schema ====================

SCHEMA_PROMPT = """You are a Neo4j Cypher expert. Given a natural language question about a knowledge graph of academic RAG papers, generate a read-only Cypher query.

## Graph Schema

**Nodes:**
- `__Document__` (properties: fileName, type, uri, domain) — represents a paper PDF; fileName is like "2604_09541v1__Trans-RAG_Query-Centric..._Re.pdf"
- `__Chunk__` (properties: id, fileName, text, position, tokens) — a text chunk of a paper
- `__Entity__` (properties: id, description) — an extracted entity. HAS secondary labels like:
  - Method, Model, Dataset, Metric, Task, Component, Author, Institution, Paper
  - (entity might also have specific labels like Algorithm, Technology, etc.)

**Key Relationships:**
- `(Chunk)-[:PART_OF]->(Document)` — chunk belongs to a paper
- `(Chunk)-[:MENTIONS]->(Entity)` — chunk mentions an entity
- `(Document)-[:FIRST_CHUNK]->(Chunk)` — first chunk pointer (not useful for most queries)
- `(Entity)-[:USES|:PROPOSES|:EVALUATED_ON|:COMPARES_WITH|:EXTENDS|:OUTPERFORMS|:ADDRESSES|:OTHER]-(Entity)` — semantic entity relationships

## Rules
1. Use **case-insensitive matching**: `toLower(e.id) CONTAINS toLower('xxx')` or `toLower(e.description) CONTAINS toLower('xxx')`
2. Don't match just the entity itself — trace back to `__Document__` via `MENTIONS` + `PART_OF` to get paper-level answers
3. Return **DISTINCT d.fileName** when the answer is "which papers"
4. Use **LIMIT** (usually 20) to keep results bounded
5. Only MATCH / RETURN / WHERE / WITH / LIMIT / ORDER BY — no CREATE / DELETE / SET / MERGE / DROP
6. If the question can't be answered with Cypher (e.g., conceptual/comparative), return literal "CANNOT_CYPHER"

## Examples

Q: "Which papers use GRPO?"
Cypher:
```
MATCH (d:__Document__)<-[:PART_OF]-(c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
WHERE toLower(e.id) CONTAINS 'grpo' OR toLower(e.description) CONTAINS 'grpo'
RETURN DISTINCT d.fileName
LIMIT 20
```

Q: "哪些论文评测了 MMLongbench-Doc？"
Cypher:
```
MATCH (d:__Document__)<-[:PART_OF]-(c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
WHERE toLower(e.id) CONTAINS 'mmlongbench' OR toLower(e.description) CONTAINS 'mmlongbench'
RETURN DISTINCT d.fileName
LIMIT 20
```

Q: "How many papers address security vulnerabilities?"
Cypher:
```
MATCH (d:__Document__)<-[:PART_OF]-(c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
WHERE toLower(e.id) CONTAINS 'security' OR toLower(e.description) CONTAINS 'adversarial attack'
   OR toLower(e.description) CONTAINS 'poisoning'
RETURN DISTINCT d.fileName, count(DISTINCT e) AS entity_count
ORDER BY entity_count DESC
LIMIT 20
```

## Question

{question}

## Output

Return ONLY the Cypher query in a code block (```cypher ... ```), nothing else.
If can't be answered with Cypher, return: CANNOT_CYPHER"""


ANSWER_PROMPT = """You are answering a question about a knowledge graph of RAG papers using Cypher query results.

**IMPORTANT: Respond in English.**

Question: {question}

Cypher query:
```
{cypher}
```

Query results:
{results}

## Rules
1. Base your answer ONLY on the query results above. Do NOT add outside knowledge.
2. If results are empty, respond: "No relevant papers found in the current knowledge graph."
3. **Start your answer directly with the key fact** (e.g., "N papers use X: ...").
4. When listing papers, **keep the arxiv id** at the start (e.g., "2604_09508v1: VISOR ..."), DO NOT remove it.
5. Remove only the `.pdf` extension and replace underscores with spaces for readability.
6. Be concise and direct. Plain paragraphs, minimal markdown.

Answer:"""


class Text2CypherTool(BaseSearchTool):
    """LLM 生成 Cypher 查询 → 执行 → LLM 渲染答案"""

    # 危险关键词黑名单
    FORBIDDEN = ["CREATE", "DELETE", "MERGE", "SET ", "DROP", "REMOVE"]

    def __init__(self, max_rows: int = 50):
        super().__init__(cache_dir="./cache/text2cypher")
        self.max_rows = max_rows
        self._setup_chains()
        # 构建 LangGraph StateGraph: generate_cypher → execute_cypher → format_answer
        self._graph = self._build_graph()

    def _setup_chains(self):
        cypher_prompt = ChatPromptTemplate.from_messages([("human", SCHEMA_PROMPT)])
        self.cypher_chain = cypher_prompt | self.llm | StrOutputParser()
        answer_prompt = ChatPromptTemplate.from_messages([("human", ANSWER_PROMPT)])
        self.answer_chain = answer_prompt | self.llm | StrOutputParser()

    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        return {"low_level": [], "high_level": []}

    def _extract_cypher(self, llm_output: str) -> Optional[str]:
        if "CANNOT_CYPHER" in llm_output:
            return None
        m = re.search(r"```(?:cypher)?\s*(.*?)```", llm_output, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return llm_output.strip()

    def _is_safe(self, cypher: str) -> bool:
        upper = cypher.upper()
        return not any(kw in upper for kw in self.FORBIDDEN)

    # ==================== Graph Nodes ====================

    def _node_generate_cypher(self, state: Text2CypherState) -> Text2CypherState:
        """Node: LLM 把自然语言 query 翻译成 Cypher"""
        query = state["query"]
        try:
            llm_output = retry_sync(max_retries=2, base_delay=1.0)(
                self.cypher_chain.invoke
            )({"question": query})
            cypher = self._extract_cypher(llm_output)
            if cypher and not self._is_safe(cypher):
                logger.warning("[t2c] Cypher 包含危险操作", extra={"cypher": cypher[:100]})
                cypher = None
            if cypher:
                logger.info(f"[t2c] 生成 Cypher: {cypher[:150]}")
            return {"cypher": cypher}
        except Exception as e:
            logger.error(f"[generate_cypher] 失败: {e}")
            return {"cypher": None}

    def _node_execute_cypher(self, state: Text2CypherState) -> Text2CypherState:
        """Node: 执行 Cypher 查询图"""
        cypher = state.get("cypher")
        if not cypher:
            return {"results": []}
        try:
            results = self.graph.query(cypher)
            logger.info(f"[t2c] 返回 {len(results)} 行")
            if len(results) > self.max_rows:
                results = results[:self.max_rows]
            return {"results": results}
        except Exception as e:
            logger.warning(f"[t2c] Cypher 执行失败: {e}", extra={"cypher": cypher[:200]})
            return {"results": []}

    def _node_format_answer(self, state: Text2CypherState) -> Text2CypherState:
        """Node: LLM 把查询结果渲染成自然语言"""
        query = state["query"]
        cypher = state.get("cypher")
        results = state.get("results", [])

        if cypher is None:
            return {"answer": "此问题不适合用 Cypher 查询（需要概念性/推理性回答）。"}

        results_str = "\n".join(str(r) for r in results[:30]) if results else "(empty result)"
        try:
            answer = retry_sync(max_retries=2, base_delay=1.0)(
                self.answer_chain.invoke
            )({"question": query, "cypher": cypher, "results": results_str})
            return {"answer": answer}
        except Exception as e:
            logger.error(f"[format_answer] 失败: {e}")
            return {"answer": f"渲染答案失败: {e}"}

    def _build_graph(self):
        g = StateGraph(Text2CypherState)
        g.add_node("generate_cypher", self._node_generate_cypher)
        g.add_node("execute_cypher", self._node_execute_cypher)
        g.add_node("format_answer", self._node_format_answer)

        g.set_entry_point("generate_cypher")
        g.add_edge("generate_cypher", "execute_cypher")
        g.add_edge("execute_cypher", "format_answer")
        g.add_edge("format_answer", END)

        return g.compile()

    # ==================== 对外接口 ====================

    def search(self, query_input: Any, session_id: Optional[str] = None,
               parent_config: Optional[Dict] = None) -> str:
        overall_start = time.time()
        query = query_input["query"] if isinstance(query_input, dict) else str(query_input)

        cache_key = f"t2c:{query}"
        cached = self.cache_manager.get(cache_key)
        if cached:
            return cached

        # Langfuse config（parent 优先）
        if parent_config:
            lf_config = parent_config
        else:
            lf_config = {}
            handler = get_langfuse_handler()
            if handler is not None:
                lf_config = {
                    "callbacks": [handler],
                    "run_name": "text2cypher_tool",
                    "metadata": {
                        "langfuse_session_id": session_id or "default",
                        "langfuse_tags": ["text2cypher"],
                        "query": query[:100],
                    },
                }

        initial_state: Text2CypherState = {"query": query}
        try:
            final_state = self._graph.invoke(initial_state, config=lf_config if lf_config else None)
            answer = final_state.get("answer", "T2C 无结果")
            self.cache_manager.set(cache_key, answer)
            total = time.time() - overall_start
            logger.info(f"[t2c] 完成 ({total:.1f}s)")
            return answer
        except Exception as e:
            logger.error(f"[t2c] graph 失败: {e}", extra={"error": str(e)})
            return f"Text2Cypher 出错: {e}"

    def get_tool(self) -> BaseTool:
        class T2CRetrievalTool(BaseTool):
            name: str = "text2cypher"
            description: str = "把自然语言问题翻译成 Cypher 查询，适合枚举/聚合类问题"

            def _run(self_tool, query: Any) -> str:
                return self.search(query)

            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError

        return T2CRetrievalTool()
