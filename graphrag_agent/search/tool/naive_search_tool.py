from typing import List, Dict, Any, TypedDict, Optional
import os
import time
import re
import math
import asyncio
import numpy as np
from collections import Counter

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from graphrag_agent.config.prompts import NAIVE_PROMPT, NAIVE_SEARCH_QUERY_PROMPT
from graphrag_agent.config.settings import response_type, naive_description, NAIVE_SEARCH_TOP_K
from graphrag_agent.search.tool.base import BaseSearchTool
from graphrag_agent.search.utils import VectorUtils
from graphrag_agent.utils.logger import get_logger
from graphrag_agent.utils.resilience import retry_sync, retry_async, async_timeout
from graphrag_agent.utils.llm_output_schemas import RerankerScore
from graphrag_agent.utils.langfuse_client import get_langfuse_handler

logger = get_logger(__name__)


class NaiveState(TypedDict, total=False):
    """Naive 检索 LangGraph 状态"""
    query: str
    query_embedding: List[float]
    vector_candidates: List[Dict]
    bm25_candidates: List[Dict]
    scored_chunks: List[Dict]    # RRF 融合后
    top_chunks: List[Dict]       # 最终 top_k（rerank 后）
    context: str
    answer: str


class SimpleBM25:
    """轻量 BM25 实现，不需要额外依赖"""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def _tokenize(self, text):
        """简单分词：按空格 + 中文字符"""
        # 英文按空格，中文按字
        tokens = re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]', text.lower())
        return tokens

    def score(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """计算 BM25 分数"""
        query_tokens = self._tokenize(query)
        if not query_tokens or not documents:
            return documents

        # 构建文档 token 列表
        doc_tokens_list = []
        for doc in documents:
            text = doc.get("text", "")
            doc_tokens_list.append(self._tokenize(text))

        # 计算平均文档长度
        avg_dl = sum(len(dt) for dt in doc_tokens_list) / len(doc_tokens_list) if doc_tokens_list else 1

        # 计算 IDF
        n_docs = len(documents)
        df = Counter()
        for dt in doc_tokens_list:
            for token in set(dt):
                df[token] += 1

        idf = {}
        for token in query_tokens:
            doc_freq = df.get(token, 0)
            idf[token] = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

        # 计算每个文档的 BM25 分数
        results = []
        for i, doc in enumerate(documents):
            dt = doc_tokens_list[i]
            dl = len(dt)
            tf = Counter(dt)

            score = 0.0
            for token in query_tokens:
                token_tf = tf.get(token, 0)
                numerator = token_tf * (self.k1 + 1)
                denominator = token_tf + self.k1 * (1 - self.b + self.b * dl / avg_dl)
                score += idf.get(token, 0) * numerator / (denominator + 1e-8)

            scored_doc = doc.copy()
            scored_doc["bm25_score"] = score
            results.append(scored_doc)

        return results

RERANK_PROMPT = """请评估以下文本段落与用户问题的相关性。只返回一个0到10的整数分数，不要任何解释。

用户问题：{query}

文本段落：
{chunk_text}

相关性分数（0-10）："""


class NaiveSearchTool(BaseSearchTool):
    """Naive RAG搜索工具，支持 cross-encoder / LLM / 无 三种 reranker 模式"""

    def __init__(self, reranker_mode: str = "cross_encoder"):
        """
        初始化Naive搜索工具

        参数:
            reranker_mode: "cross_encoder" | "llm" | "none"
        """
        # 调用父类构造函数
        super().__init__(cache_dir="./cache/naive_search")

        # 搜索参数设置
        self.top_k = NAIVE_SEARCH_TOP_K
        self.reranker_mode = reranker_mode
        self.rerank_candidates = self.top_k * 3

        # Cross-encoder 模型（延迟加载）
        self._cross_encoder = None

        # 设置处理链
        self._setup_chains()

        # 构建 LangGraph StateGraph（检索流程的可视化状态机）
        self._graph = self._build_graph()

    def _build_graph(self):
        """构建 Naive 检索 StateGraph: embed → recall → fuse → rerank → generate"""
        g = StateGraph(NaiveState)

        g.add_node("embed_query", self._node_embed)
        g.add_node("dual_recall", self._node_dual_recall)
        g.add_node("rrf_fuse", self._node_rrf_fuse)
        g.add_node("rerank", self._node_rerank_node)
        g.add_node("generate_answer", self._node_generate)

        g.set_entry_point("embed_query")
        g.add_edge("embed_query", "dual_recall")
        g.add_edge("dual_recall", "rrf_fuse")
        # conditional: 有 rerank 模式就走 rerank，否则直接 generate
        if self.reranker_mode != "none":
            g.add_edge("rrf_fuse", "rerank")
            g.add_edge("rerank", "generate_answer")
        else:
            g.add_edge("rrf_fuse", "generate_answer")
        g.add_edge("generate_answer", END)

        return g.compile()

    # ==================== Graph Nodes ====================

    def _node_embed(self, state: NaiveState) -> NaiveState:
        """Node: 生成 query embedding"""
        query = state["query"]
        emb = retry_sync(max_retries=2, base_delay=1.0)(
            self.embeddings.embed_query
        )(query)
        return {"query_embedding": emb}

    def _node_dual_recall(self, state: NaiveState) -> NaiveState:
        """Node: 双路独立召回（向量 HNSW + BM25 fulltext）"""
        query = state["query"]
        query_embedding = state["query_embedding"]
        path_recall = int(os.environ.get("NAIVE_RECALL_PER_PATH", "50"))

        # 向量路
        try:
            vector_results = self.graph.query(
                """CALL db.index.vector.queryNodes('chunk_vector', $k, $vec)
                YIELD node, score
                RETURN node.id AS id, coalesce(node.original_text, node.text) AS text, score""",
                {"k": path_recall, "vec": query_embedding}
            )
        except Exception:
            vector_results = self.graph.query("""
                MATCH (c:__Chunk__) WHERE c.embedding IS NOT NULL
                RETURN c.id AS id, coalesce(c.original_text, c.text) AS text, c.embedding AS embedding
                LIMIT 200
            """)
            vector_results = VectorUtils.rank_by_similarity(
                query_embedding, vector_results, "embedding", path_recall
            )

        # BM25 路
        try:
            ft_query = re.sub(r'[+\-!(){}\[\]^"~*?:\\/]', ' ', query)
            bm25_results = self.graph.query(
                """CALL db.index.fulltext.queryNodes('chunk_fulltext', $q)
                YIELD node, score
                RETURN node.id AS id, coalesce(node.original_text, node.text) AS text, score
                LIMIT $k""",
                {"q": ft_query, "k": path_recall}
            )
        except Exception as e:
            logger.warning("fulltext 失败，降级 in-pool BM25", extra={"error": str(e)})
            bm25_results = SimpleBM25().score(query, vector_results)
            bm25_results = sorted(bm25_results, key=lambda x: x.get("bm25_score", 0), reverse=True)

        return {"vector_candidates": vector_results, "bm25_candidates": bm25_results}

    def _node_rrf_fuse(self, state: NaiveState) -> NaiveState:
        """Node: RRF 融合两路候选"""
        vector_results = state["vector_candidates"]
        bm25_results = state["bm25_candidates"]

        merged = {}
        vec_rank_map = {}
        for rank, c in enumerate(vector_results, start=1):
            merged[c["id"]] = {"id": c["id"], "text": c.get("text", "")}
            vec_rank_map[c["id"]] = rank
        bm25_rank_map = {}
        for rank, c in enumerate(bm25_results, start=1):
            if c["id"] not in merged:
                merged[c["id"]] = {"id": c["id"], "text": c.get("text", "")}
            bm25_rank_map[c["id"]] = rank

        K = 60
        big_rank = len(merged) + 1
        for cid, chunk in merged.items():
            chunk["hybrid_score"] = (
                1.0 / (K + vec_rank_map.get(cid, big_rank))
                + 1.0 / (K + bm25_rank_map.get(cid, big_rank))
            )

        use_rerank = self.reranker_mode != "none"
        recall_k = self.rerank_candidates if use_rerank else self.top_k
        scored = sorted(merged.values(), key=lambda x: x["hybrid_score"], reverse=True)[:recall_k]
        return {"scored_chunks": scored}

    def _node_rerank_node(self, state: NaiveState) -> NaiveState:
        """Node: cross-encoder 精排（仅当启用）"""
        query = state["query"]
        scored = state["scored_chunks"]
        reranked = self._rerank(query, scored)
        return {"top_chunks": reranked[:self.top_k]}

    def _node_generate(self, state: NaiveState) -> NaiveState:
        """Node: LLM 基于 top chunks 生成答案"""
        query = state["query"]
        # 如果没走 rerank，直接用 scored_chunks 的前 top_k
        top_chunks = state.get("top_chunks") or state.get("scored_chunks", [])[:self.top_k]

        if not top_chunks:
            return {"context": "", "answer": f"没有找到与'{query}'相关的信息。\n\n{{'data': {{'Chunks':[] }} }}"}

        chunks_content = []
        chunk_ids = []
        for item in top_chunks:
            cid = item.get("id", "unknown")
            text = item.get("text", "")
            if text:
                chunks_content.append(f"Chunk ID: {cid}\n{text}")
                chunk_ids.append(cid)
        context = "\n\n---\n\n".join(chunks_content)

        def _invoke():
            return self.query_chain.invoke(
                {"query": query, "context": context, "response_type": response_type}
            )
        try:
            answer = retry_sync(max_retries=2, base_delay=1.0)(_invoke)()
        except Exception as e:
            logger.error(f"[generate] 失败: {e}")
            answer = f"生成失败: {e}"

        if "{'data': {'Chunks':" not in answer:
            chunk_refs = ", ".join([f"'{i}'" for i in chunk_ids[:5]])
            answer += f"\n\n{{'data': {{'Chunks':[{chunk_refs}] }} }}"

        return {"context": context, "top_chunks": top_chunks, "answer": answer}

    def _get_cross_encoder(self):
        """不再使用本地模型，改用远程 API"""
        return None

    def _setup_chains(self):
        """设置处理链"""
        self.query_prompt = ChatPromptTemplate.from_messages([
            ("system", NAIVE_PROMPT),
            ("human", NAIVE_SEARCH_QUERY_PROMPT),
        ])
        self.query_chain = self.query_prompt | self.llm | StrOutputParser()

        # LLM Reranker chain
        self.rerank_prompt = ChatPromptTemplate.from_template(RERANK_PROMPT)
        self.rerank_chain = self.rerank_prompt | self.llm | StrOutputParser()

    def _rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """根据 reranker_mode 选择重排策略"""
        if self.reranker_mode == "cross_encoder":
            return self._rerank_cross_encoder(query, chunks)
        elif self.reranker_mode == "llm":
            return self._rerank_llm(query, chunks)
        else:
            return chunks

    def _fetch_chunk_paper_titles(self, chunk_ids: List[str]) -> Dict[str, str]:
        """
        一次查询拿到每个 chunk 对应的 paper 标题（fileName）
        用于 reranker 注入"论文归属"信息，解决 shared terminology 问题
        """
        if not chunk_ids:
            return {}
        try:
            result = self.graph.query(
                """MATCH (c:__Chunk__) WHERE c.id IN $ids
                   RETURN c.id AS id, c.fileName AS fname""",
                {"ids": chunk_ids},
            )
            mapping = {}
            for row in result:
                fname = row.get("fname") or ""
                # 清理 fileName：去掉 arxiv id 前缀和扩展名
                # "2604_09508v1__VISOR_Agentic_Visual_..." → "VISOR Agentic Visual..."
                import re
                cleaned = re.sub(r'^\d+_\d+v?\d*__', '', fname)
                cleaned = re.sub(r'\.(pdf|txt|md)$', '', cleaned, flags=re.IGNORECASE)
                cleaned = cleaned.replace('_', ' ').strip()
                mapping[row["id"]] = cleaned or fname
            return mapping
        except Exception as e:
            logger.warning("获取 chunk paper 标题失败", extra={"error": str(e)})
            return {}

    def _rerank_cross_encoder(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        用远程 Qwen3-Reranker API 重排
        改进：chunk text 前加 paper 标题，让 reranker 感知"这段属于哪篇论文"
        """
        import requests as _requests
        import concurrent.futures

        reranker_url = os.environ.get("RERANKER_URL", "http://localhost:8008/score")
        reranker_model = os.environ.get("RERANKER_MODEL", "Qwen/Qwen3-Reranker-4B")

        valid_chunks = [c for c in chunks if c.get("text")]
        if not valid_chunks:
            return chunks

        rerank_text_limit = int(os.environ.get("RERANKER_TEXT_LIMIT", "2000"))
        enable_paper_prefix = os.environ.get("RERANKER_PAPER_PREFIX", "1") == "1"

        # 批量拉 chunk → paper 映射（一次 DB 查询）
        paper_titles = {}
        if enable_paper_prefix:
            chunk_ids = [c["id"] for c in valid_chunks]
            paper_titles = self._fetch_chunk_paper_titles(chunk_ids)

        def score_one(chunk):
            try:
                text = chunk["text"][:rerank_text_limit]
                # 注入 paper 归属信息
                if enable_paper_prefix:
                    title = paper_titles.get(chunk["id"], "")
                    if title:
                        text = f"[From paper: {title}]\n{text}"

                resp = _requests.post(reranker_url, json={
                    "model": reranker_model,
                    "text_1": query,
                    "text_2": text,
                }, timeout=15)
                data = resp.json()
                return data["data"][0]["score"]
            except Exception:
                return 0.0

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(valid_chunks)) as executor:
            scores = list(executor.map(score_one, valid_chunks))

        for i, chunk in enumerate(valid_chunks):
            chunk["rerank_score"] = scores[i]

        valid_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return valid_chunks

    def _rerank_llm(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """用LLM对候选chunk重新打分排序"""
        reranked = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if not text:
                continue
            try:
                score_str = self.rerank_chain.invoke({
                    "query": query,
                    "chunk_text": text[:500]
                })
                score = RerankerScore.parse_llm_output(score_str).score
            except Exception:
                score = 0
            reranked.append({**chunk, "rerank_score": score})

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词（naive rag不需要复杂的关键词提取）
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 空的关键词字典
        """
        return {"low_level": [], "high_level": []}
    
    def _cosine_similarity(self, vec1, vec2):
        """
        计算两个向量的余弦相似度
        
        参数:
            vec1: 第一个向量
            vec2: 第二个向量
            
        返回:
            float: 相似度值
        """
        # 确保向量是numpy数组
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)
            
        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        # 避免被零除
        if norm_a == 0 or norm_b == 0:
            return 0
            
        return dot_product / (norm_a * norm_b)
    
    def search(self, query_input: Any, session_id: str = None,
               parent_config: dict = None) -> str:
        """
        执行 Naive RAG 搜索（通过 LangGraph StateGraph 执行）

        内部流程: embed → dual_recall → rrf_fuse → [rerank?] → generate_answer

        参数:
            query_input: 查询字符串或 {"query": ...}
            session_id: 可选 Langfuse session
            parent_config: 父 RunnableConfig（来自 Router 等），有则复用以嵌套 trace
        """
        overall_start = time.time()

        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
        else:
            query = str(query_input)

        # 缓存
        cache_key = f"naive:{query}"
        cached = self.cache_manager.get(cache_key)
        if cached:
            logger.debug("缓存命中", extra={"query": query[:30]})
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
                    "run_name": "naive_rag_search",
                    "metadata": {
                        "langfuse_session_id": session_id or "default",
                        "langfuse_tags": ["naive", "rag_search"],
                        "query": query[:100],
                    },
                }

        # 用 StateGraph 跑完整流程（每个 node 在 Langfuse 里独立显示）
        initial_state: NaiveState = {"query": query}
        try:
            final_state = self._graph.invoke(initial_state, config=lf_config if lf_config else None)
            answer = final_state.get("answer", "搜索无结果")
            self.cache_manager.set(cache_key, answer)
            total = time.time() - overall_start
            self.performance_metrics["total_time"] = total
            return answer
        except Exception as e:
            logger.error("Naive 搜索失败", extra={"error": str(e), "query": query[:50]})
            return f"搜索过程中出错: {str(e)}\n\n{{'data': {{'Chunks':[] }} }}"

    async def async_search(self, query_input: Any) -> str:
        """
        异步版 Naive RAG 搜索
        - embedding: aembed_query (异步)
        - Neo4j: async_db_query (异步)
        - BM25 + RRF: CPU 计算，直接同步
        - LLM: ainvoke (异步)
        """
        overall_start = time.time()

        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
        else:
            query = str(query_input)

        cache_key = f"naive:{query}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result

        try:
            search_start = time.time()
            # 异步 embedding（带 retry + timeout）
            @retry_async(max_retries=2, base_delay=1.0)
            async def _embed():
                result = await async_timeout(
                    self.embeddings.aembed_query(query), timeout_seconds=30
                )
                if result is None:
                    raise TimeoutError("embedding 超时")
                return result
            query_embedding = await _embed()

            use_rerank = self.reranker_mode != "none"
            recall_k = self.rerank_candidates if use_rerank else self.top_k
            vector_recall = recall_k * 3

            # 异步 Neo4j 向量检索
            try:
                vector_results = await self.async_db_query(
                    """CALL db.index.vector.queryNodes('chunk_vector', $k, $vec)
                    YIELD node, score
                    RETURN node.id AS id, node.text AS text, score""",
                    {"k": vector_recall, "vec": query_embedding}
                )
            except Exception:
                vector_results = await self.async_db_query(
                    """MATCH (c:__Chunk__) WHERE c.embedding IS NOT NULL
                    RETURN c.id AS id, c.text AS text, c.embedding AS embedding
                    LIMIT 200""",
                )
                vector_results = VectorUtils.rank_by_similarity(
                    query_embedding, vector_results, "embedding", vector_recall
                )

            # BM25 + RRF（纯 CPU，同步即可）
            bm25 = SimpleBM25()
            bm25_scored = bm25.score(query, vector_results)

            K = 60
            vec_sorted = sorted(vector_results, key=lambda x: x.get("score", 0), reverse=True)
            vec_rank = {c["id"]: rank + 1 for rank, c in enumerate(vec_sorted)}
            bm25_sorted_list = sorted(bm25_scored, key=lambda x: x.get("bm25_score", 0), reverse=True)
            bm25_rank = {c["id"]: rank + 1 for rank, c in enumerate(bm25_sorted_list)}

            for chunk in vector_results:
                cid = chunk["id"]
                chunk["hybrid_score"] = (
                    1.0 / (K + vec_rank.get(cid, len(vector_results)))
                    + 1.0 / (K + bm25_rank.get(cid, len(vector_results)))
                )

            scored_chunks = sorted(vector_results, key=lambda x: x.get("hybrid_score", 0), reverse=True)
            scored_chunks = scored_chunks[:recall_k]

            if use_rerank and scored_chunks:
                reranked = self._rerank(query, scored_chunks)
                results = reranked[:self.top_k]
            else:
                results = scored_chunks[:self.top_k]

            search_time = time.time() - search_start
            self.performance_metrics["query_time"] = search_time

            if not results:
                return f"没有找到与'{query}'相关的信息。\n\n{{'data': {{'Chunks':[] }} }}"

            chunks_content = []
            chunk_ids = []
            for item in results:
                chunk_id = item.get("id", "unknown")
                text = item.get("text", "")
                if text:
                    chunks_content.append(f"Chunk ID: {chunk_id}\n{text}")
                    chunk_ids.append(chunk_id)

            context = "\n\n---\n\n".join(chunks_content)

            # 异步 LLM 生成（带 retry + timeout）
            llm_start = time.time()

            @retry_async(max_retries=2, base_delay=1.0)
            async def _llm_generate():
                return await async_timeout(
                    self.query_chain.ainvoke({
                        "query": query, "context": context, "response_type": response_type
                    }),
                    timeout_seconds=60,
                    default="LLM 生成超时，请重试。"
                )
            answer = await _llm_generate()
            llm_time = time.time() - llm_start
            self.performance_metrics["llm_time"] = llm_time

            if "{'data': {'Chunks':" not in answer:
                chunk_references = ", ".join([f"'{id}'" for id in chunk_ids[:5]])
                answer += f"\n\n{{'data': {{'Chunks':[{chunk_references}] }} }}"

            self.cache_manager.set(cache_key, answer)

            total_time = time.time() - overall_start
            self.performance_metrics["total_time"] = total_time
            return answer

        except Exception as e:
            logger.error("Naive 异步搜索失败", extra={"error": str(e), "query": query[:50]})
            return f"搜索过程中出错: {str(e)}\n\n{{'data': {{'Chunks':[] }} }}"

    def get_tool(self) -> BaseTool:
        """获取搜索工具"""
        class NaiveRetrievalTool(BaseTool):
            name: str = "naive_retriever"
            description: str = naive_description

            def _run(self_tool, query: Any) -> str:
                return self.search(query)

            async def _arun(self_tool, query: Any) -> str:
                return await self.async_search(query)

        return NaiveRetrievalTool()
    
    def close(self):
        """关闭资源"""
        super().close()
