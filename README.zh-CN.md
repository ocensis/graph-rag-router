**[English](README.md) · [简体中文](README.zh-CN.md)**

# GraphRAG 多路径智能体 · RAG 论文知识库

基于 **LangGraph + Neo4j + Langfuse** 的企业级 GraphRAG 系统，覆盖 51 篇 arXiv RAG 论文。核心是一个**分类器驱动的 3 路分派 Router**，根据 query 语义把请求分派到最合适的检索路径（Classic / Graph / Agentic），而不是让复杂 agent 兜底所有问题。

在自建的 **eng200 benchmark**（200 题跨文档评测）上，Router 达到 **LLM-Acc 0.57**，超过单策略最强的 Agentic（0.55），同时**延迟减半、成本降 57%**——这是这个项目的核心工程产出。

---

## 架构

```
用户 query
   ↓
┌──────────────────────────────────────────────────────────────┐
│ RouterAgent (LangGraph classifier)                           │
│  LLM few-shot 分类 → "classic | graph | agentic" + 置信度    │
│  低置信度回退到 classic（最安全）                              │
└───────┬───────────────────┬──────────────────────┬───────────┘
        │ ~40%              │ ~48%                 │ ~12%
        ▼                   ▼                      ▼
┌─────────────┐   ┌──────────────────┐   ┌─────────────────────┐
│ Classic     │   │ Graph Path       │   │ Agentic Path        │
│  hybrid     │   │ local / global   │   │ planner → ReAct     │
│ vec+BM25    │   │ · entity 邻域    │   │  工具集:             │
│ + RRF       │   │ · community 摘要 │   │  - classic_search   │
│ + 可选 rerank│   │ · chunks         │   │  - graph_search     │
│ + 带引用答案 │   │ · tiktoken 预算  │   │  - fetch_document   │
│              │  │   (0.25/0.25/   │   │  - entity_search    │
│              │  │    0.50) 借鉴    │   │  - path_search      │
│              │  │   微软 GraphRAG  │   │ → aggregate         │
└─────────────┘   └──────────────────┘   └─────────────────────┘
```

### 几个关键设计决策（面试 talking points）

**① 为什么用 classifier 不用 cascade？**

最早我做过 cascade 版本（naive 先试 → 不够升级到 graph → 还不够升级到 agentic），结果被 **classifier-first 直接打赢 13pp**（0.41 vs 0.54）。原因是 cascade 的中间档答案会污染最终——naive 输出"I don't know"的判决有 noise，常被 self-check 误判为 sufficient。Classifier 一次性分派避开这个耦合。

**② Agentic 的工具是 path 级别，不是原语级别**

最初版本 Agentic 的 ReAct 工具是 `hybrid_search` / `graph_lookup` / `vector_search` 等**原语**，LLM 需要在原语层面硬搓逻辑——结果测出 LLM 倾向"偷懒"，只调 1 次工具就停，method_comparison 类准确率暴跌到 0.13。

改成 `classic_search` / `graph_search` 作为 **path 级 sub-tools** 后，LLM 只需决策"这个子问题该走哪条 path"，不用自己拼 vector + BM25 + RRF，**同类问题涨到 0.50**。

**③ 借鉴微软 GraphRAG LocalSearch 的 token 预算管理**

之前是 `chunk[:3000]` 字符硬截断——会在句子中间切断，导致 embedding 退化。改成 `tiktoken` 精确计数 + **按比例动态分配**：`community_summary : entity_graph : chunks = 0.25 : 0.25 : 0.50`，某个来源短了剩余预算自动转给下一个来源。

**④ 分层社区摘要（Hierarchical Community Summary）**

Leiden 产出 level 0-4 五层社区，但原代码只摘 level 0 的前 1000 个（bug），level 1-4 都是空壳节点。

对齐微软 GraphRAG：**bottom-up 递归摘要**——level 0 用 entities+relations 摘；level 2 用 level 0 的摘要作输入再摘；level 4 用 level 2 的摘要再摘。这样 Global Search 能按 query 宽度动态选 level（`Summarize entire corpus` 用 level 4 最快，`main approaches` 用 level 2）。

**⑤ Contextual Retrieval**

借鉴 Anthropic 论文，每个 chunk 额外加一段 LLM 生成的 **40-80 词 context prefix**（说明这个 chunk 所属论文的 method / dataset），拼在原文前作为 embedding 和 BM25 的索引输入。查询时 LLM compose 看到的还是 `original_text` 原文（避免被 context summary 污染）。

---

## 知识图谱

数据源：51 篇 arXiv RAG 论文（平均 ~90K 字符）。

**Ingestion pipeline**（一条命令跑完）：

```
PyPDF2 提取 → regex 修软连字符折行 (LaTeX 自动断字导致的 "It-\nerative") 
           → RecursiveCharacterTextSplitter 分块 (对英文优于原来的 jieba)
           → OpenAI embedding (text-embedding-3-large)
           → LLM 抽实体+关系 (gemini-3-flash-preview 并发 20)
           → APOC case-variant dedup (兜底 WCC 漏掉的大小写重复)
           → GDS KNN + WCC + LLM 判决合并
           → Leiden 多层社区
           → Hierarchical summary level 0/2/4
           → Contextual retrieval (LLM 给每 chunk 加 context prefix)
           → Create chunk_vector (HNSW) + chunk_fulltext (BM25) 索引
```

**产出规模**：

| 指标 | 值 |
|------|---|
| Documents | 51 |
| Chunks | 1,904（contextual-enriched） |
| Entities | 11,315 |
| Relationships | 64,884 |
| Community lv 0 / 2 / 4 | 3,769 / 1,748 / 1,706 |
| Level-0 摘要覆盖率 | **71%** |
| LLM-judge 边精度采样 | **86-93%** |

---

## 评测

**Benchmark**：自建 200 题跨文档评测集，10 种问题类型 × 3 个难度档：
- Simple（4 类，80 题）：method / dataset / component / result identification
- Medium（2 类，60 题）：method_comparison / shared_problem_different_solutions
- Complex（4 类，60 题）：enumeration / statistics / topical_grouping / multi_comparison

**跑评测的命令**：

```bash
# Router（推荐，主要结果）
python benchmarks/run_hotpotqa_bench.py \
    --agent router \
    --questions benchmarks/eng_cross_doc_questions.json \
    --tag eng200 \
    --workers 8

# 对照组
python benchmarks/run_hotpotqa_bench.py --agent naive_rag_agent   --tag eng200_naive
python benchmarks/run_hotpotqa_bench.py --agent graph_agent       --tag eng200_graph
python benchmarks/run_hotpotqa_bench.py --agent agentic_agent     --tag eng200_agentic
```

**结果**：

| 策略 | LLM-Acc | 延迟 | 单次成本 | 相对 Agentic 成本 |
|------|---------|------|---------|------------------|
| Naive (hybrid) | 0.43 | 4.5s | $0.0005 | 38% |
| Graph | 0.51 | 5.8s | $0.0005 | 38% |
| Agentic (独跑) | 0.55 | 11.5s | $0.0013 | 100% |
| **Router (3-way)** | **0.57** | **5.8s** | **$0.0006** | **43%** |

Router 的质量 **超过** 独跑 Agentic（0.57 vs 0.55），而成本和延迟跟 Naive/Graph 差不多。**不是 Agentic 比 Classic 强**——简单事实查询用 Agentic 是浪费（拿全身搞不动手），关键是 Router 能识别 query 类型。

**详见** `benchmarks/results/` 下 4 个 JSON（含每题详细对比）。

**KG 质量报告** 在 `docs/kg_quality_report.md`。

---

## 仓库结构

```
graphrag_agent/           核心 Python 包
├── agents/
│   ├── router_agent.py       3-way 分类器分派器
│   ├── naive_rag_agent.py    Classic 路径封装（baseline）
│   ├── graph_agent.py        Graph 路径封装
│   ├── agentic_agent.py      Agentic 路径封装（不走 router）
│   └── paths/
│       ├── classic_path.py   hybrid + RRF + answer
│       ├── graph_path.py     local/global + community + chunks (token budget)
│       ├── agentic_path.py   planner → ReAct → aggregator
│       └── context_packer.py tiktoken 动态预算分配
├── search/tool/
│   ├── primitives.py         5 个检索原语
│   ├── naive_search_tool.py  Classic 的 LangGraph StateGraph 实现
│   ├── agentic_react_tool.py create_react_agent 封装
│   └── global_search_tool.py Map-Reduce 社区聚合
├── graph/                    KG 构建（抽取/索引/消歧/社区）
├── community/summary/hierarchical.py  微软风格分层摘要
├── pipelines/ingestion/      PDF→dehyphen→chunking
└── utils/                    日志 / langfuse_client / 重试

frontend/                  Streamlit UI (4 页)
├── app.py                    主对话页
└── pages/
    ├── 1_🗺️_KG_Health.py     知识图谱质量面板
    ├── 2_📊_Analytics.py     Langfuse 聚合指标面板
    └── 3_🔍_Trace_Viewer.py  单次查询调用树渲染

server/                    FastAPI 后端 (chat / feedback)

scripts/
├── build/
│   ├── nuke_and_rebuild.py               Docker 重置 + 全量建图
│   ├── full_rebuild_pipeline.py          编排器 nuke → dedup → summary → quality
│   └── rebuild_hierarchical_summaries.py 分层摘要独立重建
├── maintenance/
│   ├── dedup_case_variants.py            大小写变体实体合并兜底
│   └── dehyphen_existing_chunks.py       修复已有 chunk 的 LaTeX 折行
├── eval/
│   ├── graph_quality_report.py           KG 结构健康度 + LLM 判决边采样
│   ├── behavior_analysis.py              从 Langfuse 分析路由分布 / 成本
│   ├── langfuse_compare.py               各 agent 成本延迟对比
│   └── compare_chunkers.py               jieba vs LangChain splitter A/B 测试
└── data_prep/
    └── download_rag_papers.py            arXiv 论文下载（支持 --from-ids 复现）

benchmarks/
├── run_hotpotqa_bench.py                 通用 bench runner
├── eng_cross_doc_questions.json          200 题评测集
├── paper_ids.json                        50 篇论文 arxiv_id（用于复现）
└── results/                              4 个权威结果文件

contextualize_chunks.py                   Anthropic 风格 contextual retrieval
docker-compose.yaml                       Neo4j
```

---

## 快速开始

```bash
# 1. 环境
conda create -n graphrag python=3.10 -y && conda activate graphrag
pip install -r requirements.txt
pip install -e .

# 2. 服务
docker compose up -d                   # Neo4j (本仓库的 docker-compose.yaml)

# Langfuse 自部署（放在项目外的另一个目录，为前端的 Analytics / Trace Viewer 两个 page 提供数据）
git clone https://github.com/langfuse/langfuse.git ../langfuse
cd ../langfuse && docker compose up -d && cd -
# 访问 http://localhost:3000 → 本地注册账号 → 建 project → 复制 SECRET_KEY + PUBLIC_KEY

# 3. 配置
cp .env.example .env
# 编辑 .env:
#   OPENAI_API_KEY, OPENAI_BASE_URL (如 OpenRouter)
#   LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL=http://localhost:3000
#   (Langfuse 是可选项——不配也能跑，只是 Analytics/Trace Viewer 两个 page 没数据)

# 4. 下载 50 篇论文（保证复现和项目一致）
pip install arxiv
python scripts/data_prep/download_rag_papers.py \
    --from-ids benchmarks/paper_ids.json \
    --dir files/rag_papers

# 5. 建 KG （40-60 分钟，API 调用密）
python scripts/build/full_rebuild_pipeline.py --log /tmp/build.log

# 6. 起服务
python server/main.py                  # 后端 (:8000)
streamlit run frontend/app.py          # 前端 (:8501)
```

---

## 可观测性

3 个前端页面 + Langfuse 嵌套 trace：

- **KG Health**：实时查 Neo4j，显示规模 / 完整性 / community 覆盖 / 疑似重复实体
- **Analytics**：从 Langfuse 聚合，按 trace tag 读 route 分布（单次 API 调用拿 200 trace 数据，不再需要慢的 deep 模式）
- **Trace Viewer**：单条 query 的完整调用树（Router → Path → Tool → LLM），每个 span 带彩色 latency bar

**Langfuse trace 结构**：
```
router_agent (顶层 trace, tag=route:graph)
├─ classifier              (~0.3s)
├─ graph_path              (~4.5s)
│   ├─ extract_entities       (LLM)
│   ├─ graph_lookup(UniDoc-RL) (Cypher)
│   ├─ hybrid_grounding        (embedding + BM25)
│   └─ graph_compose           (LLM)
```

---

## 主要技术栈

- **LangGraph / LangChain** — 所有 agent / path / tool 编排（StateGraph + conditional_edges）
- **Neo4j + GDS** — 图存储 + Leiden 社区 + KNN
- **Langfuse** — trace 观测 + cost/latency 聚合
- **OpenRouter** — 模型网关，测过 GPT-4o-mini / Gemini Flash
- **tiktoken** — token 精确预算
- **Streamlit** — 前端
- **FastAPI** — 后端 API
- **APOC** — Neo4j 扩展（mergeNodes 做实体合并）

---

## License

MIT
