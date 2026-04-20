**[English](README.md) · [简体中文](README.zh-CN.md)**

# GraphRAG Multi-Path Agent · RAG Papers KG

A production-style GraphRAG system over arXiv RAG research papers, built on **LangGraph + Neo4j + Langfuse**, with a **classifier-based 3-way router** that dispatches queries to the most appropriate retrieval path (Classic / Graph / Agentic).

Benchmark on 200 self-constructed cross-document questions over 51 RAG papers: **LLM-Acc 0.57** with the router, matching the strongest single-strategy (Agentic 0.55) while cutting latency 50% and API cost 57%.

---

## Architecture

```
User query
   ↓
┌──────────────────────────────────────────────────────────────┐
│ RouterAgent (LangGraph classifier)                           │
│ "classic | graph | agentic"  (LLM few-shot + confidence)     │
└───────┬───────────────────┬──────────────────────┬───────────┘
        │ ~40%              │ ~48%                 │ ~12%
        ▼                   ▼                      ▼
┌─────────────┐   ┌──────────────────┐   ┌─────────────────────┐
│ Classic     │   │ Graph Path       │   │ Agentic Path        │
│  hybrid     │   │ local / global   │   │ planner → ReAct     │
│  (vec+BM25  │   │  · entity lookup │   │  (tools: classic,   │
│   + RRF)    │   │  · community     │   │   graph, fetch_doc, │
│  + rerank?  │   │    summary (MS   │   │   entity, path)     │
│  + answer   │   │    GraphRAG)     │   │ → aggregate         │
│  + citations│   │  · chunks        │   │                     │
│             │   │  · tiktoken      │   │                     │
│             │   │    budget        │   │                     │
│             │   │    (0.25/0.25/   │   │                     │
│             │   │     0.50)        │   │                     │
└─────────────┘   └──────────────────┘   └─────────────────────┘
```

**Key design choices**:

- **Classifier, not cascade** — one-shot route decision; empirically beat cascade (+13pp) because cascade intermediate tiers polluted answers.
- **Path-level tool composition** — the agentic path's ReAct uses `classic_search` / `graph_search` as **sub-tools**, not primitive retrieval ops. LLM orchestrates at the path level, not the primitive level.
- **MS GraphRAG alignment** — Local path packs `community_summary : entity_graph : chunks = 0.25 : 0.25 : 0.50` via `tiktoken` budget; Global path uses dynamic level selection (0/2/4) based on query scope keywords.
- **Contextual Retrieval** (Anthropic pattern) — each chunk embedded with an LLM-generated 40-80 word context prefix; `original_text` preserved separately so LLM compose sees clean text.
- **Hierarchical community summaries** — level 0 (fine) + level 2 + level 4 (coarse), generated bottom-up (level N uses level N-1 summaries as input), enabling query-scope-aware Global Search.

## Knowledge graph

Built from 51 arXiv RAG papers (~90K chars each, PyPDF2 → dehyphenation post-process → LangChain `RecursiveCharacterTextSplitter`):

| Metric | Value |
|---|---|
| Documents | 51 |
| Chunks | 1,904 (contextual-enriched) |
| Entities | 11,315 |
| Relationships | 64,884 |
| Community (level 0/2/4) | 3,769 / 1,748 / 1,706 |
| Level-0 summary coverage | **71%** |
| LLM-judge edge precision | **86-93%** (sampled) |

Indexes: `chunk_vector` (HNSW cosine), `chunk_fulltext` (BM25), entity `vector` (HNSW cosine).

## Evaluation

Self-constructed 200-question benchmark over the 51 papers, 10 question types × 3 difficulty tiers (simple fact / medium cross-doc comparison / complex aggregation).

| Strategy | LLM-Acc | Latency | $/query |
|---|---|---|---|
| Naive (hybrid) | 0.43 | 4.5s | $0.0005 |
| Graph | 0.51 | 5.8s | $0.0005 |
| Agentic (standalone) | 0.55 | 11.5s | $0.0013 |
| **Router (3-way)** | **0.57** | **5.8s** | **$0.0006** |

Router **outperforms** running Agentic on all queries (0.57 vs 0.55) while running ~50% faster and ~57% cheaper, because it avoids Agentic's expensive multi-round retrieval for simple factual queries where hybrid retrieval is already saturated (method / dataset / component identification reached 95-100% accuracy).

See [`benchmarks/results/`](./benchmarks/results/) for per-question breakdowns.

## Repository layout

```
graphrag_agent/           Core package
├── agents/
│   ├── router_agent.py       3-way classifier dispatcher
│   ├── naive_rag_agent.py    Baseline hybrid (Classic path wrapper)
│   ├── graph_agent.py        Graph path wrapper
│   ├── agentic_agent.py      Agentic path wrapper (bypass router)
│   └── paths/
│       ├── classic_path.py   hybrid + RRF + answer
│       ├── graph_path.py     local/global + community + chunks (tiktoken budget)
│       ├── agentic_path.py   planner → ReAct → aggregator
│       └── context_packer.py tiktoken-based dynamic context packing
├── search/tool/
│   ├── primitives.py         5 retrieval primitives (hybrid, graph_lookup, path_search, fetch_doc, entity_search)
│   ├── naive_search_tool.py  LangGraph StateGraph for Classic
│   ├── agentic_react_tool.py create_react_agent wrapper
│   ├── global_search_tool.py Map-Reduce community aggregation
│   └── ...
├── graph/                    KG construction (extraction/indexing/processing/community)
├── community/summary/
│   └── hierarchical.py       MS GraphRAG-style bottom-up level 0/2/4 summaries
├── pipelines/ingestion/      PDF → dehyphenation → chunking (jieba / RecursiveCharacterTextSplitter)
├── config/                   settings.py + prompts
└── utils/                    logging, langfuse_client, resilience

frontend/                  Streamlit UI (4 pages)
├── app.py                    Main chat
└── pages/
    ├── 1_🗺️_KG_Health.py     KG quality dashboard
    ├── 2_📊_Analytics.py     Langfuse aggregated metrics
    └── 3_🔍_Trace_Viewer.py  Per-trace call-tree renderer

server/                    FastAPI backend (chat + feedback endpoints)

scripts/
├── build/
│   ├── nuke_and_rebuild.py               Docker reset + full ingestion
│   ├── full_rebuild_pipeline.py          Orchestrator: nuke → dedup → summaries → quality
│   └── rebuild_hierarchical_summaries.py MS-style layer 0/2/4 summary rebuild
├── maintenance/
│   ├── dedup_case_variants.py            Merge case-variant entity duplicates
│   └── dehyphen_existing_chunks.py       Post-fix LaTeX hyphenation on existing chunks
├── eval/
│   ├── graph_quality_report.py           KG structural health + LLM-judge edge sampling
│   ├── behavior_analysis.py              Route distribution + cost from Langfuse
│   ├── langfuse_compare.py               Per-agent cost/latency comparison
│   └── compare_chunkers.py               jieba vs LangChain RecursiveCharacterTextSplitter A/B
└── data_prep/                            arXiv downloader + question generator

benchmarks/
├── run_hotpotqa_bench.py                 Router-agnostic bench runner (any agent_type)
├── run_agentic_hotpot_bench.py           Standalone agentic bench (no router)
├── eng_cross_doc_questions.json          200-question eval set
└── results/                              4 canonical result files + kg_quality.md

contextualize_chunks.py                   Anthropic-style contextual retrieval (standalone)
docker-compose.yaml                       Neo4j
```

## Quick start

```bash
# 1. Env
conda create -n graphrag python=3.10 -y && conda activate graphrag
pip install -r requirements.txt
pip install -e .

# 2. Services
docker compose up -d                   # Neo4j (this repo's compose)

# Langfuse self-host (in a separate directory) — for observability & the Analytics / Trace Viewer pages
git clone https://github.com/langfuse/langfuse.git ../langfuse
cd ../langfuse && docker compose up -d && cd -
# UI at http://localhost:3000 — sign up locally, create project, copy SECRET_KEY + PUBLIC_KEY

# 3. Config
cp .env.example .env
# Edit .env:
#   OPENAI_API_KEY, OPENAI_BASE_URL (e.g. OpenRouter)
#   LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL=http://localhost:3000
#   (Langfuse keys optional — app runs without them, only Analytics/Trace pages degrade)

# 4. Download the 50 RAG papers used in this project (for reproducibility)
pip install arxiv
python scripts/data_prep/download_rag_papers.py \
    --from-ids benchmarks/paper_ids.json \
    --dir files/rag_papers
# Or fetch fresh RAG papers by query:
#   python scripts/data_prep/download_rag_papers.py --n 50

# 5. Build the KG (40-60 min with default LLM, heavy on API)
#    Includes: chunking, embedding, entity extraction, Leiden, hierarchical summaries, indexes
python scripts/build/full_rebuild_pipeline.py --log /tmp/build.log

# 6. Run
python server/main.py                  # backend (:8000)
streamlit run frontend/app.py          # frontend (:8501)
```

## Benchmark

```bash
python benchmarks/run_hotpotqa_bench.py \
  --agent router \
  --questions benchmarks/eng_cross_doc_questions.json \
  --tag eng200 --workers 8
```

## Observability

- **Langfuse** — every retrieval step emits nested spans (Router → Path → Tool → LLM). Trace Viewer page renders the call tree with latency bar per span.
- **KG Health** page — structural metrics, completeness, community coverage, duplicate-suspect detection.
- **Analytics** page — route distribution, per-route latency/cost/LLM calls (reads `route:xxx` tag from traces; 1 API call for 200 traces, no heavy per-trace fetch).

## License

MIT
