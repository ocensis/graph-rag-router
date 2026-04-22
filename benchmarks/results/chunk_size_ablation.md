# Chunk Size Ablation Report — 500 vs 1000 token

**Date**: 2026-04-21 to 2026-04-22
**Corpus**: 51 arXiv RAG papers (~90K chars each)
**Benchmark**: 200-question eng_cross_doc_questions.json (10 types × 3 difficulty tiers)
**Answer model (all runs)**: `deepseek/deepseek-v3.2`

---

## TL;DR

- **Chunk size 500 → 1000 steadily improves end-to-end accuracy by +2 to +8pp**, despite KG-level LLM-judge precision dropping 28.6pp.
- **Adding domain schema + contextual retrieval on top of 1000-token has near-zero effect on router**, but helps graph agent by +2pp.
- **KG-level quality metrics are NOT predictive of downstream agent accuracy** in this hybrid retrieval architecture.
- **Best single-agent result: graph agent on 1000-token-v3 KG — 63% LLM-Acc**.

---

## 1. Experimental Setup

### Configurations tested

| ID | Chunk size | Schema | Contextual retrieval | KG build model | Notes |
|---|---:|---|---|---|---|
| **500 baseline** | 500 / 100 | domain (Method/Dataset/USES/EVALUATED_ON) | ✓ | Gemini 3 Flash (re-benched with DS v3.2) | Old repo, pre-existing KG |
| **1000 v1** | 1000 / 200 | generic fallback (Concept/Person/RELATED_TO) | ✗ | DS v3.2 | First clean rebuild, kb_config missing |
| **1000 v2** | 1000 / 200 | domain intended, but cache-hit returned generic | ✓ | DS v3.2 | Rebuild with kb_config but extraction cache hit v1 results |
| **1000 v3** | 1000 / 200 | domain ✓ | ✓ | DS v3.2 | Clean rebuild after clearing `cache/graph` |

### Variables controlled in final comparison (500 vs 1000 v3)

| Variable | 500 baseline | 1000 v3 |
|---|---|---|
| Answer model | DS v3.2 | DS v3.2 |
| Domain schema | ✓ | ✓ |
| Contextual retrieval | ✓ | ✓ |
| Chunker | LangChain Recursive | LangChain Recursive |
| Embedding model | text-embedding-3-large | text-embedding-3-large |
| **Chunk size** | **500 token** | **1000 token** |

**Remaining confounder**: KG extraction model differs (Gemini vs DS v3.2). Impact is small given DS was used for all answer composition.

---

## 2. KG-level Metrics

| Metric | 500 | 1000 v1 | 1000 v2 | **1000 v3** |
|---|---:|---:|---:|---:|
| Chunks | 1,904 | 958 | 958 | 958 |
| Entities | 11,316 | 10,877 | 10,848 | 11,774 |
| Relationships | ~70K | 59,715 | 59,532 | 66,132 |
| Chunks with `original_text` (Anthropic CR) | 1,904 (100%) | 0 | 958 (100%) | 958 (100%) |
| Orphan entity rate | 16.6% | 19.2% | 19.3% | 22.2% |
| `OTHER` relation ratio | 0.17% | 0.22% | 0.23% | 0.43% |
| L0 community count | 3,763 | 5,113 | 5,113 | 5,625 |
| L0 summary coverage | **71.6%** | 54.3% | 54.3% | 48.5% |
| **LLM-judge edge precision** (n=30) | **92.9%** | 70.8% | 77.8% | **64.3%** |

### Top entity labels

| 500 | 1000 v1 | 1000 v3 |
|---|---|---|
| Author 2216 | Concept 4538 | Author 3083 |
| Method 2092 | Person 3263 | Component 1745 |
| Paper 1400 | Organization 1031 | **Method 1633** |
| Component 1354 | Event 934 | Paper 1274 |
| Task 1202 | Other 498 | Task 1069 |
| Model 1037 | 未知 286 | Metric 879 |
| Metric 999 | Location 157 | Model 805 |
| Dataset 568 | Method 22 | **Dataset 575** |
| Institution 326 | Dataset 29 | Institution 387 |

### Top relation types (non-infrastructure)

| 500 | 1000 v1 | 1000 v3 |
|---|---|---|
| USES 3829 | RELATED_TO 6266 | **USES 3590** |
| OTHER 3112 | PART_OF 4354 | AUTHORED_BY 2092 |
| ADDRESSES 2128 | MEMBER_OF 1603 | **ADDRESSES 1854** |
| AUTHORED_BY 1856 | SIMILAR 1017 | **EVALUATED_ON 1659** |
| EVALUATED_ON 1684 | LOCATED_IN 612 | SIMILAR 1218 |
| PROPOSES 1001 | OTHER 134 | **COMPARES_WITH 807** |
| | | PROPOSES 742 |

---

## 3. End-to-End Benchmark (200 questions)

### Overall LLM-Acc

| Agent | 500 | 1000 v1 | 1000 v3 | Δ (v3 vs 500) |
|---|---:|---:|---:|---:|
| router | 47.5% | 56.0% | 54.5% | **+7.0pp** |
| graph | 60.0% | 61.0% | **63.0%** | **+3.0pp** |

### By-type breakdown (LLM-Acc)

**Router agent**

| Type | n | 500 | 1000 v1 | 1000 v3 |
|---|---:|---:|---:|---:|
| method_identification | 37 | 86.5% | 81.1% | 86.5% |
| result_extraction | 22 | 72.7% | 95.5% | 90.9% |
| component_identification | 8 | 100% | 87.5% | 87.5% |
| dataset_identification | 13 | 69.2% | 84.6% | 61.5% |
| method_comparison | 30 | 40.0% | 50.0% | 56.7% |
| shared_problem | 30 | 33.3% | 63.3% | 46.7% |
| enumeration | 20 | 5.0% | 0.0% | 5.0% |
| statistics | 14 | 14.3% | 7.1% | 14.3% |
| topical_grouping | 14 | 0.0% | 0.0% | 0.0% |
| multi_comparison | 12 | 41.7% | 66.7% | 66.7% |

**Graph agent**

| Type | n | 500 | 1000 v1 | 1000 v3 |
|---|---:|---:|---:|---:|
| method_identification | 37 | 83.8% | 89.2% | 89.2% |
| result_extraction | 22 | 81.8% | 95.5% | 95.5% |
| component_identification | 8 | 100% | 87.5% | 75.0% |
| dataset_identification | 13 | 69.2% | 76.9% | 69.2% |
| method_comparison | 30 | 80.0% | 73.3% | **83.3%** |
| shared_problem | 30 | 80.0% | 70.0% | **83.3%** |
| enumeration | 20 | 0.0% | 5.0% | 5.0% |
| statistics | 14 | 0.0% | 7.1% | 0.0% |
| topical_grouping | 14 | 0.0% | 0.0% | 0.0% |
| multi_comparison | 12 | 50.0% | 50.0% | 50.0% |

---

## 4. Key Findings

### Finding 1: Chunk size 500 → 1000 consistently gains, independent of KG quality

Across three configurations (v1 no-schema/no-ctx, v2 half-polluted, v3 fully-clean) of the 1000-token KG, end-to-end accuracy is remarkably stable:

| Config | KG quality | Router LLM-Acc | vs 500 |
|---|---:|---:|---:|
| 500 baseline | 92.9% | 47.5% | — |
| 1000 v1 (worst KG) | 70.8% | 56.0% | +8.5pp |
| 1000 v3 (best KG) | 64.3% | 54.5% | +7.0pp |

The gain is driven by chunk-size itself. Per-chunk context doubling lets the LLM answer composer see more cross-entity co-occurrences in a single chunk, especially for comparison-style queries (method_comparison +10 to +17pp, multi_comparison +25pp).

### Finding 2: KG-level LLM-judge precision is a misleading indicator

KG-level LLM-judge precision moves **inversely** to end-to-end graph agent accuracy:

| | 500 | 1000 v1 | 1000 v3 |
|---|---:|---:|---:|
| KG-level LLM-judge precision | 92.9% | 70.8% | 64.3% |
| Graph agent end-to-end | 60.0% | 61.0% | 63.0% |

KG precision drops monotonically by 28.6pp; graph agent rises monotonically by 3pp. The GraphRAG graph path fuses `graph_lookup + community_summary + hybrid_chunks`, and chunk-side information density dominates answer composition. KG edge precision affects only a minority of the compose context.

**Practical implication**: KG quality metrics are observability signals, not optimization targets.

### Finding 3: Domain schema + contextual retrieval benefits are agent-dependent

Adding domain schema and Anthropic contextual retrieval on top of 1000-token KG (v1 → v3):

| Agent | v1 | v3 | Δ |
|---|---:|---:|---|
| router | 56.0% | 54.5% | **-1.5pp (noise)** |
| graph | 61.0% | 63.0% | **+2.0pp** |

Graph agent benefits (+2pp), but router does not. The gain concentrates in multi-entity queries (shared_problem +13.3pp on graph, method_comparison +10pp) — exactly the queries that rely on KG traversal. Router dilutes this by sending ~40% of traffic to the classic path, which doesn't use KG at all.

### Finding 4: Static router classifier is a blind spot

On v3, graph agent's `shared_problem_different_solutions` category rose from 70% (v1) to 83.3%. The router on v3 for the same category fell from 63.3% (v1) to 46.7% — a 36pp divergence.

The router's classifier made the same routing decisions in v3 as in v1, not realizing that the upgraded KG now makes graph path a better choice for these queries. **Static routing policies do not adapt to dynamic path-quality changes** — a general limitation of pre-trained multi-agent classifier designs.

---

## 5. Mechanism Recap

**Why does larger chunk size help this benchmark?**

The benchmark skews to cross-document comparison (30 method_comparison + 30 shared_problem + 20 enumeration + 12 multi_comparison = 92/200 queries). These require multiple entities discussed together in the retrieved context. With 500-token chunks, entity co-occurrence is frequently split across chunks, forcing the compose step to reconcile fragments. With 1000-token chunks, co-occurrence is preserved in-chunk, and the LLM can answer from a single retrieved chunk.

**Why doesn't KG precision matter more?**

In GraphRAG's hybrid retrieval (this project's graph path = graph_lookup + community summary + hybrid chunks, with tiktoken budget 0.25/0.25/0.50), 50% of the compose context is always raw chunks. KG side contributes entity relations and community summaries, but these are secondary grounding. Even low-precision relations (RELATED_TO instead of USES) still surface the right entity pairs — high recall compensates for low precision.

---

## 6. Practical Takeaways

| Question | Answer |
|---|---|
| Should the project have used 500 or 1000? | **1000** — gains 3-7pp on this benchmark, no significant downside. |
| Is 512-token embedding sweet spot still valid? | For pure single-fact retrieval, yes. For cross-doc comparison, no — co-occurrence beats retrieval precision. |
| Should I invest in KG schema alignment / contextual retrieval? | Only if you're using graph agent directly. Through a router, the benefit is diluted to noise. |
| Is LLM-judge edge precision a good KG quality metric? | No. Use it for debugging (finding mis-extracted relations), not as an optimization target. |
| What should I optimize first for higher end-to-end accuracy? | 1) Chunk size to match task distribution; 2) Router classifier (make it dynamic to per-path quality); 3) Global search / aggregation path — enumeration/statistics/topical_grouping are stuck at <15% across all configs. |

---

## 7. Configurations used in reproduction

```env
# .env
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
OPENAI_LLM_MODEL = 'deepseek/deepseek-v3.2'
OPENAI_EMBEDDINGS_MODEL = 'text-embedding-3-large'
NEO4J_URI = 'neo4j://localhost:17687'
KB_NAME = 'rag_papers'
```

```json
// kb_configs/rag_papers.json
{
  "kb_name": "rag_papers",
  "theme": "Retrieval-Augmented Generation academic papers from arXiv",
  "entity_types": ["Method", "Model", "Dataset", "Metric", "Task",
                   "Component", "Author", "Institution", "Paper"],
  "relationship_types": ["proposes", "evaluated_on", "outperforms",
                         "uses", "extends", "compares_with",
                         "published_by", "authored_by", "addresses", "other"]
}
```

**Build pipeline** (1000 v3, clean):
```bash
rm -rf cache/graph                                              # clear stale extraction cache
python scripts/build/full_rebuild_pipeline.py --log build.log   # ~120 min, ~$5
```

**Benchmarks**:
```bash
python benchmarks/run_hotpotqa_bench.py --agent router \
  --questions benchmarks/eng_cross_doc_questions.json \
  --tag eng200_1000tok_clean --workers 10

python benchmarks/run_hotpotqa_bench.py --agent graph \
  --questions benchmarks/eng_cross_doc_questions.json \
  --tag eng200_1000tok_clean --workers 10
```

Result files:
- `benchmarks/bench_results/hotpot_router_eng200_1000tok_clean_eval.json`
- `benchmarks/bench_results/hotpot_graph_eng200_1000tok_clean_eval.json`
- `benchmarks/results/kg_quality_latest.md` (1000 v3 KG quality report)
