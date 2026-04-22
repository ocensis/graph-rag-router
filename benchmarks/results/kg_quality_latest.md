# KG Quality Report

Generated: 2026-04-22T01:54:48

## 1. Scale
- Entities:        **11,774**
- Chunks:          **958**
- Documents:       **51**
- Relationships:   **66,132**

## 2. Type Distribution
### Relationship types (top 10)
| Type | Count |
|---|---:|
| IN_COMMUNITY | 29,741 |
| MENTIONS | 20,092 |
| USES | 3,590 |
| AUTHORED_BY | 2,092 |
| ADDRESSES | 1,854 |
| EVALUATED_ON | 1,659 |
| SIMILAR | 1,218 |
| PART_OF | 992 |
| NEXT_CHUNK | 910 |
| COMPARES_WITH | 807 |

### Entity labels (top 10)
| Label | Count |
|---|---:|
| Author | 3,083 |
| Component | 1,745 |
| Method | 1,633 |
| Paper | 1,274 |
| Task | 1,069 |
| Metric | 879 |
| Model | 805 |
| Dataset | 575 |
| 未知 | 551 |
| Institution | 387 |

## 3. Completeness
- Description coverage:  **100.0%** (11,774/11,774)
- Orphan entities:       **2,618** (22.2%)
- Chunks with MENTIONS:  **96.9%** (928/958)

## 4. Connectivity
- Avg degree:       **2.41**
- Degree p50/p95/max: 1.0 / 8.0 / 346
- Isolate nodes:    2,618

## 5. Relationship Semantic Health
- `OTHER` relation ratio: **0.43%** (285/66,132)
  - Interpretation: high OTHER% means LLM extraction wasn't aligned to schema well

## 6. Duplicate Suspects (name Jaccard ≥ 0.8)
- Checked first 2,000 entities → **64 suspect pairs**

| A | B | Jaccard |
|---|---|---:|
| CONTEXT COMPRESSION MODEL | CONTEXT COMPRESSION MODELS | 0.958 |
| CT enterography vision-language learning | CT enterography vision-language learning论文 | 0.95 |
| ATTENTION MECHANISM | ATTENTION MECHANISMS | 0.944 |
| CONSOLIDATION CYCLES | Consolidation cycle | 0.944 |
| ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS 35 | Advances in Neural Information Processing Systems | 0.938 |
| Chen et al., 2024 | Chen et al., 2024a | 0.938 |
| Chen et al., 2024 | Chen et al., 2024b | 0.938 |
| Bai et al., 2025 | Bai et al., 2025a | 0.933 |
| Bai et al., 2025 | Bai et al., 2025b | 0.933 |
| Context Operator | Context Operators | 0.933 |

## 7. Relationship Precision (LLM-sampled)
- Sampled: **30** non-MENTIONS edges
- Correct: 18 | Incorrect: 10 | Unclear: 2
- **Precision on judged: 64.3%**

Examples:
- `(THIS PAPER) -[COMPARES_WITH]-> (LLM WIKI V2)` → *unclear*
- `(Wei Dai) -[AUTHORED_BY]-> (CIRCUITSYNTH PAPER)` → *incorrect*
- `(伴侣记忆框架) -[USES]-> (DECAY)` → *correct*
- `(检索增强生成) -[EVALUATED_ON]-> (余弦相似度)` → *correct*
- `(PAYING ONLY THE INTEREST) -[LEADS_TO]-> (LOAN)` → *correct*
