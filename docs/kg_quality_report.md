# KG Quality Report

Generated: 2026-04-19T14:38:37

## 1. Scale
- Entities:        **14,697**
- Chunks:          **2,917**
- Documents:       **51**
- Relationships:   **81,941**

## 2. Type Distribution
### Relationship types (top 10)
| Type | Count |
|---|---:|
| IN_COMMUNITY | 31,140 |
| MENTIONS | 27,433 |
| AUTHORED_BY | 4,006 |
| PART_OF | 2,928 |
| NEXT_CHUNK | 2,866 |
| USES | 2,705 |
| EVALUATED_ON | 2,576 |
| ADDRESSES | 2,551 |
| COMPARES_WITH | 1,491 |
| SIMILAR | 922 |

### Entity labels (top 10)
| Label | Count |
|---|---:|
| Author | 4,869 |
| Method | 1,497 |
| Model | 1,446 |
| Component | 1,322 |
| Paper | 1,230 |
| Metric | 1,023 |
| Task | 1,011 |
| Dataset | 1,004 |
| 未知 | 511 |
| Institution | 290 |

## 3. Completeness
- Description coverage:  **100.0%** (14,697/14,697)
- Orphan entities:       **2,440** (16.6%)
- Chunks with MENTIONS:  **99.1%** (2,892/2,917)

## 4. Connectivity
- Avg degree:       **2.39**
- Degree p50/p95/max: 1.0 / 7.0 / 646
- Isolate nodes:    2,440

## 5. Relationship Semantic Health
- `OTHER` relation ratio: **0.17%** (139/81,941)
  - Interpretation: high OTHER% means LLM extraction wasn't aligned to schema well

## 6. Duplicate Suspects (name Jaccard ≥ 0.8)
- Checked first 2,000 entities → **112 suspect pairs**

| A | B | Jaccard |
|---|---|---:|
| 10-query Experiment | 10-query experiment | 1.0 |
| 1024-Dimensional Embedding | 1024-dimensional embedding | 1.0 |
| 2WIKI | 2Wiki | 1.0 |
| A-MEM | A-Mem | 1.0 |
| A-MEM | A-mem | 1.0 |
| A-Mem | A-mem | 1.0 |
| Ablation Study | Ablation study | 1.0 |
| ACC | Acc | 1.0 |
| Active Frontier | Active frontier | 1.0 |
| Active Retrieval Augmented Generation | Active retrieval augmented generation | 1.0 |

## 7. Relationship Precision (LLM-sampled)
- Sampled: **30** non-MENTIONS edges
- Correct: 26 | Incorrect: 2 | Unclear: 2
- **Precision on judged: 92.9%**

Examples:
- `(MM-Doc-R1) -[EVALUATED_ON]-> (MMLongBench-Doc)` → *correct*
- `(E2) -[USES]-> (OIDA)` → *unclear*
- `(Reasoning Framework) -[GUIDES]-> (LLM)` → *correct*
- `(DETR) -[EXTENDS]-> (DSHP-LLM)` → *correct*
- `(ACL 2023) -[SIMILAR]-> (ACL 2025)` → *correct*
