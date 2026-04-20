# Agent性能对比表

| 指标 | naive | hybrid | graph |
| --- | --- | --- | --- |
| **答案质量指标** |  |  |  |
| answer_comprehensiveness | 0.5233 | 0.4000 | 0.1000 |
| em | 0.3600 | 0.1333 | 0.0600 |
| f1 | 0.3967 | 0.2390 | 0.0818 |
| factual_consistency | 0.6900 | 0.9433 | 0.7933 |
| response_coherence | 0.6800 | 0.9633 | 0.6433 |
| **LLM评估指标** |  |  |  |
| Comprehensiveness | 0.5000 | 0.4967 | 0.1333 |
| Directness | 0.7000 | 0.7867 | 0.7500 |
| Empowerment | 0.5333 | 0.5667 | 0.1833 |
| Relativeness | 0.5900 | 0.8800 | 0.5833 |
| Total | 0.5708 | 0.6680 | 0.3817 |
| **检索性能指标** |  |  |  |
| retrieval_latency | 3.8666 | 0.0007 | 5.0496 |
| retrieval_precision | 0.3000 | 0.5333 | 0.5333 |
| retrieval_utilization | 0.3000 | 0.3000 | 0.3000 |