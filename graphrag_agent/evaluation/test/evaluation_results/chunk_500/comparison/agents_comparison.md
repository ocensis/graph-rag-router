# Agent性能对比表

| 指标 | naive | hybrid | graph |
| --- | --- | --- | --- |
| **答案质量指标** |  |  |  |
| answer_comprehensiveness | 0.1733 | 0.4133 | 0.4567 |
| em | 0.0400 | 0.1167 | 0.3900 |
| f1 | 0.0867 | 0.2424 | 0.4533 |
| factual_consistency | 0.5333 | 0.9367 | 0.9433 |
| response_coherence | 0.3833 | 0.9600 | 0.9400 |
| **LLM评估指标** |  |  |  |
| Comprehensiveness | 0.1933 | 0.5833 | 0.5000 |
| Directness | 0.4967 | 0.8367 | 0.7567 |
| Empowerment | 0.2200 | 0.5467 | 0.5267 |
| Relativeness | 0.3067 | 0.9100 | 0.8267 |
| Total | 0.2890 | 0.7065 | 0.6397 |
| **检索性能指标** |  |  |  |
| retrieval_latency | 13.2683 | 21.1643 | 9.9114 |
| retrieval_precision | 0.3000 | 0.4067 | 0.3000 |
| retrieval_utilization | 0.3000 | 0.3067 | 0.3000 |