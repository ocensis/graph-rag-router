# Agent性能对比表

| 指标 | naive | hybrid | graph |
| --- | --- | --- | --- |
| **答案质量指标** |  |  |  |
| answer_comprehensiveness | 0.2600 | 0.4133 | 0.0733 |
| em | 0.0267 | 0.1400 | 0.0400 |
| f1 | 0.1068 | 0.2157 | 0.0633 |
| factual_consistency | 0.6733 | 0.9367 | 0.5400 |
| response_coherence | 0.6600 | 0.9600 | 0.5100 |
| **LLM评估指标** |  |  |  |
| Comprehensiveness | 0.3233 | 0.4967 | 0.0667 |
| Directness | 0.7333 | 0.7767 | 0.5500 |
| Empowerment | 0.3467 | 0.5733 | 0.0833 |
| Relativeness | 0.5600 | 0.8400 | 0.3500 |
| Total | 0.4703 | 0.6577 | 0.2383 |
| **检索性能指标** |  |  |  |
| retrieval_latency | 3.7669 | 0.0056 | 4.9244 |
| retrieval_precision | 0.3000 | 0.5333 | 0.7667 |
| retrieval_utilization | 0.3000 | 0.3000 | 0.3000 |