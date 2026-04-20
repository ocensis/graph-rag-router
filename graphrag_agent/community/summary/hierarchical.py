"""
Hierarchical Community Summarizer —— 借鉴微软 GraphRAG 的分层摘要设计

现有问题:
  - Leiden 产出 level 0-4 五层社区
  - 但旧代码只摘 level 0 的 top-200 ~ top-1000（且 batch > 20 break 导致上限 1000）
  - Level 1-4 全是空壳节点，没摘要

新做法:
  - Level 0: 全量摘（≥2 实体的社区都摘，修掉原来的 batch cap bug）
  - Level 1-N: 每个社区用它的子社区（N-1 级）摘要作输入，LLM 合成本层摘要
  - 这样任何粒度的全局查询都有现成 summary 可用

可视化 (Neo4j 里的 IN_COMMUNITY 边串起分层):
  Entity ─IN_COMMUNITY→ Community(lv=0) ─IN_COMMUNITY→ Community(lv=1) ─IN_COMMUNITY→ ... ─IN_COMMUNITY→ Community(lv=4)

每层摘要语义对照:
  lv 0: "这几个实体都是 GRPO 变体"
  lv 2: "这个区域围绕 RL-based RAG 训练方法"
  lv 4: "整个 KB 的大主题之一: 强化学习训练"
"""
from __future__ import annotations

import time
import json
import concurrent.futures
from typing import Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.config.settings import MAX_WORKERS
from graphrag_agent.config.prompts import (
    COMMUNITY_LEVEL0_SUMMARY_PROMPT,
    COMMUNITY_HIGHER_LEVEL_SUMMARY_PROMPT,
)
from graphrag_agent.models.get_models import get_llm_model
from graphrag_agent.utils.logger import get_logger
from graphrag_agent.utils.resilience import retry_sync

logger = get_logger(__name__)


class HierarchicalCommunitySummarizer:
    """
    分层社区摘要生成器。
    可以重跑在已有 KG 上（不需要重新做 Leiden），只覆盖 summary / full_content 字段。

    用法:
        from graphrag_agent.community.summary.hierarchical import HierarchicalCommunitySummarizer
        s = HierarchicalCommunitySummarizer(max_workers=8)
        s.summarize_all_levels(max_level=4, skip_odd_levels=True)
    """

    def __init__(self, max_workers: int = None):
        self.graph = get_db_manager().get_graph()
        self.llm = get_llm_model()
        self.max_workers = max_workers or MAX_WORKERS

        self.level0_chain = (
            ChatPromptTemplate.from_messages([("human", COMMUNITY_LEVEL0_SUMMARY_PROMPT)])
            | self.llm | StrOutputParser()
        )
        self.higher_chain = (
            ChatPromptTemplate.from_messages([("human", COMMUNITY_HIGHER_LEVEL_SUMMARY_PROMPT)])
            | self.llm | StrOutputParser()
        )

    # ==================== Level 0 ====================

    def _collect_level0_info(self, min_entities: int = 2) -> List[Dict]:
        """
        拉 level 0 所有社区的内部实体 + 关系，用于摘要。
        修掉原来的 batch cap bug —— 这次是全量。
        """
        query_start = time.time()
        result = self.graph.query(
            """
            MATCH (c:__Community__ {level: 0})
            WITH c ORDER BY coalesce(c.community_rank, 0) DESC

            MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
            WITH c, collect(e) AS nodes
            WHERE size(nodes) >= $min_entities

            CALL {
                WITH nodes
                UNWIND nodes AS n1
                UNWIND nodes AS n2
                WITH n1, n2 WHERE id(n1) < id(n2)
                MATCH (n1)-[r]->(n2)
                WHERE type(r) <> 'IN_COMMUNITY' AND type(r) <> 'MENTIONS'
                RETURN collect(DISTINCT r) AS relationships
            }

            RETURN c.id AS community_id,
                   [n IN nodes | {
                     id: n.id,
                     description: coalesce(n.description, ''),
                     type: CASE WHEN size([l IN labels(n) WHERE l <> '__Entity__']) > 0
                                THEN [l IN labels(n) WHERE l <> '__Entity__'][0]
                                ELSE 'Entity' END
                   }] AS nodes,
                   [r IN relationships | {
                     start: startNode(r).id,
                     rel: type(r),
                     end: endNode(r).id,
                     description: coalesce(r.description, '')
                   }] AS rels
            """,
            {"min_entities": min_entities},
        )
        elapsed = time.time() - query_start
        logger.info(f"[lv0] 收集到 {len(result)} 个社区 ({elapsed:.1f}s)")
        return result

    @staticmethod
    def _format_level0_info(info: Dict) -> str:
        """把实体+关系格式化成 LLM 可读的简短文本"""
        lines = [f"## Community {info['community_id']}"]
        lines.append("\n### Entities:")
        for n in info["nodes"][:30]:
            desc = n["description"][:150]
            lines.append(f"- [{n['type']}] {n['id']}: {desc}")
        if len(info["nodes"]) > 30:
            lines.append(f"... and {len(info['nodes']) - 30} more")
        if info.get("rels"):
            lines.append("\n### Relationships:")
            for r in info["rels"][:30]:
                lines.append(f"- ({r['start']}) -[{r['rel']}]-> ({r['end']})")
            if len(info["rels"]) > 30:
                lines.append(f"... and {len(info['rels']) - 30} more")
        return "\n".join(lines)

    def _summarize_level0_one(self, info: Dict) -> Dict:
        formatted = self._format_level0_info(info)
        try:
            summary = retry_sync(max_retries=2, base_delay=1.0)(self.level0_chain.invoke)(
                {"community_info": formatted[:4000]}
            )
            return {"community_id": info["community_id"], "summary": summary, "full_content": formatted}
        except Exception as e:
            logger.error(f"[lv0] 摘要 {info['community_id']} 失败: {e}")
            return {"community_id": info["community_id"], "summary": None, "full_content": formatted}

    def summarize_level0(self, min_entities: int = 2) -> int:
        """摘 level 0 所有 ≥min_entities 的社区"""
        infos = self._collect_level0_info(min_entities=min_entities)
        if not infos:
            logger.warning("[lv0] 无可摘社区")
            return 0

        start = time.time()
        summaries = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            futs = {exe.submit(self._summarize_level0_one, i): i for i in infos}
            done = 0
            for f in concurrent.futures.as_completed(futs):
                r = f.result()
                if r.get("summary"):
                    summaries.append(r)
                done += 1
                if done % 50 == 0 or done == len(infos):
                    logger.info(f"[lv0] {done}/{len(infos)} ({time.time()-start:.0f}s)")

        # 写回
        self._write_summaries(summaries)
        logger.info(f"[lv0] 完成 {len(summaries)}/{len(infos)} 摘要 ({time.time()-start:.0f}s)")
        return len(summaries)

    # ==================== Level N>0 ====================

    def _collect_higher_level_info(self, level: int) -> List[Dict]:
        """
        对每个 level=N 社区，找它**任意深度下**带有 summary 的最近子代。
        用变长路径 IN_COMMUNITY*1..$max_depth 跨越空层（如跳过了 level 1 时仍能从 level 0 取）。
        只保留每个高层社区最近一层有 summary 的子代（避免把孙子辈和儿子辈混在一起）。
        """
        query_start = time.time()
        max_depth = level  # 最多向下走 level 层（从 level 0 到 level N）
        result = self.graph.query(
            f"""
            MATCH (c:__Community__ {{level: $level}})
            MATCH (child:__Community__)-[:IN_COMMUNITY*1..{max_depth}]->(c)
            WHERE child.level < c.level
              AND child.summary IS NOT NULL AND child.summary <> ''
            // 为每个 c 取最近一层（level 最高）的子代
            WITH c, child
            ORDER BY child.level DESC
            WITH c,
                 max(child.level) AS closest_child_level,
                 collect(DISTINCT {{id: child.id, summary: child.summary, lv: child.level}}) AS all_children
            WITH c, closest_child_level,
                 [ch IN all_children WHERE ch.lv = closest_child_level] AS children
            WHERE size(children) >= 1
            RETURN c.id AS community_id, children, closest_child_level
            ORDER BY size(children) DESC
            """,
            {"level": level},
        )
        elapsed = time.time() - query_start
        logger.info(
            f"[lv{level}] 收集到 {len(result)} 个有子摘要的社区 "
            f"(从 level {result[0]['closest_child_level'] if result else '?'} 取子代, {elapsed:.1f}s)"
        )
        return result

    @staticmethod
    def _format_children_summaries(children: List[Dict], max_children: int = 15) -> str:
        lines = []
        for i, c in enumerate(children[:max_children], 1):
            lines.append(f"### Sub-community {i} ({c['id']})\n{c['summary']}")
        if len(children) > max_children:
            lines.append(f"... and {len(children) - max_children} more sub-communities")
        return "\n\n".join(lines)

    def _summarize_higher_one(self, info: Dict, level: int) -> Dict:
        formatted = self._format_children_summaries(info["children"])
        try:
            summary = retry_sync(max_retries=2, base_delay=1.0)(self.higher_chain.invoke)(
                {"children_summaries": formatted[:6000], "level": level}
            )
            return {"community_id": info["community_id"], "summary": summary, "full_content": formatted}
        except Exception as e:
            logger.error(f"[lv{level}] 摘要 {info['community_id']} 失败: {e}")
            return {"community_id": info["community_id"], "summary": None, "full_content": formatted}

    def summarize_higher_level(self, level: int) -> int:
        """摘一个高层（level ≥1）"""
        infos = self._collect_higher_level_info(level)
        if not infos:
            logger.warning(f"[lv{level}] 无可摘社区（子层可能还没摘）")
            return 0

        start = time.time()
        summaries = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            futs = {exe.submit(self._summarize_higher_one, i, level): i for i in infos}
            done = 0
            for f in concurrent.futures.as_completed(futs):
                r = f.result()
                if r.get("summary"):
                    summaries.append(r)
                done += 1
                if done % 50 == 0 or done == len(infos):
                    logger.info(f"[lv{level}] {done}/{len(infos)} ({time.time()-start:.0f}s)")

        self._write_summaries(summaries)
        logger.info(f"[lv{level}] 完成 {len(summaries)}/{len(infos)} 摘要 ({time.time()-start:.0f}s)")
        return len(summaries)

    # ==================== Storage ====================

    def _write_summaries(self, summaries: List[Dict], batch: int = 100):
        """批量写回 Neo4j"""
        if not summaries:
            return
        valid = [s for s in summaries if s.get("summary")]
        for i in range(0, len(valid), batch):
            chunk = valid[i : i + batch]
            self.graph.query(
                """
                UNWIND $rows AS row
                MATCH (c:__Community__ {id: row.community_id})
                SET c.summary = row.summary,
                    c.full_content = row.full_content,
                    c.summarized_at = datetime()
                """,
                {"rows": chunk},
            )

    # ==================== 对外入口 ====================

    def summarize_all_levels(self, max_level: int = 4, skip_odd_levels: bool = False,
                              min_entities: int = 2) -> Dict[str, int]:
        """
        完整跑：level 0 → 1 → 2 → ... → max_level
        skip_odd_levels=True 时跳过 1、3 层（中间态不摘，省 50% cost）
        返回每层产出的摘要数
        """
        stats = {}
        t_total = time.time()

        # Level 0
        logger.info("=" * 60)
        logger.info(f"Level 0 开始（min_entities={min_entities}）")
        logger.info("=" * 60)
        stats["level_0"] = self.summarize_level0(min_entities=min_entities)

        # Higher levels
        for level in range(1, max_level + 1):
            if skip_odd_levels and level % 2 == 1:
                logger.info(f"Level {level}: SKIP (skip_odd_levels=True)")
                continue
            logger.info("=" * 60)
            logger.info(f"Level {level} 开始（输入 = level {level-1} 摘要）")
            logger.info("=" * 60)
            stats[f"level_{level}"] = self.summarize_higher_level(level)

        logger.info(f"\n全部完成，总耗时 {time.time()-t_total:.0f}s")
        logger.info(f"覆盖统计: {json.dumps(stats, indent=2)}")
        return stats
