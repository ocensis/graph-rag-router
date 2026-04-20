from typing import Dict, Any, Tuple

class GraphProjectionMixin:
    """图投影功能的混入类，兼容 graphdatascience 1.12"""

    def create_projection(self) -> Tuple[Any, Dict]:
        """创建图投影"""
        print("开始创建社区检测的图投影...")

        # 删除已存在的投影
        try:
            self.gds.graph.drop(self.projection_name, failIfMissing=False)
        except Exception:
            pass

        # 标准投影：无向图，不带 weight
        try:
            self.G, result = self.gds.graph.project(
                self.projection_name,
                "__Entity__",
                {"_ALL_": {"type": "*", "orientation": "UNDIRECTED"}},
            )
            node_count = result.get("nodeCount", 0) if isinstance(result, dict) else getattr(result, "node_count", 0)
            rel_count = result.get("relationshipCount", 0) if isinstance(result, dict) else getattr(result, "relationship_count", 0)
            print(f"图投影创建成功: {node_count} 节点, {rel_count} 关系")
            return self.G, result
        except Exception as e:
            print(f"标准投影失败: {e}")
            return self._create_cypher_projection()

    def _create_cypher_projection(self) -> Tuple[Any, Dict]:
        """使用 Cypher 投影作为降级方案"""
        print("尝试 Cypher 投影...")

        try:
            self.gds.graph.drop(self.projection_name, failIfMissing=False)
        except Exception:
            pass

        try:
            self.G, result = self.gds.graph.project.cypher(
                self.projection_name,
                "MATCH (e:__Entity__) RETURN id(e) AS id",
                "MATCH (e1:__Entity__)-[r]-(e2:__Entity__) RETURN id(e1) AS source, id(e2) AS target, type(r) AS type",
            )
            print(f"Cypher 投影创建成功")
            return self.G, result
        except Exception as e:
            print(f"Cypher 投影也失败: {e}")
            raise ValueError(f"无法创建图投影: {e}")