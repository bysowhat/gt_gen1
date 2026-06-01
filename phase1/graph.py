"""M1.5 持久化全局图 + Dijkstra.

GBPlanner2 风格: 节点是已采样的关节构型 + 元数据, 边是已碰撞验证的连接.
图跨 cycle 累积, 不重置.

主要操作:
    add_node(node)              加点
    add_edge(id_a, id_b, weight) 加边 (调用方负责验证 free)
    nodes_within(theta, radius) 关节空间近邻 (BallTree)
    nearest_node(theta)
    dijkstra(src_id, dst_id)    单源单目标最短路径
    dijkstra_to_many(src_id, dst_ids)
    frontier_nodes(threshold)   gain >= 阈值的所有节点
    update_node_gain(id, gain, cycle)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
from sklearn.neighbors import BallTree


# ---------------------------------------------------------------------- node


@dataclass
class GraphNode:
    """节点元数据."""

    node_id: str
    theta: np.ndarray              # (dof,) 关节角
    ee_pos_world: np.ndarray       # (3,) 末端 world 位置
    ee_dir_world: np.ndarray       # (3,) 相机光轴 world 方向
    cycle_added: int               # 哪个 cycle 加入
    last_gain: float = 0.0         # 上次评估的增益 (M1.6)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.theta = np.asarray(self.theta, dtype=np.float64).reshape(-1)
        self.ee_pos_world = np.asarray(self.ee_pos_world, dtype=np.float64).reshape(3)
        self.ee_dir_world = np.asarray(self.ee_dir_world, dtype=np.float64).reshape(3)


def make_node_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------- graph


class ExplorationGraph:
    """持久化无向图 + BallTree 近邻."""

    def __init__(self):
        self._g = nx.Graph()
        self._tree: Optional[BallTree] = None
        self._tree_dirty = True
        self._node_thetas: list[np.ndarray] = []  # 与 _node_ids 同序
        self._node_ids: list[str] = []

    # ------------------------------------------------------------------ basic

    def __len__(self) -> int:
        return self._g.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._g.number_of_edges()

    @property
    def num_nodes(self) -> int:
        return self._g.number_of_nodes()

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._g

    # ------------------------------------------------------------------ nodes

    def add_node(self, node: GraphNode) -> str:
        if node.node_id in self._g:
            raise ValueError(f"node id '{node.node_id}' already exists")
        self._g.add_node(node.node_id, data=node)
        self._node_thetas.append(node.theta.copy())
        self._node_ids.append(node.node_id)
        self._tree_dirty = True
        return node.node_id

    def get_node(self, node_id: str) -> GraphNode:
        return self._g.nodes[node_id]["data"]

    def all_nodes(self) -> list[GraphNode]:
        return [self._g.nodes[nid]["data"] for nid in self._g.nodes]

    # ------------------------------------------------------------------ edges

    def add_edge(self, id_a: str, id_b: str, weight: float,
                 metadata: dict | None = None) -> None:
        if id_a not in self._g or id_b not in self._g:
            raise KeyError(f"unknown node: {id_a} or {id_b}")
        if id_a == id_b:
            raise ValueError("self-edge not allowed")
        if weight < 0:
            raise ValueError(f"weight must be >= 0; got {weight}")
        self._g.add_edge(id_a, id_b, weight=float(weight),
                         metadata=metadata or {})

    def remove_edge(self, id_a: str, id_b: str) -> None:
        if self._g.has_edge(id_a, id_b):
            self._g.remove_edge(id_a, id_b)

    def has_edge(self, id_a: str, id_b: str) -> bool:
        return self._g.has_edge(id_a, id_b)

    def edges_of(self, node_id: str) -> list[tuple[str, str, float]]:
        return [(a, b, d.get("weight", 0.0))
                for a, b, d in self._g.edges(node_id, data=True)]

    # ------------------------------------------------------------------ nearest neighbors

    def _maybe_rebuild_tree(self) -> None:
        if not self._tree_dirty:
            return
        if not self._node_thetas:
            self._tree = None
        else:
            arr = np.stack(self._node_thetas)
            # leaf_size 经验值; 6-D BallTree 比 KDTree 更稳
            self._tree = BallTree(arr, leaf_size=20)
        self._tree_dirty = False

    def nodes_within(self, theta: np.ndarray, radius: float) -> list[GraphNode]:
        """关节空间球内的所有节点 (含距离 = 0 的自己)."""
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        self._maybe_rebuild_tree()
        if self._tree is None:
            return []
        idx = self._tree.query_radius(theta.reshape(1, -1), r=float(radius))[0]
        return [self.get_node(self._node_ids[i]) for i in idx]

    def nearest_node(self, theta: np.ndarray) -> Optional[GraphNode]:
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        self._maybe_rebuild_tree()
        if self._tree is None:
            return None
        _, idx = self._tree.query(theta.reshape(1, -1), k=1)
        return self.get_node(self._node_ids[int(idx[0, 0])])

    def k_nearest(self, theta: np.ndarray, k: int) -> list[GraphNode]:
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        self._maybe_rebuild_tree()
        if self._tree is None:
            return []
        k = min(k, len(self._node_ids))
        _, idx = self._tree.query(theta.reshape(1, -1), k=k)
        return [self.get_node(self._node_ids[i]) for i in idx[0]]

    # ------------------------------------------------------------------ shortest path

    def dijkstra(self, src_id: str, dst_id: str) -> Optional[list[GraphNode]]:
        """从 src 到 dst 的最短路径节点列表. 不可达 → None."""
        if src_id not in self._g or dst_id not in self._g:
            return None
        try:
            path_ids = nx.dijkstra_path(self._g, src_id, dst_id, weight="weight")
        except nx.NetworkXNoPath:
            return None
        return [self.get_node(i) for i in path_ids]

    def dijkstra_length(self, src_id: str, dst_id: str) -> Optional[float]:
        if src_id not in self._g or dst_id not in self._g:
            return None
        try:
            return float(nx.dijkstra_path_length(self._g, src_id, dst_id, weight="weight"))
        except nx.NetworkXNoPath:
            return None

    def dijkstra_to_many(
        self, src_id: str, dst_ids: list[str],
    ) -> dict[str, list[GraphNode]]:
        """单源到多目标. 返回 dict {dst_id: path}, 不可达者不在 dict 里."""
        if src_id not in self._g:
            return {}
        lengths, paths = nx.single_source_dijkstra(self._g, src_id, weight="weight")
        out: dict[str, list[GraphNode]] = {}
        for did in dst_ids:
            if did in paths:
                out[did] = [self.get_node(i) for i in paths[did]]
        return out

    # ------------------------------------------------------------------ gain / frontier

    def update_node_gain(self, node_id: str, gain: float,
                         cycle: int | None = None) -> None:
        node = self.get_node(node_id)
        node.last_gain = float(gain)
        if cycle is not None:
            node.metadata["last_gain_cycle"] = cycle

    def frontier_nodes(self, gain_threshold: float = 5.0) -> list[GraphNode]:
        """gain >= 阈值的所有节点, 按 gain 降序."""
        nodes = [n for n in self.all_nodes() if n.last_gain >= gain_threshold]
        nodes.sort(key=lambda n: n.last_gain, reverse=True)
        return nodes

    # ------------------------------------------------------------------ misc

    def edges_with_node_in_box(
        self, theta_center: np.ndarray, half_extent: float,
    ) -> list[tuple[str, str]]:
        """返回所有"两个端点中至少一个在 theta_center 附近"的边. 用于地图局部
        变化后重检. half_extent 是关节空间球半径."""
        near = set(n.node_id for n in self.nodes_within(theta_center, half_extent))
        return [(a, b) for a, b in self._g.edges if a in near or b in near]

    def stats(self) -> dict:
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_components": nx.number_connected_components(self._g),
            "max_gain": max((n.last_gain for n in self.all_nodes()), default=0.0),
        }


__all__ = ["GraphNode", "ExplorationGraph", "make_node_id"]
