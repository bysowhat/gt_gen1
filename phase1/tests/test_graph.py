"""M1.5 单元测试: ExplorationGraph.

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python -m pytest \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/test_graph.py -v
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from phase1.graph import ExplorationGraph, GraphNode, make_node_id


def _node(name: str, theta, gain: float = 0.0, cycle: int = 0) -> GraphNode:
    return GraphNode(
        node_id=name,
        theta=np.asarray(theta, dtype=np.float64),
        ee_pos_world=np.zeros(3),
        ee_dir_world=np.array([0, 0, 1.0]),
        cycle_added=cycle,
        last_gain=gain,
    )


# ---------------------------------------------------------------------- basic


def test_init_empty():
    g = ExplorationGraph()
    assert len(g) == 0
    assert g.num_edges == 0
    assert g.dijkstra("x", "y") is None


def test_add_node():
    g = ExplorationGraph()
    g.add_node(_node("a", np.zeros(6)))
    assert len(g) == 1
    assert "a" in g
    assert g.get_node("a").theta.shape == (6,)


def test_add_node_duplicate_raises():
    g = ExplorationGraph()
    g.add_node(_node("a", np.zeros(6)))
    with pytest.raises(ValueError):
        g.add_node(_node("a", np.ones(6)))


def test_make_node_id_unique():
    ids = {make_node_id() for _ in range(100)}
    assert len(ids) == 100


# ---------------------------------------------------------------------- edges


def test_add_edge_basic():
    g = ExplorationGraph()
    g.add_node(_node("a", np.zeros(6)))
    g.add_node(_node("b", np.ones(6) * 0.1))
    g.add_edge("a", "b", weight=1.0)
    assert g.has_edge("a", "b")
    assert g.has_edge("b", "a")  # undirected
    assert g.num_edges == 1


def test_add_edge_self_loop_raises():
    g = ExplorationGraph()
    g.add_node(_node("a", np.zeros(6)))
    with pytest.raises(ValueError):
        g.add_edge("a", "a", weight=1.0)


def test_add_edge_unknown_node_raises():
    g = ExplorationGraph()
    g.add_node(_node("a", np.zeros(6)))
    with pytest.raises(KeyError):
        g.add_edge("a", "missing", weight=1.0)


def test_add_edge_negative_weight_raises():
    g = ExplorationGraph()
    g.add_node(_node("a", np.zeros(6)))
    g.add_node(_node("b", np.ones(6)))
    with pytest.raises(ValueError):
        g.add_edge("a", "b", weight=-0.1)


def test_remove_edge():
    g = ExplorationGraph()
    g.add_node(_node("a", np.zeros(6)))
    g.add_node(_node("b", np.ones(6) * 0.1))
    g.add_edge("a", "b", weight=1.0)
    g.remove_edge("a", "b")
    assert not g.has_edge("a", "b")
    g.remove_edge("a", "b")  # 重复 remove 不报错


# ---------------------------------------------------------------------- dijkstra


def test_dijkstra_simple_chain():
    """a-b-c, dijkstra(a, c) = [a, b, c]."""
    g = ExplorationGraph()
    g.add_node(_node("a", np.zeros(6)))
    g.add_node(_node("b", np.full(6, 0.1)))
    g.add_node(_node("c", np.full(6, 0.2)))
    g.add_edge("a", "b", weight=1.0)
    g.add_edge("b", "c", weight=1.0)
    path = g.dijkstra("a", "c")
    assert [n.node_id for n in path] == ["a", "b", "c"]


def test_dijkstra_picks_shortest_of_two():
    """两条路径, dijkstra 选短的."""
    g = ExplorationGraph()
    for i, name in enumerate(["a", "b", "c", "d"]):
        g.add_node(_node(name, np.full(6, i * 0.1)))
    # 长路: a→b→c (cost 20)
    g.add_edge("a", "b", weight=10.0)
    g.add_edge("b", "c", weight=10.0)
    # 短路: a→d→c (cost 2)
    g.add_edge("a", "d", weight=1.0)
    g.add_edge("d", "c", weight=1.0)
    path = g.dijkstra("a", "c")
    assert [n.node_id for n in path] == ["a", "d", "c"]


def test_dijkstra_no_path_returns_none():
    g = ExplorationGraph()
    g.add_node(_node("a", np.zeros(6)))
    g.add_node(_node("b", np.zeros(6)))
    assert g.dijkstra("a", "b") is None


def test_dijkstra_unknown_node_returns_none():
    g = ExplorationGraph()
    g.add_node(_node("a", np.zeros(6)))
    assert g.dijkstra("a", "missing") is None
    assert g.dijkstra("missing", "a") is None


def test_dijkstra_length():
    g = ExplorationGraph()
    g.add_node(_node("a", np.zeros(6)))
    g.add_node(_node("b", np.zeros(6)))
    g.add_node(_node("c", np.zeros(6)))
    g.add_edge("a", "b", weight=2.0)
    g.add_edge("b", "c", weight=3.0)
    assert g.dijkstra_length("a", "c") == 5.0
    assert g.dijkstra_length("a", "missing") is None


def test_dijkstra_to_many():
    g = ExplorationGraph()
    for n in ["a", "b", "c", "iso"]:
        g.add_node(_node(n, np.zeros(6)))
    g.add_edge("a", "b", weight=1.0)
    g.add_edge("b", "c", weight=1.0)
    out = g.dijkstra_to_many("a", ["b", "c", "iso"])
    assert "b" in out and "c" in out
    assert "iso" not in out  # 不可达
    assert [n.node_id for n in out["b"]] == ["a", "b"]
    assert [n.node_id for n in out["c"]] == ["a", "b", "c"]


# ---------------------------------------------------------------------- ball tree


def test_nodes_within_basic():
    g = ExplorationGraph()
    np.random.seed(0)
    for i in range(50):
        theta = np.random.randn(6) * 0.3
        g.add_node(_node(f"n{i}", theta))

    center = np.zeros(6)
    near = g.nodes_within(center, radius=0.4)
    # 验证: 返回的全部确实在球内
    for n in near:
        assert np.linalg.norm(n.theta - center) <= 0.4 + 1e-9
    # 验证: 球外的不在结果里 (full check)
    near_ids = {n.node_id for n in near}
    for n in g.all_nodes():
        d = np.linalg.norm(n.theta - center)
        if d <= 0.4 - 1e-9:
            assert n.node_id in near_ids


def test_nearest_node():
    g = ExplorationGraph()
    g.add_node(_node("a", np.zeros(6)))
    g.add_node(_node("b", np.full(6, 0.5)))
    g.add_node(_node("c", np.full(6, 1.0)))
    near = g.nearest_node(np.full(6, 0.4))
    assert near.node_id == "b"


def test_nearest_after_add_invalidates_tree():
    """加节点之后, nearest 必须返回新节点 (BallTree 必须 rebuild)."""
    g = ExplorationGraph()
    g.add_node(_node("a", np.zeros(6)))
    g.nearest_node(np.full(6, 0.5))  # 触发 build
    g.add_node(_node("b", np.full(6, 0.5)))  # 标 dirty
    near = g.nearest_node(np.full(6, 0.5))
    assert near.node_id == "b"


def test_k_nearest():
    g = ExplorationGraph()
    for i in range(10):
        g.add_node(_node(f"n{i}", np.full(6, i * 0.1)))
    res = g.k_nearest(np.zeros(6), k=3)
    assert [n.node_id for n in res] == ["n0", "n1", "n2"]


def test_nearest_empty_graph():
    g = ExplorationGraph()
    assert g.nearest_node(np.zeros(6)) is None
    assert g.nodes_within(np.zeros(6), 1.0) == []


# ---------------------------------------------------------------------- frontier


def test_frontier_nodes_threshold():
    g = ExplorationGraph()
    for i in range(5):
        g.add_node(_node(f"n{i}", np.zeros(6), gain=float(i * 5)))
    # gain ∈ {0, 5, 10, 15, 20}; threshold 10 → n2,n3,n4 通过
    fronts = g.frontier_nodes(gain_threshold=10.0)
    assert [n.node_id for n in fronts] == ["n4", "n3", "n2"]  # 按 gain 降序


def test_update_node_gain():
    g = ExplorationGraph()
    g.add_node(_node("a", np.zeros(6)))
    g.update_node_gain("a", 42.0, cycle=7)
    assert g.get_node("a").last_gain == 42.0
    assert g.get_node("a").metadata["last_gain_cycle"] == 7


# ---------------------------------------------------------------------- 跨 cycle 持久化


def test_persistent_across_cycles():
    """图不应在 cycle 间清空 (跟 DSVP 不同)."""
    g = ExplorationGraph()
    for c in range(3):
        for i in range(10):
            g.add_node(_node(f"c{c}_n{i}", np.random.randn(6) * 0.3, cycle=c))
    assert len(g) == 30
    # cycle 0 的节点仍在
    assert "c0_n0" in g


# ---------------------------------------------------------------------- box query


def test_edges_with_node_in_box():
    g = ExplorationGraph()
    g.add_node(_node("a", np.zeros(6)))
    g.add_node(_node("b", np.full(6, 0.05)))
    g.add_node(_node("far", np.full(6, 5.0)))
    g.add_edge("a", "b", weight=0.1)
    g.add_edge("a", "far", weight=10.0)
    edges = g.edges_with_node_in_box(np.zeros(6), half_extent=0.2)
    # a 和 b 在范围内, 边 (a,b) 应返回; (a, far) 因为 a 也在范围内, 也返回
    edge_set = {tuple(sorted(e)) for e in edges}
    assert ("a", "b") in edge_set
    assert ("a", "far") in edge_set


# ---------------------------------------------------------------------- 性能


def test_dijkstra_scales_to_1000():
    """1000 节点 + ~5K 边, dijkstra 应 < 100 ms."""
    g = ExplorationGraph()
    np.random.seed(42)
    nodes = []
    for i in range(1000):
        n = _node(f"n{i}", np.random.randn(6) * 0.5)
        g.add_node(n)
        nodes.append(n)

    # 每个节点连 5 个最近邻
    for i, n in enumerate(nodes):
        nbs = g.nodes_within(n.theta, radius=0.5)[:6]
        for nb in nbs:
            if nb.node_id != n.node_id and not g.has_edge(n.node_id, nb.node_id):
                w = float(np.linalg.norm(nb.theta - n.theta))
                g.add_edge(n.node_id, nb.node_id, weight=w)

    # warmup
    g.dijkstra("n0", "n999")

    n_runs = 20
    t0 = time.time()
    for _ in range(n_runs):
        g.dijkstra("n0", "n999")
    elapsed_ms = (time.time() - t0) / n_runs * 1000
    print(f"\n  Dijkstra (1000 nodes, {g.num_edges} edges): {elapsed_ms:.2f} ms")
    assert elapsed_ms < 100


def test_balltree_rebuild_perf():
    """加 1000 节点后第一次 nearest_node 触发 rebuild, 应 < 200 ms."""
    g = ExplorationGraph()
    for i in range(1000):
        g.add_node(_node(f"n{i}", np.random.randn(6) * 0.3))
    t0 = time.time()
    g.nearest_node(np.zeros(6))
    elapsed_ms = (time.time() - t0) * 1000
    print(f"\n  BallTree build (1000 nodes): {elapsed_ms:.2f} ms")
    assert elapsed_ms < 200


# ---------------------------------------------------------------------- stats


def test_stats():
    g = ExplorationGraph()
    g.add_node(_node("a", np.zeros(6), gain=2.0))
    g.add_node(_node("b", np.zeros(6), gain=10.0))
    g.add_node(_node("iso", np.zeros(6), gain=5.0))
    g.add_edge("a", "b", weight=1.0)
    s = g.stats()
    assert s["num_nodes"] == 3
    assert s["num_edges"] == 1
    assert s["num_components"] == 2  # {a,b} 一个连通分量, {iso} 一个
    assert s["max_gain"] == 10.0
