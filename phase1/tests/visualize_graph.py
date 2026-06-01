"""M1.5 视觉验证: 用 2D 平面演示图的增量构建 + 各种查询.

每一步保存一张 PNG, 文件名按步骤排序:
    01_init.png
    02_add_5_nodes.png
    03_connect_knn.png
    04_query_nearest.png
    05_query_within.png
    06_dijkstra_short_path.png
    07_dijkstra_blocked.png
    08_update_gain.png
    09_frontier_nodes.png
    10_growing_50_nodes.png

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/visualize_graph.py

默认存到 项目下 tmp/m1_5_graph/ (gitignored).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from phase1.graph import ExplorationGraph, GraphNode

# 中文字体
def _find_cjk_font():
    candidates = [f.name for f in fm.fontManager.ttflist]
    for kw in ["noto sans cjk", "source han", "yahei", "wenquanyi", "fallback"]:
        for name in candidates:
            if kw in name.lower():
                return name
    return None
_cjk = _find_cjk_font()
if _cjk:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [_cjk] + list(plt.rcParams["font.sans-serif"])
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------- 绘图


def make_2d_node(name: str, x: float, y: float, gain: float = 0.0) -> GraphNode:
    """为可视化方便, 用 2D theta. 实际工程是 6D, 行为一致."""
    return GraphNode(
        node_id=name,
        theta=np.array([x, y], dtype=np.float64),
        ee_pos_world=np.array([x, y, 0.0]),
        ee_dir_world=np.array([0.0, 0.0, 1.0]),
        cycle_added=0,
        last_gain=gain,
    )


def plot_graph(
    g: ExplorationGraph,
    title: str,
    out_path: Path,
    *,
    highlight_nodes: list[str] | None = None,
    highlight_edges: list[tuple[str, str]] | None = None,
    highlight_color: str = "#FF3B3B",
    target: np.ndarray | None = None,
    target_label: str = "目标",
    radius: float | None = None,
    color_by_gain: bool = False,
    legend_extra: list[str] | None = None,
    info_text: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    # ---- 边 ----
    for a, b in g._g.edges:
        na = g.get_node(a); nb = g.get_node(b)
        is_highlight = bool(highlight_edges) and (
            (a, b) in highlight_edges or (b, a) in highlight_edges
        )
        ax.plot(
            [na.theta[0], nb.theta[0]], [na.theta[1], nb.theta[1]],
            color=highlight_color if is_highlight else "#A0A0A0",
            linewidth=2.5 if is_highlight else 1.0,
            alpha=0.9 if is_highlight else 0.6,
            zorder=2 if is_highlight else 1,
        )

    # ---- 节点 ----
    for node in g.all_nodes():
        x, y = node.theta
        is_h = bool(highlight_nodes) and node.node_id in highlight_nodes
        if color_by_gain:
            # 颜色映射: 0 = 浅蓝, max = 红
            max_g = max((n.last_gain for n in g.all_nodes()), default=1.0) or 1.0
            t = node.last_gain / max_g
            color = (t, 0.3 + 0.5 * (1 - t), 1.0 - t)  # 红蓝渐变
            edge = "black" if is_h else "none"
            size = 200 if is_h else 110
        else:
            color = highlight_color if is_h else "#3070C0"
            edge = "black" if is_h else "white"
            size = 220 if is_h else 130
        ax.scatter([x], [y], s=size, c=[color], edgecolors=edge,
                   linewidths=2 if is_h else 0.7, zorder=3)
        ax.annotate(node.node_id, (x, y),
                    textcoords="offset points", xytext=(7, 7),
                    fontsize=9, color="black", zorder=4)

    # ---- 目标点 ----
    if target is not None:
        ax.scatter([target[0]], [target[1]], marker="*", s=320,
                   c="orange", edgecolors="black", linewidths=1.2, zorder=5,
                   label=target_label)
        if radius is not None:
            circ = mpatches.Circle(target[:2], radius, fill=False,
                                    color="orange", linewidth=2,
                                    linestyle="--", zorder=4)
            ax.add_patch(circ)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("theta[0]")
    ax.set_ylabel("theta[1]")

    if info_text:
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                ha="left", va="top", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.85, pad=4,
                          edgecolor="gray"))

    if target is not None or color_by_gain:
        handles = []
        if target is not None:
            handles.append(plt.Line2D([], [], marker="*", color="orange",
                                       linestyle="", markersize=14,
                                       markeredgecolor="black",
                                       label=target_label))
        if color_by_gain:
            handles.append(plt.Line2D([], [], marker="o", color="red",
                                       linestyle="", markersize=10,
                                       label="高 gain"))
            handles.append(plt.Line2D([], [], marker="o", color="blue",
                                       linestyle="", markersize=10,
                                       label="低 gain"))
        if legend_extra:
            for line in legend_extra:
                handles.append(plt.Line2D([], [], color=highlight_color,
                                           linewidth=3, label=line))
        ax.legend(handles=handles, loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- 主流程


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _default = (Path(__file__).resolve().parent.parent.parent
                / "tmp" / "m1_5_graph")
    parser.add_argument("--output-dir", default=str(_default))
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if args.clear and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_dir}")

    g = ExplorationGraph()

    # =========== Step 1: 空图 ==========
    plot_graph(g, "Step 1  空图", out_dir / "01_init.png",
               info_text="num_nodes=0\nnum_edges=0")
    print("01 init")

    # =========== Step 2: 加 5 个节点 ==========
    initial = [
        ("a", -0.4, -0.3),
        ("b",  0.0,  0.5),
        ("c",  0.6,  0.0),
        ("d",  0.3, -0.5),
        ("e", -0.5,  0.4),
    ]
    for name, x, y in initial:
        g.add_node(make_2d_node(name, x, y))
    plot_graph(g, "Step 2  添加 5 个节点 (无边)", out_dir / "02_add_5_nodes.png",
               info_text=f"num_nodes={g.num_nodes}\nnum_edges={g.num_edges}")
    print(f"02 add_5_nodes  nodes={g.num_nodes}")

    # =========== Step 3: 每个节点连最近 2 个邻居 ==========
    for n in list(g.all_nodes()):
        nbs = g.k_nearest(n.theta, k=3)  # 第 0 个是自己
        for nb in nbs:
            if nb.node_id != n.node_id and not g.has_edge(n.node_id, nb.node_id):
                w = float(np.linalg.norm(nb.theta - n.theta))
                g.add_edge(n.node_id, nb.node_id, weight=w)
    plot_graph(g, "Step 3  每个节点连最近 2 个邻居", out_dir / "03_connect_knn.png",
               info_text=f"num_nodes={g.num_nodes}\nnum_edges={g.num_edges}")
    print(f"03 connect_knn  edges={g.num_edges}")

    # =========== Step 4: nearest_node 查询 ==========
    target = np.array([0.2, 0.1])
    near = g.nearest_node(target)
    plot_graph(
        g, f"Step 4  nearest_node({target.tolist()}) = {near.node_id!r}",
        out_dir / "04_query_nearest.png",
        target=target, target_label="查询点",
        highlight_nodes=[near.node_id],
        info_text=f"target = {target.tolist()}\nnearest = {near.node_id!r}\n"
                  f"distance = {np.linalg.norm(near.theta - target):.3f}",
    )
    print(f"04 query_nearest  → {near.node_id}")

    # =========== Step 5: nodes_within 查询 ==========
    radius = 0.45
    within = g.nodes_within(target, radius=radius)
    plot_graph(
        g, f"Step 5  nodes_within(target, radius={radius})  → {len(within)} 个",
        out_dir / "05_query_within.png",
        target=target, target_label=f"查询点 (r={radius})",
        radius=radius,
        highlight_nodes=[n.node_id for n in within],
        info_text=f"radius = {radius}\nfound: " +
                  ", ".join(repr(n.node_id) for n in within),
    )
    print(f"05 query_within  → {[n.node_id for n in within]}")

    # =========== Step 6: dijkstra 最短路径 (a → c) ==========
    path = g.dijkstra("a", "c")
    if path is None:
        plot_graph(g, "Step 6  dijkstra(a → c) = 不可达",
                   out_dir / "06_dijkstra_short_path.png")
    else:
        path_edges = [(path[i].node_id, path[i + 1].node_id)
                      for i in range(len(path) - 1)]
        path_str = " → ".join(n.node_id for n in path)
        plot_graph(
            g, f"Step 6  dijkstra(a → c) = {path_str}",
            out_dir / "06_dijkstra_short_path.png",
            highlight_nodes=[n.node_id for n in path],
            highlight_edges=path_edges,
            highlight_color="#E74C3C",
            info_text=f"length = {g.dijkstra_length('a', 'c'):.3f}\n"
                      f"path: {path_str}",
            legend_extra=["最短路径"],
        )
    print(f"06 dijkstra_a_to_c  → {[n.node_id for n in path] if path else 'None'}")

    # =========== Step 7: 删一条关键边后再 dijkstra ==========
    # 把 path 中第一条边删了, 看是否还能找到其他路径
    if path and len(path) > 2:
        kill_a, kill_b = path[0].node_id, path[1].node_id
        g.remove_edge(kill_a, kill_b)
        path2 = g.dijkstra("a", "c")
        if path2 is None:
            plot_graph(
                g, f"Step 7  删除边 ({kill_a}-{kill_b}) 后, "
                   "dijkstra(a → c) = 不可达",
                out_dir / "07_dijkstra_blocked.png",
                highlight_nodes=["a", "c"],
                info_text=f"删除边 ({kill_a}-{kill_b})\n"
                          "新路径: 不可达",
            )
        else:
            path2_edges = [(path2[i].node_id, path2[i + 1].node_id)
                           for i in range(len(path2) - 1)]
            path2_str = " → ".join(n.node_id for n in path2)
            plot_graph(
                g, f"Step 7  删除边 ({kill_a}-{kill_b}) 后, 改走 {path2_str}",
                out_dir / "07_dijkstra_blocked.png",
                highlight_nodes=[n.node_id for n in path2],
                highlight_edges=path2_edges,
                highlight_color="#E67E22",
                info_text=f"删除边 ({kill_a}-{kill_b})\n"
                          f"新路径: {path2_str}\n"
                          f"length = {g.dijkstra_length('a', 'c'):.3f}",
                legend_extra=["新最短路径"],
            )
        # 恢复
        w = float(np.linalg.norm(g.get_node(kill_a).theta - g.get_node(kill_b).theta))
        g.add_edge(kill_a, kill_b, weight=w)
    print("07 dijkstra_blocked")

    # =========== Step 8: update_node_gain ==========
    np.random.seed(42)
    for n in g.all_nodes():
        # 离原点越远 gain 越高 (模拟"远处空间还没探完")
        g_val = float(np.linalg.norm(n.theta) * 30)
        g.update_node_gain(n.node_id, g_val, cycle=0)
    plot_graph(
        g, "Step 8  update_node_gain (颜色 = gain)",
        out_dir / "08_update_gain.png",
        color_by_gain=True,
        info_text="\n".join(f"{n.node_id}: {n.last_gain:.1f}"
                              for n in g.all_nodes()),
    )
    print("08 update_gain")

    # =========== Step 9: frontier_nodes 查询 ==========
    threshold = 18.0
    fronts = g.frontier_nodes(gain_threshold=threshold)
    plot_graph(
        g, f"Step 9  frontier_nodes(threshold={threshold}) → {len(fronts)} 个",
        out_dir / "09_frontier_nodes.png",
        color_by_gain=True,
        highlight_nodes=[n.node_id for n in fronts],
        info_text=f"threshold = {threshold}\nfrontiers (按 gain 降序):\n" +
                  "\n".join(f"  {n.node_id}: {n.last_gain:.1f}"
                            for n in fronts),
    )
    print(f"09 frontier_nodes  → {[n.node_id for n in fronts]}")

    # =========== Step 10: 增量加 50 个节点 (模拟 cycle 累积) ==========
    rng = np.random.default_rng(1)
    for i in range(50):
        x = rng.uniform(-0.8, 0.8); y = rng.uniform(-0.8, 0.8)
        node = make_2d_node(f"r{i}", x, y, gain=float(rng.uniform(0, 30)))
        g.add_node(node)
        # 连最近 2 个 (排除自己)
        for nb in g.k_nearest(node.theta, k=3):
            if nb.node_id != node.node_id and not g.has_edge(node.node_id, nb.node_id):
                w = float(np.linalg.norm(nb.theta - node.theta))
                g.add_edge(node.node_id, nb.node_id, weight=w)
    plot_graph(
        g, f"Step 10  累计 {g.num_nodes} 节点 / {g.num_edges} 边 (+50 增量)",
        out_dir / "10_growing_50_nodes.png",
        color_by_gain=True,
        info_text=f"num_nodes = {g.num_nodes}\nnum_edges = {g.num_edges}\n"
                  f"max gain = {max(n.last_gain for n in g.all_nodes()):.1f}",
    )
    print(f"10 growing  nodes={g.num_nodes}, edges={g.num_edges}")

    # ============ 总结 ============
    print()
    print(f"[done] 10 PNGs to {out_dir}")
    print(f"  xdg-open {out_dir}")
    print()
    print(f"[final stats]  {g.stats()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
