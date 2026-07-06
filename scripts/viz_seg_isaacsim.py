"""独立【干净进程】：读 main_loop._dump_seg_isaacsim 落盘的执行段轨迹 + Scene pkl，用 **isaacsim**
回放机械臂走这些段——工件 + 障碍物 + 机械臂，**不含 voxmap**（对应 _debug_viz_seg 的 isaacsim 版）。

为何独立进程：show_scene_isaacsim 的 SimulationApp 必须在【未加载 warp】的干净进程里最先启动，不能与
跑 generate_gt(warp/curobo) 的进程同框（见 gt_gen/scene.py Scene.save/load 与
gt_gen/main_loop._dump_seg_isaacsim 说明）。故 generate_gt 那边只落盘轨迹，回放交给本脚本。

用法（另起干净进程，勿先 import warp/curobo）：
  python scripts/viz_seg_isaacsim.py --scene <scene.pkl> [--segs <dump.pkl>] [--seg-index N] [--fps 30]

  --scene     : Scene pkl（须已 set_init_pose → base 系，才画机械臂；同 place_obstacles_to_gt2 的 --scene）。
  --segs      : _dump_seg_isaacsim 落盘（默认 configs/_seg_dump_isaacsim.pkl，与 main_loop 默认一致）。
  --seg-index : 只看第几段（支持负索引）；默认 None=把所有段拼成一条连续轨迹整体回放。
  --fps       : 回放帧率。
"""
import argparse
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_SEGS = os.path.join(ROOT, "configs", "_seg_dump_isaacsim.pkl")
DEFAULT_SCENE = "/media/a/新加卷/tempt/4/scene1.pkl"


def main():
    ap = argparse.ArgumentParser(
        description="isaacsim 回放 generate_gt 落盘的执行段 seg（工件+障碍+机械臂，无 voxmap）")
    ap.add_argument("--scene", default=DEFAULT_SCENE, help="Scene pkl（须已 set_init_pose）")
    ap.add_argument("--segs", default=DEFAULT_SEGS, help="_dump_seg_isaacsim 落盘 pkl")
    ap.add_argument("--seg-index", type=int, default=None, help="只看第几段；默认所有段拼接回放")
    ap.add_argument("--fps", type=int, default=30, help="回放帧率")
    ap.add_argument("--goal-variant", type=int, default=0, help="画 goal 视锥用 cam_pose 变体 K（默认 0）")
    ap.add_argument("--headless", action="store_true", help="无显示器自检（spawn 跑几帧即退）")
    args = ap.parse_args()

    with open(args.segs, "rb") as f:
        segs = pickle.load(f)
    if not segs:
        raise RuntimeError(f"落盘无段可回放：{args.segs}（generate_gt 跑过、且 rnd 走到过 seg 落盘点吗？）")

    if args.seg_index is not None:
        n = len(segs)
        if not (-n <= args.seg_index < n):
            raise IndexError(f"--seg-index 越界：{args.seg_index}，共 {n} 段")
        chosen = [segs[args.seg_index]]
    else:
        chosen = segs

    parts = []
    for k, e in enumerate(chosen):
        p = np.asarray(e["positions"], float)
        parts.append(p if k == 0 else p[1:])                 # 拼接相邻段时去掉重复衔接点
    traj = np.vstack(parts)
    print(f"[viz-seg] 段数={len(chosen)}/{len(segs)}  回放路点={len(traj)}  "
          f"rnd={[e.get('rnd') for e in chosen]}")

    # 干净进程里 Scene.load（不碰 warp），show_scene_isaacsim 内部再最先启动 SimulationApp。
    from gt_gen.scene import Scene
    from gt_gen.scene_viz import Open3DSceneVisualizer

    scene = Scene.load(args.scene)
    if scene.cur_init_pose is None:
        raise RuntimeError("Scene 未设 init pose（base 系才画机械臂）：需 set_init_pose 后再 save")
    Open3DSceneVisualizer(scene).show_scene_isaacsim(
        headless=args.headless, trajectory=traj, fps=args.fps, goal_variant=args.goal_variant)


if __name__ == "__main__":
    main()
