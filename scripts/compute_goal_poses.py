"""compute_goal_poses —— 移植 stomp_planner 的观测位姿优化器（per-seam .pkl 输入），只算 goal pose。

功能：给定一条焊缝的 per-seam `.pkl`（含 robot_pose / piece_pose / horizontal / seam_line /
seam_tangent / seam_limits，与 gt_overall `optimize_pose.py` 读的格式一致）和工件 USD，
完整复用本项目 `stomp_planner/` 的进化策略优化器（视觉代价用同款 isaaclab/warp `raycast_mesh`，
碰撞用接触传感器，IK 用解析 UR12e），计算覆盖整条焊缝的**观测位姿序列**（goal pose 序列），
可选用 `--save` 落盘。可视化交给独立脚本 `scripts/viz_goal_poses_isaacsim.py`（读 --save 落盘结果）。

与 gt_overall/optimize_pose.py 的区别（详见 docs 计划）：
  - 本脚本处理「单个 seam pkl」，去掉多 GPU 文件夹批处理 / 黑名单 / 超时 marker；
  - 改为内存返回 dict（+ 可选 `--save` 落盘）；
  - 优化器内核 / 场景 / 代价 / 输入口径与 stomp_planner 副本**完全一致**（不改一行）。

运行（env_isaaclab）：
  python scripts/compute_goal_poses.py --headless \
    --seam-pkl "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_25.pkl" \
    --save /tmp/goal_poses/goal_poses.pkl
  （--usd 默认取 pkl 同目录的 *_part.usd；落盘后用 viz_goal_poses_isaacsim.py 可视化）
"""

import os
import sys
import argparse

# ---- 第 1 步：必须最先启动 Isaac Sim App（与 stomp_planner/optimize_pose.py 同模式）----
from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOMP_DIR = os.path.join(PROJECT_ROOT, "stomp_planner")

parser = argparse.ArgumentParser(description="移植 stomp_planner 观测位姿优化器并用 Isaac Sim 可视化")
parser.add_argument("--seam-pkl", type=str, required=True, help="per-seam .pkl 路径（gt_overall 格式）")
parser.add_argument("--usd", type=str, default=None, help="工件 USD；默认取 pkl 同目录 *_part.usd")
parser.add_argument("--num-batches", type=int, default=None,
                    help="进化策略并行批数（config_pose.num_batches，默认 8）；与 --num-randoms-new 共同决定 num_envs")
parser.add_argument("--num-randoms-new", type=int, default=None,
                    help="每批新采样数（config_pose.num_randoms_new，默认 100）；"
                         "num_envs = num_batches*(num_randoms_new+num_randoms_old)")
parser.add_argument("--save", type=str, default=None, help="可选：把 goal poses 存到该 .pkl")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
device = args_cli.device

# ---- 第 2 步：App 启动后再 import（torch / isaaclab / stomp_planner 模块）----
import pickle
import numpy as np
import torch

if STOMP_DIR not in sys.path:
    sys.path.insert(0, STOMP_DIR)  # 让 config_pose / optimizer_pose / scene_pose / robot.* 可被 import

from config_pose import ConfigurationPose as Configuration   # noqa: E402
from optimizer_pose import OptimizerPose as Optimizer         # noqa: E402
from scene_pose import ScenePose as Scene                     # noqa: E402


def compute_goal_poses(cfg, scene, seam_data, device="cuda"):
    """忠实复刻 stomp_planner/optimize_pose.py:OptimizePose.process 的 per-seam 主体。

    Args:
        cfg:        ConfigurationPose
        scene:      ScenePose（已构造）
        seam_data:  per-seam pkl 解出的 dict（robot_pose/piece_pose/horizontal/seam_*）
    Returns:
        list[dict]，每个 robot_pose（通常 M=1）一项：
            cam_pose      : (K, B, 7)  观测位姿，piece/mesh 系，四元数 wxyz；B=覆盖焊缝的位姿序列长度
            joints        : (K, B, 6)  对应 IK 关节角
            start_pts/end_pts : (K, B) 每个位姿覆盖的焊缝点区间
            robot_pose_rel: (7,)       reset 返回的相对位姿
        无解（solve 返回 None）则该项被跳过。
    """
    robot_pose = torch.as_tensor(seam_data["robot_pose"], dtype=torch.float, device=device)      # (M, 7)
    piece_pose = torch.as_tensor(seam_data["piece_pose"], dtype=torch.float, device=device)      # (M, 7)
    seam_line = torch.as_tensor(seam_data["seam_line"], dtype=torch.float, device=device)        # (N, 3)
    seam_tangent = torch.as_tensor(seam_data["seam_tangent"], dtype=torch.float, device=device)  # (N, 3)
    seam_limits = torch.as_tensor(seam_data["seam_limits"], dtype=torch.float, device=device)    # (N, 2, 3)
    horizontal = torch.as_tensor(seam_data["horizontal"], dtype=torch.int, device=device)        # (M,)

    results = []
    total = robot_pose.shape[0]
    for i in range(total):
        print(f"[compute_goal_poses] 第 {i + 1}/{total} 个 robot_pose 开始优化 …")
        # reset scene（与 gt_overall 调用签名完全一致）
        robot_pose_rel = scene.reset(robot_pose[i], int(horizontal[i]), piece_pose[i])
        optimizer = Optimizer(cfg=cfg, scene=scene, device=device)
        optimizer.resetSeamData(seam_line, seam_tangent, seam_limits)

        cam_pose, joints, start_pts, end_pts = optimizer.solve()
        if cam_pose is None:
            print(f"[compute_goal_poses] 第 {i + 1}/{total} 个无观测位姿解")
            continue
        results.append({
            "cam_pose": cam_pose.detach().clone(),
            "joints": joints.detach().clone(),
            "start_pts": start_pts.detach().clone(),
            "end_pts": end_pts.detach().clone(),
            "robot_pose_rel": robot_pose_rel.detach().clone(),
        })
        print(f"[compute_goal_poses] 第 {i + 1}/{total} 个优化成功："
              f"cam_pose {tuple(cam_pose.shape)} joints {tuple(joints.shape)}")
    return results


def main():
    seam_pkl = args_cli.seam_pkl
    if not os.path.isfile(seam_pkl):
        raise FileNotFoundError(f"seam pkl 不存在: {seam_pkl}")

    # 工件 USD：默认取 pkl 同目录的 *_part.usd
    usd_path = args_cli.usd
    if usd_path is None:
        d = os.path.dirname(seam_pkl)
        cands = [f for f in os.listdir(d) if f.endswith("_part.usd")]
        if not cands:
            raise FileNotFoundError(f"未在 {d} 找到 *_part.usd，请用 --usd 指定")
        usd_path = os.path.join(d, sorted(cands)[0])
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"工件 USD 不存在: {usd_path}")
    print(f"[main] seam pkl : {seam_pkl}")
    print(f"[main] 工件 USD : {usd_path}")

    with open(seam_pkl, "rb") as f:
        seam_data = pickle.load(f)

    # 构造 config + 场景（usd_path 必须在构造 ScenePose 前设好）
    cfg = Configuration()
    cfg.usd_path = usd_path
    cfg.pc_path = ""
    # 用 num_batches / num_randoms_new 覆盖（优化器按 num_batches×num_randoms 做 reshape，
    # 故 scene.num_envs 必须 = num_batches*(num_randoms_new+num_randoms_old)，不能任意指定）
    if args_cli.num_batches is not None:
        cfg.num_batches = args_cli.num_batches
    if args_cli.num_randoms_new is not None:
        cfg.num_randoms_new = args_cli.num_randoms_new
    cfg.num_randoms_all = cfg.num_randoms_new + cfg.num_randoms_old + 1
    cfg.num_envs = cfg.num_batches * (cfg.num_randoms_new + cfg.num_randoms_old)
    print(f"[main] num_batches={cfg.num_batches} num_randoms_new={cfg.num_randoms_new} "
          f"-> num_envs={cfg.num_envs}")
    scene = Scene(cfg, num_envs=cfg.num_envs, device=device)

    results = compute_goal_poses(cfg, scene, seam_data, device=device)
    if not results:
        print("[main] 没有任何 robot_pose 求得观测位姿解，退出。")
        simulation_app.close()
        return

    # 汇总打印
    for i, r in enumerate(results):
        K, Bn = r["cam_pose"].shape[:2]
        print(f"[main] robot_pose#{i}: {K} 个变体 × {Bn} 个覆盖位姿 (cam_pose {tuple(r['cam_pose'].shape)})")

    if args_cli.save:
        with open(args_cli.save, "wb") as f:
            pickle.dump([{k: v.cpu().numpy() for k, v in r.items()} for r in results], f)
        print(f"[main] 已保存 goal poses → {args_cli.save}")
        print("[main] 可视化： python scripts/viz_goal_poses_isaacsim.py "
              f"--seam-pkl '{seam_pkl}' --usd '{usd_path}' --save '{args_cli.save}'")

    simulation_app.close()


if __name__ == "__main__":
    main()
