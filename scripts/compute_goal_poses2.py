"""compute_goal_poses2 —— compute_goal_poses 的【去 Isaac】版：用 cuRobo + warp + trimesh 替代
isaacsim / isaaclab，在不启动 Isaac Sim 的情况下产出同样语义的观测位姿（goal pose）序列。

与 scripts/compute_goal_poses.py 的区别（见 docs/compute_goal_poses-说明.md 替代表 + 计划文档）：
  · 不 import isaaclab；不启动 AppLauncher / SimulationContext。
  · 场景换成 stomp_planner/scene_pose2.py 的 ScenePose2：
      - 遮挡 visionBlock：warp `mesh_query_ray`（替代 isaaclab.utils.warp.raycast_mesh）
      - 碰撞 getJoints：cuRobo RobotWorld 整臂碰撞球 vs 工件 mesh + 自碰撞 + 解析 field 锥
        （替代 ContactSensor + PhysX）
      - 工件几何：trimesh 读 *_part.obj（替代 omni.usd / UsdGeom）
  · 优化器 / IK / 代价 / 数学 / 配置（optimizer_pose / ik_cam_pose / config_pose / opt_*）原样复用，未改一行。

输出 dict schema 与原脚本完全一致（cam_pose / joints / start_pts / end_pts / robot_pose_rel），
可被同样的下游/可视化读取。

运行（env_isaaclab，无需 --headless / Isaac 相关参数）：
  python scripts/compute_goal_poses2.py \
    --seam-pkl "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_xxx_part/seam_25.pkl" \
    --save /tmp/goal_poses/goal_poses2.pkl
  （--obj 默认取 pkl 同目录的 *_part.obj）
"""

import os
import sys
import types
import argparse
import pickle

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOMP_DIR = os.path.join(PROJECT_ROOT, "stomp_planner")
for _p in (PROJECT_ROOT, STOMP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 关键：optimizer_pose.py 顶层 `from scene_pose import ScenePose as Scene` 仅用作类型注解，
# 但会连带 import isaaclab。注入一个轻量 stub 顶掉它（不改原文件），其 ScenePose 仅是占位符。
_stub = types.ModuleType("scene_pose")
class ScenePose:  # noqa: E742  仅供 optimizer_pose 的类型注解，运行期不被使用
    pass
_stub.ScenePose = ScenePose
sys.modules.setdefault("scene_pose", _stub)

from config_pose import ConfigurationPose as Configuration   # noqa: E402
from optimizer_pose import OptimizerPose as Optimizer         # noqa: E402
from scene_pose2 import ScenePose2 as Scene                   # noqa: E402


def compute_goal_poses(cfg, scene, seam_data, device="cuda"):
    """忠实复刻 compute_goal_poses.py 的 per-seam 主体（仅 scene 实现不同）。"""
    robot_pose = torch.as_tensor(seam_data["robot_pose"], dtype=torch.float, device=device)      # (M, 7)
    piece_pose = torch.as_tensor(seam_data["piece_pose"], dtype=torch.float, device=device)      # (M, 7)
    seam_line = torch.as_tensor(seam_data["seam_line"], dtype=torch.float, device=device)        # (N, 3)
    seam_tangent = torch.as_tensor(seam_data["seam_tangent"], dtype=torch.float, device=device)  # (N, 3)
    seam_limits = torch.as_tensor(seam_data["seam_limits"], dtype=torch.float, device=device)    # (N, 2, 3)
    horizontal = torch.as_tensor(seam_data["horizontal"], dtype=torch.int, device=device)        # (M,)

    results = []
    total = robot_pose.shape[0]
    for i in range(total):
        print(f"[compute_goal_poses2] 第 {i + 1}/{total} 个 robot_pose 开始优化 …")
        robot_pose_rel = scene.reset(robot_pose[i], int(horizontal[i]), piece_pose[i])
        optimizer = Optimizer(cfg=cfg, scene=scene, device=device)
        optimizer.resetSeamData(seam_line, seam_tangent, seam_limits)

        cam_pose, joints, start_pts, end_pts = optimizer.solve()
        if cam_pose is None:
            print(f"[compute_goal_poses2] 第 {i + 1}/{total} 个无观测位姿解")
            continue
        results.append({
            "cam_pose": cam_pose.detach().clone(),
            "joints": joints.detach().clone(),
            "start_pts": start_pts.detach().clone(),
            "end_pts": end_pts.detach().clone(),
            "robot_pose_rel": robot_pose_rel.detach().clone(),
        })
        print(f"[compute_goal_poses2] 第 {i + 1}/{total} 个优化成功："
              f"cam_pose {tuple(cam_pose.shape)} joints {tuple(joints.shape)}")
    return results


def main():
    parser = argparse.ArgumentParser(description="去 Isaac 的观测位姿优化器（cuRobo + warp + trimesh）")
    parser.add_argument("--seam-pkl", type=str, required=True, help="per-seam .pkl 路径（gt_overall 格式）")
    parser.add_argument("--obj", type=str, default=None, help="工件 obj；默认取 pkl 同目录 *_part.obj")
    parser.add_argument("--robot-cfg", type=str, default=None, help="cuRobo 机器人 cfg；默认取 gt_gen default.yaml")
    parser.add_argument("--num-batches", type=int, default=None,
                        help="进化策略并行批数（config_pose.num_batches，默认 8）")
    parser.add_argument("--num-randoms-new", type=int, default=None,
                        help="每批新采样数（默认 100）。num_envs = num_batches*(num_randoms_new+num_randoms_old)")
    parser.add_argument("--save", type=str, default=None, help="可选：把 goal poses 存到该 .pkl")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备（默认 cuda）")
    args_cli = parser.parse_args()
    device = args_cli.device

    seam_pkl = args_cli.seam_pkl
    if not os.path.isfile(seam_pkl):
        raise FileNotFoundError(f"seam pkl 不存在: {seam_pkl}")

    # 工件 obj：默认取 pkl 同目录的 *_part.obj（与 USD 同几何，非 watertight）
    obj_path = args_cli.obj
    if obj_path is None:
        d = os.path.dirname(seam_pkl)
        cands = [f for f in os.listdir(d) if f.endswith("_part.obj")]
        if not cands:
            raise FileNotFoundError(f"未在 {d} 找到 *_part.obj，请用 --obj 指定")
        obj_path = os.path.join(d, sorted(cands)[0])
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"工件 obj 不存在: {obj_path}")
    print(f"[main] seam pkl : {seam_pkl}")
    print(f"[main] 工件 obj : {obj_path}")

    with open(seam_pkl, "rb") as f:
        seam_data = pickle.load(f)

    cfg = Configuration()
    cfg.usd_path = ""
    cfg.pc_path = ""
    if args_cli.num_batches is not None:
        cfg.num_batches = args_cli.num_batches
    if args_cli.num_randoms_new is not None:
        cfg.num_randoms_new = args_cli.num_randoms_new
    cfg.num_randoms_all = cfg.num_randoms_new + cfg.num_randoms_old + 1
    cfg.num_envs = cfg.num_batches * (cfg.num_randoms_new + cfg.num_randoms_old)
    print(f"[main] num_batches={cfg.num_batches} num_randoms_new={cfg.num_randoms_new} "
          f"-> num_envs={cfg.num_envs}")

    scene = Scene(cfg, num_envs=cfg.num_envs, device=device,
                  obj_path=obj_path, robot_cfg_path=args_cli.robot_cfg or "")

    results = compute_goal_poses(cfg, scene, seam_data, device=device)
    if not results:
        print("[main] 没有任何 robot_pose 求得观测位姿解，退出。")
        return

    for i, r in enumerate(results):
        K, Bn = r["cam_pose"].shape[:2]
        print(f"[main] robot_pose#{i}: {K} 个变体 × {Bn} 个覆盖位姿 (cam_pose {tuple(r['cam_pose'].shape)})")

    if args_cli.save:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.save)), exist_ok=True)
        with open(args_cli.save, "wb") as f:
            pickle.dump([{k: v.cpu().numpy() for k, v in r.items()} for r in results], f)
        print(f"[main] 已保存 goal poses → {args_cli.save}")


if __name__ == "__main__":
    main()
