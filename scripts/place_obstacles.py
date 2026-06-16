"""沿「关键 link 扫掠走廊」自动放置障碍物，并用三条件验证（见 docs/障碍物位置.md）。

流程：用工件 obj 当已知障碍，规划 retract→焊缝目标的【默认无障碍轨迹】→ 沿轨迹算各关键 link 的扫掠走廊
→ 每个关键 link 产 1 个场景（障碍类型未用优先随机），上层 search_placement 反复试放并判定三条件
（①默认路径碰撞 ②仍有绕行解 ③绕行明显不同），失败就改尺寸/角度/位置重试至 N 次。

每个成功场景存一个 npz（默认轨迹 + 绕行轨迹 + 障碍原语 + 工件信息），供 viz_placed_obstacle_isaacsim.py 回放。

运行（本机 conda）：
    conda run -n env_isaaclab --no-capture-output python -u scripts/place_obstacles.py \
        --seam /media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_22.pkl --seed 0
M（每场景障碍数）、N（重试次数）、关键 link 等在 configs/default.yaml 的 obstacle_placement 段。
"""
import argparse
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

DEFAULT_SEAM = ("/media/a/新加卷/hanfeng/segment_sub_output/"
                "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_22.pkl")


def _serialize_prims(prims):
    """Box/Tube → 可存 npz 的 dict 列表。"""
    from gt_gen import obstacles as ob
    out = []
    for p in prims:
        if isinstance(p, ob.Box):
            out.append(dict(kind="box", name=p.name, dims=list(map(float, p.dims)),
                            pose=list(map(float, p.pose)), color=list(map(float, p.color))))
        else:
            out.append(dict(kind="tube", name=p.name, radius=float(p.radius), height=float(p.height),
                            pose=list(map(float, p.pose)), color=list(map(float, p.color))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", default=DEFAULT_SEAM)
    ap.add_argument("--offset_cm", type=float, default=10.0)
    ap.add_argument("--out_dir", default="/tmp/placed_obstacles")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--links", default=None, help="逗号分隔覆盖 key_links")
    ap.add_argument("--types", default=None, help="逗号分隔覆盖 obstacle_types")
    ap.add_argument("--max_attempts", type=int, default=None, help="覆盖 N（单 link 重试次数）")
    args = ap.parse_args()

    import random
    import torch
    from gt_gen.config import load_config
    from gt_gen import curobo_iface as ci
    from gt_gen import obstacle_placement as opl
    from curobo.geom.types import WorldConfig, Mesh
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.types.math import Pose
    from plan_seam import find_obj, seam_ee_pose

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    cfg = load_config()
    # 配置覆盖（命令行优先）
    if args.links:
        cfg.raw.setdefault("obstacle_placement", {})["key_links"] = args.links.split(",")
    if args.types:
        cfg.raw.setdefault("obstacle_placement", {})["obstacle_types"] = args.types.split(",")
    if args.max_attempts is not None:
        cfg.raw.setdefault("obstacle_placement", {})["max_attempts"] = args.max_attempts
    op = cfg.obstacle_placement

    d = pickle.load(open(args.seam, "rb"))
    goal_pose, seam_dbg = seam_ee_pose(d, offset_m=args.offset_cm / 100.0)
    obj_path = find_obj(args.seam)
    robot_pos = np.asarray(d["robot_pose"][0], float)
    piece_pose = np.asarray(d["piece_pose"][0], float)

    print("seam   :", args.seam)
    print("obj    :", obj_path)
    print("关键 link:", op["key_links"], " M=", op["max_per_scene"], " N=", op["max_attempts"])

    # 工件 mesh @ 机械臂基座系（同 plan_seam）
    def _pose(p7):
        return Pose(position=torch.tensor([p7[:3]], dtype=torch.float32, device="cuda"),
                    quaternion=torch.tensor([p7[3:7]], dtype=torch.float32, device="cuda"))
    T_robot_piece = _pose(robot_pos).inverse().multiply(_pose(piece_pose))
    mesh_pose = T_robot_piece.get_pose_vector()[0].detach().cpu().numpy().tolist()
    workpiece = Mesh(name="workpiece", file_path=obj_path, pose=mesh_pose)
    world0 = WorldConfig(mesh=[workpiece])

    print("\n== init_curobo（MESH 世界，仅工件） ==")
    h = ci.init_curobo(cfg, world_model=world0, collision_checker_type=CollisionCheckerType.MESH,
                       position_threshold=0.05, rotation_threshold=0.5)
    retract = cfg.retract_config
    metric = ci.free_pose_metric(h, free_rot=(0,))

    print("== 规划默认无障碍轨迹 retract→goal ==")
    traj_default = opl.plan_default_traj(h, retract, goal_pose, metric, cfg.plan_max_attempts)
    if traj_default is None:
        print("PLACE_OBSTACLES_FAIL: 默认轨迹规划失败（无障碍都到不了 goal）")
        return
    print(f"默认轨迹点数: {traj_default.shape}")

    print("== 沿默认轨迹算各关键 link 扫掠球 ==")
    per_wp, origin = opl.compute_link_sweep(cfg, traj_default, op["key_links"])
    print("有扫掠球的 link:", {k: per_wp[k].shape for k in per_wp})

    print("\n== 逐 link 放障碍（每 link 1 场景） ==")
    scenes = opl.generate_scenes(h, workpiece, per_wp, origin, traj_default,
                                 retract, goal_pose, metric, cfg, rng)

    os.makedirs(args.out_dir, exist_ok=True)
    robot_usd = cfg.robot_cfg["robot_cfg"]["kinematics"].get("usd_path", "")
    n_ok = 0
    used_types = []
    for i, sc in enumerate(scenes):
        if sc["ok"]:
            n_ok += 1
            used_types.append(sc["otype"])
            out = os.path.join(args.out_dir, f"scene_{i:02d}_{sc['link']}_{sc['otype']}.npz")
            detours = sc["detour_trajs"]                       # list[ndarray]：多条互不相同的绕行解
            np.savez(
                out,
                positions=traj_default,
                detour_positions=detours[0],                  # 主绕行解（首条，IK 误差最小）：viz 默认回放它
                detour_positions_all=np.array(detours, dtype=object),  # 全部候选（供存多份 GT / 逐条核对）
                n_detour=len(detours),
                joint_names=np.array(cfg.joint_names),
                retract=np.array(retract),
                obstacle_prims=np.array(_serialize_prims(sc["prims"]), dtype=object),
                anchor=np.array(sc["anchor"], float),
                link=sc["link"], otype=sc["otype"],
                piece_pose_to_robot=np.array(mesh_pose, float),
                obj_path=obj_path,
                robot_usd=robot_usd,
                seam_mid=np.array(seam_dbg["seam_mid"], float),
                seam_bisector=np.array(seam_dbg["seam_bisector"], float),
                dt=0.02,
            )
            print(f"  [{i:02d}] {sc['link']:<22} {sc['otype']:<16} ✓ 第{sc['attempt']}次成功 "
                  f"碰撞点={sc['n_bad']} 绕行偏差={sc['dist']:.3f}rad 绕行解={len(detours)}条 "
                  f"→ {os.path.basename(out)}")
        else:
            print(f"  [{i:02d}] {sc['link']:<22} {sc['otype']:<16} ✗ 试{sc['attempt']}次失败 "
                  f"(末次原因={sc['last_reason']})")

    print(f"\n成功 {n_ok}/{len(scenes)} 个场景；类型覆盖: {sorted(set(used_types))} "
          f"({len(set(used_types))} 种)")
    print("npz 输出目录:", args.out_dir)
    print("PLACE_OBSTACLES_OK")


if __name__ == "__main__":
    main()
