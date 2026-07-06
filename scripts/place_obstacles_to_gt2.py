"""从 Scene pkl（scripts/demo_scene.py 的 demo_main 存的）产【边看边走】探索 GT。

与 place_obstacles_to_gt.py（读 place_obstacles 的 npz）不同，本脚本的数据源是 **Scene pkl**
（Scene.save 存的数据态，见 gt_gen/scene.py）。默认读
    /media/a/新加卷/tempt/4/scene1.pkl
从中取：工件 obj、已放障碍（ObstacleSpec）、机械臂起始关节角（scene.cur_cfg）、当前焊缝、
当前 init pose（工件↔base 摆放 T_workpiece_in_base）、goal 关节角（compute_goal_pose 的 joints）。

goal —— 【直接用关节角当目标】，不再算相机 goal pose：
  · 从 compute_goal_pose 的结果里取 joints[variant, goal_index]（base 系关节角）直接当规划目标；
  · 交给 generate_gt 的 goal_cfg，STOMP 后端步①(直达)/步②(P*) 走 plan_joint_single 直接规划到该
    关节角，不再解 IK / 换算焊枪尖等价位姿。generate_gt 内部按需 FK(goal_cfg) 得 goal_pose 供 NBV 用。

四个世界同 gt_gen.main_loop / scene.plan_explore_path：
  · h_truth（MESH，工件+障碍实体，按 workpiece_pose7 摆到 base 系）——全知教练；
  · h_expl（VOXEL 三态）——机械臂真正规划/执行；
  · truth_scene（base 系 trimesh，工件+障碍）——raycast 几何源；
  · vm（三态体素图 + 初始 FREE 圆柱冷启动立足）。

机械臂从 scene.cur_cfg 起步（作为 start_cfg 传入 generate_gt；缺省时 generate_gt 用 cfg.retract_config）。

产出：把这条 GT 作为一条 trajectory 追加进 scene.trajectories 后另存新 pkl（默认 <scene>_gt.pkl），
可另起干净进程 Scene.load 再 Open3DSceneVisualizer(scene).show_trajectory_isaacsim() 回放。

运行（本机 conda，无需 Isaac；须在未启动 SimulationApp 的干净进程）：
    conda run -n env_isaaclab --no-capture-output python -u scripts/place_obstacles_to_gt2.py \
        --scene '/media/a/新加卷/tempt/4/scene1.pkl' --variant 0 --goal-index 0
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

DEFAULT_SCENE = "/media/a/新加卷/tempt/4/scene1.pkl"


def goal_joints_from_scene(scene, variant, goal_index, goal_index_seq):
    """从 compute_goal_pose 的结果里直接取【关节角目标】joints[variant, goal_index]（不再算 goal pose）。

    scene.goal_poses[seam_id][goal_index_seq]['joints'] 形如 (K,B,DOF)：K=候选变体、B=观测位姿、DOF=6。
    返回 (goal_joints, dbg)：goal_joints=长度 DOF 的 list（base 系关节角，直接当 generate_gt 的 goal_cfg）；
    dbg 记 variant/goal_index/K/B 供打印。
    """
    res = scene.goal_poses[scene.seam_id][goal_index_seq]
    jt = np.asarray(res["joints"])                           # (K,B,DOF)
    K, B = jt.shape[:2]
    goal_joints = np.asarray(jt[variant, goal_index], float).tolist()
    dbg = dict(variant=variant, goal_index=goal_index, goal_index_seq=goal_index_seq, K=K, B=B, goal_joints=goal_joints)
    return goal_joints, dbg


def build_worlds(scene, include_obstacles=True):
    """由 Scene 重建 h_truth(MESH) / h_expl(VOXEL) / truth_scene(trimesh) / vm / cam。

    镜像 scene.plan_explore_path 的世界构建：工件 + 障碍实体按当前 init pose 的 workpiece_pose7
    摆到 base 系。返回 dict。
    """
    import trimesh as _trimesh
    import gt_gen.compat as _compat  # noqa: F401  warp shim（须在 import curobo 前）
    _compat.apply_trimesh_shim()

    from gt_gen import curobo_iface as ci
    from gt_gen.voxmap import build_roi_voxmap
    from gt_gen.sensor import load_camera_model, load_truth_scene
    from gt_gen.init_free import set_initial_free_cylinder, set_initial_free_box
    from curobo.geom.types import WorldConfig, Mesh as CuMesh
    from curobo.geom.sdf.world import CollisionCheckerType

    cfg = scene.cfg
    wp_pose7 = np.asarray(scene.cur_init_pose.workpiece_pose7, float).tolist()
    T = np.asarray(scene.cur_init_pose.T_workpiece_in_base, float)

    obs_tms = scene._obstacle_solid_trimeshes() if include_obstacles else []
    meshes = [CuMesh(name="workpiece", file_path=scene.workpiece_obj, pose=wp_pose7)]
    if obs_tms:
        merged = _trimesh.util.concatenate(obs_tms)
        meshes.append(CuMesh(name="obstacles",
                             vertices=np.asarray(merged.vertices, float).tolist(),
                             faces=np.asarray(merged.faces, np.int64).reshape(-1, 3).tolist(),
                             pose=wp_pose7))
    world = WorldConfig(mesh=meshes)

    print(f"[gt2] 建 h_truth（MESH，工件 + {len(obs_tms)} 障碍实体）...")
    h_truth = ci.init_curobo(cfg, world_model=world,
                             collision_checker_type=CollisionCheckerType.MESH,
                             position_threshold=0.05, rotation_threshold=0.5)
    print("[gt2] 建 h_expl（VOXEL 三态）...")
    h_expl = ci.init_curobo(cfg)

    work_mesh = load_truth_scene(scene.workpiece_obj, mesh_pose=wp_pose7)
    tms = [work_mesh]
    for tm in obs_tms:
        tmc = tm.copy(); tmc.apply_transform(T); tms.append(tmc)
    truth_scene = _trimesh.util.concatenate(tms) if len(tms) > 1 else work_mesh
    print(f"[gt2] truth_scene 顶点={len(truth_scene.vertices)} 面={len(truth_scene.faces)}")

    cam = load_camera_model(cfg)
    vm = build_roi_voxmap(cfg)
    method = cfg.init_free_method_for_init          # cylinder | box（default.yaml init_free.method_for_init）
    if method == "box":
        box_min, box_max = cfg.init_free_box_min_for_init, cfg.init_free_box_max_for_init
        n_free = set_initial_free_box(h_truth, vm, config=cfg, box_min=box_min, box_max=box_max)
        print(f"[gt2] 初始 FREE 空间=box {box_min}~{box_max} 体素={n_free}")
    else:
        n_free = set_initial_free_cylinder(h_truth, vm, config=cfg)
        print(f"[gt2] 初始 FREE 空间=圆柱 体素={n_free}")
    return dict(cfg=cfg, h_truth=h_truth, h_expl=h_expl, truth_scene=truth_scene,
                vm=vm, cam=cam, world=world)


def run(args):
    from gt_gen.scene import Scene
    from gt_gen.main_loop import generate_gt

    scene = Scene.load(args.scene)
    if scene.cur_init_pose is None:
        raise RuntimeError("pkl 未设当前 init pose（需 set_init_pose 后再 save）")
    if not scene.goal_poses.get(scene.seam_id):
        raise RuntimeError("pkl 无 goal pose（需 compute_goal_pose 后再 save）")

    W = build_worlds(scene, include_obstacles=not args.no_obstacles)

    goal_joints, dbg = goal_joints_from_scene(scene, args.variant, args.goal_index, args.goal_index_seq)
    print(f"[gt2] goal=关节角目标 joints#变体{dbg['variant']}/{dbg['K']} "
          f"位姿{dbg['goal_index']}/{dbg['B']}  goal_joints={np.round(dbg['goal_joints'],4)}")

    # 起点=scene.cur_cfg：作为 start_cfg 传入 generate_gt（无需再覆盖 retract_config）
    start = [float(v) for v in scene.cur_cfg]
    print(f"[gt2] 起始关节角(scene.cur_cfg)={np.round(start,4)}；开跑 generate_gt（边走边看）...")
    GT, status, info = generate_gt(
        W["h_truth"], W["h_expl"], W["vm"], W["truth_scene"], None,
        camera_model=W["cam"], world_plan=W["world"], goal_cfg=goal_joints, start_cfg=start)

    positions = np.asarray([np.asarray(q, float) for q in GT])
    print(f"[gt2] status={status}  GT 路点={len(positions)}  轮数={info['rounds']}  "
          f"P*长={info['P_len']}")
    if status != "reached":
        print(f"[gt2][warn] 未到达目标（status={status}）——仍存已走出的部分 GT 供检查")

    # 轨迹追加进 scene 并另存（供 show_trajectory_isaacsim 回放）
    entry = dict(positions=positions, status=status, goal_index=dbg["goal_index"],
                 variant=dbg["variant"], cur_joints=np.asarray(start, float),
                 goal_joints=np.asarray(goal_joints, float), goal_source="joint_target_K_variant",
                 info=info)
    scene.trajectories.setdefault(scene.seam_id, []).append(entry)
    out = args.out or (os.path.splitext(args.scene)[0] + "_gt.pkl")
    scene.save(out)
    print(f"\nPLACE_OBSTACLES_TO_GT2_{'OK' if status == 'reached' else 'PARTIAL'} "
          f"[status={status}, 路点={len(positions)}]  存盘: {out}")
    print(f"可视化（另起干净进程）：Scene.load('{out}') → "
          f"Open3DSceneVisualizer(scene).show_trajectory_isaacsim()")
    return status


def main():
    ap = argparse.ArgumentParser(description="读 Scene pkl 产边看边走 GT（相机当 ee_link 的 goal）")
    ap.add_argument("--scene", default=DEFAULT_SCENE, help="Scene pkl（demo_scene.demo_main 存）")
    ap.add_argument("--variant", type=int, default=0, help="joints 的变体 K（默认 0）")
    ap.add_argument("--goal-index", type=int, default=0, help="观测位姿的第几个合格解")
    ap.add_argument("--goal-index-seq", type=int, default=0, help="观测位姿序列里第几个作终点（默认 0，支持负索引）")
    ap.add_argument("--no-obstacles", action="store_true", help="真值世界不并入障碍（仅工件）")
    ap.add_argument("--out", default=None, help="GT 存盘 pkl；缺省 <scene>_gt.pkl")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
