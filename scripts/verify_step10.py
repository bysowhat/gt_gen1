"""Step 10 验证：完整主循环 generate_gt（goal 后退 standoff + 纯三态 voxel 探索世界）。

场景（参考 verify_step8.build_scene，但 goal 带 standoff、并额外建 h_expl(VOXEL)）：
  - 读 seam_22 → goal_pose 沿两面角平分方向后退 cfg.goal_standoff_m（10cm）；
  - h_truth：MESH 世界（工件 mesh），全知特权专家；
  - h_expl：VOXEL 世界（纯三态，无 mesh），机械臂真正执行的世界；
  - truth_scene：base 系 trimesh（raycast 几何源）；
  - vm：三态体素图 + 圆柱初始 FREE（冷启动立足之地，参数取自 default.yaml）。

跑 generate_gt 产出整条 GT，断言：
  ① status == reached；
  ② 保守性：GT 每对相邻路点整臂扫掠 ⊆ 最终 vm 的 FREE（0 保守违例；FREE 近似见注）；
  ③ 0 真值碰撞：每个路点 check_state(h_truth) feasible；
  ④ GT 终点末端位姿 ≈ goal（确认停在 standoff）。

运行：conda run -n env_isaaclab python scripts/verify_step10.py [--seam ...] [--viz]
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

SEAM = ("/media/a/新加卷/hanfeng/segment_sub_output/"
        "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_22.pkl")


def build_scene_step10(args):
    """建 h_truth(MESH) + h_expl(VOXEL) + truth_scene + vm(+圆柱FREE) + cam + standoff goal。"""
    from gt_gen.config import load_config
    from gt_gen import curobo_iface as ci
    from gt_gen.voxmap import build_roi_voxmap
    from gt_gen.sensor import load_camera_model, load_truth_scene
    from gt_gen.init_free import set_initial_free_cylinder
    from plan_seam import seam_ee_pose
    from curobo.geom.types import WorldConfig, Mesh
    from curobo.geom.sdf.world import CollisionCheckerType
    import torch
    from curobo.types.math import Pose

    cfg = load_config()
    d = pickle.load(open(args.seam, "rb"))
    robot_pose = np.asarray(d["robot_pose"][0], float)
    piece_pose = np.asarray(d["piece_pose"][0], float)
    goal_pose, seam_dbg = seam_ee_pose(d, offset_m=cfg.goal_standoff_m)   # 带 standoff 后退
    obj = sorted(glob.glob(os.path.dirname(args.seam) + "/*_watertight.obj"))[0]

    def _p(p7):
        return Pose(position=torch.tensor(np.array([p7[:3]]), dtype=torch.float32, device="cuda"),
                    quaternion=torch.tensor(np.array([p7[3:7]]), dtype=torch.float32, device="cuda"))
    mp = _p(robot_pose).inverse().multiply(_p(piece_pose)).get_pose_vector()[0].cpu().numpy().tolist()

    print(f"standoff={cfg.goal_standoff_m*100:.0f}cm  goal_pos={np.round(goal_pose[0], 3)}")
    world = WorldConfig(mesh=[Mesh(name="workpiece", file_path=obj, pose=mp)])
    print("初始化 h_truth（MESH，含工件）...")
    h_truth = ci.init_curobo(cfg, world_model=world, collision_checker_type=CollisionCheckerType.MESH)
    print("初始化 h_expl（VOXEL，纯三态，无 mesh）...")
    h_expl = ci.init_curobo(cfg)                              # 默认 VOXEL 全自由

    cam = load_camera_model(cfg)
    scene = load_truth_scene(obj, mesh_pose=mp)

    vm = build_roi_voxmap(cfg)
    n = set_initial_free_cylinder(h_truth, vm, config=cfg)
    print(f"初始圆柱 FREE 体素={n} (R={cfg.init_free_cyl_radius}m h={cfg.init_free_cyl_height}m)")

    return dict(cfg=cfg, h_truth=h_truth, h_expl=h_expl, vm=vm, cam=cam, scene=scene,
                obj=obj, mp=mp, goal_pose=goal_pose, seam_dbg=seam_dbg)


def verify_generate_gt(ctx, viz):
    """跑 generate_gt 并做 4 项断言。"""
    from gt_gen.main_loop import generate_gt
    from gt_gen.swept import motion_stays_in_free
    from gt_gen import curobo_iface as ci
    from scipy.spatial.transform import Rotation as Rsp

    cfg, h_truth, h_expl = ctx["cfg"], ctx["h_truth"], ctx["h_expl"]
    vm, cam, scene, goal_pose = ctx["vm"], ctx["cam"], ctx["scene"], ctx["goal_pose"]

    # 独立核校验（不走 _world_centers 自洽回读）：retract 整臂在初始 FREE 圆柱内，sync 后
    # 必须被 cuRobo voxel 核判 feasible。否则说明三态占据被放进了错位的 cuRobo 体素
    # （历史 bug：ROI 维度非 voxel 整数倍 → CUDA/Python 栅格维度取整不一致 → 沿轴抹开）。
    from gt_gen.collision_sync import sync_collision_world
    # _viz_collision_world(ctx, "before")          # sync 前：h_expl 应全自由（红切片为空）
    sync_collision_world(h_expl, vm)
    # _viz_collision_world(ctx, "after")           # sync 后：应只剩 FREE 圆柱"挖空"的洞
    rfeas, rcon = ci.check_state(h_expl, list(cfg.retract_config))
    assert rfeas, (f"retract 在 h_expl(voxel) 被判假碰撞(constraint={rcon:.1f})——"
                   f"voxel 占据放置错位（检查 ROI 维度是否 voxel_size 整数倍）")
    print(f"  ✓ 核校验：retract 在 h_expl(voxel) feasible（voxel 占据帧对齐）")

    print("\n== generate_gt 主循环 ==")
    GT, status, info = generate_gt(h_truth, h_expl, vm, scene, goal_pose, camera_model=cam)
    print(f"  status={status}  GT 路点={len(GT)}  轮数={info['rounds']}  P*长={info['P_len']}")
    print(f"  |B| 轨迹={info['n_B']}")
    print(f"  status 轨迹={info['status_seq']}")
    print(f"  FREE 增长={info['free'][:1]}…{info['free'][-3:] if info['free'] else []}")

    # ① 到达
    assert status == "reached", f"主循环未到达目标：status={status}"

    # ② 保守性：每对相邻 GT 整臂扫掠 ⊆ 最终 vm 的 FREE
    #    注：FREE 随观测增长，这里用【最终】vm 做近似校验（比规划当时的 FREE 更宽松）。
    n_viol = 0
    for i in range(len(GT) - 1):
        ok, nnf = motion_stays_in_free(h_truth, vm, GT[i], GT[i + 1])
        if not ok:
            n_viol += 1
    assert n_viol == 0, f"保守违例 {n_viol}/{len(GT)-1} 段扫掠含非 FREE 体素"
    print(f"  ✓ 保守性：{len(GT)-1} 段相邻运动整臂扫掠全 ⊆ FREE（0 违例）")

    # ③ 0 真值碰撞
    n_coll = sum(0 if ci.check_state(h_truth, GT[i].tolist())[0] else 1 for i in range(len(GT)))
    assert n_coll == 0, f"真值碰撞路点 {n_coll}/{len(GT)}"
    print(f"  ✓ 0 真值碰撞：{len(GT)} 路点全 feasible")

    # ④ 终点 ≈ standoff goal
    eep, eeq, _ = ci.fk(h_truth, GT[-1].tolist())
    perr = float(np.linalg.norm(eep - np.asarray(goal_pose[0], float)))
    Rg = Rsp.from_quat(np.r_[goal_pose[1][1:], goal_pose[1][0]]).as_matrix()
    Re = Rsp.from_quat(np.r_[eeq[1:], eeq[0]]).as_matrix()
    axis_ang = float(np.degrees(np.arccos(np.clip(np.dot(Re[:, 0], Rg[:, 0]), -1.0, 1.0))))
    pos_tol = cfg.position_threshold
    rot_tol_deg = float(np.degrees(2 * np.arcsin(min(1.0, cfg.rotation_threshold))))
    assert perr <= pos_tol + 1e-3, f"终点位置误差 {perr*1000:.1f}mm > 阈值 {pos_tol*1000:.0f}mm"
    assert axis_ang <= rot_tol_deg + 1e-6, f"终点接近轴夹角 {axis_ang:.1f}° > 阈值 {rot_tol_deg:.1f}°"
    print(f"  ✓ 终点≈standoff goal：位置误差={perr*1000:.1f}mm(阈值{pos_tol*1000:.0f}) "
          f"接近轴夹角={axis_ang:.1f}°(阈值{rot_tol_deg:.0f})")

    if viz:
        _viz_gt(ctx, GT)
    return GT, status, info


def _save_gt(ctx, GT, out):
    """把整条 GT + 场景信息存成 npz，可直接喂 scripts/viz_seam_isaacsim.py 播放轨迹。

    键与 scripts/plan_seam.py 的输出对齐（viz 直接消费）：
      positions (T,dof) 关节轨迹、joint_names、piece_pose_to_robot(工件@基座 wxyz)、
      obj_path(viz 据此换 .usd)、seam_mid/seam_bisector(焊缝中点+角平分方向, base 系)。
    不存时间戳（GT 只是构型序列，节拍由 viz 的 --fps 决定）。
    """
    cfg, seam_dbg = ctx["cfg"], ctx["seam_dbg"]
    positions = np.asarray([np.asarray(q, float) for q in GT])     # (T,dof)
    np.savez(
        out,
        positions=positions,
        joint_names=np.array(cfg.joint_names),
        piece_pose_to_robot=np.asarray(ctx["mp"], float),
        obj_path=ctx["obj"],
        seam_mid=np.asarray(seam_dbg["seam_mid"], float),
        seam_bisector=np.asarray(seam_dbg["seam_bisector"], float),
    )
    print(f"  ✓ GT 已存: {out}  (positions={positions.shape})")
    print(f"    可视化: conda run -n env_isaaclab python scripts/viz_seam_isaacsim.py --traj {out}")


def _viz_collision_world(ctx, stage):
    """可视化 sync_collision_world 前/后 h_expl(voxel) 的占据情况——目视核对 voxmap 的「非 FREE」
    是否被正确灌进 cuRobo（历史 Y 轴帧错位 bug 的人工复核口）。

    cuRobo 占据体素整块灌满后 ≈ 全 ROI 减去 FREE 圆柱（~80 万格），直接画立方体会卡死；故 cuRobo
    实际判占据的体素只取【过 ROI 盒中心 z 的一层水平切片】(暗红)，让 FREE 圆柱挖出的「洞」可见；
    vm 的 FREE 圆柱整体画半透明蓝（探索立足之地）。叠加 retract 整臂(绿) + 工件(灰)。

    stage ∈ {"before","after"}：
      before — h_expl 尚未 sync（默认 VOXEL 全自由）→ 红切片应为空（cuRobo 占据=0）；
      after  — 已 sync → 红切片布满整层、仅圆柱处留洞（cuRobo 占据≫0）。
    """
    from verify_step8 import _arm_mesh, _work_mesh, _cells_mesh, _draw, _roi_and_base
    from gt_gen.collision_sync import curobo_occupied_centers
    from gt_gen.voxmap import FREE, OCCUPIED

    cfg, h_truth, h_expl, vm, scene = (ctx["cfg"], ctx["h_truth"], ctx["h_expl"],
                                       ctx["vm"], ctx["scene"])
    vs = vm.voxel_size
    z0 = float(cfg.roi_center[2])                              # 切片高度 = ROI 盒中心 z

    geoms = [("work", _work_mesh(scene), "lit", None),
             ("arm", _arm_mesh(h_truth, list(cfg.retract_config)), "lit", None)]

    # vm 的 FREE（圆柱）整体半透明蓝 —— 这块在 cuRobo 里应被「挖空」成自由
    fc = vm.state_centers(FREE)
    if fc.shape[0]:
        geoms.append(("vm_free", _cells_mesh(vm, fc), "fill", [0.20, 0.45, 0.95, 0.12]))
    # vm 的 OCCUPIED（此刻冷启动一般为空，留作通用）红
    oc = vm.state_centers(OCCUPIED)
    if oc.shape[0]:
        om = _cells_mesh(vm, oc); om.paint_uniform_color([0.92, 0.12, 0.12])
        geoms.append(("vm_occ", om, "lit", None))

    # cuRobo 实际判占据的体素中心 → 只取过 z0 的一层水平切片（控制渲染量），暗红
    occ = curobo_occupied_centers(h_expl)
    n_occ = int(occ.shape[0])
    if n_occ:
        slab = occ[np.abs(occ[:, 2] - z0) <= 0.6 * vs]
        if slab.shape[0]:
            sm = _cells_mesh(vm, slab); sm.paint_uniform_color([0.65, 0.05, 0.05])
            geoms.append(("curobo_occ_slab", sm, "lit", None))

    geoms += _roi_and_base(vm)
    tag = "之前(应全自由,红切片空)" if stage == "before" else "之后(整层布满,仅圆柱留洞)"
    _draw(geoms, f"step10 sync {tag}: cuRobo占据={n_occ}格 "
                 f"红=过z={z0:.2f}m切片 蓝=vm FREE圆柱 绿=retract整臂")


def _viz_gt(ctx, GT):
    """复用 verify_step8 的 open3d 工具：最终 voxmap + 工件 + 沿 GT 抽样若干构型整臂 + 起终点。"""
    from verify_step8 import _arm_mesh, _work_mesh, _cells_mesh, _draw, _roi_and_base
    from gt_gen.voxmap import FREE, OCCUPIED

    h_truth, vm, scene = ctx["h_truth"], ctx["vm"], ctx["scene"]
    geoms = [("work", _work_mesh(scene), "lit", None)]
    fc = vm.state_centers(FREE)
    if fc.shape[0]:
        geoms.append(("free", _cells_mesh(vm, fc), "fill", [0.20, 0.45, 0.95, 0.12]))
    oc = vm.state_centers(OCCUPIED)
    if oc.shape[0]:
        om = _cells_mesh(vm, oc); om.paint_uniform_color([0.92, 0.12, 0.12])
        geoms.append(("occ", om, "lit", None))
    # 沿 GT 抽样 ~8 个构型画整臂（起点深绿、终点亮绿）
    idxs = np.linspace(0, len(GT) - 1, min(8, len(GT))).astype(int)
    for j, i in enumerate(idxs):
        arm = _arm_mesh(h_truth, GT[i].tolist())
        t = j / max(1, len(idxs) - 1)
        arm.paint_uniform_color([0.10 + 0.1 * t, 0.45 + 0.4 * t, 0.20])
        geoms.append((f"arm{i}", arm, "lit", None))
    geoms += _roi_and_base(vm)
    _draw(geoms, f"step10 GT: {len(GT)}点 status=reached（蓝FREE/红OCC/绿=沿GT整臂）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", default=SEAM)
    ap.add_argument("--viz", action="store_true", help="弹 open3d 窗口看整条 GT（需显示器，默认关）")
    ap.add_argument("--out", default="/tmp/seam_traj_step10.npz",
                    help="GT 存盘路径（npz，可直接喂 scripts/viz_seam_isaacsim.py）")
    args = ap.parse_args()

    ctx = build_scene_step10(args)
    GT, _, _ = verify_generate_gt(ctx, args.viz)
    _save_gt(ctx, GT, args.out)

    print("\nVERIFY_STEP10_OK [generate_gt: reached, 0 保守违例, 0 真值碰撞, 终点≈standoff goal]")


if __name__ == "__main__":
    main()
