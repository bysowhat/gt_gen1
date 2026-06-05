"""Step 7 自测：每个 verify_* 函数验证一个被测函数；main() 只负责搭环境 + 依次调用它们。

被测函数（逐步补充）：
  - verify_plan_on_truth     → 验证 curobo_iface.plan_on_truth，并按 plan_seam 的 npz 格式保存供可视化
  - verify_compute_reach_pt  → 验证 reach_b.compute_reach_pt（方式1 carve_first_n + 方式2 初始引导 FREE 空间）
  - (待补) verify_compute_blocking_B / ...

运行：
  conda run -n env_isaaclab python scripts/verify_step7_my.py --out /tmp/seam_traj_step7.npz
可视化保存的轨迹：
  conda run -n env_isaaclab python scripts/viz_seam_isaacsim.py --traj /tmp/seam_traj_step7.npz
  conda run -n env_isaaclab python scripts/viz_swept_o3d.py --traj /tmp/seam_traj_step7.npz --mode save
"""
import argparse
import glob
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

DEFAULT_SEAM = ("/media/a/新加卷/hanfeng/segment_sub_output/"
                "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_24.pkl")


def build_context(args):
    """搭建各 verify_* 共享的环境：真值 handle、retract、goal_pose、metric、工件位姿等。"""
    from gt_gen.config import load_config
    from gt_gen import curobo_iface as ci
    from plan_seam import seam_ee_pose
    from curobo.geom.types import WorldConfig, Mesh
    from curobo.geom.sdf.world import CollisionCheckerType
    import torch
    from curobo.types.math import Pose

    cfg = load_config()
    d = pickle.load(open(args.seam, "rb"))
    robot_pose = np.asarray(d["robot_pose"][0], float)
    piece_pose = np.asarray(d["piece_pose"][0], float)
    goal_pose = seam_ee_pose(d)                              # 由 pkl 几何算，非存的关节角
    obj = sorted(glob.glob(os.path.dirname(args.seam) + "/*_watertight.obj"))[0]

    def _p(p7):
        return Pose(position=torch.tensor([p7[:3]], dtype=torch.float32, device="cuda"),
                    quaternion=torch.tensor([p7[3:7]], dtype=torch.float32, device="cuda"))
    mp = _p(robot_pose).inverse().multiply(_p(piece_pose)).get_pose_vector()[0].cpu().numpy().tolist()

    world = WorldConfig(mesh=[Mesh(name="workpiece", file_path=obj, pose=mp)])
    print("初始化真值 cuRobo（含工件 MESH；position/rotation_threshold、drop link 从 default.yaml 读）...")
    h = ci.init_curobo(cfg, world_model=world, collision_checker_type=CollisionCheckerType.MESH)
    return SimpleNamespace(cfg=cfg, d=d, h=h, retract=cfg.retract_config,
                           goal_pose=goal_pose, metric=ci.free_pose_metric(h, free_rot=(0,)),
                           mesh_pose=mp, obj=obj, out=args.out, max_attempts=args.max_attempts,
                           viz=args.viz)


# ============ 每个函数验证一个被测函数 ============

def verify_plan_on_truth(ctx):
    """验证 curobo_iface.plan_on_truth：4 条判据 + 按 plan_seam 格式保存 P*。"""
    from gt_gen import curobo_iface as ci
    from scipy.spatial.transform import Rotation as Rsp

    print("\n== verify plan_on_truth ==")
    print("goal_pose:", np.round(ctx.goal_pose[0], 3), np.round(ctx.goal_pose[1], 3))
    P = ci.plan_on_truth(ctx.h, ctx.retract, ctx.goal_pose,
                         max_attempts=ctx.max_attempts, pose_cost_metric=ctx.metric)
    # 1) 形状（不可达返回 None）
    assert P is not None and P.ndim == 2 and P.shape[1] == len(ctx.retract), "P* 规划失败/形状异常"
    # 2) 端点：起点重合 + FK(终点)≈goal_pose；容差按 config 阈值（放开 roll，故比位置 + 接近轴夹角）
    d0 = float(np.linalg.norm(P[0] - np.asarray(ctx.retract, float)))
    eep, eeq, _ = ci.fk(ctx.h, P[-1].tolist())
    perr = float(np.linalg.norm(eep - np.asarray(ctx.goal_pose[0], float)))
    Rg = Rsp.from_quat(np.r_[ctx.goal_pose[1][1:], ctx.goal_pose[1][0]]).as_matrix()
    Re = Rsp.from_quat(np.r_[eeq[1:], eeq[0]]).as_matrix()
    axis_dot = float(np.clip(np.dot(Re[:, 0], Rg[:, 0]), -1.0, 1.0))
    axis_ang = float(np.degrees(np.arccos(axis_dot)))
    pos_tol = ctx.cfg.position_threshold
    rot_tol_deg = float(np.degrees(2 * np.arcsin(min(1.0, ctx.cfg.rotation_threshold))))
    assert d0 < 1e-3, f"起点偏差 {d0}"
    assert perr <= pos_tol + 1e-3, f"位置误差 {perr} > 阈值 {pos_tol}"
    assert axis_ang <= rot_tol_deg + 1e-6, f"接近轴夹角 {axis_ang:.1f}° > 阈值 {rot_tol_deg:.1f}°"
    # 3) 真值无碰撞（handle=真值 → check_state 即真实障碍下判碰撞）
    feas = [ci.check_state(ctx.h, P[i].tolist())[0] for i in range(len(P))]
    assert all(feas), f"碰撞路点 {feas.count(False)}/{len(P)}"
    # 4) 关节限位内
    lim = ctx.h.mg.kinematics.get_joint_limits().position.detach().cpu().numpy()
    assert np.all(P >= lim[0] - 1e-3) and np.all(P <= lim[1] + 1e-3), "越限"
    print(f"  路点={len(P)} 起点偏差={d0:.5f} 终点位置误差={perr*1000:.1f}mm(阈值{pos_tol*1000:.0f}) "
          f"接近轴夹角={axis_ang:.1f}°(阈值{rot_tol_deg:.0f}) 真值无碰撞={sum(feas)}/{len(P)}")

    # 按 plan_seam 的 npz 格式保存（供 viz_seam_isaacsim.py / viz_swept_o3d.py）
    np.savez(ctx.out, positions=P, joint_names=np.array(ctx.cfg.joint_names),
             retract=np.array(ctx.retract), target=np.array(P[-1]),
             piece_pose_to_robot=np.array(ctx.mesh_pose), obj_path=ctx.obj,
             robot_usd=ctx.cfg.robot_cfg["robot_cfg"]["kinematics"].get("usd_path", ""), dt=0.02)
    print("  已保存:", ctx.out)
    ctx.p_star = P                      # 存给下游 verify_*（如 compute_reach_pt）复用
    return P


def verify_compute_reach_pt(ctx):
    """验证 reach_b.compute_reach_pt，两种互补的造 voxmap 方式：

    方式1 carve_first_n（沿轨迹）：把 P* 前 n 段整臂横扫体素标 FREE 当"已确认自由区"，
      预期 reach_idx == n（前 n 段全 FREE 能走，第 n 段碰未知即停）——测"首个越界即停"逻辑。
    方式2 初始引导 FREE 空间（场景无关，set_initial_free_space）：只在 retract 邻域标一小块
      FREE（GT 生成的真实起步方式），预期机械臂能起步 reach_idx≥1，对比全 UNKNOWN 的 0——
      测"给定初始自由区即可起步，否则末端相机视野太小动不了"。
    """
    from gt_gen.voxmap import build_roi_voxmap, FREE
    from gt_gen.swept import swept_volume
    from gt_gen.reach_b import compute_reach_pt
    from gt_gen.init_free import set_initial_free_space

    P = ctx.p_star
    assert P is not None, "需先有 P*（verify_plan_on_truth 应先跑）"
    T = len(P)
    print("\n== verify compute_reach_pt ==  (P* 路点", T, ")")

    def carve_first_n(n):
        vm = build_roi_voxmap(ctx.cfg)                       # 全 UNKNOWN
        for i in range(n):
            vm.set_many(swept_volume(ctx.h, vm, P[i], P[i + 1]), FREE)
        return vm

    # 全 UNKNOWN → 0（两种方式共用的基线：没有任何已知自由区，整臂扫掠全落 UNKNOWN）
    ri0 = compute_reach_pt(ctx.h, build_roi_voxmap(ctx.cfg), P)
    assert ri0 == 0, f"全UNKNOWN应得 0，实际 {ri0}"
    print(f"  全 UNKNOWN → reach_idx={ri0} ✓（无初始自由区则机械臂起步不了）")

    # —— 方式1：carve_first_n（沿轨迹标 FREE）——
    print("  [方式1] 沿 P* 前 n 段横扫标 FREE：")
    for n in (T // 4, T // 2, 3 * T // 4):
        ri = compute_reach_pt(ctx.h, carve_first_n(n), P)
        assert ri == n, f"前 {n} 段FREE应得 reach_idx={n}，实际 {ri}"
        print(f"    前 {n}/{T-1} 段 → reach_idx={ri} ✓")
    ri_full = compute_reach_pt(ctx.h, carve_first_n(T - 1), P)
    assert ri_full == T - 1, f"整条FREE应得 {T-1}，实际 {ri_full}"
    print(f"    整条横扫标FREE → reach_idx={ri_full} (=终点) ✓")

    # —— 方式2：初始引导 FREE 空间（set_initial_free_space，GT 生成同款）——
    vm_init = build_roi_voxmap(ctx.cfg)                      # 全 UNKNOWN 起
    n_free = set_initial_free_space(ctx.h, vm_init, config=ctx.cfg)
    # viz_voxmap_arm(ctx, vm_init, ctx.retract)

    ri_init = compute_reach_pt(ctx.h, vm_init, P)
    assert n_free > 0, "初始 FREE 空间为空（set_initial_free_space 未标到任何体素）"
    assert 1 <= ri_init <= T - 1, \
        f"初始 FREE 空间应让机械臂起步且不越过终点：reach_idx={ri_init} 不在 [1,{T-1}]"
    print(f"  [方式2] retract 邻域初始FREE空间 {n_free} 体素 → reach_idx={ri_init} "
          f"(基线全UNKNOWN={ri0}) ✓（给定初始自由区即可起步）")

    # 可视化（open3d 交互窗口，需显示器；--viz 开启）：
    if getattr(ctx, "viz", False):
        viz_voxmap_arm(ctx, carve_first_n(ri_init), ctx.retract)   # 方式1：沿轨迹 FREE
        viz_voxmap_arm(ctx, vm_init, ctx.retract)             # 方式2：初始引导 FREE 空间


def viz_voxmap_arm(ctx, vm, q, spheres=None):
    """open3d 交互窗口可视化网格 + 机械臂本地碰撞球：
    voxmap 的 FREE(蓝)/OCCUPIED(红) 格(半透明实心) + 构型 q 的整臂碰撞球(绿) + ROI 线框 + base 轴。
    （交互窗口，可旋转/缩放；不存图。需本机有显示器。）
    """
    import open3d as o3d
    from open3d.visualization import rendering
    from gt_gen.voxmap import FREE, OCCUPIED
    from gt_gen.swept import fk_spheres_batch
    from gt_gen import sensor

    vs = vm.voxel_size

    def cells_mesh(centers):
        m = o3d.geometry.TriangleMesh()
        for c in centers:
            b = o3d.geometry.TriangleMesh.create_box(vs, vs, vs); b.translate(c - vs / 2); m += b
        if len(m.vertices):
            m.compute_vertex_normals()
        return m

    sph = fk_spheres_batch(ctx.h, [q])[0] if spheres is None else spheres
    arm = o3d.geometry.TriangleMesh()
    for s in sph:
        r = float(s[3])
        if r <= 1e-4:
            continue
        b = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=8); b.translate(s[:3]); arm += b
    arm.paint_uniform_color([0.20, 0.72, 0.32]); arm.compute_vertex_normals()

    geoms = [("arm", arm, "lit", None)]
    # 工件 mesh（灰色实心）
    scene = sensor.load_truth_scene(ctx.obj, mesh_pose=ctx.mesh_pose)
    work = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(scene.vertices)),
                                     o3d.utility.Vector3iVector(np.asarray(scene.faces)))
    work.paint_uniform_color([0.72, 0.72, 0.72]); work.compute_vertex_normals()
    geoms.append(("work", work, "lit", None))
    fc = vm.state_centers(FREE)
    if fc.shape[0]:
        geoms.append(("free", cells_mesh(fc), "fill", [0.20, 0.45, 0.95, 0.30]))
    oc = vm.state_centers(OCCUPIED)
    if oc.shape[0]:
        om = cells_mesh(oc); om.paint_uniform_color([0.92, 0.12, 0.12])   # OCCUPIED 不透明实心
        geoms.append(("occ", om, "lit", None))
    aabb = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(vm.origin, vm.upper)); aabb.paint_uniform_color([0.6, 0.6, 0.6])
    geoms += [("roi", aabb, "line", None),
              ("base", o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3), "lit", None)]

    items = []
    for name, g, kind, rgba in geoms:
        mat = rendering.MaterialRecord()
        if kind == "fill":
            mat.shader = "defaultLitTransparency"; mat.base_color = rgba
        elif kind == "line":
            mat.shader = "unlitLine"; mat.line_width = 2.0
        else:
            mat.shader = "defaultLit"
        items.append({"name": name, "geometry": g, "material": mat})
    print("  打开 open3d 交互窗口（关闭窗口后继续）...")
    o3d.visualization.draw(items, title="step7 网格(蓝FREE/红OCC) + 碰撞球(绿)",
                           width=1400, height=1000, bg_color=(1.0, 1.0, 1.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", default=DEFAULT_SEAM)
    ap.add_argument("--out", default="/tmp/seam_traj_step7.npz")
    ap.add_argument("--max_attempts", type=int, default=200)
    ap.add_argument("--viz", action="store_true",
                    help="弹 open3d 交互窗口看 voxmap+整臂碰撞球（需本机有显示器；默认关，便于无头自测）")
    args = ap.parse_args()

    ctx = build_context(args)

    # —— 每行验证一个被测函数（后续按需追加）——
    verify_plan_on_truth(ctx)
    verify_compute_reach_pt(ctx)

    print("\nVERIFY_STEP7_MY_OK")


if __name__ == "__main__":
    main()
