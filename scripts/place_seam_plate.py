"""在指定焊缝旁放置一块 plate 障碍物，与两面构成 C 型断面（独立脚本）。

几何（机械臂 base 系）：两面 a/b 相夹成焊缝，方向 d1/d2、焊缝切线 t、角平分线 bis。
板 c 平行于其中一个面（--parallel b → c∥b、c⊥脊面 a），沿【脊面 a 的表面方向 d1】挪到
距焊缝 n_cm 处 → b（焊缝处臂）–a（脊）–c（远端臂）三片构成 C/U 槽断面：
  · 板平面 = span{t, 所选面内⊥t 的方向 w}，法向 n̂ = 所选面法向（取朝 bis 张开侧）；
  · 板心 = 焊缝中点 mid + n_cm·â（â=脊面 a 表面方向，即 C 型两臂间距方向）；
  · 长沿 t（=焊缝弧长×length_pct%）、宽 width_cm、厚 thickness_cm；
  · 角度 p（--angle_deg）：朝向绕焊缝切线 t 自转 p°（p=0 即正好平行所选面）。
  a⊥b（90° 角焊缝）时 â 与 b 的法向重合；夹角≠90° 时 c 仍贴着脊面 a 延伸，C 型不散。

产出与 place_obstacles.py 同格式 npz（含规划好的 retract→焊缝【默认（不含板）】轨迹），可直接用
scripts/viz_placed_obstacle_isaacsim.py 回放（工件 + plate + 机械臂走到焊缝）。板仅遮挡用、不做避障，
故 detour 字段填同一条默认轨迹，让 viz 的 --which default / detour 两种都能放机械臂。

运行（本机 conda）：
    conda run -n env_isaaclab --no-capture-output python -u scripts/place_seam_plate.py \
        --seam /media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_22.pkl \
        --parallel b --n_cm 10 --length_pct 80 --width_cm 30 --thickness_cm 3 --angle_deg 0 \
        --out /tmp/seam_plate/scene_seam_plate.npz
可视化：
    conda run -n env_isaaclab python scripts/viz_placed_obstacle_isaacsim.py \
        --scene /tmp/seam_plate/scene_seam_plate.npz --which default
"""
import argparse
import os
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rsp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

DEFAULT_SEAM = ("/media/a/新加卷/hanfeng/segment_sub_output/"
                "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_22.pkl")


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def seam_frame(d, index=None):
    """焊缝几何 → base 系：返回 (mid, t, d1, d2, bis, seam_len)。

    复刻 plan_seam.seam_ee_pose 的世界→base 变换（robot_pose 求逆）：
      mid = 焊缝中点；t = seam_tangent；d1/d2 = seam_limits 两面方向；bis = normalize(d1+d2)；
      seam_len = seam_line 折线弧长（刚体变换不变，直接在世界系算）。
    """
    line = np.asarray(d["seam_line"], float)         # (N,3) world
    tang = np.asarray(d["seam_tangent"], float)      # (N,3) world
    limits = np.asarray(d["seam_limits"], float)     # (N,2,3) world
    i = int(d.get("middle", len(line) // 2)) if index is None else index

    rp = np.asarray(d["robot_pose"][0], float)       # [x,y,z,qw,qx,qy,qz]
    Rwr = Rsp.from_quat(np.r_[rp[4:7], rp[3]]).as_matrix()   # wxyz -> xyzw
    to_base_p = lambda p: Rwr.T @ (p - rp[:3])
    to_base_v = lambda v: Rwr.T @ v

    mid = to_base_p(line[i])
    t = _unit(to_base_v(tang[i]))
    d1 = to_base_v(limits[i, 0])
    d2 = to_base_v(limits[i, 1])
    bis = _unit(d1 + d2)
    seam_len = float(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)))
    return mid, t, d1, d2, bis, seam_len


def plate_for_seam(d, *, parallel, n_cm, length_pct, width_cm, thickness_cm, angle_deg):
    """据焊缝几何造一块遮挡 plate，返回 (prims, anchor, info)。

    构造 C 型：两面 a/b 相夹成焊缝，板 c 平行于面 b（--parallel b）、垂直于脊面 a，
    沿脊面 a 的表面方向(d1)挪到距焊缝 n_cm 处 → b–a–c 三片构成 C/U 槽断面。
    parallel ∈ {a,b}：板平行于该面；位移沿【另一面】的表面方向（C 型两臂间距）。
    板局部轴：X=板法向 n̂(⊥所选面 → 板∥该面)、Y≈面内⊥t 的宽方向、Z=焊缝切线 t；再绕 t 自转 angle_deg°。
    """
    from gt_gen import obstacles as ob
    mid, t, d1, d2, bis, seam_len = seam_frame(d)
    face_par = d2 if parallel == "b" else d1            # 板要平行的面（b）
    face_perp = d1 if parallel == "b" else d2           # 与之垂直的脊面（a），C 型沿它延伸
    # 所选面内、垂直焊缝切线的宽方向 w（去掉沿 t 的分量）
    w = _unit(face_par - float(np.dot(face_par, t)) * t)
    # 板法向 n̂ ⊥ 板平面 span{t,w} → 板平行于所选面；取朝 bis 张开侧
    n_hat = _unit(np.cross(t, w))
    if float(np.dot(n_hat, bis)) < 0:
        n_hat = -n_hat
    # C 型臂间距方向：沿脊面 a 的表面方向（d1，去掉沿 t 分量），而非板法向
    arm_dir = _unit(face_perp - float(np.dot(face_perp, t)) * t)
    # 右手系：X=法向, Z=焊缝切线, Y=Z×X（≈±w，对称板符号无关）
    x_axis = n_hat
    z_axis = t
    y_axis = _unit(np.cross(z_axis, x_axis))
    R0 = np.column_stack([x_axis, y_axis, z_axis])
    R_final = Rsp.from_matrix(R0) * Rsp.from_euler("z", float(angle_deg), degrees=True)
    rpy = [float(v) for v in R_final.as_euler("xyz", degrees=True)]

    anchor = (mid + (float(n_cm) / 100.0) * arm_dir).astype(float)
    length = float(length_pct) / 100.0 * seam_len      # 板长沿 t（焊缝走向）
    width = float(width_cm) / 100.0
    thickness = float(thickness_cm) / 100.0
    # ob.plate: dims=[thickness, width, length] 对齐局部 X/Y/Z；tilt_deg=0（倾斜已由 rpy 含入）
    prims = ob.build("plate", anchor.tolist(), anchor_rpy_deg=tuple(rpy),
                     length=length, width=width, thickness=thickness)
    info = dict(parallel=parallel, n_cm=float(n_cm), length_pct=float(length_pct),
                width_cm=float(width_cm), thickness_cm=float(thickness_cm),
                angle_deg=float(angle_deg), seam_len=seam_len, length=length,
                width=width, thickness=thickness, arm_dir=arm_dir.tolist(),
                rpy_deg=rpy, mid=mid.tolist(), bisector=bis.tolist())
    return prims, anchor, info


def _serialize_prims(prims):
    """Box/Tube → 可存 npz 的 dict 列表（与 place_obstacles.py 约定一致）。"""
    from gt_gen import obstacles as ob
    out = []
    for p in prims:
        if isinstance(p, ob.Box):
            out.append(dict(kind="box", name=p.name, dims=list(map(float, p.dims)),
                            pose=list(map(float, p.pose)), color=list(map(float, p.color))))
        else:
            out.append(dict(kind="tube", name=p.name, radius=float(p.radius),
                            height=float(p.height), pose=list(map(float, p.pose)),
                            color=list(map(float, p.color))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", default=DEFAULT_SEAM)
    ap.add_argument("--parallel", choices=["a", "b"], default="b",
                    help="板平行于哪个面（a=seam_limits[0] / b=seam_limits[1]）；位移沿【另一面】表面方向构成 C 型")
    ap.add_argument("--n_cm", type=float, default=10.0, help="C 型两臂间距：沿脊面(另一面)表面方向离焊缝的距离 n(cm)")
    ap.add_argument("--length_pct", type=float, default=80.0, help="板长占焊缝弧长的百分比 m(%)")
    ap.add_argument("--width_cm", type=float, default=30.0, help="板宽 l(cm)")
    ap.add_argument("--thickness_cm", type=float, default=3.0, help="板厚 o(cm)")
    ap.add_argument("--angle_deg", type=float, default=0.0, help="朝向绕焊缝切线自转角度 p(度)")
    ap.add_argument("--offset_cm", type=float, default=10.0,
                    help="轨迹目标沿 bis 的 standoff(cm，与 place_obstacles 一致)")
    ap.add_argument("--out", default="/tmp/seam_plate/scene_seam_plate.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from gt_gen.config import load_config
    from gt_gen import curobo_iface as ci
    from gt_gen import obstacle_placement as opl
    from curobo.geom.types import WorldConfig, Mesh
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.types.math import Pose
    from plan_seam import find_obj, seam_ee_pose

    np.random.seed(args.seed)

    cfg = load_config()

    d = pickle.load(open(args.seam, "rb"))
    goal_pose, seam_dbg = seam_ee_pose(d, offset_m=args.offset_cm / 100.0)
    obj_path = find_obj(args.seam)
    robot_pos = np.asarray(d["robot_pose"][0], float)
    piece_pose = np.asarray(d["piece_pose"][0], float)

    print("seam     :", args.seam)
    print("obj      :", obj_path)
    print("规划后端 :", cfg.planner_backend)

    # —— 造遮挡板（解析，几何唯一确定）——
    prims, anchor, info = plate_for_seam(
        d, parallel=args.parallel, n_cm=args.n_cm, length_pct=args.length_pct,
        width_cm=args.width_cm, thickness_cm=args.thickness_cm, angle_deg=args.angle_deg)
    print(f"plate    : 平行面={args.parallel} 臂间距={args.n_cm}cm 沿面 a 方向={np.round(info['arm_dir'], 3)}")
    print(f"           长={info['length']:.3f}m({args.length_pct}%×焊缝{info['seam_len']:.3f}m) "
          f"宽={info['width']:.3f}m 厚={info['thickness']:.3f}m 自转={args.angle_deg}° "
          f"板心={np.round(anchor, 3)}")

    # —— 工件 mesh @ 机械臂基座系（同 plan_seam）——
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

    print("== 规划默认轨迹 retract→焊缝（不含板）==")
    traj = opl.plan_default_traj(h, retract, goal_pose, metric, cfg.plan_max_attempts,
                                 cfg=cfg, world=world0, checker_type=CollisionCheckerType.MESH)
    if traj is None:
        print("PLACE_SEAM_PLATE_FAIL: 默认轨迹规划失败（无障碍都到不了 goal）")
        return
    print(f"默认轨迹点数: {traj.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    robot_usd = cfg.robot_cfg["robot_cfg"]["kinematics"].get("usd_path", "")
    traj_dt = float(cfg.stomp_params["delta_t"]) if cfg.planner_backend == "stomp" else 0.02
    np.savez(
        args.out,
        positions=traj,
        detour_positions=traj,                            # 板不做避障 → detour 即默认轨迹（viz 两种 --which 都能放臂）
        detour_positions_all=np.array([traj], dtype=object),
        n_detour=1,
        joint_names=np.array(cfg.joint_names),
        retract=np.array(retract),
        obstacle_prims=np.array(_serialize_prims(prims), dtype=object),
        anchor=np.array(anchor, float),
        link="seam", otype="plate",
        piece_pose_to_robot=np.array(mesh_pose, float),
        obj_path=obj_path,
        robot_usd=robot_usd,
        seam_mid=np.array(seam_dbg["seam_mid"], float),
        seam_bisector=np.array(seam_dbg["seam_bisector"], float),
        dt=traj_dt,
    )
    print("npz 输出:", args.out)
    print("PLACE_SEAM_PLATE_OK")


if __name__ == "__main__":
    main()
