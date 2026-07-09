"""规划机械臂初始位姿（起步姿态）相关工具（独立脚本，不动业务代码）。

本脚本两块：

① init 空间可视化（show_init_space / init_space_geometries）——
   用 open3d 可视化【机械臂某构型下的整臂碰撞球】+【init_free 起步引导空间（立方体盒）】，
   肉眼核对碰撞球是否整只落在 init 盒内（盒太小→球冒出盒外那层体素恒 UNKNOWN，起步会假阳性非 FREE）。
     · 碰撞球：在构型 q（默认 = retract 固定安全 home）下，对全部 collision_link_names 做 FK，
       把各 link 的 collision_spheres 变换到 base 系，按真实半径画成红色线框球。
     · init 盒：base_link 系下的轴对齐长方体 [box_min, box_max]（configs/default.yaml:
       init_free.box_min_m / box_max_m），画成青色线框。盒中心落在 base_link。
     · base 坐标系：原点处一个小三轴坐标架，便于判读朝向。
   运行（本机 conda，需要显示器）：
       conda run -n env_isaaclab --no-capture-output python scripts/plan_init_pose.py
   可选：--solid 把碰撞球画成实心球、--q j0 j1 ... 指定构型（缺省用 retract）。

② lookup 式初始位姿求解（InitPoseLookupSolver / --solve）——
   移植自 /home/a/Projects/xiaoyu/ifc_analyzer/baiyu/arm_pose/solve_arm_pose_lookup.py
   （文档同目录 docs/solve_arm_pose_lookup.md）。给定工件 mesh(_part.obj) + 焊缝
   (_weld_angle3.json)，离线一次性预计算 N=n_per_dof^6 个关节角的 FK（EE 位置/焊枪轴/连杆碰撞球），
   在线对每条焊缝反解出唯一工件位姿 (R,t)（焊枪头到焊缝中点、焊枪轴对准 bisector），只用「连杆球 +
   retract 球 vs 工件 ESDF」做碰撞过滤，按「绕自身三轴离 90° 整倍数偏差和」评分取最优。碰撞后端用裸 cuRobo
   RobotWorld + WorldVoxelCollision + 工件 ESDF（本项目 CuroboHandle 是固定位姿 MESH 世界，
   做不了"工件逐候选移动"的 batch 球碰撞查询）。
   运行（需 GPU；可选 --viz 复用 ① 的可视化）：
       conda run -n env_isaaclab --no-capture-output python -u scripts/plan_init_pose.py --solve \
           --obj <…_part.obj> --weld-json <…_weld_angle3.json> --limit 3 --diagnostic
"""
import argparse
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DEFAULT_LAY_FLAT_OBJ = ("/media/a/新加卷/hanfeng/segment/A3Changfang/"
                        "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part_watertight.obj")


def init_space_geometries(cfg, q=None, solid_spheres=False):
    """构造 open3d 几何列表：构型 q 下整臂碰撞球 + init_free 立方体盒（线框）+ base 坐标架。

    参数：
      cfg           : Config（load_config()）。
      q             : 构型（rad，长度=关节数）；None 时取 cfg.retract_config（起步固定 home）。
      solid_spheres : True 把碰撞球画成实心球；False（默认）画成红色线框球。

    返回：(geoms, stat)。geoms 为 open3d 几何列表；stat 为 dict（球数 / 盒尺寸等，便于打印）。
    """
    import open3d as o3d
    from gt_gen.obstacle_placement import compute_link_sweep

    if q is None:
        q = cfg.retract_config
    q = [float(v) for v in q]

    geoms = []

    # —— 整臂碰撞球：FK 到 base 系，按真实半径逐球画 ——
    per_wp, _ = compute_link_sweep(cfg, [q], cfg.collision_link_names)
    n_sph = 0
    for ln, s in per_wp.items():
        for c in np.asarray(s, float)[0]:                    # (S,4)：取唯一路点
            cx, cy, cz, r = (float(v) for v in c)
            if r <= 1e-4:
                continue
            ball = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=8)
            ball.translate((cx, cy, cz))
            if solid_spheres:
                ball.compute_vertex_normals()
                ball.paint_uniform_color([0.85, 0.1, 0.1])
                geoms.append(ball)
            else:
                ls = o3d.geometry.LineSet.create_from_triangle_mesh(ball)
                ls.paint_uniform_color([0.85, 0.1, 0.1])
                geoms.append(ls)
            n_sph += 1

    # —— init_free 立方体盒：base 系轴对齐 [box_min, box_max]，青色线框 ——
    lo = np.asarray(cfg.init_free_box_min, float)
    hi = np.asarray(cfg.init_free_box_max, float)
    aabb = o3d.geometry.AxisAlignedBoundingBox(lo.tolist(), hi.tolist())
    aabb.color = (0.0, 0.75, 0.75)
    geoms.append(aabb)

    # —— base 坐标架（原点，0.3m）——
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=(0.0, 0.0, 0.0))
    geoms.append(frame)

    stat = dict(n_spheres=n_sph, box_min=lo.tolist(), box_max=hi.tolist(),
                box_size=(hi - lo).tolist(), box_center=((lo + hi) / 2.0).tolist())
    return geoms, stat


def show_init_space(cfg, q=None, solid_spheres=False):
    """开窗显示「整臂碰撞球 + init_free 立方体盒」（关闭窗口结束）。"""
    import open3d as o3d
    geoms, stat = init_space_geometries(cfg, q=q, solid_spheres=solid_spheres)
    print(f"碰撞球   : {stat['n_spheres']} 个（红色线框）")
    print(f"init 盒  : min={np.round(stat['box_min'], 3)} max={np.round(stat['box_max'], 3)} "
          f"尺寸={np.round(stat['box_size'], 3)}m 中心={np.round(stat['box_center'], 3)}（青色线框）")
    print("显示中（关闭窗口结束）…")
    o3d.visualization.draw_geometries(geoms, window_name="init pose: 碰撞球 + init_free box")


# ============================================================================
# ② lookup 式初始位姿求解（移植自 solve_arm_pose_lookup.py / solve_arm_pose_parallel.py）
# ============================================================================
# 下面的纯几何/位姿工具、焊缝 I/O、求解器类都是从外部参考脚本【复制/移植】而来（不 import
# 外部项目：它有 sys.path 篡改、setup_logger 等模块级副作用，与本项目 curobo 前置不同）。
# 碰撞后端用裸 cuRobo RobotWorld + WorldVoxelCollision + 工件 ESDF。

# ---- 纯几何工具（torch batch；源 solve_arm_pose_lookup.py L61–174 原样） ----
def batch_axis_angle_rotmat(axis, angle):
    """Rodrigues batch 版。axis (...,3) 单位向量，angle (...) rad → R (...,3,3)。"""
    import torch
    if angle.dim() < axis.dim() - 1:
        angle = angle.expand(*axis.shape[:-1])
    cos_a = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)
    sin_a = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
    one_m = 1.0 - cos_a
    x = axis[..., 0:1].unsqueeze(-1)
    y = axis[..., 1:2].unsqueeze(-1)
    z = axis[..., 2:3].unsqueeze(-1)
    R = torch.cat([
        torch.cat([cos_a + one_m * x * x,
                   one_m * x * y - sin_a * z,
                   one_m * x * z + sin_a * y], dim=-1),
        torch.cat([one_m * y * x + sin_a * z,
                   cos_a + one_m * y * y,
                   one_m * y * z - sin_a * x], dim=-1),
        torch.cat([one_m * z * x - sin_a * y,
                   one_m * z * y + sin_a * x,
                   cos_a + one_m * z * z], dim=-1),
    ], dim=-2)
    return R


def batch_align_rotation(a_unit, b_unit):
    """最小角旋转 R 满足 R @ a_unit = b_unit。a_unit (3,)，b_unit (N,3) → R (N,3,3)。"""
    import torch
    N = b_unit.shape[0]
    device = b_unit.device
    dtype = b_unit.dtype
    a_b = a_unit.unsqueeze(0).expand(N, -1)
    cos_th = torch.sum(a_b * b_unit, dim=-1)
    cross = torch.cross(a_b, b_unit, dim=-1)
    sin_th = torch.norm(cross, dim=-1)
    axis = cross / (sin_th.unsqueeze(-1) + 1e-12)
    angle = torch.atan2(sin_th, cos_th)
    R_general = batch_axis_angle_rotmat(axis, angle)
    aligned = cos_th > 0.9999
    I = torch.eye(3, device=device, dtype=dtype).expand(N, -1, -1)
    flipped = cos_th < -0.9999
    e_z = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    perp = torch.cross(a_unit, e_z, dim=-1)
    if torch.norm(perp) < 1e-6:
        e_x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        perp = torch.cross(a_unit, e_x, dim=-1)
    perp = perp / (torch.norm(perp) + 1e-12)
    perp_b = perp.unsqueeze(0).expand(N, -1)
    R_180 = batch_axis_angle_rotmat(
        perp_b, torch.full((N,), float(np.pi), device=device, dtype=dtype))
    mask_aligned = aligned.unsqueeze(-1).unsqueeze(-1)
    mask_flipped = flipped.unsqueeze(-1).unsqueeze(-1)
    R = torch.where(mask_flipped, R_180, R_general)
    R = torch.where(mask_aligned, I, R)
    return R


def axis_align_score_batch(R):
    """工件姿态(R=T_workpiece_in_base 旋转部分)按【绕自身三轴 内旋 X→Y→Z】拆成 (rx,ry,rz)，
    每轴各算「离最近 90° 整倍数的偏差」d=|a−90°×round(a/90°)|∈[0°,45°]，三轴 d 相加取负作分数。
    =0 表示三轴恰好都落在 90° 整倍数(工件三轴对齐 base 三轴)，分最高；越斜各轴偏差越大、分越低。
    （round-to-nearest 自动覆盖 …−2,−1,0,1,2… 各 90° 倍数，过 45° 即归到下一倍数→单轴偏差恒≤45°。）"""
    import torch
    half_pi = float(np.pi / 2.0)
    rad2deg = float(180.0 / np.pi)
    # 内旋 XYZ 分解：R = Rx(rx)·Ry(ry)·Rz(rz)
    sy = torch.clamp(R[..., 0, 2], -1.0, 1.0)
    ry = torch.asin(sy)
    rx = torch.atan2(-R[..., 1, 2], R[..., 2, 2])
    rz = torch.atan2(-R[..., 0, 1], R[..., 0, 0])

    def _dev_deg(a):
        nearest = torch.round(a / half_pi) * half_pi   # 最近的 90° 整倍数(rad)
        return torch.abs(a - nearest) * rad2deg        # 偏差(度)，∈[0,45]

    return -(_dev_deg(rx) + _dev_deg(ry) + _dev_deg(rz))   # ∈[-135,0]，越大越对齐


def quat_wxyz_to_x_axis_batch(quat):
    """四元数 (B,4 wxyz) → 旋转矩阵第 0 列（局部 +x 轴在世界的方向）(B,3)。"""
    import torch
    w = quat[..., 0]; x = quat[..., 1]; y = quat[..., 2]; z = quat[..., 3]
    return torch.stack([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    ], dim=-1)


def quat_wxyz_to_rotmat_batch(quat):
    """四元数 (B,4 wxyz) → 旋转矩阵 (B,3,3)，各列 = 末端局部 x/y/z 轴在 base 的方向。
    （第 0 列与 quat_wxyz_to_x_axis_batch 一致。）"""
    import torch
    q = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-12)
    w = q[..., 0]; x = q[..., 1]; y = q[..., 2]; z = q[..., 3]
    col0 = torch.stack([1.0 - 2.0 * (y * y + z * z),
                        2.0 * (x * y + w * z),
                        2.0 * (x * z - w * y)], dim=-1)
    col1 = torch.stack([2.0 * (x * y - w * z),
                        1.0 - 2.0 * (x * x + z * z),
                        2.0 * (y * z + w * x)], dim=-1)
    col2 = torch.stack([2.0 * (x * z + w * y),
                        2.0 * (y * z - w * x),
                        1.0 - 2.0 * (x * x + y * y)], dim=-1)
    return torch.stack([col0, col1, col2], dim=-1)  # (B,3,3) 列向量为各轴


# ---- 纯位姿工具（numpy/scipy；父 solve_arm_pose_parallel.py L59–79 原样） ----
def quat_wxyz_to_rotmat(q):
    from scipy.spatial.transform import Rotation as sR
    w, x, y, z = q
    return sR.from_quat([x, y, z, w]).as_matrix()


def rotmat_to_quat_wxyz(rm):
    from scipy.spatial.transform import Rotation as sR
    x, y, z, w = sR.from_matrix(rm).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def pose7_to_mat44(p7):
    """p7 = (x,y,z, w,qx,qy,qz)。"""
    M = np.eye(4)
    M[:3, :3] = quat_wxyz_to_rotmat(p7[3:])
    M[:3, 3] = p7[:3]
    return M


def mat44_to_pose7(M):
    q = rotmat_to_quat_wxyz(M[:3, :3])
    return np.concatenate([M[:3, 3], q])


# ---- 焊缝 I/O（父 solve_arm_pose_parallel.py L191–304 移植） ----
SHORT_SEAM_LEN_M = 0.03   # 短焊缝阈值（米）：焊缝端点距 < 此值视为过短（=3cm），--filter-short 时丢弃不求解


def _seam_len_m(weld: Dict) -> float:
    """焊缝长度（米）= ‖p1 - p0‖（mesh/世界坐标，与 standoff 同单位=米）。"""
    return float(np.linalg.norm(np.asarray(weld["p1_world"], dtype=np.float64)
                                - np.asarray(weld["p0_world"], dtype=np.float64)))


def load_welds(json_fp: str) -> List[Dict]:
    """读 _weld_angle3.json；corrected_p0/p1 + bisector 直接当世界坐标（工件 mesh 系）。"""
    import json
    with open(json_fp, "r") as f:
        data = json.load(f)
    out = []
    for i, w in enumerate(data):
        p0 = np.array(w["corrected_p0"], dtype=np.float64)
        p1 = np.array(w["corrected_p1"], dtype=np.float64)
        bis = np.array(w["bisector"], dtype=np.float64)
        bis = bis / (np.linalg.norm(bis) + 1e-12)
        bd = w.get("boundary_dirs")
        bd_arr = np.array(bd, dtype=np.float64) if (bd is not None and len(bd) == 2) \
            else np.zeros((2, 3))
        out.append({
            "idx": i,
            "p0_world": p0,
            "p1_world": p1,
            "mid_world": 0.5 * (p0 + p1),
            "bisector_world": bis,
            "boundary_dirs": bd_arr,
            "raw": w,
        })
    return out


def _interpolate_line(p0, p1, n: int = 10):
    ts = np.linspace(0.0, 1.0, n)
    return p0[None, :] * (1.0 - ts[:, None]) + p1[None, :] * ts[:, None]


def save_seam_pkl(out_dir, workpiece_stem, weld, target, obj_fp, n_seg: int = 20):
    """把单条焊缝解存成 reformate_save 风格 pickle（schema 与 solve_arm_pose_* 一致）。
    target=None（失败）不写文件——与示例数据一致（失败焊缝缺 pkl）。"""
    if target is None:
        return None
    save_subdir = os.path.join(out_dir, workpiece_stem)
    os.makedirs(save_subdir, exist_ok=True)
    fp = os.path.join(save_subdir, f"seam_{weld['idx']}.pkl")

    p0 = np.asarray(weld["p0_world"], dtype=np.float64)
    p1 = np.asarray(weld["p1_world"], dtype=np.float64)
    seam_line = _interpolate_line(p0, p1, n=n_seg)
    direction = p1 - p0
    tangent = direction / (np.linalg.norm(direction) + 1e-12)
    seam_tangent = np.tile(tangent[None, :], (n_seg, 1))
    bd = np.asarray(weld.get("boundary_dirs", np.zeros((2, 3))), dtype=np.float64)
    seam_limits = np.tile(bd[None, :, :], (n_seg, 1, 1))

    piece_pose7 = np.asarray(target["wp_world_pose7"], dtype=np.float64)
    base_pose7 = np.asarray(target["base_pose_world_pose7"], dtype=np.float64)
    T_workpiece_world = pose7_to_mat44(piece_pose7)
    T_base_world = pose7_to_mat44(base_pose7)
    robot_pose7 = mat44_to_pose7(np.linalg.inv(T_workpiece_world) @ T_base_world)

    rp0 = np.asarray(target["weld_p0_world"], dtype=np.float64)
    rp1 = np.asarray(target["weld_p1_world"], dtype=np.float64)
    z_offset = abs(rp0[2] - rp1[2])
    xy_offset = float(np.linalg.norm(rp0[:2] - rp1[:2]))
    degree = 90.0 if xy_offset < 1e-6 else float(np.rad2deg(np.arctan(z_offset / xy_offset)))
    horiz = 0 if degree < 3.0 else (1 if degree > 87.0 else 2)

    save_data = {
        "seam_line":         seam_line,
        "seam_tangent":      seam_tangent,
        "seam_limits":       seam_limits,
        "robot_pose":        robot_pose7[None, :],
        "piece_pose":        piece_pose7[None, :],
        "horizontal":        np.array([horiz], dtype=np.int64),
        "horizontal_degree": np.array([degree]),
        "obj_fp":            obj_fp,
        "middle":            n_seg // 2,
        "hanfeng_i":         weld["idx"],
        "joint_angles":      np.asarray(target.get("joint_angles", []), dtype=np.float64),
        "joint_names":       list(target.get("joint_names", [])),
    }
    tmp_fp = fp + ".tmp"
    with open(tmp_fp, "wb") as f:
        pickle.dump(save_data, f)
    os.replace(tmp_fp, fp)
    return fp


def to_save_format(weld: Dict, sol: Dict, joint_names: List[str]) -> Dict:
    """best 解 (R,t)=T_workpiece_in_base → save_seam_pkl 期待的 target dict。

    工件在 world = identity（没动），base 在 world = inv(T_workpiece_in_base)。
    （源 solve_arm_pose_lookup.py L545–580 原样。）
    """
    R, t = sol["R"], sol["t"]
    T_wp_in_base = np.eye(4)
    T_wp_in_base[:3, :3] = R
    T_wp_in_base[:3, 3] = t
    T_workpiece_world = np.eye(4)
    T_base_world = np.linalg.inv(T_wp_in_base)
    return {
        "wp_world_pose7": mat44_to_pose7(T_workpiece_world).tolist(),
        "base_pose_world_pose7": mat44_to_pose7(T_base_world).tolist(),
        "weld_p0_world": weld["p0_world"].tolist(),
        "weld_p1_world": weld["p1_world"].tolist(),
        "joint_angles": sol["q"].tolist(),
        "joint_names": joint_names,
        "_rot_x_deg": sol["rot_x_deg"],
        "_rot_y_deg": sol["rot_y_deg"],
        "_rot_z_deg": sol["rot_z_deg"],
        "_align_score": sol["align_score"],
        "_d_link": sol["d_link"],
        "_d_retract": sol["d_retract"],
    }


def _as_n_per_dof_list(n_per_dof, n_dof: int = 6) -> List[int]:
    """把 n_per_dof 规整成「每关节采样档数」列表 [n0..n_{n_dof-1}]：
      · 标量 int   → 各关节同档（与旧行为完全一致，q_table = n^n_dof）；
      · 列表/元组  → 各关节各自档数（须长度 = n_dof，q_table = ∏ ni）。
    用于 link1..link6 分别给不同采样档数。每项须 ≥1。"""
    if isinstance(n_per_dof, (list, tuple, np.ndarray)):
        ns = [int(v) for v in n_per_dof]
        if len(ns) != n_dof:
            raise ValueError(f"n_per_dof 列表长度 {len(ns)} 与关节数 {n_dof} 不一致：{n_per_dof}")
    else:
        ns = [int(n_per_dof)] * n_dof
    if any(v < 1 for v in ns):
        raise ValueError(f"n_per_dof 每项须 ≥1：{ns}")
    return ns


# ---- 求解器（移植 ArmPoseSolver voxel 子集 + Lookup 子类） ----
class InitPoseLookupSolver:
    """查表式初始位姿求解器（裸 cuRobo RobotWorld + 工件 ESDF voxel 碰撞）。

    机器人 yml 取 cfg.robot_cfg_path，retract/joint_names 取 cfg。precompute_joint_table 一次性
    预计算 N=n_per_dof^6 个 q 的 (ee_pos, ee_x, link_spheres)；solve_one_weld_lookup 每条焊缝 batch
    反解 + 碰撞过滤 + 评分。
    """

    def __init__(self, cfg, obj_fp: Optional[str],
                 collision_tolerance: float = 0.03, voxel_size: float = 0.02,
                 n_per_dof: int = 7, clearance_inflate: float = 0.0):
        import gt_gen.compat  # noqa: F401  warp shim，须在 import curobo 前
        from curobo.types.base import TensorDeviceType

        self.cfg = cfg
        self.tensor_args = TensorDeviceType()
        self.obj_fp = obj_fp
        self.collision_tolerance = collision_tolerance
        self.voxel_size = voxel_size
        # 间隙膨胀：所有碰撞球半径 +clearance_inflate 再判碰，等效要求整臂离工件留间隙；0=关闭。
        self.clearance_inflate = float(clearance_inflate)
        self.n_per_dof = n_per_dof

        self._load_robot()
        self.robot_world = None
        self.world_voxel_coll = None
        self._esdf_feature = None
        self._esdf_dims = None
        self._esdf_center_world = None
        # lookup 表
        self.q_table_t = None
        self.ee_pos_t = None
        self.ee_x_t = None
        self.link_spheres_t = None
        self.retract_spheres_t = None
        self.K_link = None
        self.N = None
        if obj_fp is not None:
            self._build_robot_world()

    # ---- 机器人 ----
    def _load_robot(self):
        import gt_gen.compat  # noqa: F401
        from curobo.types.robot import RobotConfig
        from curobo.util_file import load_yaml
        rd = load_yaml(self.cfg.robot_cfg_path)
        self.robot_cfg_dict = rd["robot_cfg"]
        self.robot_cfg = RobotConfig.from_dict(self.robot_cfg_dict, self.tensor_args)
        self.joint_names = self.robot_cfg_dict["kinematics"]["cspace"]["joint_names"]
        self.retract_config = np.array(
            self.robot_cfg_dict["kinematics"]["cspace"]["retract_config"], dtype=np.float64)

    # ---- 工件 ESDF ----
    def _compute_esdf(self, obj_fp: str):
        """用临时 WorldMeshCollision 把工件 mesh 体素化成 signed ESDF（正=内，负=外），缓存。
        （父 solve_arm_pose_parallel.py L503–581 原样；libigl 广义缠绕数纠 sign。）"""
        import gt_gen.compat
        import torch
        import trimesh as _trimesh
        from curobo.geom.types import WorldConfig, Cuboid
        from curobo.geom.sdf.world import CollisionCheckerType, WorldCollisionConfig
        from curobo.geom.sdf.world_mesh import WorldMeshCollision
        gt_gen.compat.apply_trimesh_shim()

        tm = _trimesh.load(obj_fp, force="mesh", process=False)
        if tm.vertices is None or len(tm.vertices) == 0:
            raise RuntimeError(f"invalid mesh: {obj_fp}")
        bbox_min = np.asarray(tm.bounds[0], dtype=np.float64)
        bbox_max = np.asarray(tm.bounds[1], dtype=np.float64)
        center = (bbox_min + bbox_max) / 2.0
        size = (bbox_max - bbox_min) + 4 * self.voxel_size

        mesh_wc = WorldConfig.from_dict({"mesh": {"workpiece": {
            "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "file_path": obj_fp, "scale": [1.0, 1.0, 1.0]}}})
        mesh_cfg = WorldCollisionConfig.load_from_dict(
            {"checker_type": CollisionCheckerType.MESH, "max_distance": 5.0,
             "n_envs": 1, "cache": {"mesh": 1, "obb": 1}},
            mesh_wc, self.tensor_args)
        mesh_coll = WorldMeshCollision(mesh_cfg)

        bbox_cuboid = Cuboid(
            name="workpiece",
            pose=[float(center[0]), float(center[1]), float(center[2]), 1.0, 0.0, 0.0, 0.0],
            dims=size.tolist())
        esdf = mesh_coll.get_esdf_in_bounding_box(bbox_cuboid, voxel_size=self.voxel_size)

        try:
            import igl
            xyzr = esdf.create_xyzr_tensor(transform_to_origin=True, tensor_args=self.tensor_args)
            voxel_centers = xyzr[:, :3].cpu().numpy().astype(np.float64)
            V = np.asarray(tm.vertices, dtype=np.float64)
            F = np.asarray(tm.faces, dtype=np.int64)
            wn = igl.fast_winding_number(V, F, voxel_centers)
            inside_mask = wn > 0.5
            unsigned = esdf.feature_tensor.abs()
            inside_t = torch.from_numpy(inside_mask).to(unsigned.device)
            sign = torch.where(inside_t, torch.ones_like(unsigned), -torch.ones_like(unsigned))
            esdf.feature_tensor = sign * unsigned
            n_in = int(inside_mask.sum()); n_tot = inside_mask.size
            print(f"      [esdf] igl winding number: inside {n_in}/{n_tot} voxels "
                  f"({100.0 * n_in / max(n_tot, 1):.1f}%)")
        except ImportError:
            print("[warn] libigl 未安装, ESDF sign 沿用 cuRobo 默认")
        except Exception as e:
            print(f"[warn] igl winding number failed: {e}; 用 cuRobo 默认 sign")

        self._esdf_feature = esdf.feature_tensor.clone()
        self._esdf_dims = list(esdf.dims)
        self._esdf_center_world = center.copy()
        return esdf

    def _build_robot_world(self):
        """建立 voxel-based RobotWorld for signed collision（父 L583–617 原样）。"""
        import gt_gen.compat  # noqa: F401
        from curobo.geom.types import WorldConfig, VoxelGrid
        from curobo.geom.sdf.world import CollisionCheckerType, WorldCollisionConfig
        from curobo.geom.sdf.world_voxel import WorldVoxelCollision
        from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

        esdf = self._compute_esdf(self.obj_fp)
        init_voxel = VoxelGrid(
            name="workpiece", dims=self._esdf_dims,
            pose=[float(self._esdf_center_world[0]), float(self._esdf_center_world[1]),
                  float(self._esdf_center_world[2]), 1.0, 0.0, 0.0, 0.0],
            voxel_size=self.voxel_size, feature_tensor=self._esdf_feature)
        world_voxel = WorldConfig(voxel=[init_voxel])
        voxel_cfg = WorldCollisionConfig.load_from_dict(
            {"checker_type": CollisionCheckerType.VOXEL, "max_distance": 5.0, "n_envs": 1},
            world_voxel, self.tensor_args)
        self.world_voxel_coll = WorldVoxelCollision(voxel_cfg)
        self.world_voxel_coll.update_voxel_data(init_voxel)

        rwconfig = RobotWorldConfig.load_from_config(
            self.robot_cfg, None, collision_activation_distance=0.0,
            collision_checker_type=CollisionCheckerType.VOXEL,
            world_collision_checker=self.world_voxel_coll, tensor_args=self.tensor_args)
        self.robot_world = RobotWorld(rwconfig)

    def set_workpiece(self, obj_fp: str):
        """切换工件 mesh，重建/更新 ESDF（不重算 q_table，与工件无关）。"""
        from curobo.geom.types import VoxelGrid
        if obj_fp == self.obj_fp and self.robot_world is not None:
            return
        self.obj_fp = obj_fp
        if self.robot_world is None:
            self._build_robot_world()
        else:
            self._compute_esdf(obj_fp)
            new_voxel = VoxelGrid(
                name="workpiece", dims=self._esdf_dims,
                pose=[float(self._esdf_center_world[0]), float(self._esdf_center_world[1]),
                      float(self._esdf_center_world[2]), 1.0, 0.0, 0.0, 0.0],
                voxel_size=self.voxel_size, feature_tensor=self._esdf_feature)
            self.world_voxel_coll.update_voxel_data(new_voxel)

    def _ensure_voxel_pose_set(self):
        """碰撞查询前确保 voxel grid 在 world 系 pose 设对（= ESDF bbox center，identity rot）。"""
        from curobo.types.math import Pose
        voxel_pose7 = np.concatenate([self._esdf_center_world, np.array([1.0, 0.0, 0.0, 0.0])])
        self.world_voxel_coll.update_obstacle_pose(
            name="workpiece",
            w_obj_pose=Pose.from_list(voxel_pose7.tolist(), self.tensor_args))

    def _voxel_collision_distance_batch(self, spheres):
        """spheres (B,K,4)(xyz,r) 在 mesh_world 系 → (B,) max-over-spheres 穿透值（>0 撞）。"""
        x_sph = spheres.unsqueeze(1)  # (B,1,K,4)
        d = self.robot_world.get_collision_distance(x_sph, env_query_idx=None)
        return d.squeeze(-1)

    def recheck_retract_collision_snapped(self, R_wp, t_wp, chunk: int = 2048):
        """snap 后姿态复检：给定候选工件位姿 (R_wp (M,3,3), t_wp (M,3), = T_workpiece_in_base)，
        只用【retract 固定 home 姿态的整臂碰撞球】反变到工件 mesh world 查 ESDF，返回 safe 掩码 (M,) bool
        （True=不撞）。判据与 solve_one_weld_lookup 的碰撞过滤一致（含 clearance_inflate、
        d<=collision_tolerance），区别仅在用 snap 后 (Rv,t_new) 而非 solve 时 (R,t)，且只查 retract、
        不查目标构型 q 的整臂球（retract 球与 q 无关，故无需 q 索引）。"""
        import torch
        device, dtype = self.tensor_args.device, self.tensor_args.dtype
        M = int(np.asarray(R_wp).shape[0])
        safe = torch.zeros(M, dtype=torch.bool, device=device)
        if M == 0:
            return safe
        self._ensure_voxel_pose_set()
        R_t = torch.as_tensor(np.asarray(R_wp, dtype=np.float64), device=device, dtype=dtype)  # (M,3,3)
        t_t = torch.as_tensor(np.asarray(t_wp, dtype=np.float64), device=device, dtype=dtype)  # (M,3)
        ret_xyz = self.retract_spheres_t[:, :3]                 # (K,3) retract 球心（base 系）
        ret_r = self.retract_spheres_t[:, 3]                    # (K,)
        if self.clearance_inflate > 0.0:                        # 间隙膨胀：与 solve 判据一致
            ret_r = ret_r + self.clearance_inflate
        with torch.no_grad():
            for i in range(0, M, chunk):
                R_c = R_t[i:i + chunk]                          # (c,3,3)
                t_c = t_t[i:i + chunk]                          # (c,3)
                c = R_c.shape[0]
                # retract 球反变到 mesh_world：p_world = R^T @ (p_base - t)（einsum 同 solve 的整臂/retract 变换）
                deltas = ret_xyz[None, :, :].expand(c, -1, -1) - t_c[:, None, :]     # (c,K,3)
                ret_xyz_w = torch.einsum("nji,nkj->nki", R_c, deltas)                # (c,K,3)
                ret_spheres_w = torch.cat(
                    [ret_xyz_w, ret_r[None, :].expand(c, -1).unsqueeze(-1)], dim=-1)  # (c,K,4)
                d_ret = self._voxel_collision_distance_batch(ret_spheres_w)          # (c,)
                safe[i:i + chunk] = d_ret <= self.collision_tolerance
        return safe

    # ---- 离线预计算 ----
    def precompute_joint_table(self):
        """6 关节限位等距 n^6 采样 → 逐块 batch FK 算 (ee_pos, ee_x, link_spheres) 并【当场过滤】。
        （源 solve_arm_pose_lookup.py L206–308；改为逐块过滤以免 n_per_dof 大时 n^6 link_spheres OOM。）

        过滤：只保留焊枪末端位姿（base_link 系）同时满足下列约束的关节角——
          ① xy 平面到原点距离 ∈ cfg.plan_init_ee_xy_range（default.yaml plan_init_pose.ee_xy_range_m）；
          ② z ∈ cfg.plan_init_ee_z_range（plan_init_pose.ee_z_range_m）；
          ③ 落在 init_free 盒 [cfg.init_free_box_min, cfg.init_free_box_max] 内（逐轴）。
        每块只把满足约束的幸存者保留并 cat 到 GPU 表，不物化全量。"""
        import torch
        kc = self.robot_cfg.kinematics.kinematics_config
        jl = kc.joint_limits.position
        if jl.shape[0] != 2:
            jl = jl.T
        low = jl[0].cpu().numpy()
        high = jl[1].cpu().numpy()
        n_dof = len(low)
        assert n_dof == 6, f"expected 6 DoF, got {n_dof}"

        ns = _as_n_per_dof_list(self.n_per_dof, n_dof)   # 每关节采样档数 [n0..n_{n_dof-1}]
        N = int(np.prod(ns))
        self.N = N
        print(f"[lookup] sampling {'×'.join(str(v) for v in ns)} = {N} joint configs in limits")
        for i, (lo, hi) in enumerate(zip(low, high)):
            print(f"  joint {i}: [{lo:.3f}, {hi:.3f}]  (n={ns[i]})")

        # —— 过滤范围（base_link 系焊枪末端约束）：从 cfg / default.yaml 读取 ——
        ee_xy_range = self.cfg.plan_init_ee_xy_range
        ee_z_range = self.cfg.plan_init_ee_z_range
        box_min = np.asarray(self.cfg.init_free_box_min, dtype=np.float32)
        box_max = np.asarray(self.cfg.init_free_box_max, dtype=np.float32)
        xy_lo, xy_hi = float(ee_xy_range[0]), float(ee_xy_range[1])
        z_lo, z_hi = float(ee_z_range[0]), float(ee_z_range[1])
        bmin = torch.tensor(box_min, device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        bmax = torch.tensor(box_max, device=self.tensor_args.device, dtype=self.tensor_args.dtype)

        # 逐块生成关节角（按 flat 索引 unravel 取网格值，免物化全量 meshgrid）→ FK → 当场过滤，
        # 只把【满足约束】的幸存者堆到 GPU，避免 n_per_dof 大时 n^6 个 link_spheres OOM。
        grids = [np.linspace(lo, hi, ni).astype(np.float32)
                 for (lo, hi), ni in zip(zip(low, high), ns)]
        shape = tuple(int(v) for v in ns)
        chunk = 128
        q_keep, ee_pos_keep, ee_x_keep, ee_rot_keep, link_keep = [], [], [], [], []
        n_before = N
        n_after = 0
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, N, chunk):
                idxs = np.arange(i, min(i + chunk, N))
                coords = np.unravel_index(idxs, shape)            # n_dof 个索引数组
                qb_np = np.stack([grids[d][coords[d]] for d in range(n_dof)], axis=1)
                qb = torch.tensor(qb_np, device=self.tensor_args.device,
                                  dtype=self.tensor_args.dtype)
                state = self.robot_world.get_kinematics(qb)
                ee_pos = state.ee_position
                ee_x = quat_wxyz_to_x_axis_batch(state.ee_quaternion)
                ee_x = ee_x / (torch.norm(ee_x, dim=-1, keepdim=True) + 1e-12)
                ee_rot = quat_wxyz_to_rotmat_batch(state.ee_quaternion)  # (B,3,3) 末端局部 xyz 轴
                link_sph = state.link_spheres_tensor

                xy_dist = torch.norm(ee_pos[:, :2], dim=-1)
                z_val = ee_pos[:, 2]
                in_box = ((ee_pos >= bmin) & (ee_pos <= bmax)).all(dim=-1)
                keep = (xy_dist >= xy_lo) & (xy_dist <= xy_hi) & \
                       (z_val >= z_lo) & (z_val <= z_hi) & in_box
                if keep.any():
                    q_keep.append(qb[keep].detach().clone())
                    ee_pos_keep.append(ee_pos[keep].detach().clone())
                    ee_x_keep.append(ee_x[keep].detach().clone())
                    ee_rot_keep.append(ee_rot[keep].detach().clone())
                    link_keep.append(link_sph[keep].detach().clone())
                    n_after += int(keep.sum().item())

        print(f"[lookup] 焊枪末端位姿过滤（xy∈[{xy_lo},{xy_hi}]m, z∈[{z_lo},{z_hi}]m, "
              f"init_free 盒 {box_min.tolist()}~{box_max.tolist()}）：{n_before} → {n_after}")
        if n_after == 0:
            raise RuntimeError("[lookup] 过滤后无任何关节角满足约束，请放宽 plan_init_pose 范围")

        self.q_table_t = torch.cat(q_keep, dim=0)
        self.ee_pos_t = torch.cat(ee_pos_keep, dim=0)
        self.ee_x_t = torch.cat(ee_x_keep, dim=0)
        self.ee_rot_t = torch.cat(ee_rot_keep, dim=0)
        self.link_spheres_t = torch.cat(link_keep, dim=0)
        self.K_link = self.link_spheres_t.shape[1]
        self.N = n_after

        retract_q = torch.tensor(self.retract_config[None],
                                 device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        with torch.no_grad():
            retract_state = self.robot_world.get_kinematics(retract_q)
        self.retract_spheres_t = retract_state.link_spheres_tensor[0].detach().clone()

        dt = time.time() - t0
        print(f"[lookup] precomputed table in {dt:.1f}s. q_table {tuple(self.q_table_t.shape)}, "
              f"ee_pos {tuple(self.ee_pos_t.shape)}, link_spheres {tuple(self.link_spheres_t.shape)}, "
              f"retract_spheres {tuple(self.retract_spheres_t.shape)}")

        # CRITICAL sanity：同一 q 用 batch vs 单 q FK 比对 ee_pos
        with torch.no_grad():
            q_test = self.q_table_t[0:1]
            single = self.robot_world.get_kinematics(q_test).ee_position[0].cpu().numpy()
            batch = self.ee_pos_t[0].cpu().numpy()
            print(f"[lookup CRITICAL sanity] BATCH ee_pos[0]={batch.tolist()}")
            print(f"[lookup CRITICAL sanity] SINGLE ee_pos ={single.tolist()}")
            print(f"[lookup CRITICAL sanity] NORM(diff)="
                  f"{float(np.linalg.norm(batch - single)):.6f}")
        print(f"[lookup sanity] ee_pos range "
              f"x=[{float(self.ee_pos_t[:,0].min()):.3f},{float(self.ee_pos_t[:,0].max()):.3f}] "
              f"y=[{float(self.ee_pos_t[:,1].min()):.3f},{float(self.ee_pos_t[:,1].max()):.3f}] "
              f"z=[{float(self.ee_pos_t[:,2].min()):.3f},{float(self.ee_pos_t[:,2].max()):.3f}]")

    # ---- 离线预计算结果落盘 ----
    def save_joint_table(self, path: Optional[str] = None) -> str:
        """把 precompute_joint_table 的结果（与工件无关的关节角查表）存成 .pt，供复用免去 n^6 重算。

        存储地址：path 缺省时取 cfg.plan_init_joint_table_path（= default.yaml plan_init_pose.joint_table_path）。
        保存内容为 precompute_joint_table 产出的全部张量 + 重建/校验所需元信息；ESDF/robot_world 与工件相关、
        不在此保存（换工件时各自重建）。"""
        import torch
        if self.q_table_t is None:
            raise RuntimeError("尚未 precompute_joint_table，无结果可存；请先调用 precompute_joint_table()")
        if path is None:
            path = self.cfg.plan_init_joint_table_path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "q_table_t": self.q_table_t.cpu(),
            "ee_pos_t": self.ee_pos_t.cpu(),
            "ee_x_t": self.ee_x_t.cpu(),
            "ee_rot_t": self.ee_rot_t.cpu(),
            "link_spheres_t": self.link_spheres_t.cpu(),
            "retract_spheres_t": self.retract_spheres_t.cpu(),
            "K_link": int(self.K_link),
            "N": int(self.N),
            "n_per_dof": _as_n_per_dof_list(self.n_per_dof),
            "joint_names": list(self.joint_names),
        }
        torch.save(payload, path)
        print(f"[lookup] joint_table saved -> {path} "
              f"(N={payload['N']}, n_per_dof={payload['n_per_dof']}, K_link={payload['K_link']})")
        return path

    def load_joint_table(self, path: Optional[str] = None) -> bool:
        """从 .pt 读回 precompute_joint_table 的结果，填回查表张量（免去 n^6 重算）。

        存储地址：path 缺省时取 cfg.plan_init_joint_table_path（= default.yaml plan_init_pose.joint_table_path）。
        文件不存在返回 False（调用方可回退到 precompute_joint_table）；n_per_dof 与当前 solver 不一致则报错
        （查表是按 n_per_dof 采样的，混用会算错）。张量按 tensor_args 搬到当前设备。
        注意：只恢复与工件无关的关节表，ESDF/robot_world 仍由构造/ set_workpiece 各自重建。"""
        import torch
        if path is None:
            path = self.cfg.plan_init_joint_table_path
        if not os.path.isfile(path):
            print(f"[lookup] joint_table 不存在，未加载：{path}")
            return False
        payload = torch.load(path, map_location=self.tensor_args.device)
        saved_raw = payload.get("n_per_dof", None)
        if saved_raw is None:
            raise RuntimeError(
                f"joint_table 缺 n_per_dof 元信息，无法校验：{path}（请重新 precompute_joint_table+save_joint_table）")
        saved_n = _as_n_per_dof_list(saved_raw)
        cur_n = _as_n_per_dof_list(self.n_per_dof)
        if saved_n != cur_n:
            raise RuntimeError(
                f"joint_table n_per_dof={saved_n} 与当前 solver n_per_dof={cur_n} 不一致："
                f"{path}（请用一致的 n_per_dof，或重新 precompute_joint_table+save_joint_table）")
        dev, dt = self.tensor_args.device, self.tensor_args.dtype
        self.q_table_t = payload["q_table_t"].to(device=dev, dtype=dt)
        self.ee_pos_t = payload["ee_pos_t"].to(device=dev, dtype=dt)
        self.ee_x_t = payload["ee_x_t"].to(device=dev, dtype=dt)
        self.ee_rot_t = payload["ee_rot_t"].to(device=dev, dtype=dt)
        self.link_spheres_t = payload["link_spheres_t"].to(device=dev, dtype=dt)
        self.retract_spheres_t = payload["retract_spheres_t"].to(device=dev, dtype=dt)
        self.K_link = int(payload["K_link"])
        self.N = int(payload["N"])
        print(f"[lookup] joint_table loaded <- {path} "
              f"(N={self.N}, n_per_dof={saved_n}, K_link={self.K_link})")
        return True

    # ---- 在线求解 ----
    def solve_one_weld_lookup(self, weld: Dict,
                              rot_x_deg: Tuple[float, ...] = (0.0,),
                              rot_y_deg: Tuple[float, ...] = (0.0,),
                              rot_z_deg: Tuple[float, ...] = (0.0,),
                              diagnostic: bool = False,
                              profile: bool = False,
                              orient_valid: Optional[list] = None,
                              orient_snap_deg: Optional[float] = None) -> Optional[Dict]:
        """每条焊缝 batch GPU 求解。先 align(bisector→名义焊枪轴 -ee_x) 得基准姿态 R0，再【绕
        xiaoyu_tip_link 末端局部 x/y/z 轴】分别转 (αx,βy,γz) 组成扰动 R_delta：R=R_delta@R0；
        t=ee_pos−R@mid → 球反变到 mesh world → ESDF batch 碰撞 → safe → 三轴 90° 偏差和评分取 best。
        （角度参数化由 θ(绕焊缝法向)×φ(绕切向) 改为绕末端自身 xyz 轴，三角全有效，绕 z 即焊枪自转 roll；
        三轴各按 default.yaml plan_init_pose.rot_{x,y,z}_deg 采样后取笛卡尔积。）"""
        import torch
        device = self.tensor_args.device
        dtype = self.tensor_args.dtype

        mid = torch.tensor(weld["mid_world"], device=device, dtype=dtype)
        bisector = torch.tensor(weld["bisector_world"], device=device, dtype=dtype)
        bisector = bisector / (torch.norm(bisector) + 1e-12)

        # 落枪点(tip 目标) = 焊缝中点沿角平分线(bisector，远离工件方向)外移 standoff 米；
        # standoff=0 即落在中点。t=ee_pos−R@target 让 tip 落在此点，真实焊缝中点随之退后 standoff。
        standoff = float(self.cfg.plan_init_standoff)
        target = mid + standoff * bisector

        # 焊缝起点/终点（mesh world）→ 在线按 R,t 变到 base 后须落在与 tip 同一组工作空间范围内
        # （复用 precompute 的 ee_xy_range_m / ee_z_range_m；中点=tip 已在 precompute 保证，无需再算）。
        p0_world = torch.tensor(weld["p0_world"], device=device, dtype=dtype)
        p1_world = torch.tensor(weld["p1_world"], device=device, dtype=dtype)
        xy_lo, xy_hi = float(self.cfg.plan_init_ee_xy_range[0]), float(self.cfg.plan_init_ee_xy_range[1])
        z_lo, z_hi = float(self.cfg.plan_init_ee_z_range[0]), float(self.cfg.plan_init_ee_z_range[1])

        N = self.N
        axis_q = -self.ee_x_t  # (N,3) 名义焊枪轴 = 末端 -x
        axis_q = axis_q / (torch.norm(axis_q, dim=-1, keepdim=True) + 1e-12)
        # 末端 tip_link 局部 x/y/z 轴在 base 的方向（ee_rot_t 各列）→ 扰动绕这三根轴转
        eax = self.ee_rot_t[:, :, 0]
        eay = self.ee_rot_t[:, :, 1]
        eaz = self.ee_rot_t[:, :, 2]
        self._ensure_voxel_pose_set()

        # 基准姿态 R0：把焊缝平分线对齐到名义焊枪轴（零扰动），R0@bisector = axis_q
        with torch.no_grad():
            R0 = batch_align_rotation(bisector, axis_q)  # (N,3,3)

        all_solutions = []
        diag_stats = []
        # solve 内部分段计时（profile=True 时启用；GPU 异步，分段边界需 synchronize 才准）
        _pf = {"pose+端点": 0.0, "碰撞查询(link+ret)": 0.0, "safe_mask+min": 0.0, "评分+取解.cpu": 0.0}
        import time as _time

        def _pnow():
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize()
            return _time.time()

        # 朝向粗筛（可选）：把「snap 到 4 种允许朝向」这个最强约束提到碰撞之前。
        # 判据用【列向量夹角】——snap(euler 三轴<θ) 的宽松超集：euler 各<θ ⇒ R_rel 总转角<3θ ⇒
        # 各列偏移<3θ，故阈值取 3θ+15° 必涵盖所有 snap 命中（召回安全，绝不漏），精确判定仍由 CPU 做。
        orient_Rv_t = None
        orient_cos_thr = None
        if orient_valid is not None and orient_snap_deg is not None and len(orient_valid) > 0:
            orient_Rv_t = torch.tensor(np.stack([np.asarray(R, dtype=np.float64) for R in orient_valid]),
                                       device=device, dtype=dtype)            # (V,3,3)
            _coarse_deg = 3.0 * float(orient_snap_deg) + 15.0
            orient_cos_thr = float(np.cos(np.deg2rad(_coarse_deg)))

        # 工作空间端点判据：对 (...,3) 通用（用 ... 索引），(N,3) 与批量 (Gz,N,3) 共用同一套代码、逐位一致。
        def _in_ws(p):
            xy = torch.norm(p[..., :2], dim=-1)
            z = p[..., 2]
            return (xy >= xy_lo) & (xy <= xy_hi) & (z >= z_lo) & (z <= z_hi)

        # 【优化 A】Rx/Ry/Rz 仅随各自角度档变化（eax/eay/eaz 循环不变）→ 循环外各算一次再复用，
        #   消除原先 325 采样里 975 次 batch_axis_angle_rotmat（实际只 13+5+5 个不同结果）的冗余重算。
        #   Rz 堆成 (Gz,N,3,3) 供【优化 B】把内层 az 一次性批处理。
        with torch.no_grad():
            Rx_list = [batch_axis_angle_rotmat(
                eax, torch.tensor(np.deg2rad(a), device=device, dtype=dtype).expand(N))
                for a in rot_x_deg]                                 # 各 (N,3,3)
            Ry_list = [batch_axis_angle_rotmat(
                eay, torch.tensor(np.deg2rad(a), device=device, dtype=dtype).expand(N))
                for a in rot_y_deg]
            Rz_stack = torch.stack([batch_axis_angle_rotmat(
                eaz, torch.tensor(np.deg2rad(a), device=device, dtype=dtype).expand(N))
                for a in rot_z_deg], dim=0)                         # (Gz,N,3,3)
        n_az = len(rot_z_deg)

        for ix, ax_deg in enumerate(rot_x_deg):
            Rx = Rx_list[ix]
            for iy, ay_deg in enumerate(rot_y_deg):
                Ry = Ry_list[iy]
                _mark = _pnow()
                with torch.no_grad():
                    # 【优化 B】内层 az（Gz 档）一次性批处理 pose+端点+朝向预筛：Python 循环 325→13×5、
                    #   kernel 启动随之减少。矩阵乘分组严格保持原版 ((Rz@Ry)@Rx)@R0——批量 matmul 对每个
                    #   [g,n] 切片即逐采样的 (N,3,3) 乘法、逐位一致；端点/朝向预筛对 Gz 档并行算。
                    T1 = torch.matmul(Rz_stack, Ry)                # (Gz,N,3,3) = Rz_g@Ry
                    T2 = torch.matmul(T1, Rx)                       # = (Rz@Ry)@Rx
                    R_g = torch.matmul(T2, R0)                      # = ((Rz@Ry)@Rx)@R0
                    R_target_g = torch.einsum("gnij,j->gni", R_g, target)
                    t_g = self.ee_pos_t.unsqueeze(0) - R_target_g          # (Gz,N,3) tip 落在 standoff 落枪点
                    R_mid_g = torch.einsum("gnij,j->gni", R_g, mid)
                    mid_base_g = R_mid_g + t_g                             # (Gz,N,3) 真实焊缝中点在 base
                    # 焊缝起/终点变到 base：p_base = R@p_world + t；须落在 ee_xy_range(xy 环)+ee_z_range(z) 内
                    p0_base_g = torch.einsum("gnij,j->gni", R_g, p0_world) + t_g   # (Gz,N,3)
                    p1_base_g = torch.einsum("gnij,j->gni", R_g, p1_world) + t_g
                    endpoints_g = _in_ws(p0_base_g) & _in_ws(p1_base_g)    # (Gz,N) 起+终都在范围内
                    # 朝向粗筛：R 各列与某允许朝向各列夹角均 < 阈值（snap 超集）→ 与端点 AND 成预筛掩码。
                    # 只缩小碰撞查询规模；safe_mask 仍只含 endpoints（朝向精筛在 CPU），结果不变。
                    if orient_Rv_t is not None:
                        cos_g = torch.einsum("gnik,vik->gnvk", R_g, orient_Rv_t)   # (Gz,N,V,3) 各列点积
                        orient_ok_g = (cos_g > orient_cos_thr).all(dim=-1).any(dim=-1)  # (Gz,N)
                        prefilter_g = endpoints_g & orient_ok_g
                    else:
                        prefilter_g = endpoints_g
                if profile:
                    _n = _pnow(); _pf["pose+端点"] += _n - _mark; _mark = _n

                for iz in range(n_az):
                    az_deg = rot_z_deg[iz]
                    with torch.no_grad():
                        R = R_g[iz]                                  # (N,3,3) 该 az 档切片，与逐采样等价
                        t_arr = t_g[iz]
                        mid_base_arr = mid_base_g[iz]
                        endpoints_ok = endpoints_g[iz]               # (N,)
                        prefilter = prefilter_g[iz]
                        # 球反变到 mesh_world：p_world = R^T @ (p_base - t)。
                        # 整臂/retract 碰撞球世界坐标张量 (N,K,4) 单条就 ~1.7GB，幸存者一多即 OOM；
                        # 故【按 chunk 逐块构造 + 查询】，峰值显存只与 chunk 相关、与幸存者总数 N 无关。
                        # 【端点预筛】碰撞是本步最贵的操作，只对「焊缝端点在工作空间内」的候选查；非端点候选
                        # 的 d 填大值（必然不 safe），d_link/d_ret 保持全 N 形状供 diag/索引；结果与全量查询一致。
                        link_xyz_b = self.link_spheres_t[..., :3]
                        link_r_b = self.link_spheres_t[..., 3]
                        ret_xyz_b = self.retract_spheres_t[:, :3]
                        ret_r = self.retract_spheres_t[:, 3]
                        # 间隙膨胀：所有碰撞球半径 +clearance（整臂本体 + retract，无尖端例外）。
                        if self.clearance_inflate > 0.0:
                            link_r_b = link_r_b + self.clearance_inflate
                            ret_r = ret_r + self.clearance_inflate

                        BIG = 1.0e6
                        d_link = torch.full((N,), BIG, device=device, dtype=dtype)
                        d_ret = torch.full((N,), BIG, device=device, dtype=dtype)
                        ep_idx = prefilter.nonzero(as_tuple=True)[0]   # (M,) 预筛(端点∩朝向粗筛)幸存者全局索引
                        M = int(ep_idx.numel())
                        chunk = 2048
                        for i in range(0, M, chunk):
                            sub = ep_idx[i:i + chunk]                 # (c,) 全局索引
                            R_c = R[sub]                              # (c,3,3)
                            t_c = t_arr[sub]                          # (c,3)
                            m = R_c.shape[0]
                            # 整臂连杆球（每条候选用各自 q 的 link_spheres + 该 R,t 反变到 mesh world）
                            link_r_c = link_r_b[sub]                  # (c,K)
                            deltas_link_c = link_xyz_b[sub] - t_c[:, None, :]
                            link_xyz_w_c = torch.einsum("nji,nkj->nki", R_c, deltas_link_c)
                            link_spheres_w_c = torch.cat(
                                [link_xyz_w_c, link_r_c.unsqueeze(-1)], dim=-1)
                            # retract 球（球本身与 q 无关，但每条候选用各自 R,t 反变）
                            deltas_ret_c = ret_xyz_b[None, :, :].expand(m, -1, -1) - t_c[:, None, :]
                            ret_xyz_w_c = torch.einsum("nji,nkj->nki", R_c, deltas_ret_c)
                            ret_spheres_w_c = torch.cat(
                                [ret_xyz_w_c, ret_r[None, :].expand(m, -1).unsqueeze(-1)], dim=-1)
                            d_link[sub] = self._voxel_collision_distance_batch(link_spheres_w_c)
                            d_ret[sub] = self._voxel_collision_distance_batch(ret_spheres_w_c)
                    if profile:
                        _n = _pnow(); _pf["碰撞查询(link+ret)"] += _n - _mark; _mark = _n

                    with torch.no_grad():
                        safe_mask = (d_link <= self.collision_tolerance) & \
                                    (d_ret <= self.collision_tolerance) & \
                                    endpoints_ok
                        n_safe = int(safe_mask.sum().item())
                        n_ep = M   # 起+终都在范围内的候选数（= 端点预筛幸存者数）
                        # min 只对端点幸存者有意义（非幸存者 d 是占位大值）；M=0 时填 0
                        dl_min = float(d_link[ep_idx].min()) if M > 0 else 0.0
                        dr_min = float(d_ret[ep_idx].min()) if M > 0 else 0.0
                        diag_stats.append((ax_deg, ay_deg, az_deg, n_safe,
                                           dl_min, dr_min, n_ep))
                    if profile:
                        _n = _pnow(); _pf["safe_mask+min"] += _n - _mark; _mark = _n
                    if diagnostic:
                        print(f"      [αx={ax_deg:+.0f}° βy={ay_deg:+.0f}° γz={az_deg:+.0f}°] "
                              f"N={N} safe={n_safe} min_d_link={dl_min:.4f} "
                              f"min_d_ret={dr_min:.4f}")
                    if not safe_mask.any():
                        if profile:
                            _mark = _pnow()   # 丢弃本档诊断打印耗时（同原版 continue→下轮重置）
                        continue
                    with torch.no_grad():
                        safe_idx = safe_mask.nonzero(as_tuple=True)[0]
                        R_safe = R[safe_idx]
                        scores = axis_align_score_batch(R_safe)   # 三轴 90° 偏差和(取负)，越大越对齐

                    top_per = min(50, safe_idx.shape[0])
                    top_local = torch.argsort(scores, descending=True)[:top_per]
                    # 批量 GPU→CPU：原先逐解 .cpu()/.item()（每条 9 次微传输 → 上万条共 ~14 万次 kernel
                    # 启动延迟）改成每张量整批传一次，再在 numpy 里按行取。索引/取值完全一致，只是省掉微传输。
                    sel = safe_idx[top_local]                       # (top_per,) 全局索引
                    sel_np = sel.cpu().numpy()
                    q_np = self.q_table_t[sel].cpu().numpy()        # (top_per, dof)
                    R_np = R[sel].cpu().numpy()                     # (top_per,3,3)
                    t_np = t_arr[sel].cpu().numpy()
                    ee_np = self.ee_pos_t[sel].cpu().numpy()
                    mid_np = mid_base_arr[sel].cpu().numpy()
                    eex_np = self.ee_x_t[sel].cpu().numpy()
                    dl_np = d_link[sel].cpu().numpy()
                    dr_np = d_ret[sel].cpu().numpy()
                    sc_np = scores[top_local].cpu().numpy()
                    for k in range(sel_np.shape[0]):
                        all_solutions.append({
                            "q_idx": int(sel_np[k]),
                            "q": q_np[k],
                            "R": R_np[k],
                            "t": t_np[k],
                            "rot_x_deg": ax_deg,
                            "rot_y_deg": ay_deg,
                            "rot_z_deg": az_deg,
                            "ee_pos_in_base": ee_np[k],
                            "mid_in_base": mid_np[k],
                            "ee_x_in_base": eex_np[k],
                            "d_link": float(dl_np[k]),
                            "d_retract": float(dr_np[k]),
                            "align_score": float(sc_np[k]),
                            "combined_score": float(sc_np[k]),
                        })
                    if profile:
                        _pf["评分+取解.cpu"] += _pnow() - _mark; _mark = _pnow()

        if profile:
            _tot = sum(_pf.values())
            n_ang = len(rot_x_deg) * len(rot_y_deg) * len(rot_z_deg)
            print(f"      [solve profile] {n_ang} 个角度采样，N={N} 候选/采样，内部分段累计：")
            for _k, _v in _pf.items():
                print(f"      [solve profile]   {_k:22s} {_v:8.3f}s  ({100.0 * _v / max(_tot, 1e-9):5.1f}%)")
            print(f"      [solve profile]   {'内部合计':22s} {_tot:8.3f}s")

        if not all_solutions:
            n_zero = sum(1 for s in diag_stats if s[3] == 0)
            if diagnostic:                       # 详细逐角度 stats 只在 --diagnostic 时打印（批处理时太吵）
                print(f"      [FAIL diag] weld {weld['idx']}: 所有 {len(diag_stats)} 个 (αx,βy,γz) 采样 stats:")
                for axd, ayd, azd, sn, dl, dr, nep in sorted(diag_stats, key=lambda x: (x[0], x[1], x[2])):
                    reason = "OK" if sn else ("端点超范围" if nep == 0
                                              else "RETRACT撞" if dr > self.collision_tolerance
                                              else "LINK撞" if dl > self.collision_tolerance else "其他")
                    print(f"        αx={axd:+4.0f}° βy={ayd:+4.0f}° γz={azd:+4.0f}°: safe={sn:6d}/{N}  "
                          f"端点OK={nep:6d}/{N}  min_d_link={dl:.4f}  min_d_ret={dr:.4f}  [{reason}]")
            print(f"      [FAIL diag] weld {weld['idx']}: 0 解（{n_zero}/{len(diag_stats)} 个采样 safe=0；"
                  f"加 --diagnostic 看逐角度明细）")
            self.last_all_solutions = []
            return None

        all_solutions.sort(key=lambda s: -s["combined_score"])
        # 二次稳定排序：真实焊缝中点(base 系，standoff>0 时 ≠ ee_pos)的 x>0 的结果排前、x<=0 的拍到后面；
        # 稳定排序保证各组内部仍保持上面的 combined_score 降序。
        all_solutions.sort(key=lambda s: 0 if float(s["mid_in_base"][0]) > 0.0 else 1)
        self.last_all_solutions = all_solutions   # 供可视化分页浏览全部候选（x>0 优先、组内按分降序）
        best = all_solutions[0]
        if diagnostic:
            R, t = best["R"], best["t"]
            target_np = np.asarray(weld["mid_world"], float) + standoff * np.asarray(weld["bisector_world"], float)
            target_in_base = R @ target_np + t
            err_tip_pos = float(np.linalg.norm(target_in_base - best["ee_pos_in_base"]))  # 落枪点应贴 ee_pos
            so3_err = float(np.linalg.norm(R @ R.T - np.eye(3)))
            print(f"      [sanity best] standoff={standoff:.3f}m err_tip_pos={err_tip_pos:.4f} "
                  f"|R*R^T-I|={so3_err:.4f} align={best['align_score']:.2f} "
                  f"αx={best['rot_x_deg']:+.0f}° βy={best['rot_y_deg']:+.0f}° γz={best['rot_z_deg']:+.0f}°")
        return best



# ---- 求解结果可视化（复用 init_space_geometries） ----
def _lookup_solution_geoms(cfg, obj_fp: str, weld: Dict, sol: Dict):
    """构建单个解的可视化几何体列表（不开窗）：init_space_geometries（整臂碰撞球 + init_free
    盒 + base 架）+ 按 T_workpiece_in_base 摆放的工件网格（浅灰半透）+ 绿色焊缝线 + 焊枪头落点小球。"""
    import open3d as o3d

    geoms, _ = init_space_geometries(cfg, q=sol["q"])

    R = np.asarray(sol["R"], float)
    t = np.asarray(sol["t"], float)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t  # T_workpiece_in_base

    # 工件网格（浅灰）
    mesh = o3d.io.read_triangle_mesh(obj_fp)
    if not mesh.has_vertices():
        print(f"[warn] open3d 读不到工件网格: {obj_fp}（跳过工件叠加）")
    else:
        mesh.transform(T)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.72])
        geoms.append(mesh)

    # 焊缝线 p0→p1（绿色）+ 焊缝中点（绿球）+ 焊枪头实际落点（红球，= standoff 落枪点，应贴在 ee_pos）
    def _to_base(p):
        return (R @ np.asarray(p, float) + t).tolist()
    p0b, p1b = _to_base(weld["p0_world"]), _to_base(weld["p1_world"])
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([p0b, p1b]),
        lines=o3d.utility.Vector2iVector([[0, 1]]))
    ls.paint_uniform_color([0.1, 0.85, 0.1])
    geoms.append(ls)
    tip = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    tip.translate(_to_base(weld["mid_world"]))
    tip.compute_vertex_normals()
    tip.paint_uniform_color([0.1, 0.85, 0.1])
    geoms.append(tip)

    # standoff 落枪点：中点沿 bisector(远离工件)外移 standoff 米，焊枪头实际落在这（standoff>0 时与中点分离）
    standoff = float(cfg.plan_init_standoff)
    if standoff != 0.0:
        target_world = np.asarray(weld["mid_world"], float) + standoff * np.asarray(weld["bisector_world"], float)
        gun = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        gun.translate(_to_base(target_world))
        gun.compute_vertex_normals()
        gun.paint_uniform_color([0.9, 0.1, 0.1])
        geoms.append(gun)
    return geoms


def show_lookup_solution(cfg, obj_fp: str, weld: Dict, sol: Dict):
    """单解可视化（关闭窗口即返回）。"""
    import open3d as o3d
    geoms = _lookup_solution_geoms(cfg, obj_fp, weld, sol)
    print(f"显示 weld {weld['idx']} 解（关闭窗口继续）…")
    o3d.visualization.draw_geometries(
        geoms, window_name=f"plan_init_pose: weld {weld['idx']} 解 + 工件 + 整臂碰撞球")


def show_lookup_solutions(cfg, obj_fp: str, weld: Dict, solutions: List[Dict]):
    """逐个可视化该焊缝【所有候选解】(按 combined_score 已降序，第 1 个=best)：同一个窗口里
    按【C 键】切到下一个，不关窗口；切换时打印「第 i/N 个结果」。到最后一个再按 C 即关闭窗口
    （进入下一条焊缝 / 结束）。中途也可直接关窗口跳过剩余。"""
    import open3d as o3d

    n = len(solutions)
    if n == 0:
        print(f"[viz] weld {weld['idx']} 无候选解，跳过可视化")
        return

    state = {"i": 0}

    def _load(vis, idx, reset):
        vis.clear_geometries()
        sol = solutions[idx]
        for g in _lookup_solution_geoms(cfg, obj_fp, weld, sol):
            vis.add_geometry(g, reset_bounding_box=reset)
        print(f"[viz] weld {weld['idx']} 第 {idx + 1}/{n} 个结果："
              f"align={sol.get('align_score', float('nan')):.2f} "
              f"αx={sol['rot_x_deg']:+.0f}° βy={sol['rot_y_deg']:+.0f}° γz={sol['rot_z_deg']:+.0f}° "
              f"d_link={sol['d_link']:.3f} d_ret={sol['d_retract']:.3f}（按 C 看下一个）")

    def _next(vis):
        state["i"] += 1
        if state["i"] >= n:
            print(f"[viz] weld {weld['idx']} 已是最后一个（{n}/{n}），关闭窗口继续")
            vis.close()
            return False
        _load(vis, state["i"], reset=False)   # 切换不重置视角，保留用户当前相机
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"plan_init_pose: weld {weld['idx']} 全部候选（按 C 切下一个）")
    vis.register_key_callback(ord("C"), _next)
    _load(vis, 0, reset=True)   # 第 1 个 fit 一次视角
    vis.run()
    vis.destroy_window()


def viz_joint_table(cfg, solver, n=3):
    """可视化 precompute_joint_table 过滤后的结果：焊枪末端 ee_pos 的 xyz 范围（青色 AABB +
    全体落点灰点云）+ 随机挑 n 个关节角各画一套整臂碰撞球（不同颜色区分）+ base 坐标架；
    每个被挑构型再在其 ee_pos 处放一个【洋红实心球】标出焊枪末端，便于核对末端位置是否正确。

    无「是否可视化」开关：由调用处自行注释 / 取消注释这行调用来控制是否运行。n 为可视化的
    关节角个数（调用处写死 3）。"""
    import open3d as o3d
    from gt_gen.obstacle_placement import compute_link_sweep

    ee = solver.ee_pos_t.cpu().numpy()        # (N,3) base_link 系焊枪末端位置
    q_tab = solver.q_table_t.cpu().numpy()    # (N, dof)
    N = int(ee.shape[0])
    if N == 0:
        print("[viz_table] 过滤后无关节角，跳过可视化")
        return
    lo = ee.min(axis=0)
    hi = ee.max(axis=0)
    print(f"[viz_table] ee_pos xyz 范围：x=[{lo[0]:.3f},{hi[0]:.3f}] "
          f"y=[{lo[1]:.3f},{hi[1]:.3f}] z=[{lo[2]:.3f},{hi[2]:.3f}]（N={N}）")

    geoms = []
    # 全体焊枪末端落点（灰点云）
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ee))
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    geoms.append(pcd)
    # xyz 范围 AABB（青色线框）
    aabb = o3d.geometry.AxisAlignedBoundingBox(lo.tolist(), hi.tolist())
    aabb.color = (0.0, 0.75, 0.75)
    geoms.append(aabb)
    # base 坐标架（原点，0.3m）
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=(0.0, 0.0, 0.0)))

    # 随机挑 n 个关节角，各画一套整臂碰撞球（不同颜色）
    k = min(int(n), N)
    pick = np.random.choice(N, size=k, replace=False)
    print(f"[viz_table] 随机挑 {k} 个关节角可视化：idx={pick.tolist()}")
    palette = [[0.85, 0.1, 0.1], [0.1, 0.1, 0.85], [0.1, 0.7, 0.1],
               [0.85, 0.6, 0.1], [0.6, 0.1, 0.85]]
    EE_COLOR = [1.0, 0.0, 1.0]                        # 焊枪末端标记：洋红实心球（与臂碰撞球区分）
    for ci, j in enumerate(pick):
        q = [float(v) for v in q_tab[j]]
        per_wp, _ = compute_link_sweep(cfg, [q], cfg.collision_link_names)
        col = palette[ci % len(palette)]
        for _ln, s in per_wp.items():
            for c in np.asarray(s, float)[0]:                # (S,4)：取唯一路点
                cx, cy, cz, r = (float(v) for v in c)
                if r <= 1e-4:
                    continue
                ball = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=8)
                ball.translate((cx, cy, cz))
                ls = o3d.geometry.LineSet.create_from_triangle_mesh(ball)
                ls.paint_uniform_color(col)
                geoms.append(ls)
        # 焊枪末端位置 ee_pos_t[j]（base 系）：洋红实心球，核对是否落在该构型臂端
        ee_mark = o3d.geometry.TriangleMesh.create_sphere(radius=0.03, resolution=12)
        ee_mark.translate(tuple(float(v) for v in ee[j]))
        ee_mark.compute_vertex_normals()
        ee_mark.paint_uniform_color(EE_COLOR)
        geoms.append(ee_mark)
        print(f"[viz_table]   idx={int(j)} ee_pos={np.round(ee[j], 3).tolist()}（洋红球）")

    print("[viz_table] 显示中（关闭窗口继续）…")
    o3d.visualization.draw_geometries(
        geoms, window_name="precompute_joint_table: ee 范围 + 随机关节角")

def viz_tip_frame(cfg, solver, anchor_idx=None,
                  torch_links=("xiaoyu_accessory_link", "xiaoyu_tip_link"),
                  axis_len=0.25):
    """可视化【焊枪碰撞球】+【xiaoyu_tip_link(ee_link) 末端坐标系 xyz 三轴】，肉眼核实焊枪实际沿哪根
    局部轴指出去 → 绕那根轴转才是「焊枪自转 roll」。

    取一个 anchor 关节角，用其 FK 的末端位姿（solver.ee_pos_t / ee_rot_t）当 tip 坐标系：在 tip
    原点画一个【有朝向的三轴坐标架】（open3d 约定 x=红 y=绿 z=蓝），并把代码里的【名义焊枪轴 -x】
    单独用洋红粗线标出；同时用该构型 FK 出 torch_links 的碰撞球（青色线框）。这样「焊枪整支朝向」
    vs「三轴方向」一眼对照：碰撞球朝哪根轴延伸，绕那根轴转就是 roll（指向不变）。

    判读要点：solve_one_weld_lookup 用 axis_q=-ee_x 当名义焊枪轴 → 理论 roll 轴 = 局部 x（rot_x_deg）。
    若碰撞球确实沿 -x（洋红线）方向延伸，则确认 rot_x_deg 才是 roll、rot_y/z_deg 是 tilt。
    无开关：调用处注释/取消注释这行调用控制是否运行。"""
    import open3d as o3d
    from gt_gen.obstacle_placement import compute_link_sweep

    ee = solver.ee_pos_t.cpu().numpy()         # (N,3) base 系末端位置
    ee_rot = solver.ee_rot_t.cpu().numpy()     # (N,3,3) 各列=局部 x/y/z 轴在 base 的方向
    q_tab = solver.q_table_t.cpu().numpy()     # (N,dof)
    N = int(ee.shape[0])
    if N == 0:
        print("[viz_tip_frame] 空表，跳过")
        return

    a0 = int(anchor_idx) if anchor_idx is not None else int(np.random.randint(N))
    p = ee[a0]                                  # tip 原点（base 系）
    R = ee_rot[a0]                              # tip 坐标系（列=x/y/z 轴）
    ax_x, ax_y, ax_z = R[:, 0], R[:, 1], R[:, 2]
    print(f"[viz_tip_frame] anchor idx={a0} tip_xyz={np.round(p, 3).tolist()}")
    print(f"[viz_tip_frame]   局部 +x={np.round(ax_x, 3).tolist()} (红)  "
          f"+y={np.round(ax_y, 3).tolist()} (绿)  +z={np.round(ax_z, 3).tolist()} (蓝)")
    print(f"[viz_tip_frame]   名义焊枪轴 -x={np.round(-ax_x, 3).tolist()}（洋红粗线；绕它转=roll）")

    # base 坐标架（原点，0.3m）
    geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=(0.0, 0.0, 0.0))]

    # tip 坐标系：有朝向的三轴架（x=红 y=绿 z=蓝），原点在 tip、姿态=R
    tip_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_len, origin=(0.0, 0.0, 0.0))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
    tip_frame.transform(T)
    geoms.append(tip_frame)

    # 名义焊枪轴 -x：洋红线（tip 原点沿 -x 方向画 1.3*axis_len）
    torch_axis_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([p.tolist(), (p - ax_x * axis_len * 1.3).tolist()]),
        lines=o3d.utility.Vector2iVector([[0, 1]]))
    torch_axis_line.paint_uniform_color([1.0, 0.0, 1.0])
    geoms.append(torch_axis_line)

    # tip 原点小球（洋红实心）
    mk = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=12)
    mk.translate(tuple(float(v) for v in p))
    mk.compute_vertex_normals()
    mk.paint_uniform_color([1.0, 0.0, 1.0])
    geoms.append(mk)

    # 焊枪碰撞球（anchor 构型 FK，青色线框）
    q_anchor = [float(v) for v in q_tab[a0]]
    per_wp, _ = compute_link_sweep(cfg, [q_anchor], cfg.collision_link_names)
    n_sph = 0
    for ln in torch_links:
        if ln not in per_wp:
            print(f"[viz_tip_frame] 警告：torch link '{ln}' 不在 collision_link_names，跳过")
            continue
        for c in np.asarray(per_wp[ln], float)[0]:                # (S,4)：取唯一路点
            cx, cy, cz, r = (float(v) for v in c)
            if r <= 1e-4:
                continue
            ball = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=8)
            ball.translate((cx, cy, cz))
            ls = o3d.geometry.LineSet.create_from_triangle_mesh(ball)
            ls.paint_uniform_color([0.1, 0.6, 0.6])
            geoms.append(ls)
            n_sph += 1
    print(f"[viz_tip_frame] 焊枪碰撞球 {n_sph} 个（青色线框；links={list(torch_links)}）")
    print("[viz_tip_frame] 坐标架：x=红 y=绿 z=蓝；洋红粗线=名义焊枪轴(-x)。显示中（关闭窗口继续）…")
    o3d.visualization.draw_geometries(
        geoms, window_name="tip_frame：焊枪碰撞球 + xiaoyu_tip_link 三轴")


# ============================================================================
# CLI
# ============================================================================
def _deg_range(spec):
    """[min,max,step]（度，闭区间）→ 角度元组。step<=0 或 max<=min 时只取 min（该轴不转）。"""
    lo, hi, step = float(spec[0]), float(spec[1]), float(spec[2])
    if step <= 0.0 or hi <= lo:
        return (lo,)
    k = int(round((hi - lo) / step))
    return tuple(round(lo + i * step, 6) for i in range(k + 1))


def _run_solve(args):
    """lookup 求解子流程（--solve）。"""
    from gt_gen.config import load_config
    cfg = load_config()
    # n_per_dof / 绕末端 xyz 轴的角度采样 全部读 default.yaml plan_init_pose（不再走命令行）
    n_per_dof = cfg.plan_init_n_per_dof
    rot_x = _deg_range(cfg.plan_init_rot_x_deg)
    rot_y = _deg_range(cfg.plan_init_rot_y_deg)
    rot_z = _deg_range(cfg.plan_init_rot_z_deg)
    print(f"[solve] obj={args.obj}")
    print(f"[solve] weld-json={args.weld_json}")
    print(f"[solve] 绕末端局部轴采样 αx={list(rot_x)}° βy={list(rot_y)}° γz={list(rot_z)}° "
          f"总采样={len(rot_x)}×{len(rot_y)}×{len(rot_z)}="
          f"{len(rot_x) * len(rot_y) * len(rot_z)}; q_table={n_per_dof}^6={n_per_dof ** 6}")

    solver = InitPoseLookupSolver(
        cfg, args.obj, collision_tolerance=cfg.plan_init_collision_tolerance,
        voxel_size=cfg.plan_init_voxel_size, n_per_dof=n_per_dof)
    # 优先读 cfg.plan_init_joint_table_path 的缓存（与工件无关）；没有则现算一次并落盘复用。
    if solver.load_joint_table():
        print("[solve] 复用已存的 joint table（跳过 n^6 预计算）")
    else:
        print("[solve] precomputing joint table（与工件无关，仅一次）…")
        solver.precompute_joint_table()
        solver.save_joint_table()   # 落盘 → cfg.plan_init_joint_table_path（下次可复用免重算）

    # 可视化 precompute_joint_table 结果（ee xyz 范围 + 随机 3 个关节角）；注释此行即关闭可视化
    # viz_joint_table(cfg, solver, n=1)

    # 可视化焊枪碰撞球 + xiaoyu_tip_link 末端三轴（核实焊枪沿哪根局部轴=roll 轴）；注释此行即关闭可视化
    # viz_tip_frame(cfg, solver)

    welds = load_welds(args.weld_json)
    if args.welds is not None:
        want = set(args.welds)
        welds = [w for w in welds if w["idx"] in want]
        missing = sorted(want - {w["idx"] for w in welds})
        if missing:
            print(f"[solve] 警告：--welds 指定的 idx {missing} 不存在，已忽略")
        if not welds:
            raise SystemExit(f"[solve] --welds {sorted(want)} 在 {args.weld_json} 中均不存在")
    stem = os.path.splitext(os.path.basename(args.obj))[0]

    n_solved = 0
    for i, w in enumerate(welds):
        ts = time.time()
        sol = solver.solve_one_weld_lookup(w, rot_x, rot_y, rot_z, diagnostic=args.diagnostic)
        dt = (time.time() - ts) * 1000.0
        if sol is None:
            print(f"[{i + 1:3d}/{len(welds)}] weld {w['idx']}: FAIL ({dt:.0f}ms)")
            continue
        n_solved += 1
        print(f"[{i + 1:3d}/{len(welds)}] weld {w['idx']}: OK ({dt:.0f}ms) "
              f"αx={sol['rot_x_deg']:+.0f}° βy={sol['rot_y_deg']:+.0f}° γz={sol['rot_z_deg']:+.0f}° "
              f"align={sol['align_score']:.2f} "
              f"d_link={sol['d_link']:.3f} d_ret={sol['d_retract']:.3f} "
              f"q={np.round(sol['q'], 3).tolist()}")
        target = to_save_format(w, sol, solver.joint_names)
        fp = save_seam_pkl(args.out_dir, stem, w, target, args.obj, n_seg=cfg.plan_init_n_seg)
        print(f"      saved → {fp}")
        if args.viz:
            show_lookup_solutions(cfg, args.obj, w, solver.last_all_solutions)

    print(f"[solve] done: {n_solved}/{len(welds)} 条求解成功，输出目录 "
          f"{os.path.join(args.out_dir, stem)}")
    print("PLAN_INIT_POSE_OK")


def main():
    ap = argparse.ArgumentParser(
        description="① 不带 --solve：init 空间可视化；② 带 --solve：lookup 初始位姿求解。")
    # ① init-viz
    ap.add_argument("--q", nargs="+", type=float, default=None,
                    help="init-viz：构型(rad)；缺省用 cfg.retract_config")
    ap.add_argument("--solid", action="store_true",
                    help="init-viz：碰撞球画实心球（默认红色线框）")
    # ② solve
    ap.add_argument("--solve", action="store_true", help="进入 lookup 初始位姿求解模式")
    ap.add_argument("--obj", default=None, help="solve：工件 _part.obj")
    ap.add_argument("--weld-json", default=None, help="solve：焊缝 _weld_angle3.json")
    ap.add_argument("--out-dir", default="/tmp/plan_init_pose", help="solve：seam pkl 输出根目录")
    ap.add_argument("--welds", nargs="+", type=int, default=None,
                    help="solve：指定处理第几个焊缝(按 weld idx，可多个，如 --welds 0 3 5)；缺省=全部")
    ap.add_argument("--viz", action="store_true",
                    help="solve：每条成功焊缝开窗可视化（复用 show_init_space 的几何）")
    ap.add_argument("--diagnostic", action="store_true", help="solve：逐 (θ,φ) 诊断打印")
    # ③ 新逻辑（lookup 关节角采样 + 允许朝向 snap + 正反手分类）
    ap.add_argument("--solve-kejian2", action="store_true",
                    help="进入新逻辑求解（lookup n^6 关节角采样 + 允许朝向 snap，分正反手返回）")
    ap.add_argument("--seam-id", type=int, default=0, help="solve-kejian2：用 weld_json 第几条焊缝（单条）")
    ap.add_argument("--all-seams", action="store_true",
                    help="solve-kejian2：一次处理 weld_json 全部焊缝（工件级 solver/ESDF/joint 表只建一次）")
    ap.add_argument("--seam-ids", nargs="+", type=int, default=None,
                    help="solve-kejian2：只处理这些焊缝 idx（多条，如 --seam-ids 0 1 3）；隐含全焊缝模式")
    ap.add_argument("--out", default=None,
                    help="solve-kejian2：输出【目录】（单焊缝/全焊缝都一样）；每条焊缝存 <out>/seam_<idx>.npy。"
                         "存 hand/joint_angles/workpiece_pose7，每只手最多 15 个"
                         "（超出按工件 pose 差距挑选：旋转差距优先、平移次之）")
    ap.add_argument("--lay-flat", action="store_true",
                    help="只跑 lay_flat 摆平工件并 open3d 可视化（用 --obj 或默认 BEAM）")
    ap.add_argument("--log", default=None,
                    help="solve-kejian2：把逐焊缝求解情况写到此 txt（成功率/失败 seam/各 seam 合格与已存条数/合计）；"
                         "每条焊缝解完就重写一次，中断也保住已完成的记录")
    ap.add_argument("--filter-short", action="store_true",
                    help=f"solve-kejian2：过滤掉长度 <{SHORT_SEAM_LEN_M * 100:.0f}cm 的短焊缝（不求解、不落盘）；"
                         "被过滤的 seam 会记入 --log 的 txt")
    args = ap.parse_args()

    if args.solve_kejian2:
        if not args.weld_json:
            ap.error("--solve-kejian2 需要 --weld-json（焊缝 _weld_angle3.json）")
        obj = args.obj or DEFAULT_LAY_FLAT_OBJ
        # —— 全焊缝模式：--all-seams 或给了 --seam-ids（工件级 ②③ 只建一次，边算边存） ——
        if args.all_seams or args.seam_ids is not None:
            # --out 是输出【目录】；每条焊缝解完立刻存 <out>/seam_<idx>.npy。（--out-dir 是旧 --solve 的参数，这里不借用）
            if not args.out:
                ap.error("--all-seams/--seam-ids 需要 --out <输出目录>（每条焊缝存 seam_<idx>.npy）")
            results = plan_init_pose_kejian2_all(obj, args.weld_json, seam_ids=args.seam_ids,
                                                 viz=args.viz, save_dir=args.out, save_k=15,
                                                 log_path=args.log, filter_short=args.filter_short)
            print(f"[kejian2] 全焊缝完成（{len(results)} 条）；输出目录 {args.out}")
            for sid, res in results.items():
                print(f"[kejian2]   seam {sid}: 正手 {len(res['forehand'])} / 反手 {len(res['backhand'])}")
            return
        # —— 单焊缝 ——
        _weld = next((w for w in load_welds(args.weld_json)
                      if int(w["idx"]) == int(args.seam_id)), None)
        if args.filter_short and _weld is not None and _seam_len_m(_weld) < SHORT_SEAM_LEN_M:
            _L = _seam_len_m(_weld)
            print(f"[kejian2] 焊缝 {args.seam_id} 长度 {_L * 100:.2f}cm < {SHORT_SEAM_LEN_M * 100:.0f}cm，"
                  "已过滤，跳过求解")
            if args.log:
                _write_kejian2_log(args.log, obj, args.weld_json, args.out, 15, [], 0,
                                   filtered=[{"idx": int(args.seam_id), "length_cm": _L * 100.0}])
                print(f"[kejian2] 求解日志已写入 {args.log}")
            return
        res = plan_init_pose_kejian2(obj, args.weld_json, seam_id=args.seam_id, viz=args.viz)
        print(f"[kejian2] 返回：正手 {len(res['forehand'])} 个 / 反手 {len(res['backhand'])} 个")
        if args.out:                                  # --out 当目录，存 <out>/seam_<seam_id>.npy
            os.makedirs(args.out, exist_ok=True)      # --out 目录不存在则创建
            _save_kejian2_npy(res, os.path.join(args.out, f"seam_{args.seam_id}.npy"), k=15, weld=_weld)
        if args.log:                                  # 单焊缝也可写日志（一行记录）
            n_f, n_b = len(res["forehand"]), len(res["backhand"])
            _write_kejian2_log(args.log, obj, args.weld_json, args.out, 15,
                               [{"idx": args.seam_id, "n_fore": n_f, "n_back": n_b,
                                 "saved_fore": min(n_f, 15), "saved_back": min(n_b, 15)}], 1)
            print(f"[kejian2] 求解日志已写入 {args.log}")
        return

    if args.lay_flat:
        lay_flat(args.obj or DEFAULT_LAY_FLAT_OBJ, viz=True)
        return

    # 计算初始化位姿（旧 lookup 法，保留）
    # if not args.obj or not args.weld_json:
    #     ap.error("--solve 需同时给 --obj 和 --weld-json")
    # _run_solve(args)

    # 可视化init空间
    # from gt_gen.config import load_config
    # cfg = load_config()
    # show_init_space(cfg, q=args.q, solid_spheres=args.solid)

    # 缺省：摆平工件可视化
    lay_flat(args.obj or DEFAULT_LAY_FLAT_OBJ, viz=True)




def _dominant_seam_points(obj_fp, long_len, weld_json=None, seam_ratio=0.9):
    """读 obj 同目录同前缀的 <stem>_weld_angle3.json，返回「最长那批焊缝」的所有端点 (M,3)。

    仅当【最长焊缝 > 工件最长轴长度的一半】时才启用焊缝逻辑（否则返回 None，交回质心兜底）；
    启用时取长度 ≥ seam_ratio·最长 的那批「差不多长」焊缝的全部端点。
    坐标为工件 mesh 系（corrected_p0/p1，与 obj 顶点同系）。
    返回 (端点(M,3) 或 None, 用到的 json 路径 或 None, 最长焊缝长度)。
    """
    import json
    fp = weld_json
    if fp is None:                                              # 自动定位：同目录同前缀
        d = os.path.dirname(obj_fp); base = os.path.basename(obj_fp)
        for suf in ("_part_watertight.obj", ".obj"):
            if base.endswith(suf):
                cand = os.path.join(d, base[:-len(suf)] + "_weld_angle3.json")
                if os.path.isfile(cand):
                    fp = cand
                    break
    if not fp or not os.path.isfile(fp):
        return None, None, 0.0
    try:
        with open(fp, "r") as f:
            data = json.load(f)
    except Exception:
        return None, fp, 0.0
    segs = []
    for w in data:
        try:
            p0 = np.asarray(w["corrected_p0"], dtype=np.float64)
            p1 = np.asarray(w["corrected_p1"], dtype=np.float64)
        except Exception:
            continue
        if p0.shape == (3,) and p1.shape == (3,):
            segs.append((p0, p1))
    if not segs:
        return None, fp, 0.0
    lens = np.array([np.linalg.norm(p1 - p0) for p0, p1 in segs])
    max_seam = float(lens.max())
    if max_seam <= 0.5 * float(long_len):                       # 最长焊缝没过最长轴一半 → 不用焊缝逻辑
        return None, fp, max_seam
    keep = lens >= seam_ratio * max_seam                        # 「差不多长」的那批
    pts = np.array([p for (p0, p1), k in zip(segs, keep) if k for p in (p0, p1)])
    return pts, fp, max_seam


def lay_flat(obj_fp, viz, weld_json=None, seam_ratio=0.9) -> np.ndarray:
    """把工件「摆平」放到 z=0 地面上，返回 4×4 变换 T（p_world = T · p_obj）。

    硬约束：相对 origin 的旋转 **必须是「绕 x/y/z 轴 90° 整数倍」的组合**——
    即 R 只能取立方体旋转群的有符号置换矩阵（det=+1）之一，
    绝不允许出现 45° 之类的任意滚转角（旧方案B 用凸包支撑边外法向定滚转，会破坏这一点）。

    做法：
      ① 在 mesh 自身坐标系里量各轴包围盒长度，最长者为「长轴」；
      ② 把长轴对到世界 +X（长轴水平），只剩「绕 +X 滚转 90° 整倍」4 种朝向；
      ③ 选滚转：若同目录 _weld_angle3.json 里存在「长度 > 最长轴一半」的长焊缝，
         则取最长那批焊缝、让它们尽量共处一个水平面（端点世界 z 跨度最小）；
         否则（无此长焊缝 / 无 json / 4 者难分）退回「落地质心最低」；
      ④ 平移使 min z = 0 贴地、xy 居中。
    长轴稳定落在世界 +X（下游 _kejian_orientations 依赖此约定）；长轴/滚转都恰好 90° 整倍，
    故 L/工字/槽型梁也停在真实平面上。

    viz=True 时用 open3d 显示摆平后的工件 + z=0 地面 + base 坐标架。
    """
    import trimesh as _trimesh

    tm = _trimesh.load(obj_fp, force="mesh", process=False)
    V = np.asarray(tm.vertices, dtype=np.float64)                # (N,3)
    # 质心：watertight 用体质心，否则退化为顶点形心
    try:
        com = np.asarray(tm.center_mass, dtype=np.float64)
        if not np.all(np.isfinite(com)):
            raise ValueError
    except Exception:
        com = V.mean(axis=0)

    # ① mesh 系各轴包围盒长度 → 最长轴（origin 已轴对齐，故长轴必是某条 mesh 轴）
    ext_mesh = V.max(axis=0) - V.min(axis=0)
    i_long = int(np.argmax(ext_mesh))

    # ② 长轴（mesh 第 i_long 轴）→ 世界 +X，其余两轴放 Y/Z，构一个 det=+1 的基准旋转 R0
    a, b = [k for k in range(3) if k != i_long]
    R0 = np.zeros((3, 3))
    R0[:, i_long] = (1.0, 0.0, 0.0)
    R0[:, a] = (0.0, 1.0, 0.0)
    R0[:, b] = (0.0, 0.0, 1.0)
    if np.linalg.det(R0) < 0:
        R0[:, b] = (0.0, 0.0, -1.0)                             # 翻一轴符号 → det=+1（右手系）

    # ③ 长轴恒沿 +X，仅剩「绕 +X 滚转 90° 整倍」4 种朝向。
    #    有「超过最长轴一半」的长焊缝 → 让最长那批焊缝尽量共处一个水平面（端点 z 跨度最小）；
    #    否则（含无焊缝、4 者 z 跨度难分）退回「落地质心最低」。
    dom_pts, used_json, max_seam = _dominant_seam_points(
        obj_fp, ext_mesh[i_long], weld_json, seam_ratio)
    z_tol = max(1e-4, 0.01 * float(np.linalg.norm(ext_mesh)))   # z 跨度差 < 1% 对角线视为不可区分

    Rx90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    best = None  # (stable, -zkey(焊缝 z 跨度桶，越小越好), -com_h, R, zspread)
    Rk = R0
    for _ in range(4):
        Vw = (Rk @ V.T).T
        cw = Rk @ com
        mn = Vw.min(axis=0); mx = Vw.max(axis=0)
        com_h = float(cw[2] - mn[2])                            # 落地后质心高度（越低越平/越稳）
        stable = (mn[0] - 1e-9 <= cw[0] <= mx[0] + 1e-9 and
                  mn[1] - 1e-9 <= cw[1] <= mx[1] + 1e-9)        # 质心 xy 落在底面投影内
        if dom_pts is not None:
            zc = (Rk @ dom_pts.T).T[:, 2]
            zspread = float(zc.max() - zc.min())                # 最长那批焊缝端点的世界 z 跨度
        else:
            zspread = 0.0
        zkey = int(round(zspread / z_tol))                      # 量化：跨度相近者同桶 → 交给质心兜底
        cand = (1 if stable else 0, -zkey, -com_h, Rk, zspread)
        if best is None or cand[:3] > best[:3]:
            best = cand
        Rk = Rx90 @ Rk                                          # 绕 +X 再滚 90°
    R = best[3]

    # ④ 平移：min z = 0 贴地，xy 居中（用 bbox 中心）
    Vw = (R @ V.T).T
    mn = Vw.min(axis=0); mx = Vw.max(axis=0)
    t = np.array([-(mn[0] + mx[0]) / 2.0, -(mn[1] + mx[1]) / 2.0, -mn[2]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t

    mode = ("焊缝面水平" if dom_pts is not None
            else ("焊缝≤半长→质心" if used_json else "无焊缝json→质心"))
    print(f"[lay_flat] mesh 三轴长(米): {ext_mesh.round(4)}  最长轴#{i_long}  依据={mode}")
    print(f"[lay_flat] 选中 90°-整倍旋转 R(行)= {R[0].astype(int)} {R[1].astype(int)} {R[2].astype(int)}")
    if dom_pts is not None:
        print(f"[lay_flat] 最长焊缝={max_seam:.4f}(>半长{0.5*float(ext_mesh[i_long]):.4f}) "
              f"参与端点={len(dom_pts)} 选中焊缝 z 跨度={best[4]:.4f}")
    print(f"[lay_flat] 落地后质心高={-best[2]:.4f} 稳定={bool(best[0])}")
    print(f"[lay_flat] 摆平后包围盒(米): 长×宽×高 = "
          f"{(mx-mn)[0]:.4f} × {(mx-mn)[1]:.4f} × {(mx-mn)[2]:.4f}")

    if viz:
        _show_lay_flat(obj_fp, T)
    return T


def _show_lay_flat(obj_fp: str, T: np.ndarray):
    """open3d 可视化：摆平后的工件 + z=0 地面 + base 三轴坐标架。"""
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(obj_fp)
    if not mesh.has_triangles():
        print(f"[lay_flat] open3d 读不到工件网格: {obj_fp}")
        return
    mesh.transform(T)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.70, 0.75, 0.80])

    # 地面：z=0 大薄盒（顶面贴 z=0）
    ext_xy = mesh.get_axis_aligned_bounding_box().get_extent()[:2]
    side = max(float(np.linalg.norm(ext_xy)) + 1.0, 2.0)
    th = 0.01
    ground = o3d.geometry.TriangleMesh.create_box(width=side, height=side, depth=th)
    ground.translate([-side / 2.0, -side / 2.0, -th])           # 顶面落在 z=0
    ground.compute_vertex_normals()
    ground.paint_uniform_color([0.85, 0.85, 0.85])

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    o3d.visualization.draw_geometries(
        [ground, mesh, frame], window_name="lay_flat：摆平工件 + 地面")


# ============================================================================
# ③ 新逻辑：lookup 关节角采样 + 允许朝向 snap + 正反手分类（--solve-kejian2）
# ----------------------------------------------------------------------------
# 复用本文件已自带的 InitPoseLookupSolver（n^6 关节角采样，与 plan_init_pose.py 一致）当候选源：
#   · solver 对本焊缝反解出候选工件位姿 (R,t,q)，并用「整臂/retract 碰撞球 vs 工件 ESDF」过滤碰撞解；
#     焊缝起/终由 solver 端点检查、尖端由 precompute 保证在 ee_xy_range_m(径向环)+ee_z_range_m(z) 内；
#   · 工件朝向由 lay_flat 锁死成 4 种允许朝向（沿最长轴 0/180° × 沿垂直地面轴 0/180°，长轴 ⊥ base-x）；
#     从候选里挑「R 与某允许朝向 xyz 三方向逐轴误差 < snap_deg」者，snap 到该朝向、重算 t 让焊枪尖端
#     仍落在焊缝 standoff 点（snap 后不复检）；
#   · 合格者按 bisector 在 base-x 的分量分「正手(与 base-x 反向=负x) / 反手(否则)」两类返回。
# 过滤范围/缓存全部走 default.yaml 的 plan_init_pose 段（solver 直接读 cfg.plan_init_*，不再派生 cfg2）。
# ============================================================================
def _rotmat_axis_angle(axis, angle: float) -> np.ndarray:
    """Rodrigues：绕单位轴 axis 转 angle(rad) 的 3×3 旋转矩阵（numpy 标量版）。"""
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    x, y, z = a
    c = float(np.cos(angle)); s = float(np.sin(angle)); C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)


def _Rx(a): return _rotmat_axis_angle([1.0, 0.0, 0.0], a)
def _Ry(a): return _rotmat_axis_angle([0.0, 1.0, 0.0], a)
def _Rz(a): return _rotmat_axis_angle([0.0, 0.0, 1.0], a)


def _align_rotmat(a, b) -> np.ndarray:
    """最小角旋转 R 满足 R·a = b（a,b 为 3 向量，内部单位化）。处理平行/反平行退化。"""
    a = np.asarray(a, dtype=np.float64); a = a / (np.linalg.norm(a) + 1e-12)
    b = np.asarray(b, dtype=np.float64); b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b); s = float(np.linalg.norm(v)); c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0.0:
            return np.eye(3)
        # 反平行：绕任一 ⟂a 轴转 180°
        perp = np.cross(a, [0.0, 0.0, 1.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [1.0, 0.0, 0.0])
        return _rotmat_axis_angle(perp, np.pi)
    return _rotmat_axis_angle(v / s, float(np.arctan2(s, c)))


def _kejian_orientations(obj_fp: str) -> List[np.ndarray]:
    """工件 4 种允许朝向（3×3，p_base = R·p_obj 的旋转部分）。

    复用 lay_flat 的平躺旋转 R_lay（长轴沿世界 +X），绕 base-Z 转 +90° 使最长轴沿 base +Y
    （⊥ base-x）得基准朝向 R_A；4 种 = R_A ×{I, 沿长轴(Y)翻180°} ×{I, 沿垂直轴(Z)翻180°}。"""
    T_lay = lay_flat(obj_fp, viz=False)
    R_lay = np.asarray(T_lay, dtype=np.float64)[:3, :3]
    R_A = _Rz(np.pi / 2.0) @ R_lay            # 长轴 +X → +Y（⊥ base-x）
    Ry180 = _Ry(np.pi)                        # 沿最长轴(base Y)翻面
    Rz180 = _Rz(np.pi)                        # 沿垂直地面轴(base Z)翻
    return [R_A, Ry180 @ R_A, Rz180 @ R_A, Ry180 @ Rz180 @ R_A]


def _kejian2_snap(R_cand, R_valid_list, snap_deg):
    """候选工件旋转 R_cand 与 4 种允许朝向逐一比对：残差 R_rel=R_valid^T·R_cand 拆成 xyz 内旋欧拉角，
    三轴误差均 < snap_deg(度) 则命中。返回首个命中的 (oid, R_valid)；全不中返回 (None, None)。"""
    from scipy.spatial.transform import Rotation as sR
    Rc = np.asarray(R_cand, dtype=np.float64)
    for oid, Rv in enumerate(R_valid_list):
        R_rel = np.asarray(Rv, dtype=np.float64).T @ Rc
        e = sR.from_matrix(R_rel).as_euler("xyz", degrees=True)
        if np.all(np.abs(e) < float(snap_deg)):
            return oid, np.asarray(Rv, dtype=np.float64)
    return None, None


# —— 固定底座 vs 工件 在 base-xy 平面投影的相交过滤（机械臂不能压在工件下 / 工件不能盖在底座上）——
_FIXED_BASE_LINK = "xiaoyu_base_link"   # 「固定底座」：不随关节运动，碰撞球 q 无关，可只算一次


def _fixed_base_xy_circles(cfg) -> List[Tuple[np.ndarray, float]]:
    """固定底座 (_FIXED_BASE_LINK) 的碰撞球在 base-xy 平面的投影圆列表 [(中心(2,), 半径), …]。
    底座不随关节动，球位姿与 q 无关，用 retract_config 算一次即可。"""
    from gt_gen.obstacle_placement import compute_link_sweep
    per_wp, _ = compute_link_sweep(cfg, [cfg.retract_config], [_FIXED_BASE_LINK])
    sph = per_wp.get(_FIXED_BASE_LINK)
    circles: List[Tuple[np.ndarray, float]] = []
    if sph is None:
        return circles
    for c in np.asarray(sph, float)[0]:                  # (S,4) 取唯一路点
        cx, cy, _cz, r = (float(v) for v in c)
        if r <= 1e-4:
            continue
        circles.append((np.array([cx, cy], dtype=np.float64), r))
    return circles


def _load_mesh_vf(obj_fp: str):
    """读工件 mesh 的顶点 V(N,3) 与三角形索引 F(M,3) int（mesh 局部系，与 InitPoseLookupSolver
    同一份 obj）。读不到顶点返回 (None, None)；有顶点但无三角形返回 (V, None)。"""
    try:
        import open3d as o3d
        m = o3d.io.read_triangle_mesh(obj_fp)
        v = np.asarray(m.vertices, dtype=np.float64)
        f = np.asarray(m.triangles, dtype=np.int64)
        if v.size == 0:
            return None, None
        return v, (f if f.size else None)
    except Exception:
        return None, None


def _voxelize_mesh_points(mesh_v, mesh_f, pitch: float):
    """把工件表面按 pitch(米) 体素化，返回占据体素中心 (K,3)（mesh 局部系）。
    用于「工件距 base 欧氏最近点 x」过滤的稠密点集——密度绑物理长度、与三角形大小解耦，
    大三角形也会被填成一排体素，不会像顶点那样漏采。
    体素化失败/无三角形则回退用原顶点 mesh_v（保证不崩），由调用方负责打印。"""
    if mesh_v is None or mesh_f is None:
        return mesh_v
    try:
        import gt_gen.compat
        import trimesh as _trimesh
        gt_gen.compat.apply_trimesh_shim()
        tm = _trimesh.Trimesh(vertices=np.asarray(mesh_v, dtype=np.float64),
                              faces=np.asarray(mesh_f, dtype=np.int64), process=False)
        pts = np.asarray(tm.voxelized(pitch=float(pitch)).points, dtype=np.float64)
        if pts.size == 0:
            return mesh_v
        return pts
    except Exception:
        return mesh_v


def _base_circles_to_arrays(base_circles):
    """把 [(中心(2,),半径),…] 转成 (cc(C,2), rr(C,), 并集AABB下界 umin(2,), 并集AABB上界 umax(2,))。
    底座圆 q 无关，转一次即可复用；空则返回 (None,None,None,None)。"""
    if not base_circles:
        return None, None, None, None
    cc = np.array([c for c, _ in base_circles], dtype=np.float64)   # (C,2)
    rr = np.array([r for _, r in base_circles], dtype=np.float64)   # (C,)
    umin = (cc - rr[:, None]).min(axis=0)                           # (2,) 所有圆并集 AABB 下界
    umax = (cc + rr[:, None]).max(axis=0)                           # (2,) 上界
    return cc, rr, umin, umax


def _circle_tri_intersect_any(cc, rr, tris) -> bool:
    """任一底座圆 (cc(C,2), rr(C,)) 与任一 2D 三角形 (tris(M,3,2)) 相交则 True（全向量化）。
    相交 = 三条件之一：①三角形某顶点落在圆内；②圆心落在三角形内；③圆心到某条边的距离 ≤ r。
    （①②③合起来覆盖所有「圆∩三角形≠∅」情形：含工件套住底座、底座套住小三角、部分交叠）。"""
    M, C = tris.shape[0], cc.shape[0]
    if M == 0 or C == 0:
        return False
    cen = cc[:, None, :]                          # (C,1,2)
    r2 = (rr * rr)[:, None]                        # (C,1)
    A, B, D = tris[:, 0, :], tris[:, 1, :], tris[:, 2, :]   # 各 (M,2)
    # ① 三角形顶点在圆内
    for vtx in (A, B, D):
        if np.any(((cen - vtx[None, :, :]) ** 2).sum(-1) <= r2):    # (C,M)
            return True
    # ②③ 遍历三条边：叉积符号（判圆心是否在三角形内）+ 圆心到线段距离
    cross = []
    for P, Q in ((A, B), (B, D), (D, A)):
        e = (Q - P)[None, :, :]                    # (1,M,2) 边向量
        w = cen - P[None, :, :]                    # (C,M,2) 圆心相对边起点
        cross.append(e[..., 0] * w[..., 1] - e[..., 1] * w[..., 0])  # (C,M) 叉积
        ee = (e ** 2).sum(-1)                       # (1,M)
        t = np.clip((w * e).sum(-1) / (ee + 1e-18), 0.0, 1.0)        # (C,M) 投影参数夹到 [0,1]
        proj = P[None, :, :] + t[..., None] * e     # (C,M,2) 边上最近点
        if np.any(((cen - proj) ** 2).sum(-1) <= r2):                # ③ 圆心到边距离 ≤ r
            return True
    c0, c1, c2 = cross                              # ② 三叉积同号 ⇒ 圆心在三角形内（兼容两种绕向）
    inside = ((c0 >= 0) & (c1 >= 0) & (c2 >= 0)) | ((c0 <= 0) & (c1 <= 0) & (c2 <= 0))
    return bool(np.any(inside))


def _base_overlaps_workpiece_tris(cc, rr, umin, umax, v_base_xy, faces) -> bool:
    """固定底座圆 vs 工件【三角形投影并集】是否相交（精确，替代凸包近似）：
    先用底座并集 AABB 粗筛三角形（底座固定在 base 原点附近小区域，绝大多数三角形被剔除→快），
    再对邻近三角形做精确圆-三角形相交。faces 为 None（无三角形）时退化为「不过滤」。"""
    if cc is None or faces is None or v_base_xy.shape[0] == 0:
        return False
    tris = v_base_xy[faces]                         # (M,3,2) 各三角形 3 个 base-xy 顶点
    tmin = tris.min(axis=1)                         # (M,2) 三角形 AABB 下界
    tmax = tris.max(axis=1)                         # (M,2) 上界
    near = (tmax >= umin).all(axis=1) & (tmin <= umax).all(axis=1)   # 与底座并集 AABB 相叠才精算
    if not np.any(near):
        return False
    return _circle_tri_intersect_any(cc, rr, tris[near])


def _link_pose_in_base_batch(cfg, q_rows, link_name: str = "Link6"):
    """批量 FK：给定关节角 q_rows (N,dof)，返回 link_name 在 base_link 系的 (原点位置 (N,3), +z 轴 (N,3))。

    复用机器人 yml（cfg.robot_cfg_path）另建一个【仅做 FK】的 CudaRobotModel——把 link_name 注入
    kinematics.link_names（yml 默认 null，只跟踪 ee_link），get_link_poses 才能取到该 link 位姿；
    不触碰 solver 的碰撞模型/查表。z 轴 = 该 link 在 base 的旋转矩阵第 3 列（quat wxyz → rotmat 第 2 列）。"""
    import gt_gen.compat  # noqa: F401  warp shim，须在 import curobo 前
    import torch
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.util_file import load_yaml
    ta = TensorDeviceType()
    rd = load_yaml(cfg.robot_cfg_path)
    kin = rd["robot_cfg"]["kinematics"]
    ln = list(kin.get("link_names") or [])
    if link_name not in ln:
        ln.append(link_name)
    kin["link_names"] = ln
    robot_cfg = RobotConfig.from_dict(rd["robot_cfg"], ta)
    kin_model = CudaRobotModel(robot_cfg.kinematics)   # RobotConfig.kinematics 是 Config，需包成 model 才能 FK
    q = torch.as_tensor(np.asarray(q_rows, dtype=np.float64), device=ta.device, dtype=ta.dtype)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    with torch.no_grad():
        pose = kin_model.get_link_poses(q, [link_name])
    pos = pose.position.reshape(-1, 3).detach().cpu().numpy().astype(np.float64)
    R = quat_wxyz_to_rotmat_batch(pose.quaternion.reshape(-1, 4))   # (N,3,3) 列=各轴在 base
    z = R[:, :, 2].detach().cpu().numpy().astype(np.float64)        # +z 轴在 base
    return pos, z


def _show_kejian2_results(cfg, obj_fp, weld, res, stride: int = 5, extra_geoms=None):
    """挨个可视化 kejian2 结果（正手→反手），每 stride 个抽 1 个（取每条手内第 1,1+stride,… 个）。

    每个位姿单开一个窗口（窗口标题写清「正手/反手 第 i/N 个」）；按【C 键】切到下一个、相机视角自动沿用
    （open3d 0.19 的 legacy Visualizer 无法对存活窗口改标题，故只能每个换新窗口才能让标题随之变化）；
    直接关窗（不按 C）则退出。几何复用 plan_init_pose.py 风格的 _lookup_solution_geoms（整臂碰撞球
    @ joint_angles + 工件 mesh @ T_workpiece_in_base + 绿色焊缝线 + standoff 落枪点）；sol 由 result 适配。

    extra_geoms：可选回调 (R, t) -> list[o3d.geometry]，每个候选窗口按该候选的 T_workpiece_in_base
    (R,t) 追加额外几何（如随工件一起摆放的障碍物）；缺省 None 时行为与原先完全一致。"""
    import open3d as o3d
    step = max(1, int(stride))
    items = []   # (hand_label, i_1based, N, sol_like, r)
    for hand_label, lst in (("正手", res.get("forehand", [])), ("反手", res.get("backhand", []))):
        for i in range(0, len(lst), step):
            r = lst[i]
            T = np.asarray(r["T_workpiece_in_base"], dtype=np.float64)
            sol = {"R": T[:3, :3], "t": T[:3, 3],
                   "q": np.asarray(r["joint_angles"], dtype=np.float64)}
            items.append((hand_label, i + 1, len(lst), sol, r))
    if not items:
        print("[kejian2] 无可视化结果")
        return
    n = len(items)
    print(f"[kejian2] 可视化 {n} 个（每 {step} 个抽 1；按 C 切下一个，直接关窗退出）")

    def _bisector_axis_geoms(origin, direction, length: float = 0.30):
        """正反手判据轴：从焊缝中心点沿 bisector（背离工件方向）画一根【带箭头的线段】（蓝色箭头）。
        蓝箭头的 base-x 分量 <0（与 base-x 反向）⇒ 正手，否则 ⇒ 反手（与分类判据一致）。"""
        import open3d as o3d
        d = np.asarray(direction, float); d = d / (np.linalg.norm(d) + 1e-12)
        p0 = np.asarray(origin, float)
        col = [0.15, 0.35, 0.95]
        cone_h = 0.30 * length
        cyl_h = length - cone_h
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.010, cone_radius=0.022,
            cylinder_height=cyl_h, cone_height=cone_h)
        arrow.rotate(_align_rotmat([0.0, 0.0, 1.0], d), center=(0.0, 0.0, 0.0))  # 默认 +Z → bisector
        arrow.translate(p0.tolist())                                            # 箭尾在焊缝中心点
        arrow.compute_vertex_normals()
        arrow.paint_uniform_color(col)
        return [arrow]

    def _near_pt_geoms(p_near, radius: float = 0.03):
        """把「工件距 base 原点欧氏最近的那个点」画成品红小球 + 从 base 原点(0,0,0)到它的连线，
        直观展示 workpiece_x_min 过滤依据的那个点落在工件哪里、离底座多近。"""
        import open3d as o3d
        p = np.asarray(p_near, float)
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
        s.translate(p.tolist())
        s.compute_vertex_normals()
        s.paint_uniform_color([1.0, 0.1, 0.6])                # 品红
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0], p])),
            lines=o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32)))
        ls.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.1, 0.6]]))
        return [s, ls]

    cam = {"params": None}   # 跨窗口沿用相机视角，避免每次切换都重置
    idx = 0
    while idx < n:
        hand_label, i, N, sol, r = items[idx]
        title = f"kejian2: {hand_label} 第 {i}/{N} 个（抽样 {idx + 1}/{n}）— 按 C 下一个 / 关窗退出"
        bis = np.asarray(r["bisector_base"], dtype=np.float64)
        seam_c = np.asarray(r["seam_center_base"], dtype=np.float64)
        print(f"[viz] {hand_label} 第 {i}/{N} 个（抽样 {idx + 1}/{n}）"
              f" bisector base-x={float(bis[0]):+.3f}")
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=title)
        for g in _lookup_solution_geoms(cfg, obj_fp, weld, sol):
            vis.add_geometry(g)
        for g in _bisector_axis_geoms(seam_c, bis):   # 蓝色 bisector 轴（正反手判据）
            vis.add_geometry(g)
        p_near = r.get("wpx_near_base", None)          # 品红球：工件距 base 原点最近点（workpiece_x_min 判据点）
        if p_near is not None:
            print(f"       最近点 base-x={float(np.asarray(p_near, float)[0]):+.3f}"
                  f"（阈值 workpiece_x_min）")
            for g in _near_pt_geoms(p_near):
                vis.add_geometry(g)
        if extra_geoms is not None:                    # 随工件摆放的额外几何（如障碍物）
            try:
                for g in extra_geoms(sol["R"], sol["t"]):
                    vis.add_geometry(g)
            except Exception as _e:
                print(f"[viz] extra_geoms 追加失败（忽略）: {_e}")
        if cam["params"] is not None:
            try:
                vis.get_view_control().convert_from_pinhole_camera_parameters(
                    cam["params"], allow_arbitrary=True)
            except Exception:
                pass            # 窗口尺寸/版本不兼容时退回默认视角，不致命
        advance = {"go": False}

        def _next(v):
            advance["go"] = True
            v.close()           # 关掉当前窗口 → 退出 run() → 外层 while 开下一个（标题随之变）
            return False

        vis.register_key_callback(ord("C"), _next)
        vis.run()
        try:
            cam["params"] = vis.get_view_control().convert_to_pinhole_camera_parameters()
        except Exception:
            pass
        vis.destroy_window()
        if not advance["go"]:
            print("[viz] 直接关窗，结束可视化")
            break               # 用户没按 C 而是关窗 → 退出
        idx += 1


def _select_diverse_poses(items: list, k: int = 15) -> list:
    """从同一只手的合格结果里挑最多 k 个「工件 pose 差距尽量大」的（farthest-point 贪心）。

    距离优先级：先旋转差距（两工件姿态 R 的测地夹角，度），再平移差距（t 的欧氏距离，米）。
    用 d = rot_deg*1000 + trans_m 把旋转设为主序、平移设为次序（旋转相同才比平移）。
    ≤k 个时原样返回；否则 FPS：从第 0 个起，每次选「到已选集合最小距离最大」的那个。"""
    if len(items) <= k:
        return items
    Rs = [np.asarray(r["T_workpiece_in_base"], dtype=np.float64)[:3, :3] for r in items]
    ts = [np.asarray(r["T_workpiece_in_base"], dtype=np.float64)[:3, 3] for r in items]
    n = len(items)

    def _dist(i, j):
        c = (np.trace(Rs[i].T @ Rs[j]) - 1.0) * 0.5
        ang = float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))
        tr = float(np.linalg.norm(ts[i] - ts[j]))
        return ang * 1000.0 + tr

    selected = [0]
    mind = [_dist(0, j) for j in range(n)]
    while len(selected) < k:
        nxt = int(np.argmax(mind))
        if nxt in selected:          # 退化：剩余全是重复 pose，提前停
            break
        selected.append(nxt)
        for j in range(n):
            dj = _dist(nxt, j)
            if dj < mind[j]:
                mind[j] = dj
    return [items[i] for i in selected]


def _save_kejian2_npy(res: Dict[str, list], path: str, k: int = 15,
                      weld: Optional[dict] = None) -> None:
    """把正/反手结果存成单个 .npy（dict 对象数组；读取用 np.load(path, allow_pickle=True).item()）。

    保存的 dict：
      · 求解结果（每条 pose 各自不同，按 picked 顺序对齐的并列数组）：
        hand (N,) 字符串、joint_angles (N, ndof)、workpiece_pose7 (N, 7)；
      · seam_idx：该焊缝在 weld_json 里的下标（=文件名 seam_<idx>）；
      · weld：_weld_angle3.json 里【这条焊缝的原始 dict，原样照搬】——mesh 坐标系不转、bisector
        不归一化、内容一字不改；json 里有什么字段就存什么，缺的字段就【不存】（不补兜底）。
    每只手超过 k 个时用 _select_diverse_poses 挑 k 个（旋转差距优先、平移差距次之）。"""
    fore = _select_diverse_poses(list(res.get("forehand", [])), k)
    back = _select_diverse_poses(list(res.get("backhand", [])), k)
    picked = fore + back
    if not picked:
        print("[kejian2] 无合格结果，跳过保存")
        return
    weld_raw = dict(weld["raw"]) if (isinstance(weld, dict) and weld.get("raw")) else {}
    data = {
        "hand": np.array([r["hand"] for r in picked]),
        "joint_angles": np.stack([np.asarray(r["joint_angles"], dtype=np.float64).reshape(-1)
                                  for r in picked]),
        "workpiece_pose7": np.stack([np.asarray(r["workpiece_pose7"], dtype=np.float64).reshape(-1)
                                     for r in picked]),
        "seam_idx": int(weld["idx"]) if (isinstance(weld, dict) and "idx" in weld) else -1,
        "weld": weld_raw,                         # json 原始字段，缺啥就没啥（不补兜底）
    }
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    np.save(path, np.array(data, dtype=object))   # 对象数组：np.load(..., allow_pickle=True).item() 读回
    print(f"[kejian2] 已保存 {len(picked)} 条到 {path}"
          f"（正手 {len(fore)} / 反手 {len(back)}；每只手最多 {k}，超出按 pose 差距挑选；"
          f"含焊缝 json 原始字段 {sorted(weld_raw)}）")


def _write_kejian2_log(log_path: str, obj_fp: str, weld_json: str,
                       save_dir: Optional[str], save_k: int,
                       stats: List[dict], total_welds: int,
                       filtered: Optional[List[dict]] = None,
                       setup_s: float = 0.0, elapsed_s: float = 0.0) -> None:
    """把逐焊缝求解情况写成一个 txt（每条焊缝解完就重写一次，中断也保住已完成的记录）。

    stats 每项：{idx, n_fore, n_back, saved_fore, saved_back, t_solve}
        （合格=求解给出数，已存=落盘数=min(合格,k)；t_solve=该焊缝 ④+⑤ 耗时秒，缺省 0）。
    filtered 每项：{idx, length_cm}——被 --filter-short 过滤掉的短焊缝（未求解）。
    setup_s：工件级一次性 ②③（solver/ESDF/joint 表）耗时秒，全焊缝共享、只算一次。
    elapsed_s：逐焊缝求解循环到此刻的累计墙钟秒。
    内容：逐焊缝成功/失败 + 合格/已存条数 + 耗时；被过滤短焊缝清单；末尾汇总成功率、失败 seam、
    合格/已存总数，以及计时（setup / 求解累计 / 平均·最慢每条 / 总墙钟）。
    成功率按【实际求解的焊缝】(stats) 计，过滤掉的不计入分母。"""
    import datetime as _dt
    filtered = filtered or []
    d = os.path.dirname(os.path.abspath(log_path))
    if d:
        os.makedirs(d, exist_ok=True)
    n_ok = sum(1 for s in stats if (s["n_fore"] + s["n_back"]) > 0)
    n_fail = len(stats) - n_ok
    fail_ids = [s["idx"] for s in stats if (s["n_fore"] + s["n_back"]) == 0]
    sum_qual_f = sum(s["n_fore"] for s in stats)
    sum_qual_b = sum(s["n_back"] for s in stats)
    sum_save_f = sum(s["saved_fore"] for s in stats)
    sum_save_b = sum(s["saved_back"] for s in stats)
    solve_times = [float(s.get("t_solve", 0.0)) for s in stats]
    sum_solve = sum(solve_times)
    lines = []
    lines.append("# plan_init_pose_kejian2 初始位姿求解日志")
    lines.append(f"工件 obj      : {obj_fp}")
    lines.append(f"焊缝 json     : {weld_json}")
    lines.append(f"输出目录      : {save_dir}")
    lines.append(f"每只手上限 k  : {save_k}")
    lines.append(f"更新时间      : {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"进度          : 已处理 {len(stats)} / 计划 {total_welds} 条"
                 + (f"（另过滤短焊缝 {len(filtered)} 条）" if filtered else ""))
    lines.append("")
    lines.append("===== 逐焊缝 =====")
    for s in stats:
        qual = s["n_fore"] + s["n_back"]
        saved = s["saved_fore"] + s["saved_back"]
        ts = float(s.get("t_solve", 0.0))
        if qual > 0:
            lines.append(f"seam {s['idx']:>4}  成功   合格 {qual:>5} (正手 {s['n_fore']:>4} / 反手 {s['n_back']:>4})"
                         f"   已存 {saved:>3} (正手 {s['saved_fore']:>3} / 反手 {s['saved_back']:>3})"
                         f"   耗时 {ts:7.2f}s")
        else:
            lines.append(f"seam {s['idx']:>4}  失败   合格     0   已存   0"
                         f"   耗时 {ts:7.2f}s")
    if filtered:
        lines.append("")
        lines.append(f"===== 过滤掉的短焊缝 (<{SHORT_SEAM_LEN_M * 100:.0f}cm，未求解) =====")
        for fz in filtered:
            lines.append(f"seam {fz['idx']:>4}  已过滤   长度 {fz['length_cm']:.2f} cm")
    lines.append("")
    lines.append("===== 汇总 =====")
    lines.append(f"焊缝总数        : {len(stats)}" + (f" / 计划 {total_welds}" if len(stats) != total_welds else ""))
    rate = (100.0 * n_ok / len(stats)) if stats else 0.0
    lines.append(f"成功            : {n_ok} 条  ({rate:.1f}%)")
    lines.append(f"失败            : {n_fail} 条" + (f"   失败 seam: {fail_ids}" if fail_ids else ""))
    if filtered:
        lines.append(f"过滤短焊缝      : {len(filtered)} 条  (<{SHORT_SEAM_LEN_M * 100:.0f}cm，未求解)"
                     f"   seam: {[fz['idx'] for fz in filtered]}")
    lines.append(f"合格 pose 总数  : {sum_qual_f + sum_qual_b}  (正手 {sum_qual_f} / 反手 {sum_qual_b})")
    lines.append(f"已存 pose 总数  : {sum_save_f + sum_save_b}  (正手 {sum_save_f} / 反手 {sum_save_b})")
    lines.append("")
    lines.append("===== 计时 =====")
    lines.append(f"工件级 ②③ setup : {setup_s:8.2f}s  (solver/ESDF/joint 表，全焊缝共享、只一次)")
    lines.append(f"逐焊缝 ④⑤ 累计  : {sum_solve:8.2f}s  (已处理 {len(stats)} 条求解时间之和)")
    if stats:
        avg = sum_solve / len(stats)
        i_max = max(range(len(stats)), key=lambda k: solve_times[k])
        lines.append(f"  每条平均      : {avg:8.2f}s")
        lines.append(f"  最慢一条      : {solve_times[i_max]:8.2f}s  (seam {stats[i_max]['idx']})")
    lines.append(f"求解墙钟        : {elapsed_s:8.2f}s  (循环 wall-clock，含落盘/写日志开销)")
    lines.append(f"合计(setup+墙钟): {setup_s + elapsed_s:8.2f}s")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _kejian2_build_ctx(obj_fp: str) -> dict:
    """构建与【工件相关、与焊缝无关】的求解上下文（只需一次，可被同一工件的所有焊缝复用）：
    cfg、InitPoseLookupSolver（含工件 ESDF 体素化 ②）、joint 表（③，已存盘则复用）、
    4 种允许朝向 R_valid_list、snap_deg、绕轴采样、ee 范围/standoff、固定底座圆 + 工件顶点/三角形。
    返回 ctx dict；ctx["prof_setup"] 记录 ①②③②b 的耗时（供单焊缝入口打印完整耗时表）。"""
    from gt_gen.config import load_config
    import time as _time
    import torch as _torch

    def _sync():
        if _torch.cuda.is_available():
            _torch.cuda.synchronize()

    prof_setup = {}
    _t = _time.time()
    cfg = load_config()                          # solver 直接读 cfg.plan_init_*（plan_init_pose 段），不再派生 cfg2
    prof_setup["①setup(cfg)"] = _time.time() - _t

    # —— ② lookup 求解器（与工件相关：含工件 ESDF 体素化） ——
    _t = _time.time()
    solver = InitPoseLookupSolver(
        cfg, obj_fp,
        collision_tolerance=cfg.plan_init_collision_tolerance,
        voxel_size=cfg.plan_init_voxel_size,
        n_per_dof=cfg.plan_init_n_per_dof,
        clearance_inflate=cfg.plan_init_clearance_inflate)
    _sync()
    prof_setup["②solver初始化(含工件ESDF体素化)"] = _time.time() - _t

    # —— ③ joint 表（与工件无关，已存盘则复用） ——
    _t = _time.time()
    if solver.load_joint_table():
        print("[kejian2] 复用已存 joint table（跳过 n^6 预计算）")
    else:
        print("[kejian2] precomputing joint table（与工件无关，仅一次）…")
        solver.precompute_joint_table()
        solver.save_joint_table()
    _sync()
    prof_setup["③joint表(load或precompute+save)"] = _time.time() - _t

    # 朝向集合（4 种允许朝向）：solve 的朝向粗筛 + CPU 精筛 _kejian2_snap 共用，snap_deg 同源
    _t = _time.time()
    R_valid_list = _kejian_orientations(obj_fp)
    prof_setup["②b朝向集合(lay_flat)"] = _time.time() - _t

    # —— ③' 「工件距 base 欧氏最近点 x」+「固定底座 vs 工件 base-xy 投影相交」两过滤的预备 ——
    #   工件顶点/三角形【始终加载】：最近点 x 过滤必需；底座相交过滤也复用同一份顶点 xy。
    #   底座圆则由 plan_init_pose.base_overlap_filter 开关控制：关闭则置 None → solve 时自然跳过相交过滤。
    mesh_v, mesh_f = _load_mesh_vf(obj_fp)
    if mesh_v is None:
        print("[kejian2] 警告：读不到工件顶点，无法做「工件距 base 欧氏最近点 x」过滤（该候选不因此过滤）")
    # 「最近点 x」过滤的稠密点集：把工件按 workpiece_x_voxel_m 体素化（与三角形大小解耦），
    # 体素化失败/无三角形自动回退到顶点。存 mesh 局部系，per-candidate 只反变换 base 原点再 argmin。
    wpx_pts = _voxelize_mesh_points(mesh_v, mesh_f, cfg.plan_init_workpiece_x_voxel)
    if mesh_v is not None:
        _n_pts = 0 if wpx_pts is None else len(wpx_pts)
        _via = "体素" if (mesh_f is not None and wpx_pts is not None and wpx_pts is not mesh_v) else "顶点(回退)"
        print(f"[kejian2] 「最近点 x」过滤点集：{_n_pts} 个（{_via}, pitch={cfg.plan_init_workpiece_x_voxel}m）")
    if cfg.plan_init_base_overlap_filter:
        base_circles = _fixed_base_xy_circles(cfg)
        bc_cc, bc_rr, bc_umin, bc_umax = _base_circles_to_arrays(base_circles)
        if not base_circles:
            print("[kejian2] 警告：取不到固定底座碰撞球，跳过「底座-工件 XY 相交」过滤")
        if mesh_v is None or mesh_f is None:
            print("[kejian2] 警告：读不到工件顶点/三角形，跳过「底座-工件 XY 相交」过滤")
    else:
        bc_cc = bc_rr = bc_umin = bc_umax = None
        print("[kejian2] base_overlap_filter=false：已关闭「底座-工件 XY 相交」过滤")

    return {
        "obj_fp": obj_fp,
        "cfg": cfg, "solver": solver,
        "R_valid_list": R_valid_list,
        "snap_deg": cfg.plan_init_snap_deg,
        "rot_x": _deg_range(cfg.plan_init_rot_x_deg),
        "rot_y": _deg_range(cfg.plan_init_rot_y_deg),
        "rot_z": _deg_range(cfg.plan_init_rot_z_deg),
        "standoff": cfg.plan_init_standoff,
        "ee_xy_range": [float(v) for v in cfg.plan_init_ee_xy_range],
        "ee_z_range": [float(v) for v in cfg.plan_init_ee_z_range],
        "workpiece_x_min": float(cfg.plan_init_workpiece_x_min),
        "wpx_pts": wpx_pts,
        "arm_collision_recheck": bool(cfg.plan_init_arm_collision_recheck),
        "bc_cc": bc_cc, "bc_rr": bc_rr, "bc_umin": bc_umin, "bc_umax": bc_umax,
        "mesh_v": mesh_v, "mesh_f": mesh_f,
        "prof_setup": prof_setup,
    }


def _kejian2_solve_weld(ctx: dict, weld: Dict, verbose: bool = True) -> Tuple[Dict[str, list], dict]:
    """对单条焊缝求解（④ lookup 碰撞过滤 + ⑤ 朝向 snap/正面过滤/正反手分类），复用 ctx 里工件级的
    solver/朝向/底座/mesh（不重建 ②③）。返回 ({"forehand":[...],"backhand":[...]}, prof_weld)，
    prof_weld 记录 ④⑤ 耗时。per-weld 结果与原 plan_init_pose_kejian2 单焊缝逐位一致。
    verbose=False 时静默 lookup 明细 / solve profile 计时（求解不变；scene.py 默认走此路）；
    逐步过滤计数（① lookup→…→⑧ 复检→合格）为结果口径信息，【无条件打印】，不受 verbose 影响。"""
    import time as _time
    import torch as _torch
    from scipy.spatial.transform import Rotation as _sR

    def _sync():
        if _torch.cuda.is_available():
            _torch.cuda.synchronize()

    solver = ctx["solver"]
    R_valid_list = ctx["R_valid_list"]
    snap_deg = ctx["snap_deg"]
    rot_x, rot_y, rot_z = ctx["rot_x"], ctx["rot_y"], ctx["rot_z"]
    standoff = ctx["standoff"]
    xy_lo, xy_hi = ctx["ee_xy_range"]
    z_lo, z_hi = ctx["ee_z_range"]
    wp_x_min = ctx["workpiece_x_min"]
    wpx_pts = ctx["wpx_pts"]
    arm_recheck = ctx["arm_collision_recheck"]
    bc_cc, bc_rr, bc_umin, bc_umax = ctx["bc_cc"], ctx["bc_rr"], ctx["bc_umin"], ctx["bc_umax"]
    mesh_v, mesh_f = ctx["mesh_v"], ctx["mesh_f"]

    prof = {}
    # —— ④ lookup（碰撞过滤 + 朝向粗筛） ——
    _t = _time.time()
    solver.solve_one_weld_lookup(weld, rot_x, rot_y, rot_z, profile=verbose,
                                 orient_valid=R_valid_list, orient_snap_deg=snap_deg)
    _sync()
    prof["④solve_one_weld_lookup(碰撞过滤)"] = _time.time() - _t
    cands = list(getattr(solver, "last_all_solutions", []) or [])
    if verbose:
        print(f"[kejian2] lookup 候选 {len(cands)} 个（绕末端轴采样 "
              f"{len(rot_x)}×{len(rot_y)}×{len(rot_z)}）")
    if not cands:
        print("[kejian2] 逐步过滤：① lookup 候选 0 个 → 合格 0"
              "（请放宽 ee_xy_range_m/ee_z_range_m 或增大 n_per_dof）")
        prof["⑤snap+正反手分类(CPU遍历候选)"] = 0.0
        return {"forehand": [], "backhand": []}, prof

    _t = _time.time()
    mid_world = np.asarray(weld["mid_world"], dtype=np.float64)
    bis_world = np.asarray(weld["bisector_world"], dtype=np.float64)
    target_world = mid_world + standoff * bis_world      # 焊枪尖端目标点（mesh world 系）

    def _T(R, t):
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        return T

    results = []
    n_hit = 0          # 朝向 snap 命中数
    n_back = 0         # 因「焊缝在背面」(bisector base-z<0) 丢弃
    n_xneg = 0         # 因「焊缝中心点 base-x<=0」丢弃
    n_wpx = 0          # 因「工件距 base 欧氏最近点 base-x <= workpiece_x_min」丢弃
    n_overlap = 0      # 因「固定底座与工件在 base-xy 投影相交」丢弃
    n_arm_collide = 0  # 因「snap 后 retract 姿态整臂 vs 工件碰撞」丢弃（最后一步复检）
    n_dedup = 0        # 因「同朝向 + 同位置(2cm)重复」轻去重丢弃
    seen = set()

    # —— 向量化预筛（批量替代逐候选 scipy/几何；与逐个版逐位等价，仅 overlap+去重+组装仍按原始顺序循环）——
    Ncand = len(cands)
    R_all = np.stack([np.asarray(s["R"], dtype=np.float64) for s in cands])              # (N,3,3)
    mid_all = np.stack([np.asarray(s["mid_in_base"], dtype=np.float64) for s in cands])  # (N,3)
    ee_all = np.stack([np.asarray(s["ee_pos_in_base"], dtype=np.float64) for s in cands])# (N,3)

    # (pre-snap) _in_ws(mid_in_base)：焊缝中点径向/z 在范围内（起/终已由 lookup 端点检查、尖端由 precompute 保证）
    r_xy = np.hypot(mid_all[:, 0], mid_all[:, 1])
    mask_ws = (r_xy >= xy_lo) & (r_xy <= xy_hi) & (mid_all[:, 2] >= z_lo) & (mid_all[:, 2] <= z_hi)

    # 朝向 snap（批量 scipy：每个允许朝向一次 from_matrix）；取【首个】命中 oid，与 _kejian2_snap 逐个版完全一致
    Rv_arr = np.stack([np.asarray(R, dtype=np.float64) for R in R_valid_list])           # (V,3,3)
    oid_best = np.full(Ncand, -1, dtype=np.int64)
    for _oid in range(Rv_arr.shape[0]):
        R_rel = np.einsum("ij,njk->nik", Rv_arr[_oid].T, R_all)                          # (N,3,3)
        e = _sR.from_matrix(R_rel).as_euler("xyz", degrees=True)                         # (N,3) 一次
        hit = (np.abs(e) < float(snap_deg)).all(axis=1) & mask_ws & (oid_best < 0)
        oid_best[hit] = _oid
    mask_hit = mask_ws & (oid_best >= 0)
    n_hit = int(mask_hit.sum())

    # 命中者：snap 到对应 Rv，批量重算 t_new（尖端仍落 standoff 点，snap 后不复检）/ bis_base
    Rv_sel = Rv_arr[np.where(oid_best >= 0, oid_best, 0)]                                # (N,3,3) 非命中处用 oid0(后续不读)
    t_new_all = ee_all - np.einsum("nij,j->ni", Rv_sel, target_world)                    # (N,3)
    bis_all = np.einsum("nij,j->ni", Rv_sel, bis_world)                                  # (N,3)
    bis_all = bis_all / (np.linalg.norm(bis_all, axis=1, keepdims=True) + 1e-12)
    seam_center_all = ee_all - standoff * bis_all                                        # (N,3)

    # （req）正面过滤：bisector(背离工件=approach 方向) base-z<0 ⇒ 焊缝朝下=背面，丢弃
    mask_back_drop = mask_hit & (bis_all[:, 2] < 0.0)
    n_back = int(mask_back_drop.sum())
    mask_front = mask_hit & ~mask_back_drop
    # （req）焊缝中心点在 base 必须 x>0
    mask_xneg_drop = mask_front & (seam_center_all[:, 0] <= 0.0)
    n_xneg = int(mask_xneg_drop.sum())
    mask_survive = mask_front & ~mask_xneg_drop          # 进入 最近点x/overlap/去重/组装的候选

    # 仅对幸存者按【原始候选顺序】循环：工件最近点x + 底座-工件相交 + 轻去重 + 组装（去重需保序，与逐个版一致）
    for i in np.nonzero(mask_survive)[0].tolist():
        oid = int(oid_best[i]); Rv = Rv_arr[oid]
        sol = cands[i]
        ee_pos = ee_all[i]
        t_new = t_new_all[i]
        bis_base = bis_all[i]
        seam_center_base = seam_center_all[i]

        # 工件顶点变换到 base 系（供「底座-工件 XY 相交」过滤用，只算一次）
        v_base = (mesh_v @ Rv.T + t_new) if mesh_v is not None else None   # (V,3) base 系

        # （req）工件距 base_link 原点【欧氏最近】的那个点，其 base-x 分量须 > workpiece_x_min，否则丢弃。
        #   点集 wpx_pts 为工件体素中心（mesh 局部系，与三角形大小解耦）。距离对刚体变换不变，
        #   故把 base 原点反变换到局部系一次做 argmin，只把赢家变回 base 读 x（省整批变换）。
        p_near_base = None                                               # 赢家点 base 系坐标（供过滤+可视化）
        if wpx_pts is not None:
            o_local = -(Rv.T @ t_new)                                    # base 原点在 mesh 局部系
            d = wpx_pts - o_local
            i_near = int(np.argmin(np.einsum("vi,vi->v", d, d)))         # argmin |p|^2（省 sqrt）
            p_near_base = Rv @ wpx_pts[i_near] + t_new                   # 离 base 原点最近的那个点（base 系）
            if float(p_near_base[0]) <= wp_x_min:                        # 看它的 base-x
                n_wpx += 1
                continue

        # （req）固定底座与工件在 base-xy 平面投影不能相交：相交 ⇒ 机械臂压在工件下/工件盖在底座上，丢弃。
        # 用工件【三角形投影并集】精确判定（不再用凸包近似，凹形工件也准确）；底座 AABB 粗筛保证速度。
        if bc_cc is not None and v_base is not None and mesh_f is not None:
            if _base_overlaps_workpiece_tris(bc_cc, bc_rr, bc_umin, bc_umax, v_base[:, :2], mesh_f):
                n_overlap += 1
                continue

        key = (oid, round(float(t_new[0]), 2), round(float(t_new[1]), 2), round(float(t_new[2]), 2))
        if key in seen:                              # 轻去重：同朝向 + 同位置(2cm 粒度)只留一份
            n_dedup += 1
            continue
        seen.add(key)

        R0_ee = _align_rotmat([1.0, 0.0, 0.0], bis_base)   # 末端局部 +x → bisector_base
        goal_quat = rotmat_to_quat_wxyz(R0_ee)
        goal_pose7 = np.concatenate([ee_pos, goal_quat])   # 位置= standoff 落枪点(=ee_pos)
        # （req）正反手按 bisector 在 base-x 的分量定：与 base-x 反向(负 x)=正手 / 否则=反手
        if abs(float(bis_base[0])) < 1e-9:
            print(f"[kejian2] 警告：候选 i={i} bisector base-x 分量≈0，归为反手")
        hand = "forehand" if float(bis_base[0]) < 0.0 else "backhand"
        results.append({
            "workpiece_pose7": mat44_to_pose7(_T(Rv, t_new)),
            "T_workpiece_in_base": _T(Rv, t_new),
            "goal_pose7": goal_pose7,
            "joint_angles": np.asarray(sol["q"], dtype=np.float64),
            "rot_x_deg": float(sol["rot_x_deg"]),
            "rot_y_deg": float(sol["rot_y_deg"]),
            "rot_z_deg": float(sol["rot_z_deg"]),
            "bisector_base": np.asarray(bis_base, dtype=np.float64),
            "seam_center_base": np.asarray(seam_center_base, dtype=np.float64),
            "wpx_near_base": (np.asarray(p_near_base, dtype=np.float64)
                              if p_near_base is not None else None),   # 距 base 原点最近点(base 系，可视化)
            "orientation_id": int(oid),
            "hand": hand,
        })

    fore = [r for r in results if r["hand"] == "forehand"]
    back = [r for r in results if r["hand"] == "backhand"]
    # —— 最后一步：snap 后姿态复检。候选 R snap 到 90°整倍朝向、重算 t 会改变工件姿态，
    #   ④ 里 snap 前的碰撞过滤在此姿态下已失效；这里对最终 (Rv,t_new) 只用 retract 姿态整臂碰撞球
    #   再查一次工件 ESDF，撞则丢弃（判据同 ④：含 clearance_inflate、d<=collision_tolerance）。
    if arm_recheck and results:
        R_res = np.stack([np.asarray(r["T_workpiece_in_base"], dtype=np.float64)[:3, :3] for r in results])
        t_res = np.stack([np.asarray(r["T_workpiece_in_base"], dtype=np.float64)[:3, 3] for r in results])
        safe = solver.recheck_retract_collision_snapped(R_res, t_res).detach().cpu().numpy()
        n_arm_collide = int((~safe).sum())
        results = [r for r, s in zip(results, safe.tolist()) if s]
        fore = [r for r in results if r["hand"] == "forehand"]
        back = [r for r in results if r["hand"] == "backhand"]
    prof["⑤snap+正反手分类(CPU遍历候选)"] = _time.time() - _t
    # —— 逐步过滤计数（前→后，括号=本步丢弃）；这是结果口径信息，无条件打印（不受 verbose 影响） ——
    after_hit = n_hit                          # ② 工作空间 + 朝向 snap 命中
    after_back = n_hit - n_back                # ③ 正面过滤（背面丢）
    after_xneg = after_back - n_xneg           # ④ 焊缝中心 base-x>0（= int(mask_survive.sum())）
    after_wpx = after_xneg - n_wpx             # ⑤ 工件最近点 base-x>wp_x_min
    after_overlap = after_wpx - n_overlap      # ⑥ 底座-工件 XY 投影不相交
    after_dedup = after_overlap - n_dedup      # ⑦ 轻去重（= 组装完、复检前的 results 数）
    n_final = len(results)                     # ⑧ snap 后 retract-工件无碰撞复检 → 合格
    print("[kejian2] 逐步过滤 候选初始位姿（前→后，括号内=本步丢弃）：")
    print(f"  ① lookup 候选（绕末端轴 {len(rot_x)}×{len(rot_y)}×{len(rot_z)} 采样） : {len(cands)}")
    print(f"  ② 工作空间 + 朝向snap 命中            : {len(cands)} → {after_hit}")
    print(f"  ③ 正面过滤(bisector base-z≥0)        : {after_hit} → {after_back}（背面丢 {n_back}）")
    print(f"  ④ 焊缝中心 base-x>0                   : {after_back} → {after_xneg}（x≤0 丢 {n_xneg}）")
    print(f"  ⑤ 工件最近点 base-x>{wp_x_min}          : {after_xneg} → {after_wpx}（丢 {n_wpx}）")
    print(f"  ⑥ 底座-工件 XY 投影不相交            : {after_wpx} → {after_overlap}（相交丢 {n_overlap}）")
    print(f"  ⑦ 轻去重(同朝向 + 2cm 同位)          : {after_overlap} → {after_dedup}（重复丢 {n_dedup}）")
    print(f"  ⑧ snap后 retract-工件无碰撞复检      : {after_dedup} → {n_final}（碰撞丢 {n_arm_collide}）")
    print(f"  ⇒ 合格 {n_final}（正手 {len(fore)} / 反手 {len(back)}；snap_deg={snap_deg}°）")
    return {"forehand": fore, "backhand": back}, prof


def _print_prof_table(prof: dict) -> None:
    """打印 === 各阶段耗时 === 表（含百分比与合计）。"""
    _total = sum(prof.values())
    print("[kejian2] === 各阶段耗时 ===")
    for _k, _v in prof.items():
        print(f"[kejian2]   {_k:32s} {_v:8.3f}s  ({100.0 * _v / max(_total, 1e-9):5.1f}%)")
    print(f"[kejian2]   {'合计':32s} {_total:8.3f}s")


def plan_init_pose_kejian2(obj_fp: str, weld_json: str, seam_id: int = 0,
                           viz: bool = False) -> Dict[str, list]:
    """新逻辑求解（lookup 关节角采样 + 允许朝向 snap + 正面过滤 + 正反手分类），单条焊缝：
    返回 {"forehand":[...], "backhand":[...]}。

    ① InitPoseLookupSolver（n^6 关节角采样，复用本文件已自带的求解器）对本焊缝反解出候选工件位姿
       (R,t,q)，并用「整臂/retract 碰撞球 vs 工件 ESDF」过滤掉碰撞解（= plan_init_pose 原逻辑碰撞）；
    ② 从候选里挑「旋转 R 与 4 种允许朝向某一种 xyz 三方向逐轴误差 < snap_deg」者，把 R snap 到该朝向、
       重算 t 让焊枪尖端（FK ee_pos）仍精确落在焊缝 standoff 点（snap 后不复检范围/碰撞）；
    ③ 过滤：焊缝中心点须在 base 系 x>0；焊缝须在「正面」——bisector（背离工件=焊枪 approach 方向）
       在 base z 分量为负 ⇒ 焊缝朝下=背面，丢弃；且【固定底座(xiaoyu_base_link)碰撞球 与 工件】在
       base-xy 平面投影不能相交（相交=机械臂压在工件下/工件盖在底座上，丢弃；工件投影取【三角形投影并集】
       精确判定，非凸包近似，凹形工件也准确）；
    ④ 正反手按【bisector】在 base-x 的分量定：与 base-x 反向(负 x)=正手，否则=反手（焊缝几何，
       与关节构型无关）。lookup 过滤范围/缓存全部走 plan_init_pose_kejian2 段。

    注：工件级 ②③（solver/ESDF/joint 表）只构建一次；要一次跑同工件多条焊缝、摊薄这次开销，
    用 plan_init_pose_kejian2_all。"""
    ctx = _kejian2_build_ctx(obj_fp)
    welds = load_welds(weld_json)
    weld = next((w for w in welds if int(w["idx"]) == int(seam_id)), None)
    if weld is None:
        raise IndexError(f"seam_id={seam_id} 不在 {weld_json}（共 {len(welds)} 条）")

    res, prof_weld = _kejian2_solve_weld(ctx, weld)
    _print_prof_table({**ctx["prof_setup"], **prof_weld})
    if viz and (res["forehand"] or res["backhand"]):
        _show_kejian2_results(ctx["cfg"], obj_fp, weld, res, stride=5)
    return res


def plan_init_pose_kejian2_all(obj_fp: str, weld_json: str,
                               seam_ids: Optional[List[int]] = None,
                               viz: bool = False,
                               save_dir: Optional[str] = None,
                               save_k: int = 15,
                               log_path: Optional[str] = None,
                               filter_short: bool = False) -> Dict[int, Dict[str, list]]:
    """一次运行处理同一工件的【多条/全部】焊缝：工件级 ②③（solver/ESDF/joint 表）只构建一次，
    再逐条焊缝跑 ④⑤。返回 {seam_id: {"forehand":[...], "backhand":[...]}}。

    seam_ids=None → weld_json 里全部焊缝；否则只处理给定 idx（缺失的报错列出）。
    filter_short=True → 先丢弃长度 < SHORT_SEAM_LEN_M(3cm) 的短焊缝（不求解、不落盘），并记入日志。
    save_dir 给定时：每条焊缝【解完立刻】存 <save_dir>/seam_<idx>.npy（边算边存，不等全部跑完）。
    log_path 给定时：每条焊缝解完就重写一次该 txt（记录成功率/失败 seam/合格与已存条数；中断也保住已完成的）。
    per-weld 结果与单焊缝 plan_init_pose_kejian2 逐位一致，区别仅在 ②③ 不再每条重复。"""
    import time as _time
    ctx = _kejian2_build_ctx(obj_fp)          # ②③：只一次
    print("[kejian2] === 工件级一次性耗时（②③，全焊缝共享）===")
    _print_prof_table(ctx["prof_setup"])
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)  # 输出目录不存在则创建（在开跑前建好）

    welds = load_welds(weld_json)
    if seam_ids is not None:
        want = [int(s) for s in seam_ids]
        by_idx = {int(w["idx"]): w for w in welds}
        miss = [s for s in want if s not in by_idx]
        if miss:
            raise IndexError(f"seam_id {miss} 不在 {weld_json}（共 {len(welds)} 条："
                             f"{sorted(by_idx)}）")
        welds = [by_idx[s] for s in want]

    filtered: List[dict] = []
    if filter_short:
        kept = []
        for w in welds:
            L = _seam_len_m(w)
            if L < SHORT_SEAM_LEN_M:
                filtered.append({"idx": int(w["idx"]), "length_cm": L * 100.0})
            else:
                kept.append(w)
        if filtered:
            print(f"[kejian2] 过滤掉 {len(filtered)} 条短焊缝(<{SHORT_SEAM_LEN_M * 100:.0f}cm)："
                  f"{[fz['idx'] for fz in filtered]}")
        welds = kept

    out: Dict[int, Dict[str, list]] = {}
    stats: List[dict] = []
    setup_s = sum(ctx["prof_setup"].values())   # 工件级 ②③ 一次性耗时，写日志计时段共享
    t_all = _time.time()
    for wi, weld in enumerate(welds):
        sid = int(weld["idx"])
        print(f"\n[kejian2] ===== 焊缝 seam_id={sid}（{wi + 1}/{len(welds)}）=====")
        res, prof_weld = _kejian2_solve_weld(ctx, weld)
        _t4 = prof_weld.get("④solve_one_weld_lookup(碰撞过滤)", 0.0)
        _t5 = prof_weld.get("⑤snap+正反手分类(CPU遍历候选)", 0.0)
        print(f"[kejian2]   seam {sid}: ④={_t4:.3f}s ⑤={_t5:.3f}s "
              f"（正手 {len(res['forehand'])} / 反手 {len(res['backhand'])}）")
        if save_dir is not None:                  # 边算边存：本条解完立刻落盘
            _save_kejian2_npy(res, os.path.join(save_dir, f"seam_{sid}.npy"), k=save_k, weld=weld)
        out[sid] = res
        n_fore, n_back = len(res["forehand"]), len(res["backhand"])
        stats.append({"idx": sid, "n_fore": n_fore, "n_back": n_back,
                      "saved_fore": min(n_fore, save_k), "saved_back": min(n_back, save_k),
                      "t_solve": _t4 + _t5})
        if log_path is not None:                  # 边算边写日志：中断也保住已完成焊缝的记录
            _write_kejian2_log(log_path, obj_fp, weld_json, save_dir, save_k, stats, len(welds),
                               filtered=filtered, setup_s=setup_s, elapsed_s=_time.time() - t_all)
    # 全被过滤（没有任何焊缝可求解）时也写一次日志，把过滤清单落盘
    if log_path is not None and not welds:
        _write_kejian2_log(log_path, obj_fp, weld_json, save_dir, save_k, stats, len(welds),
                           filtered=filtered, setup_s=setup_s, elapsed_s=_time.time() - t_all)
    _setup = setup_s
    print(f"\n[kejian2] 全部 {len(welds)} 条焊缝完成：工件级 ②③ 一次 {_setup:.3f}s "
          f"+ 逐焊缝 ④⑤ 共 {_time.time() - t_all:.3f}s")
    if log_path is not None:
        print(f"[kejian2] 求解日志已写入 {log_path}")
    if viz and welds:
        w0 = welds[0]
        if out[int(w0['idx'])]["forehand"] or out[int(w0['idx'])]["backhand"]:
            _show_kejian2_results(ctx["cfg"], obj_fp, w0, out[int(w0['idx'])], stride=5)
    return out




if __name__ == "__main__":
    main()
