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
   retract 球 vs 工件 ESDF」做碰撞过滤，按 upright/cube 评分取最优。碰撞后端用裸 cuRobo
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


def cube_score_batch(R):
    """R 各元素接近 {-1,0,1} 的程度（负的"到最近整值距离和"）；越大越接近 90° 倍数正交姿态。"""
    import torch
    abs_r = torch.abs(R)
    dist = torch.minimum(abs_r, 1.0 - abs_r)
    return -torch.sum(dist, dim=(-2, -1))


def upright_score_batch(R):
    """工件竖直程度 |R[2,2]| ∈ [0,1]，≥0.9 表示不歪。"""
    import torch
    return torch.abs(R[..., 2, 2])


def quat_wxyz_to_x_axis_batch(quat):
    """四元数 (B,4 wxyz) → 旋转矩阵第 0 列（局部 +x 轴在世界的方向）(B,3)。"""
    import torch
    w = quat[..., 0]; x = quat[..., 1]; y = quat[..., 2]; z = quat[..., 3]
    return torch.stack([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    ], dim=-1)


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
        "_theta_deg": sol["theta_deg"],
        "_phi_deg": sol.get("phi_deg", 0.0),
        "_cube_score": sol["cube_score"],
        "_d_link": sol["d_link"],
        "_d_retract": sol["d_retract"],
    }


# ---- 求解器（移植 ArmPoseSolver voxel 子集 + Lookup 子类） ----
class InitPoseLookupSolver:
    """查表式初始位姿求解器（裸 cuRobo RobotWorld + 工件 ESDF voxel 碰撞）。

    机器人 yml 取 cfg.robot_cfg_path，retract/joint_names 取 cfg。precompute_joint_table 一次性
    预计算 N=n_per_dof^6 个 q 的 (ee_pos, ee_x, link_spheres)；solve_one_weld_lookup 每条焊缝 batch
    反解 + 碰撞过滤 + 评分。
    """

    def __init__(self, cfg, obj_fp: Optional[str],
                 collision_tolerance: float = 0.03, voxel_size: float = 0.02,
                 n_per_dof: int = 7):
        import gt_gen.compat  # noqa: F401  warp shim，须在 import curobo 前
        from curobo.types.base import TensorDeviceType

        self.cfg = cfg
        self.tensor_args = TensorDeviceType()
        self.obj_fp = obj_fp
        self.collision_tolerance = collision_tolerance
        self.voxel_size = voxel_size
        self.n_per_dof = n_per_dof

        self._load_robot()
        self.robot_world = None
        self.world_voxel_coll = None
        # 精确碰撞用：直接查工件 mesh 的 signed distance（不经体素，无量化误差）。
        # _voxel 路径的体素分辨率(voxel_size≈0.02)远大于最小碰撞球半径(0.003)，
        # 球心落在表面附近的体素时 ESDF 仍读“自由”→ 漏判；mesh 查询按三角面解析距离，无此问题。
        self.mesh_coll = None
        self._mesh_weight = self.tensor_args.to_device([1.0])
        self._mesh_act = self.tensor_args.to_device([0.0])
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
        self.mesh_coll = mesh_coll  # 留住：碰撞判定直接查 mesh（精确），不查体素化后的 ESDF

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

    def _mesh_penetration_batch(self, spheres):
        """spheres (B,K,4)(xyz,r) 在 mesh 系 → (B,) 最深单球穿透深度（米，>0 撞）。

        直接查工件 mesh 的精确 signed distance（compute_esdf）：每个球心到工件面的
        带符号距离 sdf（约定 正=工件内、负=工件外，由 mesh 解析三角面距离给出，无体素量化）。
        球壳穿透深度 = sdf + r：>0 即该球扎入工件。取所有球的最大值=最坏穿透。
        半径<0 的“关闭球”由 kernel 返回 sdf=0，penetration=负，自然被 max 忽略（再额外屏蔽以防万一）。
        """
        import torch
        from curobo.geom.sdf.world import CollisionQueryBuffer
        x_sph = spheres.unsqueeze(1)  # (B,1,K,4)
        buf = CollisionQueryBuffer.initialize_from_shape(
            x_sph.shape, self.tensor_args, self.mesh_coll.collision_types)
        sdf = self.mesh_coll.get_sphere_distance(
            x_sph, buf, self._mesh_weight, self._mesh_act, compute_esdf=True)  # (B,1,K) 正=工件内
        B, _, K = sdf.shape
        sdf = sdf.view(B, K)
        radii = spheres[..., 3]                       # (B,K)
        penetration = sdf + radii                     # >0 即球壳扎入工件
        penetration = torch.where(radii > 0, penetration,
                                  torch.full_like(penetration, -1e9))  # 屏蔽关闭球
        return penetration.amax(dim=-1)               # (B,) 最坏单球穿透

    # ---- 离线预计算 ----
    def precompute_joint_table(self):
        """6 关节限位等距 n^6 采样 → batch FK 算 (ee_pos, ee_x, link_spheres)。
        （源 solve_arm_pose_lookup.py L206–308 原样。）"""
        import torch
        kc = self.robot_cfg.kinematics.kinematics_config
        jl = kc.joint_limits.position
        if jl.shape[0] != 2:
            jl = jl.T
        low = jl[0].cpu().numpy()
        high = jl[1].cpu().numpy()
        n_dof = len(low)
        assert n_dof == 6, f"expected 6 DoF, got {n_dof}"

        n = self.n_per_dof
        N = n ** n_dof
        self.N = N
        print(f"[lookup] sampling {n}^{n_dof} = {N} joint configs in limits")
        for i, (lo, hi) in enumerate(zip(low, high)):
            print(f"  joint {i}: [{lo:.3f}, {hi:.3f}]")

        grids = [np.linspace(lo, hi, n) for lo, hi in zip(low, high)]
        mesh = np.meshgrid(*grids, indexing="ij")
        q_grid = np.stack([m.ravel() for m in mesh], axis=1).astype(np.float32)
        q_grid_t = torch.tensor(q_grid, device=self.tensor_args.device,
                                dtype=self.tensor_args.dtype)

        chunk = 4096
        ee_pos_list, ee_x_list, link_spheres_list = [], [], []
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, N, chunk):
                qb = q_grid_t[i:i + chunk]
                state = self.robot_world.get_kinematics(qb)
                ee_pos_list.append(state.ee_position.detach().clone())
                ee_x = quat_wxyz_to_x_axis_batch(state.ee_quaternion)
                ee_x_list.append(ee_x.detach().clone())
                link_spheres_list.append(state.link_spheres_tensor.detach().clone())

        self.ee_pos_t = torch.cat(ee_pos_list, dim=0)
        self.ee_x_t = torch.cat(ee_x_list, dim=0)
        self.ee_x_t = self.ee_x_t / (torch.norm(self.ee_x_t, dim=-1, keepdim=True) + 1e-12)
        self.link_spheres_t = torch.cat(link_spheres_list, dim=0)
        self.K_link = self.link_spheres_t.shape[1]
        self.q_table_t = q_grid_t

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

    # ---- 在线求解 ----
    def solve_one_weld_lookup(self, weld: Dict,
                              thetas_deg: Tuple[float, ...] = (0.0, 15.0, -15.0, 30.0, -30.0),
                              phis_deg: Tuple[float, ...] = (0.0, 15.0, -15.0, 30.0, -30.0),
                              diagnostic: bool = False) -> Optional[Dict]:
        """每条焊缝 batch GPU 求解。θ×φ axis 偏移 → align(bisector→axis) → t=ee_pos−R@mid →
        球反变到 mesh world → mesh 精确穿透 batch 碰撞 → safe → upright*10+cube 评分取 best。
        （源 solve_arm_pose_lookup.py L346–539 原样。）"""
        import torch
        device = self.tensor_args.device
        dtype = self.tensor_args.dtype

        mid = torch.tensor(weld["mid_world"], device=device, dtype=dtype)
        bisector = torch.tensor(weld["bisector_world"], device=device, dtype=dtype)
        bisector = bisector / (torch.norm(bisector) + 1e-12)
        p0 = torch.tensor(weld["p0_world"], device=device, dtype=dtype)
        p1 = torch.tensor(weld["p1_world"], device=device, dtype=dtype)
        tangent_w = (p1 - p0) / (torch.norm(p1 - p0) + 1e-12)
        normal_w = torch.cross(bisector, tangent_w, dim=-1)
        normal_w = normal_w / (torch.norm(normal_w) + 1e-12)

        N = self.N
        axis_q = -self.ee_x_t  # (N,3)
        self._ensure_voxel_pose_set()

        all_solutions = []
        diag_stats = []
        for theta_deg in thetas_deg:
            for phi_deg in phis_deg:
                theta_rad = torch.tensor(np.deg2rad(theta_deg), device=device, dtype=dtype)
                phi_rad = torch.tensor(np.deg2rad(phi_deg), device=device, dtype=dtype)
                with torch.no_grad():
                    R_normal_th = batch_axis_angle_rotmat(
                        normal_w.unsqueeze(0), theta_rad.expand(1))[0]
                    R_tangent_negph = batch_axis_angle_rotmat(
                        tangent_w.unsqueeze(0), (-phi_rad).expand(1))[0]
                    R_extra = R_tangent_negph @ R_normal_th
                    axis_tp = axis_q @ R_extra.T
                    axis_tp = axis_tp / (torch.norm(axis_tp, dim=-1, keepdim=True) + 1e-12)
                    R = batch_align_rotation(bisector, axis_tp)  # (N,3,3)
                    R_mid = torch.einsum("nij,j->ni", R, mid)
                    t_arr = self.ee_pos_t - R_mid  # (N,3)

                    # 球反变到 mesh_world：p_world = R^T @ (p_base - t)
                    link_xyz_b = self.link_spheres_t[..., :3]
                    link_r = self.link_spheres_t[..., 3]
                    deltas_link = link_xyz_b - t_arr[:, None, :]
                    link_xyz_w = torch.einsum("nji,nkj->nki", R, deltas_link)
                    link_spheres_w = torch.cat([link_xyz_w, link_r.unsqueeze(-1)], dim=-1)

                    ret_xyz_b = self.retract_spheres_t[:, :3]
                    ret_r = self.retract_spheres_t[:, 3]
                    ret_xyz_b_n = ret_xyz_b[None, :, :].expand(N, -1, -1)
                    deltas_ret = ret_xyz_b_n - t_arr[:, None, :]
                    ret_xyz_w = torch.einsum("nji,nkj->nki", R, deltas_ret)
                    ret_r_n = ret_r[None, :].expand(N, -1)
                    ret_spheres_w = torch.cat([ret_xyz_w, ret_r_n.unsqueeze(-1)], dim=-1)

                    chunk = 8192
                    d_link_list, d_ret_list = [], []
                    for i in range(0, N, chunk):
                        d_link_list.append(
                            self._mesh_penetration_batch(link_spheres_w[i:i + chunk]))
                        d_ret_list.append(
                            self._mesh_penetration_batch(ret_spheres_w[i:i + chunk]))
                    d_link = torch.cat(d_link_list)
                    d_ret = torch.cat(d_ret_list)

                    safe_mask = (d_link <= self.collision_tolerance) & \
                                (d_ret <= self.collision_tolerance)
                    n_safe = int(safe_mask.sum().item())
                    diag_stats.append((theta_deg, phi_deg, n_safe,
                                       float(d_link.min()), float(d_ret.min())))
                    if diagnostic:
                        print(f"      [θ={theta_deg:+.0f}° φ={phi_deg:+.0f}°] N={N} safe={n_safe} "
                              f"min_d_link={float(d_link.min()):.4f} "
                              f"min_d_ret={float(d_ret.min()):.4f}")
                    if not safe_mask.any():
                        continue
                    safe_idx = safe_mask.nonzero(as_tuple=True)[0]
                    R_safe = R[safe_idx]
                    upright = upright_score_batch(R_safe)
                    cube = cube_score_batch(R_safe)
                    scores = upright * 10.0 + cube

                top_per = min(50, safe_idx.shape[0])
                top_local = torch.argsort(scores, descending=True)[:top_per]
                for li in top_local:
                    gi = int(safe_idx[li].item())
                    all_solutions.append({
                        "q_idx": gi,
                        "q": self.q_table_t[gi].cpu().numpy(),
                        "R": R[gi].cpu().numpy(),
                        "t": t_arr[gi].cpu().numpy(),
                        "theta_deg": theta_deg,
                        "phi_deg": phi_deg,
                        "ee_pos_in_base": self.ee_pos_t[gi].cpu().numpy(),
                        "ee_x_in_base": self.ee_x_t[gi].cpu().numpy(),
                        "d_link": float(d_link[gi].item()),
                        "d_retract": float(d_ret[gi].item()),
                        "cube_score": float(cube[li].item()),
                        "upright_score": float(upright[li].item()),
                        "combined_score": float(scores[li].item()),
                    })

        if not all_solutions:
            print(f"      [FAIL diag] weld {weld['idx']}: 所有 {len(diag_stats)} 个 (θ,φ) 采样 stats:")
            for td, pd, sn, dl, dr in sorted(diag_stats, key=lambda x: (x[0], x[1])):
                reason = "OK" if sn else ("RETRACT撞" if dr > self.collision_tolerance
                                          else "LINK撞" if dl > self.collision_tolerance else "其他")
                print(f"        θ={td:+4.0f}° φ={pd:+4.0f}°: safe={sn:6d}/{N}  "
                      f"min_d_link={dl:.4f}  min_d_ret={dr:.4f}  [{reason}]")
            n_zero = sum(1 for s in diag_stats if s[2] == 0)
            print(f"      [FAIL diag] 共 {n_zero}/{len(diag_stats)} 个采样完全无解 (safe=0)，"
                  f"全局 min_d_link={min(s[3] for s in diag_stats):.4f}, "
                  f"min_d_ret={min(s[4] for s in diag_stats):.4f} (tol={self.collision_tolerance:.4f})")
            return None

        all_solutions.sort(key=lambda s: -s["combined_score"])
        best = all_solutions[0]
        if diagnostic:
            R, t = best["R"], best["t"]
            mid_in_base = R @ weld["mid_world"] + t
            err_mid_pos = float(np.linalg.norm(mid_in_base - best["ee_pos_in_base"]))
            so3_err = float(np.linalg.norm(R @ R.T - np.eye(3)))
            print(f"      [sanity best] err_mid_pos={err_mid_pos:.4f} |R*R^T-I|={so3_err:.4f} "
                  f"upright={best['upright_score']:.3f} "
                  f"θ={best['theta_deg']:+.0f}° φ={best['phi_deg']:+.0f}°")
        return best


# ---- 求解结果可视化（复用 init_space_geometries） ----
def show_lookup_solution(cfg, obj_fp: str, weld: Dict, sol: Dict):
    """复用 init_space_geometries（整臂碰撞球 + init_free 盒 + base 架），再叠加按解出的
    T_workpiece_in_base 摆放的工件网格（浅灰半透）+ 绿色焊缝线 + 焊枪头落点小球。"""
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

    # 焊缝线 p0→p1（绿色）+ 焊枪头落点（= R@mid + t，应贴在 ee_pos）
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

    print(f"显示 weld {weld['idx']} 解（关闭窗口继续）…")
    o3d.visualization.draw_geometries(
        geoms, window_name=f"plan_init_pose: weld {weld['idx']} 解 + 工件 + 整臂碰撞球")


# ============================================================================
# CLI
# ============================================================================
def _run_solve(args):
    """lookup 求解子流程（--solve）。"""
    from gt_gen.config import load_config
    cfg = load_config()
    thetas = tuple(float(x) for x in args.thetas_deg.split(",") if x.strip())
    phis = tuple(float(x) for x in args.phis_deg.split(",") if x.strip())
    print(f"[solve] obj={args.obj}")
    print(f"[solve] weld-json={args.weld_json}")
    print(f"[solve] θ={list(thetas)}° φ={list(phis)}° 总采样={len(thetas)}×{len(phis)}="
          f"{len(thetas) * len(phis)}; q_table={args.n_per_dof}^6={args.n_per_dof ** 6}")

    solver = InitPoseLookupSolver(
        cfg, args.obj, collision_tolerance=args.collision_tolerance,
        voxel_size=args.voxel_size, n_per_dof=args.n_per_dof)
    print("[solve] precomputing joint table（与工件无关，仅一次）…")
    solver.precompute_joint_table()

    welds = load_welds(args.weld_json)
    if args.limit > 0:
        welds = welds[:args.limit]
    stem = os.path.splitext(os.path.basename(args.obj))[0]

    n_solved = 0
    for i, w in enumerate(welds):
        ts = time.time()
        sol = solver.solve_one_weld_lookup(w, thetas, phis, diagnostic=args.diagnostic)
        dt = (time.time() - ts) * 1000.0
        if sol is None:
            print(f"[{i + 1:3d}/{len(welds)}] weld {w['idx']}: FAIL ({dt:.0f}ms)")
            continue
        n_solved += 1
        print(f"[{i + 1:3d}/{len(welds)}] weld {w['idx']}: OK ({dt:.0f}ms) "
              f"θ={sol['theta_deg']:+.0f}° φ={sol['phi_deg']:+.0f}° "
              f"upright={sol['upright_score']:.2f} cube={sol['cube_score']:.2f} "
              f"d_link={sol['d_link']:.3f} d_ret={sol['d_retract']:.3f} "
              f"q={np.round(sol['q'], 3).tolist()}")
        target = to_save_format(w, sol, solver.joint_names)
        fp = save_seam_pkl(args.out_dir, stem, w, target, args.obj, n_seg=args.n_seg)
        print(f"      saved → {fp}")
        if args.viz:
            show_lookup_solution(cfg, args.obj, w, sol)

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
    ap.add_argument("--n-per-dof", type=int, default=7, help="solve：每关节采样档数，q_table=n^6")
    ap.add_argument("--thetas-deg", default="0,15,-15,30,-30",
                    help="solve：axis 沿 tangent 偏（drag/push）采样（度）")
    ap.add_argument("--phis-deg", default="0,15,-15,30,-30",
                    help="solve：axis 沿 normal 偏采样（度）。总采样=thetas×phis")
    ap.add_argument("--collision-tolerance", type=float, default=0.03,
                    help="solve：允许穿透阈值（米）")
    ap.add_argument("--voxel-size", type=float, default=0.02, help="solve：工件 ESDF 体素大小（米）")
    ap.add_argument("--n-seg", type=int, default=20, help="solve：seam_line 插值点数")
    ap.add_argument("--limit", type=int, default=-1, help="solve：只处理前 N 条焊缝（-1=全部）")
    ap.add_argument("--viz", action="store_true",
                    help="solve：每条成功焊缝开窗可视化（复用 show_init_space 的几何）")
    ap.add_argument("--diagnostic", action="store_true", help="solve：逐 (θ,φ) 诊断打印")
    args = ap.parse_args()

    if args.solve:
        if not args.obj or not args.weld_json:
            ap.error("--solve 需同时给 --obj 和 --weld-json")
        _run_solve(args)
        return

    from gt_gen.config import load_config
    cfg = load_config()
    show_init_space(cfg, q=args.q, solid_spheres=args.solid)


if __name__ == "__main__":
    main()
