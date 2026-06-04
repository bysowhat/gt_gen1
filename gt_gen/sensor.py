"""Step 3: 传感器模拟（对真值 obj mesh 做 raycast）。

见 docs/gt-generation-curobo-implementation.md §4.1、privileged-nbv.md §4.5 ②/⑥。

约定（与 configs/default.yaml、scripts/viz_arm_camera.py 一致）：
- 相机帧 = base 系下 T_base_cam = T_base_Link6 @ T(extrinsic_pos, extrinsic_quat_wxyz)。
- 光学帧 OpenCV：+Z 朝前(视线)、+X 右、+Y 下；像素射线 d=normalize([(u-cx)/fx,(v-cy)/fy,1])。
- raycast 纯几何：输入相机位姿(4x4,base)+内参+真值 mesh(base 系)，
  输出 base 系下的点 (free_points, occ_points)；体素化交给 Step 4。
- truth_scene 是已变换到 base 系的 trimesh.Trimesh。
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


# ---- 位姿小工具（wxyz 约定，同 viz_arm_camera） ----
def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def pose_to_T(pos, quat_wxyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_R(quat_wxyz)
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def load_camera_model(config) -> dict:
    """从 config 读相机内参 + 手眼外参(相对 Link6) + 深度上限，扁平成一个 dict。"""
    cam = config.camera
    intr = cam["intrinsics"]
    return {
        "fx": float(intr["fx"]), "fy": float(intr["fy"]),
        "cx": float(intr["cx"]), "cy": float(intr["cy"]),
        "width": int(cam["width"]), "height": int(cam["height"]),
        "extrinsic_pos": list(cam["extrinsic_pos"]),
        "extrinsic_quat_wxyz": list(cam["extrinsic_quat_wxyz"]),
        "mount_link": cam.get("mount_link", "Link6"),
        "max_depth": float(config.max_depth_m),
    }


def build_kinematics(config):
    """轻量 CudaRobotModel（仅 FK，不起 MotionGen）；link_names 含 Link6 供相机外参挂载。"""
    import gt_gen.compat  # noqa: F401
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.util_file import load_yaml

    gt_gen.compat.apply_trimesh_shim()
    d = load_yaml(config.robot_cfg_path)
    d["robot_cfg"]["kinematics"]["link_names"] = ["Link6"]
    ta = TensorDeviceType()
    rc = RobotConfig.from_dict(d["robot_cfg"], ta)
    return CudaRobotModel(rc.kinematics)


def link6_pose(kin_provider, q) -> Tuple[np.ndarray, np.ndarray]:
    """FK 求 Link6 在 base 系的位姿。kin_provider 可为 CuroboHandle 或裸 CudaRobotModel。
    返回 (pos[3], quat_wxyz[4])。
    """
    import torch
    mg = getattr(kin_provider, "mg", None)
    kin = mg.kinematics if mg is not None else kin_provider
    qt = torch.tensor([list(q)], dtype=torch.float32, device="cuda")
    st = kin.get_state(qt)
    l6 = st.link_pose["Link6"]
    return (l6.position[0].detach().cpu().numpy(),
            l6.quaternion[0].detach().cpu().numpy())


def camera_pose_from_config(kin_provider, q, camera_model) -> np.ndarray:
    """由关节构型 q + 手眼外参，求相机在 base 系的 4x4 位姿 T_base_cam。"""
    pos, quat = link6_pose(kin_provider, q)
    T_base_l6 = pose_to_T(pos, quat)
    T_l6_cam = pose_to_T(camera_model["extrinsic_pos"], camera_model["extrinsic_quat_wxyz"])
    return T_base_l6 @ T_l6_cam


def pixel_ray_dirs(camera_model, pixel_stride: int = 16) -> np.ndarray:
    """生成（子采样）像素射线方向（光学帧，单位向量）。返回 (K,3)。"""
    cm = camera_model
    us = np.arange(0, cm["width"], pixel_stride)
    vs = np.arange(0, cm["height"], pixel_stride)
    uu, vv = np.meshgrid(us, vs)
    px = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float64)
    d = np.stack([(px[:, 0] - cm["cx"]) / cm["fx"],
                  (px[:, 1] - cm["cy"]) / cm["fy"],
                  np.ones(px.shape[0])], axis=1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def load_truth_scene(obj_path, mesh_pose: Optional[Sequence[float]] = None):
    """加载 obj 为 trimesh，并按 mesh_pose=[x,y,z,qw,qx,qy,qz] 变换到 base 系。"""
    import trimesh
    m = trimesh.load(obj_path, force="mesh")
    if mesh_pose is not None:
        mp = np.asarray(mesh_pose, dtype=float)
        m.apply_transform(pose_to_T(mp[:3], mp[3:7]))
    return m


def raycast_observe(camera_pose, camera_model, truth_scene, max_depth,
                    pixel_stride: int = 16, free_step: Optional[float] = None,
                    near: Optional[float] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """从相机位姿向真值 mesh 投射光线。

    camera_pose: 4x4 base 系相机位姿（T_base_cam，列向量 [:3,2] 为 +Z 视线）。
    truth_scene: base 系 trimesh.Trimesh。
    返回 (free_points, occ_points)，均为 base 系点 (·,3)：
      free_points —— 各射线从 near 到 min(命中距, max_depth) 前一格的采样点（自由空间）；
      occ_points  —— 命中距 ≤ max_depth 的命中点（占据）。
    """
    free_step = 0.02 if free_step is None else float(free_step)
    near = free_step if near is None else float(near)

    R = np.asarray(camera_pose)[:3, :3]
    org = np.asarray(camera_pose)[:3, 3]
    d_opt = pixel_ray_dirs(camera_model, pixel_stride)        # (K,3) 光学帧
    d_base = d_opt @ R.T                                      # (K,3) base 系
    nray = d_base.shape[0]
    origins = np.tile(org, (nray, 1))

    # multiple_hits=True：取回每条射线【全部】交点，再在下面保留最近的一个。
    # （trimesh 纯 Python 求交在 multiple_hits=False 时只给一个交点且不保证最近，
    #  对工字梁等多层薄壁会误把远侧内壁当命中 → 必须自己挑最近。）
    locs, idx_ray, _ = truth_scene.ray.intersects_location(
        origins, d_base, multiple_hits=True)

    t_hit = np.full(nray, np.inf)
    hit_loc = np.zeros((nray, 3))
    if len(idx_ray):
        dist = np.linalg.norm(locs - origins[idx_ray], axis=1)
        # 按距离降序写入 → 每条射线最近命中最后写、获胜
        for k in np.argsort(dist)[::-1]:
            r = int(idx_ray[k])
            t_hit[r] = dist[k]
            hit_loc[r] = locs[k]

    free_chunks, occ = [], []
    for r in range(nray):
        th = t_hit[r]
        hit = np.isfinite(th) and th <= max_depth
        end = min(th, max_depth) if np.isfinite(th) else max_depth
        s = np.arange(near, max(end - free_step, near), free_step)
        if s.size:
            free_chunks.append(origins[r] + np.outer(s, d_base[r]))
        if hit:
            occ.append(hit_loc[r])

    free_points = (np.concatenate(free_chunks) if free_chunks
                   else np.empty((0, 3)))
    occ_points = np.asarray(occ) if occ else np.empty((0, 3))
    return free_points, occ_points
