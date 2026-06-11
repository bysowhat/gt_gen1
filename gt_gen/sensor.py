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


# ============================================================================
# warp 视锥体素雕刻（主循环观测路径）—— 替代逐像素稀疏 raycast，产实心自由锥。
#
# 思路：取视锥内每个体素中心，与相机中心连线，wp.mesh_query_ray 判最近命中 t_hit，
# 按体素到相机的距离 r 分类：r<t-½vox→FREE、|r-t|≤½vox→OCC、r>t→被挡(不动)、无命中→FREE。
# 产出实心 FREE（vs trimesh 稀疏细管），扛得住 sync 的 inflate=1 障碍膨胀。
#
# 与 trimesh raycast_observe 并存：那条仍是 Step 3 几何原语/可视化工具，本节专供主循环。
# 见 docs/implementation-steps.md、scripts/proto_warp_carve.py（原型对照）。
# ============================================================================

_WP_INIT = False                 # wp.init() 只跑一次（惰性）
_WP_MESH_CACHE: dict = {}        # id(truth_scene) -> wp.Mesh（一次运行一个 scene）
_ROI_CENTERS_CACHE: dict = {}    # id(voxmap) -> (idx_all, centers_all)（ROI 固定）
_CARVE_KERNEL = None             # 编译一次的 @wp.kernel


def _ensure_wp():
    """惰性 wp.init()（首次调用时初始化 CUDA 设备）。"""
    global _WP_INIT
    if not _WP_INIT:
        import warp as wp
        wp.init()
        _WP_INIT = True


def _carve_kernel():
    """编译并缓存逐体素雕刻 kernel：每线程一个体素，连线相机中心查最近命中并分类。

    out: 1=FREE、2=OCCUPIED、0=被挡(UNKNOWN,不写)。
    """
    global _CARVE_KERNEL
    if _CARVE_KERNEL is not None:
        return _CARVE_KERNEL
    import warp as wp

    @wp.kernel
    def carve(mesh: wp.uint64, cam_o: wp.vec3,
              centers: wp.array(dtype=wp.vec3), ranges: wp.array(dtype=wp.float32),
              max_t: wp.float32, half_vox: wp.float32, out: wp.array(dtype=wp.int32)):
        tid = wp.tid()
        c = centers[tid]
        r = ranges[tid]
        dn = wp.normalize(c - cam_o)
        query = wp.mesh_query_ray(mesh, cam_o, dn, max_t)
        if query.result:
            th = query.t
            if r < th - half_vox:
                out[tid] = 1            # FREE：表面前方、无遮挡
            elif r <= th + half_vox:
                out[tid] = 2            # OCCUPIED：命中表面那一层
            else:
                out[tid] = 0            # 被挡 → UNKNOWN（不动）
        else:
            out[tid] = 1                # 该方向 max_depth 内无命中 → FREE

    _CARVE_KERNEL = carve
    return carve


def _truth_warp_mesh(truth_scene):
    """由 truth_scene.vertices/faces 建 wp.Mesh，按 id(truth_scene) 缓存（避免每帧重建）。"""
    import warp as wp
    key = id(truth_scene)
    mesh = _WP_MESH_CACHE.get(key)
    if mesh is None:
        _ensure_wp()
        verts = np.asarray(truth_scene.vertices, dtype=np.float32)
        faces = np.asarray(truth_scene.faces, dtype=np.int32).reshape(-1)
        mesh = wp.Mesh(points=wp.array(verts, dtype=wp.vec3, device="cuda"),
                       indices=wp.array(faces, dtype=wp.int32, device="cuda"))
        _WP_MESH_CACHE[key] = mesh
    return mesh


def _roi_centers(voxmap):
    """ROI 全体素下标 idx_all (N,3) + 中心 centers_all (N,3)，按 id(voxmap) 缓存（ROI 固定）。"""
    key = id(voxmap)
    cached = _ROI_CENTERS_CACHE.get(key)
    if cached is None:
        sh = voxmap.shape
        gi, gj, gk = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]), np.arange(sh[2]),
                                 indexing="ij")
        idx_all = np.stack([gi.ravel(), gj.ravel(), gk.ravel()], axis=1).astype(np.int64)
        centers_all = voxmap.voxel_to_world(idx_all)
        cached = (idx_all, centers_all)
        _ROI_CENTERS_CACHE[key] = cached
    return cached


def carve_observe(voxmap, camera_pose, camera_model, truth_scene, max_depth=None
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """warp 视锥体素雕刻：单次观测，返回应翻为 (FREE, OCCUPIED) 的体素下标。

    camera_pose : 4x4 base 系相机位姿（T_base_cam，[:3,2] 为 +Z 视线）。
    max_depth   : None → camera_model["max_depth"]。
    返回 (free_idx, occ_idx)，均为 (·,3) int64 体素下标（已在网格内）；不修改 voxmap（由调用方写）。

    被遮挡(r>t_hit)的体素归 UNKNOWN（不在返回里），保留三态语义；写状态/粘滞由调用方负责。
    """
    import warp as wp
    if max_depth is None:
        max_depth = camera_model["max_depth"]
    max_depth = float(max_depth)

    mesh = _truth_warp_mesh(truth_scene)
    kernel = _carve_kernel()
    idx_all, centers_all = _roi_centers(voxmap)

    org = np.asarray(camera_pose)[:3, 3]
    R = np.asarray(camera_pose)[:3, :3]
    fx, fy = camera_model["fx"], camera_model["fy"]
    cx, cy = camera_model["cx"], camera_model["cy"]
    W, H = camera_model["width"], camera_model["height"]
    vs = voxmap.voxel_size
    near = vs

    # CPU 向量化选视锥内体素：投影到画幅 + 深度/连线范围
    Xc = (centers_all - org) @ R                                  # = R^T (C-org)
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * Xc[:, 0] / z + cx
        v = fy * Xc[:, 1] / z + cy
    rr = np.linalg.norm(centers_all - org, axis=1)
    infr = ((z > near) & (z <= max_depth) & (u >= 0) & (u <= W) &
            (v >= 0) & (v <= H) & (rr <= max_depth))
    idx_f = idx_all[infr]
    if idx_f.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty((0, 3), dtype=np.int64)

    cw = wp.array(centers_all[infr].astype(np.float32), dtype=wp.vec3, device="cuda")
    rw = wp.array(rr[infr].astype(np.float32), dtype=wp.float32, device="cuda")
    ow = wp.zeros(idx_f.shape[0], dtype=wp.int32, device="cuda")
    wp.launch(kernel, dim=idx_f.shape[0],
              inputs=[mesh.id, wp.vec3(float(org[0]), float(org[1]), float(org[2])),
                      cw, rw, max_depth, float(0.5 * vs), ow], device="cuda")
    wp.synchronize()
    out = ow.numpy()

    return idx_f[out == 1], idx_f[out == 2]
