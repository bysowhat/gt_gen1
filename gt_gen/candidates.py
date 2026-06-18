"""Step 8: 候选视点生成。

见 docs/privileged-nbv.md §4.5 ③(1)(2)(3)。

一次 NBV 迭代里，③把"看清阻塞段 B"这件事，落成一批【既看得见 B、又走得到】的候选关节构型：

  cluster_centroids(B)          —— 对 B 空间聚类，取 1~3 个簇心当"要看的目标点 T"
  standoff_poses_looking_at(T)  —— 每个 T 按 (几档站位距离 d × 半球若干方向) 撒相机位姿，
                                   过滤：相机眼睛 p 必须站在已确认 FREE、视线 p→T 不被 OCCUPIED 挡
  generate_candidates(...)      —— 上面两步 + 眼在手 IK + 看向校验 + 保守可达性过滤

【眼在手关键】相机装在 Link6 上，而 cuRobo IK 解的是 ee_link(xiaoyu_tip_link)。
要让【相机】到达某位姿 T_base_cam，需先换算成对应的【ee_link】位姿再 IK：
  T_base_cam = T_base_Link6 @ T_l6_cam            (T_l6_cam = 手眼外参，已知)
  T_base_ee  = T_base_cam   @ T_cam_ee            (T_cam_ee 为 cam→ee 的固定刚体变换)
T_cam_ee 与构型无关，用任一构型 FK 一次性求出（ee 位姿 + Link6 位姿反解）。

候选最终要：①IK 有解(关节限位+自碰撞，由 handle.ik 保证)；②相机光轴确实对准 T
（cuRobo 朝向阈值较松，这里再用真实 FK 光轴夹角兜一道）；③从当前构型【经自由区】走得到
（motion_stays_in_free：直线插值整臂扫掠 ⊆ FREE，保守——走不到的丢弃）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Candidate:
    """一个候选视点。config=关节构型；cam_pose=该构型下相机在 base 的 4x4 实际位姿；
    target=该候选要看的目标点 T(base)；ik_pos_err=IK 位置误差(m)；look_err_deg=光轴对准 T 的夹角(°)。"""
    config: list
    cam_pose: np.ndarray
    target: np.ndarray
    ik_pos_err: float
    look_err_deg: float


# ---------- 小工具：旋转/位姿 ----------

def _normalize(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _R_to_quat_wxyz(R) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rsp
    q = Rsp.from_matrix(np.asarray(R, float)).as_quat()      # xyzw
    return np.array([q[3], q[0], q[1], q[2]], float)         # -> wxyz


def _rot_z_to(axis) -> np.ndarray:
    """返回把 +Z 轴转到 axis 方向的旋转矩阵（用于把半球采样模板对齐到任意轴）。"""
    from scipy.spatial.transform import Rotation as Rsp
    z = np.array([0.0, 0.0, 1.0]); a = _normalize(axis)
    c = float(np.dot(z, a))
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        return Rsp.from_rotvec([np.pi, 0.0, 0.0]).as_matrix()
    v = np.cross(z, a); s = float(np.linalg.norm(v))
    return Rsp.from_rotvec(v / s * np.arccos(np.clip(c, -1, 1))).as_matrix()


def _hemisphere_dirs(axis, n: int) -> np.ndarray:
    """绕 axis 的半球内 n 个近似均匀方向（Fibonacci 半球；z∈(0,1] 那半，再对齐到 axis）。"""
    if n <= 0:
        return np.empty((0, 3))
    ga = np.pi * (3.0 - np.sqrt(5.0))                        # 黄金角
    out = []
    for i in range(n):
        z = (i + 0.5) / n                                    # (0,1] 上半球
        r = np.sqrt(max(0.0, 1.0 - z * z))
        phi = i * ga
        out.append([r * np.cos(phi), r * np.sin(phi), z])
    return np.asarray(out, float) @ _rot_z_to(axis).T


def look_at_pose(p, T, world_up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """相机眼睛在 p、光轴对准 T 的 4x4 位姿（OpenCV 光学帧：+Z 视线、+X 右、+Y 下）。"""
    p = np.asarray(p, float); T = np.asarray(T, float)
    f = _normalize(T - p)                                    # +Z 视线
    up = np.asarray(world_up, float)
    if abs(float(np.dot(f, _normalize(up)))) > 0.95:         # 光轴近平行世界上方 → 换参考
        up = np.array([1.0, 0.0, 0.0])
    x = _normalize(np.cross(f, up))                          # +X 右 = 视线 × 世界上方
    y = _normalize(np.cross(f, x))                           # +Y 下 (x×y=z, 右手系)
    M = np.eye(4)
    M[:3, 0] = x; M[:3, 1] = y; M[:3, 2] = f; M[:3, 3] = p
    return M


# ---------- ③(1) 对 B 聚类 ----------

def cluster_centroids(B, max_clusters: int = 3, link_dist: float = 0.15,
                      method: str = "single") -> np.ndarray:
    """对阻塞集 B（(M,3) base 系世界点）空间聚类，取簇心当"要看的目标点" T。

    用 scipy 层次聚类按距离阈值 link_dist 自然分片；若片数 > max_clusters，保留最大的
    max_clusters 个簇心，再把【全部点】重新指派到最近簇心做一次 Lloyd 更新（保证所有点都被代表）。
    返回 (k,3) 簇心，k ∈ [1, max_clusters]（B 为空时返回 (0,3)）。

    参数（均可由 configs/default.yaml 的 params.nbv.cluster 配置）：
      max_clusters : 簇数上限。
      link_dist    : fcluster criterion='distance' 的距离阈值 t（簇内最近邻间距 ≤ 此值归一簇）。
      method       : linkage 连接法（single 链式，适合连通体素块；也可 complete/average/ward）。
    （criterion 固定 'distance'、metric 固定欧氏——本函数就是按"欧氏距离阈值"做空间分片。）
    """
    pts = np.asarray(B, float).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.empty((0, 3))
    if pts.shape[0] <= 1:
        return pts.copy()

    from scipy.cluster.hierarchy import linkage, fcluster
    labels = fcluster(linkage(pts, method=method), t=link_dist, criterion="distance")
    groups = [pts[labels == l] for l in np.unique(labels)]
    groups.sort(key=lambda g: g.shape[0], reverse=True)
    cents = np.array([g.mean(axis=0) for g in groups])

    if cents.shape[0] <= max_clusters:
        return cents
    # 片数过多：保留最大的 max_clusters 个簇心，全部点重指派到最近簇心后重算（覆盖所有 B）
    cents = cents[:max_clusters]
    d = np.linalg.norm(pts[:, None, :] - cents[None, :, :], axis=2)   # (M,k)
    assign = d.argmin(axis=1)
    return np.array([pts[assign == j].mean(axis=0) if (assign == j).any() else cents[j]
                     for j in range(max_clusters)])


# ---------- ③(2) 每个 T 撒相机位姿 ----------

def _sight_clear(voxmap, p, T) -> bool:
    """视线 p→T 是否未被【已知障碍 OCCUPIED】挡（UNKNOWN 允许——那正是要去看的）。
    沿线按半体素步采样，止于 T 前一格（T 本身在 B/未知区，不算被挡）。"""
    from gt_gen.voxmap import OCCUPIED
    p = np.asarray(p, float); T = np.asarray(T, float)
    L = float(np.linalg.norm(T - p))
    if L < 1e-6:
        return True
    step = 0.5 * voxmap.voxel_size
    ts = np.arange(step, max(L - voxmap.voxel_size, step), step)
    if ts.size == 0:
        return True
    samples = p[None] + np.outer(ts, (T - p) / L)
    return not bool((np.asarray(voxmap.get(voxmap.world_to_voxel(samples))) == OCCUPIED).any())


def flange_origin(handle, q) -> np.ndarray:
    """xiaoyu_flange_link 原点在 base 系的位置（半球轴锚点：相机/腕部当前所在侧）。

    需 init_curobo 把 'xiaoyu_flange_link' 加进 link_names（已加）。
    """
    import torch
    st = handle.mg.kinematics.get_state(
        torch.tensor([list(q)], dtype=torch.float32, device="cuda"))
    return st.link_pose["xiaoyu_flange_link"].position[0].detach().cpu().numpy()


def standoff_poses_looking_at(T, voxmap, camera_model, anchor=None,
                              standoff_d=(0.25, 0.35, 0.45),
                              n_view_dirs: int = 8) -> List[np.ndarray]:
    """对目标点 T，按 (几档站位距离 d × 半球若干方向) 生成朝向 T、站位在 FREE 区的相机位姿(4x4)。

    半球轴取 T→anchor 方向（anchor 默认 base 原点；实际取当前构型下 xiaoyu_flange_link 原点，
    使半球面朝机械臂腕部/相机当前所在一侧——眼在手时可达视点都在这一侧）。过滤：
      ① 相机眼睛 p 落在已确认 FREE 体素；② 视线 p→T 不被 OCCUPIED 挡；
      ③ 站距 d ≤ 相机量程 max_depth（拍得清）。光轴对准 T，滚转用世界上方定。
    """
    from gt_gen.voxmap import FREE
    T = np.asarray(T, float)
    a = np.zeros(3) if anchor is None else np.asarray(anchor, float)
    axis = _normalize(a - T) if float(np.linalg.norm(a - T)) > 1e-6 else np.array([0.0, 0.0, 1.0])
    dirs = _hemisphere_dirs(axis, n_view_dirs)
    max_depth = float(camera_model.get("max_depth", 3.0))
    poses = []
    for d in standoff_d:
        if d > max_depth:
            continue
        for u in dirs:
            p = T + d * u
            if int(voxmap.get(voxmap.world_to_voxel(p))) != FREE:      # 必须站在确认自由区
                continue
            if not _sight_clear(voxmap, p, T):                          # 视线别被已知障碍挡
                continue
            poses.append(look_at_pose(p, T))
    return poses


# ---------- 眼在手：相机位姿 → ee_link 位姿 ----------

def _cam_to_ee_transform(handle, camera_model) -> np.ndarray:
    """求 cam→ee 的固定刚体变换 T_cam_ee（与构型无关）：T_base_ee = T_base_cam @ T_cam_ee。"""
    from gt_gen.sensor import pose_to_T
    from gt_gen import curobo_iface as ci
    q0 = list(handle.config.retract_config)
    ee_pos, ee_quat, lp = ci.fk(handle, q0)
    T_base_ee = pose_to_T(ee_pos, ee_quat)
    l6 = lp["Link6"]
    T_base_l6 = pose_to_T(l6.position[0].detach().cpu().numpy(),
                          l6.quaternion[0].detach().cpu().numpy())
    T_l6_cam = pose_to_T(camera_model["extrinsic_pos"], camera_model["extrinsic_quat_wxyz"])
    T_base_cam = T_base_l6 @ T_l6_cam
    return np.linalg.inv(T_base_cam) @ T_base_ee


def _debug_viz_candidate(handle, voxmap, cur_cfg, cfg, B):
    """调试用：可视化 cur_cfg / 候选 cfg 两条整臂 + voxmap(FREE/OCCUPIED) + 阻塞集 B。
    被调用即弹窗(需显示器+open3d)；无显示器/服务器跑时把调用行注释掉即可（调用行本身就是开关）。

    画 cur_cfg 整臂(绿) + cfg 整臂(青) + FREE 格(蓝半透明) + OCCUPIED 格(橙) + B(品红) +
    ROI/base。UNKNOWN 几乎是整个网格、画出来挡视线，故不画。依赖 verify_step8 的 open3d 工具。
    """
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from verify_step8 import _arm_mesh, _cells_mesh, _draw, _roi_and_base
    from gt_gen.voxmap import FREE, OCCUPIED

    a_cur = _arm_mesh(handle, list(cur_cfg)); a_cur.paint_uniform_color([0.20, 0.72, 0.32])
    a_cfg = _arm_mesh(handle, list(cfg)); a_cfg.paint_uniform_color([0.10, 0.75, 0.85])
    geoms = [("cur_cfg", a_cur, "lit", None), ("cfg", a_cfg, "lit", None)]

    free_c = voxmap.state_centers(FREE)
    occ_c = voxmap.state_centers(OCCUPIED)
    n_free = int(free_c.shape[0]); n_occ = int(occ_c.shape[0])
    if n_free:
        geoms.append(("free", _cells_mesh(voxmap, free_c), "fill", [0.20, 0.45, 0.95, 0.20]))
    if n_occ:
        om = _cells_mesh(voxmap, occ_c); om.paint_uniform_color([1.0, 0.55, 0.0])
        geoms.append(("occupied", om, "lit", None))

    Bw = np.asarray(B).reshape(-1, 3)
    n_b = int(Bw.shape[0])
    if n_b:
        bm = _cells_mesh(voxmap, voxmap.voxel_to_world(Bw)); bm.paint_uniform_color([1.0, 0.0, 0.85])
        geoms.append(("B", bm, "lit", None))

    geoms += _roi_and_base(voxmap)
    _draw(geoms, f"candidate: 绿=cur_cfg 青=候选cfg | FREE={n_free}(蓝) "
                 f"OCCUPIED={n_occ}(橙) B={n_b}(品红)")


# ---------- ③ 顶层：生成候选关节构型 ----------

def generate_candidates(handle, voxmap, B, camera_model, cur_cfg,
                        standoff_d=None, n_view_dirs: Optional[int] = None,
                        max_clusters: Optional[int] = None, ik_per_pose: int = 2,
                        max_look_deg: Optional[float] = None) -> List[Candidate]:
    """聚类 → standoff 位姿 → 眼在手 IK → 看向校验 + 保守可达性过滤，返回候选构型列表。

    参数：
      handle      : CuroboHandle（IK 用其 handle.ik，与世界无关；只需运动学）。
      voxmap      : 当前三态图（FREE 判站位/视线/可达）。
      B           : 阻塞集体素下标 (M,3)（compute_blocking_B 的输出）。
      camera_model: load_camera_model(cfg) 的 dict。
      cur_cfg     : 当前关节构型（可达性的起点）。
      standoff_d / n_view_dirs / max_clusters / max_look_deg：None 时从 cfg.params.nbv 取
                    （standoff_d_m / n_view_dirs / cluster.* / max_look_deg，default.yaml）。
      ik_per_pose : 每个相机位姿尝试的 IK 分支数（不同腕姿，提高可达命中）。
    返回：通过全部过滤的 Candidate 列表（可空——无可达视点时由主循环转"就近揭示"兜底）。
    """
    from gt_gen import curobo_iface as ci
    from gt_gen.sensor import camera_pose_from_config
    from gt_gen.swept import motion_stays_in_free

    nbv = handle.config.params.get("nbv", {})
    if standoff_d is None:
        standoff_d = tuple(nbv.get("standoff_d_m", (0.25, 0.35, 0.45)))
    if n_view_dirs is None:
        n_view_dirs = int(nbv.get("n_view_dirs", 8))
    if max_look_deg is None:
        max_look_deg = float(nbv.get("max_look_deg", 20.0))
    cl = nbv.get("cluster", {})
    if max_clusters is None:
        max_clusters = int(cl.get("max", 3))

    B = np.asarray(B).reshape(-1, 3)
    if B.shape[0] == 0:
        return []
    Bw = voxmap.voxel_to_world(B)                              # 体素下标 → base 世界点
    targets = cluster_centroids(Bw, max_clusters=max_clusters,
                                link_dist=float(cl.get("link_dist_m", 0.15)),
                                method=str(cl.get("method", "single")))

    T_cam_ee = _cam_to_ee_transform(handle, camera_model)
    anchor = flange_origin(handle, cur_cfg)                  # 半球轴锚点：当前 flange 原点
    out: List[Candidate] = []
    for T in targets:
        for cam_pose in standoff_poses_looking_at(T, voxmap, camera_model, anchor=anchor,
                                                  standoff_d=standoff_d, n_view_dirs=n_view_dirs):
            T_base_ee = cam_pose @ T_cam_ee                   # 眼在手：相机位姿 → ee_link 位姿
            ee_pos = T_base_ee[:3, 3]
            ee_quat = _R_to_quat_wxyz(T_base_ee[:3, :3])
            ik = ci.solve_ik(handle, (ee_pos, ee_quat), return_seeds=handle.config.ik_num_seeds)
            cfgs = ci.ik_configs(handle, ik)[:ik_per_pose]
            for cfg, perr in cfgs:
                # ② 看向校验：实际相机光轴 vs (T - 相机位置) 的夹角
                Tc = camera_pose_from_config(handle, cfg, camera_model)
                axis = Tc[:3, 2]
                look = _normalize(T - Tc[:3, 3])
                look_deg = float(np.degrees(np.arccos(np.clip(np.dot(axis, look), -1, 1))))
                if look_deg > max_look_deg:
                    continue
                # _debug_viz_candidate(handle, voxmap, cur_cfg, cfg, B)  # 看 cur_cfg/候选整臂 + voxmap(FREE/OCC) + B（去注释开窗）
                # ③ 保守可达：当前构型 → 候选，整臂扫掠 ⊆ FREE
                ok, _ = motion_stays_in_free(handle, voxmap, cur_cfg, cfg)
                if not ok:
                    continue
                out.append(Candidate(config=list(cfg), cam_pose=Tc, target=np.asarray(T, float),
                                     ik_pos_err=float(perr), look_err_deg=look_deg))
    return out
