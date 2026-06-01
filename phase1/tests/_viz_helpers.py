"""三个 visualize_*.py 共用的 helper.

主要功能:
  - load_seam_data(pkl_path)        加载 pkl, 返回 dict
  - make_seam_geoms(seam_data, ...) 焊缝可视化 (粗线 + 端点 + 中点高亮)
  - look_at(cam_pos, target)
  - make_camera_frustum(pose, K)
  - make_sphere(center, radius, color)
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------- seam


@dataclass
class SeamData:
    """从 pkl 抽出的焊缝信息."""
    pkl_path: Path
    seam_line: np.ndarray         # (N, 3)
    seam_tangent: np.ndarray      # (N, 3)
    seam_limits: np.ndarray       # (N, 2, 3) 焊缝处两侧表面法向量 (单位向量, 互相垂直)
    p_weld: np.ndarray            # (3,) 中点
    p_weld_idx: int
    tangent: np.ndarray           # (3,) p_weld 的切线
    seam_length: float            # 焊缝总长 (米)
    robot_pose: np.ndarray        # (4, 4)
    piece_pose: np.ndarray        # (4, 4)
    joint_angles: np.ndarray      # (6,)

    @property
    def label(self) -> str:
        """供 Open3D 窗口标题用. 例: 'seam_0.pkl  (20 pts, len=0.010m)'."""
        return (f"{self.pkl_path.name}  "
                f"({len(self.seam_line)} pts, len={self.seam_length*1000:.0f}mm)")


def _pose7_to_mat4(p7: np.ndarray) -> np.ndarray:
    p = np.asarray(p7).reshape(-1)
    pos = p[:3]
    qw, qx, qy, qz = p[3:7]
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos
    return T


def load_seam_data(pkl_path: str | Path) -> SeamData:
    pkl_path = Path(pkl_path)
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    seam_line = np.asarray(d["seam_line"])
    seam_tangent = np.asarray(d["seam_tangent"])
    seam_limits = np.asarray(d["seam_limits"])
    mid = int(d.get("middle", len(seam_line) // 2))
    seam_length = float(np.linalg.norm(np.diff(seam_line, axis=0), axis=1).sum())
    return SeamData(
        pkl_path=pkl_path,
        seam_line=seam_line,
        seam_tangent=seam_tangent,
        seam_limits=seam_limits,
        p_weld=seam_line[mid],
        p_weld_idx=mid,
        tangent=seam_tangent[mid],
        seam_length=seam_length,
        robot_pose=_pose7_to_mat4(d["robot_pose"][0]),
        piece_pose=_pose7_to_mat4(d["piece_pose"][0]),
        joint_angles=np.asarray(d["joint_angles"], dtype=np.float64),
    )


def make_seam_geoms(
    seam: SeamData,
    show_tangent: bool = True,
    line_color=(1.0, 0.05, 0.55),     # 品红, 工件灰背景上很显眼
    pt_color=(1.0, 0.3, 0.7),
    p_weld_color=(1.0, 0.55, 0.0),
    start_color=(0.0, 1.0, 0.2),       # 起点 亮绿
    end_color=(1.0, 0.05, 0.05),       # 终点 亮红
    endpoint_radius: float = 0.015,    # 端点球半径 (大, 显眼)
    p_weld_radius: float = 0.012,
    midpoint_radius: float = 0.004,
) -> list:
    """画焊缝.

    几何:
        粗线段串起 N 个点
        起点 (idx=0)         亮绿大球
        终点 (idx=N-1)       亮红大球
        P_weld (中点)         橙球
        其他中间点            小粉球
        切线方向              蓝色短线段 (P_weld 处)
    """
    geoms = []
    pts = seam.seam_line.astype(np.float64)
    N = len(pts)
    if N == 0:
        return geoms

    # 1. 把 N 个点用 N-1 条线段串起来
    if N >= 2:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(N - 1)])
        ls.colors = o3d.utility.Vector3dVector([list(line_color)] * (N - 1))
        geoms.append(ls)

    # 2. 起点: 亮绿大球
    sph_start = o3d.geometry.TriangleMesh.create_sphere(radius=endpoint_radius, resolution=14)
    sph_start.translate(pts[0])
    sph_start.paint_uniform_color(list(start_color))
    sph_start.compute_vertex_normals()
    geoms.append(sph_start)

    # 3. 终点: 亮红大球
    if N >= 2:
        sph_end = o3d.geometry.TriangleMesh.create_sphere(radius=endpoint_radius, resolution=14)
        sph_end.translate(pts[-1])
        sph_end.paint_uniform_color(list(end_color))
        sph_end.compute_vertex_normals()
        geoms.append(sph_end)

    # 4. 其他中间点 (除起点、终点、中点之外)
    for i, p in enumerate(pts):
        if i in (0, N - 1, seam.p_weld_idx):
            continue
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=midpoint_radius, resolution=8)
        sph.translate(p)
        sph.paint_uniform_color(list(pt_color))
        sph.compute_vertex_normals()
        geoms.append(sph)

    # 5. P_weld 中点 (橙球, 中等大)
    if 0 < seam.p_weld_idx < N - 1:
        pw = o3d.geometry.TriangleMesh.create_sphere(radius=p_weld_radius, resolution=12)
        pw.translate(seam.p_weld)
        pw.paint_uniform_color(list(p_weld_color))
        pw.compute_vertex_normals()
        geoms.append(pw)

    # 6. 切线方向 (蓝色短线段, 长度跟焊缝长度成比例, 至少 5cm 让人看得见)
    if show_tangent:
        t_len = max(seam.seam_length * 1.0, 0.05)
        ls_t = o3d.geometry.LineSet()
        ls_t.points = o3d.utility.Vector3dVector(np.vstack([
            seam.p_weld - seam.tangent * t_len,
            seam.p_weld + seam.tangent * t_len,
        ]))
        ls_t.lines = o3d.utility.Vector2iVector([[0, 1]])
        ls_t.colors = o3d.utility.Vector3dVector([[0.1, 0.4, 1.0]])
        geoms.append(ls_t)

    return geoms


# ---------------------------------------------------------------------- camera


def look_at(
    cam_pos: np.ndarray, target: np.ndarray,
    up_hint: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """OpenCV 系: x 右, y 下, z 前. 返回 (4,4) world→camera 位姿."""
    z = target - cam_pos
    z = z / np.linalg.norm(z)
    if abs(np.dot(z, up_hint)) > 0.99:
        up_hint = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
    x = np.cross(up_hint, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cam_pos
    return T


def make_camera_frustum(
    camera_pose: np.ndarray, intrinsics, far: float = 0.3,
    color=(0.1, 0.4, 1.0),
) -> o3d.geometry.LineSet:
    corners_uv = np.array([
        [0, 0], [intrinsics.width - 1, 0],
        [intrinsics.width - 1, intrinsics.height - 1], [0, intrinsics.height - 1],
    ])
    dirs_cam = np.stack([
        (corners_uv[:, 0] - intrinsics.cx) / intrinsics.fx,
        (corners_uv[:, 1] - intrinsics.cy) / intrinsics.fy,
        np.ones(4),
    ], axis=1)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    far_corners = t + far * (dirs_cam @ R.T)
    points = np.vstack([t.reshape(1, 3), far_corners])
    lines = [[0, 1], [0, 2], [0, 3], [0, 4],
             [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([list(color)] * len(lines))
    return ls


def make_sphere(center, radius, color=(1.0, 0.2, 0.2)) -> o3d.geometry.TriangleMesh:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
    s.translate(np.asarray(center).reshape(3))
    s.paint_uniform_color(list(color))
    s.compute_vertex_normals()
    return s


def make_line(p_a, p_b, color=(1.0, 0.9, 0.0)) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.vstack([np.asarray(p_a), np.asarray(p_b)]))
    ls.lines = o3d.utility.Vector2iVector([[0, 1]])
    ls.colors = o3d.utility.Vector3dVector([list(color)])
    return ls
