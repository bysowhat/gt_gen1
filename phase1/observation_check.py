"""M1.3 合格观测检查器 — 4 条硬约束.

判断: 一个 viewpoint v 能否"合格观测"焊缝点 P_i.

4 条全部满足才返回 valid=True:
    ① 距离        d_min ≤ ‖cam_pos - P_i‖ ≤ d_max
    ② 视锥        P_i 投影到图像 (u,v) 在 [0,W) × [0,H), 且 z_cam > 0
    ③ 视线无遮挡  raycast(cam_pos → P_i) 不被 occupied 体素挡, 且 (默认) 不穿 unknown
    ④ 入射角      view_direction 与焊缝切线的夹角 ∈ [angle_min, angle_max]
                  避免视线和焊缝平行 (会看不清焊缝)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from phase1.depth_source import CameraIntrinsics
    from phase1.mapping import VoxelMap


@dataclass
class Viewpoint:
    """相机位姿.

    Attrs:
        pos:  (3,) 相机位置 (world).
        dir:  (3,) 单位向量, 相机光轴方向 = z_camera 在 world 的方向.
        up:   (3,) 可选, 相机 "向上" 参考方向 (用于决定 R 的 yaw); 默认 world +z.
    """
    pos: np.ndarray
    dir: np.ndarray
    up: Optional[np.ndarray] = None

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=np.float64).reshape(3)
        d = np.asarray(self.dir, dtype=np.float64).reshape(3)
        n = np.linalg.norm(d)
        if n < 1e-9:
            raise ValueError("Viewpoint.dir must be non-zero")
        self.dir = d / n
        if self.up is not None:
            self.up = np.asarray(self.up, dtype=np.float64).reshape(3)


@dataclass
class ObservationResult:
    """每条约束的具体结果, 调试用. 主流程只看 .valid."""
    valid: bool
    distance_ok: bool
    in_frustum_ok: bool
    line_of_sight_ok: bool
    wedge_ok: bool
    distance: float
    wedge_min_dot: float        # cos_to_bis - cos_half. >0 = 在 wedge 内
    fail_reason: Optional[str]


# ---------------------------------------------------------------------- helpers


def _camera_basis(cam_dir: np.ndarray, up_hint: Optional[np.ndarray] = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据 cam_dir 和 up_hint 构造相机 (x_cam, y_cam, z_cam) 在 world 中的基向量.

    OpenCV 系: x 右, y 下, z 前. 公式:
        z = cam_dir
        x = (up_hint × z) / ||...||
        y = z × x  (向下)
    """
    z = cam_dir / np.linalg.norm(cam_dir)
    if up_hint is None:
        up_hint = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(z, up_hint))) > 0.999:
        # 与 z 平行, 换 up
        up_hint = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(z, up_hint))) > 0.999:
            up_hint = np.array([1.0, 0.0, 0.0])
    x = np.cross(up_hint, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z


# ---------------------------------------------------------------------- 4 条约束


def check_distance(
    viewpoint: Viewpoint, p_seam: np.ndarray,
    d_min: float, d_max: float,
) -> tuple[bool, float]:
    """① 距离."""
    p_seam = np.asarray(p_seam, dtype=np.float64).reshape(3)
    d = float(np.linalg.norm(viewpoint.pos - p_seam))
    return d_min <= d <= d_max, d


def check_in_frustum(
    viewpoint: Viewpoint, p_seam: np.ndarray,
    intrinsics: "CameraIntrinsics",
) -> bool:
    """② 视锥内: P_i 投影到 (u, v) 必须落在 [0, W) × [0, H), 且 z_cam > 0."""
    p_seam = np.asarray(p_seam, dtype=np.float64).reshape(3)
    rel_world = p_seam - viewpoint.pos
    x_cam_w, y_cam_w, z_cam_w = _camera_basis(viewpoint.dir, viewpoint.up)
    # world → cam: rel_cam[i] = rel_world · cam_basis[i]
    z_cam = float(np.dot(rel_world, z_cam_w))
    if z_cam <= 0:
        return False
    x_cam = float(np.dot(rel_world, x_cam_w))
    y_cam = float(np.dot(rel_world, y_cam_w))
    u = intrinsics.fx * x_cam / z_cam + intrinsics.cx
    v = intrinsics.fy * y_cam / z_cam + intrinsics.cy
    return (0.0 <= u < intrinsics.width) and (0.0 <= v < intrinsics.height)


def check_line_of_sight(
    viewpoint: Viewpoint, p_seam: np.ndarray,
    voxel_map: "VoxelMap",
    unknown_as_block: bool = True,
) -> bool:
    """③ 视线无遮挡.

    沿 cam_pos → P_i 方向 raycast, 走到 P_i 距离之前是否撞 occupied / 穿 unknown.
    """
    p_seam = np.asarray(p_seam, dtype=np.float64).reshape(3)
    rel = p_seam - viewpoint.pos
    distance = float(np.linalg.norm(rel))
    if distance < 1e-6:
        return True   # 重合视为通过
    direction = rel / distance

    origins = torch.tensor(viewpoint.pos.reshape(1, 3),
                           dtype=torch.float32, device=voxel_map.device)
    dirs = torch.tensor(direction.reshape(1, 3),
                        dtype=torch.float32, device=voxel_map.device)

    # P_i 在工件表面附近, 它自己的体素(以及紧邻体素)往往是 occupied.
    # 把 raycast 终点提前 margin, 跳过 P_i 周围的"自身"体素.
    margin = voxel_map.res * 2.0
    effective_max = distance - margin
    if effective_max <= voxel_map.res:
        # 距离太近, 没有 raycast 空间; 视为通过 (距离检查应已挡住)
        return True
    out = voxel_map.raycast(
        origins, dirs,
        max_range=effective_max,
    )
    if bool(out["hit"][0].item()):
        # raycast 在 P_i 之前击中了 occupied
        return False
    if unknown_as_block and int(out["unknown_count"][0].item()) > 0:
        return False
    return True


def wedge_bisector(
    seam_limits: np.ndarray,
    tangent: np.ndarray | None = None,
) -> np.ndarray | None:
    """返回 wedge 中线 (角平分线) 单位方向, 从 seam 指向开口外.

    与 check_seam_wedge 里的 bisector 算法一致, 抽出来给评分复用 (M1.8):
    让最优观测候选的 (cam_pos - p_seam) 方向尽量对准这条中线 → 最"正"地看焊缝.

    `seam_limits.shape = (2, 3)` (boundary_dirs). tangent 给了就投影到 ⊥tangent 平面.
    退化 (limits 不合法 / 反向) 返回 None.
    """
    sl = np.asarray(seam_limits, dtype=np.float64)
    if sl.shape != (2, 3):
        return None
    n1, n2 = np.linalg.norm(sl[0]), np.linalg.norm(sl[1])
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    d1, d2 = sl[0] / n1, sl[1] / n2

    if tangent is not None:
        t = np.asarray(tangent, dtype=np.float64).reshape(3)
        tn = np.linalg.norm(t)
        if tn > 1e-9:
            t = t / tn
            d1 = d1 - np.dot(d1, t) * t
            d2 = d2 - np.dot(d2, t) * t
            m1, m2 = np.linalg.norm(d1), np.linalg.norm(d2)
            if m1 < 1e-9 or m2 < 1e-9:
                return None
            d1 /= m1; d2 /= m2

    bis = d1 + d2
    bn = np.linalg.norm(bis)
    if bn < 1e-9:
        return None                      # d1, d2 反向 (gap=180°), 无明确中线
    return bis / bn


def check_seam_wedge(
    viewpoint: Viewpoint, p_seam: np.ndarray, seam_limits: np.ndarray,
    tangent: np.ndarray | None = None,
    margin_deg: float = 0.0,
) -> tuple[bool, float]:
    """⑤ 凹角 wedge: 相机必须在焊缝两侧"面方向"定义的开口 wedge 内.

    `seam_limits.shape = (2, 3)`: 焊缝两侧的"面方向" (boundary_dirs, 单位向量).
        d1, d2 都从 seam 指向开口外 (即 wedge 内). 它们之间的夹角就是 gap_deg
        (开口角度): 90° 表示直角焊缝 (L 形), <90° 表示锐角凹槽, >90° 表示钝角.

    判定: rel = cam_pos - P_seam 投影到 ⊥ tangent 平面, 必须落在
        d1 和 d2 张成的 wedge 角内 (即与 bisector 的夹角 ≤ gap_deg/2).

    Args:
        tangent: 焊缝切线 (1D方向). 若 None, 不做投影 (假设 limits 已在 ⊥ tangent 平面).
        margin_deg: 收紧 wedge 边界的角度余量 (度); 默认 0.
                    >0: cam 必须严格在 wedge 内一定角度; <0: 允许稍微越界.

    Returns:
        (ok, cos_angle_to_bisector_minus_cos_half).
        正值 = 在 wedge 内, 越大越居中.
    """
    p_seam = np.asarray(p_seam, dtype=np.float64).reshape(3)
    sl = np.asarray(seam_limits, dtype=np.float64)
    if sl.shape != (2, 3):
        raise ValueError(f"seam_limits must be (2, 3), got {sl.shape}")

    d1 = sl[0] / np.linalg.norm(sl[0])
    d2 = sl[1] / np.linalg.norm(sl[1])

    # 如果给了切线, 投影到 ⊥ tangent 平面 (理论上 limits 应该已经在这个平面里)
    if tangent is not None:
        t = np.asarray(tangent, dtype=np.float64).reshape(3)
        t = t / np.linalg.norm(t)
        d1 = d1 - np.dot(d1, t) * t
        d2 = d2 - np.dot(d2, t) * t
        n1 = np.linalg.norm(d1); n2 = np.linalg.norm(d2)
        if n1 < 1e-9 or n2 < 1e-9:
            return False, -1.0
        d1 /= n1; d2 /= n2

    rel = viewpoint.pos - p_seam
    if tangent is not None:
        rel = rel - np.dot(rel, t) * t
    rel_norm = np.linalg.norm(rel)
    if rel_norm < 1e-9:
        return False, -1.0
    rel /= rel_norm

    # bisector = (d1 + d2) / norm; 方向是 wedge 中线
    bis = d1 + d2
    bn = np.linalg.norm(bis)
    if bn < 1e-9:
        # d1, d2 反向 (gap_deg = 180°), wedge 实际上是半空间; 用法向定向
        # 退化情况: 不太可能出现, 用 d1 当 bisector 凑合
        bis = d1
    else:
        bis /= bn

    # gap_deg = 两 dir 夹角. half_angle = gap_deg / 2.
    cos_gap = float(np.clip(np.dot(d1, d2), -1.0, 1.0))
    gap_deg = float(np.degrees(np.arccos(cos_gap)))
    half_angle_deg = gap_deg / 2.0 - margin_deg
    cos_half = float(np.cos(np.radians(half_angle_deg)))

    cos_to_bis = float(np.dot(rel, bis))
    in_wedge = cos_to_bis >= cos_half
    return in_wedge, cos_to_bis - cos_half


# ---------------------------------------------------------------------- main


def is_valid_observation(
    viewpoint: Viewpoint,
    p_seam: np.ndarray,
    tangent: np.ndarray,
    voxel_map: "VoxelMap",
    intrinsics: "CameraIntrinsics",
    *,
    seam_limits: Optional[np.ndarray] = None,
    d_min: float = 0.3,
    d_max: float = 0.6,
    los_unknown_as_block: bool = True,
) -> ObservationResult:
    """主函数: 4 条 short-circuit 检查, 任一失败即返回.

    4 条:
      ①  距离        d_min ≤ ‖cam_pos - P_seam‖ ≤ d_max
      ②  视锥        P_seam 投影到 (u,v) 在 [0,W) × [0,H) 且 z_cam>0
      ③  视线        raycast(cam_pos → P_seam) 不撞 occupied (可选不穿 unknown)
      ④  wedge       (可选) cam_pos 在 seam_limits 定义的开口 wedge 内.
                     需要传 seam_limits=(2,3) + tangent; 不传 seam_limits 则跳过.

    `tangent` 仍是必填参数(用作 wedge 投影到 ⊥ tangent 平面).
    """
    dist_ok, dist = check_distance(viewpoint, p_seam, d_min, d_max)
    if not dist_ok:
        return ObservationResult(False, dist_ok, False, False, False,
                                 dist, 0.0, "distance")

    frustum_ok = check_in_frustum(viewpoint, p_seam, intrinsics)
    if not frustum_ok:
        return ObservationResult(False, dist_ok, False, False, False,
                                 dist, 0.0, "frustum")

    los_ok = check_line_of_sight(viewpoint, p_seam, voxel_map, los_unknown_as_block)
    if not los_ok:
        return ObservationResult(False, dist_ok, frustum_ok, False, False,
                                 dist, 0.0, "line_of_sight")

    if seam_limits is not None:
        wedge_ok, min_dot = check_seam_wedge(
            viewpoint, p_seam, seam_limits, tangent=tangent,
        )
        if not wedge_ok:
            return ObservationResult(False, dist_ok, frustum_ok, los_ok, False,
                                     dist, min_dot, "wedge")
    else:
        wedge_ok, min_dot = True, 0.0

    return ObservationResult(True, True, True, True, True,
                             dist, min_dot, None)


__all__ = [
    "wedge_bisector",
    "Viewpoint",
    "ObservationResult",
    "is_valid_observation",
    "check_distance",
    "check_in_frustum",
    "check_line_of_sight",
    "check_seam_wedge",
]
