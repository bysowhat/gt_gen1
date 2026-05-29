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
    incidence_ok: bool
    distance: float
    incidence_angle_deg: float
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


def check_incidence(
    viewpoint: Viewpoint, p_seam: np.ndarray, tangent: np.ndarray,
    angle_min_deg: float, angle_max_deg: float,
) -> tuple[bool, float]:
    """④ 入射角: 视线 (cam_pos→P_i) 与焊缝切线的夹角.

    用 |cos(angle)| (因为切线方向有正反两个等价方向).
    返回 angle ∈ [0°, 90°].
    """
    p_seam = np.asarray(p_seam, dtype=np.float64).reshape(3)
    tangent = np.asarray(tangent, dtype=np.float64).reshape(3)
    t_n = tangent / np.linalg.norm(tangent)

    rel = p_seam - viewpoint.pos
    rel_n = rel / np.linalg.norm(rel)
    cos_a = abs(float(np.dot(rel_n, t_n)))
    cos_a = float(np.clip(cos_a, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_a)))
    return angle_min_deg <= angle_deg <= angle_max_deg, angle_deg


# ---------------------------------------------------------------------- main


def is_valid_observation(
    viewpoint: Viewpoint,
    p_seam: np.ndarray,
    tangent: np.ndarray,
    voxel_map: "VoxelMap",
    intrinsics: "CameraIntrinsics",
    *,
    d_min: float = 0.3,
    d_max: float = 0.6,
    incidence_min_deg: float = 30.0,
    incidence_max_deg: float = 90.0,
    los_unknown_as_block: bool = True,
) -> ObservationResult:
    """主函数: 4 条 short-circuit 检查, 任一失败即返回."""
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

    inc_ok, inc_angle = check_incidence(
        viewpoint, p_seam, tangent, incidence_min_deg, incidence_max_deg,
    )
    if not inc_ok:
        return ObservationResult(False, dist_ok, frustum_ok, los_ok, False,
                                 dist, inc_angle, "incidence")

    return ObservationResult(True, True, True, True, True,
                             dist, inc_angle, None)


__all__ = [
    "Viewpoint",
    "ObservationResult",
    "is_valid_observation",
    "check_distance",
    "check_in_frustum",
    "check_line_of_sight",
    "check_incidence",
]
