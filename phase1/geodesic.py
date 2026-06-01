"""M1.8.5 测地距离场 — 绕障导航的核心.

为什么需要: approach 项原来用欧氏距离 |cam - P_weld|, 它无视障碍 —— 当合格观测
区在工件另一面时, 绕行会让欧氏距离变大, approach 反而把机械臂往近侧表面顶, 绕不过去.

测地距离 = 沿自由空间实际要走的路程 (绕开 occupied 格). 把 approach 换成
"候选格到合格观测区(shell∩wedge)的测地距离", 梯度就会自动沿绕路把相机引到另一面.

实现: 在体素网格上做 GPU 波前 BFS (并行 Bellman-Ford):
    - 种子 = shell∩wedge 的可通行格, 距离 0
    - occupied 格挡住; free + unknown 都算可通行 (未知区乐观当通, 边走边修)
    - 26-邻接, 边权 = res·√(dx²+dy²+dz²)
    - 反复松弛直到收敛

主接口:
    build_shell_wedge_goal_mask(...)  -> 目标种子布尔 mask
    geodesic_field(voxel_map, goal_mask) -> 每格测地距离 (米) tensor, 不可达=inf
    sample_field_at(field, voxel_map, pts) -> 查任意 world 点所在格的距离
"""
from __future__ import annotations

import numpy as np
import torch

from phase1.mapping import STATE_OCCUPIED, VoxelMap


# 26-邻接偏移 + 边权 (排除 (0,0,0))
def _neighbor_offsets() -> tuple[list[tuple[int, int, int]], list[float]]:
    offs, wts = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offs.append((dx, dy, dz))
                wts.append(float(np.sqrt(dx * dx + dy * dy + dz * dz)))
    return offs, wts


_OFFSETS, _WEIGHTS = _neighbor_offsets()


def build_shell_wedge_goal_mask(
    voxel_map: VoxelMap,
    p_target: np.ndarray,
    d_min: float,
    d_max: float,
    seam_limits: np.ndarray | None = None,
    tangent: np.ndarray | None = None,
    wedge_margin_deg: float = 0.0,
) -> torch.Tensor:
    """合格观测区的体素种子: 到 p_target 距离 ∈[d_min,d_max] 且 (可选) 在 wedge 内,
    且可通行 (非 occupied).

    返回 (nx,ny,nz) bool tensor.
    """
    dev = voxel_map.device
    nx, ny, nz = voxel_map.shape
    # 所有体素中心 world 坐标
    ix = torch.arange(nx, device=dev)
    iy = torch.arange(ny, device=dev)
    iz = torch.arange(nz, device=dev)
    gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing="ij")
    idx = torch.stack([gx, gy, gz], dim=-1)                      # (nx,ny,nz,3)
    centers = voxel_map.index_to_world(idx)                      # (nx,ny,nz,3)

    p = torch.tensor(np.asarray(p_target, dtype=np.float64).reshape(3),
                     device=dev, dtype=torch.float32)
    rel = centers - p                                            # (...,3)
    d = rel.norm(dim=-1)                                         # (...)
    mask = (d >= d_min) & (d <= d_max)

    if seam_limits is not None:
        sl = np.asarray(seam_limits, dtype=np.float64)
        d1 = sl[0] / (np.linalg.norm(sl[0]) + 1e-12)
        d2 = sl[1] / (np.linalg.norm(sl[1]) + 1e-12)
        t = None
        if tangent is not None:
            t = np.asarray(tangent, dtype=np.float64).reshape(3)
            tn = np.linalg.norm(t)
            t = t / tn if tn > 1e-9 else None
        if t is not None:
            d1 = d1 - np.dot(d1, t) * t
            d2 = d2 - np.dot(d2, t) * t
            d1 /= (np.linalg.norm(d1) + 1e-12)
            d2 /= (np.linalg.norm(d2) + 1e-12)
        bis = d1 + d2
        bn = np.linalg.norm(bis)
        if bn > 1e-9:
            bis = bis / bn
            cos_gap = float(np.clip(np.dot(d1, d2), -1.0, 1.0))
            half = np.degrees(np.arccos(cos_gap)) / 2.0 - wedge_margin_deg
            cos_half = float(np.cos(np.radians(max(half, 0.0))))

            rel_u = rel / (d.unsqueeze(-1) + 1e-9)
            if t is not None:
                t_t = torch.tensor(t, device=dev, dtype=torch.float32)
                rel_u = rel_u - (rel_u * t_t).sum(-1, keepdim=True) * t_t
                rel_u = rel_u / (rel_u.norm(dim=-1, keepdim=True) + 1e-9)
            bis_t = torch.tensor(bis, device=dev, dtype=torch.float32)
            cos_to_bis = (rel_u * bis_t).sum(-1)
            mask = mask & (cos_to_bis >= cos_half)

    traversable = voxel_map.state != STATE_OCCUPIED
    return mask & traversable


def geodesic_field(
    voxel_map: VoxelMap,
    goal_mask: torch.Tensor,
    max_iters: int | None = None,
    tol_iters: int = 2,
) -> torch.Tensor:
    """波前 BFS: 从 goal_mask 出发, 在可通行格上传播测地距离 (米).

    occupied 格 = inf; 不可达 = inf.
    max_iters: 上限 (默认 nx+ny+nz); tol_iters: 连续多少轮无变化即停.
    """
    dev = voxel_map.device
    nx, ny, nz = voxel_map.shape
    INF = float("inf")
    res = voxel_map.res

    traversable = (voxel_map.state != STATE_OCCUPIED)
    dist = torch.full((nx, ny, nz), INF, device=dev, dtype=torch.float32)
    seed = goal_mask & traversable
    dist[seed] = 0.0

    if max_iters is None:
        max_iters = nx + ny + nz

    no_change = 0
    for _ in range(max_iters):
        # 一轮松弛: 用 inf 填充边界, 再按偏移切片对齐邻居
        pad = torch.nn.functional.pad(
            dist, (1, 1, 1, 1, 1, 1), mode="constant", value=INF,
        )                                                         # (nx+2,ny+2,nz+2)
        new = dist
        for (dx, dy, dz), w in zip(_OFFSETS, _WEIGHTS):
            # 邻居值对齐到中心格 c: neighbor at c+off → slice 起点 1+off
            sl = pad[1 + dx:1 + dx + nx,
                     1 + dy:1 + dy + ny,
                     1 + dz:1 + dz + nz]
            new = torch.minimum(new, sl + w * res)
        # 不可通行格强制 inf
        new = torch.where(traversable, new, torch.full_like(new, INF))
        changed = (new != dist) & torch.isfinite(new)
        dist = new
        if not bool(changed.any()):
            no_change += 1
            if no_change >= tol_iters:
                break
        else:
            no_change = 0
    return dist


def sample_field_at(
    field: torch.Tensor,
    voxel_map: VoxelMap,
    pts: np.ndarray,
    oob_value: float = float("inf"),
) -> np.ndarray:
    """查 world 点 pts (N,3) 所在格的测地距离. 越界返回 oob_value."""
    dev = voxel_map.device
    pts_t = torch.tensor(np.asarray(pts, dtype=np.float64).reshape(-1, 3),
                         device=dev, dtype=torch.float32)
    idx = voxel_map.world_to_index(pts_t)                        # (N,3), 越界=-1
    valid = (idx >= 0).all(dim=-1)
    out = torch.full((pts_t.shape[0],), float(oob_value), device=dev,
                     dtype=torch.float32)
    if valid.any():
        vi = idx[valid]
        out[valid] = field[vi[:, 0], vi[:, 1], vi[:, 2]]
    return out.cpu().numpy()


__all__ = [
    "workspace_bounds",
    "build_shell_wedge_goal_mask",
    "geodesic_field",
    "sample_field_at",
]


def workspace_bounds(
    robot_pose_world: np.ndarray,
    arm_reach: float,
    pad: float = 0.8,
) -> np.ndarray:
    """机械臂工作空间立方盒: base 为心, 半边长 = arm_reach + pad.

    测地场活在网格里, 网格必须罩住臂能到达 + 相机会经过的整个空间.

    robot_pose_world: (4,4) base 在 world 的位姿, 或 (3,) base 位置.
    返回 (2,3) [[xmin,ymin,zmin],[xmax,ymax,zmax]].
    """
    p = np.asarray(robot_pose_world, dtype=np.float64)
    base = p[:3, 3] if p.shape == (4, 4) else p.reshape(3)
    half = float(arm_reach) + float(pad)
    return np.stack([base - half, base + half])

