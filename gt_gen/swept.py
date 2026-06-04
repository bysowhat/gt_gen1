"""Step 6: 整臂扫掠体积 + 是否 ⊆ FREE。

见 privileged-nbv.md §4.5 ②(2)。决策 A = whole_arm：整条臂的碰撞球扫掠体积。

做法：把 q_from→q_to 在关节空间线性插值成若干子步（细到每个碰撞球每步移动 ≤ 分辨率，
不漏体素），逐步 FK 整臂碰撞球，把每个球覆盖的体素并起来 = 扫掠体积。
保守判定：扫掠体积内只要有一个体素非 FREE（OCCUPIED 或 UNKNOWN）→ 该段运动不通过。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def fk_spheres_batch(handle, qs):
    """批量 FK：qs (K,dof) -> 碰撞球 (K,S,4) xyz+r（base 系）。"""
    import torch
    qt = torch.as_tensor(np.asarray(qs, dtype=np.float32), device=handle.ta.device)
    st = handle.mg.kinematics.get_state(qt)
    return st.link_spheres_tensor.detach().cpu().numpy()


_fk_spheres_batch = fk_spheres_batch   # 兼容内部旧名


def voxelize_spheres(voxmap, spheres) -> np.ndarray:
    """把一组碰撞球 (M,4 xyz+r) 保守体素化到 voxmap：返回覆盖的体素下标 (M,3)（去重、在界内）。

    每球覆盖其 AABB 与球相交的所有体素（中心到球心 ≤ r + 半个体素对角线）。
    """
    spheres = np.asarray(spheres, float)
    spheres = spheres[spheres[:, 3] > 1e-4]
    vs = voxmap.voxel_size
    pad = 0.5 * vs * np.sqrt(3.0)
    occ = set()
    for c0, c1, c2, r in spheres:
        c = np.array([c0, c1, c2]); R = r + pad
        lo = voxmap.world_to_voxel(c - R); hi = voxmap.world_to_voxel(c + R)
        rs = [np.arange(lo[k], hi[k] + 1) for k in range(3)]
        ii, jj, kk = np.meshgrid(*rs, indexing="ij")
        idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], 1)
        ctr = voxmap.voxel_to_world(idx)
        for t in map(tuple, idx[np.linalg.norm(ctr - c, axis=1) <= R]):
            occ.add(t)
    if not occ:
        return np.empty((0, 3), dtype=np.int64)
    idx = np.array(sorted(occ), dtype=np.int64)
    return idx[voxmap.in_bounds(idx)]


def _interp_count(handle, q_from, q_to, resolution):
    """按"任一碰撞球单步位移 ≤ resolution"定插值子步数（>=2）。"""
    sph = fk_spheres_batch(handle, [q_from, q_to])              # (2,S,4)
    move = np.linalg.norm(sph[1, :, :3] - sph[0, :, :3], axis=1) # (S,)
    max_move = float(move.max()) if move.size else 0.0
    return max(2, int(np.ceil(max_move / max(resolution, 1e-4))) + 1)


def swept_volume(handle, voxmap, q_from, q_to, resolution: Optional[float] = None) -> np.ndarray:
    """一段运动 q_from→q_to 的整臂扫掠体素下标集合（去重），形状 (M,3)。

    碰撞球近似 + 细插值；每球保守覆盖与之相交的体素。仅返回落在 voxmap 范围内的下标。
    """
    res = voxmap.voxel_size if resolution is None else float(resolution)
    K = _interp_count(handle, q_from, q_to, res)
    q_from = np.asarray(q_from, float); q_to = np.asarray(q_to, float)
    ts = np.linspace(0.0, 1.0, K)
    qs = q_from[None] + ts[:, None] * (q_to - q_from)[None]      # (K,dof)
    spheres = fk_spheres_batch(handle, qs).reshape(-1, 4)        # (K*S,4)
    return voxelize_spheres(voxmap, spheres)


def motion_stays_in_free(handle, voxmap, q_from, q_to,
                         resolution: Optional[float] = None) -> Tuple[bool, int]:
    """该段扫掠体积是否全部落在 FREE 体素内。

    返回 (ok, n_nonfree)：ok=True 表示整臂扫掠 ⊆ FREE；n_nonfree = 扫掠体积内非 FREE 体素数。
    保守：扫掠体积越界(voxmap 外)的部分按 voxmap.get 约定返回 UNKNOWN → 计入非 FREE。
    """
    from gt_gen.voxmap import FREE

    cells = swept_volume(handle, voxmap, q_from, q_to, resolution=resolution)
    if cells.shape[0] == 0:
        return True, 0
    states = np.asarray(voxmap.get(cells))
    n_nonfree = int((states != FREE).sum())
    return n_nonfree == 0, n_nonfree
