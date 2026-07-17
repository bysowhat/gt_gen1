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


# 固定底座 link（都在 Link1 关节之前，永不随关节运动）：其碰撞球恒定不动，不可能"新"碰到障碍，
# 却常扎进「基座平面以下 / 初始 FREE 圆柱外」的从未观测区(UNKNOWN)，使每段运动都假阳性非 FREE。
# 故从扫掠体积里整体剔除（实测：z<0 的格全部来自 xiaoyu_base_link）。
_FIXED_BASE_LINKS = ("base_link", "xiaoyu_base_link", "xiaoyu_arm_base_link")


def _moving_sphere_mask(handle):
    """整臂碰撞球里「会随关节运动」的球掩码 (S,) bool：剔除 _FIXED_BASE_LINKS 的固定底座球。
    结果按 handle 缓存（kinematics 不变）。某 handle 取不到映射时退回全 True（不剔除，保守不漏检）。"""
    cached = getattr(handle, "_moving_sphere_mask_cache", None)
    if cached is not None:
        return cached
    try:
        kc = handle.mg.kinematics.kinematics_config
        idx_map = kc.link_sphere_idx_map.detach().cpu().numpy()       # (S,) 每球所属 link 下标
        name_to_idx = kc.link_name_to_idx_map
        drop = [name_to_idx[n] for n in _FIXED_BASE_LINKS if n in name_to_idx]
        mask = ~np.isin(idx_map, drop)
    except Exception:
        mask = None
    handle._moving_sphere_mask_cache = mask
    return mask


def link_sphere_mask(handle, link_name):
    """指定 link 的碰撞球掩码 (S,) bool：只保留属于 link_name 的球。
    结果按 (handle, link_name) 缓存（kinematics 不变）。取不到映射时返回 None。"""
    cache = getattr(handle, "_link_sphere_mask_cache", None)
    if cache is None:
        cache = handle._link_sphere_mask_cache = {}
    if link_name in cache:
        return cache[link_name]
    try:
        kc = handle.mg.kinematics.kinematics_config
        idx_map = kc.link_sphere_idx_map.detach().cpu().numpy()       # (S,) 每球所属 link 下标
        name_to_idx = kc.link_name_to_idx_map
        mask = (idx_map == name_to_idx[link_name])
    except Exception:
        mask = None
    cache[link_name] = mask
    return mask


def voxelize_spheres(voxmap, spheres) -> np.ndarray:
    """把一组碰撞球 (M,4 xyz+r) 体素化到 voxmap：返回真正与球相交的体素下标 (M,3)（去重、在界内）。

    判据为精确的「球-AABB(体素立方体)相交」：球心到体素盒的最近距离 ≤ r 才算覆盖
    （逐轴 clamp：d_k = max(0, |c_k - ctr_k| - vs/2)，∑d_k² ≤ r²）。
    候选枚举仍用 r + 半体对角线 框 AABB（保证候选集是超集、不漏），再用精确判据剔除"只擦到角外"的体素。

    向量化实现：用【全体球的每轴最大 AABB 跨度】搭一个统一相对偏移网格，广播到每个球（lo+off），
    再用 off ≤ 该球自身 (hi-lo) 掩掉多出的偏移——等价于原逐球 meshgrid(lo..hi)，但去掉 Python 逐球循环。
    """
    spheres = np.asarray(spheres, float)
    spheres = spheres[spheres[:, 3] > 1e-4]
    if spheres.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int64)
    vs = voxmap.voxel_size
    half = 0.5 * vs                          # 体素半边长（精确判据用）
    pad = 0.5 * vs * np.sqrt(3.0)            # 半体对角线，仅用于框候选 AABB（超集、不漏）
    c = spheres[:, :3]                       # (M,3)
    r = spheres[:, 3]                        # (M,)
    R = (r + pad)[:, None]                   # (M,1)
    lo = voxmap.world_to_voxel(c - R).astype(np.int64)          # (M,3)
    hi = voxmap.world_to_voxel(c + R).astype(np.int64)          # (M,3)
    span = hi - lo                                              # (M,3) 每轴 AABB 跨度（含端 = span+1 格）
    ext = np.maximum(span.max(axis=0) + 1, 1)                   # (3,) 统一偏移网格尺寸
    ox, oy, oz = np.meshgrid(np.arange(ext[0]), np.arange(ext[1]), np.arange(ext[2]),
                             indexing="ij")
    off = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1).astype(np.int64)  # (E,3)
    idx = lo[:, None, :] + off[None, :, :]                      # (M,E,3) 候选下标
    within = np.all(off[None, :, :] <= span[:, None, :], axis=2)   # (M,E) 掩掉超出各球自身 AABB 的偏移
    ctr = voxmap.voxel_to_world(idx.reshape(-1, 3)).reshape(spheres.shape[0], -1, 3)  # (M,E,3) 体素中心
    d = np.maximum(0.0, np.abs(ctr - c[:, None, :]) - half)     # 逐轴超出体素盒的距离 (M,E,3)
    hit = ((d * d).sum(axis=2) <= (r * r)[:, None]) & within    # (M,E) 精确球-AABB 相交
    sel = idx[hit]                                              # (K,3) 命中体素（含重复）
    if sel.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int64)
    idx_u = np.unique(sel, axis=0)
    return idx_u[voxmap.in_bounds(idx_u)]


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
    spheres = fk_spheres_batch(handle, qs)                       # (K,S,4)
    mask = _moving_sphere_mask(handle)                           # 剔除固定底座球（None=不剔除）
    if mask is not None:
        spheres = spheres[:, mask, :]
    return voxelize_spheres(voxmap, spheres.reshape(-1, 4))


def _debug_viz_cells(handle, voxmap, q_from, q_to, cells):
    """调试用【整臂扫掠体素 cells】：可视化 swept_volume 返回的扫掠体素（按三态着色）。
    被调用即弹窗(需显示器+open3d)；无显示器/服务器跑时把调用行注释掉即可（调用行本身就是开关）。

    画 cells 中 FREE 格(蓝半透明) + 非 FREE 格(OCCUPIED/UNKNOWN，橙不透明) + 两端整臂碰撞球
    (q_from 绿 / q_to 青) + ROI/base。一眼看出「这段运动整臂扫过哪些格、是否擦到非 FREE」。
    依赖 verify_step8 的 open3d 工具。
    """
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from verify_step8 import _arm_mesh, _cells_mesh, _draw, _roi_and_base
    from gt_gen.voxmap import FREE

    af = _arm_mesh(handle, list(q_from)); af.paint_uniform_color([0.10, 0.80, 0.20])
    at = _arm_mesh(handle, list(q_to)); at.paint_uniform_color([0.10, 0.75, 0.85])
    geoms = [("q_from", af, "lit", None), ("q_to", at, "lit", None)]

    n_free = n_nonfree = 0
    if cells.shape[0]:
        states = np.asarray(voxmap.get(cells))
        centers = voxmap.voxel_to_world(cells)
        free_c = centers[states == FREE]
        nonfree_c = centers[states != FREE]
        n_free = int(free_c.shape[0]); n_nonfree = int(nonfree_c.shape[0])
        if n_free:
            geoms.append(("swept_free", _cells_mesh(voxmap, free_c), "fill", [0.20, 0.45, 0.95, 0.25]))
        if n_nonfree:
            nm = _cells_mesh(voxmap, nonfree_c); nm.paint_uniform_color([1.0, 0.35, 0.0])
            geoms.append(("swept_nonfree", nm, "lit", None))     # 非 FREE：撞到的格(橙)
    geoms += _roi_and_base(voxmap)
    _draw(geoms, f"swept 扫掠体素: FREE={n_free}格(蓝) 非FREE={n_nonfree}格(橙) "
                 f"绿=q_from整臂 青=q_to整臂")


def _debug_viz_cells2(handle, voxmap, q_from, q_to, cells):
    """调试用【整臂扫掠体素 cells，单色版】：cells【整体只用一种颜色(黄)】，其他什么都不画
    (无整臂、无 ROI/base)。被调用即弹窗(需显示器+open3d)；服务器/无显示器时把调用行注释掉即可。

    只看「整臂扫过的体素几何形状/占多大一片」，不区分 FREE/非 FREE。依赖 verify_step8 的 open3d 工具。
    """
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from verify_step8 import _cells_mesh, _draw

    geoms = []
    n = int(cells.shape[0])
    if n:
        centers = voxmap.voxel_to_world(cells)
        geoms.append(("swept", _cells_mesh(voxmap, centers), "fill", [1.0, 0.92, 0.0, 0.45]))  # 全部黄（单色）
    _draw(geoms, f"swept 扫掠体素(单色黄): cells={n}格")


def motion_stays_in_free(handle, voxmap, q_from, q_to,
                         resolution: Optional[float] = None) -> Tuple[bool, int]:
    """该段扫掠体积是否全部落在 FREE 体素内。

    返回 (ok, n_nonfree)：ok=True 表示整臂扫掠 ⊆ FREE；n_nonfree = 扫掠体积内非 FREE 体素数。
    保守：扫掠体积越界(voxmap 外)的部分按 voxmap.get 约定返回 UNKNOWN → 计入非 FREE。
    """
    from gt_gen.voxmap import FREE

    cells = swept_volume(handle, voxmap, q_from, q_to, resolution=resolution)
    # _debug_viz_cells2(handle, voxmap, q_from, q_to, cells)        # 看整臂扫掠体素 cells（注释此行可关）
    # _debug_viz_cells(handle, voxmap, q_from, q_to, cells)        # 看整臂扫掠体素 cells（注释此行可关）
    if cells.shape[0] == 0:
        return True, 0
    states = np.asarray(voxmap.get(cells))
    n_nonfree = int((states != FREE).sum())
    return n_nonfree == 0, n_nonfree
