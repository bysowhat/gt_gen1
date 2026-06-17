"""初始引导 FREE 空间：在 retract 邻域把整臂小幅活动扫过的体素标 FREE。

动机：相机装在机械臂末端(Link6)，初始视野极小。探索起点 voxmap 全 UNKNOWN 时，
整臂扫掠体积必含 UNKNOWN → reach_pt=0，机械臂第一步就动不了。给 retract（固定安全
home）邻域一小块"已知自由"活动空间，机械臂即可起步、转动相机、逐步观测把可行区往外扩。

本函数既用于 Step7 自测（verify_compute_reach_pt 的第二种验证），也用于正式 GT 生成的
起步引导——两处共用同一份定义与参数（dq 来自 configs/default.yaml 的 init_free 段）。

提供两种方案（均把"起步活动空间"标 FREE，可在 calibrate_init_free.py 中并排标定/对比）：
- set_initial_free_space ：按各关节 ±dq 整臂扫掠并集，精确贴合活动 blob，体素更省。
- set_initial_free_cylinder：以 base 竖直轴为中心、外接 retract 整臂 + margin 的圆柱整块，
  规整、与轨迹无关，但体素更多、可能更靠近工件。

空间大小由 tmp/plan_seam 已规划轨迹标定（见 docs/initial-free-space.md 与
scripts/calibrate_init_free.py）：dq=0.05rad 已让 122/122 条轨迹 reach_idx≥4，
默认取 dq=0.10rad 留余量（blob≈1900 体素）。退回 retract 是固定安全 home，其小邻域
必为自由，标 FREE 安全。
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def set_initial_free_space(handle, voxmap, config=None,
                           retract: Optional[Sequence[float]] = None,
                           dq: Optional[float] = None,
                           return_cells: bool = False):
    """把 retract 邻域整臂活动空间标 FREE。

    做法：以 retract 为中心，对每个关节做 ±dq 单关节扰动得到一组构型，对每个
    retract→构型 段做整臂扫掠(swept_volume)，并集去重后置 FREE。
    （单关节 ±dq = "原地小幅旋转/摆动各关节"，对应末端相机的小幅扫视。）

    参数：
      handle  : CuroboHandle（只用其运动学做 FK，不需要 MESH 碰撞世界）。
      voxmap  : ThreeStateVoxelMap（就地修改，把 blob 体素置 FREE）。
      config  : Config；retract/dq 为 None 时从它取（retract_config / init_free_dq）。
      retract : 起点构型（rad）；None 时取 config.retract_config。
      dq      : 各关节活动半幅（rad）；None 时取 config.init_free_dq。
      return_cells : True 则额外返回标记的体素下标 (M,3)。

    返回：标记为 FREE 的体素数 n（return_cells=True 时返回 (n, cells)）。
    """
    from gt_gen.voxmap import FREE
    from gt_gen.swept import swept_volume

    if retract is None:
        if config is None:
            raise ValueError("需要 retract 或 config 之一来确定起点构型")
        retract = config.retract_config
    if dq is None:
        dq = config.init_free_dq if config is not None else 0.10

    q0 = np.asarray(retract, dtype=float)
    dq = float(dq)

    chunks = []
    for j in range(q0.shape[0]):
        for s in (1.0, -1.0):
            q = q0.copy()
            q[j] += s * dq
            cells = swept_volume(handle, voxmap, q0, q)   # 已去重、在界内
            if cells.shape[0]:
                chunks.append(cells)

    if chunks:
        cells = np.unique(np.concatenate(chunks, axis=0), axis=0)
    else:
        cells = np.empty((0, 3), dtype=np.int64)

    n = voxmap.set_many(cells, FREE)
    return (n, cells) if return_cells else n


def base_cylinder_bounds(handle, retract, margin: float = 0.0):
    """返回刚好罩住整条 retract 机械臂的最小竖直圆柱尺寸 (radius, height)（轴过 base 原点 x=y=0）。

    仅作参考/默认值来源（如想让圆柱自动贴合整臂时取这个尺寸）：
      radius = 各碰撞球(水平距轴 + 球半径)的最大值 + margin
      height = 各球(z + 半径)的最大值 + margin            —— 底面取 base_link 平面 z=0，故高=顶z
    """
    from gt_gen.swept import fk_spheres_batch

    q0 = np.asarray(retract, dtype=float)
    sph = np.asarray(fk_spheres_batch(handle, [q0])[0], dtype=float)
    sph = sph[sph[:, 3] > 1e-4]
    rad_xy = np.linalg.norm(sph[:, :2], axis=1) + sph[:, 3]
    radius = float(rad_xy.max()) + float(margin)
    height = float((sph[:, 2] + sph[:, 3]).max()) + float(margin)
    return radius, height


def set_initial_free_cylinder(handle, voxmap, config=None,
                              radius: Optional[float] = None,
                              height: Optional[float] = None,
                              z_min: Optional[float] = None,
                              return_cells: bool = False):
    """把一个竖立在 base_link 平面、给定半径/高度的圆柱体内的体素整块标 FREE（初始 FREE 空间的另一方案）。

    相比 set_initial_free_space（精确按各关节小幅扫掠勾出活动 blob），本法更"规整、与轨迹无关"：
    取一个轴过 base 原点(x=y=0)、底面 z=z_min(默认 0 = base_link 平面)、顶面 z=z_min+height、半径
    radius 的竖直圆柱，整块置 FREE。半径/高度直接给定（不自动贴合整臂）。缺点是同样起步效果下体素
    数通常更多、更可能靠近工件（用 calibrate_init_free.py 的安全核查 + base_cylinder_bounds 参考）。

    参数：
      handle  : CuroboHandle（本方案不做 FK，保留以与 set_initial_free_space 同签名；可传 None）。
      voxmap  : ThreeStateVoxelMap（就地修改）。
      config  : Config；radius/height 为 None 时从它取（init_free_cyl_radius / init_free_cyl_height）。
      radius  : 圆柱半径（米）；None 时取 config.init_free_cyl_radius。
      height  : 圆柱高度（米，从 z_min 往上）；None 时取 config.init_free_cyl_height。
      z_min   : 圆柱底面 z（米，base 系）；None 时取 config.init_free_cyl_z_min（默认 -0.02，
                盖住固定底座扎到 z<0 的那层体素）。
      return_cells : True 则额外返回标记的体素下标 (M,3)。

    返回：标记为 FREE 的体素数 n（return_cells=True 时返回 (n, cells)）。
    """
    from gt_gen.voxmap import FREE

    if radius is None:
        if config is None:
            raise ValueError("需要 radius 或 config 之一来确定圆柱半径")
        radius = config.init_free_cyl_radius
    if height is None:
        if config is None:
            raise ValueError("需要 height 或 config 之一来确定圆柱高度")
        height = config.init_free_cyl_height
    if z_min is None:
        z_min = config.init_free_cyl_z_min if config is not None else 0.0
    radius = float(radius)
    z_max = float(z_min) + float(height)

    # 枚举 voxmap 内中心落在圆柱体内的体素（向量化）
    nx, ny, nz = voxmap.shape
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
    c = voxmap.voxel_to_world(idx)
    rxy = np.hypot(c[:, 0], c[:, 1])                      # 到 base 竖直轴(x=y=0)的水平距离
    inside = (rxy <= radius) & (c[:, 2] >= float(z_min)) & (c[:, 2] <= z_max)
    cells = idx[inside]

    n = voxmap.set_many(cells, FREE)
    return (n, cells) if return_cells else n
