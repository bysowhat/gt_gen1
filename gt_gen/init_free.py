"""初始引导 FREE 空间：在 retract 邻域把整臂小幅活动扫过的体素标 FREE。

动机：相机装在机械臂末端(Link6)，初始视野极小。探索起点 voxmap 全 UNKNOWN 时，
整臂扫掠体积必含 UNKNOWN → reach_pt=0，机械臂第一步就动不了。给 retract（固定安全
home）邻域一小块"已知自由"活动空间，机械臂即可起步、转动相机、逐步观测把可行区往外扩。

本函数既用于 Step7 自测（verify_compute_reach_pt 的第二种验证），也用于正式 GT 生成的
起步引导——两处共用同一份定义与参数（dq 来自 configs/default.yaml 的 init_free 段）。

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
