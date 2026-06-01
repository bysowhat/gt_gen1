"""M1.7 单 cycle 探索的所有可调超参集中.

修改这个文件就能调实验, 不用改主流程代码.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Phase 1 固定初始关节角 (= ur12e.yml 里的 retract_config).
# 注意: pkl 里的 joint_angles 是焊接到位姿 (EE 贴在 P_weld 上), 不是初始姿态.
# Phase 1 从这个 home 位姿出发, 通过 cycle 探索找观测位姿.
INITIAL_JOINT_ANGLES = np.array([
    1.5707824230194092,
    -2.0071660480894984,
    1.3613484541522425,
    -0.9599629205516357,
    -1.570770565663473,
    0.0,
], dtype=np.float64)


@dataclass
class Phase1Config:
    """Phase 1 探索全部超参. 默认值适合 BEAM 工件 + UR12e + RealSense D455."""

    # ---- 候选采样 ----
    sample_box_half: float = 0.15        # 末端笛卡尔采样盒半边长 (米)
    n_samples: int = 100                  # 每 cycle 撒多少候选
    yaw_jitter_deg: float = 30.0          # 朝向 yaw 抖动幅度 (度)
    pitch_jitter_deg: float = 30.0

    # ---- gain (M1.6) ----
    w_vol: float = 1.0                    # 体积增益权重
    w_target: float = 2000.0              # 目标偏置权重 (跟 vol_gain 量级)
    gain_n_rays_h: int = 8
    gain_n_rays_v: int = 8
    gain_max_range: float = 2.0
    gain_fov_h_deg: float = 86.0
    gain_fov_v_deg: float = 57.0

    # ---- 评分 ----
    lambda_cost: float = 50.0             # 关节空间距离惩罚 (rad → score 单位)
    g_high: float = 100.0                 # 立即执行阈值 (M1.8 用)
    g_low: float = 5.0                    # 卡住阈值 (M1.8 用)

    # ---- 合格观测 (M1.3) ----
    obs_d_min: float = 0.30
    obs_d_max: float = 0.60
    obs_los_unknown_as_block: bool = True   # 严格模式: unknown 当障碍

    # ---- voxel 地图 (M1.2) ----
    voxel_res: float = 0.02
    voxel_max_range: float = 2.0
    voxel_bounds_pad: float = 0.5         # bbox 外扩多少米

    # ---- 相机 ----
    cam_fov_h_deg: float = 86.0
    cam_fov_v_deg: float = 57.0
    cam_height: int = 360
    cam_width: int = 640

    # ---- cuRobo (M1.4) ----
    edge_step_rad: float = 0.05           # 边碰撞检测插值步长

    # ---- 主循环 ----
    max_cycles: int = 100                 # 安全终止
    patience: int = 3                     # M1.8: 多少次低分后切 Global


__all__ = ["Phase1Config", "INITIAL_JOINT_ANGLES"]
