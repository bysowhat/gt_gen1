"""M1.7 单 cycle 探索的所有可调超参集中.

修改这个文件就能调实验, 不用改主流程代码.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    # 0.30: 末端笛卡尔采样盒半边长 (米). 注意不能太小 — 从远 home 起点 (~1.3m)
    # 接近观测壳需要跨 ~0.7m, 0.15 的小盒里 IK+碰撞通过的最近候选无法单调逼近
    # (机械臂要大幅重构型), 会卡在 ~1.1m; 0.30 给足空间, 3-5 cycle 即可进壳.
    sample_box_half: float = 0.30
    n_samples: int = 100                  # 每 cycle 撒多少候选
    # yaw/pitch 抖动: 让光轴在"瞄准 P_weld"基础上小幅偏摆, 探索略不同视角.
    # 必须 < 视锥半角 (竖直 28.5°), 否则目标会被晃出视野 → frustum 失败、无法合格观测.
    # wedge 是由相机"位置"决定的 (位置由采样盒探索), 朝向只需精确瞄准目标, 故抖动要小.
    yaw_jitter_deg: float = 12.0
    pitch_jitter_deg: float = 12.0
    # 绕相机光轴的滚转角抖动 (度). roll 不改变"看向哪"(光轴不变, wedge/LOS/距离都不受影响),
    # 但显著影响 IK 可行性 —— 同一观测位置, 某些 roll 解不出 IK, 换个 roll 就行.
    # 默认 180 = 全范围, 让采样覆盖所有 roll, 否则远侧/刁钻位姿会因 roll 卡死 IK 而被漏掉.
    roll_jitter_deg: float = 180.0
    # 每 cycle 候选里, 多大比例是"目标导向观测候选"(直接在 shell∩wedge 精确瞄准目标),
    # 其余是"探索候选"(EE 周围盒). 目标候选够不到时被 IK 滤掉, 够得到时直接锁定合格观测.
    frac_goal_candidates: float = 0.4
    # 每个目标观测位置系统扫多少个绕光轴 roll. roll 是 IK 关键 DOF —— 远侧/刁钻位姿
    # 往往只有窄窄一段 roll 能解 IK, 随机单 roll 命中率极低 (~0.5%), 系统扫 12 个能到 ~5%.
    goal_roll_sweep: int = 12

    # ---- gain (M1.6) ----
    w_vol: float = 1.0                    # 体积增益权重
    w_target: float = 2000.0              # 目标偏置权重 (跟 vol_gain 量级)
    gain_n_rays_h: int = 8
    gain_n_rays_v: int = 8
    gain_max_range: float = 2.0
    gain_fov_h_deg: float = 86.0
    gain_fov_v_deg: float = 57.0

    # ---- 接近观测壳 (M1.8) ----
    # 关键: target_bias 只奖励"朝向"目标, 不奖励"距离". 从远 home 起点出发时,
    # 所有候选都朝向 P_weld → target_bias 不分远近 → vol_gain 把 EE 推向开阔空间.
    # 这一项把 EE 往观测壳 [obs_d_min, obs_d_max] 的中位距离拉, 提供位置牵引.
    # 量级要能压住 vol_gain 候选间的差异 (~±1000 / 0.15m).
    w_approach: float = 12000.0           # |d - d_mid| 的惩罚权重
    # 对准 wedge 角平分线: 奖励 (cam_pos - p_target) 方向贴近 seam_limits 中线,
    # 让观测尽量"正对"焊缝开口中央 (cos∈[-1,1], 越大越居中). 没传 seam_limits 则无效.
    w_bisector: float = 3000.0

    # ---- 测地距离场绕障 (M1.8.5) ----
    # approach 项改用"候选→合格观测区(shell∩wedge)的测地距离"(穿自由空间, 绕开 occupied),
    # 这样当观测区在工件另一面时, 梯度会沿绕路把相机引过去 (欧氏距离做不到).
    use_geodesic_approach: bool = True    # False 退回欧氏 |d - d_mid|
    geo_field_clamp: float = 5.0          # 测地距离上限/不可达填充 (米), 防 inf 主导
    geo_wedge_margin_deg: float = 0.0     # 目标种子 wedge 收紧余量

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
    # 网格范围 = base ± (arm_reach + workspace_pad) 立方盒 (M1.8.5).
    # 测地场活在网格里, 所以网格必须罩住机械臂能到达 + 相机会经过的整个工作空间.
    arm_reach: float = 1.9                # UR12e 臂展 (米)
    workspace_pad: float = 0.8            # 臂展外再留多少米

    # ---- 相机 ----
    cam_fov_h_deg: float = 86.0
    cam_fov_v_deg: float = 57.0
    cam_height: int = 360
    cam_width: int = 640

    # ---- cuRobo (M1.4) ----
    edge_step_rad: float = 0.05           # 边碰撞检测插值步长

    # ---- 合格位姿求解器 (M1.8.5, 针对针尖可行域的刁钻焊缝) ----
    # 卡住时调用: 大批量在 shell∩wedge 搜 IK+碰撞+合格 的观测位姿. 慢但能啃下深凹角/远侧.
    use_pose_solver: bool = True
    solver_budget: int = 20000            # 每次求解最多试多少候选
    solver_max_calls: int = 3             # 整个 Phase1 最多调用几次 (控制总开销)

    # ---- 主循环 ----
    max_cycles: int = 100                 # 安全终止
    patience: int = 3                     # M1.8: 多少次"没接近观测壳"后切 Global
    # 一个 cycle 让"到观测壳的距离"至少缩短这么多米, 才算有进展 (否则 stuck++).
    progress_eps: float = 0.01

    # ---- Global step (M1.8) ----
    # frontier 阈值: 节点 gain >= 这个值才算"值得过去". None 表示用 g_low*2.
    global_frontier_gain_threshold: Optional[float] = None
    # Global score = gain · exp(-global_lambda_cost · path_cost), 选最高的.
    # 注意单位: path_cost 是关节空间距离 (rad), 这里 lambda 比 lambda_cost 小很多.
    global_lambda_cost: float = 0.5


__all__ = ["Phase1Config", "INITIAL_JOINT_ANGLES"]
