"""M1.7 单 cycle 探索 — 把 M1.1–M1.6 串起来.

每个 cycle 7 步:
    1. 采样候选 EE pose (笛卡尔小立方体 + 朝向偏置)
    2. IK 过滤 (cuRobo, M1.4)
    3. 整臂碰撞过滤 (cuRobo, M1.4)
    4. 算每个候选的 gain (M1.6)
    5. 算 score = gain - λ·trajectory_cost
    6. 选 max score → next_theta
    7. 加节点+边到 graph (M1.5), 检查 phase 切换

主入口: run_one_cycle()
返回: CycleResult (含决策 + 各步骤候选数 + phase 切换标志)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from phase1.arm_kin import ArmKin
from phase1.config import Phase1Config
from phase1.depth_source import CameraIntrinsics
from phase1.gain import GainConfig, gain as compute_gain
from phase1.geodesic import (
    build_shell_wedge_goal_mask,
    geodesic_field,
    sample_field_at,
)
from phase1.graph import ExplorationGraph, GraphNode
from phase1.mapping import VoxelMap
from phase1.observation_check import Viewpoint, is_valid_observation, wedge_bisector


# ---------------------------------------------------------------------- 数据结构


@dataclass
class CycleResult:
    """单 cycle 的输出."""

    success: bool                                # 是否选出可执行候选
    next_theta: Optional[np.ndarray] = None      # (6,) 下一步关节角
    next_ee_pose_world: Optional[np.ndarray] = None  # (4, 4)
    best_score: float = 0.0
    best_gain: float = 0.0
    best_traj_cost: float = 0.0
    best_node_id: Optional[str] = None

    n_candidates_total: int = 0
    n_candidates_after_ik: int = 0
    n_candidates_after_collision: int = 0

    phase_switch_triggered: bool = False
    p_seam_observed: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------- 子步骤


def _rotate_about_axis(v: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
    """绕 axis 旋转 v 角度 ang (Rodrigues)."""
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(ang), np.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def _look_at_pose(cam_pos: np.ndarray, p_target: np.ndarray,
                  roll: float, rng: np.random.Generator) -> np.ndarray:
    """构造"相机精确看向 p_target"的 4×4 位姿, 绕光轴滚转 roll. OpenCV 系 (z=光轴)."""
    z = np.asarray(p_target) - np.asarray(cam_pos)
    zn = np.linalg.norm(z)
    z = z / zn if zn > 1e-9 else np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    x = _rotate_about_axis(x, z, roll)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = cam_pos
    return T


def sample_observation_candidates(
    p_target: np.ndarray,
    cfg: Phase1Config,
    n: int,
    seam_limits: np.ndarray | None = None,
    tangent: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """目标导向候选: 在观测壳 [d_min,d_max] (∩wedge) 里精确瞄准 p_target 的位姿.

    这些候选位置不依赖当前 EE —— 它们是"理想观测点". 够不到时 IK 会滤掉 (靠盒
    候选绕行接近); 够得到时直接通过 IK+碰撞+合格检查 → 锁定. roll 全采样保 IK 可行.
    返回 (n,4,4).
    """
    if rng is None:
        rng = np.random.default_rng()
    p = np.asarray(p_target, dtype=np.float64).reshape(3)

    # wedge 中线 + 半角 (有 seam_limits 时), 否则全球面
    bis = None
    cos_half = -1.0
    if seam_limits is not None:
        bis = wedge_bisector(seam_limits, tangent=tangent)
        sl = np.asarray(seam_limits, dtype=np.float64)
        d1 = sl[0] / (np.linalg.norm(sl[0]) + 1e-12)
        d2 = sl[1] / (np.linalg.norm(sl[1]) + 1e-12)
        cos_gap = float(np.clip(np.dot(d1, d2), -1.0, 1.0))
        cos_half = float(np.cos(np.radians(np.degrees(np.arccos(cos_gap)) / 2.0)))

    poses = []
    n_roll = max(1, int(cfg.goal_roll_sweep))
    n_pos = max(1, int(np.ceil(n / n_roll)))
    rolls = np.linspace(-np.pi, np.pi, n_roll, endpoint=False)
    tries = 0
    while len(poses) < n and tries < n_pos * 50:
        tries += 1
        v = rng.normal(size=3)
        v /= np.linalg.norm(v) + 1e-12
        if bis is not None:
            if np.dot(v, bis) < cos_half:        # 不在 wedge 内, 丢
                continue
        d = rng.uniform(cfg.obs_d_min, cfg.obs_d_max)
        cam_pos = p + v * d
        # 同一观测位置系统扫多个 roll: roll 是 IK 关键 DOF, 刁钻位姿只有窄段 roll 可解
        jit = rng.uniform(-np.pi / n_roll, np.pi / n_roll)
        for r in rolls:
            poses.append(_look_at_pose(cam_pos, p, float(r + jit), rng))
            if len(poses) >= n:
                break
    return np.array(poses[:n]) if poses else np.zeros((0, 4, 4))


def sample_candidate_ee_poses(
    ee_pos_now: np.ndarray,
    p_target: np.ndarray,
    cfg: Phase1Config,
    n: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """围 ee_pos_now 撒 n 个候选末端位姿 (4×4 SE(3)).

    位置: 笛卡尔立方体 [-half, half]^3 内均匀
    朝向: z_axis 大致指向 p_target, 加 yaw/pitch 抖动
    返回 (n, 4, 4) world frame.
    """
    if rng is None:
        rng = np.random.default_rng()
    ee_pos_now = np.asarray(ee_pos_now, dtype=np.float64).reshape(3)
    p_target = np.asarray(p_target, dtype=np.float64).reshape(3)

    half = cfg.sample_box_half
    pos_offsets = rng.uniform(-half, half, size=(n, 3))
    positions = ee_pos_now + pos_offsets

    poses = np.zeros((n, 4, 4))
    yaw_max = np.radians(cfg.yaw_jitter_deg)
    pitch_max = np.radians(cfg.pitch_jitter_deg)
    roll_max = np.radians(cfg.roll_jitter_deg)

    for i, pos in enumerate(positions):
        # 主朝向: 指向 p_target
        z_axis = p_target - pos
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-6:
            z_axis = np.array([0.0, 0.0, 1.0])     # 退化, 随便给一个
        else:
            z_axis = z_axis / z_norm

        # yaw 抖动 (绕 world z), pitch 抖动 (绕 ⊥ z 的某轴)
        yaw = rng.uniform(-yaw_max, yaw_max)
        pitch = rng.uniform(-pitch_max, pitch_max)
        z_axis = _rotate_about_axis(z_axis, np.array([0, 0, 1.0]), yaw)
        # pitch 轴: 在 ⊥ z_axis 的平面里随便取
        up_hint = np.array([0, 0, 1.0])
        if abs(np.dot(z_axis, up_hint)) > 0.99:
            up_hint = np.array([0, 1.0, 0])
        pitch_axis = np.cross(up_hint, z_axis)
        pitch_axis /= np.linalg.norm(pitch_axis)
        z_axis = _rotate_about_axis(z_axis, pitch_axis, pitch)

        # 构造 OpenCV 系基 (z 前, x 右, y 下)
        if abs(np.dot(z_axis, up_hint)) > 0.99:
            up_hint = np.array([0, 1.0, 0])
        x_axis = np.cross(up_hint, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        # 绕光轴 roll 抖动: 不改变光轴(看向不变), 只换 IK 可行性
        roll = rng.uniform(-roll_max, roll_max)
        x_axis = _rotate_about_axis(x_axis, z_axis, roll)
        y_axis = np.cross(z_axis, x_axis)
        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        poses[i, :3, :3] = R
        poses[i, :3, 3] = pos
        poses[i, 3, 3] = 1.0

    return poses


def filter_by_ik(
    poses: np.ndarray,
    arm_kin: ArmKin,
) -> tuple[np.ndarray, torch.Tensor]:
    """对每个候选 pose 跑 IK, 保留可解的.

    返回 (poses_pass (M, 4, 4), thetas_pass (M, dof) tensor).
    """
    if len(poses) == 0:
        return poses[:0], torch.zeros((0, arm_kin.dof), device=arm_kin.tensor_args.device,
                                       dtype=torch.float32)

    ee_pos = poses[:, :3, 3]                                          # (N, 3)
    # 从 R 提 quaternion (scipy)
    from scipy.spatial.transform import Rotation as ScipyRot
    R_mats = poses[:, :3, :3]                                          # (N, 3, 3)
    quats_xyzw = ScipyRot.from_matrix(R_mats).as_quat()               # (N, 4) [x,y,z,w]
    quats_wxyz = np.stack([quats_xyzw[:, 3], quats_xyzw[:, 0],
                            quats_xyzw[:, 1], quats_xyzw[:, 2]], axis=1)

    res = arm_kin.ik(ee_pos, quats_wxyz)
    success = res["success"].numpy().astype(bool)
    if success.sum() == 0:
        return poses[:0], torch.zeros((0, arm_kin.dof),
                                       device=arm_kin.tensor_args.device,
                                       dtype=torch.float32)

    poses_pass = poses[success]
    thetas_pass = res["theta"][torch.from_numpy(success)]              # (M, dof)
    return poses_pass, thetas_pass


def filter_by_collision(
    thetas: torch.Tensor,
    poses: np.ndarray,
    arm_kin: ArmKin,
) -> tuple[np.ndarray, torch.Tensor]:
    """整臂碰撞过滤. 假设 arm_kin.update_world() 已调过."""
    if len(thetas) == 0:
        return poses, thetas
    coll = arm_kin.collides(thetas)                                    # (N,) bool
    keep = ~coll.cpu().numpy().astype(bool)
    return poses[keep], thetas[torch.from_numpy(keep).to(thetas.device)]


def score_candidates(
    poses: np.ndarray,
    thetas: torch.Tensor,
    theta_now: np.ndarray,
    voxel_map: VoxelMap,
    p_target: np.ndarray,
    arm_kin: ArmKin,
    cfg: Phase1Config,
    target_bisector: np.ndarray | None = None,
    geo_field: "torch.Tensor | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """为每个候选 (pose, theta) 算 score.

    score = gain (M1.6: w_vol·vol + w_target·target_bias)
          + approach (M1.8.5: -w_approach·测地距离(候选→合格观测区), 绕障导航)
          + bisector (M1.8: +w_bisector·cos(rel, 中线), 让观测正对 wedge 中央)
          - λ·trajectory_cost (关节空间 L2)

    target_bisector: p_target 处 wedge 角平分线; None 跳过 bisector 项.
    geo_field: 测地距离场 (nx,ny,nz). 给了就用测地距离做 approach (绕障);
        None 则退回欧氏 |d - d_mid| (无障碍/单测用).

    返回 (scores, gains, traj_costs), 都是 (K,) numpy.
    注: gains 只含 M1.6 gain (不含 approach/bisector), 方便 CycleResult 报告与 frontier 复用.
    """
    K = len(poses)
    if K == 0:
        return (np.zeros(0), np.zeros(0), np.zeros(0))

    gain_cfg = GainConfig(
        w_vol=cfg.w_vol, w_target=cfg.w_target,
        n_rays_h=cfg.gain_n_rays_h, n_rays_v=cfg.gain_n_rays_v,
        max_range=cfg.gain_max_range,
        fov_h_deg=cfg.gain_fov_h_deg, fov_v_deg=cfg.gain_fov_v_deg,
    )

    p_target_arr = np.asarray(p_target, dtype=np.float64).reshape(3)
    d_mid = 0.5 * (cfg.obs_d_min + cfg.obs_d_max)
    bis = None
    if target_bisector is not None:
        bis = np.asarray(target_bisector, dtype=np.float64).reshape(3)
        bn = np.linalg.norm(bis)
        bis = bis / bn if bn > 1e-9 else None

    cam_positions = poses[:, :3, 3]                          # (K, 3)
    d_euclid = np.linalg.norm(cam_positions - p_target_arr, axis=1)
    eucl_pen = np.abs(d_euclid - d_mid)                      # 到 d_mid 球的欧氏距离
    # ---- approach: 混合. 网格内可达 → 测地 (绕障); 网格外/不可达 → 欧氏 (远程牵引) ----
    # 两者在网格边界处 ≈ 相等 (开阔空间测地≈欧氏), 平滑衔接.
    if geo_field is not None:
        gd = sample_field_at(geo_field, voxel_map, cam_positions,
                             oob_value=float("inf"))
        finite = np.isfinite(gd)
        eff = np.where(finite, np.minimum(gd, cfg.geo_field_clamp), eucl_pen)
        approach = -cfg.w_approach * eff
    else:
        approach = -cfg.w_approach * eucl_pen

    gains = np.zeros(K, dtype=np.float64)
    bisr = np.zeros(K, dtype=np.float64)
    for i in range(K):
        T = poses[i]
        cam_pos = T[:3, 3]
        cam_dir = T[:3, 2]
        cam_up = -T[:3, 1]                # OpenCV: y 朝下 → up = -y
        gains[i] = compute_gain(cam_pos, cam_dir, cam_up,
                                 voxel_map, p_target, gain_cfg)
        if bis is not None:
            rel = cam_pos - p_target_arr
            dn = np.linalg.norm(rel)
            if dn > 1e-9:
                bisr[i] = cfg.w_bisector * float(np.dot(rel / dn, bis))

    # 轨迹代价 (关节空间 L2)
    theta_now_t = torch.tensor(theta_now, dtype=torch.float32,
                                device=arm_kin.tensor_args.device).unsqueeze(0)
    theta_now_b = theta_now_t.expand(K, -1)
    traj_costs = arm_kin.trajectory_cost(theta_now_b, thetas).cpu().numpy()

    scores = gains + approach + bisr - cfg.lambda_cost * traj_costs
    return scores, gains, traj_costs


def detect_phase_switch(
    voxel_map: VoxelMap,
    seam_points: np.ndarray,
    seam_tangents: np.ndarray,
    seam_limits: np.ndarray | None,
    current_viewpoint: Viewpoint,
    cfg: Phase1Config,
    intrinsics: CameraIntrinsics,
) -> list[int]:
    """在当前 viewpoint, 遍历所有 P_i, 看哪些已经能合格观测.

    返回所有 valid 观测的 P_i 索引列表.
    """
    observed = []
    N = len(seam_points)
    for i in range(N):
        slim_i = seam_limits[i] if seam_limits is not None else None
        r = is_valid_observation(
            current_viewpoint, seam_points[i], seam_tangents[i],
            voxel_map, intrinsics,
            seam_limits=slim_i,
            d_min=cfg.obs_d_min, d_max=cfg.obs_d_max,
            los_unknown_as_block=cfg.obs_los_unknown_as_block,
        )
        if r.valid:
            observed.append(i)
    return observed


# ---------------------------------------------------------------------- 主入口


def solve_observation_pose(
    arm_kin: ArmKin,
    voxel_map: VoxelMap,
    seam_points: np.ndarray,
    seam_tangents: np.ndarray,
    seam_limits: np.ndarray | None,
    p_target: np.ndarray,
    cfg: Phase1Config,
    intrinsics: CameraIntrinsics,
    rng: np.random.Generator | None = None,
    budget: int = 30000,
    chunk: int = 1000,
) -> Optional[tuple[np.ndarray, np.ndarray, list[int]]]:
    """专门的合格观测位姿求解器 (针对可行域极小的刁钻焊缝, 如深凹角/远侧/臂展边缘).

    探索循环每 cycle 只撒 ~百个候选, 命中针尖可行域 (~0.01%) 几乎不可能. 这里一次性
    大批量 (budget 个) 在 shell∩wedge 内精确瞄准 + roll 扫描, 批量 IK→碰撞→4 条件验证,
    返回最"居中"(贴 wedge 中线) 的合格 (theta, pose, observed). 找不到返回 None.

    用当前 voxel_map 做碰撞 + LOS, 所以越往后(地图越全)越容易找到.
    LOS 用宽松判定 (unknown 不算挡): 找"几何上可观测的可达位姿"; 相机移过去拍一帧后
    unknown→free, 主循环下一 cycle 的严格判定再确认.
    """
    import dataclasses
    if rng is None:
        rng = np.random.default_rng()
    arm_kin.update_world(voxel_map)
    relaxed_cfg = dataclasses.replace(cfg, obs_los_unknown_as_block=False)

    sp = np.asarray(seam_points, dtype=np.float64)
    ti = int(np.argmin(np.linalg.norm(sp - np.asarray(p_target).reshape(1, 3),
                                       axis=1)))
    tan_i = seam_tangents[ti] if seam_tangents is not None else None
    sl_i = seam_limits[ti] if seam_limits is not None else None
    bis = wedge_bisector(sl_i, tangent=tan_i) if sl_i is not None else None
    p = np.asarray(p_target, dtype=np.float64).reshape(3)

    best: Optional[tuple[np.ndarray, np.ndarray, list[int]]] = None
    best_centering = -np.inf
    spent = 0
    while spent < budget:
        n = min(chunk, budget - spent)
        spent += n
        poses = sample_observation_candidates(
            p, cfg, n, seam_limits=sl_i, tangent=tan_i, rng=rng,
        )
        if len(poses) == 0:
            continue
        poses_ik, thetas_ik = filter_by_ik(poses, arm_kin)
        if len(poses_ik) == 0:
            continue
        poses_col, thetas_col = filter_by_collision(thetas_ik, poses_ik, arm_kin)
        found_in_chunk = False
        for j in range(len(poses_col)):
            T = poses_col[j]
            vp = Viewpoint(pos=T[:3, 3], dir=T[:3, 2], up=-T[:3, 1])
            observed = detect_phase_switch(
                voxel_map, seam_points, seam_tangents, seam_limits,
                vp, relaxed_cfg, intrinsics,
            )
            if not observed:
                continue
            # 居中度 = (cam-P)方向 与 wedge 中线的对齐 (无 limits 时用到目标距离的负偏)
            rel = T[:3, 3] - p
            dn = np.linalg.norm(rel)
            centering = (float(np.dot(rel / dn, bis)) if (bis is not None and dn > 1e-9)
                         else -abs(dn - 0.5 * (cfg.obs_d_min + cfg.obs_d_max)))
            if centering > best_centering:
                best_centering = centering
                best = (thetas_col[j].cpu().numpy(), T.copy(), observed)
                found_in_chunk = True
        if found_in_chunk:
            break          # 这一批已找到合格位姿, 取最居中的即可, 不必耗尽 budget
        torch.cuda.empty_cache()
    return best


def run_one_cycle(
    theta_now: np.ndarray,
    voxel_map: VoxelMap,
    graph: ExplorationGraph,
    arm_kin: ArmKin,
    seam_points: np.ndarray,
    seam_tangents: np.ndarray,
    seam_limits: np.ndarray | None,
    p_target: np.ndarray,
    cycle_idx: int,
    cfg: Phase1Config,
    intrinsics: CameraIntrinsics,
    rng: np.random.Generator | None = None,
) -> CycleResult:
    """单 cycle 主函数: 采样 → IK + 碰撞过滤 → 评分 → 选最优 → 加图.

    Args:
        theta_now: 当前关节角 (6,)
        voxel_map: 当前体素地图 (调用方负责更新)
        graph: 全局图 (此函数只追加, 不重置)
        arm_kin: cuRobo 封装 (此函数会调用 update_world)
        seam_points / seam_tangents / seam_limits: pkl 数据 (注: limits 可能为 None)
        p_target: 目标点 (= seam_points[mid] 通常)
        cycle_idx: 第几个 cycle
        cfg: 超参
        intrinsics: 相机内参
        rng: 控制采样随机性 (单测用)
    """
    if rng is None:
        rng = np.random.default_rng()

    # ---- 1. 当前 EE pose ----
    fk = arm_kin.fk(torch.tensor(theta_now, dtype=torch.float32,
                                  device=arm_kin.tensor_args.device))
    ee_pos_now = fk["ee_pos_world"][0].cpu().numpy()

    # ---- 0. 把 voxel map 更新给 cuRobo (碰撞用) ----
    arm_kin.update_world(voxel_map)

    # 目标点在 seam 上的索引 → 取其 limits + tangent (供目标候选 + 中线 + 测地场)
    sp = np.asarray(seam_points, dtype=np.float64)
    ti = int(np.argmin(np.linalg.norm(sp - np.asarray(p_target).reshape(1, 3),
                                       axis=1)))
    tan_i = seam_tangents[ti] if seam_tangents is not None else None
    sl_i = seam_limits[ti] if seam_limits is not None else None

    # ---- 1. 采样: 探索候选 (EE 周围盒) + 目标导向候选 (shell∩wedge 精确瞄准) ----
    n_goal = int(round(cfg.n_samples * cfg.frac_goal_candidates))
    n_box = cfg.n_samples - n_goal
    box_poses = sample_candidate_ee_poses(ee_pos_now, p_target, cfg, n_box, rng=rng)
    if n_goal > 0:
        goal_poses = sample_observation_candidates(
            p_target, cfg, n_goal, seam_limits=sl_i, tangent=tan_i, rng=rng,
        )
        candidate_poses = (np.concatenate([box_poses, goal_poses], axis=0)
                           if len(goal_poses) else box_poses)
    else:
        candidate_poses = box_poses
    n_total = len(candidate_poses)

    # ---- 2. IK 过滤 ----
    poses_ik, thetas_ik = filter_by_ik(candidate_poses, arm_kin)
    n_after_ik = len(poses_ik)

    # ---- 3. 碰撞过滤 ----
    poses_col, thetas_col = filter_by_collision(thetas_ik, poses_ik, arm_kin)
    n_after_col = len(poses_col)

    if n_after_col == 0:
        # 没有可行候选, 直接返回失败
        return CycleResult(
            success=False,
            n_candidates_total=n_total,
            n_candidates_after_ik=n_after_ik,
            n_candidates_after_collision=0,
        )

    # ---- 4-5. 评分 + 选最优 ----
    # 目标点 wedge 中线 + 测地距离场 (复用上面算的 ti/sl_i/tan_i).
    target_bis = wedge_bisector(sl_i, tangent=tan_i) if sl_i is not None else None
    geo_field = None
    if cfg.use_geodesic_approach:
        goal_mask = build_shell_wedge_goal_mask(
            voxel_map, p_target, cfg.obs_d_min, cfg.obs_d_max,
            seam_limits=sl_i, tangent=tan_i,
            wedge_margin_deg=cfg.geo_wedge_margin_deg,
        )
        if bool(goal_mask.any()):
            geo_field = geodesic_field(voxel_map, goal_mask)

    scores, gains, traj_costs = score_candidates(
        poses_col, thetas_col, theta_now, voxel_map, p_target, arm_kin, cfg,
        target_bisector=target_bis, geo_field=geo_field,
    )

    # ---- 锁定检测 + 边碰撞检查 ----
    # 关键: 只检查候选位姿本身不撞还不够 —— theta_now→候选 的关节空间直线运动也不能
    # 穿过工件. 用 edge_free 逐点插值检查; 选"边无碰撞"的候选, 否则路径会切过工件.
    # 锁定: phase switch 在"找到合格观测"时触发 (不要求是 argmax); 目标导向候选靠此生效.
    theta_now_arr = np.asarray(theta_now, dtype=np.float64).reshape(-1)

    def _edge_ok(j: int) -> bool:
        return arm_kin.edge_free(theta_now_arr, thetas_col[j].cpu().numpy(),
                                 step_rad=cfg.edge_step_rad)

    # 锁定: 在能合格观测的候选里, 取第一个边也无碰撞的
    lock_idx = -1
    lock_observed: list[int] = []
    for j in range(len(poses_col)):
        Tj = poses_col[j]
        vpj = Viewpoint(pos=Tj[:3, 3], dir=Tj[:3, 2], up=-Tj[:3, 1])
        obs_j = detect_phase_switch(
            voxel_map, seam_points, seam_tangents, seam_limits,
            vpj, cfg, intrinsics,
        )
        if obs_j and _edge_ok(j):
            lock_idx = j
            lock_observed = obs_j
            break

    if lock_idx >= 0:
        best_idx = lock_idx
    else:
        # 按分数降序, 取第一个边无碰撞的候选 (路径不穿工件)
        best_idx = -1
        for j in np.argsort(-scores):
            if _edge_ok(int(j)):
                best_idx = int(j)
                break
        if best_idx < 0:
            # 没有任何边无碰撞的候选 → 本 cycle 无法安全移动, 视为失败 (触发回退)
            return CycleResult(
                success=False,
                n_candidates_total=n_total,
                n_candidates_after_ik=n_after_ik,
                n_candidates_after_collision=n_after_col,
            )
    best_pose = poses_col[best_idx]
    best_theta = thetas_col[best_idx].cpu().numpy()

    # ---- 6. 加节点+边 (边也要无碰撞才连, 保证 global step 的 Dijkstra 路径可执行) ----
    new_node = GraphNode(
        node_id=uuid.uuid4().hex[:12],
        theta=best_theta,
        ee_pos_world=best_pose[:3, 3],
        ee_dir_world=best_pose[:3, 2],
        cycle_added=cycle_idx,
        last_gain=float(gains[best_idx]),
    )
    graph.add_node(new_node)
    nearest = graph.nearest_node(theta_now)
    if (nearest is not None and nearest.node_id != new_node.node_id
            and arm_kin.edge_free(nearest.theta, best_theta,
                                  step_rad=cfg.edge_step_rad)):
        edge_w = float(np.linalg.norm(best_theta - nearest.theta))
        graph.add_edge(nearest.node_id, new_node.node_id, weight=edge_w)

    # ---- 7. phase switch: 锁定到的合格观测, 或在 best_pose 处复查 ----
    if lock_idx >= 0:
        observed = lock_observed
    else:
        cur_vp = Viewpoint(
            pos=best_pose[:3, 3],
            dir=best_pose[:3, 2],
            up=-best_pose[:3, 1],
        )
        observed = detect_phase_switch(
            voxel_map, seam_points, seam_tangents, seam_limits,
            cur_vp, cfg, intrinsics,
        )

    return CycleResult(
        success=True,
        next_theta=best_theta,
        next_ee_pose_world=best_pose,
        best_score=float(scores[best_idx]),
        best_gain=float(gains[best_idx]),
        best_traj_cost=float(traj_costs[best_idx]),
        best_node_id=new_node.node_id,
        n_candidates_total=n_total,
        n_candidates_after_ik=n_after_ik,
        n_candidates_after_collision=n_after_col,
        phase_switch_triggered=len(observed) > 0,
        p_seam_observed=observed,
    )


__all__ = [
    "CycleResult",
    "run_one_cycle",
    "sample_candidate_ee_poses",
    "sample_observation_candidates",
    "filter_by_ik",
    "filter_by_collision",
    "score_candidates",
    "detect_phase_switch",
    "solve_observation_pose",
    "Phase1State",
    "Phase1Result",
    "compute_camera_pose",
    "is_local_stuck",
    "dist_to_shell",
    "reevaluate_node_gains",
    "pick_frontier_node",
    "do_global_step",
    "run_phase1",
]


# ====================================================================
# M1.8: 多 cycle + 卡住回退
# ====================================================================


def compute_camera_pose(
    theta: np.ndarray,
    arm_kin: ArmKin,
) -> np.ndarray:
    """关节角 → 相机 4×4 world pose.

    约定: 相机贴 EE, 用 OpenCV 系 (z 前 = 光轴, x 右, y 下).
    R 来自 ee_quat_world.
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    theta_t = torch.tensor(theta, dtype=torch.float32,
                            device=arm_kin.tensor_args.device)
    fk = arm_kin.fk(theta_t)
    pos = fk["ee_pos_world"][0].detach().cpu().numpy()
    quat_wxyz = fk["ee_quat_world"][0].detach().cpu().numpy()

    from scipy.spatial.transform import Rotation as ScipyRot
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2],
                           quat_wxyz[3], quat_wxyz[0]])
    R = ScipyRot.from_quat(quat_xyzw).as_matrix()

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = pos
    return pose


@dataclass
class Phase1State:
    """跨 cycle 运行时状态."""

    cycle_idx: int = 0
    theta_now: Optional[np.ndarray] = None

    voxel_map: Optional[VoxelMap] = None
    graph: Optional[ExplorationGraph] = None
    arm_kin: Optional[ArmKin] = None

    seam_points: Optional[np.ndarray] = None
    seam_tangents: Optional[np.ndarray] = None
    seam_limits: Optional[np.ndarray] = None
    p_target: Optional[np.ndarray] = None

    p_observed: set[int] = field(default_factory=set)

    stuck_count: int = 0
    last_global_cycle: int = -100
    best_dist_to_shell: float = np.inf       # 已达到的"到观测壳"最短距离 (米)
    solver_calls: int = 0                    # 合格位姿求解器已调用次数

    cycle_history: list[CycleResult] = field(default_factory=list)
    global_cycle_indices: list[int] = field(default_factory=list)
    ee_pos_history: list[np.ndarray] = field(default_factory=list)
    ee_pose_history: list[np.ndarray] = field(default_factory=list)  # 每路点 4×4 相机位姿
    theta_history: list[np.ndarray] = field(default_factory=list)    # 每路点关节角 (6,)

    cfg: Optional[Phase1Config] = None
    intrinsics: Optional[CameraIntrinsics] = None


@dataclass
class Phase1Result:
    """整个 Phase 1 的最终输出."""

    success: bool
    final_state: Phase1State
    phase_switch_at: Optional[int] = None
    p_observed_first: Optional[int] = None
    total_time_s: float = 0.0
    fail_reason: Optional[str] = None


def is_local_stuck(state: Phase1State) -> bool:
    """连续 patience 次"没接近观测壳" → 卡住. 上一次 global step 后至少先做 1 次 local."""
    if state.cfg is None:
        return False
    return (state.stuck_count >= state.cfg.patience
            and (state.cycle_idx - state.last_global_cycle) > 1)


def dist_to_shell(ee_pos: np.ndarray, p_target: np.ndarray,
                  cfg: Phase1Config) -> float:
    """EE 到观测壳 [obs_d_min, obs_d_max] 的距离 (米). 壳内返回 0.

    只看到 P_target 的欧氏距离落在 [d_min, d_max] 没有 — 但作为"接近进度"的
    标量足够: 壳外越远越大, 壳内为 0.
    """
    d = float(np.linalg.norm(np.asarray(ee_pos).reshape(3)
                             - np.asarray(p_target).reshape(3)))
    if d < cfg.obs_d_min:
        return cfg.obs_d_min - d
    if d > cfg.obs_d_max:
        return d - cfg.obs_d_max
    return 0.0


def reevaluate_node_gains(
    graph: ExplorationGraph,
    voxel_map: VoxelMap,
    p_target: np.ndarray,
    cfg: Phase1Config,
) -> None:
    """对图中所有节点重算 gain (体积+目标偏置).

    Global step 触发前调用 — 地图已变, 旧的 last_gain 过时了.
    """
    gain_cfg = GainConfig(
        w_vol=cfg.w_vol, w_target=cfg.w_target,
        n_rays_h=cfg.gain_n_rays_h, n_rays_v=cfg.gain_n_rays_v,
        max_range=cfg.gain_max_range,
        fov_h_deg=cfg.gain_fov_h_deg, fov_v_deg=cfg.gain_fov_v_deg,
    )
    for node in graph.all_nodes():
        d = node.ee_dir_world
        d_norm = np.linalg.norm(d)
        d_unit = d / d_norm if d_norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        node.last_gain = float(compute_gain(
            node.ee_pos_world, d_unit, np.array([0.0, 0.0, 1.0]),
            voxel_map, p_target, gain_cfg,
        ))


def pick_frontier_node(
    graph: ExplorationGraph,
    theta_now: np.ndarray,
    cfg: Phase1Config,
) -> Optional[GraphNode]:
    """从 frontier 节点 (gain 高的) 里挑 score = gain·exp(-λ·cost) 最大的."""
    threshold = (cfg.global_frontier_gain_threshold
                 if cfg.global_frontier_gain_threshold is not None
                 else cfg.g_low * 2.0)
    fronts = graph.frontier_nodes(gain_threshold=threshold)
    if not fronts:
        return None
    cur_node = graph.nearest_node(theta_now)
    if cur_node is None:
        return None
    cost_map = graph.dijkstra_to_many(
        cur_node.node_id, [f.node_id for f in fronts],
    )
    best, best_score = None, -np.inf
    for f in fronts:
        if f.node_id not in cost_map:
            continue
        if f.node_id == cur_node.node_id:
            continue
        path = cost_map[f.node_id]
        cost = sum(np.linalg.norm(path[i].theta - path[i + 1].theta)
                   for i in range(len(path) - 1))
        score = f.last_gain * np.exp(-cfg.global_lambda_cost * cost)
        if score > best_score:
            best_score, best = score, f
    return best


def do_global_step(
    state: Phase1State,
    depth_source,
) -> bool:
    """Global step: 重评 gain → 选 frontier → Dijkstra 走过去, 沿途建图.

    返回 True 成功, False 失败 (没 frontier / 没路径).
    """
    cfg = state.cfg
    reevaluate_node_gains(state.graph, state.voxel_map, state.p_target, cfg)

    cur_node = state.graph.nearest_node(state.theta_now)
    if cur_node is None:
        return False
    f_star = pick_frontier_node(state.graph, state.theta_now, cfg)
    if f_star is None or f_star.node_id == cur_node.node_id:
        return False

    path = state.graph.dijkstra(cur_node.node_id, f_star.node_id)
    if path is None or len(path) < 2:
        return False

    # 沿 path 走, 每个节点拍一帧
    for next_node in path[1:]:
        cam_pose = compute_camera_pose(next_node.theta, state.arm_kin)
        depth = depth_source.render(cam_pose)
        state.voxel_map.integrate(depth, state.intrinsics, cam_pose)
        state.theta_now = next_node.theta.copy()
        state.ee_pos_history.append(cam_pose[:3, 3].copy())
        state.ee_pose_history.append(cam_pose.copy())
        state.theta_history.append(next_node.theta.copy())
    return True


def run_phase1(
    theta_init: np.ndarray,
    seam_points: np.ndarray,
    seam_tangents: np.ndarray,
    seam_limits: Optional[np.ndarray],
    p_target: np.ndarray,
    arm_kin: ArmKin,
    depth_source,
    intrinsics: CameraIntrinsics,
    voxel_map: VoxelMap,
    cfg: Optional[Phase1Config] = None,
    rng: Optional[np.random.Generator] = None,
) -> Phase1Result:
    """Phase 1 主入口: 多 cycle loop + Global 回退.

    Args:
        theta_init: 初始关节角 (= INITIAL_JOINT_ANGLES, 不是 pkl 的 joint_angles)
        seam_points / seam_tangents / seam_limits: pkl 数据
        p_target: 目标点 (一般 = seam_points 中点)
        arm_kin: cuRobo IK + collision
        depth_source: 任何带 .render(camera_pose: (4,4)) → (H,W) 的对象
        intrinsics: 相机内参
        voxel_map: 调用方建好 (定 bounds + resolution 的责任在外面)
        cfg: 超参
        rng: 控制 sampling 随机性
    """
    import time
    if cfg is None:
        cfg = Phase1Config()
    if rng is None:
        rng = np.random.default_rng()

    t_start = time.time()

    # ---- 初始化 state + 加 init 节点 ----
    theta_init = np.asarray(theta_init, dtype=np.float64).reshape(-1)
    cam_pose_init = compute_camera_pose(theta_init, arm_kin)
    state = Phase1State(
        theta_now=theta_init.copy(),
        voxel_map=voxel_map,
        graph=ExplorationGraph(),
        arm_kin=arm_kin,
        seam_points=seam_points,
        seam_tangents=seam_tangents,
        seam_limits=seam_limits,
        p_target=np.asarray(p_target, dtype=np.float64),
        cfg=cfg,
        intrinsics=intrinsics,
    )
    init_node = GraphNode(
        node_id="init",
        theta=theta_init.copy(),
        ee_pos_world=cam_pose_init[:3, 3].copy(),
        ee_dir_world=cam_pose_init[:3, 2].copy(),
        cycle_added=0,
    )
    state.graph.add_node(init_node)
    state.ee_pos_history.append(cam_pose_init[:3, 3].copy())
    state.ee_pose_history.append(cam_pose_init.copy())
    state.theta_history.append(theta_init.copy())
    state.best_dist_to_shell = dist_to_shell(
        cam_pose_init[:3, 3], state.p_target, cfg,
    )

    # ---- 主循环 ----
    while state.cycle_idx < cfg.max_cycles:
        # 1) 拍 depth + 建图
        cam_pose = compute_camera_pose(state.theta_now, arm_kin)
        depth = depth_source.render(cam_pose)
        state.voxel_map.integrate(depth, intrinsics, cam_pose)

        # 2) phase switch?
        cur_vp = Viewpoint(
            pos=cam_pose[:3, 3],
            dir=cam_pose[:3, 2],
            up=-cam_pose[:3, 1],
        )
        observed_now = detect_phase_switch(
            state.voxel_map, seam_points, seam_tangents, seam_limits,
            cur_vp, cfg, intrinsics,
        )
        for o in observed_now:
            state.p_observed.add(int(o))
        if state.p_observed:
            return Phase1Result(
                success=True,
                final_state=state,
                phase_switch_at=state.cycle_idx,
                p_observed_first=min(state.p_observed),
                total_time_s=time.time() - t_start,
            )

        # 3) local 还是 global?
        if is_local_stuck(state):
            # 3a) 先试专门的合格位姿求解器 (针对针尖可行域焊缝). 找到即锁定成功.
            if cfg.use_pose_solver and state.solver_calls < cfg.solver_max_calls:
                state.solver_calls += 1
                sol = solve_observation_pose(
                    arm_kin, state.voxel_map, seam_points, seam_tangents,
                    seam_limits, state.p_target, cfg, intrinsics,
                    rng=rng, budget=cfg.solver_budget,
                )
                if sol is not None:
                    # 求解器找到几何可观测的可达位姿: 移过去 (主循环下一 cycle 会拍一帧
                    # 把 LOS 上的 unknown 扫成 free, 再用严格判定确认 → 触发成功).
                    sol_theta, sol_pose, _sol_obs = sol
                    state.theta_now = sol_theta.copy()
                    state.ee_pos_history.append(sol_pose[:3, 3].copy())
                    state.ee_pose_history.append(sol_pose.copy())
                    state.theta_history.append(sol_theta.copy())
                    state.stuck_count = 0
                    state.last_global_cycle = state.cycle_idx
                    state.cycle_idx += 1
                    continue
            # 3b) 求解器没找到 → GBPlanner2 风格 global step 回退
            ok = do_global_step(state, depth_source)
            if not ok:
                return Phase1Result(
                    success=False,
                    final_state=state,
                    fail_reason="no_frontier_for_global_step",
                    total_time_s=time.time() - t_start,
                )
            state.global_cycle_indices.append(state.cycle_idx)
            state.last_global_cycle = state.cycle_idx
            state.stuck_count = 0
        else:
            cyc = run_one_cycle(
                theta_now=state.theta_now,
                voxel_map=state.voxel_map,
                graph=state.graph,
                arm_kin=arm_kin,
                seam_points=seam_points,
                seam_tangents=seam_tangents,
                seam_limits=seam_limits,
                p_target=state.p_target,
                cycle_idx=state.cycle_idx,
                cfg=cfg,
                intrinsics=intrinsics,
                rng=rng,
            )
            state.cycle_history.append(cyc)
            if not cyc.success:
                # 没有 IK+碰撞通过的候选 → 卡住
                state.stuck_count += 1
            else:
                # 贪心执行最佳候选 (标准 NBV: 总是走向当前最优 viewpoint).
                # 是否"卡住"看接近观测壳的进度, 不看绝对分数 —— approach 项
                # 让绝对分数带大负偏置, 绝对阈值会误判.
                state.theta_now = cyc.next_theta.copy()
                ee_new = cyc.next_ee_pose_world[:3, 3]
                state.ee_pos_history.append(ee_new.copy())
                state.ee_pose_history.append(cyc.next_ee_pose_world.copy())
                state.theta_history.append(cyc.next_theta.copy())
                d_shell = dist_to_shell(ee_new, state.p_target, cfg)
                if d_shell <= cfg.progress_eps:
                    # 已在观测壳内: 处于正确区域, 继续本地精修 (每 cycle 扫图清 LOS
                    # + 微调位姿对准 wedge), 不要触发 global 把自己弹走 (否则在壳边
                    # 反复横跳, 永远扫不通 unknown 视线). 靠 max_cycles 兜底.
                    state.stuck_count = 0
                    state.best_dist_to_shell = 0.0
                elif d_shell < state.best_dist_to_shell - cfg.progress_eps:
                    state.best_dist_to_shell = d_shell
                    state.stuck_count = 0
                else:
                    state.stuck_count += 1

        state.cycle_idx += 1

    return Phase1Result(
        success=False,
        final_state=state,
        fail_reason=f"max_cycles_reached ({cfg.max_cycles})",
        total_time_s=time.time() - t_start,
    )
