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
from phase1.graph import ExplorationGraph, GraphNode
from phase1.mapping import VoxelMap
from phase1.observation_check import Viewpoint, is_valid_observation


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """为每个候选 (pose, theta) 算 score.

    score = w_vol·vol_gain + w_target·target_bias - λ·trajectory_cost
          (gain 部分 = M1.6.gain, traj_cost = 关节空间 L2 距离)

    返回 (scores, gains, traj_costs), 都是 (K,) numpy.
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

    gains = np.zeros(K, dtype=np.float64)
    for i in range(K):
        T = poses[i]
        cam_pos = T[:3, 3]
        cam_dir = T[:3, 2]
        cam_up = -T[:3, 1]                # OpenCV: y 朝下 → up = -y
        gains[i] = compute_gain(cam_pos, cam_dir, cam_up,
                                 voxel_map, p_target, gain_cfg)

    # 轨迹代价 (关节空间 L2)
    theta_now_t = torch.tensor(theta_now, dtype=torch.float32,
                                device=arm_kin.tensor_args.device).unsqueeze(0)
    theta_now_b = theta_now_t.expand(K, -1)
    traj_costs = arm_kin.trajectory_cost(theta_now_b, thetas).cpu().numpy()

    scores = gains - cfg.lambda_cost * traj_costs
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

    # ---- 1. 采样 ----
    candidate_poses = sample_candidate_ee_poses(
        ee_pos_now, p_target, cfg, cfg.n_samples, rng=rng,
    )
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
    scores, gains, traj_costs = score_candidates(
        poses_col, thetas_col, theta_now, voxel_map, p_target, arm_kin, cfg,
    )
    best_idx = int(np.argmax(scores))
    best_pose = poses_col[best_idx]
    best_theta = thetas_col[best_idx].cpu().numpy()

    # ---- 6. 加节点+边 ----
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
    if nearest is not None and nearest.node_id != new_node.node_id:
        edge_w = float(np.linalg.norm(best_theta - nearest.theta))
        graph.add_edge(nearest.node_id, new_node.node_id, weight=edge_w)

    # ---- 7. phase switch 检测: 在 best_pose 处会看到哪些 P_i? ----
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
    "filter_by_ik",
    "filter_by_collision",
    "score_candidates",
    "detect_phase_switch",
]
