"""STOMP 规划后端：封装参考项目 gt_overall/stomp_planning_api（与 curobo_iface 并列）。

当 configs/default.yaml 的 planner.backend == 'stomp' 时，obstacle_placement 的
「默认轨迹 / 绕行解」改走这里；碰撞判定(check_state)、扫掠 FK、三条件验证逻辑均不变。

设计要点（见对话确认）：
- 固定朝向：STOMP 的 IK 解【全位姿】(pos+quat)，不支持放开末端 roll（cuRobo free_pose_metric 才支持）；
  obstacle_placement 传进来的 metric 在 STOMP 分支被忽略。
- buffer 不重复叠加：STOMP 自身 buffer≈0.001（activation_distance）；间隙完全来自障碍物理膨胀
  （obstacle_buffer_m，工件不膨胀）。两后端在【同一个 world】里规划——默认=仅工件、绕行=障碍已膨胀——
  只换规划器。
- 目标位姿就是 seam_ee_pose 返回的 (pos, quat) 元组：stomp_planning_api._normalize_pose 原生支持。
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional

import numpy as np


def _ensure_api(cfg):
    """把 gt_overall 目录注入 sys.path 并 import stomp_planning_api（懒加载，避免无 STOMP 时报错）。

    目录优先级：环境变量 GT_OVERALL_DIR > config.stomp_params['gt_overall_dir']。
    """
    d = os.environ.get("GT_OVERALL_DIR") or cfg.stomp_params["gt_overall_dir"]
    if d and d not in sys.path:
        sys.path.insert(0, d)
    import stomp_planning_api  # noqa: E402
    return stomp_planning_api


def _plan_kwargs(cfg, buffer, checker_type):
    """组装透传给 stomp_planning_api.plan_to_pose_* 的公共参数。"""
    sp = cfg.stomp_params
    if buffer is None:
        buffer = float(sp["buffer_m"])
    return dict(
        robot_yml=cfg.robot_cfg_path, checker_type=checker_type, buffer=float(buffer),
        num_iterations=int(sp["num_iterations"]), num_batch=int(sp["num_batch"]),
        num_timesteps=int(sp["num_timesteps"]), delta_t=float(sp["delta_t"]),
        collision_weight=float(sp["collision_weight"]),
    )


def plan_pose_single(cfg, world, cur_cfg, goal_pose, *, buffer: Optional[float] = None,
                     checker_type: Any = None) -> Optional[np.ndarray]:
    """用 STOMP 规划 cur_cfg -> goal_pose，返回【单条最优】避障轨迹（内部先 IK 解目标关节角）。

    调 stomp_planning_api.plan_to_pose_single：在所有「无碰撞(n_collision_steps==0)且在限位
    (in_limit)」的候选里取 state_cost 最小那条；无合格候选返回 None（合格判定全在 API 内部，
    本层不再解析 info）。

    入参
    ----
    cfg          : gt_gen.config.Config（读 robot_cfg_path 与 stomp_params）。
    world        : cuRobo WorldConfig（默认轨迹=仅工件；绕行=障碍已膨胀、工件不膨胀）。
    cur_cfg      : 起点关节角（list/序列，长度=dof）。
    goal_pose    : (pos[3], quat_wxyz[4]) 元组 或 [x,y,z,qw,qx,qy,qz]（base 系）。
    buffer       : STOMP 碰撞激活距离(m)；None 时取 cfg.stomp_params['buffer_m']（≈0.001）。
    checker_type : cuRobo CollisionCheckerType；None 时由 stomp_planning_api 按 world 内容自动推断。

    返回：(T,dof) ndarray 或 None。
    """
    api = _ensure_api(cfg)
    traj = api.plan_to_pose_single(
        list(map(float, cur_cfg)), goal_pose, world,
        **_plan_kwargs(cfg, buffer, checker_type))
    return None if traj is None else np.asarray(traj, float)


def plan_pose_multi(cfg, world, cur_cfg, goal_pose, *, buffer: Optional[float] = None,
                    checker_type: Any = None, ik_position_threshold: Optional[float] = None,
                    ik_rotation_threshold: Optional[float] = None) -> list:
    """用 STOMP 规划 cur_cfg -> goal_pose，返回【所有合格】避障轨迹（按 state_cost 升序）。

    调 stomp_planning_api.plan_to_pose_multi：返回所有「无碰撞且在限位」的候选轨迹；这些候选
    都指向同一个 IK 目标关节角（plan_pose 先解一次 IK 再批量规划），差在中段绕障路线、终点相同。
    去重（按路线偏差挑互不相同的若干条）由调用方 obstacle_placement.detour_exists 负责。

    ik_position_threshold / ik_rotation_threshold：透传给 STOMP 内部 cuRobo IKSolver 的位置/朝向
    收敛门限（None 时用 stomp_planning_api 默认 0.005/0.05）。这两个参数仅 detour_exists 的绕行
    规划会传入（来自 cfg.obstacle_placement 的 detour_ik_*_threshold），plan_pose_single 不受影响。

    入参其余同 plan_pose_single。返回：list[(T,dof) ndarray]（无合格候选时为 []）。
    """
    api = _ensure_api(cfg)
    kw = _plan_kwargs(cfg, buffer, checker_type)
    if ik_position_threshold is not None:
        kw["ik_position_threshold"] = float(ik_position_threshold)
    if ik_rotation_threshold is not None:
        kw["ik_rotation_threshold"] = float(ik_rotation_threshold)
    trajs = api.plan_to_pose_multi(
        list(map(float, cur_cfg)), goal_pose, world, **kw)
    return [np.asarray(t, float) for t in trajs] if trajs else []
