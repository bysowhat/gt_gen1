"""Step 1: cuRobo 封装（IK + 规划）。

见 docs/gt-generation-curobo-implementation.md §3、§5。
"""
from __future__ import annotations


def init_curobo(config):
    """初始化 MotionGen（VOXEL 碰撞世界）+ warmup，返回封装句柄。"""
    raise NotImplementedError("Step 1")


def solve_ik(handle, pose):
    """目标位姿 -> 关节构型（碰撞感知 IK）。"""
    raise NotImplementedError("Step 1")


def plan_to_pose(handle, start_cfg, goal_pose):
    """在当前（保守）碰撞世界里规划 start_cfg -> goal_pose 的无碰撞轨迹。"""
    raise NotImplementedError("Step 1")


def plan_to_config(handle, start_cfg, goal_cfg):
    """关节空间到关节空间的无碰撞规划。"""
    raise NotImplementedError("Step 1")


def plan_on_truth(handle, start_cfg, goal_cfg, truth_scene):
    """在【真值场景】上规划全知最优路径 P*（特权，仅用于指导探索）。见 Step 7。"""
    raise NotImplementedError("Step 7")
