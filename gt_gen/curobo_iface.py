"""Step 1: cuRobo 封装（IK + 规划）。

见 docs/gt-generation-curobo-implementation.md §3、§5。

约定：
- 关节构型 cfg 用 python list / 1D 序列（长度 = len(joint_names)）。
- 位姿 pose 用 (pos[3], quat_wxyz[4]) 元组，或直接传 cuRobo Pose。
- 碰撞世界用 VOXEL 类型；init 时为"全自由"（ESDF=-max），Step 5 再灌障碍。
- 机器人 kinematics 额外暴露 Link6（供 Step 3 相机 FK 复用）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import gt_gen.compat  # noqa: F401  warp shim，须在 import curobo 前


@dataclass
class CuroboHandle:
    mg: Any                 # MotionGen
    ik: Any                 # 专用多种子 IKSolver（关节限位+自碰撞，准）
    ta: Any                 # TensorDeviceType
    config: Any             # gt_gen.config.Config
    voxel: dict             # {name, dims, pose, voxel_size}

    @property
    def joint_names(self):
        return self.config.joint_names


def _to_pose(handle: CuroboHandle, pose):
    """把 (pos, quat_wxyz) 或 Pose 统一成 cuRobo Pose（batch=1）。"""
    from curobo.types.math import Pose
    if isinstance(pose, Pose):
        return pose
    pos, quat = pose
    return Pose(
        position=handle.ta.to_device(list(pos)).view(1, 3),
        quaternion=handle.ta.to_device(list(quat)).view(1, 4),
    )


def _js(handle: CuroboHandle, cfg: Sequence[float]):
    from curobo.types.state import JointState
    return JointState.from_position(
        handle.ta.to_device(list(cfg)).view(1, -1),
        joint_names=list(handle.joint_names),
    )


def init_curobo(
    config,
    roi_dims: Optional[Sequence[float]] = None,
    roi_center: Optional[Sequence[float]] = None,
    interpolation_dt: float = 0.02,
    world_model: Optional[Any] = None,
    collision_checker_type: Optional[Any] = None,
    drop_collision_links: Optional[Sequence[str]] = None,
    position_threshold: float = 0.005,
    rotation_threshold: float = 0.05,
    num_seeds: Optional[int] = None,
) -> CuroboHandle:
    """初始化 MotionGen + warmup。

    默认：VOXEL 全自由世界（roi_dims/roi_center 占位，Step 2 精化）。
    可传 world_model（如含 obj mesh 的 WorldConfig）+ collision_checker_type 覆盖。
    drop_collision_links：从碰撞检查中剔除的 link（如焊枪 xiaoyu_accessory_link，
        允许其接触工件——焊接接触是预期的，机械臂本体仍避障）。
    position_threshold：位置收敛门限（米，默认 5mm）。
    rotation_threshold：朝向收敛门限（四元数测度，默认 0.05；越大越松）。
        注意 cuRobo 在 position_threshold<=1mm 时会自动收紧，别设太小。
    num_seeds：IK 并行优化的随机起点数；None 时取 config.ik_num_seeds（default.yaml）。
    """
    from curobo.types.base import TensorDeviceType
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.util_file import load_yaml
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    gt_gen.compat.apply_trimesh_shim()  # 修 curobo geom/types.py 缺 trimesh 导入

    ta = TensorDeviceType()
    rd = load_yaml(config.robot_cfg_path)
    kin = rd["robot_cfg"]["kinematics"]
    kin["link_names"] = ["Link6"]  # 供相机 FK

    if drop_collision_links:
        drop = set(drop_collision_links)
        kin["collision_link_names"] = [n for n in kin["collision_link_names"] if n not in drop]
        if isinstance(kin.get("collision_spheres"), dict):
            for n in list(kin["collision_spheres"]):
                if n in drop:
                    kin["collision_spheres"].pop(n)
        sci = kin.get("self_collision_ignore")
        if isinstance(sci, dict):
            for n in list(sci):
                if n in drop:
                    sci.pop(n)
                else:
                    sci[n] = [x for x in sci[n] if x not in drop]
        scb = kin.get("self_collision_buffer")
        if isinstance(scb, dict):
            for n in list(scb):
                if n in drop:
                    scb.pop(n)

    robot_cfg = rd["robot_cfg"]

    vs = float(config.voxel_size_m)
    dims = list(roi_dims) if roi_dims is not None else [3.0, 3.0, 3.0]
    center = list(roi_center) if roi_center is not None else [0.0, 0.0, 1.0]
    pose = center + [1.0, 0.0, 0.0, 0.0]

    if world_model is not None:
        checker = collision_checker_type or CollisionCheckerType.MESH
    else:
        world_model = {"voxel": {"world": {"dims": dims, "pose": pose, "voxel_size": vs}}}
        checker = collision_checker_type or CollisionCheckerType.VOXEL

    mg_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_model,
        ta,
        collision_checker_type=checker,
        interpolation_dt=interpolation_dt,
        position_threshold=position_threshold,
        rotation_threshold=rotation_threshold,
    )
    mg = MotionGen(mg_cfg)
    mg.warmup(warmup_js_trajopt=False)

    # 专用 IKSolver：MotionGen 内置 IK 对远目标收敛差，这里用多种子、无世界
    # （世界碰撞交给规划/扫掠检查），关 cuda_graph 避免 solve/solve_batch 冲突。
    # num_seeds=100：每个 IK 问题并行优化的随机起点数（cuRobo 默认 100）。越大越稳但越慢；
    #   IK 跨调用随机撒种，边缘目标的"过阈值解数"会抖动，提高 num_seeds 可缓解。
    #   plan_to_pose 的 num_solutions(=return_seeds) 须 ≤ num_seeds。详见
    #   docs/gt-generation-curobo-implementation.md「IK 多种子与多解轮询」。
    if num_seeds is None:
        num_seeds = config.ik_num_seeds
    ik_cfg = IKSolverConfig.load_from_robot_config(
        robot_cfg, None, num_seeds=num_seeds,
        self_collision_check=True, self_collision_opt=True,
        use_cuda_graph=False, tensor_args=ta,
        position_threshold=position_threshold,
        rotation_threshold=rotation_threshold,
    )
    ik = IKSolver(ik_cfg)

    return CuroboHandle(
        mg=mg, ik=ik, ta=ta, config=config,
        voxel={"name": "world", "dims": dims, "pose": pose, "voxel_size": vs},
    )


def fk(handle: CuroboHandle, cfg: Sequence[float]):
    """正运动学：返回 (ee_pos, ee_quat_wxyz, link_pose_dict)。"""
    st = handle.mg.kinematics.get_state(handle.ta.to_device(list(cfg)).view(1, -1))
    ee_pos = st.ee_position[0].detach().cpu().numpy()
    ee_quat = st.ee_quaternion[0].detach().cpu().numpy()
    return ee_pos, ee_quat, st.link_pose


def free_pose_metric(handle: CuroboHandle, free_rot=(), free_pos=()):
    """构造放开指定轴的 PoseCostMetric（在目标位姿坐标系下）。

    轴索引：旋转 0=x,1=y,2=z；平移 0=x,1=y,2=z。
    hold_vec_weight=[rx,ry,rz, x,y,z]，置 0 = 不约束该轴。
    放开末端 x 轴自转(roll)：free_rot=(0,)。
    """
    from curobo.rollout.cost.pose_cost import PoseCostMetric
    w = handle.ta.to_device([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    for a in free_rot:
        w[a] = 0.0
    for a in free_pos:
        w[3 + a] = 0.0
    return PoseCostMetric(hold_partial_pose=True, hold_vec_weight=w)


def _reset_ik_metric(handle: CuroboHandle):
    from curobo.rollout.cost.pose_cost import PoseCostMetric
    handle.ik.update_pose_cost_metric(PoseCostMetric(hold_partial_pose=False))


def solve_ik(handle: CuroboHandle, pose, pose_cost_metric=None, return_seeds: int = 1):
    """目标位姿 -> 关节构型（多种子 IKSolver）。返回 cuRobo IKResult。

    pose_cost_metric：可选 PoseCostMetric（如 free_pose_metric 放开 roll）。
        IKSolver 的 metric 是有状态的，这里用完即复位，避免污染后续调用。
    return_seeds：每个问题返回多少个解（默认 1）。>1 时 IKResult 含多组解，
        供 ik_configs / plan_to_pose 轮询（不同 IK 分支）。
    """
    if pose_cost_metric is not None:
        handle.ik.update_pose_cost_metric(pose_cost_metric)
    try:
        return handle.ik.solve_batch(_to_pose(handle, pose), return_seeds=return_seeds)
    finally:
        if pose_cost_metric is not None:
            _reset_ik_metric(handle)


def ik_configs(handle: CuroboHandle, ik_result):
    """从 IKResult 取所有【成功】解，按位置误差升序，返回 [(cfg_list, pos_err), ...]。"""
    if not bool(ik_result.success.any().item()):
        return []
    dof = len(handle.joint_names)
    sol = ik_result.solution.reshape(-1, dof)
    errs = ik_result.position_error.reshape(-1)
    succ = ik_result.success.reshape(-1)
    out = []
    for i in range(sol.shape[0]):
        if bool(succ[i].item()):
            out.append((sol[i].detach().cpu().numpy().tolist(), float(errs[i].item())))
    out.sort(key=lambda x: x[1])
    return out


def ik_best_config(handle: CuroboHandle, ik_result):
    """从 IKResult 取位置误差最小的成功解；不成功返回 None。"""
    cs = ik_configs(handle, ik_result)
    return cs[0][0] if cs else None


def plan_to_pose(handle: CuroboHandle, start_cfg, goal_pose, max_attempts: int = 5,
                 pose_cost_metric=None, num_solutions: Optional[int] = None):
    """规划 start_cfg -> goal_pose：IK 求【多个】解，按误差升序逐个尝试关节规划，
    第一个成功的即返回（避免"只挑一个 IK 解、恰好不可达/碰撞"的问题）。

    pose_cost_metric：可选，传给 IK 放开某些轴（如 free_pose_metric(free_rot=(0,))）。
    num_solutions：最多尝试多少个 IK 解；None 时取 config.ik_return_seeds（default.yaml）。
    返回：成功的 MotionGenResult；全失败返回最后一次的 MotionGenResult（便于诊断）；
        IK 完全无解返回 None。
    """
    if num_solutions is None:
        num_solutions = handle.config.ik_return_seeds
    res_ik = solve_ik(handle, goal_pose, pose_cost_metric=pose_cost_metric,
                      return_seeds=num_solutions)
    cands = ik_configs(handle, res_ik)
    if not cands:
        return None
    last = None
    for goal_cfg, _err in cands:
        last = plan_to_config(handle, start_cfg, goal_cfg, max_attempts=max_attempts)
        if last is not None and bool(last.success.item()):
            return last
    return last  # 全失败，返回最后一次结果供 explain 诊断


def plan_to_pose2(handle: CuroboHandle, start_cfg, goal_pose, max_attempts: int = 5,
                  pose_cost_metric=None):
    """直接用 cuRobo `mg.plan_single(末端位姿)` 规划，**不预先 IK 选构型**。

    cuRobo 内部自管多种子 IK + 轨迹优化，联合挑选终点构型（理论上比"先 IK 选一个解"
    更优）。但注意两点局限（已实测）：
    1. MotionGen 内置 IK 对远离 retract 的目标收敛差，常 `IK_FAIL`——这正是 plan_to_pose
       当初改走自建 IKSolver 的原因；
    2. pose_cost_metric 在 MotionGen 里是【路径约束】语义（hold_partial_pose=保持指定轴
       在起点→终点间不变），要求 start/goal 在被约束轴上一致，"只放开目标 roll"的用法
       通常会被拒绝（update_pose_cost_metric 返回 False）。

    返回 MotionGenResult（失败看 .status）。
    """
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
    from curobo.rollout.cost.pose_cost import PoseCostMetric
    start = _js(handle, start_cfg)
    goal = _to_pose(handle, goal_pose)
    applied = False
    if pose_cost_metric is not None:
        applied = bool(handle.mg.update_pose_cost_metric(
            pose_cost_metric, start_state=start, goal_pose=goal))
    try:
        return handle.mg.plan_single(start, goal, MotionGenPlanConfig(max_attempts=max_attempts))
    finally:
        if applied:
            handle.mg.update_pose_cost_metric(PoseCostMetric(hold_partial_pose=False))


def plan_to_config(handle: CuroboHandle, start_cfg, goal_cfg, max_attempts: int = 5):
    """关节空间到关节空间的无碰撞规划。返回 MotionGenResult。"""
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
    return handle.mg.plan_single_js(
        _js(handle, start_cfg),
        _js(handle, goal_cfg),
        MotionGenPlanConfig(max_attempts=max_attempts),
    )


def check_state(handle: CuroboHandle, cfg):
    """检查单个构型是否可行（不在碰撞/限位/自碰撞）。

    用 MotionGen 内部带世界的检查器，与规划器判定一致。
    返回 (feasible: bool, constraint: float)；constraint>0 表示违反约束。
    """
    m = handle.mg.check_constraints(_js(handle, cfg))
    feasible = bool(m.feasible.view(-1)[0].item())
    constraint = float(m.constraint.view(-1)[0].item()) if m.constraint is not None else float("nan")
    return feasible, constraint


def explain_endpoints(handle: CuroboHandle, start_cfg, goal_cfg):
    """分别检查起点/终点合法性，返回明确字符串（替代含糊的 'Start or End ...'）。"""
    sf, sc = check_state(handle, start_cfg)
    gf, gc = check_state(handle, goal_cfg)
    parts = [f"起点(start) {'OK' if sf else f'在碰撞/违约 constraint={sc:.3f}'}",
             f"终点(end) {'OK' if gf else f'在碰撞/违约 constraint={gc:.3f}'}"]
    if sf and gf:
        parts.append("两端都合法 → 失败在中间路径(找不到无碰撞连线)")
    return " | ".join(parts)


def plan_on_truth(handle, start_cfg, goal_cfg, truth_scene):
    """在真值场景上规划全知最优路径 P*。见 Step 7。"""
    raise NotImplementedError("Step 7")
