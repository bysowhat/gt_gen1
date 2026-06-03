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
) -> CuroboHandle:
    """初始化 MotionGen + warmup。

    默认：VOXEL 全自由世界（roi_dims/roi_center 占位，Step 2 精化）。
    可传 world_model（如含 obj mesh 的 WorldConfig）+ collision_checker_type 覆盖。
    drop_collision_links：从碰撞检查中剔除的 link（如焊枪 xiaoyu_accessory_link，
        允许其接触工件——焊接接触是预期的，机械臂本体仍避障）。
    position_threshold：位置收敛门限（米，默认 5mm）。
    rotation_threshold：朝向收敛门限（四元数测度，默认 0.05；越大越松）。
        注意 cuRobo 在 position_threshold<=1mm 时会自动收紧，别设太小。
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
    ik_cfg = IKSolverConfig.load_from_robot_config(
        robot_cfg, None, num_seeds=50,
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


def solve_ik(handle: CuroboHandle, pose):
    """目标位姿 -> 关节构型（多种子 IKSolver）。返回 cuRobo IKResult。"""
    return handle.ik.solve_batch(_to_pose(handle, pose))


def ik_best_config(handle: CuroboHandle, ik_result):
    """从 IKResult 取位置误差最小的关节解；不成功返回 None。"""
    if not bool(ik_result.success.any().item()):
        return None
    dof = len(handle.joint_names)
    sol = ik_result.solution.reshape(-1, dof)
    errs = ik_result.position_error.reshape(-1)
    return sol[int(errs.argmin())].detach().cpu().numpy().tolist()


def plan_to_pose(handle: CuroboHandle, start_cfg, goal_pose, max_attempts: int = 5):
    """规划 start_cfg -> goal_pose：先 IK 成 goal_cfg，再关节空间规划（更稳）。

    返回 MotionGenResult；IK 失败返回 None。
    """
    goal_cfg = ik_best_config(handle, solve_ik(handle, goal_pose))
    if goal_cfg is None:
        return None
    return plan_to_config(handle, start_cfg, goal_cfg, max_attempts=max_attempts)


def plan_to_config(handle: CuroboHandle, start_cfg, goal_cfg, max_attempts: int = 5):
    """关节空间到关节空间的无碰撞规划。返回 MotionGenResult。"""
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
    return handle.mg.plan_single_js(
        _js(handle, start_cfg),
        _js(handle, goal_cfg),
        MotionGenPlanConfig(max_attempts=max_attempts),
    )


def plan_on_truth(handle, start_cfg, goal_cfg, truth_scene):
    """在真值场景上规划全知最优路径 P*。见 Step 7。"""
    raise NotImplementedError("Step 7")
