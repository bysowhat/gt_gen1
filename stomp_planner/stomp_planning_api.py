"""对外暴露本项目「带障碍物 STOMP 路径规划」能力的稳定 API（供别的项目调用）。

本模块把 plan_path_stomp.py / plan_path_stomp_obstacle.py 里的核心规划逻辑
封装成几个即拿即用的函数 + 一个可复用的 StompPlanner 类：

    函数1  plan_to_joint_config(cur_cfg, target_cfg, world, ...)
            输入：当前关节角、目标关节角、cuRobo 碰撞世界
            输出：(trajs, infos) —— 所有 batch 候选轨迹 (B,T,6) 及各自诊断

    函数1-single  plan_to_joint_single(cur_cfg, target_cfg, world, ...)
            在满足「无碰撞 + 在限位内」的候选里取 state_cost 最小的一条 traj (T,6)；
            不返回 info；若无满足者返回 None。

    函数1-multi   plan_to_joint_multi(cur_cfg, target_cfg, world, ...)
            返回所有满足「无碰撞 + 在限位内」的 traj 列表（每条 (T,6)）；
            不返回 info；若无满足者返回 None。

    函数2-single  plan_to_pose_single(cur_cfg, target_pose, world, ...)
            在满足「无碰撞 + 在限位内」的候选里取 state_cost 最小的一条 traj (T,6)；
            不返回 info；若无满足者返回 None。

    函数2-multi   plan_to_pose_multi(cur_cfg, target_pose, world, ...)
            返回所有满足「无碰撞 + 在限位内」的 traj 列表（每条 (T,6)）；
            不返回 info；若无满足者返回 None。
    函数2 内部先用 cuRobo IKSolver 解出避障+不自碰的目标关节角，再调函数1。

碰撞世界 `world` 的类型 = cuRobo `curobo.geom.types.WorldConfig`
    —— 这是 cuRobo 通用的碰撞世界表示。你可以：
      (a) 自己用 WorldConfig(cuboid=[...], cylinder=[...], mesh=[...]) 直接构造；
      (b) 用本模块的 build_world_from_obstacles(...) 从 gt_gen_hanfeng 原语+工件 mesh 构造；
      (c) 若你手里是 gt_gen 的 h_expl(CuroboHandle) 这类封装，其底层世界即一个 WorldConfig，
          把那个 WorldConfig 传进来即可（见 build_world_from_obstacles / 你自己的建场代码）。
    注意：world 内所有位姿、以及 plan_to_pose 的 target_pose，都在**机器人 base 系**下，
    四元数为 wxyz —— 与 gt_gen_hanfeng / cuRobo 约定一致。

依赖：cuRobo（env_isaaclab 已装，RobotWorld/IKSolver 仅 CUDA）；
      gt_gen_hanfeng 仅用于其 trimesh shim 与可选的障碍物原语（缺失时本模块仍可用纯 cuRobo 世界）。

最小用法见同目录 example_use_planner.py。
"""

import os
import sys

import numpy as np
import torch

# 复用本项目已实现并验证过的规划核心（避免重复造轮子）
_PROJ_DIR = os.path.dirname(os.path.realpath(__file__))
if _PROJ_DIR not in sys.path:
    sys.path.insert(0, _PROJ_DIR)

from plan_path_stomp import load_robot_cfg, SEED            # noqa: E402
from plan_path_stomp_obstacle import (                      # noqa: E402
    CollisionOracle, ObstacleStomp, build_world, DEFAULT_OBSTACLES,
)

torch.set_default_dtype(torch.float)

# 与本项目其它脚本一致的机器人配置（关节名/限位/retract 从该 yml + 其 URDF 读）
DEFAULT_ROBOT_YML = "/home/a/Datas/curobo/example_new_robot/urdf_12e/ur12e.yml"


# --------------------------------------------------------------------------- #
# 辅助：trimesh shim（curobo geom/types.py 缺 trimesh 导入；gt_gen.compat 修它）
# --------------------------------------------------------------------------- #
def _apply_shim():
    try:
        import gt_gen.compat  # noqa: F401
        gt_gen.compat.apply_trimesh_shim()
    except Exception:  # noqa: BLE001  gt_gen 不可用时忽略（多数 curobo 版本无需 shim）
        pass


def _infer_checker_type(world):
    """根据 WorldConfig 内容推断碰撞检查器类型：有 mesh 用 MESH，否则 PRIMITIVE。"""
    from curobo.geom.sdf.world import CollisionCheckerType
    meshes = getattr(world, "mesh", None) or []
    return CollisionCheckerType.MESH if len(meshes) > 0 else CollisionCheckerType.PRIMITIVE


# --------------------------------------------------------------------------- #
# 碰撞世界构造辅助（可选；你也可以自己构造 WorldConfig 后直接传入）
# --------------------------------------------------------------------------- #
def build_world_from_obstacles(workpiece_mesh=None, workpiece_pose=None,
                               obstacle_spec=None, use_obstacles=True):
    """用 gt_gen_hanfeng 原语 + 可选工件 mesh 构造 cuRobo (world_config, checker_type)。

    直接复用 plan_path_stomp_obstacle.build_world，障碍与本项目规划/可视化完全同源。
    obstacle_spec=None 时用 DEFAULT_OBSTACLES（一块挡路的 box_beam）。
    """
    spec = obstacle_spec if obstacle_spec is not None else DEFAULT_OBSTACLES
    return build_world(workpiece_mesh=workpiece_mesh, workpiece_pose=workpiece_pose,
                       use_obstacles=use_obstacles, obstacle_spec=spec)


def empty_world():
    """空碰撞世界（仅自碰撞，无外部障碍）。返回 (world_config, checker_type)。"""
    _apply_shim()
    from curobo.geom.types import WorldConfig
    from curobo.geom.sdf.world import CollisionCheckerType
    return WorldConfig(cuboid=[], cylinder=[], mesh=[]), CollisionCheckerType.PRIMITIVE


# --------------------------------------------------------------------------- #
# 可复用规划器：碰撞世界 + 机器人只构建一次，可反复规划（推荐重复调用时用）
# --------------------------------------------------------------------------- #
class StompPlanner:
    """绑定一个 cuRobo 碰撞世界 + 机器人，提供到关节角 / 到位姿的 STOMP 规划。

    构建一次（CollisionOracle 即 cuRobo RobotWorld）后可多次调用 plan_joint / plan_pose，
    避免每次规划都重建碰撞世界。
    """

    def __init__(self, world, *, robot_yml=DEFAULT_ROBOT_YML, checker_type=None,
                 device="cuda", buffer=0.1):
        """
        world        : cuRobo WorldConfig（碰撞世界）
        robot_yml    : cuRobo 机器人 yml（默认本项目 ur12e.yml）
        checker_type : cuRobo CollisionCheckerType；None 时按 world 内容自动推断
        device       : 必须为 'cuda'（cuRobo RobotWorld/IKSolver 仅支持 CUDA）
        buffer       : 碰撞安全缓冲/激活距离(m)，>0 即视为进入缓冲（代价>0）
        """
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("STOMP 规划依赖 cuRobo RobotWorld/IKSolver，必须在 CUDA 上运行。")
        _apply_shim()

        self.device = device
        self.robot_yml = robot_yml
        self.world = world
        self.checker_type = checker_type or _infer_checker_type(world)
        self.buffer = buffer

        self.robot = load_robot_cfg(robot_yml)          # joint_names / lower / upper / retract
        self.lower = self.robot["lower_limit"]
        self.upper = self.robot["upper_limit"]

        # cuRobo RobotWorld 封装：批量关节角 -> (世界碰撞+自碰撞) 距离
        self.oracle = CollisionOracle(robot_yml, world, self.checker_type,
                                      activation_distance=buffer)
        self._ik_solver = None  # 懒构建并缓存

    # --------------------------- 函数1：到关节角 --------------------------- #
    def plan_joint(self, cur_cfg, target_cfg, *, num_timesteps=51, num_iterations=80,
                   num_batch=4, delta_t=0.1, collision_weight=80.0, noise_scale=1.5,
                   filter_scale=2.0):
        """规划 cur_cfg -> target_cfg 的避障关节轨迹（**所有 batch 候选都返回**）。

        返回 (trajs, infos)：
          trajs : ndarray (B, T, 6)，B=num_batch 条候选轨迹（不再只挑最优一条）
          infos : list[dict]（长度 B），每条对应一个诊断 dict：
                  batch、state_cost、total_cost、collision_max/mean、
                  n_collision_steps、n_steps、in_limit。
        """
        cur_cfg = list(map(float, cur_cfg))
        target_cfg = list(map(float, target_cfg))
        assert len(cur_cfg) == 6 and len(target_cfg) == 6, "关节角必须是 6 维"

        stomp = ObstacleStomp(
            oracle=self.oracle, collision_cost_weight=collision_weight,
            lower_limit=self.lower, upper_limit=self.upper, device=self.device,
            num_batch=num_batch, num_timesteps=num_timesteps, delta_t=delta_t,
            num_iterations=num_iterations, noise_scale=noise_scale, filter_scale=filter_scale)

        cur = torch.tensor(cur_cfg, device=self.device)
        target = torch.tensor(target_cfg, device=self.device)
        # fixed_pts: (B, 2, D) —— 每条 batch 都固定同一对首末点
        fixed_pts = torch.stack([cur, target], dim=0).unsqueeze(0).repeat(num_batch, 1, 1)

        traj_all, total_cost = stomp.solve(fixed_pts)            # (B,D,T), (B,)
        state_cost = stomp.parameters_state_cost                 # (B,)
        B = traj_all.shape[0]
        lower = np.array(self.lower)
        upper = np.array(self.upper)

        trajs, infos = [], []
        for b in range(B):
            traj = traj_all[b]                                   # (D,T)
            traj_np = traj.transpose(0, 1).contiguous().cpu().numpy()    # (T,6)
            dtraj = self.oracle.distance(traj.transpose(0, 1)).cpu().numpy()  # (T,)
            info = dict(
                batch=b,
                state_cost=float(state_cost[b]),
                total_cost=float(total_cost[b]),
                collision_max=float(dtraj.max()),
                collision_mean=float(dtraj.mean()),
                n_collision_steps=int((dtraj > 1e-4).sum()),
                n_steps=int(traj_np.shape[0]),
                in_limit=bool(np.all(traj_np >= lower - 1e-5)
                              and np.all(traj_np <= upper + 1e-5)),
            )
            trajs.append(traj_np)
            infos.append(info)
        return np.stack(trajs, axis=0), infos

    # --------------------------- 函数2：到位姿 --------------------------- #
    def ik(self, target_pose, *, num_seeds=100, position_threshold=0.005,
           rotation_threshold=0.05, collision_tol=1e-4, return_info=False):
        """末端目标位姿 -> 避障+不自碰的目标关节角（cuRobo 多种子 IKSolver）。

        target_pose: [x,y,z, qw,qx,qy,qz]（base 系, 四元数 wxyz），或 (pos3, quat4)。
        在所有成功 IK 解中，优先选「用本碰撞世界判定为无碰撞」且位置误差最小的解；
        若无无碰解，则退回位置误差最小的解并给出警告。
        返回 target_cfg(list,6)；return_info=True 时返回 (target_cfg, info_dict)。
        """
        pos, quat = _normalize_pose(target_pose)
        solver = self._get_ik_solver(num_seeds, position_threshold, rotation_threshold)

        from curobo.types.math import Pose
        ta = self.oracle.ta
        goal = Pose(position=ta.to_device(pos).view(1, 3),
                    quaternion=ta.to_device(quat).view(1, 4))
        res = solver.solve_batch(goal, return_seeds=num_seeds)

        dof = len(self.robot["joint_names"])
        sol = res.solution.reshape(-1, dof)
        errs = res.position_error.reshape(-1)
        succ = res.success.reshape(-1)
        cands = [(sol[i], float(errs[i].item()))
                 for i in range(sol.shape[0]) if bool(succ[i].item())]
        if not cands:
            raise RuntimeError("IK 求解失败：目标位姿无可行解（检查可达性/位姿是否在 base 系）。")
        cands.sort(key=lambda x: x[1])

        # 用碰撞世界过滤：优先「无碰撞」且位置误差最小的解
        chosen, chosen_err, chosen_dist, collision_free = None, None, None, False
        for q, err in cands:
            d = float(self.oracle.distance(q.view(1, -1))[0].item())
            if d <= collision_tol:
                chosen, chosen_err, chosen_dist, collision_free = q, err, d, True
                break
        if chosen is None:
            q, err = cands[0]
            chosen, chosen_err = q, err
            chosen_dist = float(self.oracle.distance(q.view(1, -1))[0].item())
            print(f"[ik][warn] 所有 IK 解都进入碰撞/缓冲(最小碰撞距离={chosen_dist:.4f})，"
                  f"已退回位置误差最小解；STOMP 终点将固定于此（可能起步即碰）。")

        target_cfg = chosen.detach().cpu().numpy().tolist()
        if not return_info:
            return target_cfg
        info = dict(position_error=chosen_err, collision_distance=chosen_dist,
                    collision_free=collision_free, num_ik_solutions=len(cands))
        return target_cfg, info

    def plan_pose(self, cur_cfg, target_pose, *, ik_num_seeds=100,
                  ik_position_threshold=0.005, ik_rotation_threshold=0.05,
                  **plan_kwargs):
        """规划 cur_cfg -> target_pose（先 IK 解目标关节角，再走 plan_joint）。

        plan_kwargs 透传给 plan_joint（num_iterations/num_batch/collision_weight/...）。
        返回 (trajs, infos)：与 plan_joint 一样【所有 batch 都返回】，每条 info 额外含
        target_cfg / ik_position_error / ik_collision_free。
        """
        target_cfg, ik_info = self.ik(
            target_pose, num_seeds=ik_num_seeds,
            position_threshold=ik_position_threshold,
            rotation_threshold=ik_rotation_threshold, return_info=True)

        trajs, infos = self.plan_joint(cur_cfg, target_cfg, **plan_kwargs)
        for info in infos:
            info["target_cfg"] = target_cfg
            info["ik_position_error"] = ik_info["position_error"]
            info["ik_collision_free"] = ik_info["collision_free"]
        return trajs, infos

    # --------------------------- 内部 --------------------------- #
    def _get_ik_solver(self, num_seeds, position_threshold, rotation_threshold):
        """构建并缓存一个多种子 IKSolver（无世界；世界碰撞用 oracle 后置过滤）。

        与 gt_gen_hanfeng init_curobo 中的 IKSolver 配置一致：多种子、关自碰撞优化、
        关 cuda_graph（避免 solve_batch 冲突）。
        """
        if self._ik_solver is not None:
            return self._ik_solver
        _apply_shim()
        from curobo.util_file import load_yaml
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

        robot_cfg = load_yaml(self.robot_yml)["robot_cfg"]
        ik_cfg = IKSolverConfig.load_from_robot_config(
            robot_cfg, None, num_seeds=num_seeds,
            self_collision_check=True, self_collision_opt=True,
            use_cuda_graph=False, tensor_args=self.oracle.ta,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
        )
        self._ik_solver = IKSolver(ik_cfg)
        return self._ik_solver


def _normalize_pose(target_pose):
    """统一目标位姿为 (pos[3], quat[4] wxyz) 两个 python list。"""
    if isinstance(target_pose, (tuple, list)) and len(target_pose) == 2 \
            and hasattr(target_pose[0], "__len__"):
        pos, quat = target_pose
        return list(map(float, pos)), list(map(float, quat))
    p = list(map(float, target_pose))
    assert len(p) == 7, "target_pose 应为 [x,y,z,qw,qx,qy,qz] 或 (pos3, quat4)"
    return p[:3], p[3:]


# --------------------------------------------------------------------------- #
# 顶层便捷函数（用完即弃地构建一次规划器；重复规划请直接用 StompPlanner）
# --------------------------------------------------------------------------- #
def plan_to_joint_config(cur_cfg, target_cfg, world, *, robot_yml=DEFAULT_ROBOT_YML,
                         checker_type=None, device="cuda", buffer=0.1, **plan_kwargs):
    """【函数1】当前关节角 + 目标关节角 + cuRobo 碰撞世界 -> 所有 batch 候选避障轨迹。

    world        : cuRobo WorldConfig
    plan_kwargs  : 透传 StompPlanner.plan_joint（num_iterations/num_batch/collision_weight/
                   num_timesteps/delta_t/noise_scale/...）
    返回 (trajs, infos)：trajs (B,T,6)，infos 为各 batch 的诊断 dict 列表。
    """
    planner = StompPlanner(world, robot_yml=robot_yml, checker_type=checker_type,
                           device=device, buffer=buffer)
    return planner.plan_joint(cur_cfg, target_cfg, **plan_kwargs)


def _valid_candidates(infos):
    """从 plan_pose/plan_joint 的 infos 里挑出满足「无碰撞 + 在限位内」的 batch 下标，
    按 state_cost 升序（越小越好）返回。"""
    valid = [(info["state_cost"], i) for i, info in enumerate(infos)
             if info["n_collision_steps"] == 0 and info["in_limit"]]
    valid.sort(key=lambda x: x[0])
    return [i for _, i in valid]


def plan_to_joint_single(cur_cfg, target_cfg, world, *, robot_yml=DEFAULT_ROBOT_YML,
                         checker_type=None, device="cuda", buffer=0.1, **plan_kwargs):
    """【函数1-single】当前关节角 + 目标关节角 + cuRobo 碰撞世界 -> 单条最优避障轨迹 (T,6)。

    只在满足 n_collision_steps==0 且 in_limit==True 的候选里，取 state_cost 最小的一条返回；
    **不返回 info**；若没有任何候选满足，返回 None。

    plan_kwargs  : 透传 StompPlanner.plan_joint（num_iterations/num_batch/collision_weight/...）
    """
    planner = StompPlanner(world, robot_yml=robot_yml, checker_type=checker_type,
                           device=device, buffer=buffer)
    trajs, infos = planner.plan_joint(cur_cfg, target_cfg, **plan_kwargs)
    idx = _valid_candidates(infos)
    if not idx:
        return None
    return trajs[idx[0]]


def plan_to_joint_multi(cur_cfg, target_cfg, world, *, robot_yml=DEFAULT_ROBOT_YML,
                        checker_type=None, device="cuda", buffer=0.1, **plan_kwargs):
    """【函数1-multi】当前关节角 + 目标关节角 + cuRobo 碰撞世界 -> 所有合格避障轨迹列表。

    返回所有满足 n_collision_steps==0 且 in_limit==True 的轨迹（list，每条 (T,6)，
    按 state_cost 升序）；**不返回 info**；若没有任何候选满足，返回 None。

    plan_kwargs  : 透传 StompPlanner.plan_joint（num_iterations/num_batch/collision_weight/...）
    """
    planner = StompPlanner(world, robot_yml=robot_yml, checker_type=checker_type,
                           device=device, buffer=buffer)
    trajs, infos = planner.plan_joint(cur_cfg, target_cfg, **plan_kwargs)
    idx = _valid_candidates(infos)
    if not idx:
        return None
    return [trajs[i] for i in idx]


def plan_to_pose_single(cur_cfg, target_pose, world, *, robot_yml=DEFAULT_ROBOT_YML,
                        checker_type=None, device="cuda", buffer=0.1,
                        ik_num_seeds=100, ik_position_threshold=0.005,
                        ik_rotation_threshold=0.05, **plan_kwargs):
    """【函数2-single】当前关节角 + 目标位姿 + cuRobo 碰撞世界 -> 单条最优避障轨迹 (T,6)。

    只在满足 n_collision_steps==0 且 in_limit==True 的候选里，取 state_cost 最小的一条返回；
    **不返回 info**；若没有任何候选满足，返回 None。

    target_pose  : [x,y,z,qw,qx,qy,qz]（base 系, 四元数 wxyz）或 (pos3, quat4)
    """
    planner = StompPlanner(world, robot_yml=robot_yml, checker_type=checker_type,
                           device=device, buffer=buffer)
    trajs, infos = planner.plan_pose(
        cur_cfg, target_pose, ik_num_seeds=ik_num_seeds,
        ik_position_threshold=ik_position_threshold,
        ik_rotation_threshold=ik_rotation_threshold, **plan_kwargs)
    idx = _valid_candidates(infos)
    if not idx:
        return None
    return trajs[idx[0]]


def plan_to_pose_multi(cur_cfg, target_pose, world, *, robot_yml=DEFAULT_ROBOT_YML,
                       checker_type=None, device="cuda", buffer=0.1,
                       ik_num_seeds=100, ik_position_threshold=0.005,
                       ik_rotation_threshold=0.05, **plan_kwargs):
    """【函数2-multi】当前关节角 + 目标位姿 + cuRobo 碰撞世界 -> 所有合格避障轨迹列表。

    返回所有满足 n_collision_steps==0 且 in_limit==True 的轨迹（list，每条 (T,6)，
    按 state_cost 升序）；**不返回 info**；若没有任何候选满足，返回 None。

    target_pose  : [x,y,z,qw,qx,qy,qz]（base 系, 四元数 wxyz）或 (pos3, quat4)
    """
    planner = StompPlanner(world, robot_yml=robot_yml, checker_type=checker_type,
                           device=device, buffer=buffer)
    trajs, infos = planner.plan_pose(
        cur_cfg, target_pose, ik_num_seeds=ik_num_seeds,
        ik_position_threshold=ik_position_threshold,
        ik_rotation_threshold=ik_rotation_threshold, **plan_kwargs)
    idx = _valid_candidates(infos)
    if not idx:
        return None
    return [trajs[i] for i in idx]
