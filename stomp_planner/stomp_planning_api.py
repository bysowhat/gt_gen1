"""对外暴露本项目「带障碍物 STOMP 路径规划」能力的稳定 API（供别的项目调用）。

本模块把 plan_path_stomp.py / plan_path_stomp_obstacle.py 里的核心规划逻辑
封装成几个即拿即用的函数 + 一个可复用的 StompPlanner 类：

    函数1  plan_to_joint_config(cur_cfg, target_cfg, world, ...)
            输入：当前关节角、目标关节角、cuRobo 碰撞世界
            输出：(trajs, infos) —— 所有 batch 候选轨迹 (B,T,6) 及各自诊断

    函数1-single  plan_to_joint_single(cur_cfg, target_cfg, world, ...)
            在满足「无碰撞 + 在限位内」的候选里取焊枪路径最短的一条 traj (T,6)；
            不返回 info；若无满足者返回 None。

    函数1-multi   plan_to_joint_multi(cur_cfg, target_cfg, world, ...)
            返回所有满足「无碰撞 + 在限位内」的 traj 列表（每条 (T,6)）；
            不返回 info；若无满足者返回 None。

    函数2-single  plan_to_pose_single(cur_cfg, target_pose, world, ...)
            在满足「无碰撞 + 在限位内」的候选里取焊枪路径最短的一条 traj (T,6)；
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
import time

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
                   filter_scale=2.0, early_stop=True, early_stop_patience=5,
                   early_stop_min_iters=20, early_stop_rel_tol=1e-3,
                   early_stop_min_delta=1e-4, early_stop_state_tol=1e-3):
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

        # 目标角相对起点做 2π 分支归一：旋转关节（如 J6 腕部 roll）若目标落在差 2π 的远端支，
        # STOMP 固定首末点后会线性插值绕整圈（焊枪自转 360°）。归一到近端等价支，末端位姿不变。
        try:
            from gt_gen.joint_wrap import wrap_goal_near_start
            target_cfg = wrap_goal_near_start(cur_cfg, target_cfg, self.lower, self.upper,
                                              tag="wrap/stomp")
        except Exception as e:  # noqa: BLE001  归一化失败不应阻断规划
            print(f"[wrap/stomp][warn] 目标角 2π 归一化跳过：{e}")

        stomp = ObstacleStomp(
            oracle=self.oracle, collision_cost_weight=collision_weight,
            lower_limit=self.lower, upper_limit=self.upper, device=self.device,
            num_batch=num_batch, num_timesteps=num_timesteps, delta_t=delta_t,
            num_iterations=num_iterations, noise_scale=noise_scale, filter_scale=filter_scale,
            early_stop=early_stop, early_stop_patience=early_stop_patience,
            early_stop_min_iters=early_stop_min_iters, early_stop_rel_tol=early_stop_rel_tol,
            early_stop_min_delta=early_stop_min_delta, early_stop_state_tol=early_stop_state_tol)

        cur = torch.tensor(cur_cfg, device=self.device)
        target = torch.tensor(target_cfg, device=self.device)
        # fixed_pts: (B, 2, D) —— 每条 batch 都固定同一对首末点
        fixed_pts = torch.stack([cur, target], dim=0).unsqueeze(0).repeat(num_batch, 1, 1)

        traj_all, total_cost = stomp.solve(fixed_pts)            # (B,D,T), (B,)
        if early_stop and stomp.iterations_run < num_iterations:
            print(f"[早停] STOMP 实跑 {stomp.iterations_run}/{num_iterations} 轮收敛退出")
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
                traj=traj_np,                                    # (T,6) 该 batch 轨迹（供可视化）
                dtraj=dtraj,                                     # (T,)  每路点碰撞距离(>1e-4=碰)
            )
            trajs.append(traj_np)
            infos.append(info)
            # a = 0
            # if a == 1:
            #     _viz_collision(self.oracle, self.world, info, self.buffer)   # 逐碰撞路点弹窗；注释此行即关闭
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

        # 所有 IK 解都进入碰撞/缓冲 → 目标位姿本身不可行，STOMP 终点只能固定在碰撞构型上、
        # 必产垃圾轨迹。此处直接判规划失败、不再跑 STOMP；返回空候选，下游 _valid_candidates
        # 取不到合格解 → plan_to_pose_single/multi 返回 None（默认轨迹判 FAIL / 绕行判 no_solution）。
        if not ik_info["collision_free"]:
            print(f"[plan_pose] 所有 IK 解都进入碰撞/缓冲"
                  f"(最小碰撞距离={ik_info['collision_distance']:.4f}) → 目标位姿不可行，"
                  f"不再继续 STOMP，直接判规划失败。")
            return [], []

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


def _valid_candidates_by_state_cost(infos):
    """【原实现，保留备查】从 plan_pose/plan_joint 的 infos 里挑出满足「无碰撞 + 在限位内」的
    batch 下标，按 state_cost 升序（越小越好）返回。"""
    valid = [(info["state_cost"], i) for i, info in enumerate(infos)
             if info["n_collision_steps"] == 0 and info["in_limit"]]
    valid.sort(key=lambda x: x[0])
    return [i for _, i in valid]


# 焊枪 link 名（与 ur12e.yml collision_link_names、gt_gen/nbv.py 一致）
_GUN_LINK_NAME = "xiaoyu_accessory_link"


def _oracle_link_sphere_mask(oracle, link_name):
    """oracle(cuRobo RobotWorld) 上指定 link 的碰撞球掩码 (S,) bool；取不到/为空返回 None。

    与 gt_gen/swept.py 的 link_sphere_mask 同源：mask = (link_sphere_idx_map == link_name_to_idx[link])。
    按 (oracle, link_name) 缓存（kinematics 不变）。
    """
    cache = getattr(oracle, "_link_sphere_mask_cache", None)
    if cache is None:
        cache = oracle._link_sphere_mask_cache = {}
    if link_name in cache:
        return cache[link_name]
    try:
        kc = oracle.rw.kinematics.kinematics_config
        idx_map = kc.link_sphere_idx_map.detach().cpu().numpy()        # (S,) 每球所属 link 下标
        name_to_idx = dict(kc.link_name_to_idx_map)
        mask = (idx_map == name_to_idx[link_name])
        if not bool(np.any(mask)):
            mask = None
    except Exception:  # noqa: BLE001
        mask = None
    cache[link_name] = mask
    return mask


def _gun_translation_cost(oracle, traj):
    """焊枪(xiaoyu_accessory_link)沿整条轨迹的平移路径长度(m)，按碰撞球半径加权。

    参照 gt_gen/nbv.py 的 _gun_translation_cost（那里只算 cur→cand 单步位移），这里把它推广到
    「整条轨迹逐路点位移之和」——因为所有 batch 候选首末点相同，单步位移无法区分，只有沿途累计
    的路径长度才有区分度。做法：对该 link 每个碰撞球，累加相邻路点球心平移量得到该球总路径长，
    再按半径加权 (w_i = r_i / r_max) 对各球求均值。半径大的球位移影响更大。

    取不到该 link 球掩码时，退回关节空间路径长度（保守不失效）。traj: (T,6) -> 标量平移代价(m)。
    """
    import torch

    traj = np.asarray(traj, float)
    mask = _oracle_link_sphere_mask(oracle, _GUN_LINK_NAME)
    if mask is None:
        return float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())   # 关节空间路径长
    qt = torch.as_tensor(traj, device=oracle.device, dtype=torch.float32)
    st = oracle.rw.kinematics.get_state(qt)
    sph = st.link_spheres_tensor.detach().cpu().numpy()               # (T,S,4) xyz+r
    c = sph[:, mask, :3]                                              # (T,Sg,3) 焊枪球心轨迹
    radii = sph[0, mask, 3]                                           # (Sg,) 球半径
    per_sphere_path = np.linalg.norm(np.diff(c, axis=0), axis=2).sum(axis=0)  # (Sg,) 各球总路径长
    r_max = float(radii.max())
    if r_max <= 0.0:
        return float(per_sphere_path.mean())
    w = radii / r_max                                                # 最大球=1，其余=r_i/r_max
    return float((w * per_sphere_path).sum() / w.sum())


# 旋转代价用的 link（Link6 = 腕部末端，其局部 +Z = 焊枪轴向；cuRobo 标准 link，FK 可直接取姿态）
_ROT_LINK_NAME = "Link6"
# 旋转代价权重：把「累计绕枪轴自转弧度」换算成等效平移(m) 后与平移代价相加。1 rad ≈ 该米数。
# 注意：现在只算绕 Z 的自转，量级比之前的总旋转小，权重可能要相应调大；按实际轨迹标定。
_ROT_WEIGHT_M_PER_RAD = 0.2


def _rot_link_fk_kinematics(oracle):
    """惰性构建并缓存一个只做 FK 的 CudaRobotModel，link_names 含 _ROT_LINK_NAME，用于取其姿态四元数。

    oracle 的 RobotWorld kinematics 建时 link_names=null，拿不到具名 link 位姿，故单独建一个 FK 模型
    （仿 gt_gen/sensor.py.build_kinematics）。取不到时返回 None（旋转代价退化为 0）。
    """
    if hasattr(oracle, "_rot_fk_kin"):
        return oracle._rot_fk_kin
    kin = None
    try:
        try:
            import gt_gen.compat  # noqa: F401
            gt_gen.compat.apply_trimesh_shim()
        except Exception:  # noqa: BLE001
            pass
        from curobo.types.robot import RobotConfig
        from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
        from curobo.util_file import load_yaml

        d = load_yaml(oracle.robot_yml)
        d["robot_cfg"]["kinematics"]["link_names"] = [_ROT_LINK_NAME]
        rc = RobotConfig.from_dict(d["robot_cfg"], oracle.ta)
        kin = CudaRobotModel(rc.kinematics)
    except Exception as e:  # noqa: BLE001
        print(f"[path_cost][warn] 构建 Link6 FK 模型失败，旋转代价按 0 计：{e}")
        kin = None
    oracle._rot_fk_kin = kin
    return kin


def _rotation_cost(oracle, traj):
    """_ROT_LINK_NAME(Link6) 沿整条轨迹【绕自身局部 Z 轴】的累计自转量(rad)：
    逐相邻路点求相对旋转、用绕 Z 的 swing-twist 分解取 twist 角、全程按绝对值求和。

    只计「绕枪轴(Link6 局部 +Z)的自转 roll」，不含枪指向的摆动(pitch/yaw)。做法：相邻姿态
    q_t、q_{t+1}(base 系单位四元数) 的相对旋转(表达在局部系) Δq = q_t⁻¹ ⊗ q_{t+1}；对 Δq 做绕 Z 的
    swing-twist 分解 —— twist 角 = 2·atan2(Δq_z, Δq_w)，即该步绕枪轴转过的弧度。只需 Δq 的 w、z
    两个分量（q_t⁻¹ = 共轭）。

    取不到 FK/姿态时返回 0（不失效，仅退化为「只按平移排序」）。traj: (T,6) -> 标量自转代价(rad)。
    """
    import torch

    kin = _rot_link_fk_kinematics(oracle)
    if kin is None:
        return 0.0
    qt = torch.as_tensor(np.asarray(traj, np.float32), device=oracle.device)
    st = kin.get_state(qt)
    lp = st.link_pose.get(_ROT_LINK_NAME) if isinstance(st.link_pose, dict) else None
    if lp is None:
        return 0.0
    q = lp.quaternion.detach().cpu().numpy()                         # (T,4) wxyz
    w0, x0, y0, z0 = q[:-1, 0], q[:-1, 1], q[:-1, 2], q[:-1, 3]      # q_t
    w1, x1, y1, z1 = q[1:, 0], q[1:, 1], q[1:, 2], q[1:, 3]          # q_{t+1}
    # Δq = q_t⁻¹ ⊗ q_{t+1}（q_t⁻¹ = 共轭）；绕 Z 的 twist 只需 Δq 的 w、z 分量
    dw = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1                       # Δq_w（= 相邻四元数点积）
    dz = w0 * z1 - x0 * y1 + y0 * x1 - z0 * w1                       # Δq_z
    twist = 2.0 * np.arctan2(dz, dw)                                # (T-1,) 每步绕 Z 的 twist 角
    twist = (twist + np.pi) % (2.0 * np.pi) - np.pi                 # 归一到 (-π, π]（防四元数双覆盖翻号）
    return float(np.abs(twist).sum())                              # 逐步绕 Z 自转角之和


def _gun_path_cost(oracle, traj):
    """轨迹综合运动代价 = 焊枪平移路径(m) + _ROT_WEIGHT_M_PER_RAD · Link6 绕枪轴累计自转(rad)。

    平移项见 _gun_translation_cost（焊枪碰撞球半径加权路径长），旋转项见 _rotation_cost
    （Link6 绕自身局部 Z 轴的逐步自转角之和）。traj: (T,6) -> 标量综合代价。
    """
    trans = _gun_translation_cost(oracle, traj)
    # rot = _rotation_cost(oracle, traj)
    # return trans + _ROT_WEIGHT_M_PER_RAD * rot
    return trans


def _valid_candidates(infos, oracle):
    """从 plan_pose/plan_joint 的 infos 里挑出满足「无碰撞 + 在限位内」的 batch 下标，
    按**综合运动代价最小**升序（越小越好）返回。

    综合代价 = 焊枪(xiaoyu_accessory_link)半径加权平移路径(m) + 权重·Link6 绕枪轴累计自转(rad)，
    见 _gun_path_cost。（原按 state_cost 排序的实现见 _valid_candidates_by_state_cost，保留备查。）
    """
    valid = [(_gun_path_cost(oracle, info["traj"]), i) for i, info in enumerate(infos)
             if info["n_collision_steps"] == 0 and info["in_limit"]]
    valid.sort(key=lambda x: x[0])
    return [i for _, i in valid]


def plan_to_joint_single(cur_cfg, target_cfg, world, *, robot_yml=DEFAULT_ROBOT_YML,
                         checker_type=None, device="cuda", buffer=0.1, **plan_kwargs):
    """【函数1-single】当前关节角 + 目标关节角 + cuRobo 碰撞世界 -> 单条最优避障轨迹 (T,6)。

    只在满足 n_collision_steps==0 且 in_limit==True 的候选里，取焊枪路径最短的一条返回；
    **不返回 info**；若没有任何候选满足，返回 None。

    plan_kwargs  : 透传 StompPlanner.plan_joint（num_iterations/num_batch/collision_weight/...）
    """
    _t0 = time.perf_counter()
    planner = StompPlanner(world, robot_yml=robot_yml, checker_type=checker_type,
                           device=device, buffer=buffer)
    _t_build = time.perf_counter()
    trajs, infos = planner.plan_joint(cur_cfg, target_cfg, **plan_kwargs)
    _t_solve = time.perf_counter()
    print(f"[计时·拆分] 建planner {_t_build - _t0:.3f}s | STOMP solve {_t_solve - _t_build:.3f}s")
    idx = _valid_candidates(infos, planner.oracle)
    if not idx:
        return None
    return trajs[idx[0]]


def plan_to_joint_multi(cur_cfg, target_cfg, world, *, robot_yml=DEFAULT_ROBOT_YML,
                        checker_type=None, device="cuda", buffer=0.1, **plan_kwargs):
    """【函数1-multi】当前关节角 + 目标关节角 + cuRobo 碰撞世界 -> 所有合格避障轨迹列表。

    返回所有满足 n_collision_steps==0 且 in_limit==True 的轨迹（list，每条 (T,6)，
    按焊枪路径升序）；**不返回 info**；若没有任何候选满足，返回 None。

    plan_kwargs  : 透传 StompPlanner.plan_joint（num_iterations/num_batch/collision_weight/...）
    """
    planner = StompPlanner(world, robot_yml=robot_yml, checker_type=checker_type,
                           device=device, buffer=buffer)
    trajs, infos = planner.plan_joint(cur_cfg, target_cfg, **plan_kwargs)
    idx = _valid_candidates(infos, planner.oracle)
    if not idx:
        return None
    return [trajs[i] for i in idx]


def plan_to_pose_single(cur_cfg, target_pose, world, *, robot_yml=DEFAULT_ROBOT_YML,
                        checker_type=None, device="cuda", buffer=0.1,
                        ik_num_seeds=100, ik_position_threshold=0.005,
                        ik_rotation_threshold=0.05, **plan_kwargs):
    """【函数2-single】当前关节角 + 目标位姿 + cuRobo 碰撞世界 -> 单条最优避障轨迹 (T,6)。

    只在满足 n_collision_steps==0 且 in_limit==True 的候选里，取焊枪路径最短的一条返回；
    **不返回 info**；若没有任何候选满足，返回 None。

    target_pose  : [x,y,z,qw,qx,qy,qz]（base 系, 四元数 wxyz）或 (pos3, quat4)
    """
    planner = StompPlanner(world, robot_yml=robot_yml, checker_type=checker_type,
                           device=device, buffer=buffer)
    trajs, infos = planner.plan_pose(
        cur_cfg, target_pose, ik_num_seeds=ik_num_seeds,
        ik_position_threshold=ik_position_threshold,
        ik_rotation_threshold=ik_rotation_threshold, **plan_kwargs)
    idx = _valid_candidates(infos, planner.oracle)
    if not idx:
        return None
    return trajs[idx[0]]


def plan_to_pose_multi(cur_cfg, target_pose, world, *, robot_yml=DEFAULT_ROBOT_YML,
                       checker_type=None, device="cuda", buffer=0.1,
                       ik_num_seeds=100, ik_position_threshold=0.005,
                       ik_rotation_threshold=0.05, **plan_kwargs):
    """【函数2-multi】当前关节角 + 目标位姿 + cuRobo 碰撞世界 -> 所有合格避障轨迹列表。

    返回所有满足 n_collision_steps==0 且 in_limit==True 的轨迹（list，每条 (T,6)，
    按焊枪路径升序）；**不返回 info**；若没有任何候选满足，返回 None。

    target_pose  : [x,y,z,qw,qx,qy,qz]（base 系, 四元数 wxyz）或 (pos3, quat4)
    """
    planner = StompPlanner(world, robot_yml=robot_yml, checker_type=checker_type,
                           device=device, buffer=buffer)
    trajs, infos = planner.plan_pose(
        cur_cfg, target_pose, ik_num_seeds=ik_num_seeds,
        ik_position_threshold=ik_position_threshold,
        ik_rotation_threshold=ik_rotation_threshold, **plan_kwargs)
    idx = _valid_candidates(infos, planner.oracle)
    if not idx:
        return None
    return [trajs[i] for i in idx]


# --------------------------------------------------------------------------- #
# 碰撞可视化（调试）：逐个碰撞路点弹 open3d 窗口，画整臂碰撞球 + 障碍
# 开关 = 是否注释掉 plan_joint 里的 `_viz_collision(...)` 调用行（无布尔变量控制）
# --------------------------------------------------------------------------- #
def _quat_to_R(quat_wxyz):
    """四元数 (w,x,y,z) -> 旋转矩阵 (3,3)。"""
    w, x, y, z = (float(v) for v in quat_wxyz)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
        [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)],
    ], float)


def _arm_spheres(oracle, q):
    """整臂碰撞球 (S,4) [x,y,z,r] + 每球所属 link 索引 (S,)，base 系。

    cuRobo FK；与 gt_gen swept.fk_spheres_batch 同源。丢弃半径<=0 的无效球
    （cuRobo 用 r<0 占位关闭的球），link 索引同步过滤。返回 (spheres, link_idx)。
    """
    qt = torch.as_tensor([list(map(float, q))], device=oracle.device, dtype=torch.float32)
    st = oracle.rw.kinematics.get_state(qt)
    sph = st.link_spheres_tensor.detach().cpu().numpy()[0]      # (S,4)
    li = oracle.rw.kinematics.kinematics_config.link_sphere_idx_map.detach().cpu().numpy()
    mask = sph[:, 3] > 1e-6
    return sph[mask], li[mask].astype(int)


def _link_idx_to_name(oracle):
    """link 索引 -> link 名字典（用于打印自碰撞是哪两个 link）。"""
    m = dict(oracle.rw.kinematics.kinematics_config.link_name_to_idx_map)
    return {int(v): str(k) for k, v in m.items()}


def _adjacent_link_pairs(oracle):
    """从运动链 link_chain_map 取「父子相邻」link 对集合 set{frozenset({la,lb})}。

    cuRobo 默认忽略相邻 link 间自碰（设计上必然贴合）。link_chain_map[a][b]=1 表示 b 是 a 的祖先
    （含自身）；深度 depth[l]=祖先数；父子 = 互为祖先且深度差 1。仅取父子（不取全部祖先，
    否则手臂折回撞自身 base 这类真自碰也会被忽略）。
    """
    lcm = getattr(oracle.rw.kinematics.kinematics_config, "link_chain_map", None)
    if lcm is None:
        return set()
    a = lcm.detach().cpu().numpy().astype(int)
    n = a.shape[0]
    depth = a.sum(axis=1) - 1                                   # 每 link 的祖先数
    adj = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j 是 i 的祖先 且 深度差 1 -> 父子相邻
            if a[i, j] == 1 and depth[i] - depth[j] == 1:
                adj.add(frozenset((i, j)))
    return adj


def _self_collision_pairs(spheres, link_idx, buffer=0.0, ignore=None, link_ignore=None):
    """跨 link 且球面重叠/进缓冲的自碰撞球对。

    返回 list[(i, j, gap)]，gap = r_i+r_j+buffer - ||c_i-c_j|| (>0 = 重叠/进缓冲)，按 gap 降序。
    跳过：同 link 的球；link 对属于 link_ignore（父子相邻，设计贴合）；球对 (i,j) 属于 ignore
    （从自碰无碰路点学到的非相邻设计贴合对，如工具与 Link6）。
    """
    S = int(spheres.shape[0])
    c = spheres[:, :3]
    r = spheres[:, 3]
    ignore = ignore or set()
    link_ignore = link_ignore or set()
    pairs = []
    for i in range(S):
        for j in range(i + 1, S):
            if link_idx[i] == link_idx[j] or (i, j) in ignore:
                continue
            if frozenset((int(link_idx[i]), int(link_idx[j]))) in link_ignore:
                continue
            d = float(np.linalg.norm(c[i] - c[j]))
            gap = float(r[i] + r[j] + float(buffer) - d)
            if gap > 0.0:
                pairs.append((i, j, gap))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs


def _sphere_hits_boxes(spheres, cuboids, buffer=0.0):
    """每球 vs 每盒 精确 sphere-box 距离（盒可带旋转）。

    返回 (hit_sphere (S,) bool, hit_box (nbox,) bool)：距离 <= r+buffer 即判碰。
    """
    S = int(spheres.shape[0])
    nb = len(cuboids)
    hit_s = np.zeros(S, bool)
    hit_b = np.zeros(nb, bool)
    if S == 0 or nb == 0:
        return hit_s, hit_b
    c = spheres[:, :3]                                          # (S,3) 球心
    r = spheres[:, 3]                                           # (S,)  半径
    for bi, box in enumerate(cuboids):
        p = np.asarray(box.pose[:3], float)
        half = np.asarray(box.dims, float) * 0.5
        R = _quat_to_R(box.pose[3:7])                          # 盒->world 旋转
        local = (c - p) @ R                                    # world->盒: R^T (c-p)
        d = np.maximum(np.abs(local) - half, 0.0)              # 到盒面外距离(内部=0)
        dist = np.linalg.norm(d, axis=1)                       # (S,)
        hits = dist <= (r + float(buffer))
        if hits.any():
            hit_b[bi] = True
            hit_s |= hits
    return hit_s, hit_b


# 碰撞双方配色：机械臂碰撞球=红，被碰障碍(盒/mesh)=绿；其余=灰
_COLOR_HIT_ARM = [0.9, 0.1, 0.1]      # 红：碰撞的机械臂球
_COLOR_HIT_OBS = [0.1, 0.8, 0.1]      # 绿：被碰的障碍(盒/mesh)
_COLOR_GRAY = [0.6, 0.6, 0.6]         # 灰：未碰


def _box_geom(o3d, box, hit=False):
    """Cuboid -> open3d：被碰=绿实心 TriangleMesh，未碰=灰线框 LineSet。"""
    dx, dy, dz = (float(v) for v in box.dims)
    p = np.asarray(box.pose[:3], float)
    R = _quat_to_R(box.pose[3:7])
    mesh = o3d.geometry.TriangleMesh.create_box(dx, dy, dz)
    mesh.translate((-dx / 2.0, -dy / 2.0, -dz / 2.0))          # 角点在原点 -> 居中
    mesh.rotate(R, center=(0.0, 0.0, 0.0))
    mesh.translate(p)
    if hit:
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(_COLOR_HIT_OBS)
        return mesh
    ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    ls.paint_uniform_color(_COLOR_GRAY)
    return ls


def _mesh_world(m):
    """world.mesh 元素 -> (verts_w (V,3), faces (F,3))，含 pose 变换。失败返回 None。"""
    try:
        tm = m.get_trimesh_mesh()
        verts = np.asarray(tm.vertices, float)
        R = _quat_to_R(m.pose[3:7])
        p = np.asarray(m.pose[:3], float)
        verts_w = verts @ R.T + p
        return verts_w, np.asarray(tm.faces, np.int32)
    except Exception as e:                                      # noqa: BLE001
        print(f"[viz] mesh 加载失败 {getattr(m, 'name', '?')}: {e}")
        return None


def _mesh_geom(o3d, mw, hit=False):
    """(verts_w, faces) -> open3d TriangleMesh：被碰=绿，未碰=灰。mw=None 返回 None。"""
    if mw is None:
        return None
    verts_w, faces = mw
    o3m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts_w),
        o3d.utility.Vector3iVector(faces))
    o3m.compute_vertex_normals()
    o3m.paint_uniform_color(_COLOR_HIT_OBS if hit else [0.7, 0.7, 0.7])
    return o3m


def _sphere_hits_meshes(spheres, meshes_world, buffer=0.0):
    """每球 vs 每个 world 系 trimesh 的最近表面距离；dist<=r+buffer 判碰。

    meshes_world: list，每项 (verts_w, faces) 或 None（加载失败）。
    返回 (hit_sphere (S,) bool, hit_mesh (nmesh,) bool)。缺 trimesh 时全 False。
    """
    S = int(spheres.shape[0])
    nm = len(meshes_world)
    hit_s = np.zeros(S, bool)
    hit_m = np.zeros(nm, bool)
    if S == 0 or nm == 0:
        return hit_s, hit_m
    try:
        import trimesh
    except Exception as e:                                      # noqa: BLE001
        print(f"[viz] trimesh 不可用，mesh 碰撞检测跳过：{e}")
        return hit_s, hit_m
    c = spheres[:, :3]                                          # (S,3) 球心
    r = spheres[:, 3]                                           # (S,)  半径
    for mi, mw in enumerate(meshes_world):
        if mw is None:
            continue
        verts_w, faces = mw
        tm = trimesh.Trimesh(vertices=verts_w, faces=faces, process=False)
        _, dist, _ = trimesh.proximity.closest_point(tm, c)    # (S,) 球心到表面距离
        hits = dist <= (r + float(buffer))
        if hits.any():
            hit_m[mi] = True
            hit_s |= hits
    return hit_s, hit_m


def _viz_collision(oracle, world, info, buffer=0.0, coll_tol=1e-4):
    """遍历【单条 batch 轨迹】的碰撞路点，每个碰撞路点弹一个 open3d 窗口：

    每个碰撞路点先分别取 cuRobo 的世界碰撞距离 dw 与自碰撞距离 ds，打印碰撞类型：
      · 世界碰撞：机械臂碰撞球=红、被碰障碍(cuboid 绿实心 / mesh 绿)=绿；
      · 自碰撞：把互相碰撞的两个球分色（低 link 号=红 / 高 link 号=绿），并打印是哪两个 link。
    其余(未碰的球/盒/mesh)=灰 + base 坐标系。开关 = 是否注释掉 plan_joint 里的调用行。

    盒用精确 sphere-box 距离判碰，mesh 用 trimesh 最近表面距离判碰(缺 trimesh 时 mesh 不高亮)；
    自碰撞球对用「跨 link + 球面重叠」判定，并从轨迹上的自碰无碰路点自动学出相邻设计贴合对忽略。
    info 需含 traj (T,6) 与 dtraj (T,)（plan_joint 已写入）。无碰撞路点则打印并 return；
    缺 open3d / 无显示时打印告警跳过，不打断规划。
    """
    if info.get("traj") is None or info.get("dtraj") is None:
        return
    traj = np.asarray(info["traj"], float)
    dtraj = np.asarray(info["dtraj"], float)
    if dtraj.size == 0:
        return
    b = info.get("batch", -1)
    T = int(traj.shape[0])
    coll_idx = np.nonzero(dtraj > coll_tol)[0]
    if coll_idx.size == 0:
        print(f"[viz] batch {b} 无碰撞路点(dist 全 <= {coll_tol})，不弹窗。")
        return
    try:
        import open3d as o3d
    except Exception as e:                                      # noqa: BLE001
        print(f"[viz] open3d 不可用，跳过可视化：{e}")
        return

    cuboids = list(getattr(world, "cuboid", None) or [])
    meshes = list(getattr(world, "mesh", None) or [])
    meshes_world = [_mesh_world(m) for m in meshes]             # 预算世界系顶点（复用于碰撞+几何）
    idx2name = _link_idx_to_name(oracle)
    link_ignore = _adjacent_link_pairs(oracle)                  # 父子相邻 link 对（设计贴合，忽略）

    # 分别取每路点的世界碰撞 dw 与自碰撞 ds（cuRobo；dtraj=两者之和）
    qt = torch.as_tensor(traj, device=oracle.device, dtype=torch.float32)
    dw_all, ds_all = oracle.rw.get_world_self_collision_distance_from_joints(qt)
    dw_all = dw_all.clamp(min=0).detach().cpu().numpy()          # (T,) 世界碰撞
    ds_all = ds_all.clamp(min=0).detach().cpu().numpy()          # (T,) 自碰撞

    # 忽略集：自碰无碰路点(ds<=tol)上「跨 link + 非父子仍重叠」的球对 = 非相邻设计贴合(如工具↔Link6)
    ignore = set()
    free_idx = np.nonzero(ds_all <= coll_tol)[0]
    for t in free_idx[:: max(1, free_idx.size // 12 or 1)][:12]:  # 均匀抽最多 12 个自碰无碰路点
        fsph, fli = _arm_spheres(oracle, traj[int(t)])
        for i, j, _ in _self_collision_pairs(fsph, fli, buffer, link_ignore=link_ignore):
            ignore.add((i, j))
    if free_idx.size == 0:
        print("[viz] 警告：整条轨迹每个路点都自碰，无法学习非相邻设计贴合对忽略集，自碰对可能有误报。")

    print(f"[viz] batch {b}: {coll_idx.size} 个碰撞路点，逐个弹窗（关闭当前窗口看下一个）。")
    for t in coll_idx.tolist():
        dw, ds = float(dw_all[t]), float(ds_all[t])
        kinds = []
        if dw > coll_tol:
            kinds.append("世界碰撞")
        if ds > coll_tol:
            kinds.append("自碰撞")
        kind = "+".join(kinds) if kinds else "未知"

        sph, li = _arm_spheres(oracle, traj[t])
        S = int(sph.shape[0])
        hit_sb, hit_b = _sphere_hits_boxes(sph, cuboids, buffer)
        hit_sm, hit_m = _sphere_hits_meshes(sph, meshes_world, buffer)
        hit_world = hit_sb | hit_sm                             # 球碰盒 或 碰 mesh

        # 每球颜色：默认灰；世界碰撞球=红；自碰对(低link=红/高link=绿)
        colors = [list(_COLOR_GRAY) for _ in range(S)]
        for si in range(S):
            if hit_world[si]:
                colors[si] = list(_COLOR_HIT_ARM)

        self_pairs = []
        if ds > coll_tol:
            self_pairs = _self_collision_pairs(sph, li, buffer, ignore=ignore, link_ignore=link_ignore)
            for i, j, gap in self_pairs:
                lo, hi = (i, j) if li[i] <= li[j] else (j, i)
                colors[lo] = list(_COLOR_HIT_ARM)               # 红
                colors[hi] = list(_COLOR_HIT_OBS)               # 绿

        # 打印诊断
        msg = (f"[viz] batch {b} 路点 {t + 1}/{T}: 类型={kind}  "
               f"dw={dw:.4f} ds={ds:.4f}  世界碰撞球={int(hit_world.sum())}")
        if ds > coll_tol:
            if self_pairs:
                top = self_pairs[:5]
                desc = "; ".join(
                    f"{idx2name.get(int(li[i]), li[i])}#{i}<->{idx2name.get(int(li[j]), li[j])}#{j}(gap={g:.4f})"
                    for i, j, g in top)
                more = "" if len(self_pairs) <= 5 else f" 等{len(self_pairs)}对"
                msg += f"  自碰球对[{len(self_pairs)}]: {desc}{more}"
            else:
                msg += "  自碰球对: 0（可能全在忽略集/深穿透，红=低link 绿=高link 未标出）"
        print(msg)

        geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)]
        for bi, box in enumerate(cuboids):
            geoms.append(_box_geom(o3d, box, hit=bool(hit_b[bi])))
        for mi, mw in enumerate(meshes_world):
            g = _mesh_geom(o3d, mw, hit=bool(hit_m[mi]))
            if g is not None:
                geoms.append(g)
        for si in range(S):
            cx, cy, cz, rr = (float(v) for v in sph[si])
            ball = o3d.geometry.TriangleMesh.create_sphere(radius=max(rr, 1e-3), resolution=8)
            ball.translate((cx, cy, cz))
            ls = o3d.geometry.LineSet.create_from_triangle_mesh(ball)
            ls.paint_uniform_color(colors[si])
            geoms.append(ls)
        title = (f"batch {b}  路点 {t + 1}/{T}  {kind}  dw={dw:.3f} ds={ds:.3f}  "
                 f"世界碰撞球(红)={int(hit_world.sum())}  自碰对(红/绿)={len(self_pairs)}")
        o3d.visualization.draw_geometries(geoms, window_name=title)
