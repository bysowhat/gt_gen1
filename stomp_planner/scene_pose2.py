"""ScenePose2 —— compute_goal_poses 的【去 Isaac】场景，公开接口与 scene_pose.ScenePose 完全一致。

替代关系（见 docs/compute_goal_poses-说明.md 替代表 + 计划 tingly-exploring-swing.md）：
  · warp raycast_mesh（遮挡判定 visionBlock）   -> 自封装 warp `mesh_query_ray`（piece 局部系，piece_pos=0）
  · ContactSensor + PhysX + Articulation（碰撞）  -> cuRobo RobotWorld 整臂碰撞球 vs 工件 mesh + 自碰撞
  · field 圆锥（机械臂自挡视野）                  -> 解析「球心是否落在镜头前方圆锥内」测试（GPU 批量）
  · USD/omni 装载工件 mesh                        -> trimesh 读 *_part.obj

优化器(optimizer_pose.py)只通过 3 个公开方法访问场景：
    reset(robot_pose, horizontal, piece_pose)
    computeCollisionCost(cam_poses, joints_opt)   -> (cost, joints)
    computeVisionCost_1(...)                       -> (vision_pos_cost, vision_rot_cost, ...)
只要这三者输入/输出形状与 ScenePose 一致，优化器无需任何改动。

其中 computeCollisionCost->getJoints、visionBlock 是本类相对 ScenePose 唯一改写的两块；
visionInsides / visionOrientation / computeVisionCost_1 / computeCollisionCost 外壳 /
generate_start_points / generate_transformation / compute_normal 全为纯 torch，照搬 scene_pose.py。
"""

import os
import sys

import torch
import numpy as np

# 让 gt_gen / 同目录 stomp 模块可被 import（脚本里也会加，这里防御性再加一次）
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for _p in (_THIS, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import time

import opt_math_pose as math
from config_pose import ConfigurationPose as Configuration
from ik_cam_pose import UR12e_t

# getJoints 细分埋点（默认关；设环境变量 POSE_PROFILE=1 才启用）。
# 开启后每次 getJoints 段间会 torch.cuda.synchronize() 累加三段耗时 —— 绝对值会因
# 频繁 sync 偏大，但 build/ik/coll 三段【相对占比】准确，用于定位每次 eval ~130ms 花在哪。
_PROFILE = bool(os.environ.get("POSE_PROFILE"))
_PROF = {"build": 0.0, "ik": 0.0, "coll": 0.0, "n": 0}


# field 圆锥几何（与 scene_pose.ScenePoseCfg.field 一致）：相机正前方的「视野禁区」锥
_FIELD_RADIUS = 0.335 / 2          # 锥底半径 (m)
_FIELD_HEIGHT = 0.4                # 锥高 = 镜头到锥底的视线距离 (m)
# 解析推导（见计划文档「field 圆锥」一节）：锥顶在镜头(cam_pos)、轴沿相机 +z(视线)、
# 从 t=0 处半径 0 线性张到 t=_FIELD_HEIGHT 处半径 _FIELD_RADIUS，半角 atan(r/h)≈22.7°。
# 仅对会摆进视野的近端大臂 Link1/Link2/Link3 做检测（与原 contact 过滤器一致）。
_FIELD_LINKS = ("Link1", "Link2", "Link3")


# ---------------------------------------------------------------------------
# warp 批量首次命中 raycast（替代 isaaclab.utils.warp.raycast_mesh）
# ---------------------------------------------------------------------------
_WP_READY = False
_RAYCAST_KERNEL = None


def _ensure_warp():
    global _WP_READY
    if not _WP_READY:
        import warp as wp
        wp.init()
        _WP_READY = True


def _raycast_kernel():
    """编译并缓存：每条射线查 mesh 最近命中距离 t（无命中写 -1）。"""
    global _RAYCAST_KERNEL
    if _RAYCAST_KERNEL is not None:
        return _RAYCAST_KERNEL
    import warp as wp

    @wp.kernel
    def _rk(mesh: wp.uint64, starts: wp.array(dtype=wp.vec3), dirs: wp.array(dtype=wp.vec3),
            max_t: wp.float32, t_out: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        q = wp.mesh_query_ray(mesh, starts[tid], dirs[tid], max_t)
        if q.result:
            t_out[tid] = q.t
        else:
            t_out[tid] = -1.0

    _RAYCAST_KERNEL = _rk
    return _rk


class ScenePose2:
    def __init__(self, cfg: Configuration, num_envs=None, device="cuda",
                 obj_path: str = "", robot_cfg_path: str = "", retract=None):
        self.cfg = cfg
        self.device = device
        self.num_envs = cfg.num_envs if num_envs is None else num_envs

        # —— 与 ScenePose.__init__ 一致的纯 torch 配置 ——
        self.num_steps = cfg.num_steps
        '''
        ┌─────────────────────────┬───────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
        │          变量            │        定义（config_pose.py）          │                                                       含义                                                       │
        ├─────────────────────────┼───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
        │ insides_cost_weight     │ 19 / dist_limit（=19/0.5=38）          │ "工件内部"代价：惩罚相机位姿落到工件内部/穿模，或相机离焊缝太近钻进实体里                                        │
        ├─────────────────────────┼───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
        │ space_cost_weight       │ 21 / dist_limit（=42）                 │ "视野空间"代价：惩罚相机与目标的距离偏离期望工作距离（太远/太近，看不清或超出景深）                              │
        ├─────────────────────────┼───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
        │ block_cost_weight       │ 22 / dist_limit（=44）                 │ "遮挡"代价：惩罚焊缝被工件自身或环境挡住（视线被 block），配合 block_radius=0.04、num_block_pts=6 做遮挡采样检测 │
        ├─────────────────────────┼───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
        │ orientation_cost_weight │ 23 / orientation_limit（=23/30≈0.77）  │ "方向/朝向"代价：惩罚相机光轴与焊缝法向的夹角偏离理想观测角（约 45°），orientation_limit=30 是软上限（度）       │
        └─────────────────────────┴───────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
        '''
        self.insides_cost_weight = cfg.insides_cost_weight
        self.space_cost_weight = cfg.space_cost_weight
        self.block_cost_weight = cfg.block_cost_weight
        self.orientation_cost_weight = cfg.orientation_cost_weight
        # 接受门槛专用的朝向放宽角度(deg)；0=不放宽=gate 与全量朝向代价一致(向后兼容)
        self.orient_gate_relax = float(getattr(cfg, "orient_gate_relax", 0.0))
        '''
        ┌───────────────┬──────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
        │     变量       │      值      │                                                                             作用                                                                              │
        ├───────────────┼──────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
        │ block_radius  │ 0.04（4 cm）  │ 遮挡判定的"容差半径"：从相机到焊缝的视线附近，用一个半径 4cm 的球（或圆柱）去检测有没有障碍物挡在视线上。半径越大，判定"被挡"越严格（视线周围一圈都得空出来） │
        ├───────────────┼──────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
        │ num_block_pts │ 6            │ 沿视线的采样点数：把"相机→焊缝"这条视线均匀切成 6 个采样点，逐点检查是否落进工件/障碍物里。点数越多，遮挡检测越细但越慢                                       │
        └───────────────┴──────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
        '''
        self.block_radius = cfg.block_radius
        self.num_block_pts = cfg.num_block_pts

        # 机器人在 cuRobo 基座系：基座即原点（无世界 spawn 偏移）
        self.robot_init_pos = torch.zeros((1, 3), dtype=torch.float, device=self.device)
        self.robot_init_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float, device=self.device)
        self.robot_base_inv_pose = torch.zeros((1, 7), device=self.device)
        self.robot_pose_init = torch.zeros((7,), device=self.device)
        self.horizontal = 0
        self.joints_lower_limit = torch.empty(0)
        self.joints_upper_limit = torch.empty(0)

        # IK 越界回退构型（仅作占位，影响极小）：取 retract
        if retract is None:
            retract = cfg.__dict__.get("retract", None)
        if retract is None:
            try:
                from gt_gen.config import load_config
                retract = load_config().retract_config
            except Exception:
                retract = [0.0] * 6
        self.initial_joint_pos = torch.tensor([list(map(float, retract))], dtype=torch.float, device=self.device)

        # —— 视锥（FOV）六面体法向，照搬 ScenePose.__init__ ——
        scl = cfg.scl
        scl_z = cfg.scl_z
        self.p1 = torch.tensor([ 0.135 * scl, -0.20 * 5/6 * scl,   0.4], dtype=torch.float, device=self.device)
        self.p2 = torch.tensor([-0.135 * scl, -0.20 * 5/6 * scl,   0.4], dtype=torch.float, device=self.device)
        self.p3 = torch.tensor([-0.135 * scl,  0.20 * 5/6 * scl,   0.4], dtype=torch.float, device=self.device)
        self.p4 = torch.tensor([ 0.135 * scl,  0.20 * 5/6 * scl,   0.4], dtype=torch.float, device=self.device)
        self.p5 = torch.tensor([ 0.135 * scl + ( 0.140 * scl_z * scl), -0.20 * 5/6 * scl + (-0.185 * 5/6 * scl_z * scl),  0.4 * (1 + scl_z)], dtype=torch.float, device=self.device)
        self.p6 = torch.tensor([-0.135 * scl + (-0.140 * scl_z * scl), -0.20 * 5/6 * scl + (-0.185 * 5/6 * scl_z * scl),  0.4 * (1 + scl_z)], dtype=torch.float, device=self.device)
        self.p7 = torch.tensor([-0.135 * scl + (-0.140 * scl_z * scl),  0.20 * 5/6 * scl + ( 0.185 * 5/6 * scl_z * scl),  0.4 * (1 + scl_z)], dtype=torch.float, device=self.device)
        self.p8 = torch.tensor([ 0.135 * scl + ( 0.140 * scl_z * scl),  0.20 * 5/6 * scl + ( 0.185 * 5/6 * scl_z * scl),  0.4 * (1 + scl_z)], dtype=torch.float, device=self.device)
        self.normal = self.compute_normal(self.p1, self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8)

        self.print = False
        self.original_costs = False

        # —— 工件 mesh：trimesh 读 obj（piece 局部系） ——
        self._load_piece(obj_path)
        # —— cuRobo 碰撞（RobotWorld：整臂碰撞球 + mesh 世界 + 自碰撞） ——
        self._init_curobo(robot_cfg_path)

        print("ScenePose2（cuRobo + warp）就绪")

    # ------------------------------------------------------------------ 资源
    def _load_piece(self, obj_path):
        import trimesh
        if not obj_path or not os.path.isfile(obj_path):
            raise FileNotFoundError(f"工件 obj 不存在: {obj_path}")
        m = trimesh.load(obj_path, force="mesh", process=False)
        self._verts = np.asarray(m.vertices, dtype=np.float64)        # (V,3) piece 系
        self._faces = np.asarray(m.faces, dtype=np.int64)             # (F,3)
        self._verts_list = self._verts.tolist()
        self._faces_list = self._faces.tolist()

        # warp mesh（raycast 用，piece 系；piece_pos=0 故无需变换）
        self._build_wp_mesh()

    def _build_wp_mesh(self, extra_verts=None, extra_faces=None):
        """（重）建 raycast 用 warp mesh（piece 系）。extra_* 给定时把额外遮挡体
        （障碍物，须同在 piece 系）拼进工件 mesh，使 visionBlock 遮挡判定含障碍物。"""
        import warp as wp
        _ensure_warp()
        verts = self._verts.astype(np.float32)
        faces = self._faces.astype(np.int32)
        if extra_verts is not None and extra_faces is not None and len(extra_verts):
            ev = np.asarray(extra_verts, dtype=np.float32).reshape(-1, 3)
            ef = np.asarray(extra_faces, dtype=np.int64).reshape(-1, 3) + len(verts)
            verts = np.concatenate([verts, ev], axis=0)
            faces = np.concatenate([faces, ef.astype(np.int32)], axis=0)
        self._wp_mesh = wp.Mesh(
            points=wp.array(verts, dtype=wp.vec3, device=self.device),
            indices=wp.array(faces.reshape(-1).astype(np.int32), dtype=wp.int32, device=self.device),
        )

    def set_occluders(self, verts, faces):
        """把额外遮挡体（障碍物 mesh，piece 系，与 self._verts 同框）并入遮挡 raycast
        用的 warp mesh。传空则恢复为仅工件。碰撞世界（self.rw）另行 update_world，两者独立。"""
        self._build_wp_mesh(extra_verts=verts, extra_faces=faces)

    def _init_curobo(self, robot_cfg_path):
        import gt_gen.compat  # noqa: F401  warp/trimesh shim 须在 import curobo 前
        gt_gen.compat.apply_trimesh_shim()
        from curobo.types.base import TensorDeviceType
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import WorldConfig, Mesh
        from curobo.util_file import load_yaml
        from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

        if not robot_cfg_path:
            from gt_gen.config import load_config
            robot_cfg_path = load_config().robot_cfg_path
        self._rd = load_yaml(robot_cfg_path)

        ta = TensorDeviceType()
        self._ta = ta
        # 占位 world（工件在原点）：先实例化 collision 对象，reset 时再按 base 系位姿更新
        m0 = Mesh(name="piece", vertices=self._verts_list, faces=self._faces_list,
                  pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        rw_cfg = RobotWorldConfig.load_from_config(
            self._rd, WorldConfig(mesh=[m0]), tensor_args=ta,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_activation_distance=0.0, n_meshes=6,#n_meshes 这个碰撞世界最多放几个 mesh"
        )
        self.rw = RobotWorld(rw_cfg)
        self._WorldConfig = WorldConfig
        self._Mesh = Mesh

        # 选出 Link1/2/3 的碰撞球掩码（field 锥仅测这些球，与原 contact 过滤器一致）
        kc = self.rw.kinematics.kinematics_config
        idx_map = kc.link_sphere_idx_map.detach().cpu().numpy()       # (S,) 每球所属 link
        name_to_idx = kc.link_name_to_idx_map
        want = [name_to_idx[n] for n in _FIELD_LINKS if n in name_to_idx]
        m = np.isin(idx_map, want)
        self._field_sphere_mask = torch.as_tensor(m, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ reset
    def reset(self, robot_pose: torch.Tensor, horizontal: int, piece_pose: torch.Tensor | None = None):
        """与 ScenePose.reset 同语义：把量换算到工件局部系，按 horizontal 选关节限位，
        并把工件 mesh 灌进 cuRobo 基座系碰撞世界。返回 robot_pose_init。"""
        # 换算到工件局部系（纯 math，与 ScenePose 一致）
        if piece_pose is not None:
            piece_inv_quat, piece_inv_pos = math.tf_inverse(piece_pose[3:], piece_pose[:3])
            robot_quat, robot_pos = math.tf_combine(piece_inv_quat, piece_inv_pos, robot_pose[3:], robot_pose[:3])
            robot_pose = torch.cat((robot_pos, robot_quat), dim=-1)
        self.robot_pose_init = robot_pose.clone()

        '''
        现在的麻烦是：机器人底座上有两个不同的原点，它们俩差了 90° + 0.26 米：
            - 一个叫 base_link（真正机械臂的第一个关节处）
            - 一个叫 靠上底座（往上抬 0.26 米、又拧了 90° 的另一个参照点）
        '''
        # 去 Isaac：cuRobo FK 根就是 base_link（固定底座 xiaoyu_base_link 已在运动学模型内），
        # 且 robot_pose 由 plan_init_pose 在 base_link 系给出（robot_pose = pose7(inv(T_workpiece_in_base))），
        # 本就是 base_link 位姿——**不能再叠底座 base_transform**，否则把 0.26m+90° 的底座重复计一遍，
        # 令工件/相机/IK 帧整体错转 90°（原版 Isaac scene_pose.py 里 root≠base_link 才需要它，这里不需要）。
        robot_base = robot_pose.clone()
        robot_base_inv_quat, robot_base_inv_pos = math.tf_inverse(robot_base[3:].unsqueeze(0), robot_base[:3].unsqueeze(0))
        self.robot_base_inv_pose = torch.cat((robot_base_inv_pos, robot_base_inv_quat), dim=-1)   # piece->base_link

        # 选关节限位
        self.horizontal = horizontal
        if self.horizontal == 0:
            self.joints_lower_limit = torch.tensor(self.cfg.joints_lower_limit_horizontal, device=self.device)
            self.joints_upper_limit = torch.tensor(self.cfg.joints_upper_limit_horizontal, device=self.device)
        elif self.horizontal == 1:
            self.joints_lower_limit = torch.tensor(self.cfg.joints_lower_limit_vertical, device=self.device)
            self.joints_upper_limit = torch.tensor(self.cfg.joints_upper_limit_vertical, device=self.device)
        else:
            self.joints_lower_limit = torch.tensor(self.cfg.joints_lower_limit_all, device=self.device)
            self.joints_upper_limit = torch.tensor(self.cfg.joints_upper_limit_all, device=self.device)

        # 把工件 mesh 按 piece->base_link 位姿摆进 cuRobo 碰撞世界
        pose = self.robot_base_inv_pose[0].detach().cpu().tolist()    # [x,y,z, qw,qx,qy,qz]
        m = self._Mesh(name="piece", vertices=self._verts_list, faces=self._faces_list, pose=pose)
        self.rw.update_world(self._WorldConfig(mesh=[m]))

        return self.robot_pose_init

    # ------------------------------------------------------------------ 碰撞代价
    def computeCollisionCost(self, cam_poses: torch.Tensor, joints_opt: torch.Tensor):
        """与 ScenePose.computeCollisionCost 完全一致（纯 reshape 外壳）。"""
        B, R = cam_poses.shape[:2]
        cam_poses = cam_poses.reshape(B * R, 7)
        joints_opt = joints_opt.unsqueeze(-2).expand(-1, R, -1).reshape(B * R, 6)
        cost, joints = self.getJoints(cam_poses, joints_opt)
        cost = cost.reshape(B, R)
        joints = joints.reshape(B, R, 6)
        return cost, joints

    def getJoints(self, poses: torch.Tensor, joints_opt: torch.Tensor):
        """据 pose 解析 IK 出关节角，并算碰撞代价。
        保留 ScenePose.getJoints 的 IK/选解逻辑；仅把「PhysX 接触」碰撞换成 cuRobo 批量判定 +
        解析 field 锥。返回 cost (M,) ∈{0,1}，joints (M,6)。"""
        M = poses.shape[0]
        K = 2

        # cam: piece 系 -> base_link 系
        robot_inv_pose = self.robot_base_inv_pose.clone().repeat(M, 1)
        cam_quat, cam_pos = math.tf_combine(robot_inv_pose[:, 3:], robot_inv_pose[:, :3], poses[:, 3:], poses[:, :3])

        # 解析 IK（与 ScenePose 一致）
        # 缓存 UR12e_t 实例：其 IK 路径 solve_fairino_ec 只读 cam_inv/da/dd/all_idx（init 后不改，
        # 与 FK 写的 dh_params[:,:,0] 无关），故同一 (M, device) 可跨 eval 复用 —— 免去每次
        # load_config 读两次 yaml + GPU 建 tensor（原占 compute_goal_pose 约 88% 耗时）。数值零影响。
        if _PROFILE:
            torch.cuda.synchronize(); _t0 = time.perf_counter()
        _ik_cache = self.__dict__.setdefault("_ik_solver_cache", {})
        _key = (int(M), str(self.device))
        ik_solver = _ik_cache.get(_key)
        if ik_solver is None:
            ik_solver = UR12e_t(num_envs=M, device=self.device)
            _ik_cache[_key] = ik_solver
            if _PROFILE:
                print(f"[计时][pose-getJoints] 新建 UR12e_t(M={M})，缓存实例数={len(_ik_cache)}")
        if _PROFILE:
            torch.cuda.synchronize(); _t1 = time.perf_counter()
        # ⚠ ik_cam_pose 的解析 IK/FK 其 DH 根是「靠上底座」(base_link 上抬 0.26m + 绕 z 转 90°)，
        #   不是 base_link。上面 cam_quat/cam_pos 在 base_link 系，直接喂 solve_fairino_ec 会因
        #   base_transform(0.26m+90°) 缺失而令关节角整体错转 90°（表现：回放轨迹手臂对不准视锥）。
        #   故 IK 目标须先从 base_link 系换到靠上底座系：cam_up = inv(base_transform) · cam_base_link。
        #   注意只转这一路——下面碰撞锥的 apex/axis(:321) 仍用 base_link 系(与 cuRobo 碰撞世界一致)。
        base_transform = torch.tensor([0.0, 0.0, 0.26, 0.7071068, 0.0, 0.0, 0.7071068],
                                      dtype=cam_quat.dtype, device=self.device)
        bt_inv_quat, bt_inv_pos = math.tf_inverse(base_transform[3:].unsqueeze(0), base_transform[:3].unsqueeze(0))
        ik_quat, ik_pos = math.tf_combine(bt_inv_quat.repeat(M, 1), bt_inv_pos.repeat(M, 1), cam_quat, cam_pos)
        R_mat = math.matrix_from_quat(ik_quat)
        T_mat = self.generate_transformation(ik_pos, R_mat)
        joints_all_ik = ik_solver.solve_fairino_ec(T_mat).float()        # (M, 8, 6)
        joints_all_ik = torch.nan_to_num(joints_all_ik, nan=-20)
        outbound = (joints_all_ik < self.joints_lower_limit) | (joints_all_ik > self.joints_upper_limit)
        outbound = torch.any(outbound, dim=-1)                            # (M, 8)
        rewards = -torch.norm(joints_all_ik - joints_opt.unsqueeze(-2), dim=-1)   # (M, 8)
        rewards[outbound] -= 500
        rewards_max, max_idx = torch.topk(rewards, k=K, dim=1)            # (M, K)
        joints_all_ik = torch.gather(input=joints_all_ik, dim=1, index=max_idx.unsqueeze(-1).expand(-1, -1, 6))  # (M,K,6)
        joints_all_ik = joints_all_ik.reshape(M * K, 6)
        cost = (rewards_max <= -400).int().reshape(M * K)                 # IK 越界/无解 → 已记 1
        outbound = torch.gather(input=outbound, dim=1, index=max_idx).reshape(M * K)
        joints_all_ik[outbound] = self.initial_joint_pos[0:1].expand(int(outbound.int().sum()), -1)

        # 锥顶/轴（每候选位姿一锥；K 个解共享同一相机）
        apex = cam_pos.unsqueeze(1).expand(M, K, 3).reshape(M * K, 3)             # (M*K,3) 镜头位置=锥顶
        cam_axis = math.matrix_from_quat(cam_quat)[:, :, 2]                       # 相机 +z（视线方向）
        axis = cam_axis.unsqueeze(1).expand(M, K, 3).reshape(M * K, 3)            # (M*K,3)

        if _PROFILE:
            torch.cuda.synchronize(); _t2 = time.perf_counter()
        collided = self._collided_batch(joints_all_ik, apex, axis)               # (M*K,) int
        cost = cost + collided
        if _PROFILE:
            torch.cuda.synchronize(); _t3 = time.perf_counter()
            _PROF["build"] += _t1 - _t0
            _PROF["ik"] += _t2 - _t1
            _PROF["coll"] += _t3 - _t2
            _PROF["n"] += 1

        cost = (cost > 0.1).int().reshape(M, K)
        joints_all_ik = joints_all_ik.reshape(M, K, 6)
        cost, min_idx = torch.min(cost, dim=-1)                           # (M,)
        batch_idx = torch.arange(M, device=self.device)
        joints = joints_all_ik[batch_idx, min_idx]
        return cost, joints

    def _collided_batch(self, joints: torch.Tensor, apex: torch.Tensor, axis: torch.Tensor):
        """批量碰撞判定：arm-vs-工件 ∨ 自碰撞 ∨ field 锥内（Link1/2/3）。返回 (N,) int。"""
        N = joints.shape[0]
        q = joints.to(self._ta.device).contiguous()
        kin = self.rw.get_kinematics(q)
        spheres = kin.link_spheres_tensor                                 # (N, S, 4)

        # arm-vs-工件 + 自碰撞（cuRobo，constraint：穿透才 >0）
        d_world = self.rw.get_collision_constraint(spheres.unsqueeze(1)).squeeze(1)   # (N,)
        d_self = self.rw.get_self_collision(spheres.unsqueeze(1)).squeeze(1)          # (N,)
        hit = (d_world > 1e-6) | (d_self > 1e-6)
        hit = hit.to(self.device)

        # field 锥：仅 Link1/2/3 球
        sph = spheres.to(self.device)
        sub = sph[:, self._field_sphere_mask, :]                          # (N, Sn, 4)
        centers, radii = sub[..., :3], sub[..., 3]                        # (N,Sn,3), (N,Sn)
        rel = centers - apex.unsqueeze(1)                                 # (N,Sn,3)
        t = (rel * axis.unsqueeze(1)).sum(-1)                             # (N,Sn) 轴向投影
        proj = t.unsqueeze(-1) * axis.unsqueeze(1)
        radial = torch.norm(rel - proj, dim=-1)                          # (N,Sn) 到轴距离
        cone_r = torch.clamp(t, min=0.0) / _FIELD_HEIGHT * _FIELD_RADIUS  # 该处锥半径
        inside = (t >= -radii) & (t <= _FIELD_HEIGHT + radii) & (radial <= cone_r + radii) & (radii > 1e-6)
        cone_hit = inside.any(dim=-1)                                     # (N,)

        return (hit | cone_hit).int()

    # ------------------------------------------------------------------ 视觉代价（_1 路径）
    def computeVisionCost_1(self, seam_line, seam_tangent, seam_limits, cam_poses, idx, block_mask,
                            num_randoms, original_costs: bool = False):
        """照搬 ScenePose.computeVisionCost_1（纯 torch；内部调 visionInsides/visionBlock/visionOrientation）。"""
        B = idx.shape[0]
        start_idx = idx[:, 0]
        end_idx = idx[:, 1]
        length = end_idx - start_idx
        idx_array = torch.arange(B, device=self.device)
        self.original_costs = original_costs
        self.print = False

        def computeCostSingle(x, i):
            x = x.reshape(num_randoms, int(length[i] / num_randoms))
            return x.sum(dim=-1)

        mask_idx, costs, costs_o = self.visionInsides(seam_line, cam_poses)
        insides_cost = [computeCostSingle(costs[s:e], int(i)) for s, e, i in zip(start_idx, end_idx, idx_array)]
        insides_cost = torch.stack(insides_cost, dim=0)
        insides_cost /= self.num_steps
        if self.original_costs:
            assert costs_o is not None
            insides_cost_o = [computeCostSingle(costs_o[s:e], int(i)) for s, e, i in zip(start_idx, end_idx, idx_array)]
            insides_cost_o = torch.stack(insides_cost_o, dim=0)
            insides_cost_o /= self.num_steps

        unblock, costs = self.visionBlock(seam_line, cam_poses[:, :3])
        block_cost = [computeCostSingle(costs[s:e], int(i)) for s, e, i in zip(start_idx, end_idx, idx_array)]
        block_cost = torch.stack(block_cost, dim=0)
        block_cost /= self.num_steps

        costs, costs_o, costs_gate = self.visionOrientation(seam_line, seam_tangent, seam_limits, cam_poses, block_mask)
        orientation_cost = [computeCostSingle(costs[s:e], int(i)) for s, e, i in zip(start_idx, end_idx, idx_array)]
        orientation_cost = torch.stack(orientation_cost, dim=0)
        orientation_cost /= self.num_steps
        # 软上限 gate：朝向放宽带后的 orientation 代价(始终计算，供接受门槛用；relax=0 时与 orientation_cost 相同)
        orientation_cost_gate = [computeCostSingle(costs_gate[s:e], int(i)) for s, e, i in zip(start_idx, end_idx, idx_array)]
        orientation_cost_gate = torch.stack(orientation_cost_gate, dim=0)
        orientation_cost_gate /= self.num_steps
        if self.original_costs:
            assert isinstance(costs_o, torch.Tensor)
            orientation_cost_o = [computeCostSingle(costs_o[s:e], int(i)) for s, e, i in zip(start_idx, end_idx, idx_array)]
            orientation_cost_o = torch.stack(orientation_cost_o, dim=0)
            orientation_cost_o /= self.num_steps

        vision_pos_cost = (self.insides_cost_weight * insides_cost + self.block_cost_weight * block_cost +
                           self.orientation_cost_weight * orientation_cost)
        vision_rot_cost = self.insides_cost_weight * insides_cost
        # gate：把 orientation 换成放宽带版本(insides/block 不变)——“能不能看到”的接受门槛专用
        vision_pos_cost_gate = (self.insides_cost_weight * insides_cost + self.block_cost_weight * block_cost +
                                self.orientation_cost_weight * orientation_cost_gate)
        if self.original_costs:
            vision_pos_cost_o = (self.insides_cost_weight * insides_cost_o + self.block_cost_weight * block_cost +
                                 self.orientation_cost_weight * orientation_cost_o)
            vision_rot_cost_o = self.insides_cost_weight * insides_cost_o
        else:
            vision_pos_cost_o = None
            vision_rot_cost_o = None
        return vision_pos_cost, vision_rot_cost, vision_pos_cost_o, vision_rot_cost_o, vision_pos_cost_gate

    def visionInsides(self, seam_line: torch.Tensor, cam_poses: torch.Tensor):
        """照搬 ScenePose.visionInsides（纯 torch）。判焊缝点是否在相机 FOV 六面体内。"""
        N = seam_line.shape[0]
        normals = self.normal.clone()
        normals = math.quat_apply(cam_poses[..., 3:].unsqueeze(-2).repeat(1, 6, 1),
                                  normals.unsqueeze(0).repeat(N, 1, 1))
        p1 = math.quat_apply(cam_poses[..., 3:], self.p1.clone().unsqueeze(0).repeat(N, 1)) + cam_poses[..., :3]
        p7 = math.quat_apply(cam_poses[..., 3:], self.p7.clone().unsqueeze(0).repeat(N, 1)) + cam_poses[..., :3]
        vector_1 = seam_line - p1
        vector_7 = seam_line - p7
        vectors = torch.cat((vector_1.unsqueeze(-2).repeat(1, 3, 1), vector_7.unsqueeze(-2).repeat(1, 3, 1)), dim=-2)
        mask_idx = torch.arange(N, device=self.device)
        vectors = vectors.reshape(N * 6, 3)
        normals = normals.reshape(N * 6, 3)
        dists = -torch.bmm(vectors.unsqueeze(-2), normals.unsqueeze(-1)).squeeze(-1).squeeze(-1).reshape(N, 6)
        if self.original_costs:
            costs_o, _ = torch.max(dists[:, :-1], dim=-1)
        else:
            costs_o = None
        dists = torch.clamp(dists, min=0)
        costs = dists.sum(-1)
        in_idx = (costs <= 1e-8)
        mask_idx = mask_idx[in_idx]
        return mask_idx, costs, costs_o

    def visionBlock(self, seam_line: torch.Tensor, pos_list: torch.Tensor):
        """判焊缝点是否被工件遮挡：从相机向焊缝点发一束射线（主射线 + num_block_pts 偏移射线），
        看是否在到达焊缝点前先命中工件 mesh。逻辑与 ScenePose.visionBlock 完全一致，
        仅把 isaaclab raycast_mesh 换成自封装 warp raycast；piece 固定原点故 piece_pos=0。"""
        M = seam_line.shape[0]
        piece_pos = torch.zeros((1, 3), dtype=torch.float, device=self.device)   # 工件在原点

        # from cam to seam line（version 2，与原一致）
        ray_dirs = seam_line - pos_list
        ray_dirs = ray_dirs / torch.clamp(torch.norm(ray_dirs, dim=-1, keepdim=True), min=1e-8)

        r = self.block_radius
        N = self.num_block_pts
        points = self.generate_start_points(ray_dirs, pos_list, r, N)            # (M, N, 3)
        pos_list = pos_list.unsqueeze(-2)
        pos_list = torch.cat((pos_list, pos_list + points), dim=-2)              # (M, N+1, 3)
        ray_starts = piece_pos.unsqueeze(-2) + pos_list                          # (M, N+1, 3)
        ray_dirs = seam_line.unsqueeze(-2) - pos_list
        ray_dirs = ray_dirs / torch.clamp(torch.norm(ray_dirs, dim=-1, keepdim=True), min=1e-8)

        hit_positions = self._raycast(ray_starts, ray_dirs, max_dist=100.0)      # (M, N+1, 3)，miss=inf

        costs = torch.zeros((M, N + 1), dtype=torch.float, device=self.device)
        unblock = torch.ones((M,), dtype=torch.int, device=self.device).bool()
        seam_line = seam_line + piece_pos
        dist_c = torch.norm(hit_positions - seam_line.unsqueeze(-2), dim=-1)     # (M, N+1)
        block = ~(dist_c < 1e-2)
        dist_c[torch.isinf(dist_c)] = 0
        dist_c[dist_c > 2] = 0
        costs[block] = 1 - dist_c[block] / 2
        costs = torch.sum(costs, dim=-1)
        block = torch.any(block, dim=-1)
        unblock[block] = False
        return unblock, costs

    def _raycast(self, ray_starts: torch.Tensor, ray_dirs: torch.Tensor, max_dist: float = 100.0):
        """批量首次命中：返回命中点 (..., 3)，无命中处为 inf（与 isaaclab.raycast_mesh 语义一致）。"""
        import warp as wp
        shape = ray_starts.shape
        o = ray_starts.reshape(-1, 3).contiguous().to(self.device, torch.float32)
        d = ray_dirs.reshape(-1, 3).contiguous().to(self.device, torch.float32)
        n = o.shape[0]
        t_out = torch.empty((n,), dtype=torch.float32, device=self.device)
        wp.launch(
            _raycast_kernel(),
            dim=n,
            inputs=[self._wp_mesh.id,
                    wp.from_torch(o, dtype=wp.vec3),
                    wp.from_torch(d, dtype=wp.vec3),
                    float(max_dist),
                    wp.from_torch(t_out)],
            device=self.device,
        )
        hit = o + t_out.unsqueeze(-1) * d
        hit[t_out < 0] = float("inf")
        return hit.reshape(shape)

    def visionOrientation(self, seam_line, seam_tangent, seam_limits, cam_poses, block_mask):
        """照搬 ScenePose.visionOrientation（纯 torch）。观测方向相对两面角的代价。"""
        M = seam_line.shape[0]
        tgt_cost_scale = 0.3
        pl_cost_scale = 0.2
        nm_cost_scale = 1 - tgt_cost_scale - pl_cost_scale

        mid_vector = seam_limits[:, 0] + seam_limits[:, 1]
        mid_vector = mid_vector / (torch.norm(mid_vector, dim=-1, keepdim=True) + 1e-8)

        cam_pos = cam_poses[:, :3]
        direction = cam_pos - seam_line
        r = self.orient_gate_relax        # 接受门槛的朝向放宽角度(deg)；每子项目标带各放宽 r 度
        tgt_cost = torch.zeros((M,), dtype=torch.float, device=self.device)
        tgt_cost_o = torch.zeros((M,), dtype=torch.float, device=self.device)
        tgt_cost_gate = torch.zeros((M,), dtype=torch.float, device=self.device)
        proj_tgt = torch.bmm(direction.unsqueeze(-2), seam_tangent.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        if block_mask.int().sum() > 0:
            cos_theta_tgt = torch.clamp(proj_tgt[block_mask] / (torch.norm(direction[block_mask], dim=-1) + 1e-8), -1, 1).abs()
            theta_tgt = torch.acos(cos_theta_tgt) / torch.pi * 180
            theta_tgt = torch.max(theta_tgt - 60, 30 - theta_tgt)
            if self.original_costs:
                tgt_cost_o[block_mask] = theta_tgt.clone() * self.num_steps
            tgt_cost[block_mask] = torch.clamp(theta_tgt, min=0) * self.num_steps
            # 放宽带 [30-r, 60+r]：目标带对称平移 r，等价于未裁剪值减 r 再裁剪
            tgt_cost_gate[block_mask] = torch.clamp(theta_tgt - r, min=0) * self.num_steps

        direction_pl = direction - proj_tgt.unsqueeze(-1) * seam_tangent
        direction_pl = direction_pl / (torch.norm(direction_pl, dim=-1, keepdim=True) + 1e-8)
        soll_dist = torch.bmm(mid_vector.unsqueeze(-2), seam_limits[:, 1].unsqueeze(-1)).squeeze(-1).squeeze(-1)
        soll_dist = torch.acos(torch.clamp(soll_dist, -1, 1)) / torch.pi * 180
        dist = torch.bmm(direction_pl.unsqueeze(-2), mid_vector.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        dist = torch.acos(torch.clamp(dist, -1, 1)) / torch.pi * 180
        pl_cost = dist - soll_dist
        pl_cost_gate = torch.clamp((dist - soll_dist) - r, min=0)      # 放宽阈值 soll+r
        if self.original_costs:
            pl_cost_o = pl_cost.clone()
        pl_cost = torch.clamp(pl_cost, min=0)

        seam_normal = torch.cross(seam_limits, seam_tangent.unsqueeze(-2).expand(-1, 2, -1), dim=-1)
        seam_normal = seam_normal / torch.norm(seam_normal, dim=-1, keepdim=True).clamp(min=1e-8)
        block_mask_neg = ~block_mask
        nm_cost_0 = torch.zeros((M,), dtype=torch.float, device=self.device)
        nm_cost_1 = torch.zeros((M,), dtype=torch.float, device=self.device)
        nm_cost = torch.zeros((M,), dtype=torch.float, device=self.device)
        nm_cost_0_gate = torch.zeros((M,), dtype=torch.float, device=self.device)
        nm_cost_1_gate = torch.zeros((M,), dtype=torch.float, device=self.device)
        nm_cost_gate = torch.zeros((M,), dtype=torch.float, device=self.device)
        nm_cost_o_0 = torch.zeros((M,), dtype=torch.float, device=self.device)
        nm_cost_o_1 = torch.zeros((M,), dtype=torch.float, device=self.device)
        nm_cost_o = torch.zeros((M,), dtype=torch.float, device=self.device)
        if block_mask_neg.int().sum() > 0:
            direction_n = direction[block_mask_neg] / torch.clamp(
                torch.norm(direction[block_mask_neg], dim=-1, keepdim=True), min=1e-8)
            dists_0 = torch.bmm(seam_normal[block_mask_neg, 0].unsqueeze(-2), direction_n.unsqueeze(-1)).squeeze(-1).squeeze(-1)
            dists_0 = torch.acos(torch.clamp(dists_0, -1, 1).abs()) / torch.pi * 180
            nm_cost_0[block_mask_neg] = torch.abs(dists_0 - 45) - 15
            # 放宽带 45°±(15+r)
            nm_cost_0_gate[block_mask_neg] = torch.clamp(torch.abs(dists_0 - 45) - 15 - r, min=0)
            if self.original_costs:
                nm_cost_o_0 = nm_cost_0.clone()
            nm_cost_0 = torch.clamp(nm_cost_0, min=0)
            dists_1 = torch.bmm(seam_normal[block_mask_neg, 1].unsqueeze(-2), direction_n.unsqueeze(-1)).squeeze(-1).squeeze(-1)
            dists_1 = torch.acos(torch.clamp(dists_1, -1, 1).abs()) / torch.pi * 180
            nm_cost_1[block_mask_neg] = torch.abs(dists_1 - 45) - 15
            nm_cost_1_gate[block_mask_neg] = torch.clamp(torch.abs(dists_1 - 45) - 15 - r, min=0)
            if self.original_costs:
                nm_cost_o_1 = nm_cost_1.clone()
                nm_cost_o = nm_cost_o_0 + nm_cost_o_1
            nm_cost_1 = torch.clamp(nm_cost_1, min=0)
            nm_cost = nm_cost_0 + nm_cost_1
            nm_cost_gate = nm_cost_0_gate + nm_cost_1_gate

        orientation_cost = tgt_cost * tgt_cost_scale + pl_cost * pl_cost_scale + nm_cost * nm_cost_scale
        orientation_cost_gate = (tgt_cost_gate * tgt_cost_scale + pl_cost_gate * pl_cost_scale
                                 + nm_cost_gate * nm_cost_scale)
        if self.original_costs:
            orientation_cost_o = tgt_cost_o * tgt_cost_scale + pl_cost_o * pl_cost_scale + nm_cost_o * nm_cost_scale
        else:
            orientation_cost_o = None
        return orientation_cost, orientation_cost_o, orientation_cost_gate

    # ------------------------------------------------------------------ Utils（照搬 ScenePose）
    def generate_transformation(self, xyz, R):
        B = xyz.shape[0]
        T = torch.zeros((B, 4, 4), dtype=torch.float64, device=self.device)
        T[:, :3, :3] = R[:, :, :].double()
        T[:, :3, 3] = xyz.double()
        T[:, 3, 3] = 1
        return T

    def normal_vector(self, p1, p2, p3):
        return torch.cross(p2 - p1, p3 - p1, dim=-1)

    def compute_normal(self, p1, p2, p3, p4, p5, p6, p7, p8):
        n1 = self.normal_vector(p1, p4, p2); n1 = n1 / torch.norm(n1, p=2, dim=-1)
        n2 = self.normal_vector(p5, p8, p1); n2 = n2 / torch.norm(n2, p=2, dim=-1)
        n3 = self.normal_vector(p6, p5, p2); n3 = n3 / torch.norm(n3, p=2, dim=-1)
        n4 = self.normal_vector(p7, p6, p3); n4 = n4 / torch.norm(n4, p=2, dim=-1)
        n5 = self.normal_vector(p8, p7, p4); n5 = n5 / torch.norm(n5, p=2, dim=-1)
        n6 = self.normal_vector(p5, p6, p8); n6 = n6 / torch.norm(n6, p=2, dim=-1)
        return torch.stack((n1, n2, n3, n4, n5, n6), dim=0)

    def generate_start_points(self, ray_dirs: torch.Tensor, ray_starts: torch.Tensor, r: float, N: int):
        """照搬 ScenePose.generate_start_points：在垂直射线的小圆盘上撒 N 个偏移点 (M,N,3)。"""
        device = ray_dirs.device
        ref1 = torch.tensor([1., 0., 0.], device=device)
        ref2 = torch.tensor([0., 1., 0.], device=device)
        use_ref1 = (torch.abs(ray_dirs[:, 0]) < 0.9).float().unsqueeze(-1)
        ref = use_ref1 * ref1 + (1 - use_ref1) * ref2
        u = torch.cross(ray_dirs, ref, dim=-1); u = u / torch.norm(u, dim=-1, keepdim=True)
        v = torch.cross(ray_dirs, u, dim=-1); v = v / torch.norm(v, dim=-1, keepdim=True)
        theta = torch.linspace(0, 2 * torch.pi, N, device=device)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        points = r * (cos_t[None, :, None] * u[:, None, :] + sin_t[None, :, None] * v[:, None, :])
        return points
