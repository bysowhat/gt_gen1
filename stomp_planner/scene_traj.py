"""
build up a simulation scene for collision detection and raycasting
version: v0
date: 25.09.12
"""

import os
import time

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.sim import SimulationContext
from isaaclab.assets import RigidObjectCfg, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.sensors.ray_caster import RayCaster, patterns
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
# from isaaclab.utils import configclass
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

import omni.usd
from pxr import UsdGeom, Gf, Usd

import gt_math_traj as math
from config_traj import ConfigurationTraj as Configuration
from robot.a1_cfg import A1_CFG
from ik_cam_traj import UR12e_t


# @configclass
class SceneTrajCfg(InteractiveSceneCfg):
    """Config the scene for a1 robots"""
    # robot
    robot = A1_CFG.copy()
    robot.prim_path = "{ENV_REGEX_NS}/robot"
    robot.init_state.pos = (0.0, 0.0, 0.0)
    # robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)

    # piece
    # piece pos: (0, 0, 0)
    # lines: (-0.14, -0.005, 0.08), (0.0, -0.005, 0.08), (0.14, -0.005, 0.08)
    # robot pose: pos = (-0.45, 0, 0)
    # pose: pos = (-0.14, -0.005, 0.08), rpy1 = (-90, 45, 135), rpy2 = (-90, 45, 120)
    
    piece = RigidObjectCfg(
        prim_path = "/World/envs/env_.*/piece",
        spawn = sim_utils.UsdFileCfg(
            usd_path = Configuration().usd_path,
            activate_contact_sensors = True,
            rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True),
            collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled = True)
        ),
        init_state = RigidObjectCfg.InitialStateCfg(
            pos = (0.0, 0.0, 0.0),
            rot = (1.0, 0.0, 0.0, 0.0),
        )
    )

    field = RigidObjectCfg(
        prim_path = "/World/envs/env_.*/field",
        spawn = sim_utils.ConeCfg(
            # activate_contact_sensors = True,
            radius = 0.335 / 2,
            height = 0.4,
            rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True),
            mass_props = sim_utils.MassPropertiesCfg(density = 5000),
            collision_props= sim_utils.CollisionPropertiesCfg(collision_enabled = True),
        ),
        init_state = RigidObjectCfg.InitialStateCfg(
            pos = (0.0, 0.0, 1e6),
            rot = (1.0, 0.0, 0.0, 0.0),
        )
    )

    contact_piece = ContactSensorCfg(prim_path="/World/envs/env_.*/piece",)
    contact_fr_base = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/xiaoyu_arm_base_link")
    contact_magnetic_base = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/xiaoyu_base_link")
    contact_j1 = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/Link1")
    contact_j2 = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/Link2")
    contact_j3 = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/Link3")
    contact_j4 = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/Link4")
    contact_j5 = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/Link5")
    contact_acc = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/xiaoyu_accessory_link")


class SceneTraj:
    def __init__(self, cfg: Configuration, num_envs: int| None = None, device = "cuda"):
        self.cfg = cfg
        self.has_pc = self.cfg.has_pc
        if self.has_pc:
            self.pc = torch.load(self.cfg.pc_path)
            self.pc_scale = torch.load(self.cfg.pc_scale_path)
            self.pc_used = torch.empty(0)
        # self.seam_line = torch.load(self.cfg.seam_line_path)    # (N, 3)
        # self.seam_tangent = torch.load(self.cfg.seam_tangent_path)   # (N, 3)
        # self.seam_limits = torch.load(self.cfg.seam_limits_path)    # (N, 2, 3)

        self.dir_cost_weight = self.cfg.dir_cost_weight
        self.visible_cost_weight = self.cfg.visible_cost_weight
        if num_envs is None:
            self.num_envs = self.cfg.num_envs
        else:
            self.num_envs = num_envs
        self.num_timesteps_init = self.cfg.num_timesteps_init
        self.num_timesteps_next = self.cfg.num_timesteps_next

        mul = 4     # 2**1
        # create sim
        self.sim_cfg = SimulationCfg(
            dt = 1/100,
            render_interval = 1,
            # disable_contact_processing = True,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode = "multiply",
                restitution_combine_mode = "multiply",
                static_friction = 1,
                dynamic_friction = 1,
                restitution = 0.0,
            ),
            physx=PhysxCfg(
                gpu_collision_stack_size = int(mul * 2**26),
                gpu_heap_capacity = int(mul * 2**26),
                gpu_temp_buffer_capacity = int(mul * 2**24),
                gpu_max_rigid_contact_count = int(mul * 2**23),
                gpu_max_rigid_patch_count = int(mul * 5 * 2**15),
                gpu_found_lost_pairs_capacity = int(mul * 2**21),
                gpu_found_lost_aggregate_pairs_capacity = int(mul * 2**25),
                gpu_total_aggregate_pairs_capacity = int(mul * 2**21),
                # gpu_max_num_partitions = 8,
                # gpu_max_soft_body_contacts = 2**20,
                # gpu_max_particle_contacts = 2**20
            ),
            device = device
        )
        self.sim = SimulationContext(self.sim_cfg)
        self.device = self.sim.device

        # setup scene
        scene_cfg = SceneTrajCfg(num_envs=self.num_envs, env_spacing=16, replicate_physics=True)
        scene_cfg.piece.spawn.usd_path = self.cfg.usd_path
        scene_cfg.robot.spawn.usd_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "robot", "a1_ur12e.usd")
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()
        self.robot = self.scene.articulations["robot"]
        self.piece = self.scene.rigid_objects["piece"]
        self.field = self.scene.rigid_objects["field"]
        self.contact_piece = self.scene["contact_piece"]
        self.contact_fr_base = self.scene.sensors["contact_fr_base"]
        self.contact_magnetic_base = self.scene.sensors["contact_magnetic_base"]
        self.contact_j1 = self.scene.sensors["contact_j1"]
        self.contact_j2 = self.scene.sensors["contact_j2"]
        self.contact_j3 = self.scene.sensors["contact_j3"]
        self.contact_j4 = self.scene.sensors["contact_j4"]
        self.contact_j5 = self.scene.sensors["contact_j5"]
        # self.contact_acc = self.scene.sensors["contact_acc"]


        # initialize raycasting computed with warp
        def get_all_meshes_under(prim):
            meshes = []
            for child in prim.GetChildren():
                if child.IsA(UsdGeom.Mesh):
                    meshes.append(child)
                else:
                    # Recursively look inside nested Xforms
                    meshes.extend(get_all_meshes_under(child))
            return meshes
                
        def combine_meshes_to_warp(xform_prim, device="cuda"):
            mesh_prims = get_all_meshes_under(xform_prim)

            all_points = []
            all_indices = []
            vertex_offset = 0

            for mesh_prim in mesh_prims:
                usd_mesh = UsdGeom.Mesh(mesh_prim)

                # --- Transform vertices to world coordinates ---
                xformable = UsdGeom.Xformable(mesh_prim)
                matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

                local_points = np.asarray(usd_mesh.GetPointsAttr().Get(), dtype=np.float32).reshape(-1, 3)
                world_points = np.array([
                    matrix.Transform(Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))) 
                    for v in local_points
                ], dtype=np.float32)
                # ----------------------------------------------

                indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

                all_points.append(world_points)
                all_indices.append(indices + vertex_offset)  # offset indices
                vertex_offset += world_points.shape[0]

            all_points = np.vstack(all_points)
            all_indices = np.hstack(all_indices)

            # print(all_points.shape)
            # print(all_points)
            # print(all_points - self.piece.data.root_pos_w[0].clone().unsqueeze(0).cpu().numpy())

            warp_mesh = convert_to_warp_mesh(all_points, all_indices, device=device)
            return warp_mesh
        
        stage = omni.usd.get_context().get_stage()
        xform_prim = stage.GetPrimAtPath("/World/envs/env_0/piece")
        if xform_prim.IsInstance():
            xform_prim = xform_prim.GetPrototype()  # follow to actual mesh definition
        self.warp_mesh = combine_meshes_to_warp(xform_prim)

        # piece_pos = self.piece.data.root_pos_w[0].clone()
        # ray_starts= piece_pos.unsqueeze(0) + (torch.rand((250, 3), dtype=torch.float, device=self.device) * 0.2 
        #                                       + torch.tensor([[0, 0, 0.2]], dtype=torch.float, device=self.device))
        # ray_dirs = torch.tensor([0, -0.3, -1], dtype=torch.float, device=self.device).unsqueeze(0).repeat(250, 1)
        # ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        # hit_positions, _, _, _ = raycast_mesh(
        #     ray_starts,
        #     ray_dirs,
        #     self.warp_mesh,
        #     max_dist=1.0,
        #     return_distance=False,
        #     return_normal=False,
        #     return_face_id=False
        # )
        # print(hit_positions - piece_pos.unsqueeze(0))

        ## init
        # robot
        self.horizontal =self.cfg.horizontal
        if self.horizontal == 0:
            self.joints_lower_limit = torch.tensor(self.cfg.joints_lower_limit_horizontal, device=self.device)
            self.joints_upper_limit = torch.tensor(self.cfg.joints_upper_limit_horizontal, device=self.device)
        elif self.horizontal == 1:
            self.joints_lower_limit = torch.tensor(self.cfg.joints_lower_limit_vertical, device=self.device)
            self.joints_upper_limit = torch.tensor(self.cfg.joints_upper_limit_vertical, device=self.device)
        else:
            self.joints_lower_limit = torch.tensor(self.cfg.joints_lower_limit_all, device=self.device)
            self.joints_upper_limit = torch.tensor(self.cfg.joints_upper_limit_all, device=self.device)
        self.robot_init_pos = self.robot.data.root_pos_w.clone()
        self.robot_init_quat = self.robot.data.root_quat_w.clone()
        self.initial_joint_pos = self.robot.data.default_joint_pos.to(device=self.device)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.robot_base_inv_pose = torch.zeros((1, 7), device=self.device)
        self.robot_pose_init = torch.zeros((7,), device=self.device)
        self.j6link_idx = self.robot.find_bodies("Link6")[0][0]
        self.base_link_idx = self.robot.find_bodies("xiaoyu_arm_base_link")[0][0]
        self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # camera
        self.cam_transfer = torch.tensor([[0, 0, 0, 0.5000, -0.5000,  0.5000, -0.5000]], dtype=torch.float, device=self.device)
        self.z_off = torch.tensor([0, 0, scene_cfg.field.spawn.height/2], dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.field_init_pose = torch.cat((self.field.data.root_pos_w.clone(), self.field.data.root_quat_w.clone()), dim=-1)
        self.ones = torch.ones((self.num_envs,), dtype=torch.float, device=self.device)
        self.zeros = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        # print("仿真，启动")

        # # 3d cam: nomal verctor of valid vision space
        # self.p1 = torch.tensor([ 0.135, -0.20,   0.4], dtype=torch.float ,device=self.device)
        # self.p2 = torch.tensor([-0.135, -0.20,   0.4], dtype=torch.float ,device=self.device)
        # self.p3 = torch.tensor([-0.135,  0.20,   0.4], dtype=torch.float ,device=self.device)
        # self.p4 = torch.tensor([ 0.135,  0.20,   0.4], dtype=torch.float ,device=self.device)
        # self.p5 = torch.tensor([ 0.275, -0.385,  0.8], dtype=torch.float ,device=self.device)
        # self.p6 = torch.tensor([-0.275, -0.385,  0.8], dtype=torch.float ,device=self.device)
        # self.p7 = torch.tensor([-0.275,  0.385,  0.8], dtype=torch.float ,device=self.device)
        # self.p8 = torch.tensor([ 0.275,  0.385,  0.8], dtype=torch.float ,device=self.device)
        # self.normal = self.compute_normal(self.p1, self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8)   # (6, 3)

        # 2d cam: nomal verctor of valid vision space
        self.p10 = torch.tensor([ 0.0,    0.0,   0.0], dtype=torch.float ,device=self.device)
        self.p11 = torch.tensor([ 1.375, -1.925, 4.0], dtype=torch.float ,device=self.device)
        self.p12 = torch.tensor([-1.375, -1.925, 4.0], dtype=torch.float ,device=self.device)
        self.p13 = torch.tensor([-1.375,  1.925, 4.0], dtype=torch.float ,device=self.device)
        self.p14 = torch.tensor([ 1.375,  1.925, 4.0], dtype=torch.float ,device=self.device)
        self.normal = self.compute_normal_1(self.p10, self.p11, self.p12, self.p13, self.p14)   # (5, 3)

        
        # seam
        self.seam_median = torch.zeros((3,), dtype=torch.float, device=self.device)


    def selectPC(self, offset:float = 0.015, voxel_size:float = 0.005):
        pc = self.pc.clone()    # (M, 3)
        seam_line = self.seam_line.clone()          # (N, 3)
        seam_limits = self.seam_limits.clone()      # (N, 2, 3)
        seam_tangent = self.seam_tangent.clone()    # (N, 3)
        seam_tangent_s = self.seam_tangent[0] * (2 * (torch.sum((self.seam_line[1] - self.seam_line[0]) 
                                                          * self.seam_tangent[0], dim=-1) > 0).int() - 1)   # (3,)
        seam_tangent_e = self.seam_tangent[-1] * (2 * (torch.sum((self.seam_line[-2] - self.seam_line[-1]) 
                                                          * self.seam_tangent[-1], dim=-1) > 0).int() - 1)  # (3,)
        N = seam_line.shape[0]

        # 计算点云筛选范围
        max_val, _ = torch.max(seam_line, dim=0)    # (3,)
        max_val += offset
        min_val, _ = torch.min(seam_line, dim=0)    # (3,)
        min_val -= offset
        mask = (pc <= max_val) * (pc >= min_val)
        pc = pc[mask]   # (M1, 3)

        # 对点云进行下采样
        # 将点量化到体素格上
        voxel_indices = torch.floor(pc / voxel_size)
        # 将 voxel 索引转为哈希（字符串或整数编码）,哈希常数随意但应互质
        keys = voxel_indices[:, 0] * 73856093 + voxel_indices[:, 1] * 19349663 + voxel_indices[:, 2] * 83492791
        # 获取唯一体素及其第一个点索引
        unique_keys, inverse_indices = torch.unique(keys, return_inverse=True)
        first_indices = torch.zeros(len(unique_keys), dtype=torch.long)
        for i in range(len(unique_keys)):
            first_indices[i] = torch.nonzero(inverse_indices == i, as_tuple=True)[0][0]
        pc = pc[first_indices]      # (M2, 3)   

        # 判断点云是否在端点之间内
        dir_pc_s = pc - seam_line[0].clone().unsqueeze(0)   # (M2, 3)
        dir_pc_e = pc - seam_line[-1].clone().unsqueeze(0)  # (M2, 3)
        M2 = dir_pc_s.shape[0]
        mask_dir_s = (torch.bmm(dir_pc_s.unsqueeze(-2), seam_tangent_s.unsqueeze(0).expand(M2, -1).unsqueeze(-1)
                                ).squeeze(-1).squeeze(-1) >= 0)
        mask_dir_e = (torch.bmm(dir_pc_e.unsqueeze(-2), seam_tangent_e.unsqueeze(0).expand(M2, -1).unsqueeze(-1)
                                ).squeeze(-1).squeeze(-1) >= 0)
        pc = pc[mask_dir_s * mask_dir_e]    # (M3, 3)

        # 判断点云是否在正面区域内
        dir_pc = pc.unsqueeze(1) - seam_line.unsqueeze(0)   # (M3, N, 3)
        M3 = dir_pc.shape[0]
        seam_tangents = seam_tangent.unsqueeze(0).expand(M3, -1, -1)   # (M3, N, 3)
        dir_proj = torch.bmm(dir_pc.reshape(M3 * N, 3).unsqueeze(-2), seam_tangents.reshape(M3 * N, 3).unsqueeze(-1)
                             ).squeeze(-1).squeeze(-1).reshape(M3, N)       # (M3, N)
        proj_min_idx = torch.argmin(dir_proj.abs(), dim=-1)     # (M3,)
        proj_seam_line = seam_line[proj_min_idx]                # (M3, 3)
        proj_seam_tangent = seam_tangent[proj_min_idx]          # (M3, 3)
        proj_seam_limits = seam_limits[proj_min_idx]            # (M3, 2, 3)
        dir_pc = pc - proj_seam_line                            # (M3, 3)
        dir_proj = dir_proj[torch.arange(M3, device=self.device), proj_min_idx]         # (M3, 3)
        dir_plane = dir_pc - dir_proj.unsqueeze(-1) * proj_seam_tangent                 # (M3, 3)
        # 计算各平面内的角度范围
        mid_vector = proj_seam_limits[:,0] + proj_seam_limits[:, 1]         # (M3, 3)
        soll_dist = torch.bmm(mid_vector.unsqueeze(-2), proj_seam_limits[:, 0].unsqueeze(-1)).squeeze(-1).squeeze(-1)    # (M3,)
        ist_dist = torch.bmm(mid_vector.unsqueeze(-2), dir_plane.unsqueeze(-1)).squeeze(-1).squeeze(-1)     # (M3,)
        mask_dist = (ist_dist >= soll_dist)
        self.pc_used = pc[mask_dist].clone()    # (M4, 3)


    def collided(self, env_ids=None):
        if env_ids == None:
            env_ids = self.robot._ALL_INDICES
        force_piece = self.contact_piece.data.net_forces_w[env_ids, 0, :]
        force_magnetic_base = self.contact_magnetic_base.data.net_forces_w[env_ids, 0, :]
        force_fr_base = self.contact_fr_base.data.net_forces_w[env_ids, 0, :]
        force_j1 = self.contact_j1.data.net_forces_w[env_ids, 0, :]
        force_j2 = self.contact_j2.data.net_forces_w[env_ids, 0, :]
        force_j3 = self.contact_j3.data.net_forces_w[env_ids, 0, :]
        force_j4 = self.contact_j4.data.net_forces_w[env_ids, 0, :]
        force_j5 = self.contact_j5.data.net_forces_w[env_ids, 0, :]
        # force_acc = self.contact_acc.data.net_forces_w[env_ids, 0, :]
        return ~(
            torch.all(force_piece == 0, dim=-1) &
            torch.all(force_magnetic_base == 0, dim=-1) 
            & torch.all(force_fr_base == 0, dim=-1) & torch.all(force_j1 == 0, dim=-1) 
            & torch.all(force_j2 == 0, dim=-1) & torch.all(force_j3 == 0, dim=-1) 
            & torch.all(force_j4 == 0, dim=-1) & torch.all(force_j5 == 0, dim=-1)
            # & torch.all(force_acc == 0, dim=-1)
            )


    def reset(self, robot_pose: torch.Tensor, seam_median: torch.Tensor):
        """
        reset scene
        Args:
            robot_pose: (7,)
            seam_median: (3,)
        """
        self.robot_pose_init = robot_pose.clone()

        robot_base = robot_pose.clone()
        # robot_base[2] += 0.26
        base_transform = torch.tensor([0, 0, 0.26, 0.7071068, 0, 0, 0.7071068], dtype=torch.float, device= self.device)
        robot_base_quat, robot_base_pos = math.tf_combine(robot_base[3:], robot_base[:3], base_transform[3:], base_transform[:3])
        robot_base = torch.cat((robot_base_pos, robot_base_quat), dim=-1)
        self.robot_pose_init = robot_base.clone()
        robot_base_inv_quat, robot_base_inv_pos = math.tf_inverse(robot_base[3:].unsqueeze(0), robot_base[:3].unsqueeze(0))
        self.robot_base_inv_pose = torch.cat((robot_base_inv_pos, robot_base_inv_quat), dim=-1)

        robot_pose = robot_pose.unsqueeze(0).repeat(self.num_envs, 1)
        robot_quat, robot_pos = math.tf_combine(self.robot_init_quat.clone(), self.robot_init_pos.clone(), 
                                     robot_pose[:, 3:], robot_pose[:, :3])
        robot_pose = torch.cat((robot_pos, robot_quat), dim=-1)
        self.robot.write_root_pose_to_sim(robot_pose)

        joint_pos = self.robot.data.default_joint_pos
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.sim_cfg.dt)

        self.seam_median = seam_median.clone()


    def computeCollisionCost(self, rollouts: torch.Tensor):
        """
        计算关节角对应的碰撞成本函数
        Args:
            joints: (B, R, D, T)
        """
        B, R, D, T = rollouts.shape
        # print("rollouts shape:", rollouts.shape)
        rollouts = rollouts.transpose(-2, -1)   # (B, R, T, D)
        rollouts = rollouts.reshape(-1, D)
        N = rollouts.shape[0]
        cost_tmp = torch.zeros((N,), device=self.device)

        # print("N:", N)
        # print("num_envs:", self.num_envs)

        # compute collision cost
        times = N // self.num_envs
        rest = N % self.num_envs
        for i in range(times):
            joints = rollouts[i*self.num_envs : (i+1)*self.num_envs]
            for _ in range(1):
                self.robot.set_joint_velocity_target(self.robot_dof_targets)
                self.robot.write_joint_state_to_sim(joints, self.robot_dof_targets)
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.sim_cfg.dt)
            cost_tmp[i*self.num_envs : (i+1)*self.num_envs] = self.collided().int()
        if rest > 0:
            all_idx = torch.arange(rest, device=self.device)
            joints = rollouts[-rest:]
            for _ in range(2):
                self.robot.set_joint_velocity_target(self.robot_dof_targets[all_idx], env_ids=all_idx)
                self.robot.write_joint_state_to_sim(joints, self.robot_dof_targets[all_idx], env_ids=all_idx)
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.sim_cfg.dt)
            cost_tmp[-rest:] = self.collided(all_idx).int()

        # reshape cost_tmp to original
        cost_tmp = cost_tmp.reshape(B, R, T).unsqueeze(-2).repeat(1, 1, D, 1)   # (B, R, D, T)
        return cost_tmp


    def computeVisionCost(self, rollouts: torch.Tensor):
        """
        计算相机内点云数量产生的成本
        Args:
            rollouts: (B, R, D, T), 轨迹, T = Ts, Ts为采样后的关键时间步数量
            pc: (P, 3), 工件附近点云
        """
        B, R, D, T = rollouts.shape
        fk_computer = UR12e_t(B*R*T)
        
        # prepare data
        rollouts = rollouts.transpose(-2, -1).reshape(-1, D)            # (B*R*T, D=6)
        cam_poses = fk_computer.forward_cam_pose(rollouts).float()      # (B*R*T, 7)
        robot_pose_init = self.robot_pose_init.clone().unsqueeze(0).repeat(B*R*T, 1)     # (B*R*T, 7)
        cam_quat, cam_pos = math.tf_combine(robot_pose_init[:, 3:], robot_pose_init[:, :3], 
                                            cam_poses[:, 3:], cam_poses[:, :3])
        cam_poses = torch.cat((cam_pos, cam_quat), dim=-1)      # (B*R*T, 7)

        if self.has_pc:
            # part1: compute number of visible
            P = self.pc.shape[0]
            pc_original = self.pc.clone()    # (P, 3)
            pc = self.pc.clone().unsqueeze(0).repeat(B*R*T, 1, 1)     # (B*R*T, P, 3)

            cam_pos_ray = cam_poses[:, :3].unsqueeze(-2).repeat(1, P, 1).reshape(B*R*T*P, 3)     # (B*R*T*P, 3)
            normals = self.normal.clone()   # (5, 3)
            normals = math.quat_apply(cam_poses[..., 3:].unsqueeze(-2).repeat(1, 5, 1), 
                                    normals.unsqueeze(0).repeat(B*R*T, 1, 1))         # (B*R*T, 5, 3)
            normals = normals.unsqueeze(1).repeat(1, P, 1, 1).reshape(B*R*T*P, 5, 3)   # (B*R*T*P, 5, 3)
            p12 = math.quat_apply(cam_poses[..., 3:], self.p12.clone().unsqueeze(0).repeat(B*R*T, 1)
                                    ) + cam_poses[..., :3]    # (B*R*T, 3)
            p14 = math.quat_apply(cam_poses[..., 3:], self.p14.clone().unsqueeze(0).repeat(B*R*T, 1)
                                    ) + cam_poses[..., :3]    # (B*R*T, 3)
            vector_12 = (pc - p12.unsqueeze(-2)).reshape(B*R*T*P, 3)       # (B*R*T*P, 3)
            vector_14 = (pc - p14.unsqueeze(-2)).reshape(B*R*T*P, 3)       # (B*R*T*P, 3)

            # step1: compute number of points insides of FOV
            mask_idx = torch.arange(B*R*T*P, device=self.device)
            # print("num all:", mask_idx.shape)
            for i in range(5):
                if i < 3:
                    vector = vector_12.clone()
                else:
                    vector = vector_14.clone()
                in_idx = (torch.bmm(vector[mask_idx].unsqueeze(-2), normals[mask_idx, i].unsqueeze(-1)).squeeze(-1).squeeze(-1) > 0)
                # in_idx = (torch.sum(vector[mask_idx] * normals[mask_idx, i], dim=-1) > 0)
                mask_idx = mask_idx[in_idx]             # (N,)
                # print("num insides:", mask_idx.shape)

            # step2: compute number of points without vision block
            pc = pc.reshape(B*R*T*P, 3)                 # (B*R*T*P, 3)
            pc = pc[mask_idx]                           # (N, 3)
            cam_pos_ray_s = cam_pos_ray[mask_idx].clone()             # (N, 3)
            ray_dirs = cam_pos_ray_s - pc
            ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
            ray_starts = pc + ray_dirs * 0.0005 + self.piece.data.root_pos_w[0].clone().unsqueeze(0)
            hit_positions, _, _, _ = raycast_mesh(
                ray_starts,
                ray_dirs,
                self.warp_mesh,
                max_dist=0.8,
                return_distance=False,
                return_normal=False,
                return_face_id=False
            )
            mask_inf = torch.isinf(hit_positions)   # (N, 3)
            unblock = torch.any(mask_inf, dim=-1)   # (N,)
            mask_idx = mask_idx[unblock]           # (M,)
            # print("num unblock:", mask_idx.shape)

            # step3: compute number of visible points        
            mask = torch.zeros(B*R*T*P, device=self.device)
            mask[mask_idx] = 1                          # (B*R*T*P,)
            mask = mask.reshape(B, R, T, P).bool()      # (B, R, T, P)
            visible = torch.any(mask, dim=-2)           # (B, R, P)
            visible_cost = torch.sum((1-visible.float()), dim=-1) / P * 50    # (B, R)
            visible_cost = visible_cost.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, D, T) / T * self.visible_cost_weight

            vision_cost = visible_cost.clone()

        else:
            # part2: 
            seam_median = self.seam_median       # (3,)
            ideal_dir = seam_median.unsqueeze(0) - cam_poses[:, :3].clone()       # (B*R*T, 3)
            ideal_dir = ideal_dir / (torch.norm(ideal_dir, dim=-1, keepdim=True) + 1e-8)
            z_dir = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device)
            z_dir = math.quat_apply(cam_poses[:, 3:], z_dir.unsqueeze(0).repeat(B*R*T, 1))      # (B*R*T, 3)
            cos_theta = torch.bmm(ideal_dir.unsqueeze(-2), z_dir.unsqueeze(-1)).squeeze(-1).squeeze(-1)     # (B*R*T,)
            scale = (0.5 * torch.ones(T, dtype=torch.float, device=self.device)) ** torch.arange(T, device=self.device)
            dir_cost = scale.unsqueeze(0).unsqueeze(0) * torch.acos(cos_theta).reshape(B, R, T) / torch.pi * 180    # (B, R, T)
            dir_cost = dir_cost.unsqueeze(-2).repeat(1, 1, D, 1) * self.dir_cost_weight

            vision_cost = dir_cost.clone()

        return vision_cost


    def getJoints(self, poses):
        """
        根据pose逆解出关节角
        Args:
            poses: (N, M, 7), 目前设定: N*M <= 750, M为125, N最大值为6

        """
        N, M = poses.shape[:2]
        all_idx = torch.arange(N * M, dtype=torch.long, device=self.device)

        # compute poses frame for IK
        poses = poses.reshape(-1, 7)    # (N*M, 7)
        robot_inv_pose = self.robot_base_inv_pose.clone().repeat(N*M, 1)
        cam_quat, cam_pos = math.tf_combine(robot_inv_pose[:, 3:], robot_inv_pose[:, :3], poses[:, 3:], poses[:, :3])
        cam_transfer = self.cam_transfer.repeat(N*M, 1)
        quat, xyz = math.tf_combine(cam_quat, cam_pos, cam_transfer[:, 3:], cam_transfer[:, :3])

        # set for field block
        field_pos = xyz + math.quat_apply(quat, self.z_off[all_idx])
        rot_x = math.quat_from_euler_xyz(torch.pi * self.ones[all_idx], self.zeros[all_idx] , self.zeros[all_idx])
        field_rot = math.quat_mul(quat, rot_x)
        field_rot, field_pos = math.tf_combine(self.robot.data.body_quat_w[all_idx, self.base_link_idx].clone(), 
                                               self.robot.data.body_pos_w[all_idx, self.base_link_idx].clone(),
                                               field_rot, field_pos)
        field_pose = torch.cat((field_pos, field_rot), dim=-1)

        # compute IK
        ik_solver = UR12e_t(num_envs=N*M, device=self.device)
        R = math.matrix_from_quat(quat)
        T = self.generate_transformation(xyz, R)
        joints_all = ik_solver.solve_fairino_ec(T).float()   # (N*M, 8, 6)

        rewards = []
        for i in range(8):
            joints = joints_all[:, i]       # (N*M, 6)
            outbound = (joints < self.joints_lower_limit) | (joints > self.joints_upper_limit)
            outbound = torch.any(outbound, dim=1)
            # all_idx_ob = all_idx[outbound]
            # joints[outbound] = self.initial_joint_pos[all_idx_ob]
            for _ in range(5):
                self.field.write_root_pose_to_sim(field_pose, env_ids=all_idx)
                self.robot.set_joint_velocity_target(self.robot_dof_targets[all_idx], env_ids=all_idx)
                self.robot.write_joint_state_to_sim(joints, self.robot_dof_targets[all_idx], env_ids=all_idx)
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.sim_cfg.dt)
            reward = self.get_single_rewards(joints, all_idx)
            reward[outbound] -= 300
            rewards.append(reward)          # (N*M,)
        rewards = torch.stack(rewards, dim=-1)      # (N*M, 8)
        mask_nan = torch.isnan(rewards)
        rewards[mask_nan] = -1000
        # print("rewards:", rewards)

        rwd, max_idx = torch.topk(input=rewards, k=2, dim=-1)     # (N*M, 2), (N*M, 2)
        max_idx = max_idx.unsqueeze(-1).repeat(1, 1, 6)
        joints = torch.gather(input=joints_all, index=max_idx, dim=1)    # (N*M, 2, 6)

        rwd = rwd.reshape(N, M, 2).reshape(N, 2*M)
        joints = joints.reshape(N, M, 2, 6).reshape(N, 2*M, 6)

        # move field far away
        field_pose = self.field_init_pose[all_idx]

        for _ in range(5):
            self.field.write_root_pose_to_sim(field_pose, env_ids=all_idx)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.sim_cfg.dt)

        # torch.set_printoptions(threshold=float('inf'))
        # print("joints:")
        # print(joints.shape)
        # print("rwd:")
        # print(rwd)
        return rwd, joints


    def visionBlock(self, seam_line: torch.Tensor, pos_list: torch.Tensor):
        """
        判断焊缝上的点是否被遮挡
        Args:
            seam_line: (N, M, L, 3)
            pos_list:  (N, M, 3) 
        """
        N, M, L = seam_line.shape[:3]
        
        pos_list = pos_list.unsqueeze(-2).repeat(1, 1, L, 1)    # (N, M, L, 3)

        # Prepare your custom rays
        piece_pos = self.piece.data.root_pos_w[0].clone().unsqueeze(0).unsqueeze(0).unsqueeze(0)    # (1, 1, 1, 3)
        ray_starts = piece_pos + seam_line       # (N, M, L, 3)
        ray_dirs = pos_list - seam_line
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)        # (N, M, L, 3)
        ray_starts += ray_dirs * 0.0005

        hit_positions, _, _, _ = raycast_mesh(
            ray_starts,
            ray_dirs,
            self.warp_mesh,
            max_dist=1.0,
            return_distance=False,
            return_normal=False,
            return_face_id=False
        )       # (N, M, L, 3)

        mask_inf = torch.isinf(hit_positions)   # (N, M, L, 3)
        unblock = torch.any(mask_inf, dim=-1)   # (N, M, L)
        
        return unblock


    #-----------------------------------------------------------------
    #----------------------------- Utils -----------------------------
    #-----------------------------------------------------------------
    def generate_transformation(self, xyz, R):
        """
        生成转移矩阵
        """
        B = xyz.shape[0]
        T = torch.zeros((B, 4, 4), dtype=torch.float64, device=self.device)
        # print(R.shape)
        T[:, :3, :3] = R[:, :, :].double()
        T[:, :3, 3] = xyz.double()
        T[:, 3, 3] = 1
        return T
    

    def get_single_rewards(self, joints, env_ids):
        # PENALTY: joint actions
        joint_pen = -torch.sum((joints - self.initial_joint_pos[env_ids])**2, dim=-1) * 0.01

        # PENALTY: Collision
        collision_pen = -self.collided(env_ids).int() * 500

        return joint_pen + collision_pen
    

    def normal_vector(self, p1, p2, p3):
        return torch.cross(p2 - p1, p3 - p1, dim=-1)


    def compute_normal(self, p1, p2, p3, p4, p5, p6, p7, p8):
        "Compute normal vectors of all surfaces of a hexahedral space for 1d input (8 vertices)"
        n1 = self.normal_vector(p1, p4, p2)
        n1 = (n1 / torch.norm(n1, p=2, dim=-1))
        n2 = self.normal_vector(p5, p8, p1)
        n2 = (n2 / torch.norm(n2, p=2, dim=-1))
        n3 = self.normal_vector(p6, p5, p2)
        n3 = (n3 / torch.norm(n3, p=2, dim=-1))
        n4 = self.normal_vector(p7, p6, p3)
        n4 = (n4 / torch.norm(n4, p=2, dim=-1))
        n5 = self.normal_vector(p8, p7, p4)
        n5 = (n5 / torch.norm(n5, p=2, dim=-1))
        n6 = self.normal_vector(p5, p6, p8)
        n6 = (n6 / torch.norm(n6, p=2, dim=-1))
        
        return torch.stack((n1, n2, n3, n4, n5, n6), dim=0)
    

    def compute_normal_1(self, p0, p1, p2, p3, p4):
        "Compute normal vectors of all surfaces of a hexa space for 1d input (5 vertices)"
        n0 = self.normal_vector(p1, p2, p3)
        n0 = (n0 / torch.norm(n0, p=2, dim=-1))
        n1 = self.normal_vector(p0, p2, p1)
        n1 = (n1 / torch.norm(n1, p=2, dim=-1))
        n2 = self.normal_vector(p0, p3, p2)
        n2 = (n2 / torch.norm(n2, p=2, dim=-1))
        n3 = self.normal_vector(p0, p4, p3)
        n3 = (n3 / torch.norm(n3, p=2, dim=-1))
        n4 = self.normal_vector(p0, p1, p4)
        n4 = (n4 / torch.norm(n4, p=2, dim=-1))

        
        return torch.stack((n0, n1, n2, n3, n4), dim=0)