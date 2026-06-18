"""
build up a simulation scene for raycasting
version: v0
date: 25.09.26
"""

import time
import os

import torch
import numpy as np
import pickle

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.sim import SimulationContext
from isaaclab.assets import RigidObjectCfg, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, RAY_CASTER_MARKER_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
# from isaaclab.utils import configclass
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

import omni.usd
from pxr import UsdGeom, Gf, Usd

import opt_math_pose as math
from config_pose import ConfigurationPose as Configuration
from robot.a1_cfg import A1_CFG
from ik_cam_pose import UR12e_t


# @configclass
class ScenePoseCfg(InteractiveSceneCfg):
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
    contact_j1 = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/Link1",
                                  filter_prim_paths_expr=["{ENV_REGEX_NS}/robot/Link3", "{ENV_REGEX_NS}/robot/Link4",
                                                          "{ENV_REGEX_NS}/robot/Link5", "{ENV_REGEX_NS}/robot/Link6",
                                                          "{ENV_REGEX_NS}/field",
                                                          "{ENV_REGEX_NS}/robot/xiaoyu_accessory_link"])
    contact_j2 = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/Link2",
                                  filter_prim_paths_expr=["{ENV_REGEX_NS}/robot/Link4", "{ENV_REGEX_NS}/robot/Link5",
                                                          "{ENV_REGEX_NS}/robot/Link6", "{ENV_REGEX_NS}/field",
                                                          "{ENV_REGEX_NS}/robot/xiaoyu_accessory_link"])
    contact_j3 = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/Link3",
                                  filter_prim_paths_expr=["{ENV_REGEX_NS}/robot/Link1", "{ENV_REGEX_NS}/robot/Link6",
                                                          "{ENV_REGEX_NS}/field",
                                                          "{ENV_REGEX_NS}/robot/xiaoyu_accessory_link"])
    contact_j4 = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/Link4",
                                  filter_prim_paths_expr=["{ENV_REGEX_NS}/robot/Link1", "{ENV_REGEX_NS}/robot/Link2"])
    contact_j5 = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/Link5",
                                  filter_prim_paths_expr=["{ENV_REGEX_NS}/robot/Link1", "{ENV_REGEX_NS}/robot/Link2"])
    # contact_acc = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/xiaoyu_accessory_link",
    #                                filter_prim_paths_expr=["{ENV_REGEX_NS}/piece"])

    # contact_piece = ContactSensorCfg(prim_path="/World/envs/env_.*/piece",)
    # contact_fr_base = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/xiaoyu_arm_base_link")
    # contact_magnetic_base = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/xiaoyu_base_link")
    # contact_j1 = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/Link1")
    # contact_j2 = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/Link2")
    # contact_j3 = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/Link3")
    # contact_j4 = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/Link4")
    # contact_j5 = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/Link5")
    # contact_acc = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/xiaoyu_accessory_link")


class ScenePose:
    def __init__(self, cfg: Configuration, num_envs: int| None = None, device = "cuda"):
        self.cfg = cfg
        if num_envs is None:
            self.num_envs = self.cfg.num_envs
        else:
            self.num_envs = num_envs

        self.num_steps = self.cfg.num_steps

        self.insides_cost_weight = self.cfg.insides_cost_weight
        self.space_cost_weight = self.cfg.space_cost_weight
        self.block_cost_weight = self.cfg.block_cost_weight
        self.orientation_cost_weight = self.cfg.orientation_cost_weight

        self.block_radius = self.cfg.block_radius
        self.num_block_pts = self.cfg.num_block_pts

        mul = 4     # 2**1
        # create sim
        self.sim_cfg = SimulationCfg(
            dt = 1/120,
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
        scene_cfg = ScenePoseCfg(num_envs=self.num_envs, env_spacing=16, replicate_physics=True)
        scene_cfg.piece.spawn.usd_path = self.cfg.usd_path
        scene_cfg.robot.spawn.usd_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "robot", "a1_ur12e_s.usd")
        # if self.cfg.test:
        #     scene_cfg.robot.spawn.usd_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "robot", "a1_ur7e_n.usd")
        # else:
        #     scene_cfg.robot.spawn.usd_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "robot", "a1_ur7e_n_c.usd")
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

        # self.contact_piece = self.scene["contact_piece"]
        # self.contact_fr_base = self.scene.sensors["contact_fr_base"]
        # self.contact_magnetic_base = self.scene.sensors["contact_magnetic_base"]
        # self.contact_j1 = self.scene.sensors["contact_j1"]
        # self.contact_j2 = self.scene.sensors["contact_j2"]
        # self.contact_j3 = self.scene.sensors["contact_j3"]
        # self.contact_j4 = self.scene.sensors["contact_j4"]
        # self.contact_j5 = self.scene.sensors["contact_j5"]
        # # self.contact_acc = self.scene.sensors["contact_acc"]

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

                if world_points.shape[0] == 8:
                    indices = np.array([0, 1, 3, 2, 0, 3, 
                              4, 7, 5, 6, 7, 4,
                              5, 7, 3, 1, 5, 3,
                              4, 2, 6, 0, 2, 4,
                              6, 3, 7, 2, 3, 6,
                              4, 5, 1, 0, 4, 1], dtype=np.int32)
                else:
                    indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

                all_points.append(world_points)
                all_indices.append(indices + vertex_offset)  # offset indices
                vertex_offset += world_points.shape[0]

            all_points = np.vstack(all_points)
            all_indices = np.hstack(all_indices)

            # print("all_points.shape", all_points.shape)
            # print("all_indices.shape", all_indices.shape)
            # print("all_points")
            # print(all_points)
            # all_points_local = torch.tensor(all_points - self.piece.data.root_pos_w[0].clone().unsqueeze(0).cpu().numpy())
            # print(all_points_local)
            # print("all_indices")
            # b = torch.tensor([0, 1, 3, 2, 0, 3, 
            #                   4, 7, 5, 6, 7, 4,
            #                   5, 7, 3, 1, 5, 3,
            #                   4, 2, 6, 0, 2, 4,
            #                   6, 3, 7, 2, 3, 6,
            #                   4, 5, 1, 0, 4, 1])
            # print(all_points_local[b].reshape(-1, 3, 3))

            warp_mesh = convert_to_warp_mesh(all_points, all_indices, device=device)
            return warp_mesh
        
        stage = omni.usd.get_context().get_stage()
        xform_prim = stage.GetPrimAtPath("/World/envs/env_0/piece")
        if xform_prim.IsInstance():
            xform_prim = xform_prim.GetPrototype()  # follow to actual mesh definition
        self.warp_mesh = combine_meshes_to_warp(xform_prim)


        ## init
        # robot
        self.horizontal = 0
        self.joints_lower_limit = torch.empty(0)
        self.joints_upper_limit = torch.empty(0)
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
        self.z_off = torch.tensor([0, 0, scene_cfg.field.spawn.height/2], dtype=torch.float, device=self.device).unsqueeze(0)
        self.field_init_pose = torch.cat((self.field.data.root_pos_w.clone(), self.field.data.root_quat_w.clone()), dim=-1)
        field_pos_all = torch.tensor([0, 0, 1e3], dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        field_quat_all = torch.tensor([1, 0, 0, 0], dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.field_pose_all = torch.cat((field_pos_all, field_quat_all), dim=-1)

        # 3d cam: nomal verctor of valid vision space
        scl = self.cfg.scl      # scl = 9.5 / 11
        scl_z = self.cfg.scl_z
        self.p1 = torch.tensor([ 0.135 * scl, -0.20 * 5/6 * scl,   0.4], dtype=torch.float ,device=self.device)
        self.p2 = torch.tensor([-0.135 * scl, -0.20 * 5/6 * scl,   0.4], dtype=torch.float ,device=self.device)
        self.p3 = torch.tensor([-0.135 * scl,  0.20 * 5/6 * scl,   0.4], dtype=torch.float ,device=self.device)
        self.p4 = torch.tensor([ 0.135 * scl,  0.20 * 5/6 * scl,   0.4], dtype=torch.float ,device=self.device)
        self.p5 = torch.tensor([ 0.135 * scl + ( 0.140 * scl_z * scl), -0.20 * 5/6 * scl + (-0.185 * 5/6 * scl_z * scl),  0.4 * (1 + scl_z)], dtype=torch.float ,device=self.device)
        self.p6 = torch.tensor([-0.135 * scl + (-0.140 * scl_z * scl), -0.20 * 5/6 * scl + (-0.185 * 5/6 * scl_z * scl),  0.4 * (1 + scl_z)], dtype=torch.float ,device=self.device)
        self.p7 = torch.tensor([-0.135 * scl + (-0.140 * scl_z * scl),  0.20 * 5/6 * scl + ( 0.185 * 5/6 * scl_z * scl),  0.4 * (1 + scl_z)], dtype=torch.float ,device=self.device)
        self.p8 = torch.tensor([ 0.135 * scl + ( 0.140 * scl_z * scl),  0.20 * 5/6 * scl + ( 0.185 * 5/6 * scl_z * scl),  0.4 * (1 + scl_z)], dtype=torch.float ,device=self.device)
        # self.p1 = torch.tensor([ 0.135, -0.20 * 5/6,   0.4], dtype=torch.float ,device=self.device)
        # self.p2 = torch.tensor([-0.135, -0.20 * 5/6,   0.4], dtype=torch.float ,device=self.device)
        # self.p3 = torch.tensor([-0.135,  0.20 * 5/6,   0.4], dtype=torch.float ,device=self.device)
        # self.p4 = torch.tensor([ 0.135,  0.20 * 5/6,   0.4], dtype=torch.float ,device=self.device)
        # self.p5 = torch.tensor([ 0.275, -0.385 * 5/6,  0.8], dtype=torch.float ,device=self.device)
        # self.p6 = torch.tensor([-0.275, -0.385 * 5/6,  0.8], dtype=torch.float ,device=self.device)
        # self.p7 = torch.tensor([-0.275,  0.385 * 5/6,  0.8], dtype=torch.float ,device=self.device)
        # self.p8 = torch.tensor([ 0.275,  0.385 * 5/6,  0.8], dtype=torch.float ,device=self.device)
        self.normal = self.compute_normal(self.p1, self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8)   # (6, 3)

        print("仿真，启动")
        self.print = False
        self.original_costs = False


    def reset(self, robot_pose: torch.Tensor, horizontal: int, piece_pose: torch.Tensor | None = None):
        """
        reset scene
        Args:
            robot_pose: (7,)
            horizontal: int
            piece_pose: (7,)
        """
        # reset robot pose
        if piece_pose is not None:
            piece_inv_quat, piece_inv_pos = math.tf_inverse(piece_pose[3:], piece_pose[:3])
            robot_quat, robot_pos = math.tf_combine(piece_inv_quat, piece_inv_pos, robot_pose[3:], robot_pose[:3])
            robot_pose = torch.cat((robot_pos, robot_quat), dim=-1)
        self.robot_pose_init = robot_pose.clone()

        robot_base = robot_pose.clone()
        # robot_base[2] += 0.26
        base_transform = torch.tensor([0, 0, 0.26, 0.7071068, 0, 0, 0.7071068], dtype=torch.float, device= self.device)
        robot_base_quat, robot_base_pos = math.tf_combine(robot_base[3:], robot_base[:3], base_transform[3:], base_transform[:3])
        robot_base = torch.cat((robot_base_pos, robot_base_quat), dim=-1)
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

        # reset joints limit
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

        return self.robot_pose_init


    def collided(self, env_ids=None):
        if env_ids == None:
            env_ids = self.robot._ALL_INDICES
        
        force_piece = self.contact_piece.data.net_forces_w[env_ids, 0, :]
        force_magnetic_base = self.contact_magnetic_base.data.net_forces_w[env_ids, 0, :]
        force_fr_base = self.contact_fr_base.data.net_forces_w[env_ids, 0, :]
        force_j1 = self.contact_j1.data.force_matrix_w[env_ids, 0, :].reshape(len(env_ids), -1)
        force_j2 = self.contact_j2.data.force_matrix_w[env_ids, 0, :].reshape(len(env_ids), -1)
        force_j3 = self.contact_j3.data.force_matrix_w[env_ids, 0, :].reshape(len(env_ids), -1)
        force_j4 = self.contact_j4.data.force_matrix_w[env_ids, 0, :].reshape(len(env_ids), -1)
        force_j5 = self.contact_j5.data.force_matrix_w[env_ids, 0, :].reshape(len(env_ids), -1)
        # force_acc = self.contact_acc.data.force_matrix_w[env_ids, 0, :].reshape(len(env_ids), -1)
        # print("force_j4_f:", self.contact_j4.data.force_matrix_w)
        # print("force_j4_a:", self.contact_j4.data.net_forces_w)
        # print("force_acc:", self.contact_acc.data.force_matrix_w)
        return ~(
            torch.all(force_piece == 0, dim=-1) &
            torch.all(force_magnetic_base == 0, dim=-1) 
            & torch.all(force_fr_base == 0, dim=-1) & torch.all(force_j1 == 0, dim=-1) 
            & torch.all(force_j2 == 0, dim=-1) & torch.all(force_j3 == 0, dim=-1) 
            & torch.all(force_j4 == 0, dim=-1) & torch.all(force_j5 == 0, dim=-1)
            # & torch.all(force_acc == 0, dim=-1)
            )

        # force_piece = self.contact_piece.data.net_forces_w[env_ids, 0, :].clone()
        # force_magnetic_base = self.contact_magnetic_base.data.net_forces_w[env_ids, 0, :].clone()
        # force_fr_base = self.contact_fr_base.data.net_forces_w[env_ids, 0, :].clone()
        # force_j1 = self.contact_j1.data.net_forces_w[env_ids, 0, :].clone()
        # force_j2 = self.contact_j2.data.net_forces_w[env_ids, 0, :].clone()
        # force_j3 = self.contact_j3.data.net_forces_w[env_ids, 0, :].clone()
        # force_j4 = self.contact_j4.data.net_forces_w[env_ids, 0, :].clone()
        # force_j5 = self.contact_j5.data.net_forces_w[env_ids, 0, :].clone()
        # # force_acc = self.contact_acc.data.net_forces_w[env_ids, 0, :].clone()
        # return ~(
        #     torch.all(force_piece == 0, dim=-1) &
        #     torch.all(force_magnetic_base == 0, dim=-1) 
        #     & torch.all(force_fr_base == 0, dim=-1) & torch.all(force_j1 == 0, dim=-1) 
        #     & torch.all(force_j2 == 0, dim=-1) & torch.all(force_j3 == 0, dim=-1) 
        #     & torch.all(force_j4 == 0, dim=-1) & torch.all(force_j5 == 0, dim=-1)
        #     # & torch.all(force_acc == 0, dim=-1)
        #     )


    def computeCollisionCost(self, cam_poses: torch.Tensor, joints_opt: torch.Tensor):
        """
        计算关节角对应的碰撞成本函数
        Args:
            cam_poses: (B, R, 7)
            joints_opt: (B, 6)
        """
        B, R = cam_poses.shape[:2]
        cam_poses = cam_poses.reshape(B * R, 7)
        joints_opt = joints_opt.unsqueeze(-2).expand(-1, R, -1).reshape(B * R, 6)

        # compute collision cost
        cost, joints = self.getJoints(cam_poses, joints_opt)        # (B*R,), (B*R, 6)
        cost = cost.reshape(B, R)       # (B, R)
        joints = joints.reshape(B, R, 6)        # (B, R, 6)

        # if R == 1:
        #     print("collsion:", cost[:, 0])

        return cost, joints


    def computeVisionCost_0(self, seam_line: torch.Tensor, seam_tangent: torch.Tensor, seam_limits: torch.Tensor, 
                            cam_poses: torch.Tensor, idx: torch.Tensor, seam_line_mul: torch.Tensor,
                            seam_tangent_mul: torch.Tensor, seam_limit_mul: torch.Tensor, seam_vertical_mul: torch.Tensor,
                            seam_theta_mul: torch.Tensor, seam_rays_2d_mul: torch.Tensor, seam_dir_mul: torch.Tensor,
                            cam_poses_mul: torch.Tensor, idx_mul: torch.Tensor, num_randoms: int):
        """
        计算焊缝是否通过相机可视,即在相机FOV内且无遮挡,不可计算梯度,使用RL方法
        Args:
            seam_line: (M, 3)   original (B, R, N, 3)
            seam_tangent: (M, 3)
            seam_limits: (M, 2, 3)
            cam_poses: (M, 7)
            idx: (B, 2),    每段起点和终点的index
        """
        M = seam_line.shape[0]
        B = idx.shape[0]
        start_idx = idx[:, 0]       # (B,)
        end_idx = idx[:, 1]       # (B,)
        length = end_idx - start_idx       # (B,)
        idx_array = torch.arange(B, device=self.device)
        
        # functions
        def computeCostSingle(x: torch.Tensor, i: int):
            x = x.reshape(num_randoms, int(length[i]/num_randoms))     # (R, N')
            return x.sum(dim=-1)       # (R)
        
        def expNorm(x: torch.Tensor, t: float):
            return 1 - torch.exp(-x / t)
        
        # step1: compute number of points insides of FOV and insides cost
        mask_idx, costs = self.visionInsides(seam_line, cam_poses)
        insides_cost = [computeCostSingle(costs[s:e], int(i)) for s, e, i in zip(start_idx, end_idx, idx_array)]    # [(R)]
        insides_cost = torch.stack(insides_cost, dim=0)     # (B, R)
        insides_cost /= self.num_steps      # (B, R)
        
        # step2: compute number of points without vision block and block cost
        unblock, costs = self.visionBlock(seam_line, cam_poses[:, :3])
        block_cost = [computeCostSingle(costs[s:e], int(i)) for s, e, i in zip(start_idx, end_idx, idx_array)]  # [(R)]
        block_cost = torch.stack(block_cost, dim=0) / self.num_steps        # (B, R)
        block_cost /= self.num_steps        # (B, R)

        # step3: compute space cost
        space_cost = self.visionSpace(seam_line_mul, seam_tangent_mul, seam_limit_mul, seam_vertical_mul, seam_theta_mul, 
                         seam_rays_2d_mul, seam_dir_mul, cam_poses_mul, idx_mul, num_randoms)       # (B, R)
        space_cost /= self.num_steps        # (B, R)

        # # step3: compute number of visible points
        # mask = torch.zeros(M, device=self.device)
        # mask[mask_idx] = 1                          # (M,)
        # def computeVisCostSingle(x: torch.Tensor, i: int):
        #     x = x.reshape(num_randoms, int(length[i]/num_randoms))     # (R, N')
        #     cost = torch.sum((1-x.float()), dim=-1)     # (R)
        #     return cost
        # vis_costs = [computeVisCostSingle(mask[s:e], int(i)) for s, e, i in zip(start_idx, end_idx, idx_array)]      # [(R)]
        # visible_cost = torch.stack(vis_costs, dim=0)    # (B, R)

        # step4: compute orientation cost
        costs = self.visionOrientation(seam_line, seam_tangent, seam_limits, cam_poses)
        orientation_cost = [computeCostSingle(costs[s:e], int(i)) for s, e, i in zip(start_idx, end_idx, idx_array)]      # [(R)]
        orientation_cost = torch.stack(orientation_cost, dim=0)         # (B, R)
        orientation_cost /= self.num_steps      # (B, R)

        return insides_cost, block_cost, space_cost, orientation_cost


    def computeVisionCost_1(self, seam_line: torch.Tensor, seam_tangent: torch.Tensor, seam_limits: torch.Tensor, 
                            cam_poses: torch.Tensor, idx: torch.Tensor, block_mask: torch.Tensor, num_randoms: int, 
                            original_costs: bool = False):
        """
        计算焊缝是否通过相机可视,即在相机FOV内且无遮挡,不可计算梯度,使用RL方法
        Args:
            seam_line: (M, 3)   original (B, R, N, 3)
            seam_tangent: (M, 3)
            seam_limits: (M, 2, 3)
            cam_poses: (M, 7)
            idx: (B, 2),    每段起点和终点的index
        """
        M = seam_line.shape[0]
        B = idx.shape[0]
        start_idx = idx[:, 0]       # (B,)
        end_idx = idx[:, 1]       # (B,)
        length = end_idx - start_idx       # (B,)
        idx_array = torch.arange(B, device=self.device)
        self.original_costs = original_costs

        self.print = False
        if num_randoms == 1:
            self.print = False
        
        # functions
        def computeCostSingle(x: torch.Tensor, i: int):
            x = x.reshape(num_randoms, int(length[i]/num_randoms))     # (R, N')
            return x.sum(dim=-1)       # (R)
        
        def expNorm(x: torch.Tensor, t: float):
            return 1 - torch.exp(-x / t)
        
        # step1: compute number of points insides of FOV and insides cost
        mask_idx, costs, costs_o = self.visionInsides(seam_line, cam_poses)
        insides_cost = [computeCostSingle(costs[s:e], int(i)) for s, e, i in zip(start_idx, end_idx, idx_array)]    # [(R)]
        insides_cost = torch.stack(insides_cost, dim=0)     # (B, R)
        insides_cost /= self.num_steps      # (B, R)
        # print("insides_cost:")
        # print(insides_cost[:, 0])
        if self.original_costs:
            assert costs_o is not None
            insides_cost_o = [computeCostSingle(costs_o[s:e], int(i)) for s, e, i in zip(start_idx, end_idx, idx_array)]    # [(R)]
            insides_cost_o = torch.stack(insides_cost_o, dim=0)     # (B, R)
            insides_cost_o /= self.num_steps      # (B, R)
        if self.print:
            print("insides_cost:", insides_cost[:, 0])
        
        # step2: compute number of points without vision block and block cost
        unblock, costs = self.visionBlock(seam_line, cam_poses[:, :3])
        block_cost = [computeCostSingle(costs[s:e], int(i)) for s, e, i in zip(start_idx, end_idx, idx_array)]  # [(R)]
        block_cost = torch.stack(block_cost, dim=0)        # (B, R)
        block_cost /= self.num_steps        # (B, R)
        if self.print:
            print("block_cost:", block_cost[:, 0])
            # print("seam_line:")
            # print(seam_line[idx[-1, 0]:idx[-1, -1]])


        # step4: compute orientation cost
        costs, costs_o = self.visionOrientation(seam_line, seam_tangent, seam_limits, cam_poses, block_mask)
        orientation_cost = [computeCostSingle(costs[s:e], int(i)) for s, e, i in zip(start_idx, end_idx, idx_array)]      # [(R)]
        orientation_cost = torch.stack(orientation_cost, dim=0)         # (B, R)
        orientation_cost /= self.num_steps      # (B, R)
        # print("orientation_cost:")
        # print(orientation_cost[:, 0])
        if self.original_costs:
            assert isinstance(costs_o, torch.Tensor)
            orientation_cost_o = [computeCostSingle(costs_o[s:e], int(i)) for s, e, i in zip(start_idx, end_idx, idx_array)]      # [(R)]
            orientation_cost_o = torch.stack(orientation_cost_o, dim=0)         # (B, R)
            orientation_cost_o /= self.num_steps      # (B, R)
        if self.print:
            print("orientation_cost:", orientation_cost[:, 0])

        
        vision_pos_cost = (self.insides_cost_weight * insides_cost + self.block_cost_weight * block_cost + 
                           self.orientation_cost_weight * orientation_cost)
        vision_rot_cost = self.insides_cost_weight * insides_cost
        # if self.print:
        #     print("total vision cost:", vision_pos_cost[:, 0] + vision_rot_cost[:, 0])
        if self.original_costs:
            vision_pos_cost_o = (self.insides_cost_weight * insides_cost_o + self.block_cost_weight * block_cost + 
                    self.orientation_cost_weight * orientation_cost_o)
            vision_rot_cost_o = self.insides_cost_weight * insides_cost_o
        else:
            vision_pos_cost_o = None
            vision_rot_cost_o = None
        return vision_pos_cost, vision_rot_cost, vision_pos_cost_o, vision_rot_cost_o


    def getJoints(self, poses: torch.Tensor, joints_opt: torch.Tensor):
        """
        根据pose逆解出关节角,并计算相应的
        Args:
            poses: (M, 7)
            joints_opt: (M, 6)
        Returns:
            cost: (M,)
            joints: (M, 6)
        """
        M = poses.shape[0]
        K = 2
        L = 1

        # print("robot_quat:")
        # print(self.robot.data.root_quat_w[:20])
        # print("robot_base_inv_pose:", self.robot_base_inv_pose)
        # raise KeyError

        # compute poses frame for IK
        robot_inv_pose = self.robot_base_inv_pose.clone().repeat(M, 1)
        cam_quat, cam_pos = math.tf_combine(robot_inv_pose[:, 3:], robot_inv_pose[:, :3], poses[:, 3:], poses[:, :3])

        # Overall
        # set for field block
        field_pos = cam_pos + math.quat_apply(cam_quat, self.z_off.repeat(M, 1))        # (M, 3)
        ones = torch.ones((M,), dtype=torch.float, device=self.device)
        zeros = torch.zeros((M,), dtype=torch.float, device=self.device)
        rot_x = math.quat_from_euler_xyz(torch.pi * ones, zeros , zeros)
        field_rot = math.quat_mul(cam_quat, rot_x)      # (M, 4)
        field_pos = field_pos.unsqueeze(-2).expand(-1, K, -1).reshape(M * K, 3)     # (M * K, 3)
        field_rot = field_rot.unsqueeze(-2).expand(-1, K, -1).reshape(M * K, 4)     # (M * K, 4)

        # compute IK
        ik_solver = UR12e_t(num_envs=M, device=self.device)
        R_mat = math.matrix_from_quat(cam_quat)
        T_mat = self.generate_transformation(cam_pos, R_mat)
        joints_all_ik = ik_solver.solve_fairino_ec(T_mat).float()   # (M, 8, 6)
        joints_all_ik = torch.nan_to_num(joints_all_ik, nan=-20)
        outbound = (joints_all_ik < self.joints_lower_limit) | (joints_all_ik > self.joints_upper_limit)
        outbound = torch.any(outbound, dim=-1)   # (M, 8)
        rewards = -torch.norm(joints_all_ik - joints_opt.unsqueeze(-2), dim=-1)   # (M, 8)
        rewards[outbound] -= 500     # (M, 8)
        rewards_max, max_idx = torch.topk(rewards, k = K, dim=1)      # (M, K), (M, K)
        joints_all_ik = torch.gather(input=joints_all_ik, dim=1, index=max_idx.unsqueeze(-1).expand(-1, -1, 6))  # (M, K, 6)
        joints_all_ik = joints_all_ik.reshape(M * K, 6)     # (M * K, 6)
        cost = (rewards_max <= -400).int().reshape(M * K)    # (M * K,)
        # if M < 10:
        #     print("outbound cost:", cost)
        
        outbound = torch.gather(input=outbound, dim=1, index=max_idx)  # (M, K)
        outbound = outbound.reshape(M * K)
        if torch.max((outbound.int() - cost).abs()) != 0:
            print("equal:", torch.max((outbound.int() - cost).abs()))
        
        joints_all_ik[outbound] = self.initial_joint_pos[0:1].expand(int(outbound.int().sum()), -1)

        if L == 2:
            field_pos_inv = field_pos.clone().flip(0)
            field_pos = torch.cat((field_pos, field_pos_inv), dim=0)
            field_rot_inv = field_rot.clone().flip(0)
            field_rot = torch.cat((field_rot, field_rot_inv), dim=0)
            joints_all_ik_inv = joints_all_ik.clone().flip(0)
            joints_all_ik = torch.cat((joints_all_ik, joints_all_ik_inv), dim=0)
            cost_inv = cost.clone().flip(0)
            cost = torch.cat((cost, cost_inv), dim=0)

        # compute collision cost
        times = (L * M * K) // self.num_envs
        rest = (L * M * K) % self.num_envs
        for i in range(times):
            # set for field block
            field_rot_p, field_pos_p = math.tf_combine(self.robot.data.body_quat_w[:, self.base_link_idx].clone(), 
                                                self.robot.data.body_pos_w[:, self.base_link_idx].clone(),
                                                field_rot[i*self.num_envs : (i+1)*self.num_envs], 
                                                field_pos[i*self.num_envs : (i+1)*self.num_envs])
            field_pose_all = torch.cat((field_pos_p, field_rot_p), dim=-1)
            # compute IK
            joints = joints_all_ik[i*self.num_envs : (i+1)*self.num_envs].clone()   # (Ne, 6)
            for _ in range(2):
                self.field.write_root_pose_to_sim(field_pose_all)
                self.robot.set_joint_velocity_target(self.robot_dof_targets)
                self.robot.write_joint_state_to_sim(joints, self.robot_dof_targets)
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.sim_cfg.dt)
            collided = self.collided().int()
            cost[i*self.num_envs : (i+1)*self.num_envs] += collided # (Ne,)
        if rest > 0:
            all_idx = torch.arange(rest, device=self.device)
            # set for field block
            field_rot_p, field_pos_p = math.tf_combine(self.robot.data.body_quat_w[all_idx, self.base_link_idx].clone(), 
                                                self.robot.data.body_pos_w[all_idx, self.base_link_idx].clone(),
                                                field_rot[-rest:], field_pos[-rest:])
            field_pose_all = torch.cat((field_pos_p, field_rot_p), dim=-1)      # (N, 7)
            # compute IK
            joints = joints_all_ik[-rest:].clone()
            for _ in range(2):
                self.field.write_root_pose_to_sim(field_pose_all, env_ids=all_idx)
                self.robot.set_joint_velocity_target(self.robot_dof_targets[all_idx], env_ids=all_idx)
                self.robot.write_joint_state_to_sim(joints, self.robot_dof_targets[all_idx], env_ids=all_idx)
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.sim_cfg.dt)
            collided = self.collided(all_idx).int()
            cost[-rest:] += collided        # (N,)
        # if rest > 0:
        #     all_idx = torch.arange(rest, device=self.device)
        #     # set for field block
        #     field_rot_p, field_pos_p = math.tf_combine(self.robot.data.body_quat_w[all_idx, self.base_link_idx].clone(), 
        #                                         self.robot.data.body_pos_w[all_idx, self.base_link_idx].clone(),
        #                                         field_rot[-rest:], field_pos[-rest:])
            
        #     field_pose = torch.cat((field_pos_p, field_rot_p), dim=-1)      # (N, 7)
        #     field_pose_all = self.field_pose_all.clone()
        #     field_pose_all[all_idx] = field_pose.clone()
        #     # compute IK
        #     joints = self.initial_joint_pos.clone()
        #     joints[all_idx] = joints_all_ik[-rest:].clone()
        #     for _ in range(2):
        #         self.field.write_root_pose_to_sim(field_pose_all)
        #         self.robot.set_joint_velocity_target(self.robot_dof_targets)
        #         self.robot.write_joint_state_to_sim(joints, self.robot_dof_targets)
        #         self.scene.write_data_to_sim()
        #         self.sim.step(render=False)
        #         self.scene.update(dt=self.sim_cfg.dt)
        #     collided = self.collided().int()
        #     cost[-rest:] += collided[all_idx]        # (N,)

        if L == 2: 
            cost = cost.reshape(L, M * K)
            cost = cost[0] + cost[1].flip(0)    # (M * K,)
            joints_all_ik = joints_all_ik.reshape(L , M * K, 6)
            joints_all_ik = joints_all_ik[0].reshape(M, K, 6)
        cost = (cost > 0.1).int()
        cost = cost.reshape(M, K)
        joints_all_ik = joints_all_ik.reshape(M, K, 6)
        cost, min_idx = torch.min(cost, dim=-1)    # (M,), (M,)
        batch_idx = torch.arange(M, device=self.device)
        joints = joints_all_ik[batch_idx, min_idx]  # (M, 6)

        # if M < 10:
        #     print("overall cost:", cost)
        return cost, joints


    def visionInsides(self, seam_line: torch.Tensor, cam_poses: torch.Tensor):
        """
        判断焊缝上的点是否在FOV内,返回FOV内的indices和距离FOV的距离
        Args:
            seam_line: (N, 3)
            pos_list:  (N, 3) 
        """
        N = seam_line.shape[0]
        normals = self.normal.clone()   # (6, 3)
        normals = math.quat_apply(cam_poses[..., 3:].unsqueeze(-2).repeat(1, 6, 1), 
                                  normals.unsqueeze(0).repeat(N, 1, 1))         # (N, 6, 3)
        p1 = math.quat_apply(cam_poses[..., 3:], self.p1.clone().unsqueeze(0).repeat(N, 1)
                                 ) + cam_poses[..., :3]    # (N, 3)
        p7 = math.quat_apply(cam_poses[..., 3:], self.p7.clone().unsqueeze(0).repeat(N, 1)
                                 ) + cam_poses[..., :3]    # (N, 3)
        vector_1 = seam_line - p1       # (N, 3)
        vector_7 = seam_line - p7       # (N, 3)
        vectors = torch.cat((vector_1.unsqueeze(-2).repeat(1, 3, 1), vector_7.unsqueeze(-2).repeat(1, 3, 1)), dim=-2)   # (N, 6, 3)

        mask_idx = torch.arange(N, device=self.device)
        # print("num all:", mask_idx.shape)

        vectors = vectors.reshape(N * 6, 3)
        normals = normals.reshape(N * 6, 3)

        dists = -torch.bmm(vectors.unsqueeze(-2), normals.unsqueeze(-1)).squeeze(-1).squeeze(-1).reshape(N, 6)   # (N, 6)
        if self.original_costs:
            # costs_o, _ = torch.max(dists, dim=-1)      # (N,)
            costs_o, _ = torch.max(dists[:, :-1], dim=-1)      # (N,)
        else:
            costs_o = None
        dists = torch.clamp(dists, min=0)       # (N, 6)
        costs = dists.sum(-1)           # (N,)
        in_idx = (costs <= 1e-8)
        mask_idx = mask_idx[in_idx]             # (M,)
        # print("num insides:", mask_idx.shape)

        return mask_idx, costs, costs_o


    def visionBlock(self, seam_line: torch.Tensor, pos_list: torch.Tensor):
        """
        判断焊缝上的点是否被遮挡,使用通过mesh进行检测的方法,不可计算梯度
        Args:
            seam_line: (M, 3)
            pos_list:  (M, 3) 
        """
        M = seam_line.shape[0]

        # Prepare your custom rays
        piece_pos = self.piece.data.root_pos_w[0].clone().unsqueeze(0)    # (1, 3)

        # # version 1: from seam line to cam
        # ray_starts_o = piece_pos + seam_line       # (M, 3)
        # ray_dirs = pos_list - seam_line
        # ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)        # (M, 3)
        # ray_starts = ray_starts_o + ray_dirs * 0.00005

        # version 2: from cam to seam line
        ray_starts = piece_pos + pos_list       # (M, 3)
        ray_dirs = seam_line - pos_list 
        # ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)        # (M, 3)
        ray_dirs = ray_dirs / torch.clamp(torch.norm(ray_dirs, dim=-1, keepdim=True), min=1e-8)        # (M, 3)
        
        # 防止射线与遮挡物过近
        r = self.block_radius
        N = self.num_block_pts
        points = self.generate_start_points(ray_dirs, ray_starts, r, N)   # (M, N, 3)
        pos_list = pos_list.unsqueeze(-2)   # (M, 1, 3)
        pos_list = torch.cat((pos_list, pos_list + points), dim=-2)     # (M, N+1, 3)
        ray_starts = piece_pos.unsqueeze(-2) + pos_list     # (M, N+1, 3)
        ray_dirs = seam_line.unsqueeze(-2) - pos_list       # (M, N+1, 3)
        ray_dirs = ray_dirs / torch.clamp(torch.norm(ray_dirs, dim=-1, keepdim=True), min=1e-8)        # (M, N+1, 3)


        hit_positions, _, _, _ = raycast_mesh(
            ray_starts,
            ray_dirs,
            self.warp_mesh,
            max_dist=100.0,
            return_distance=False,
            return_normal=False,
            return_face_id=False
        )       # (M, N+1, 3)

        # # version 1: from seam line to cam
        # idx = torch.arange(M, device=self.device)
        # costs = torch.zeros((M,), dtype=torch.float, device=self.device)    # (M,)
        # unblock = torch.ones((M,), dtype=torch.int, device=self.device).bool()
        # mask_inf = torch.isinf(hit_positions)   # (M, 3)
        # mask_block = ~torch.any(mask_inf, dim=-1)   # (M,)
        # idx_block = idx[mask_block]     # (M',)
        # dists = torch.norm(pos_list[mask_block] - seam_line[mask_block], dim=-1)        # (M',)
        # dists_c = torch.norm(hit_positions[mask_block] - ray_starts_o[mask_block], dim=-1)      # (M',)
        # mask_block = (dists - dists_c) > 1e-5       # (M',)
        # idx_block = idx_block[mask_block]       # (M'',)
        # costs[idx_block] = 1 - dists_c[mask_block].clone() / 2
        # unblock[idx_block] = False

        # version 2: from cam to seam line
        costs = torch.zeros((M, N+1), dtype=torch.float, device=self.device)    # (M, N+1)
        unblock = torch.ones((M,), dtype=torch.int, device=self.device).bool()
        seam_line  = seam_line + piece_pos      # (M, 3)
        dist_c = torch.norm(hit_positions - seam_line.unsqueeze(-2), dim=-1)    # (M, N+1)
        block = ~(dist_c < 1e-2)    # (M, N+1)
        dist_c[torch.isinf(dist_c)] = 0
        dist_c[dist_c > 2] = 0
        costs[block] = 1 - dist_c[block] / 2    # (M, N+1)
        costs = torch.sum(costs, dim=-1)    # (M,)
        block = torch.any(block, dim=-1)
        unblock[block] = False

        # if self.print:
        #     print("hit_positions:")
        #     print(hit_positions[-5:] - piece_pos)
        #     print("ray_starts:")
        #     print(ray_starts[-5:] - piece_pos)
        #     print("ray_dirs:")
        #     print(ray_dirs[-5:])
        #     # print("ray_dirs diff:")
        #     # print((pos_list[-5:] - seam_line[-5:]) / torch.norm(pos_list[-5:] - seam_line[-5:], dim=-1, keepdim=True) - ray_dirs[-5:])
        #     # print("ray_starts diff:")
        #     # print(seam_line[-5:] - (ray_starts[-5:] - piece_pos - ray_dirs[-5:] * 0.0005))
        #     # print("mask inf:")
        #     # print(mask_inf[-5:])
        
        return unblock, costs


    def visionSpace(self, seam_lines: torch.Tensor, seam_tangent: torch.Tensor, seam_limit: torch.Tensor, 
                    seam_vertical: torch.Tensor, seam_theta: torch.Tensor, seam_rays_2d: torch.Tensor, 
                    seam_dir: torch.Tensor, cam_poses: torch.Tensor, idx: torch.Tensor, num_randoms: int):
        """
        判断焊缝上的点是否被遮挡,使用通过空间进行检测的方法,可以计算梯度
        Args:
            seam_lines:         (M, 2, 3)
            seam_tangent:       (M, 3)
            seam_limit:         (M, 3)
            seam_vertical:      (M, 3)
            seam_theta:         (M, L)
            seam_rays_2d:       (M, L, 2)
            seam_dir:           (M, 3)
            cam_poses:          (M, 7)
            idx:                (B, 2)
        """
        M = seam_lines.shape[0]
        B = idx.shape[0]
        start_idx = idx[:, 0]           # (B,)
        end_idx = idx[:, 1]             # (B,)
        length = end_idx - start_idx        # (B,)
        idx_array = torch.arange(B, device=self.device)

        def line_plane_intersection(n: torch.Tensor, p0: torch.Tensor, r0: torch.Tensor, v: torch.Tensor, eps=1e-12):
            """
            批量计算直线与平面的交点
            Args:
                n:   (N, 3) 平面法向量
                p0:  (N, 3) 平面上一点
                r0:  (N, 3) 直线上一点
                v:   (N, 3) 直线方向向量
            Returns: 
                (N, 3) 每条直线的交点 (若无解则为 NaN)
            """
            device = n.device
            # 分母 n · v
            den = torch.sum(n * v, dim=-1)   # (N,)
            # 分子 n · (p0 - r0)
            num = torch.sum(n * (p0 - r0), dim=-1)  # (N,)

            # 初始化交点
            intersection = torch.full_like(r0, float("nan"), device=device)

            # 可解情况：den != 0
            mask = torch.abs(den) > eps
            t = num[mask] / den[mask]          # (M,)
            intersection[mask] = r0[mask] + t.unsqueeze(-1) * v[mask]

            return intersection

        # 计算视线向量与每个平面的交点
        seam_line = seam_lines[:, 0].clone()
        seam_line_prime = seam_lines[:, 1].clone()
        cam_pos = cam_poses[:, :3].clone()
        inter_plane = line_plane_intersection(seam_tangent, seam_line, cam_pos, seam_dir)   # (M, 3), 会有NaN值存在
        mask_value = (torch.sum((inter_plane - seam_line_prime) * seam_dir, dim=-1) > 0)          # (M,)
        idx_value = torch.arange(M, device=self.device)[mask_value]         # (K,)
        # mask_zeros = (torch.sum((inter_plane - seam_line_prime) * seam_dir, dim=-1) <= 0)         # (M)
        
        # 计算每个平面内交点
        vectors = inter_plane[mask_value].clone() - seam_line[mask_value].clone()       # (K, 3)
        dist_vec = torch.norm(vectors, dim=-1)      # (K,)
        vectors_norm = vectors / torch.clamp(dist_vec.unsqueeze(-1), min=1e-8)
        cos_theta = torch.sum(vectors_norm * seam_limit[mask_value].clone(), dim=-1)    # (K,)
        theta = torch.acos(torch.clamp(cos_theta, min=-1, max=1))                       # (K,)
        diff_theta = torch.abs(theta.unsqueeze(-1) - seam_theta[mask_value].clone())    # (K, L)
        _, idx_top = torch.topk(input=diff_theta, k=2, dim=-1, largest=False)           # (K, 2)
        
        # 投影到2d平面
        near_pts = torch.gather(input=seam_rays_2d[mask_value], dim=1, index=idx_top.unsqueeze(-1).repeat(1, 1, 2))    # (K, 2, 2)
        vectors_2d = torch.stack((
            torch.sum(vectors * seam_limit[mask_value], dim=-1), 
            torch.sum(vectors * seam_vertical[mask_value], dim=-1)
        ), dim=-1)      # (M, 2)

        def intersect_lines(p1: torch.Tensor, d1: torch.Tensor, p2: torch.Tensor, d2: torch.Tensor, eps=1e-12):
            """
            批量计算二维直线交点
            p1, d1, p2, d2: (N,2)
            返回: (N,2) 交点，若两直线平行则返回 nan
            """
            # 向量差
            dp = p2 - p1  # (N,2)

            # 2D 叉积
            def cross(a, b):
                return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]

            denom = cross(d1, d2)  # (N,)
            num = cross(dp, d2)    # (N,)

            # 避免除零：平行时用 nan
            t = torch.where(torch.abs(denom) < eps,
                            torch.full_like(denom, float('nan')),
                            num / denom)

            inter = p1 + t.unsqueeze(-1) * d1
            return inter

        inter_2d = intersect_lines(near_pts[:, 0], near_pts[:, 0] - near_pts[:, 1], 
                        torch.zeros_like(vectors_2d, dtype=torch.float, device=self.device), vectors_2d)    # (K, 2)
        inter_3d = inter_2d[:,0:1] * seam_limit[mask_value] + inter_2d[:,1:2] * seam_vertical[mask_value]   # (K, 3)

        # 计算成本
        costs = torch.zeros((M,), dtype=torch.float, device=self.device)    # (M,)
        vector_diff = vectors - inter_3d        # (K, 3)
        mask_in = (torch.bmm(vectors.unsqueeze(-2), vector_diff.unsqueeze(-1)).squeeze(-1).squeeze(-1) < 0)     # (K,)
        dists_diff = torch.norm(vector_diff, dim=-1)        # (K,)
        dists_diff[mask_in] = 0         # (K,)
        dists_diff = torch.nan_to_num(dists_diff, nan=2)
        costs[mask_value] = dists_diff.clone()              # (M,)
        
        def computeCostSingle(x: torch.Tensor, i: int):
            x = x.reshape(num_randoms, int(length[i]/num_randoms))     # (R, N'* N')
            return x.sum(dim=-1)        # (R)
        vs_costs = [computeCostSingle(costs[s:e], int(i)) for s, e, i in zip(start_idx, end_idx, idx_array)]      # [(R)]
        vs_costs = torch.stack(vs_costs, dim=0)    # (B, R)
        return vs_costs


    def visionOrientation(self, seam_line: torch.Tensor, seam_tangent: torch.Tensor, seam_limits: torch.Tensor, 
                            cam_poses: torch.Tensor, block_mask: torch.Tensor):
        """
        计算观测方向的成本
        Args:
            seam_line: (M, 3),
            seam_tangent: (M, 3)
            seam_limits: (M, 2, 3)
            cam_poses: (M, 7)
            block_mask: (M,)
        """
        M = seam_line.shape[0]
        tgt_cost_scale = 0.3
        pl_cost_scale = 0.2
        nm_cost_scale = 1 - tgt_cost_scale - pl_cost_scale

        mid_vector = seam_limits[:, 0] + seam_limits[:, 1]                      # (M, 3)
        mid_vector = mid_vector / (torch.norm(mid_vector, dim=-1, keepdim=True) + 1e-8)

        # tangent cost
        cam_pos = cam_poses[:, :3]      # (M, 3)
        direction = cam_pos - seam_line      # (M, 3)
        tgt_cost = torch.zeros((M,), dtype=torch.float, device=self.device)
        tgt_cost_o = torch.zeros((M,), dtype=torch.float, device=self.device)
        proj_tgt = torch.bmm(direction.unsqueeze(-2), seam_tangent.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (M,)
        if block_mask.int().sum() > 0:
            cos_theta_tgt = torch.clamp(proj_tgt[block_mask] / (torch.norm(direction[block_mask], dim=-1) + 1e-8), -1, 1).abs()   # (M',)
            theta_tgt = torch.acos(cos_theta_tgt) / torch.pi * 180       # (M',)
            theta_tgt = torch.max(theta_tgt - 60, 30 - theta_tgt)
            if self.original_costs:
                tgt_cost_o[block_mask] = theta_tgt.clone() * self.num_steps
            tgt_cost[block_mask] = torch.clamp(theta_tgt, min=0) * self.num_steps

        # plane cost
        direction_pl = direction - proj_tgt.unsqueeze(-1) * seam_tangent        # (M, 3)
        direction_pl = direction_pl / (torch.norm(direction_pl, dim=-1, keepdim=True) + 1e-8)         # (M, 3)
        soll_dist = torch.bmm(mid_vector.unsqueeze(-2), seam_limits[:, 1].unsqueeze(-1)).squeeze(-1).squeeze(-1)    # (M,)
        soll_dist = torch.acos(torch.clamp(soll_dist, -1, 1)) / torch.pi * 180       # (M,)
        dist = torch.bmm(direction_pl.unsqueeze(-2), mid_vector.unsqueeze(-1)).squeeze(-1).squeeze(-1) # (M,)
        dist = torch.acos(torch.clamp(dist, -1, 1)) / torch.pi * 180       # (M,)
        pl_cost = dist - soll_dist
        if self.original_costs:
            pl_cost_o = pl_cost.clone()
        pl_cost = torch.clamp(pl_cost, min=0)

        # normal cost
        seam_normal = torch.cross(seam_limits, seam_tangent.unsqueeze(-2).expand(-1, 2, -1), dim=-1)    # (M, 2, 3)
        seam_normal = seam_normal / torch.norm(seam_normal, dim=-1, keepdim=True).clamp(min=1e-8)
        block_mask_neg = ~block_mask
        nm_cost_0 = torch.zeros((M,), dtype=torch.float, device=self.device)
        nm_cost_1 = torch.zeros((M,), dtype=torch.float, device=self.device)
        nm_cost = torch.zeros((M,), dtype=torch.float, device=self.device)
        nm_cost_o_0 = torch.zeros((M,), dtype=torch.float, device=self.device)
        nm_cost_o_1 = torch.zeros((M,), dtype=torch.float, device=self.device)
        nm_cost_o = torch.zeros((M,), dtype=torch.float, device=self.device)
        if block_mask_neg.int().sum() > 0:
            direction_n = direction[block_mask_neg] / torch.clamp(
                torch.norm(direction[block_mask_neg], dim=-1, keepdim=True), min = 1e-8)      # (M', 3)
            dists_0 = torch.bmm(seam_normal[block_mask_neg, 0].unsqueeze(-2), direction_n.unsqueeze(-1)).squeeze(-1).squeeze(-1) # (M',)
            dists_0 = torch.acos(torch.clamp(dists_0, -1, 1).abs()) / torch.pi * 180      # (M')
            nm_cost_0[block_mask_neg] = torch.abs(dists_0 - 45) - 15    # (M,)
            if self.original_costs:
                nm_cost_o_0 = nm_cost_0.clone()         # (M,)
            nm_cost_0 = torch.clamp(nm_cost_0, min=0)   # (M,)
            dists_1 = torch.bmm(seam_normal[block_mask_neg, 1].unsqueeze(-2), direction_n.unsqueeze(-1)).squeeze(-1).squeeze(-1) # (M',)
            dists_1 = torch.acos(torch.clamp(dists_1, -1, 1).abs()) / torch.pi * 180      # (M')
            nm_cost_1[block_mask_neg] = torch.abs(dists_1 - 45) - 15    # (M,)
            if self.original_costs:
                nm_cost_o_1 = nm_cost_1.clone()         # (M,)
                nm_cost_o = nm_cost_o_0 + nm_cost_o_1
            nm_cost_1 = torch.clamp(nm_cost_1, min=0)   # (M,)
            nm_cost = nm_cost_0 + nm_cost_1
            
        
        # direction_pl = direction - proj_tgt.unsqueeze(-1) * seam_tangent        # (M, 3)
        # direction_pl = direction_pl / (torch.norm(direction_pl, dim=-1, keepdim=True) + 1e-8)         # (M, 3)
        # soll_dist = torch.bmm(mid_vector.unsqueeze(-2), seam_limits[:, 1].unsqueeze(-1)).squeeze(-1).squeeze(-1)    # (M,)
        # soll_dist = torch.acos(torch.clamp(soll_dist, -1, 1)) / torch.pi * 180       # (M,)
        # soll_dist = torch.where(soll_dist > 15, 15, soll_dist)
        # dist = torch.bmm(direction_pl.unsqueeze(-2), mid_vector.unsqueeze(-1)).squeeze(-1).squeeze(-1) # (M,)
        # dist = torch.acos(torch.clamp(dist, -1, 1)) / torch.pi * 180       # (M,)
        # pl_cost = dist - soll_dist
        # if self.original_costs:
        #     pl_cost_o = pl_cost.clone()
        # pl_cost = torch.clamp(pl_cost, min=0)
        # nm_cost = torch.zeros((M,), dtype=torch.float, device=self.device)
        # nm_cost_o = torch.zeros((M,), dtype=torch.float, device=self.device)

        orientation_cost = tgt_cost * tgt_cost_scale + pl_cost * pl_cost_scale + nm_cost * nm_cost_scale
        if self.original_costs:
            orientation_cost_o = tgt_cost_o * tgt_cost_scale + pl_cost_o * pl_cost_scale + nm_cost_o * nm_cost_scale
        else:
            orientation_cost_o = None
        
        return orientation_cost, orientation_cost_o
    

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
    

    def get_single_rewards(self, joints, env_ids=None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
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
    

    def generateRay(self, pos_list: torch.Tensor, seam_line: torch.Tensor, ):
        """
        create ray from cam to seam line
        Args:
            pos_list: (N, 3)
            seam_line: (M, 3)
        """
        N = pos_list.shape[0]
        M = seam_line.shape[0]
        pos_list = pos_list.unsqueeze(1).expand(-1, M, -1).reshape(N * M, 3)
        seam_line = seam_line.unsqueeze(0).expand(N, -1, -1).reshape(N * M, 3)  # (N * M, 3)
        L = 500
        count = torch.linspace(0, 1, L, device=self.device).unsqueeze(0)   # (1, L)
        vs_x = count * (seam_line[:, 0] - pos_list[:, 0]).unsqueeze(-1)
        vs_y = count * (seam_line[:, 1] - pos_list[:, 1]).unsqueeze(-1)
        vs_z = count * (seam_line[:, 2] - pos_list[:, 2]).unsqueeze(-1)     # (N * M, L)
        vs = (torch.stack((vs_x, vs_y, vs_z), dim=-1) + pos_list.unsqueeze(-2)).reshape(N * M * L, 3)     # (N * M * L, 3)

        base_pos = self.robot.data.root_pos_w.clone() - self.robot_pose_init[:3].unsqueeze(0)   # (B, 3)

        return (base_pos.unsqueeze(-2) + vs.unsqueeze(0)).reshape(-1, 3)


    def generate_start_points(self, ray_dirs: torch.Tensor, ray_starts: torch.Tensor, r: float, N: int):
        """
        ray_dirs: (M,3) 法向量
        ray_starts: (M, 3)
        r: 半径
        N: 圆上点的数量
        return: (M, N, 3)
        """
        device = ray_dirs.device

        # 选参考向量
        ref1 = torch.tensor([1.,0.,0.], device=device)
        ref2 = torch.tensor([0.,1.,0.], device=device)
        
        # 挑选 ref，避免和 n 平行
        use_ref1 = (torch.abs(ray_dirs[:,0]) < 0.9).float().unsqueeze(-1)  # (M, 1)
        ref = use_ref1 * ref1 + (1-use_ref1) * ref2   # (M, 3)
        
        # 基向量 u, v
        u = torch.cross(ray_dirs, ref, dim=-1)
        u = u / torch.norm(u, dim=-1, keepdim=True)
        v = torch.cross(ray_dirs, u, dim=-1)
        v = v / torch.norm(v, dim=-1, keepdim=True)
        
        # 角度
        theta = torch.linspace(0, 2*torch.pi, N, device=device)  # (N,)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)        # (N,)
        
        # 广播组合
        points = r * (cos_t[None,:,None]*u[:,None,:] + sin_t[None,:,None]*v[:,None,:])  # (M, N, 3)
        return points