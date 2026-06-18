"""
configuration
version: v0
date: 25.09.22
"""
import os
import json
import torch

class ConfigurationPose:
    def __init__(self):
        self.test = False
        
        self.code_test = False

        # 进化策略优化设置
        self.num_batches: int = 8 * 1
        self.num_iterations: int = 100
        self.max_iterations: int = 300                    # 300
        self.max_iterations_refine: int = 40    # 18, 42
        self.num_randoms_new: int = 100  #50
        self.num_randoms_old: int = 0
        self.num_randoms_all: int = self.num_randoms_new + self.num_randoms_old + 1
        self.num_select:int  = 2
        self.num_steps: int = 5
        self.num_next: int = 30 # 30
        self.num_poses: int = 4
        self.span: int =  1 # 3, 7

        self.exponentiated_cost_sensitivity: float = 5    # h值: 0.5


        # 使用进化策略优化
        self.collision_cost_weight: float = 25
        self.vision_rot_cost_weight: float = 1
        self.vision_pos_cost_weight: float = 1
        # self.visible_cost_weight: float = 25 / self.num_steps
        self.dist_limit: float = 0.5
        self.insides_cost_weight: float = 19 / self.dist_limit
        self.space_cost_weight: float = 21 / self.dist_limit
        self.block_cost_weight: float = 22 / self.dist_limit
        self.orientation_limit: float = 30
        self.orientation_cost_weight: float = 23 / self.orientation_limit
        self.block_radius: float = 0.04     # 0.04
        self.num_block_pts: int = 6         # 6
        self.forced: bool = False               # 是否单独拍起点和终点的点云
        self.has_space_cost: bool = False
        self.refine: bool = True

        self.has_cov: bool = False
        self.cov_path = ""
        self.sigma: torch.Tensor = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=torch.float)
        # self.sigma: torch.Tensor = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=torch.float)

        # self.dir_path = "/home/kejian/Downloads/3/part"
        # self.seam_data_path = os.path.join(self.dir_path, "seam_1.pkl")

        ## 仿真环境
        self.num_envs: int = self.num_batches * (self.num_randoms_new + self.num_randoms_old)
        self.usd_path = ""
        self.pc_path = ""

        self.scl: float = 7 / 11     # 9.5 / 11
        self.scl_z: float = 9.5 / 11

        # ur12e
        self.joints_lower_limit_horizontal = [-0.77920353, -2.60093713, -0.07581812, -3.29959822, -3.16031218, -6.19999981]
        self.joints_upper_limit_horizontal = [ 3.92079639,  0.35906282,  2.54208183, -0.44959825,  0.88968790,  6.19999981]
        self.joints_lower_limit_vertical = [-0.77920353, -2.60093713,  0.42418188, -3.29959822, -3.16031218, -6.19999981]
        self.joints_upper_limit_vertical = [ 3.92079639,  0.35906282,  2.54208183, -0.24959825,  2.93968773,  6.19999981]
        self.joints_lower_limit_all = [-0.77920353, -2.60093713, -0.07581812, -3.29959822, -3.16031218, -6.19999981]
        self.joints_upper_limit_all = [ 3.92079639,  0.35906282,  2.54208183, -0.24959825,  2.93968773,  6.19999981]
        