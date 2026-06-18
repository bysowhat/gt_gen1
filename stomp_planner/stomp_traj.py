"""
generate gt trajectories given env voxels as well as target position(s)
version: v0
date: 25.08.25
"""
import time

import torch
import numpy as np
import random

from typing import Optional

import stomp_utils_traj as stomp_utils
from stomp_utils_traj import TrajectoryInitializations, DerivativeOrder
from stomp_utils_traj import DEFAULT_NOISY_COST_IMPORTANCE_WEIGHT, MIN_COST_DIFFERENCE, MIN_CONTROL_COST_WEIGHT
from config_traj import ConfigurationTraj as Configuration
from scene_traj import SceneTraj as Scene

import matplotlib.pyplot as plt


# --- 设置默认数据类型 ---
torch.set_default_dtype(torch.float)


# Set the seed value
seed = 3
# Python's built-in random
random.seed(seed)
# NumPy
np.random.seed(seed)
# PyTorch
torch.manual_seed(seed)
# CUDA
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multi-GPU


# --- STOMP 主类 ---
class StompTraj:
    def __init__(self, config: Configuration, scene: Scene, device="cuda"):
        self.device = device
        self.scene = scene
        # config配置
        self.config = config
        self.num_batch = self.config.num_batch
        self.num_dimensions = self.config.num_dimensions
        self.num_timesteps_init = self.config.num_timesteps_init
        self.num_timesteps_next = self.config.num_timesteps_next
        self.num_timesteps_all = 0
        self.num_iterations = self.config.num_iterations
        self.num_iterations_after_valid = self.config.num_iterations_after_valid
        self.num_rollouts_new = self.config.num_rollouts_new
        self.num_rollouts_old = self.config.num_rollouts_old
        self.num_rollouts_all = self.num_rollouts_new + self.num_rollouts_old + 1
        self.delta_t = self.config.delta_t
        self.control_cost_weight = self.config.control_cost_weight
        self.collision_cost_weight = self.config.collision_cost_weight
        self.vision_cost_weight = self.config.vision_cost_weight
        self.initialization_method = self.config.initialization_method
        self.exponentiated_cost_sensitivity = self.config.exponentiated_cost_sensitivity
        self.horizontal =self.config.horizontal
        self.noise_scale_high = self.config.noise_scale_high
        self.noise_scale_low = self.config.noise_scale_low
        self.filter_scale_high = self.config.filter_scale_high
        self.filter_scale_low = self.config.filter_scale_low
        if self.horizontal == 0:
            self.joints_lower_limit = torch.tensor(self.config.joints_lower_limit_horizontal, device=self.device)
            self.joints_upper_limit = torch.tensor(self.config.joints_upper_limit_horizontal, device=self.device)
        elif self.horizontal == 1:
            self.joints_lower_limit = torch.tensor(self.config.joints_lower_limit_vertical, device=self.device)
            self.joints_upper_limit = torch.tensor(self.config.joints_upper_limit_vertical, device=self.device)
        else:
            self.joints_lower_limit = torch.tensor(self.config.joints_lower_limit_all, device=self.device)
            self.joints_upper_limit = torch.tensor(self.config.joints_upper_limit_all, device=self.device)

        self.pre_filter = self.config.pre_filter
        self.post_filter = self.config.post_filter
        self.has_pc = self.config.has_pc
        self.has_vision = False

        self.init_it = True
        self.num_transfer = 0

        # Rollouts变量
        self.noise = torch.empty(0)
        self.generate_noise = torch.empty(0)
        # self.parameters_noise = torch.empty(0)

        self.control_costs = torch.empty(0)
        self.state_costs = torch.empty(0)
        self.total_costs = torch.empty(0)
        self.total_cost = torch.empty(0)

        self.probabilities = torch.empty(0)
        self.importance_weights = torch.empty(0)

        self.stored_rollouts = torch.empty(0)
        self.parameters_optimized = torch.empty(0)

        self.parameters_control_costs = torch.empty(0)
        self.parameters_state_costs = torch.empty(0)
        self.parameters_total_costs = torch.empty(0)
        self.parameters_total_cost = torch.empty(0)
        self.parameters_state_cost = torch.empty(0)
        self.parameters_collision_cost = torch.empty(0)
        self.parameters_vision_cost = torch.empty(0)

        self.noise_scale = torch.empty(0)
        self.filter_scale = torch.empty(0)
        
        self.control_cost_matrix_R_padded_init, self.control_cost_matrix_R_init, self.inv_control_cost_matrix_R_init = \
            stomp_utils.generateControlCostMatrix(self.num_timesteps_init, self.delta_t, True, self.device)
        self.control_cost_matrix_R_padded_next, self.control_cost_matrix_R_next, self.inv_control_cost_matrix_R_next = \
            stomp_utils.generateControlCostMatrix(self.num_timesteps_next, self.delta_t, True, self.device)

        self.control_cost_matrix_R = torch.empty(0)
        self.control_cost_matrix_R_padded = torch.empty(0)
        self.filter_matrix_R = torch.empty(0)


    def resetVariables(self, fixed_pts: torch.Tensor):
        """
        根据固定点(机械臂起点以及观测点)的动态重新定义变量
        Args:
            fixed_pts: (B, N+1 D), N为拍点云次数,1为起始位姿
        """
        self.init_it = True

        self.num_transfer = fixed_pts.shape[1] - 2     #观测位姿间的轨迹数量
        self.num_timesteps_all = self.num_timesteps_init + (self.num_timesteps_next - 1) * self.num_transfer    # 整条轨迹总长度

        self.noise = torch.zeros((self.num_batch, self.num_rollouts_all, self.num_dimensions, self.num_timesteps_all), 
                                 device=self.device)   # 所有stored rollouts的noise
        self.generate_noise = torch.zeros((self.num_batch, self.num_rollouts_new, self.num_dimensions, self.num_timesteps_all), 
                                          device=self.device)  # 生成的rollouts的noise
        # self.parameters_noise = torch.zeros((self.num_batch, self.num_dimensions, self.num_timesteps_all), 
        #                                     device=self.device)   # 最优rollout的noise
    
        self.control_costs = torch.zeros((self.num_batch, self.num_rollouts_all, self.num_dimensions, self.num_timesteps_all), 
                                         device=self.device)
        self.state_costs = torch.zeros((self.num_batch, self.num_rollouts_all, self.num_dimensions, self.num_timesteps_all), 
                                       device=self.device)   
        self.total_costs = torch.zeros((self.num_batch, self.num_rollouts_all, self.num_dimensions, self.num_timesteps_all), 
                                       device=self.device)
        self.total_cost = torch.zeros((self.num_batch, self.num_rollouts_all), device=self.device)
        
        self.probabilities = torch.zeros((self.num_batch, self.num_dimensions, self.num_timesteps_all, self.num_rollouts_all), 
                                         device=self.device)        # 张量的顺序不同，(B, D, T, R)
        self.importance_weights = DEFAULT_NOISY_COST_IMPORTANCE_WEIGHT * torch.ones_like(self.probabilities)      # =1.0
 
        self.stored_rollouts = torch.zeros((self.num_batch, self.num_rollouts_all, self.num_dimensions, self.num_timesteps_all), 
                                           device=self.device)          # 生成轨迹 + 复用轨迹
        self.parameters_optimized = torch.zeros((self.num_batch, self.num_dimensions, self.num_timesteps_all), 
                                                device=self.device)     # 最优轨迹(单条),是否使用滤波处理有post_filter判定
     
        self.parameters_control_costs = torch.zeros((self.num_batch, self.num_dimensions, self.num_timesteps_all), device=self.device)
        self.parameters_state_costs = torch.zeros((self.num_batch, self.num_dimensions, self.num_timesteps_all), device=self.device)
        self.parameters_total_costs = torch.zeros((self.num_batch, self.num_dimensions, self.num_timesteps_all), device=self.device)
        self.parameters_total_cost = torch.zeros((self.num_batch,), device=self.device)
        self.parameters_state_cost = torch.zeros((self.num_batch,), device=self.device)
        self.parameters_collision_cost = torch.zeros((self.num_batch,), device=self.device)
        self.parameters_vision_cost = torch.zeros((self.num_batch,), device=self.device)

        self.noise_scale = torch.zeros((self.num_batch,), device=self.device)
        self.filter_scale = torch.zeros((self.num_batch,), device=self.device)

        self.control_cost_matrix_R_padded, self.control_cost_matrix_R, _ = stomp_utils.generateControlCostMatrix(self.num_timesteps_all, self.delta_t, 
                                                                                 True, self.device)
        self.filter_matrix_R = stomp_utils.generateSmoothingMatrix_1(self.inv_control_cost_matrix_R_init, 
                                                                   self.num_transfer, self.num_timesteps_next, damping=5)

        # mat3 = self.filter_matrix_R
        # for col in range(mat3.shape[1]):
        #     plt.plot(mat3[:, col].cpu().numpy(), label=f'Column {col}')
        # plt.title('Each Column as a Curve')
        # plt.xlabel('Row Index')
        # plt.ylabel('Value')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        # print(self.filter_matrix_R.sum(dim=0))
        # print(self.filter_matrix_R.sum(dim=1))


    def solve(self, fixed_pts: torch.Tensor, has_vision: bool = False):
        """
        Args:
            fixed_pts: (B, N+1 D), N为拍点云次数,1为起始位姿
            parameters_optimized: (B, D, T)
        """
        self.has_vision = has_vision
        self.resetVariables(fixed_pts)
        
        # 初始化最优轨迹
        self.InitializeTrajectory(fixed_pts)
        # print("self.parameters_total_cost:")
        # print(self.parameters_total_cost)
        # print("self.parameters_collision_cost:")
        # print(self.parameters_collision_cost)
        # if self.has_vision:
        #     print("self.parameters_vision_cost:")
        #     print(self.parameters_vision_cost / self.vision_cost_weight)
        # print("self.parameters_control_cost:")
        # print(self.parameters_total_cost - self.parameters_state_cost)

        # iteration block
        for _ in range(self.num_iterations):
            self.generateNoisyRollouts()
            self.computeNoisyRolloutsCosts()
            self.computeProbabilities()
            self.updateParameters()
        
        # print("self.parameters_total_cost:")
        # print(self.parameters_total_cost)
        # print("self.parameters_collision_cost:")
        # print(self.parameters_collision_cost)
        # if self.has_vision:
        #     print("self.parameters_vision_cost:")
        #     print(self.parameters_vision_cost / self.vision_cost_weight)
        # print("self.parameters_control_cost:")
        # print(self.parameters_total_cost - self.parameters_state_cost)

        # parameters_control_costs = stomp_utils.computeParametersControlCosts_2(
        #     self.parameters_optimized.unsqueeze(1),
        #     self.delta_t,
        #     self.control_cost_weight).squeeze(1)
        # parameters_total_cost = self.parameters_state_cost.clone() + parameters_control_costs[:, :, 0].sum(dim=-1)
        parameters_total_cost = self.parameters_total_cost.clone()

        return self.parameters_optimized, parameters_total_cost


    def InitializeTrajectory(self, fixed_pts: torch.Tensor):
        """
        Args:
            fixed_pts: (B, N+1 D), N为拍点云次数,1为起始位姿
            parameters_optimized: (B, D, T)
        """
        # 初始化最优轨迹
        # init段
        first = fixed_pts[:, 0]
        last = fixed_pts[:, 1]
        self.parameters_optimized[..., :self.num_timesteps_init] = self.computeInitialTrajectory(
            first, last, self.num_timesteps_init, self.control_cost_matrix_R_padded_init, self.inv_control_cost_matrix_R_init)
        # next段
        for i in range(self.num_transfer):
            first = fixed_pts[:, i+1]
            last = fixed_pts[:, i+2]
            start_idx = self.num_timesteps_init + i * (self.num_timesteps_next - 1) - 1
            end_idx = self.num_timesteps_init + (i+1) * (self.num_timesteps_next - 1)
            self.parameters_optimized[..., start_idx:end_idx] = self.computeInitialTrajectory(
            first, last, self.num_timesteps_next, self.control_cost_matrix_R_padded_next, self.inv_control_cost_matrix_R_init)

        # 计算最优轨迹成本
        # Compute state costs
        collision_cost, vision_cost = self.computeStateCost(self.parameters_optimized.unsqueeze(1))     # (B, D, T)
        collision_cost = collision_cost.squeeze(1)
        vision_cost = vision_cost.squeeze(1)
        self.parameters_state_costs = collision_cost + vision_cost
        # Compute state costs
        if self.control_cost_weight > MIN_CONTROL_COST_WEIGHT:
            # self.parameters_control_costs = stomp_utils.computeParametersControlCosts(
            #     self.parameters_optimized.unsqueeze(1),
            #     self.delta_t,
            #     self.control_cost_weight,
            #     self.control_cost_matrix_R).squeeze(1)
            self.parameters_control_costs = stomp_utils.computeParametersControlCosts_1(
                            self.parameters_optimized.unsqueeze(1),
                            self.delta_t,
                            self.control_cost_weight).squeeze(1)
            # self.parameters_control_costs = stomp_utils.computeParametersControlCosts_2(
            #                 self.parameters_optimized.unsqueeze(1),
            #                 self.delta_t,
            #                 self.control_cost_weight).squeeze(1)
        # Compute total costs, total_costs[d, t]
        self.parameters_total_costs = self.parameters_state_costs + self.parameters_control_costs       # (B, D, T)
        self.parameters_total_cost = self.parameters_state_costs[:, 0].sum(dim=-1) + self.parameters_control_costs[:, :, 0].sum(dim=-1)  # (B,)
        self.parameters_state_cost = self.parameters_state_costs[:, 0].sum(dim=-1)
        self.parameters_collision_cost = collision_cost[:, 0].sum(dim=-1)
        self.parameters_vision_cost = vision_cost[:, 0].sum(dim=-1)
        self.noise_scale = torch.where(
            (self.parameters_collision_cost + self.parameters_vision_cost / self.vision_cost_weight)
            > (self.collision_cost_weight - 0.1), self.noise_scale_high, self.noise_scale_low)
        self.filter_scale = torch.where(
            (self.parameters_collision_cost + self.parameters_vision_cost / self.vision_cost_weight)
            > (self.collision_cost_weight - 0.1), self.filter_scale_high, self.filter_scale_low)


    def computeInitialTrajectory(self, first: torch.Tensor, last: torch.Tensor, num_timesteps: int, 
                                 control_cost_matrix_R_padded: torch.Tensor, inv_control_cost_matrix_R: torch.Tensor):
        """
        生成初始轨迹
        Args:
            first: (B, D)
            last: (B, D)
            parameters_optimized: (B, D, T)
        """
        if self.initialization_method == TrajectoryInitializations.CUBIC_POLYNOMIAL_INTERPOLATION:
            parameters_optimized =  stomp_utils.computeCubicInterpolation(first, last, num_timesteps, self.delta_t)
        elif self.initialization_method == TrajectoryInitializations.LINEAR_INTERPOLATION:
            parameters_optimized = stomp_utils.computeLinearInterpolation(first, last, num_timesteps)
        elif self.initialization_method == TrajectoryInitializations.MININUM_CONTROL_COST:
            parameters_optimized = stomp_utils.computeMinCostTrajectory(
                first, last, control_cost_matrix_R_padded, inv_control_cost_matrix_R)
        parameters_optimized = torch.clamp(input=parameters_optimized.transpose(-2, -1), 
                                           min=self.joints_lower_limit, max=self.joints_upper_limit).transpose(-2, -1)
        return parameters_optimized


    def generateNoisyRollouts(self):
        """
        生成加噪声的轨迹
        """
        rollouts_stored = self.num_rollouts_all         # take the optimized rollout into account
        rollouts_generate = self.num_rollouts_new
        rollouts_reuse = self.num_rollouts_old         

        def getSamples(inv_control_cost_matrix_R, num_rollouts, num_timesteps):
            mean = torch.zeros((inv_control_cost_matrix_R.shape[0],), device=self.device)
            cov = inv_control_cost_matrix_R
            distribution = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)
            samples = distribution.sample((self.num_batch * num_rollouts * self.num_dimensions,))
            samples = samples.reshape(self.num_batch, num_rollouts, self.num_dimensions, num_timesteps)
            return samples

        if self.init_it:
            # 上一次迭代的最优rollout
            self.noise[:,-1] = 0
            self.stored_rollouts[:, -1] = self.parameters_optimized
            self.control_costs[:, -1] = self.parameters_control_costs
            self.state_costs[:, -1] = self.parameters_state_costs
            self.total_costs[:, -1] = self.parameters_total_costs
            self.total_cost[:, -1] = self.parameters_total_cost

            # 生成新的noisy rollouts
            samples = []
            # init段
            samples.append(getSamples(self.inv_control_cost_matrix_R_init, 
                                      self.num_rollouts_new + self.num_rollouts_old, self.num_timesteps_init))
            # next段
            for i in range(self.num_transfer):
                samples.append(getSamples(self.inv_control_cost_matrix_R_next, 
                                      self.num_rollouts_new + self.num_rollouts_old, self.num_timesteps_next)[..., 1:])
            samples = torch.cat(samples, dim=-1)
            samples, samples_rollout = self.filterNoisyRollouts(samples)
            self.noise[:, :(rollouts_generate + rollouts_reuse)] = samples
            self.stored_rollouts[:, :(rollouts_generate + rollouts_reuse)] = samples_rollout

        else:
            # 复用的rollouts
            noise_stored = self.stored_rollouts - self.parameters_optimized.unsqueeze(1)
            _, idx_top = torch.topk(self.total_cost, rollouts_reuse, dim=-1, largest=False)
            idx_top_4 = idx_top.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.num_dimensions, self.num_timesteps_all)
            self.noise[:, -1-rollouts_reuse:-1] = torch.gather(input=noise_stored, dim=1, index=idx_top_4)
            self.stored_rollouts[:, -1-rollouts_reuse:-1] = torch.gather(input=self.stored_rollouts, dim=1, index=idx_top_4)
            self.control_costs[:, -1-rollouts_reuse:-1] = torch.gather(input=self.control_costs, dim=1, index=idx_top_4)
            self.state_costs[:, -1-rollouts_reuse:-1] = torch.gather(input=self.state_costs, dim=1, index=idx_top_4)
            self.total_costs[:, -1-rollouts_reuse:-1] = torch.gather(input=self.total_costs, dim=1, index=idx_top_4)
            self.total_cost[:, -1-rollouts_reuse:-1] = torch.gather(input=self.total_cost, dim=1, index=idx_top)
            
            # 上一次迭代的最优rollout
            self.noise[:,-1] = 0
            self.stored_rollouts[:, -1] = self.parameters_optimized
            self.control_costs[:, -1] = self.parameters_control_costs
            self.state_costs[:, -1] = self.parameters_state_costs
            self.total_costs[:, -1] = self.parameters_total_costs
            self.total_cost[:, -1] = self.parameters_total_cost

            # 生成新的noisy rollouts
            samples = []
            # init段
            samples.append(getSamples(self.inv_control_cost_matrix_R_init, 
                                      self.num_rollouts_new, self.num_timesteps_init))
            # next段
            for i in range(self.num_transfer):
                samples.append(getSamples(self.inv_control_cost_matrix_R_next, 
                                      self.num_rollouts_new, self.num_timesteps_next)[..., 1:])
            samples = torch.cat(samples, dim=-1)
            samples, samples_rollout = self.filterNoisyRollouts(samples)
            self.noise[:, :rollouts_generate] = samples
            self.stored_rollouts[:, :rollouts_generate] = samples_rollout


    def filterNoisyRollouts(self, samples: torch.Tensor):
        """
        预先对噪音进行滤波,并约束超出限位的rollouts
        Args:
            samples: (B, R, D, T)
        """
        # scale noise
        samples = self.noise_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * samples

        # filter joints
        if self.pre_filter:
            samples = (self.filter_matrix_R.unsqueeze(0).unsqueeze(0) @ samples.transpose(-2, -1)).transpose(-2, -1)    # (B, R, D, T)

        # clamp joints
        samples_rollout = samples + self.parameters_optimized.unsqueeze(1)          # (B, R, D, T)
        samples_rollout = samples_rollout.transpose(-2, -1)                         # (B, R, T, D)
        samples_rollout = torch.clamp(input=samples_rollout, min=self.joints_lower_limit, max=self.joints_upper_limit)
        samples_rollout = samples_rollout.transpose(-2, -1)                         # (B, R, D, T)
        samples = samples_rollout - self.parameters_optimized.unsqueeze(1)

        return samples, samples_rollout
        

    def computeNoisyRolloutsCosts(self):
        """
        计算轨迹中各关节d和各时间点t对应的成本函数,cost(d, t)
        """
        if self.init_it:
            # Compute state costs
            collision_cost, vision_cost = self.computeStateCost(self.stored_rollouts)
            self.state_costs = collision_cost + vision_cost
            # Compute control costs
            if self.control_cost_weight > MIN_CONTROL_COST_WEIGHT:
                # self.control_costs = stomp_utils.computeParametersControlCosts(self.stored_rollouts,
                #                                                                self.delta_t,
                #                                                                self.control_cost_weight,
                #                                                                self.control_cost_matrix_R)
                self.control_costs = stomp_utils.computeParametersControlCosts_1(self.stored_rollouts,
                                                                               self.delta_t,
                                                                               self.control_cost_weight)
                # self.control_costs = stomp_utils.computeParametersControlCosts_2(self.stored_rollouts,
                #                                                                self.delta_t,
                #                                                                self.control_cost_weight)

            # Compute total costs, total_costs[d, t]
            self.total_costs = self.state_costs + self.control_costs    # (B, R, D, T)
            self.total_cost = self.state_costs[:, :, 0].sum(dim=-1) + self.control_costs[:, :, :, 0].sum(dim=-1)

        else:
            # Compute state costs
            collision_cost, vision_cost = self.computeStateCost(
                self.stored_rollouts[:, :self.num_rollouts_new])
            self.state_costs[:, :self.num_rollouts_new] = collision_cost + vision_cost
            # Compute state costs
            if self.control_cost_weight > MIN_CONTROL_COST_WEIGHT:
                # self.control_costs[:, :self.num_rollouts_new] = stomp_utils.computeParametersControlCosts(
                #     self.stored_rollouts[:, :self.num_rollouts_new],
                #     self.delta_t,
                #     self.control_cost_weight,
                #     self.control_cost_matrix_R)
                self.control_costs[:, :self.num_rollouts_new] = stomp_utils.computeParametersControlCosts_1(
                    self.stored_rollouts[:, :self.num_rollouts_new],
                    self.delta_t,
                    self.control_cost_weight)
                # self.control_costs[:, :self.num_rollouts_new] = stomp_utils.computeParametersControlCosts_2(
                #     self.stored_rollouts[:, :self.num_rollouts_new],
                #     self.delta_t,
                #     self.control_cost_weight)

            # Compute total costs, total_costs[d, t]
            self.total_costs[:, :self.num_rollouts_new] = (self.state_costs[:, :self.num_rollouts_new] 
                                                           + self.control_costs[:, :self.num_rollouts_new])    # (B, R, D, T)
            self.total_cost[:, :self.num_rollouts_new] = (self.state_costs[:, :self.num_rollouts_new, 0].sum(dim=-1) 
                                                          + self.control_costs[:, :self.num_rollouts_new, :, 0].sum(dim=-1))


    def computeStateCost(self, rollouts:torch.Tensor):
        """"
        计算状态成本
        Args:
            rollouts: (B, R, D, T)
        """
        collision_cost = self.scene.computeCollisionCost(rollouts) * self.collision_cost_weight     # (B, R, D, T)
        vision_cost = torch.zeros_like(collision_cost, dtype=torch.float, device=self.device)
        if self.has_vision:
            # vision_cost[:, :, :, 10:51:10] =\
            #     self.scene.computeVisionCost(rollouts[:, :, :, 10:51:10]) * self.vision_cost_weight      # (B, R, D, T)
            vision_cost[:, :, :, 10:41:10] =\
                self.scene.computeVisionCost(rollouts[:, :, :, 10:41:10]) * self.vision_cost_weight      # (B, R, D, T)
            
        return collision_cost, vision_cost


    def computeProbabilities(self):
        """
        计算在每个关节维度d上每个时间步t时每个rollouts的概率
        """
        h = self.exponentiated_cost_sensitivity
        total_costs_T = self.total_costs.clone().permute(0, 2, 3, 1)                        # (B, D, T, R)
        
        # Find min and max cost over all rollouts at each timestep
        min_costs_per_t = torch.min(input=total_costs_T, dim=-1)[0].unsqueeze(-1)           # (B, D, T, 1)
        max_costs_per_t = torch.max(input=total_costs_T, dim=-1)[0].unsqueeze(-1)           # (B, D, T, 1)

        denom_per_t = max_costs_per_t - min_costs_per_t                                     # (B, D, T, 1)
        denom_per_t = torch.clamp(denom_per_t, min=MIN_COST_DIFFERENCE)                     # prevent division by zero
        
        # Compute probs for all rollouts and timesteps at once for dimension d
        exponents = -h * (total_costs_T - min_costs_per_t) / denom_per_t                    # (B, D, T, R)
        probs_unnormalized = self.importance_weights * torch.exp(exponents)                 # (B, D, T, R)
        probs_sum = torch.clamp(probs_unnormalized.sum(dim=-1, keepdim=True), min=1e-12)    # (B, D, T, 1)
        self.probabilities = probs_unnormalized / probs_sum                                 # (B, D, T, R)


    def updateParameters(self):
        """
        更新最优轨迹
        """
        noise = self.noise.clone().permute(0, 2, 3, 1)                          # (B, D, T, R)
        
        # computing updates from probabilities using convex combination
        parameters_noise = (noise * self.probabilities).sum(dim=-1)             # (B, D, T)

        if self.post_filter:
            # update filter scale
            self.noise_scale = torch.where(
                (self.parameters_collision_cost + self.parameters_vision_cost / self.vision_cost_weight)
                > (self.collision_cost_weight - 0.1), self.noise_scale_high, self.noise_scale_low)
            self.filter_scale = torch.where(
                (self.parameters_collision_cost + self.parameters_vision_cost / self.vision_cost_weight)
                > (self.collision_cost_weight - 0.1), self.filter_scale_high, self.filter_scale_low)
            parameters_noise = (self.filter_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                                 * self.filter_matrix_R.unsqueeze(0).unsqueeze(0) 
                                 @ parameters_noise.unsqueeze(-1)).squeeze(-1)
        parameters_noise[..., 0] = 0
        parameters_noise[..., -1] = 0
        parameters_optimized = parameters_noise + self.parameters_optimized     # (B, D, T)

        # print("parameters_noise:")
        # print(parameters_noise[0])
        
        # get update indices
        update_idx = self.computeOptimizedCost(parameters_optimized)
        # self.parameters_noise[update_idx] = parameters_noise[update_idx]
        self.parameters_optimized[update_idx] = parameters_optimized[update_idx]

        self.init_it = False
        

    def computeOptimizedCost(self, parameters_optimized):
        """
        计算更新后最优轨迹的成本
        """
        # Compute state costs
        collision_cost, vision_cost = self.computeStateCost(parameters_optimized.unsqueeze(1))      # (B, D, T)
        collision_cost = collision_cost.squeeze(1)
        vision_cost = vision_cost.squeeze(1)
        parameters_state_costs = collision_cost + vision_cost
        # Compute state costs
        if self.control_cost_weight > MIN_CONTROL_COST_WEIGHT:
            # parameters_control_costs = stomp_utils.computeParametersControlCosts(
            #     parameters_optimized.unsqueeze(1),
            #     self.delta_t,
            #     self.control_cost_weight,
            #     self.control_cost_matrix_R).squeeze(1)
            parameters_control_costs = stomp_utils.computeParametersControlCosts_1(
                parameters_optimized.unsqueeze(1),
                self.delta_t,
                self.control_cost_weight).squeeze(1)
            # parameters_control_costs = stomp_utils.computeParametersControlCosts_2(
            #     parameters_optimized.unsqueeze(1),
            #     self.delta_t,
            #     self.control_cost_weight).squeeze(1)
        # Compute total costs, total_costs[d, t]
        parameters_total_costs = parameters_state_costs + parameters_control_costs       # (B, D, T)
        parameters_total_cost = parameters_state_costs[:, 0].sum(dim=-1) + parameters_control_costs[:, :, 0].sum(dim=-1)  # (B,)
        parameters_state_cost = parameters_state_costs[:, 0].sum(dim=-1)
        # print("state cost:")
        # print(parameters_state_costs[:, 0].sum(dim=-1))
        # print("control cost:")
        # print(parameters_control_costs[:, :, 0].sum(dim=-1))
        parameters_collision_cost = collision_cost[:, 0].sum(dim=-1)
        parameters_vision_cost = vision_cost[:, 0].sum(dim=-1)

        # Update parameters, meaning optimized rollout
        update_idx = parameters_total_cost < self.parameters_total_cost
        # print(update_idx)
        # print(parameters_total_cost)
        self.parameters_state_costs[update_idx] = parameters_state_costs[update_idx]                     # (B, D, T)
        if self.control_cost_weight > MIN_CONTROL_COST_WEIGHT:
            self.parameters_control_costs[update_idx] = parameters_control_costs[update_idx]             # (B, D, T)
        self.parameters_total_costs[update_idx] = parameters_total_costs[update_idx]
        self.parameters_total_cost[update_idx] = parameters_total_cost[update_idx]
        self.parameters_state_cost[update_idx] = parameters_state_cost[update_idx]
        self.parameters_collision_cost[update_idx] = parameters_collision_cost[update_idx]
        self.parameters_vision_cost[update_idx] = parameters_vision_cost[update_idx]

        return update_idx
