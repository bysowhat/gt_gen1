"""
configuration
version: v0
date: 25.09.19
"""
import os
import json

from stomp_utils_traj import TrajectoryInitializations

class ConfigurationTraj:
    def __init__(self):
        # STOMP部分
        self.num_batch: int = 24                        # batch数量，即每条焊缝gt轨迹数量
        self.num_dimensions: int = 6                    # 机械臂自由度
        self.num_timesteps_init: int = 51               # 从起始到达第一个观测点轨迹时间点数,包括两个端点
        self.num_timesteps_next: int = 51               # 到达下一个观测点轨迹时间点数,包括两个端点
        self.num_iterations: int = 20                   # 迭代时间步数
        self.num_iterations_after_valid: int = 0         
        self.num_rollouts_new: int = 30                 # 生成轨迹数
        self.num_rollouts_old: int = 20                 # 复用轨迹数
        self.delta_t: float = 0.1                       # 间隔时间
        self.control_cost_weight: float = 0.1           # 控制成本权重(acc)
        # self.control_cost_weight: float = 0.3              # 控制成本权重(vel)
        self.collision_cost_weight: float = 20             # 碰撞成本权重
        self.vision_cost_weight: float = 1                # 视野成本权重  (1.0)
        self.visible_cost_weight: float = 1
        self.dir_cost_weight: float = 0.25                 # 相机方向成本权重    (0.25)
        self.initialization_method: TrajectoryInitializations = TrajectoryInitializations.LINEAR_INTERPOLATION     # 轨迹初始化方式
        self.exponentiated_cost_sensitivity: float = 2      # h值 = 0.5
        self.pre_filter: bool = False                       # 预先对噪音进行滤波
        self.post_filter: bool = True                       # 对优化后的噪音做后处理滤波
        self.noise_scale_high = 1
        self.noise_scale_low = 1
        self.filter_scale_high = 5
        self.filter_scale_low = 2

        self.joints_lower_limit_horizontal = [-0.7792, -2.6009, -0.0758, -3.2996, -3.1603, -3.1479]
        self.joints_upper_limit_horizontal = [ 3.9208,  0.3591,  2.5421, -0.4496,  0.8897,  1.6921]
        self.joints_lower_limit_vertical = [-0.7792, -2.6009, -0.0758, -3.2996, -3.1603, -6.2800]
        self.joints_upper_limit_vertical = [ 3.9208,  0.3591,  2.5421, -0.2496,  2.9397,  6.2800]
        self.joints_lower_limit_all = [-0.77920353, -2.60093713, -0.07581812, -3.29959822, -3.16031218, -6.19999981]
        self.joints_upper_limit_all = [ 3.92079639,  0.35906282,  2.54208183, -0.24959825,  2.93968773,  6.19999981]

        # 仿真环境
        # self.num_envs: int = self.num_select * self.num_rollouts_new * 8
        self.num_envs: int = self.num_batch * self.num_rollouts_new * 2
        self.usd_path = ""
        self.has_vision: bool = True
        self.has_pc: bool = False
        self.pc_path = ""
        # data_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
        # self.dir_path = os.path.join(data_dir_path, "seam_hs1", "p3")
        # self.seam_line_path = os.path.join(self.dir_path, "seam_line.pth")
        # self.seam_tangent_path = os.path.join(self.dir_path, "seam_tangent.pth")
        # self.seam_limits_path = os.path.join(self.dir_path, "seam_limits.pth")

        self.horizontal: int = 0