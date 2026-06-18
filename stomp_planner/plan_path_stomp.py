"""
基于本项目 STOMP 方法的「自由空间单段路径规划」独立脚本。

用途：
    给定起始关节角 cur_cfg 与目标关节角 target_cfg（均在 free 空间，无障碍），
    用本项目的 STOMP 算法规划一条平滑的关节空间轨迹。

与本项目 Traj 阶段（stomp_traj.py）的关系：
    - STOMP 的全部核心数学（控制代价矩阵 R、N(0,R⁻¹) 噪声采样、概率指数加权、
      多段平滑滤波矩阵）直接复用本项目的 `stomp_utils_traj.py`，逐函数一致；
    - 唯一区别：原项目的「状态代价」来自 Isaac Lab 仿真（碰撞 + 视野），需要起仿真器；
      本脚本场景在 free 空间，故把状态代价（碰撞/视野）置 0，于是 STOMP 退化为
      在关节限位内、固定首末点、最小化加速度（控制代价）的平滑规划。
    - 单段规划（cur → target，无中间观测点）对应 num_transfer = 0，
      generateSmoothingMatrix_1 自然退化为标准单段平滑矩阵。

机械臂配置：从 curobo 的 ur12e.yml 读取关节名与 retract_config，
    关节限位从其引用的 URDF 读取（与项目 config_traj 的 *_all 限位一致）。

运行：
    python plan_path_stomp.py                       # 用脚本内默认 cur/target
    python plan_path_stomp.py --output traj.npy     # 保存结果
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import torch

# 复用本项目的 STOMP 数学工具（纯 torch/numpy，无 Isaac 依赖）
_PROJ_DIR = os.path.dirname(os.path.realpath(__file__))
if _PROJ_DIR not in sys.path:
    sys.path.insert(0, _PROJ_DIR)

import stomp_utils_traj as stomp_utils
from stomp_utils_traj import (
    TrajectoryInitializations,
    DEFAULT_NOISY_COST_IMPORTANCE_WEIGHT,
    MIN_COST_DIFFERENCE,
    MIN_CONTROL_COST_WEIGHT,
)

torch.set_default_dtype(torch.float)

# 与本项目一致的固定随机种子
SEED = 3
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ----------------------------------------------------------------------------- #
# 读取机械臂配置（关节名 / 限位 / retract_config）
# ----------------------------------------------------------------------------- #
def load_robot_cfg(yml_path: str):
    """从 curobo ur12e.yml + 其引用的 URDF 读取关节名、关节限位、retract_config。"""
    import yaml

    with open(yml_path, "r") as f:
        cfg = yaml.safe_load(f)
    kin = cfg["robot_cfg"]["kinematics"]
    joint_names = kin["cspace"]["joint_names"]
    retract = kin["cspace"].get("retract_config", None)

    urdf_path = kin["urdf_path"]
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.join(os.path.dirname(yml_path), urdf_path)

    lower, upper = [], []
    root = ET.parse(urdf_path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    for name in joint_names:
        lim = joints[name].find("limit")
        lower.append(float(lim.get("lower")))
        upper.append(float(lim.get("upper")))

    return {
        "joint_names": joint_names,
        "lower_limit": lower,
        "upper_limit": upper,
        "retract": retract,
    }


# ----------------------------------------------------------------------------- #
# 自由空间 STOMP（复用本项目核心，状态代价置 0）
# ----------------------------------------------------------------------------- #
class FreeSpaceStomp:
    """
    本项目 StompTraj 的自由空间精简版。
    流程/张量布局/更新规则与 stomp_traj.py 完全一致，仅 computeStateCost 恒为 0。
    """

    def __init__(self, lower_limit, upper_limit, device="cuda",
                 num_batch=1, num_timesteps=51, delta_t=0.1,
                 num_iterations=40, num_rollouts_new=30, num_rollouts_old=20,
                 control_cost_weight=0.1, exponentiated_cost_sensitivity=2.0,
                 noise_scale=1.0, filter_scale=2.0,
                 initialization_method=TrajectoryInitializations.LINEAR_INTERPOLATION,
                 post_filter=True, pre_filter=False):
        self.device = device
        self.num_batch = num_batch
        self.num_dimensions = 6
        # 单段规划：init 段 = next 段 = 整条轨迹
        self.num_timesteps_init = num_timesteps
        self.num_timesteps_next = num_timesteps
        self.num_timesteps_all = num_timesteps
        self.num_iterations = num_iterations
        self.num_rollouts_new = num_rollouts_new
        self.num_rollouts_old = num_rollouts_old
        self.num_rollouts_all = num_rollouts_new + num_rollouts_old + 1
        self.delta_t = delta_t
        self.control_cost_weight = control_cost_weight
        self.exponentiated_cost_sensitivity = exponentiated_cost_sensitivity
        self.initialization_method = initialization_method
        self.post_filter = post_filter
        self.pre_filter = pre_filter
        self.noise_scale_val = noise_scale
        self.filter_scale_val = filter_scale

        self.joints_lower_limit = torch.tensor(lower_limit, device=device)
        self.joints_upper_limit = torch.tensor(upper_limit, device=device)

        self.init_it = True
        self.num_transfer = 0

        # 控制代价矩阵 R 及其逆（噪声协方差），与 StompTraj.__init__ 一致
        (self.control_cost_matrix_R_padded_init,
         self.control_cost_matrix_R_init,
         self.inv_control_cost_matrix_R_init) = stomp_utils.generateControlCostMatrix(
            self.num_timesteps_init, self.delta_t, True, self.device)
        (self.control_cost_matrix_R_padded_next,
         self.control_cost_matrix_R_next,
         self.inv_control_cost_matrix_R_next) = stomp_utils.generateControlCostMatrix(
            self.num_timesteps_next, self.delta_t, True, self.device)

    # --- 状态代价：free 空间恒为 0（替换原项目的仿真碰撞/视野代价）---
    def computeStateCost(self, rollouts: torch.Tensor):
        collision_cost = torch.zeros_like(rollouts)
        vision_cost = torch.zeros_like(rollouts)
        return collision_cost, vision_cost

    def resetVariables(self, fixed_pts: torch.Tensor):
        self.init_it = True
        self.num_transfer = fixed_pts.shape[1] - 2
        self.num_timesteps_all = (self.num_timesteps_init
                                  + (self.num_timesteps_next - 1) * self.num_transfer)
        B, RA, D, T = self.num_batch, self.num_rollouts_all, self.num_dimensions, self.num_timesteps_all

        self.noise = torch.zeros((B, RA, D, T), device=self.device)
        self.control_costs = torch.zeros((B, RA, D, T), device=self.device)
        self.state_costs = torch.zeros((B, RA, D, T), device=self.device)
        self.total_costs = torch.zeros((B, RA, D, T), device=self.device)
        self.total_cost = torch.zeros((B, RA), device=self.device)

        self.probabilities = torch.zeros((B, D, T, RA), device=self.device)
        self.importance_weights = DEFAULT_NOISY_COST_IMPORTANCE_WEIGHT * torch.ones_like(self.probabilities)

        self.stored_rollouts = torch.zeros((B, RA, D, T), device=self.device)
        self.parameters_optimized = torch.zeros((B, D, T), device=self.device)
        self.parameters_control_costs = torch.zeros((B, D, T), device=self.device)
        self.parameters_state_costs = torch.zeros((B, D, T), device=self.device)
        self.parameters_total_costs = torch.zeros((B, D, T), device=self.device)
        self.parameters_total_cost = torch.zeros((B,), device=self.device)
        self.parameters_state_cost = torch.zeros((B,), device=self.device)

        # free 空间下 noise/filter scale 取「低代价」分支（恒定）
        self.noise_scale = self.noise_scale_val * torch.ones((B,), device=self.device)
        self.filter_scale = self.filter_scale_val * torch.ones((B,), device=self.device)

        (self.control_cost_matrix_R_padded,
         self.control_cost_matrix_R, _) = stomp_utils.generateControlCostMatrix(
            self.num_timesteps_all, self.delta_t, True, self.device)
        self.filter_matrix_R = stomp_utils.generateSmoothingMatrix_1(
            self.inv_control_cost_matrix_R_init, self.num_transfer,
            self.num_timesteps_next, damping=5)

    def solve(self, fixed_pts: torch.Tensor):
        self.resetVariables(fixed_pts)
        self.InitializeTrajectory(fixed_pts)
        for _ in range(self.num_iterations):
            self.generateNoisyRollouts()
            self.computeNoisyRolloutsCosts()
            self.computeProbabilities()
            self.updateParameters()
        return self.parameters_optimized, self.parameters_total_cost.clone()

    def InitializeTrajectory(self, fixed_pts: torch.Tensor):
        first = fixed_pts[:, 0]
        last = fixed_pts[:, 1]
        self.parameters_optimized[..., :self.num_timesteps_init] = self.computeInitialTrajectory(
            first, last, self.num_timesteps_init,
            self.control_cost_matrix_R_padded_init, self.inv_control_cost_matrix_R_init)
        for i in range(self.num_transfer):
            first = fixed_pts[:, i + 1]
            last = fixed_pts[:, i + 2]
            start_idx = self.num_timesteps_init + i * (self.num_timesteps_next - 1) - 1
            end_idx = self.num_timesteps_init + (i + 1) * (self.num_timesteps_next - 1)
            self.parameters_optimized[..., start_idx:end_idx] = self.computeInitialTrajectory(
                first, last, self.num_timesteps_next,
                self.control_cost_matrix_R_padded_next, self.inv_control_cost_matrix_R_init)

        collision_cost, vision_cost = self.computeStateCost(self.parameters_optimized.unsqueeze(1))
        collision_cost = collision_cost.squeeze(1)
        vision_cost = vision_cost.squeeze(1)
        self.parameters_state_costs = collision_cost + vision_cost
        if self.control_cost_weight > MIN_CONTROL_COST_WEIGHT:
            self.parameters_control_costs = stomp_utils.computeParametersControlCosts_1(
                self.parameters_optimized.unsqueeze(1), self.delta_t,
                self.control_cost_weight).squeeze(1)
        self.parameters_total_costs = self.parameters_state_costs + self.parameters_control_costs
        self.parameters_total_cost = (self.parameters_state_costs[:, 0].sum(dim=-1)
                                      + self.parameters_control_costs[:, :, 0].sum(dim=-1))
        self.parameters_state_cost = self.parameters_state_costs[:, 0].sum(dim=-1)

    def computeInitialTrajectory(self, first, last, num_timesteps,
                                 control_cost_matrix_R_padded, inv_control_cost_matrix_R):
        if self.initialization_method == TrajectoryInitializations.CUBIC_POLYNOMIAL_INTERPOLATION:
            traj = stomp_utils.computeCubicInterpolation(first, last, num_timesteps, self.delta_t)
        elif self.initialization_method == TrajectoryInitializations.LINEAR_INTERPOLATION:
            traj = stomp_utils.computeLinearInterpolation(first, last, num_timesteps)
        elif self.initialization_method == TrajectoryInitializations.MININUM_CONTROL_COST:
            traj = stomp_utils.computeMinCostTrajectory(
                first, last, control_cost_matrix_R_padded, inv_control_cost_matrix_R)
        traj = torch.clamp(traj.transpose(-2, -1),
                           min=self.joints_lower_limit, max=self.joints_upper_limit).transpose(-2, -1)
        return traj

    def _getSamples(self, inv_R, num_rollouts, num_timesteps):
        mean = torch.zeros((inv_R.shape[0],), device=self.device)
        dist = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=inv_R)
        s = dist.sample((self.num_batch * num_rollouts * self.num_dimensions,))
        return s.reshape(self.num_batch, num_rollouts, self.num_dimensions, num_timesteps)

    def generateNoisyRollouts(self):
        rollouts_generate = self.num_rollouts_new
        rollouts_reuse = self.num_rollouts_old

        if self.init_it:
            self.noise[:, -1] = 0
            self.stored_rollouts[:, -1] = self.parameters_optimized
            self.control_costs[:, -1] = self.parameters_control_costs
            self.state_costs[:, -1] = self.parameters_state_costs
            self.total_costs[:, -1] = self.parameters_total_costs
            self.total_cost[:, -1] = self.parameters_total_cost

            samples = [self._getSamples(self.inv_control_cost_matrix_R_init,
                                        rollouts_generate + rollouts_reuse, self.num_timesteps_init)]
            for _ in range(self.num_transfer):
                samples.append(self._getSamples(self.inv_control_cost_matrix_R_next,
                                                rollouts_generate + rollouts_reuse,
                                                self.num_timesteps_next)[..., 1:])
            samples = torch.cat(samples, dim=-1)
            samples, samples_rollout = self.filterNoisyRollouts(samples)
            self.noise[:, :(rollouts_generate + rollouts_reuse)] = samples
            self.stored_rollouts[:, :(rollouts_generate + rollouts_reuse)] = samples_rollout
        else:
            noise_stored = self.stored_rollouts - self.parameters_optimized.unsqueeze(1)
            _, idx_top = torch.topk(self.total_cost, rollouts_reuse, dim=-1, largest=False)
            idx4 = idx_top.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.num_dimensions, self.num_timesteps_all)
            self.noise[:, -1 - rollouts_reuse:-1] = torch.gather(noise_stored, 1, idx4)
            self.stored_rollouts[:, -1 - rollouts_reuse:-1] = torch.gather(self.stored_rollouts, 1, idx4)
            self.control_costs[:, -1 - rollouts_reuse:-1] = torch.gather(self.control_costs, 1, idx4)
            self.state_costs[:, -1 - rollouts_reuse:-1] = torch.gather(self.state_costs, 1, idx4)
            self.total_costs[:, -1 - rollouts_reuse:-1] = torch.gather(self.total_costs, 1, idx4)
            self.total_cost[:, -1 - rollouts_reuse:-1] = torch.gather(self.total_cost, 1, idx_top)

            self.noise[:, -1] = 0
            self.stored_rollouts[:, -1] = self.parameters_optimized
            self.control_costs[:, -1] = self.parameters_control_costs
            self.state_costs[:, -1] = self.parameters_state_costs
            self.total_costs[:, -1] = self.parameters_total_costs
            self.total_cost[:, -1] = self.parameters_total_cost

            samples = [self._getSamples(self.inv_control_cost_matrix_R_init,
                                        rollouts_generate, self.num_timesteps_init)]
            for _ in range(self.num_transfer):
                samples.append(self._getSamples(self.inv_control_cost_matrix_R_next,
                                                rollouts_generate, self.num_timesteps_next)[..., 1:])
            samples = torch.cat(samples, dim=-1)
            samples, samples_rollout = self.filterNoisyRollouts(samples)
            self.noise[:, :rollouts_generate] = samples
            self.stored_rollouts[:, :rollouts_generate] = samples_rollout

    def filterNoisyRollouts(self, samples: torch.Tensor):
        samples = self.noise_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * samples
        if self.pre_filter:
            samples = (self.filter_matrix_R.unsqueeze(0).unsqueeze(0)
                       @ samples.transpose(-2, -1)).transpose(-2, -1)
        samples_rollout = samples + self.parameters_optimized.unsqueeze(1)
        samples_rollout = samples_rollout.transpose(-2, -1)
        samples_rollout = torch.clamp(samples_rollout,
                                      min=self.joints_lower_limit, max=self.joints_upper_limit)
        samples_rollout = samples_rollout.transpose(-2, -1)
        samples = samples_rollout - self.parameters_optimized.unsqueeze(1)
        return samples, samples_rollout

    def computeNoisyRolloutsCosts(self):
        if self.init_it:
            collision_cost, vision_cost = self.computeStateCost(self.stored_rollouts)
            self.state_costs = collision_cost + vision_cost
            if self.control_cost_weight > MIN_CONTROL_COST_WEIGHT:
                self.control_costs = stomp_utils.computeParametersControlCosts_1(
                    self.stored_rollouts, self.delta_t, self.control_cost_weight)
            self.total_costs = self.state_costs + self.control_costs
            self.total_cost = (self.state_costs[:, :, 0].sum(dim=-1)
                               + self.control_costs[:, :, :, 0].sum(dim=-1))
        else:
            n = self.num_rollouts_new
            collision_cost, vision_cost = self.computeStateCost(self.stored_rollouts[:, :n])
            self.state_costs[:, :n] = collision_cost + vision_cost
            if self.control_cost_weight > MIN_CONTROL_COST_WEIGHT:
                self.control_costs[:, :n] = stomp_utils.computeParametersControlCosts_1(
                    self.stored_rollouts[:, :n], self.delta_t, self.control_cost_weight)
            self.total_costs[:, :n] = self.state_costs[:, :n] + self.control_costs[:, :n]
            self.total_cost[:, :n] = (self.state_costs[:, :n, 0].sum(dim=-1)
                                      + self.control_costs[:, :n, :, 0].sum(dim=-1))

    def computeProbabilities(self):
        h = self.exponentiated_cost_sensitivity
        total_costs_T = self.total_costs.clone().permute(0, 2, 3, 1)
        min_t = torch.min(total_costs_T, dim=-1)[0].unsqueeze(-1)
        max_t = torch.max(total_costs_T, dim=-1)[0].unsqueeze(-1)
        denom = torch.clamp(max_t - min_t, min=MIN_COST_DIFFERENCE)
        exponents = -h * (total_costs_T - min_t) / denom
        probs = self.importance_weights * torch.exp(exponents)
        probs_sum = torch.clamp(probs.sum(dim=-1, keepdim=True), min=1e-12)
        self.probabilities = probs / probs_sum

    def updateParameters(self):
        noise = self.noise.clone().permute(0, 2, 3, 1)
        parameters_noise = (noise * self.probabilities).sum(dim=-1)
        if self.post_filter:
            parameters_noise = (self.filter_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                                * self.filter_matrix_R.unsqueeze(0).unsqueeze(0)
                                @ parameters_noise.unsqueeze(-1)).squeeze(-1)
        parameters_noise[..., 0] = 0
        parameters_noise[..., -1] = 0
        parameters_optimized = parameters_noise + self.parameters_optimized
        update_idx = self.computeOptimizedCost(parameters_optimized)
        self.parameters_optimized[update_idx] = parameters_optimized[update_idx]
        self.init_it = False

    def computeOptimizedCost(self, parameters_optimized):
        collision_cost, vision_cost = self.computeStateCost(parameters_optimized.unsqueeze(1))
        collision_cost = collision_cost.squeeze(1)
        vision_cost = vision_cost.squeeze(1)
        parameters_state_costs = collision_cost + vision_cost
        if self.control_cost_weight > MIN_CONTROL_COST_WEIGHT:
            parameters_control_costs = stomp_utils.computeParametersControlCosts_1(
                parameters_optimized.unsqueeze(1), self.delta_t, self.control_cost_weight).squeeze(1)
        parameters_total_costs = parameters_state_costs + parameters_control_costs
        parameters_total_cost = (parameters_state_costs[:, 0].sum(dim=-1)
                                 + parameters_control_costs[:, :, 0].sum(dim=-1))
        parameters_state_cost = parameters_state_costs[:, 0].sum(dim=-1)

        update_idx = parameters_total_cost < self.parameters_total_cost
        self.parameters_state_costs[update_idx] = parameters_state_costs[update_idx]
        if self.control_cost_weight > MIN_CONTROL_COST_WEIGHT:
            self.parameters_control_costs[update_idx] = parameters_control_costs[update_idx]
        self.parameters_total_costs[update_idx] = parameters_total_costs[update_idx]
        self.parameters_total_cost[update_idx] = parameters_total_cost[update_idx]
        self.parameters_state_cost[update_idx] = parameters_state_cost[update_idx]
        return update_idx


# ----------------------------------------------------------------------------- #
# 主流程
# ----------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="自由空间 STOMP 单段路径规划（基于本项目方法）")
    parser.add_argument("--robot_yml", type=str,
                        default="/home/a/Datas/curobo/example_new_robot/urdf_12e/ur12e.yml")
    parser.add_argument("--cur", type=float, nargs=6, default=None, help="起始关节角(6)")
    parser.add_argument("--target", type=float, nargs=6, default=None, help="目标关节角(6)")
    parser.add_argument("--num_timesteps", type=int, default=51)
    parser.add_argument("--num_iterations", type=int, default=40)
    parser.add_argument("--delta_t", type=float, default=0.1)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="保存轨迹的 .npy 路径(可选)")
    parser.add_argument("--plot", action="store_true", help="绘制各关节随时间变化曲线")
    args = parser.parse_args()

    cur_cfg = args.cur if args.cur is not None else [
        1.5707824230194092, -2.0071660480894984, 1.3613484541522425,
        -0.9599629205516357, -1.570770565663473, 0.0]
    target_cfg = args.target if args.target is not None else [
        3.442432165145874, -2.472576379776001, 1.4129290580749512,
        -0.4071854054927826, -2.634472131729126, -3.2598252296447754]

    robot = load_robot_cfg(args.robot_yml)
    lower = robot["lower_limit"]
    upper = robot["upper_limit"]
    print("机械臂关节:", robot["joint_names"])
    print("关节下限:", [round(x, 4) for x in lower])
    print("关节上限:", [round(x, 4) for x in upper])

    # 越限检查（free 空间，仅检查端点在限位内）
    for name, q in [("cur_cfg", cur_cfg), ("target_cfg", target_cfg)]:
        for i, (v, lo, hi) in enumerate(zip(q, lower, upper)):
            if v < lo - 1e-6 or v > hi + 1e-6:
                print(f"[警告] {name} 第 {i} 关节 {v:.4f} 超出限位 [{lo:.4f}, {hi:.4f}]")

    device = args.device
    cur = torch.tensor(cur_cfg, device=device)
    target = torch.tensor(target_cfg, device=device)

    # fixed_pts: (B, N+1, D)；单段规划 N+1 = 2（起点 + 终点）
    fixed_pts = torch.stack([cur, target], dim=0).unsqueeze(0)  # (1, 2, 6)

    stomp = FreeSpaceStomp(
        lower_limit=lower, upper_limit=upper, device=device,
        num_batch=1, num_timesteps=args.num_timesteps,
        delta_t=args.delta_t, num_iterations=args.num_iterations)

    traj, total_cost = stomp.solve(fixed_pts)   # traj: (B=1, D=6, T)
    traj_np = traj[0].transpose(0, 1).cpu().numpy()   # (T, 6)

    print(f"\n规划完成：轨迹形状 (T, D) = {traj_np.shape}，控制代价 = {float(total_cost[0]):.6f}")
    print("起点 (应=cur_cfg):   ", np.round(traj_np[0], 4))
    print("终点 (应=target_cfg):", np.round(traj_np[-1], 4))
    err_start = np.abs(traj_np[0] - np.array(cur_cfg)).max()
    err_end = np.abs(traj_np[-1] - np.array(target_cfg)).max()
    print(f"端点误差: start={err_start:.2e}, end={err_end:.2e}")
    in_limit = np.all(traj_np >= np.array(lower) - 1e-5) and np.all(traj_np <= np.array(upper) + 1e-5)
    print("全程在关节限位内:", bool(in_limit))

    if args.output:
        np.save(args.output, traj_np)
        print(f"轨迹已保存到: {args.output}")

    if args.plot:
        import matplotlib.pyplot as plt
        t = np.arange(traj_np.shape[0]) * args.delta_t
        for d in range(6):
            plt.plot(t, traj_np[:, d], label=f"joint{d+1}")
        plt.xlabel("time (s)")
        plt.ylabel("joint angle (rad)")
        plt.title("STOMP free-space trajectory")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
