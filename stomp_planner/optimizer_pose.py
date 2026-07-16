"""
Optimizing the cam pose parameter of 6-DOF
version: v0
date: 25.09.22
"""

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import numpy as np

import random
import time

from config_pose import ConfigurationPose as Configuration
import opt_utils_pose as opt_utils
import opt_math_pose as math
from scene_pose import ScenePose as Scene


MIN_COST_DIFFERENCE = 1e-8
DEFAULT_NOISY_COST_IMPORTANCE_WEIGHT = 1.0          # 用于在计算概率时调整不同轨迹的贡献。


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


torch.set_printoptions(threshold=float('inf'), precision=6)


class OptimizerPose:
    def __init__(self, cfg: Configuration, scene: Scene, device="cuda"):      
        self.device = device
        self.cfg = cfg
        self.scene = scene

        # config
        # 进化策略优化设置：
        self.num_batches = self.cfg.num_batches
        self.num_iterations = self.cfg.num_iterations
        self.max_iterations = self.cfg.max_iterations
        self.max_iterations_refine = self.cfg.max_iterations_refine
        self.num_steps = self.cfg.num_steps
        self.num_randoms_new = self.cfg.num_randoms_new
        self.num_randoms_old = self.cfg.num_randoms_old
        self.num_randoms_all = self.cfg.num_randoms_all
        self.num_select = self.cfg.num_select
        self.num_next = self.cfg.num_next
        self.num_poses = self.cfg.num_poses

        self.collision_cost_weight = self.cfg.collision_cost_weight
        self.vision_pos_cost_weight = self.cfg.vision_pos_cost_weight
        self.vision_rot_cost_weight = self.cfg.vision_rot_cost_weight
        self.forced = self.cfg.forced
        self.has_space_cost = self.cfg.has_space_cost
        self.refine = self.cfg.refine

        self.exponentiated_cost_sensitivity = self.cfg.exponentiated_cost_sensitivity

        self.has_cov = self.cfg.has_cov
        if self.has_cov:
            self.covariance = torch.load(self.cfg.cov_path).to(self.device)
        else:
            sigma = self.cfg.sigma.to(self.device)
            self.covariance = torch.diag(sigma ** 2)
        
        # seam
        self.seam_line = torch.empty(0)             # (N, 3),   N为焊缝采样点的数量
        self.seam_lines = torch.empty(0)
        self.seam_tangent = torch.empty(0)          # (N, 3)
        self.seam_tangents = torch.empty(0)
        self.seam_limits = torch.empty(0)           # (N, 2, 3)
        self.seam_limits_s = torch.empty(0)
        self.num_points = self.seam_line.shape[0]

        # 定义更新变量
        # noise: delta (x, y, z), (wx, wy, wz)
        self.noise_optimized = torch.zeros((self.num_batches, 6), dtype=torch.float, device=self.device) 
        self.noise_stored = torch.zeros((self.num_batches, self.num_randoms_all, 6), dtype=torch.float, device=self.device)

        self.cam_pos_optimized = torch.zeros((self.num_batches, 3), dtype=torch.float, device=self.device)
        self.cam_rot_optimized = torch.zeros((self.num_batches, 3, 3), dtype=torch.float, device=self.device)
        self.cam_quat_optimized = torch.zeros((self.num_batches, 4), dtype=torch.float, device=self.device)
        self.cam_pose_optimized = torch.zeros((self.num_batches, 7), dtype=torch.float, device=self.device)
        self.joints_optimized = torch.zeros((self.num_batches, 6), dtype=torch.float, device=self.device)
        self.cam_pose_stored = torch.zeros((self.num_batches, self.num_randoms_all, 7), dtype=torch.float, device=self.device)
        self.cam_pose_backup = torch.zeros((self.num_batches, 4, 7), dtype=torch.float, device=self.device)
        self.cam_pose_init = torch.zeros((self.num_batches, 7), dtype=torch.float, device=self.device)

        self.pos_costs = torch.zeros((self.num_batches, self.num_randoms_all), dtype=torch.float, device=self.device)
        self.rot_costs = torch.zeros((self.num_batches, self.num_randoms_all), dtype=torch.float, device=self.device)
        self.pos_optimized_cost = torch.zeros((self.num_batches,), dtype=torch.float, device=self.device)
        self.pos_optimized_gate = torch.zeros((self.num_batches,), dtype=torch.float, device=self.device)
        self.rot_optimized_cost = torch.zeros((self.num_batches,), dtype=torch.float, device=self.device)
        self.original_costs = False
        self.pos_costs_o = torch.zeros((self.num_batches, self.num_randoms_all), dtype=torch.float, device=self.device)
        self.rot_costs_o = torch.zeros((self.num_batches, self.num_randoms_all), dtype=torch.float, device=self.device)
        self.pos_optimized_cost_o = torch.zeros((self.num_batches,), dtype=torch.float, device=self.device)
        self.rot_optimized_cost_o = torch.zeros((self.num_batches,), dtype=torch.float, device=self.device)

        self.probabilities_pos = torch.zeros((self.num_batches, self.num_select), dtype=torch.float, device=self.device)
        self.probabilities_rot = torch.zeros((self.num_batches, self.num_select), dtype=torch.float, device=self.device)
        self.importance_weights = DEFAULT_NOISY_COST_IMPORTANCE_WEIGHT * torch.ones_like(self.probabilities_pos)    # =1.0
        self.idx_pos = torch.zeros((self.num_batches, self.num_select), dtype=torch.int, device=self.device)
        self.idx_rot = torch.zeros((self.num_batches, self.num_select), dtype=torch.int, device=self.device)

        self.start_pts = torch.zeros((self.num_batches,), dtype=torch.int, device=self.device)
        self.end_pts = torch.zeros((self.num_batches,), dtype=torch.int, device=self.device) + 1
        self.end_update = torch.zeros((self.num_batches,), device=self.device).bool()
        self.start_update = torch.zeros((self.num_batches,), device=self.device).bool()
        self.iter_count = torch.zeros((self.num_batches,), dtype=torch.float, device=self.device)
        self.next_count = torch.zeros((self.num_batches,), dtype=torch.float, device=self.device)
        self.pose_count = torch.zeros((self.num_batches,), dtype=torch.long, device=self.device)

        self.cam_pose_saved = torch.zeros((self.num_batches, 7), dtype=torch.float, device=self.device)
        self.joints_saved = torch.zeros((self.num_batches, 6), dtype=torch.float, device=self.device)
        self.start_pts_saved = torch.zeros((self.num_batches,), dtype=torch.int, device=self.device)
        self.end_pts_saved = torch.zeros((self.num_batches,), dtype=torch.int, device=self.device)
        self.cam_pose_list = [[] for _ in range(self.num_batches)]
        self.joints_list = [[] for _ in range(self.num_batches)]
        self.start_pts_list = [[] for _ in range(self.num_batches)]
        self.end_pts_list = [[] for _ in range(self.num_batches)]
        self.cam_pose_select = torch.empty(0)
        self.joints_select = torch.empty(0)
        self.start_pts_select = torch.empty(0)
        self.end_pts_select = torch.empty(0)
        self.idx_select = torch.empty(0)

        self.stop = False
        self.finish = torch.zeros((self.num_batches,), device=self.device).bool()

        self.choose_idx = torch.tensor([[1, 3, 5, 7], [0, 2, 4, 6], [3, 1, 7, 5], [2, 0, 6, 4], 
                                        [5, 7, 1, 3], [4, 6, 0, 2], [7, 5, 3, 1], [6, 4, 2, 0]], 
                                        dtype=torch.long, device=self.device)
        self.choose_iden_idx = torch.tensor([[2, 4, 6, 0], [3, 5, 7, 1], [0, 6, 4, 2], [1, 7, 5, 3],
                                             [6, 0, 2, 4], [7, 1, 3, 5], [4, 2, 0, 6], [5, 3, 1, 7]],
                                             dtype=torch.long, device=self.device)
        self.pts_list = torch.arange(0, self.num_points+1, self.num_steps, dtype=torch.int, device=self.device)
        self.pts_list = torch.cat((self.pts_list[:1],
                                   torch.tensor([1], dtype=torch.int, device=self.device),
                                   self.pts_list[1:-1],
                                   torch.tensor([self.num_points - 1], dtype=torch.int, device=self.device),
                                   self.pts_list[-1:]), dim=0)
        self.start_pts_list_idx = torch.zeros((self.num_batches,), dtype=torch.long, device=self.device)
        self.end_pts_list_idx = torch.ones((self.num_batches,), dtype=torch.long, device=self.device)


    def defineVariables(self,):
        # 定义更新变量
        # noise: delta (x, y, z), (wx, wy, wz)
        self.noise_optimized = torch.zeros((self.num_batches, 6), dtype=torch.float, device=self.device) 
        self.noise_stored = torch.zeros((self.num_batches, self.num_randoms_all, 6), dtype=torch.float, device=self.device)

        self.cam_pos_optimized = torch.zeros((self.num_batches, 3), dtype=torch.float, device=self.device)
        self.cam_rot_optimized = torch.zeros((self.num_batches, 3, 3), dtype=torch.float, device=self.device)
        self.cam_quat_optimized = torch.zeros((self.num_batches, 4), dtype=torch.float, device=self.device)
        self.cam_pose_optimized = torch.zeros((self.num_batches, 7), dtype=torch.float, device=self.device)
        self.joints_optimized = torch.zeros((self.num_batches, 6), dtype=torch.float, device=self.device)
        self.cam_pose_stored = torch.zeros((self.num_batches, self.num_randoms_all, 7), dtype=torch.float, device=self.device)
        self.cam_pose_backup = torch.zeros((self.num_batches, 4, 7), dtype=torch.float, device=self.device)
        self.cam_pose_init = torch.zeros((self.num_batches, 7), dtype=torch.float, device=self.device)

        self.pos_costs = torch.zeros((self.num_batches, self.num_randoms_all), dtype=torch.float, device=self.device)
        self.rot_costs = torch.zeros((self.num_batches, self.num_randoms_all), dtype=torch.float, device=self.device)
        self.pos_optimized_cost = torch.zeros((self.num_batches,), dtype=torch.float, device=self.device)
        self.pos_optimized_gate = torch.zeros((self.num_batches,), dtype=torch.float, device=self.device)
        self.rot_optimized_cost = torch.zeros((self.num_batches,), dtype=torch.float, device=self.device)
        self.original_costs = False
        self.pos_costs_o = torch.zeros((self.num_batches, self.num_randoms_all), dtype=torch.float, device=self.device)
        self.rot_costs_o = torch.zeros((self.num_batches, self.num_randoms_all), dtype=torch.float, device=self.device)
        self.pos_optimized_cost_o = torch.zeros((self.num_batches,), dtype=torch.float, device=self.device)
        self.rot_optimized_cost_o = torch.zeros((self.num_batches,), dtype=torch.float, device=self.device)

        self.probabilities_pos = torch.zeros((self.num_batches, self.num_select), dtype=torch.float, device=self.device)
        self.probabilities_rot = torch.zeros((self.num_batches, self.num_select), dtype=torch.float, device=self.device)
        self.importance_weights = DEFAULT_NOISY_COST_IMPORTANCE_WEIGHT * torch.ones_like(self.probabilities_pos)    # =1.0
        self.idx_pos = torch.zeros((self.num_batches, self.num_select), dtype=torch.int, device=self.device)
        self.idx_rot = torch.zeros((self.num_batches, self.num_select), dtype=torch.int, device=self.device)

        self.start_pts = torch.zeros((self.num_batches,), dtype=torch.int, device=self.device)
        self.end_pts = torch.zeros((self.num_batches,), dtype=torch.int, device=self.device) + 1
        self.end_update = torch.zeros((self.num_batches,), device=self.device).bool()
        self.start_update = torch.zeros((self.num_batches,), device=self.device).bool()
        self.iter_count = torch.zeros((self.num_batches,), dtype=torch.float, device=self.device)
        self.next_count = torch.zeros((self.num_batches,), dtype=torch.float, device=self.device)
        self.pose_count = torch.zeros((self.num_batches,), dtype=torch.long, device=self.device)

        self.cam_pose_saved = torch.zeros((self.num_batches, 7), dtype=torch.float, device=self.device)
        self.joints_saved = torch.zeros((self.num_batches, 6), dtype=torch.float, device=self.device)
        self.start_pts_saved = torch.zeros((self.num_batches,), dtype=torch.int, device=self.device)
        self.end_pts_saved = torch.zeros((self.num_batches,), dtype=torch.int, device=self.device)
        self.cam_pose_list = [[] for _ in range(self.num_batches)]
        self.joints_list = [[] for _ in range(self.num_batches)]
        self.start_pts_list = [[] for _ in range(self.num_batches)]
        self.end_pts_list = [[] for _ in range(self.num_batches)]
        self.cam_pose_select = torch.empty(0)
        self.joints_select = torch.empty(0)
        self.start_pts_select = torch.empty(0)
        self.end_pts_select = torch.empty(0)
        self.idx_select = torch.empty(0)

        self.stop = False
        self.finish = torch.zeros((self.num_batches,), device=self.device).bool()

        self.choose_idx = torch.tensor([[1, 3, 5, 7], [0, 2, 4, 6], [3, 1, 7, 5], [2, 0, 6, 4], 
                                        [5, 7, 1, 3], [4, 6, 0, 2], [7, 5, 3, 1], [6, 4, 2, 0]], 
                                        dtype=torch.long, device=self.device)
        self.choose_iden_idx = torch.tensor([[2, 4, 6, 0], [3, 5, 7, 1], [0, 6, 4, 2], [1, 7, 5, 3],
                                             [6, 0, 2, 4], [7, 1, 3, 5], [4, 2, 0, 6], [5, 3, 1, 7]],
                                             dtype=torch.long, device=self.device)
        self.pts_list = torch.arange(0, self.num_points+1, self.num_steps, dtype=torch.int, device=self.device)
        self.pts_list = torch.cat((self.pts_list[:1],
                                   torch.tensor([1], dtype=torch.int, device=self.device),
                                   self.pts_list[1:-1],
                                   torch.tensor([self.num_points - 1], dtype=torch.int, device=self.device),
                                   self.pts_list[-1:]), dim=0)
        self.start_pts_list_idx = torch.zeros((self.num_batches,), dtype=torch.long, device=self.device)
        self.end_pts_list_idx = torch.ones((self.num_batches,), dtype=torch.long, device=self.device)


    def resetSeamData(self, seam_line: torch.Tensor, seam_tangent: torch.Tensor, seam_limits: torch.Tensor):
        self.seam_line = seam_line.to(self.device)            # (N, 3),   N为焊缝采样点的数量
        self.seam_lines = torch.empty(0)
        self.seam_tangent = seam_tangent.to(self.device)      # (N, 3)
        self.seam_tangents = torch.empty(0)
        self.seam_limits = seam_limits.to(self.device)        # (N, 2, 3)
        self.seam_limits_s = torch.empty(0)
        self.num_points = self.seam_line.shape[0]


    def solve(self,):
        """
        Args:
        """
        def _sync():
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize()

        self._n_cost_calls = 0                       # computeCosts 调用次数（每次=1 次 UR12e_t 重建 + IK/碰撞）
        _t_all = time.perf_counter()
        # 定义变量值
        self.defineVariables()
        # 初始化相机位姿
        self.initializePose()
        _sync(); _t_init = time.perf_counter()
        # print("init:", self.pos_optimized_cost + self.rot_optimized_cost)
        i = 1
        # iteration block
        while i <= self.max_iterations:
            self.generateNoise()
            self.computeNoisyPoseCosts()
            self.computeProbabilities()
            self.updateParameters()
            # print(f"{i}:{self.pos_optimized_cost + self.rot_optimized_cost}")
            self.resetVariables()
            if self.stop:
                break
            i += 1
        _sync(); _t_main = time.perf_counter()
        _main_iters = i
        _cost_calls_main = self._n_cost_calls
        self.selectOutputs()
        # print(self.cam_pose_select)
        # print(self.joints_select)
        if self.cam_pose_select is None:
            _sync()
            print(f"[计时][pose] 无解：init={_t_init - _t_all:.3f}s "
                  f"主循环={_t_main - _t_init:.3f}s/{_main_iters}轮 "
                  f"(num_batches={self.num_batches} num_randoms_all={self.num_randoms_all} "
                  f"eval={_cost_calls_main}次) 总={time.perf_counter() - _t_all:.3f}s")
            return None, None, None, None
        else:
            _sync(); _t_select = time.perf_counter()
            A0, B0 = self.cam_pose_select.shape[:2]
            _cost_calls_refine = self._n_cost_calls
            if self.refine:
                self.refineOutputs()
            _sync(); _t_refine = time.perf_counter()
            print(f"[计时][pose] init={_t_init - _t_all:.3f}s "
                  f"主循环={_t_main - _t_init:.3f}s/{_main_iters}轮(eval={_cost_calls_main}次) "
                  f"select={_t_select - _t_main:.3f}s "
                  f"refine={_t_refine - _t_select:.3f}s/{self.max_iterations_refine}轮"
                  f"(A0={A0} B={B0} K={self.cam_pose_select.shape[0]} eval={self._n_cost_calls - _cost_calls_refine}次) "
                  f"总={_t_refine - _t_all:.3f}s")
            try:
                import scene_pose2 as _sp
                if _sp._PROF["n"]:
                    _p = _sp._PROF
                    print(f"[计时][pose-getJoints] 累计{_p['n']}次 "
                          f"build={_p['build']:.3f}s ik={_p['ik']:.3f}s coll={_p['coll']:.3f}s | "
                          f"每次 build={_p['build']/_p['n']*1000:.1f}ms "
                          f"ik={_p['ik']/_p['n']*1000:.1f}ms "
                          f"coll={_p['coll']/_p['n']*1000:.1f}ms")
                    _p.update(build=0.0, ik=0.0, coll=0.0, n=0)
            except Exception:
                pass
            # Convert inverted-seam indices to original-seam indices.
            # initializePose builds 8 seam variants: even indices (0,2,4,6) use the original
            # seam, odd indices (1,3,5,7) use seam_line.flip(0). The start/end_pts stored
            # during optimization are indices into each batch's own seam variant.  Callers
            # always hold the original seam, so we map odd-batch indices back:
            #   [s_inv, e_inv)  →  [N - e_inv, N - s_inv)  in the original seam.
            N = self.num_points
            A_orig = len(self.idx_select)
            for m in range(A_orig):
                j = int(self.idx_select[m])
                if j % 2 == 1:  # inverted-seam batch
                    total_rows = self.start_pts_select.shape[0]
                    rows = list(range(m, total_rows, A_orig))
                    old_s = self.start_pts_select[rows, :].clone()
                    old_e = self.end_pts_select[rows, :].clone()
                    self.start_pts_select = self.start_pts_select.clone()
                    self.start_pts_select[rows, :] = N - old_e
                    self.end_pts_select = self.end_pts_select.clone()
                    self.end_pts_select[rows, :] = N - old_s
            return self.cam_pose_select, self.joints_select, self.start_pts_select, self.end_pts_select


    def selectOutputs(self,):
        """
        筛选出观测次数少的相机位姿组合
        """
        if not self.finish.any():
            self.cam_pose_select = None
            self.joints_select = None
            return

        length = 1e3
        for i, cam_poses in enumerate(self.cam_pose_list):
            if self.finish[i]:
                cam_pose = torch.stack(cam_poses, dim=0)
                if length > cam_pose.shape[0]:
                    length = cam_pose.shape[0]
        
        idx = torch.arange(self.num_batches, device=self.device)
        cam_pose_select = []
        joint_select = []
        start_pts_select = []
        end_pts_select = []
        idx_select = []
        for i, cam_poses, joints, start_pts, end_pts in zip(idx, self.cam_pose_list, self.joints_list, 
                                                            self.start_pts_list, self.end_pts_list):
            if self.finish[i]:
                cam_pose = torch.stack(cam_poses, dim=0)
                if cam_pose.shape[0] == length:
                    cam_pose_select.append(cam_pose)

                    joint = torch.stack(joints, dim=0)
                    joint_select.append(joint)

                    start_pt = torch.tensor(start_pts, dtype=torch.long, device=self.device)
                    start_pts_select.append(start_pt)

                    end_pt = torch.tensor(end_pts, dtype=torch.long, device=self.device)
                    end_pts_select.append(end_pt)

                    idx_select.append(i)

        self.cam_pose_select = torch.stack(cam_pose_select, dim=0)
        self.joints_select = torch.stack(joint_select, dim=0)
        self.start_pts_select = torch.stack(start_pts_select, dim=0)
        # print(self.start_pts_select)
        self.end_pts_select = torch.stack(end_pts_select, dim=0)
        # print(self.end_pts_select)
        self.idx_select = torch.tensor(idx_select, dtype=torch.long, device=self.device)
        # print(self.idx_select)


    def refineOutputs(self,):
        """
        优化每次观测的位姿到最优观测位姿
        """
        A, B = self.cam_pose_select.shape[:2]
        # reset variables
        self.num_batches = A * B
        self.original_costs = True

        self.noise_optimized = torch.zeros((self.num_batches, 6), dtype=torch.float, device=self.device) 
        self.noise_stored = torch.zeros((self.num_batches, self.num_randoms_all, 6), dtype=torch.float, device=self.device)

        self.pos_costs = torch.zeros((self.num_batches, self.num_randoms_all), dtype=torch.float, device=self.device)
        self.rot_costs = torch.zeros((self.num_batches, self.num_randoms_all), dtype=torch.float, device=self.device)
        self.pos_costs_o = torch.zeros((self.num_batches, self.num_randoms_all), dtype=torch.float, device=self.device)
        self.rot_costs_o = torch.zeros((self.num_batches, self.num_randoms_all), dtype=torch.float, device=self.device)

        self.cam_pose_optimized = self.cam_pose_select.clone().reshape(A * B, 7)
        self.cam_pos_optimized = self.cam_pose_optimized[:, :3].clone()
        self.cam_quat_optimized = self.cam_pose_optimized[:, 3:].clone()
        self.cam_rot_optimized = math.matrix_from_quat(self.cam_quat_optimized.clone())
        self.cam_pose_stored = torch.zeros((self.num_batches, self.num_randoms_all, 7), dtype=torch.float, device=self.device)
        self.joints_optimized = self.joints_select.clone().reshape(A * B, 6)

        self.importance_weights = DEFAULT_NOISY_COST_IMPORTANCE_WEIGHT * torch.ones((self.num_batches, self.num_select), 
                                                                                    dtype=torch.float, device=self.device)
        self.start_pts = self.start_pts_select.clone().reshape(A * B)
        self.end_pts = self.end_pts_select.clone().reshape(A * B)

        # generate seam
        N = self.seam_line.shape[0]
        self.seam_lines = self.seam_lines[self.idx_select].clone().unsqueeze(1).expand(-1, B, -1, -1).reshape(A * B, N, 3)
        self.seam_tangents = self.seam_tangents[self.idx_select].clone().unsqueeze(1).expand(-1, B, -1, -1).reshape(A * B, N, 3)
        self.seam_limits_s = self.seam_limits_s[self.idx_select].clone().unsqueeze(1).expand(-1, B, -1, -1, -1).reshape(A * B, N, 2, 3)

        # compute cost of initial poses
        pos_cost, rot_cost, joints, pos_cost_o, rot_cost_o = self.computeCosts(self.cam_pose_optimized.unsqueeze(-2))
        self.pos_optimized_cost = pos_cost.clone().squeeze(-1)
        self.pos_optimized_gate = self._pos_costs_gate.clone().squeeze(-1)
        self.rot_optimized_cost = rot_cost.clone().squeeze(-1)
        self.pos_optimized_cost_o = pos_cost_o.clone().squeeze(-1)
        self.rot_optimized_cost_o = rot_cost_o.clone().squeeze(-1)
        self.joints_optimized = joints[:, 0]
        # print("init:", self.pos_optimized_cost + self.rot_optimized_cost)
        # print("init_o:", self.pos_optimized_cost_o + self.rot_optimized_cost_o)


        # refine
        i = 1
        cam_pose_list = []
        joints_list = []
        span = self.cfg.span
        # iteration block
        while i <= self.max_iterations_refine:
            self.generateNoise()
            self.computeNoisyPoseCosts()
            self.computeProbabilities()
            self.updateParameters()
            # print(f"{i}:{self.pos_optimized_cost + self.rot_optimized_cost}")
            # print(f"{i}:{self.pos_optimized_cost_o + self.rot_optimized_cost_o}")
            if i % span == 0:
                cam_pose_list.append(self.cam_pose_optimized.clone().reshape(A, B, 7))
                joints_list.append(self.joints_optimized.clone().reshape(A, B, 6))
            i += 1

        # self.cam_pose_select = self.cam_pose_optimized.clone().reshape(A, B, 7)
        # self.joints_select = self.joints_optimized.clone().reshape(A, B, 6)
        num_snapshots = len(cam_pose_list)
        start_pts_select_orig = self.start_pts_select.clone()   # (A, B)
        end_pts_select_orig = self.end_pts_select.clone()       # (A, B)
        cam_pose_list.reverse()
        joints_list.reverse()
        self.cam_pose_select = torch.cat(cam_pose_list, dim=0)
        self.joints_select = torch.cat(joints_list, dim=0)
        # expand segment info to match new (num_snapshots*A, B) shape
        self.start_pts_select = start_pts_select_orig.unsqueeze(0).expand(num_snapshots, -1, -1).reshape(num_snapshots * A, B)
        self.end_pts_select = end_pts_select_orig.unsqueeze(0).expand(num_snapshots, -1, -1).reshape(num_snapshots * A, B)

        # # 去重：仅在每个 batch 自己的 num_snapshots 个快照内部去重
        # num_snapshots = self.cam_pose_select.shape[0] // A
        # keep_list = []
        # for a in range(A):
        #     idx = torch.arange(a, num_snapshots * A, A, device=self.device)   # (num_snapshots,)
        #     flat = self.cam_pose_select[idx].reshape(num_snapshots, -1)        # (num_snapshots, B*7)
        #     dist = torch.cdist(flat, flat)                                      # (num_snapshots, num_snapshots)
        #     dup = torch.tril(dist < 1e-4, diagonal=-1).any(dim=1)              # True 表示与前面某快照重复
        #     keep_list.append(idx[~dup])
        # keep = torch.cat(keep_list).sort().values
        # self.cam_pose_select = self.cam_pose_select[keep]
        # self.joints_select = self.joints_select[keep]
        # self.start_pts_select = self.start_pts_select[keep]
        # self.end_pts_select = self.end_pts_select[keep]


    def resetVariables(self,):
        """
        更新目标焊缝采样点
        """
        # update start and end updates, as well as cam poses
        # 是否选定焊缝点完成观测——接受门槛用 gate 代价(朝向放宽为偏好，大角度也接受；relax=0 时等价旧版)
        self.end_update = ((self.pos_optimized_gate + self.rot_optimized_cost) <= 0)
        # 存储最新的成功观测位姿
        self.cam_pose_saved[self.end_update] = self.cam_pose_optimized[self.end_update].clone()
        self.joints_saved[self.end_update] = self.joints_optimized[self.end_update].clone()

        self.iter_count += 1
        # 是否未能找到观测位姿，重置焊缝点
        self.start_update = ((self.iter_count > self.num_iterations) + (self.next_count > self.num_next-1)) * (~self.end_update)

        # reset start and end indices
        if self.forced:
            # 是否看完第一段和倒数第二段焊缝
            forced = self.end_update * ((self.end_pts == 1) + 
                                        (self.end_pts == (self.num_points - 1)))    # old version #
            # forced = self.end_update * ((self.end_pts == self.num_steps) + 
            #                             (self.end_pts == (self.num_points - self.num_steps)))    # old version #
            self.end_update = self.end_update * (~forced)   # 从end_update中剔除forced
            # 存储并更新start点和end点
            self.start_pts_saved[forced] = self.start_pts[forced].clone()
            self.end_pts_saved[forced] = self.end_pts[forced].clone()
            self.start_pts_list_idx[forced] = self.end_pts_list_idx[forced].clone()
            self.start_pts[forced] = self.pts_list[self.start_pts_list_idx[forced]].clone()
            self.end_pts_list_idx[forced] += 1
            self.end_pts[forced] = self.pts_list[self.end_pts_list_idx[forced]].clone()
            # self.start_pts[forced] = self.end_pts[forced].clone()                     # old version #
            # self.end_pts[forced] = self.start_pts[forced].clone() + self.num_steps    # old version #
        else:
            forced = torch.zeros((self.num_batches,), device=self.device).bool()

        # 是否完成所有焊缝
        mask_finish = self.end_update * (self.end_pts >= self.num_points)
        # mask_finish = self.end_update * (self.end_pts >= (self.num_points-1))     # old version #
        # 是否第一次观测完所有焊缝
        first_finish = mask_finish * (~self.finish)
        # 当第一次观测完所有焊缝时，存储start点和end点
        self.start_pts_saved[first_finish] = self.start_pts[first_finish].clone()
        self.end_pts_saved[first_finish] = self.end_pts[first_finish].clone()
        # 记录完成观测
        self.finish[mask_finish] = True
        # 更新新的end点
        self.end_pts_list_idx[self.end_update] += 1 * (1 - self.finish[self.end_update].int())
        self.end_pts[self.end_update] = self.pts_list[self.end_pts_list_idx[self.end_update]].clone()
        # self.end_pts[self.end_update] = self.end_pts[self.end_update] + \
        #                                     self.num_steps * (1 - self.finish[self.end_update].int())   # old version #

        # 是否重置start点
        mask_next = self.start_update * ((self.end_pts_list_idx - self.start_pts_list_idx) 
                                           > 1)
        mask_repeat = self.start_update * ((self.end_pts_list_idx - self.start_pts_list_idx) 
                                           == 1)        
        # mask_next = self.start_update * ((self.end_pts - self.start_pts) 
        #                                    > self.num_steps)                      # old version #
        # mask_repeat = self.start_update * ((self.end_pts - self.start_pts) 
        #                                    == self.num_steps)                     # old version #
        # 重置start点和end点
        # 重置前存储start点和end点
        self.start_pts_saved[mask_next] = self.start_pts[mask_next].clone()
        self.start_pts_list_idx[mask_next] = self.end_pts_list_idx[mask_next].clone() - 1
        self.start_pts[mask_next] = self.pts_list[self.start_pts_list_idx[mask_next]].clone()
        # self.start_pts[mask_next] = self.end_pts[mask_next].clone() - self.num_steps      # old version #
        self.end_pts_saved[mask_next] = self.start_pts[mask_next].clone()
        self.end_pts_list_idx[mask_next] = self.start_pts_list_idx[mask_next] + 1
        self.end_pts[mask_next] = self.pts_list[self.end_pts_list_idx[mask_next]].clone()
        # self.end_pts[mask_next] = self.start_pts[mask_next].clone() + self.num_steps      # old version #

        # save data
        idx = torch.arange(self.num_batches, dtype=torch.int, device=self.device)
        idx = idx[mask_next + first_finish + forced]
        for i in idx:
            # print(f"{i}: {self.cam_pose_saved[i]}")
            self.cam_pose_list[i].append(self.cam_pose_saved[i].clone())
            self.joints_list[i].append(self.joints_saved[i].clone())
            self.start_pts_list[i].append(self.start_pts_saved[i].clone())
            self.end_pts_list[i].append(self.end_pts_saved[i].clone())

        # reset
        self.iter_count[self.start_update + self.end_update + forced] = 0
        self.next_count[self.start_update + self.end_update + forced] = 0
        self.pose_count[self.end_update + forced + mask_next] = 0
        if mask_repeat.int().sum() > 0:
            self.retrievePoses(mask_repeat)
        if (self.end_update + self.start_update + forced).int().sum() > 0:
            pos_cost, rot_cost, joints, _, _ = self.computeCosts(self.cam_pose_optimized.unsqueeze(-2))
            self.pos_optimized_cost[self.end_update + self.start_update + forced] = \
                pos_cost[self.end_update + self.start_update + forced].squeeze(-1)
            self.pos_optimized_gate[self.end_update + self.start_update + forced] = \
                self._pos_costs_gate[self.end_update + self.start_update + forced].squeeze(-1)
            self.rot_optimized_cost[self.end_update + self.start_update + forced] = \
                rot_cost[self.end_update + self.start_update + forced].squeeze(-1)
            self.joints_optimized[self.end_update + self.start_update + forced] = \
                joints[self.end_update + self.start_update + forced, 0]

        # update stop sign
        num_finish = self.finish.int().sum()
        # print("finish:", self.finish)
        # print("start_pts:", self.start_pts)
        # print("end_pts:", self.end_pts)
        if num_finish >= (self.num_poses):
            self.stop = True


    def initializePose(self,):
        """
        生成初始相机位姿(旋转矩阵和四元数)和焊缝采样点
        """
        # print("seam_line:")
        # print(self.seam_line)
        # print("seam_tangent:")
        # print(self.seam_tangent)
        # print("seam_limits:")
        # print(self.seam_limits)
        # raise KeyError
        z_dir_s = -(self.seam_limits[0, 0] + self.seam_limits[0, -1]) / 2        # (3,)
        z_dir_s = z_dir_s / torch.clamp(torch.norm(z_dir_s, dim=-1, keepdim=True), min=1e-8)
        z_dir_e = -(self.seam_limits[-1, 0] + self.seam_limits[-1, -1]) / 2
        z_dir_e = z_dir_e / torch.clamp(torch.norm(z_dir_e, dim=-1, keepdim=True), min=1e-8)
        y_dir_s =  self.seam_tangent[0].clone()       # (3,)
        y_dir_e =  self.seam_tangent[-1].clone()
        pos_s = self.seam_line[0]       # (3,)
        pos_e = self.seam_line[-1]
        seam_line_inv = self.seam_line.clone().flip(0)
        seam_tangent_inv = self.seam_tangent.clone().flip(0)
        seam_limits_inv = self.seam_limits.clone().flip(0)

        z_dir_lists = torch.stack((z_dir_s,z_dir_e), dim=0)     # (2, 3)
        y_dir_lists = torch.stack((y_dir_s, y_dir_e), dim=0)     # (2, 3)
        x_dir_lists = torch.cross(y_dir_lists, z_dir_lists, dim=-1)     # (2, 3)
        pos_lists = torch.stack((pos_s, pos_e), dim=0)      # (2, 3)
        rot0 = opt_utils.matrix_from_vectors(x_dir_lists, y_dir_lists, z_dir_lists)         # (2, 3, 3)
        seam_lines = torch.stack((self.seam_line.clone(), seam_line_inv), dim=0)            # (2, N, 3)
        seam_tangents = torch.stack((self.seam_tangent.clone(), seam_tangent_inv), dim=0)   # (2, N, 3)
        seam_limits_s = torch.stack((self.seam_limits.clone(), seam_limits_inv), dim=0)     # (2, N, 2, 3)

        ones = torch.ones((rot0.shape[0]), dtype=torch.float, device=self.device)
        rot1 = rot0 @ math.matrix_from_euler("X", ones * 45 / 180 * torch.pi)       # (2, 3, 3)
        rot2 = rot0 @ math.matrix_from_euler("X", ones * -45 / 180 * torch.pi)      # (2, 3, 3)
        rot_a = torch.cat((rot1, rot2), dim=0)              # (4, 3, 3)
        pos_lists = pos_lists.repeat(2, 1)                  # (4, 3)
        seam_lines = seam_lines.repeat(2, 1, 1)             # (4, N, 3)
        seam_tangents = seam_tangents.repeat(2, 1, 1)       # (4, N, 3)
        seam_limits_s = seam_limits_s.repeat(2, 1, 1, 1)    # (4, N, 2, 3)

        ones = torch.ones((rot_a.shape[0]), dtype=torch.float, device=self.device)
        rot_o = rot_a @ math.matrix_from_euler("Z", ones * 180 / 180 * torch.pi)    # (4, 3, 3)
        rot_init = torch.cat((rot_a, rot_o), dim=0)         # (8, 3, 3)
        pos_lists = pos_lists.repeat(2, 1)                  # (8, 3)
        seam_lines = seam_lines.repeat(2, 1, 1)             # (8, N, 3)
        seam_tangents = seam_tangents.repeat(2, 1, 1)       # (8, N, 3)
        seam_limits_s = seam_limits_s.repeat(2, 1, 1, 1)    # (8, N, 2, 3)
        quat_init = math.quat_from_matrix(rot_init)         # (8, 4)
        dists = torch.tensor([0, 0, -0.5], dtype=torch.float, device=self.device).unsqueeze(0).repeat(quat_init.shape[0], 1)     # (8, 3)
        pos_init = pos_lists + math.quat_apply(quat_init, dists)

        length_multiple = int(self.num_batches // pos_init.shape[0])
        self.cam_pos_optimized = pos_init.repeat(length_multiple, 1)
        self.cam_rot_optimized = rot_init.repeat(length_multiple, 1, 1)
        self.cam_quat_optimized = quat_init.repeat(length_multiple, 1)
        self.cam_pose_optimized = torch.cat((self.cam_pos_optimized, self.cam_quat_optimized), dim=-1)      # (B, 7)
        self.seam_lines = seam_lines.repeat(length_multiple, 1, 1)
        self.seam_tangents = seam_tangents.repeat(length_multiple, 1, 1)
        self.seam_limits_s = seam_limits_s.repeat(length_multiple, 1, 1, 1)

        # compute cost of initial poses
        pos_cost, rot_cost, joints, _, _ = self.computeCosts(self.cam_pose_optimized.unsqueeze(-2))
        self.pos_optimized_cost = pos_cost.clone().squeeze(-1)
        self.pos_optimized_gate = self._pos_costs_gate.clone().squeeze(-1)
        self.rot_optimized_cost = rot_cost.clone().squeeze(-1)
        self.joints_optimized = joints[:, 0]

        self.cam_pose_init = self.cam_pose_optimized.clone()


    def generateNoise(self,):
        def getSamples(covariance, num_randoms):
            mean = torch.zeros((covariance.shape[0],), device=self.device)
            cov = covariance
            distribution = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)
            samples = distribution.sample((self.num_batches * num_randoms,))
            samples = samples.reshape(self.num_batches, num_randoms, covariance.shape[0])
            return samples
        
        if self.original_costs:
            # 上一次迭代的最优rollout
            self.noise_stored[:,-1] = 0         # (B, R, 6)
            self.cam_pose_stored[:, -1] = self.cam_pose_optimized
            self.pos_costs[:, -1] = self.pos_optimized_cost
            self.rot_costs[:, -1] = self.rot_optimized_cost
            self.pos_costs_o[:, -1] = self.pos_optimized_cost_o
            self.rot_costs_o[:, -1] = self.rot_optimized_cost_o

            # 生成新的noisy rollouts
            samples = getSamples(self.covariance, self.num_randoms_new)    # (B, Rn, 6)
            scale = 0.2
            samples = samples * scale
            if self.num_randoms_old > 0:
                self.noise_stored[:, -1 - self.num_randoms_old] = self.noise_optimized.clone()
                if self.num_randoms_old > 1:
                    _, idx_pos_top = torch.topk(self.pos_costs_o, self.num_randoms_old-1, dim=-1, largest=False)       # (B, Ro-1)
                    idx_pos_top = idx_pos_top.unsqueeze(-1).expand(-1, -1, 3)
                    noise_pos_top = torch.gather(input=self.noise_stored[..., :3], dim=1, index=idx_pos_top)
                    _, idx_rot_top = torch.topk(self.rot_costs_o, self.num_randoms_old-1, dim=-1, largest=False)       # (B, Ro-1)
                    idx_rot_top = idx_rot_top.unsqueeze(-1).expand(-1, -1, 3)
                    noise_rot_top = torch.gather(input=self.noise_stored[..., 3:], dim=1, index=idx_rot_top)
                    self.noise_stored[:, -self.num_randoms_old:-1] = torch.cat((noise_pos_top, noise_rot_top), dim=-1)
        
        else:
            # 上一次迭代的最优rollout
            self.noise_stored[:,-1] = 0         # (B, R, 6)
            self.cam_pose_stored[:, -1] = self.cam_pose_optimized
            # self.pose_costs[:, -1] = self.pose_optimized_cost
            self.pos_costs[:, -1] = self.pos_optimized_cost
            self.rot_costs[:, -1] = self.rot_optimized_cost

            # 生成新的noisy rollouts
            samples = getSamples(self.covariance, self.num_randoms_new)    # (B, Rn, 6)
            scale = 0.1 * torch.clamp(self.pos_optimized_cost + self.rot_optimized_cost, min=2, max=60).unsqueeze(-1).unsqueeze(-1)
            samples = samples * scale
            if self.num_randoms_old > 0:
                self.noise_stored[:, -1 - self.num_randoms_old] = self.noise_optimized.clone()
                if self.num_randoms_old > 1:
                    _, idx_pos_top = torch.topk(self.pos_costs, self.num_randoms_old-1, dim=-1, largest=False)       # (B, Ro-1)
                    idx_pos_top = idx_pos_top.unsqueeze(-1).expand(-1, -1, 3)
                    noise_pos_top = torch.gather(input=self.noise_stored[..., :3], dim=1, index=idx_pos_top)
                    _, idx_rot_top = torch.topk(self.rot_costs, self.num_randoms_old-1, dim=-1, largest=False)       # (B, Ro-1)
                    idx_rot_top = idx_rot_top.unsqueeze(-1).expand(-1, -1, 3)
                    noise_rot_top = torch.gather(input=self.noise_stored[..., 3:], dim=1, index=idx_rot_top)
                    self.noise_stored[:, -self.num_randoms_old:-1] = torch.cat((noise_pos_top, noise_rot_top), dim=-1)
        
        self.noise_stored[:, :self.num_randoms_new] = samples
        pos_new, rot_new, quat_new = self.computePoses(self.cam_pos_optimized, self.cam_rot_optimized, 
                                                       self.noise_stored[:, :-1].clone())
        self.cam_pose_stored[:, :-1] = torch.cat((pos_new, quat_new), dim=-1)


    def computePoses(self, pos: torch.Tensor, rot: torch.Tensor, delta: torch.Tensor):
        """
        计算加高斯噪音后的相机位姿
        Args:
            pos: (B, 3)
            rot: (B, 3, 3)
            delta: (B, R, 6)
        Returns:
            pos_new: (B, R, 3)
            rot_new: (B, R, 3, 3)
            quat_new: (B, R, 3)
        """
        B, R = delta.shape[:2]
        pos = pos.unsqueeze(-2).repeat(1, R, 1)
        rot = rot.unsqueeze(-3).repeat(1, R, 1, 1)
        quat = math.quat_from_matrix(rot)

        delta_pos = delta[..., :3].clone()
        delta_rot = math.matrix_from_so3(delta[..., 3:].clone())     # (B, R, 3, 3)
        
        rot_new = rot @ delta_rot       # (B, R, 3, 3)
        quat_new = math.quat_from_matrix(rot_new)       # (B, R, 4)
        pos_new = math.quat_apply(quat, delta_pos) + pos         # (B, R, 3)
        
        return pos_new, rot_new, quat_new
    

    def computeNoisyPoseCosts(self):
        """
        计算个相机位姿对应的成本函数
        """
        # compute costs
        if self.original_costs:
            self.pos_costs[:, :-1], self.rot_costs[:, :-1], _, self.pos_costs_o[:, :-1], self.rot_costs_o[:, :-1] = \
                self.computeCosts(self.cam_pose_stored[:, :-1].clone())
        else:
            self.pos_costs[:, :-1], self.rot_costs[:, :-1], _, _, _ = self.computeCosts(self.cam_pose_stored[:, :-1].clone())
    

    def computeCosts(self, cam_poses):
        """
        计算个相机位姿对应的成本函数
        Args:
            cam_poses: (B, N, 7)
        """
        self._n_cost_calls = getattr(self, "_n_cost_calls", 0) + 1   # 计时用：统计 IK/碰撞评估次数
        # compute collision costs
        collision_costs, joints = self.scene.computeCollisionCost(cam_poses, self.joints_optimized)     # (B, R')
        # collision_costs = torch.zeros((cam_poses.shape[:2]), dtype=torch.float, device=self.device)
        # joints = torch.zeros((cam_poses.shape[:2] + (6,)), dtype=torch.float, device=self.device)

        # compute vision costs
        # vision_costs = self.scene.computeVisionCost_0(*self.flattenInputs_0(start_idx=self.start_pts,
        #                                                    end_idx=self.end_pts,
        #                                                    seam_lines=self.seam_lines,
        #                                                    seam_tangent=self.seam_tangents,
        #                                                    seam_limits=self.seam_limits_s,
        #                                                    seam_vertical=self.seam_vertical_s,
        #                                                    seam_theta=self.seam_theta_s,
        #                                                    seam_rays_2d=self.seam_rays_2d_s,
        #                                                    cam_poses=cam_poses))     # (B, R')
        _vc = self.scene.computeVisionCost_1(
            *self.flattenInputs_1(start_idx=self.start_pts,
                                  end_idx=self.end_pts,
                                  seam_lines=self.seam_lines,
                                  seam_tangent=self.seam_tangents,
                                  seam_limits=self.seam_limits_s,
                                  cam_poses=cam_poses), original_costs=self.original_costs)     # (B, R')
        vision_pos_costs, vision_rot_costs, vision_pos_costs_o, vision_rot_costs_o = _vc[:4]
        # scene_pose2 会多返回第5项(软上限 gate 的 vision_pos)；scene_pose(Isaac)只返回4项→gate 回退=全量，
        # 故原 Isaac 管线行为不变(gate==全量代价)，无需改 scene_pose.py。
        vision_pos_costs_gate = _vc[4] if len(_vc) > 4 else vision_pos_costs

        pos_costs = (self.collision_cost_weight * collision_costs + self.vision_pos_cost_weight * vision_pos_costs)
        rot_costs = (self.collision_cost_weight * collision_costs + self.vision_rot_cost_weight * vision_rot_costs)
        # gate 代价(侧信道，不改返回元数)：接受门槛(end_update)与 refine 可行性因子用它，其余(采样/排序)仍用全量
        self._pos_costs_gate = (self.collision_cost_weight * collision_costs
                                + self.vision_pos_cost_weight * vision_pos_costs_gate)
        if self.original_costs:
            pos_costs_o = (self.collision_cost_weight * collision_costs + self.vision_pos_cost_weight * vision_pos_costs_o)
            rot_costs_o = (self.collision_cost_weight * collision_costs + self.vision_rot_cost_weight * vision_rot_costs_o)
        else:
            pos_costs_o = torch.empty(0)
            rot_costs_o = torch.empty(0)
        return pos_costs, rot_costs, joints, pos_costs_o, rot_costs_o       # (B, R), (B, R, 6)


    def computeProbabilities(self):
        """
        计算在每个高斯噪声下新的pose的概率
        """
        h = self.exponentiated_cost_sensitivity
        
        # Select top k min costs
        if self.original_costs:
            mask = (self.pos_costs + self.rot_costs) > 0
            pos_costs = self.pos_costs_o.clone()
            rot_costs = self.rot_costs_o.clone()
            pos_costs[mask] += 1e3
            rot_costs[mask] += 1e3
            pos_costs, self.idx_pos = torch.topk(input=pos_costs, k = self.num_select, dim=1, largest=False)
            rot_costs, self.idx_rot = torch.topk(input=rot_costs, k = self.num_select, dim=1, largest=False)
        else:
            pos_costs, self.idx_pos = torch.topk(input=self.pos_costs, k = self.num_select, dim=1, largest=False)
            rot_costs, self.idx_rot = torch.topk(input=self.rot_costs, k = self.num_select, dim=1, largest=False)

        # Find min and max cost over all rollouts at each timestep
        min_pos_costs_per_t = torch.min(input=pos_costs, dim=-1)[0].unsqueeze(-1)               # (B, 1)
        max_pos_costs_per_t = torch.max(input=pos_costs, dim=-1)[0].unsqueeze(-1)               # (B, 1)
        min_rot_costs_per_t = torch.min(input=rot_costs, dim=-1)[0].unsqueeze(-1)               # (B, 1)
        max_rot_costs_per_t = torch.max(input=rot_costs, dim=-1)[0].unsqueeze(-1)               # (B, 1)

        denom_pos_per_t = max_pos_costs_per_t - min_pos_costs_per_t                             # (B, 1)
        denom_pos_per_t = torch.clamp(denom_pos_per_t, min=MIN_COST_DIFFERENCE)                 # prevent division by zero
        denom_rot_per_t = max_rot_costs_per_t - min_rot_costs_per_t                             # (B, 1)
        denom_rot_per_t = torch.clamp(denom_rot_per_t, min=MIN_COST_DIFFERENCE)                 # prevent division by zero
        
        # Compute probs for all rollouts and timesteps at once for dimension d
        exponents_pos = -h * (pos_costs - min_pos_costs_per_t) / denom_pos_per_t                        # (B, S)
        probs_pos_unnormalized = self.importance_weights * torch.exp(exponents_pos)                     # (B, S)
        probs_pos_sum = torch.clamp(probs_pos_unnormalized.sum(dim=-1, keepdim=True), min=1e-12)        # (B, 1)
        self.probabilities_pos = probs_pos_unnormalized / probs_pos_sum                                 # (B, S)
        exponents_rot = -h * (rot_costs - min_rot_costs_per_t) / denom_rot_per_t                        # (B, S)
        probs_rot_unnormalized = self.importance_weights * torch.exp(exponents_rot)                     # (B, S)
        probs_rot_sum = torch.clamp(probs_rot_unnormalized.sum(dim=-1, keepdim=True), min=1e-12)        # (B, 1)
        self.probabilities_rot = probs_rot_unnormalized / probs_rot_sum                                 # (B, S)

    
    def updateParameters(self):
        """
        更新最优轨迹
        """
        noise = self.noise_stored.clone()                          # (B, R, 6)
        
        # computing updates from probabilities using convex combination
        noise_pos = torch.gather(input=noise[..., :3], dim=1, index=self.idx_pos.unsqueeze(-1).expand(-1, -1, 3))
        noise_rot = torch.gather(input=noise[..., 3:], dim=1, index=self.idx_rot.unsqueeze(-1).expand(-1, -1, 3))
        noise_pos_optimized = (noise_pos * self.probabilities_pos.unsqueeze(-1)).sum(dim=-2)             # (B, 3)
        noise_rot_optimized = (noise_rot * self.probabilities_rot.unsqueeze(-1)).sum(dim=-2)             # (B, 3)
        noise_optimized = torch.cat((noise_pos_optimized, noise_rot_optimized), dim=-1)
        
        # get update indices
        pos_new, _, quat_new = self.computePoses(self.cam_pos_optimized, self.cam_rot_optimized, noise_optimized.unsqueeze(-2))
        cam_pose_optimized = torch.cat((pos_new, quat_new), dim=-1)     # (B, 1, R)        
        update_idx = self.computeOptimizedCost(cam_pose_optimized)
        self.cam_pose_optimized[update_idx] = cam_pose_optimized[update_idx, 0].clone()
        self.cam_pos_optimized[update_idx] = self.cam_pose_optimized[update_idx, :3].clone()
        self.cam_quat_optimized[update_idx] = self.cam_pose_optimized[update_idx, 3:].clone()
        self.cam_rot_optimized[update_idx] = math.matrix_from_quat(self.cam_quat_optimized[update_idx].clone())
        self.noise_optimized[update_idx] = noise_optimized[update_idx]
        
        if not self.original_costs:
            self.next_count[~update_idx] += 1
        

    def computeOptimizedCost(self, cam_pose_optimized):
        """
        计算更新后相机位姿的成本
        """
        if self.original_costs:
            pos_optimized_cost, rot_optimized_cost, joints, pos_optimized_cost_o, rot_optimized_cost_o = \
                self.computeCosts(cam_pose_optimized)
            pos_optimized_cost = pos_optimized_cost.squeeze(-1)
            rot_optimized_cost = rot_optimized_cost.squeeze(-1)
            pos_optimized_cost_o = pos_optimized_cost_o.squeeze(-1)
            rot_optimized_cost_o = rot_optimized_cost_o.squeeze(-1)
            pos_optimized_gate = self._pos_costs_gate.squeeze(-1)      # 软上限 gate 版 pos 代价

            # Update parameters, meaning optimized rollout
            # 可行性因子改用 gate(接受大角度)；排序仍按全量 _o 代价(朝 45° 偏好保留)
            update_idx = \
                ((pos_optimized_cost_o + rot_optimized_cost_o) < (self.pos_optimized_cost_o + self.rot_optimized_cost_o)) * \
                    ((pos_optimized_gate + rot_optimized_cost) <= 0)

            # print(update_idx)
            self.pos_optimized_cost_o[update_idx] = pos_optimized_cost_o[update_idx]                    # (B,)
            self.rot_optimized_cost_o[update_idx] = rot_optimized_cost_o[update_idx]                    # (B,)
            self.pos_optimized_cost[update_idx] = pos_optimized_cost[update_idx]                        # (B,)
            self.pos_optimized_gate[update_idx] = pos_optimized_gate[update_idx]                        # (B,)
            self.rot_optimized_cost[update_idx] = rot_optimized_cost[update_idx]                        # (B,)
            self.joints_optimized[update_idx] = joints[update_idx, 0]
        else:
            pos_optimized_cost, rot_optimized_cost, joints, _, _ = self.computeCosts(cam_pose_optimized)
            pos_optimized_cost = pos_optimized_cost.squeeze(-1)
            rot_optimized_cost = rot_optimized_cost.squeeze(-1)
            pos_optimized_gate = self._pos_costs_gate.squeeze(-1)

            # print("a:", pos_optimized_cost + rot_optimized_cost)

            # Update parameters, meaning optimized rollout（纯排序，用全量代价→朝 45° 偏好保留）
            update_idx = ((pos_optimized_cost + rot_optimized_cost) < (self.pos_optimized_cost + self.rot_optimized_cost))

            # print(update_idx)
            self.pos_optimized_cost[update_idx] = pos_optimized_cost[update_idx]                        # (B,)
            self.pos_optimized_gate[update_idx] = pos_optimized_gate[update_idx]                        # (B,)
            self.rot_optimized_cost[update_idx] = rot_optimized_cost[update_idx]                        # (B,)
            self.joints_optimized[update_idx] = joints[update_idx, 0]

        return update_idx
    

    #################################################################################
    ################################## Utils ########################################
    #################################################################################
    def retrievePoses(self, mask: torch.Tensor):
        """
        选取在对称方向已确定的相机位姿
        Args:
            mask: (B,), bool
        """
        B = mask.shape[0]
        L = self.seam_line.shape[0]
        mask_repeat = mask.clone()
        mask_reset = mask.clone() * (self.end_pts != 1)
           
        idx = torch.arange(mask.shape[0], device=self.device)
        mask_start_idx = idx[mask * (self.end_pts == 1)]     # (M1,)
        mask_reset_idx = idx[mask * (self.end_pts != 1)]     # (M2,)

        cam_pose_new = torch.zeros((B, 7), dtype=torch.float, device=self.device)   # (B, 7)
        num_choose = ((self.pose_count // 2) + 1) * 2   # (B,)

        if (mask * (self.end_pts == 1)).int().sum() > 0:
            iden_idx = self.choose_iden_idx[mask_start_idx, self.pose_count[mask_start_idx]]    # (M1,)
            cam_pose_new[mask_start_idx] = self.cam_pose_init[iden_idx].clone()

        for i in mask_reset_idx:
            batch_idx = self.choose_idx[i, :num_choose[i]].clone()
            start_pts = L - self.end_pts[i]
            end_pts = L - self.start_pts[i]
            cam_pose_cand_list = []
            joints_cand_list = []
            found = False
            
            for j in batch_idx:
                start_list = self.start_pts_list[j].copy()      # sub-list
                end_list = self.end_pts_list[j].copy()          # sub-list
                cam_pose_list = self.cam_pose_list[j].copy()    # sub-list
                joints_list = self.joints_list[j].copy()        # sub-list
                for k, l, cam_pose, joints in zip(start_list, end_list, cam_pose_list, joints_list):
                    if (k <= start_pts) & (l >= end_pts):
                        cam_pose_cand_list.append(cam_pose.clone())
                        joints_cand_list.append(joints.clone())
                        mask_reset[i] = False
                        found = True
                        # print(f"..............found a substitute cam pose for env {i}................")
                        break
            
            if found:
                cam_pose_cand = torch.stack(cam_pose_cand_list, dim=0)  # (M, 7)
                joints_cand = torch.stack(joints_cand_list, dim=0)      # (M, 6)
                min_idx = torch.argmin(torch.norm(joints_cand - self.joints_optimized[i].unsqueeze(0), dim=-1))
                cam_pose_new[i] = cam_pose_cand[min_idx]
        if mask_reset.int().sum() > 0:
            cam_pose_new[mask_reset] = self.resetPoses(mask_reset)
        self.cam_pose_optimized[mask_repeat] = cam_pose_new[mask_repeat].clone()
        self.cam_pos_optimized[mask_repeat] = self.cam_pose_optimized[mask_repeat, :3].clone()
        self.cam_quat_optimized[mask_repeat] = self.cam_pose_optimized[mask_repeat, 3:].clone()
        self.cam_rot_optimized[mask_repeat] = math.matrix_from_quat(self.cam_quat_optimized[mask_repeat].clone())


    def resetPoses(self, mask):
        """
        重置相机位姿的初值
        Args:
            mask: (B,), bool
        """
        idx = torch.arange(mask.shape[0], device=self.device)
        idx = idx[mask]
        mask_reset = mask.clone()

        mid_pts = ((self.start_pts[mask] + self.end_pts[mask]) / 2).floor()
        seam_line_m = torch.gather(self.seam_lines[mask].clone(), 1, mid_pts.clone()\
                                    .long().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3))[:, 0]
        seam_tangent_m = torch.gather(self.seam_tangents[mask].clone(), 1, mid_pts.clone()\
                                        .long().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3))[:, 0]
        seam_limits_m = torch.gather(self.seam_limits_s[mask].clone(), 1, mid_pts.clone()\
                                        .long().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, 3))[:, 0]
        cam_pose_new = self.generatePoses(
            seam_line_m = seam_line_m,
            seam_tangent_m = seam_tangent_m,
            seam_limits_m = seam_limits_m,
            cam_rot = self.cam_rot_optimized[mask].clone(),
        )

        self.pose_count[mask_reset] = (self.pose_count[mask_reset] + 1) % 4
        
        return cam_pose_new


    def resetPoses_0(self, mask):
        """
        重置相机位姿的初值
        Args:
            mask: (B,), bool
        """
        seam_line_s = torch.gather(self.seam_lines[mask].clone(), 1, self.start_pts[mask].clone()\
                                    .long().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3))[:, 0]
        seam_tangent_s = torch.gather(self.seam_tangents[mask].clone(), 1, self.start_pts[mask].clone()\
                                        .long().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3))[:, 0]
        seam_limits_s = torch.gather(self.seam_limits_s[mask].clone(), 1, self.start_pts[mask].clone()\
                                        .long().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, 3))[:, 0]
        seam_line_e = torch.gather(self.seam_lines[mask].clone(), 1, (self.end_pts[mask].clone() - 1)\
                                    .long().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3))[:, 0]
        seam_tangent_e = torch.gather(self.seam_tangents[mask].clone(), 1, (self.end_pts[mask].clone() - 1)\
                                        .long().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3))[:, 0]
        seam_limits_e = torch.gather(self.seam_limits_s[mask].clone(), 1, (self.end_pts[mask].clone() - 1)\
                                        .long().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, 3))[:, 0]
        seam_dir_s = torch.gather(self.seam_lines[mask].clone(), 1, (self.start_pts[mask].clone() + 1)\
                                    .long().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3))[:, 0] - seam_line_s
        seam_dir_e = torch.gather(self.seam_lines[mask].clone(), 1, (self.end_pts[mask].clone() - 2)\
                                    .long().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3))[:, 0] - seam_line_e
        self.cam_pose_optimized[mask] = self.generatePoses_0(
            seam_line_s = seam_line_s,
            seam_tangent_s = seam_tangent_s,
            seam_limits_s = seam_limits_s,
            seam_line_e = seam_line_e,
            seam_tangent_e = seam_tangent_e,
            seam_limits_e = seam_limits_e,
            seam_dir_s = seam_dir_s,
            seam_dir_e = seam_dir_e,
            cam_rot = self.cam_rot_optimized[mask].clone(),
            single = True
        )
        self.cam_pos_optimized[mask] = self.cam_pose_optimized[mask, :3].clone()
        self.cam_quat_optimized[mask] = self.cam_pose_optimized[mask, 3:].clone()
        self.cam_rot_optimized[mask] = math.matrix_from_quat(self.cam_quat_optimized[mask].clone())


    def generatePoses_0(self, seam_line_s: torch.Tensor, seam_tangent_s: torch.Tensor, seam_limits_s: torch.Tensor, 
                      seam_line_e: torch.Tensor, seam_tangent_e: torch.Tensor, seam_limits_e: torch.Tensor,
                      seam_dir_s: torch.Tensor, seam_dir_e: torch.Tensor, cam_rot: torch.Tensor, single: bool = True):
        """
        生成新的基于焊缝两个端点的相机位姿, old version
        Args:
            seam_line_s, seam_line_e: (B, 3)
            seam_tangent_s, seam_tangent_e: (B, 3)
            seam_limits_s, seam_limits_e: (B, 2, 3)
            seam_dir_s, seam_dir_e: (B, 3)
            cam_rot: (B, 3, 3)
        Rerurns:
            z_dir: (B, 3)
        """
        B = seam_line_s.shape[0]
        # generate 8 possible cam poses
        z_dir_s = -(seam_limits_s[:, 0] + seam_limits_s[:, -1]) / 2        # (B ,3)
        z_dir_s = z_dir_s / torch.clamp(torch.norm(z_dir_s, dim=-1, keepdim=True), min=1e-8)
        z_dir_e = -(seam_limits_e[:, 0] + seam_limits_e[:, -1]) / 2
        z_dir_e = z_dir_e / torch.clamp(torch.norm(z_dir_e, dim=-1, keepdim=True), min=1e-8)
        y_dir_s =  seam_tangent_s.clone()       # (B, 3)
        y_dir_e =  seam_tangent_e.clone()
        pos_s = seam_line_s               # (B, 3)
        pos_e = seam_line_e

        z_dir_lists = torch.stack((z_dir_s, z_dir_e), dim=1)            # (B, 2, 3)
        y_dir_lists = torch.stack((y_dir_s, y_dir_e), dim=1)            # (B, 2, 3)
        x_dir_lists = torch.cross(y_dir_lists, z_dir_lists, dim=-1)     # (B, 2, 3)
        pos_lists = torch.stack((pos_s, pos_e), dim=1)      # (B, 2, 3)
        rot0 = opt_utils.matrix_from_vectors(x_dir_lists, y_dir_lists, z_dir_lists)         # (B, 2, 3, 3)

        ones = torch.ones((rot0.shape[:2]), dtype=torch.float, device=self.device)
        rot1 = rot0 @ math.matrix_from_euler("X", ones * 45 / 180 * torch.pi)       # (B, 2, 3, 3)
        rot2 = rot0 @ math.matrix_from_euler("X", ones * -45 / 180 * torch.pi)      # (B, 2, 3, 3)
        rot_a = torch.cat((rot1, rot2), dim=1)              # (B, 4, 3, 3)
        pos_lists = pos_lists.repeat(1, 2, 1)                  # (B, 4, 3)

        ones = torch.ones((rot_a.shape[:2]), dtype=torch.float, device=self.device)
        rot_o = rot_a @ math.matrix_from_euler("Z", ones * 180 / 180 * torch.pi)    # (B, 4, 3, 3)
        rot_init = torch.cat((rot_a, rot_o), dim=1)         # (B, 8, 3, 3)
        pos_lists = pos_lists.repeat(1, 2, 1)                  # (B, 8, 3)

        quat_init = math.quat_from_matrix(rot_init)         # (B, 8, 4)
        dists = torch.tensor([0, 0, -0.5], dtype=torch.float, device=self.device).unsqueeze(0).unsqueeze(0).repeat(B, 8, 1) # (B, 8, 3)
        pos_init = pos_lists + math.quat_apply(quat_init, dists)    # (B, 8, 3)
        pose_init = torch.cat((pos_init, quat_init), dim=-1)        # (B, 8, 7)

        # select cam orientations
        seam_dirs = torch.stack((seam_dir_s, seam_dir_e), dim=-2)      # (B, 2, 3)
        seam_dirs = seam_dirs.repeat(1, 4, 1)       # (B, 8, 3)
        z_dirs = rot_init[:, :, :, 2].clone()
        mask = (torch.sum(seam_dirs * z_dirs, dim=-1) > 0)      # (B, 8)
        rot_sel = rot_init[mask].reshape(B, 4, 3, 3)            # (B, 4, 3, 3)
        pose_sel = pose_init[mask].reshape(B, 4, 7)             # (B, 4, 7)
        loss = (torch.sum(rot_sel[: ,:, :, 2] * cam_rot[:, :, 2].unsqueeze(-2), dim=-1) 
                - 0.2 * torch.sum(rot_sel[:, :, :, 0] * cam_rot[:, :, 0].unsqueeze(-2), dim=-1))    # (B, 4)
        
        if single:
            # version1
            idx = torch.argmin(loss, dim=-1)  # (B,)
            pose_sel = torch.gather(input=pose_sel, dim=1, index=idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 7))[:, 0] # (B, 7)
        else:
            # version2
            idx = torch.argsort(loss, dim=-1)    # (B, 4)
            pose_sel = torch.gather(input=pose_sel, dim=1, index=idx.unsqueeze(-1).repeat(1, 1, 7))     # (B, 4, 7)

        return pose_sel


    def generatePoses(self, seam_line_m: torch.Tensor, seam_tangent_m: torch.Tensor, seam_limits_m: torch.Tensor,
                      cam_rot: torch.Tensor):
        """
        生成新的基于焊缝两个端点的相机位姿
        Args:
            seam_line_m: (B, 3)
            seam_tangent_m: (B, 3)
            seam_limits_m: (B, 2, 3)
            cam_rot: (B, 3, 3)
        Rerurns:
            z_dir: (B, 7)
        """
        B = seam_line_m.shape[0]
        # generate 8 possible cam poses
        z_dir_m = -(seam_limits_m[:, 0] + seam_limits_m[:, -1]) / 2        # (B ,3)
        z_dir_m = z_dir_m / torch.clamp(torch.norm(z_dir_m, dim=-1, keepdim=True), min=1e-8)
        y_dir_m =  seam_tangent_m.clone()       # (B, 3)
        pos_m = seam_line_m               # (B, 3)

        z_dir_lists = torch.stack((z_dir_m,  z_dir_m), dim=1)            # (B, 2, 3)
        y_dir_lists = torch.stack((y_dir_m, -y_dir_m), dim=1)            # (B, 2, 3)
        x_dir_lists = torch.cross(y_dir_lists, z_dir_lists, dim=-1)     # (B, 2, 3)
        pos_lists = torch.stack((pos_m, pos_m), dim=1)      # (B, 2, 3)
        rot_lists = opt_utils.matrix_from_vectors(x_dir_lists, y_dir_lists, z_dir_lists)         # (B, 2, 3, 3)
        
        # select cam orientations
        diff_rot = torch.norm(rot_lists - cam_rot.unsqueeze(1), dim=(-2, -1))    # (B, 2)
        min_idx = torch.argmin(diff_rot, dim=-1)    # (B)
        batch_idx = torch.arange(B, device=self.device)
        pos_sel = pos_lists[batch_idx, min_idx]     # (B, 3)
        rot_sel = rot_lists[batch_idx, min_idx]     # (B, 3, 3)
        quat_sel = math.quat_from_matrix(rot_sel)         # (B, 4)       
        dists = torch.tensor([0, 0, -0.5], dtype=torch.float, device=self.device).unsqueeze(0).repeat(B, 1)     # (B, 3)
        pos_sel = pos_sel + math.quat_apply(quat_sel, dists)    # (B, 3)
        pose_sel = torch.cat((pos_sel, quat_sel), dim=-1)        # (B, 7)

        return pose_sel


    def flattenInputs_0(self, start_idx: torch.Tensor, end_idx: torch.Tensor, seam_lines: torch.Tensor, 
                      seam_tangent: torch.Tensor, seam_limits: torch.Tensor, seam_vertical: torch.Tensor, 
                      seam_theta: torch.Tensor, seam_rays_2d: torch.Tensor, cam_poses: torch.Tensor):
        """
        将计算vision cost的输入数据flatten化,便于并行计算
        Args:
            start_idx, end_idx: (B,)
            seam_lines: (B, N, 3)
            seam_tangent: (B, N, 3)
            seam_limits: (B, N, 2, 3)
            seam_vertical: (B, N, 3)
            seam_theta: (B, N, L)
            sema_rays_2d: (B, N, L, 2)
            cam_poses: (B, R, 7)
        Returns:
            seam_line, seam_tangent, seam_limits, cam_poses, idx
        """
        B, R = cam_poses.shape[:2]
        L = seam_theta.shape[2]

        idx_s = torch.zeros((B), dtype=torch.int, device=self.device)   # (B,)
        idx_e = torch.zeros((B), dtype=torch.int, device=self.device)
        idx_mul_s = torch.zeros((B), dtype=torch.int, device=self.device)   # (B,)
        idx_mul_e = torch.zeros((B), dtype=torch.int, device=self.device)
        seam_line_lists = []
        seam_tangent_lists = []
        seam_limits_lists = []
        cam_poses_lists = []
        seam_line_mul_lists = []
        seam_tangent_mul_lists = []
        seam_limit_mul_lists = []
        seam_vertical_mul_lists = []
        seam_theta_mul_lists = []
        seam_rays_2d_mul_lists = []
        seam_dir_mul_lists = []
        cam_poses_mul_lists = []

        for i in range(B):
            seam_line_base = seam_lines[i, start_idx[i]:end_idx[i]].clone()             # (N', 3)
            N = seam_line_base.shape[0]
            seam_tangent_base = seam_tangent[i, start_idx[i]:end_idx[i]].clone()        # (N', 3)
            seam_limits_base = seam_limits[i, start_idx[i]:end_idx[i]].clone()          # (N', 2, 3)
            seam_vertical_base = seam_vertical[i, start_idx[i]:end_idx[i]].clone()      # (N', 3)
            seam_theta_base = seam_theta[i, start_idx[i]:end_idx[i]].clone()            # (N', L)
            seam_rays_2d_base = seam_rays_2d[i, start_idx[i]:end_idx[i]].clone()        # (N', L, 2)
            cam_poses_base = cam_poses[i].clone()                                 # (R, 7)

            # single
            seam_line_single = seam_line_base.unsqueeze(0).expand(R, N, 3).reshape(R*N, 3).clone()              # (R*N', 3)            
            seam_tangent_single = seam_tangent_base.unsqueeze(0).expand(R, N, 3).reshape(R*N, 3).clone()        # (R*N', 3)            
            seam_limits_single = seam_limits_base.unsqueeze(0).expand(R, N, 2, 3).reshape(R*N, 2, 3).clone()    # (R*N', 2, 3)
            cam_poses_single = cam_poses_base.unsqueeze(1).expand(R, N, 7).reshape(R*N, 7)        # (R*N', 7)            
            seam_line_lists.append(seam_line_single)
            seam_tangent_lists.append(seam_tangent_single)
            seam_limits_lists.append(seam_limits_single)
            cam_poses_lists.append(cam_poses_single)
            if i == 0:
                idx_s[i] = 0
            else:
                idx_s[i] = idx_e[i-1].clone()
            idx_e[i] = idx_s[i] + R * N

            # multiple
            seam_line_mul_o = seam_line_base.unsqueeze(0).unsqueeze(0).expand(R, N, N, 3).reshape(R*N*N, 3).clone()   # (R*N'*N', 3)
            seam_line_mul_p = seam_line_base.unsqueeze(0).unsqueeze(-2).expand(R, N, N, 3).reshape(R*N*N, 3).clone()   # (R*N'*N', 3)
            seam_line_mul = torch.stack((seam_line_mul_o, seam_line_mul_p), dim=-2)         # (R*N'*N', 2, 3)
            seam_tangent_mul = seam_tangent_base.unsqueeze(0).unsqueeze(0).expand(R, N, N, 3).reshape(R*N*N, 3).clone()   # (R*N'*N', 3)
            seam_limit_mul = seam_limits_base[:, 0].unsqueeze(0).unsqueeze(0).expand(R, N, N, 3).reshape(R*N*N, 3).clone()   # (R*N'*N', 3)
            seam_vertical_mul = seam_vertical_base.unsqueeze(0).unsqueeze(0).expand(R, N, N, 3).reshape(R*N*N, 3).clone()   # (R*N'*N', 3)
            seam_theta_mul = seam_theta_base.unsqueeze(0).unsqueeze(0).expand(R, N, N, L).reshape(R*N*N, L).clone()   # (R*N'*N', L)
            seam_rays_2d_mul = seam_rays_2d_base.unsqueeze(0).unsqueeze(0).expand(R, N, N, L, 2).reshape(R*N*N, L, 2).clone()   # (R*N'*N', L, 2)
            cam_poses_mul = cam_poses_base.unsqueeze(1).unsqueeze(1).expand(R, N, N, 7).reshape(R*N*N, 7)        # (R*N'*N', 7)
            seam_dir_mul = cam_poses_mul[:, :3] - seam_line_mul_p       # (R*N'*N', 3)
            seam_line_mul_lists.append(seam_line_mul)
            seam_tangent_mul_lists.append(seam_tangent_mul)
            seam_limit_mul_lists.append(seam_limit_mul)
            seam_vertical_mul_lists.append(seam_vertical_mul)
            seam_theta_mul_lists.append(seam_theta_mul)
            seam_rays_2d_mul_lists.append(seam_rays_2d_mul)
            cam_poses_mul_lists.append(cam_poses_mul)
            seam_dir_mul_lists.append(seam_dir_mul)
            if i == 0:
                idx_mul_s[i] = 0
            else:
                idx_mul_s[i] = idx_mul_e[i-1].clone()
            idx_mul_e[i] = idx_mul_s[i] + R * N * N

        seam_line_lists = torch.cat(seam_line_lists, dim=0)         # (M, 3)
        seam_tangent_lists = torch.cat(seam_tangent_lists, dim=0)   # (M, 3)
        seam_limits_lists = torch.cat(seam_limits_lists, dim=0)     # (M, 2, 3)
        cam_poses_lists = torch.cat(cam_poses_lists, dim=0)         # (M, 7)

        seam_line_mul_lists = torch.cat(seam_line_mul_lists, dim=0)
        seam_tangent_mul_lists = torch.cat(seam_tangent_mul_lists, dim=0)
        seam_limit_mul_lists = torch.cat(seam_limit_mul_lists, dim=0)
        seam_vertical_mul_lists = torch.cat(seam_vertical_mul_lists, dim=0)
        seam_theta_mul_lists = torch.cat(seam_theta_mul_lists, dim=0)
        seam_rays_2d_mul_lists = torch.cat(seam_rays_2d_mul_lists, dim=0)
        seam_dir_mul_lists = torch.cat(seam_dir_mul_lists, dim=0)
        cam_poses_mul_lists = torch.cat(cam_poses_mul_lists, dim=0)

        idx = torch.stack((idx_s, idx_e), dim=1)        # (B, 2)
        idx_mul = torch.stack((idx_mul_s, idx_mul_e), dim=1)        # (B, 2)

        return (
            seam_line_lists,
            seam_tangent_lists,
            seam_limits_lists,
            cam_poses_lists,
            idx,
            seam_line_mul_lists,
            seam_tangent_mul_lists,
            seam_limit_mul_lists,
            seam_vertical_mul_lists,
            seam_theta_mul_lists,
            seam_rays_2d_mul_lists,
            seam_dir_mul_lists,
            cam_poses_mul_lists,
            idx_mul,
            R
        )

    
    def flattenInputs_1(self, start_idx: torch.Tensor, end_idx: torch.Tensor, seam_lines: torch.Tensor, 
                      seam_tangent: torch.Tensor, seam_limits: torch.Tensor, cam_poses: torch.Tensor):
        """
        将计算vision cost的输入数据flatten化,便于并行计算
        Args:
            start_idx, end_idx: (B,)
            seam_lines: (B, N, 3)
            seam_tangent: (B, N, 3)
            seam_limits: (B, N, 2, 3)
            cam_poses: (B, R, 7)
        Returns:
            seam_line, seam_tangent, seam_limits, cam_poses, idx
        """
        B, R = cam_poses.shape[:2]

        idx_s = torch.zeros((B), dtype=torch.int, device=self.device)   # (B,)
        idx_e = torch.zeros((B), dtype=torch.int, device=self.device)
        seam_line_lists = []
        seam_tangent_lists = []
        seam_limits_lists = []
        cam_poses_lists = []
        block_mask_lists = []

        for i in range(B):
            seam_line_single = seam_lines[i, start_idx[i]:end_idx[i]].clone()       # (N', 3)
            N = seam_line_single.shape[0]
            seam_tangent_single = seam_tangent[i, start_idx[i]:end_idx[i]].clone()     # (N', 3)
            seam_limits_single = seam_limits[i, start_idx[i]:end_idx[i]].clone()       # (N', 2, 3)
            cam_poses_single = cam_poses[i].clone()                                 # (R, 7)
            block_mask_single = torch.arange(int(start_idx[i]), int(end_idx[i]), device=self.device)
            block_mask_single = (block_mask_single == 0) + (block_mask_single == (self.num_points - 1))   # (N',)
            # block_mask_single = (torch.any(block_mask_single).int() * torch.ones((N,), dtype=torch.int, device=self.device)).bool()

            seam_line_single = seam_line_single.unsqueeze(0).repeat(R, 1, 1).reshape(R*N, 3)        # (R*N', 3)            
            seam_tangent_single = seam_tangent_single.unsqueeze(0).repeat(R, 1, 1).reshape(R*N, 3)  # (R*N', 3)            
            seam_limits_single = seam_limits_single.unsqueeze(0).repeat(R, 1, 1, 1).reshape(R*N, 2, 3)   # (R*N', 2, 3)            
            cam_poses_single = cam_poses_single.unsqueeze(1).repeat(1, N, 1).reshape(R*N, 7)        # (R*N', 7)
            block_mask_single = block_mask_single.unsqueeze(0).repeat(R, 1).reshape(R*N)            # (R*N,)   
            seam_line_lists.append(seam_line_single)
            seam_tangent_lists.append(seam_tangent_single)
            seam_limits_lists.append(seam_limits_single)
            cam_poses_lists.append(cam_poses_single)
            block_mask_lists.append(block_mask_single)
            if i == 0:
                idx_s[i] = 0
            else:
                idx_s[i] = idx_e[i-1]
            idx_e[i] = idx_s[i] + R * N
        
        seam_line_lists = torch.cat(seam_line_lists, dim=0)         # (M, 3)
        seam_tangent_lists = torch.cat(seam_tangent_lists, dim=0)   # (M, 3)
        seam_limits_lists = torch.cat(seam_limits_lists, dim=0)     # (M, 2, 3)
        cam_poses_lists = torch.cat(cam_poses_lists, dim=0)         # (M, 7)
        block_mask_lists = torch.cat(block_mask_lists, dim=0)       # (M,)
        idx = torch.stack((idx_s, idx_e), dim=1)        # (B, 2)
        
        return seam_line_lists, seam_tangent_lists, seam_limits_lists, cam_poses_lists, idx, block_mask_lists, R


    def generateSeamData(self, start_idx: torch.Tensor, end_idx: torch.Tensor, seam_lines: torch.Tensor, 
                      seam_tangent: torch.Tensor, seam_limits: torch.Tensor, seam_vertical: torch.Tensor, 
                      seam_theta: torch.Tensor, seam_rays_2d: torch.Tensor):
        """
        将计算vision cost的输入数据flatten化,便于并行计算
        Args:
            start_idx, end_idx: (B,)
            seam_lines: (B, N, 3)
            seam_tangent: (B, N, 3)
            seam_limits: (B, N, 2, 3)
            seam_vertical: (B, N, 3)
            seam_theta: (B, N, L)
            sema_rays_2d: (B, N, L, 2)
        Returns:
            seam_line, seam_tangent, seam_limits, cam_poses, idx
        """
        B = seam_lines.shape[0]
        L = seam_theta.shape[2]

        idx_s = torch.zeros((B), dtype=torch.int, device=self.device)   # (B,)
        idx_e = torch.zeros((B), dtype=torch.int, device=self.device)
        idx_mul_s = torch.zeros((B), dtype=torch.int, device=self.device)   # (B,)
        idx_mul_e = torch.zeros((B), dtype=torch.int, device=self.device)
        seam_line_lists = []
        seam_tangent_lists = []
        seam_limits_lists = []
        seam_line_mul_lists = []
        seam_tangent_mul_lists = []
        seam_limit_mul_lists = []
        seam_vertical_mul_lists = []
        seam_theta_mul_lists = []
        seam_rays_2d_mul_lists = []

        idx_s_1 = torch.zeros((B), dtype=torch.int, device=self.device)   # (B,)
        idx_e_1 = torch.zeros((B), dtype=torch.int, device=self.device)
        idx_mul_s_1 = torch.zeros((B), dtype=torch.int, device=self.device)   # (B,)
        idx_mul_e_1 = torch.zeros((B), dtype=torch.int, device=self.device)
        seam_line_lists_1 = []
        seam_tangent_lists_1 = []
        seam_limits_lists_1 = []
        seam_line_mul_lists_1 = []
        seam_tangent_mul_lists_1 = []
        seam_limit_mul_lists_1 = []
        seam_vertical_mul_lists_1 = []
        seam_theta_mul_lists_1 = []
        seam_rays_2d_mul_lists_1 = []

        for i in range(B):
            seam_line_base = seam_lines[i, start_idx[i]:end_idx[i]].clone()             # (N', 3)
            N = seam_line_base.shape[0]
            seam_tangent_base = seam_tangent[i, start_idx[i]:end_idx[i]].clone()        # (N', 3)
            seam_limits_base = seam_limits[i, start_idx[i]:end_idx[i]].clone()          # (N', 2, 3)
            seam_vertical_base = seam_vertical[i, start_idx[i]:end_idx[i]].clone()      # (N', 3)
            seam_theta_base = seam_theta[i, start_idx[i]:end_idx[i]].clone()            # (N', L)
            seam_rays_2d_base = seam_rays_2d[i, start_idx[i]:end_idx[i]].clone()        # (N', L, 2)

            R = 50
            # single
            seam_line_single = seam_line_base.unsqueeze(0).expand(R, N, 3).reshape(R*N, 3).clone()              # (R*N', 3)            
            seam_tangent_single = seam_tangent_base.unsqueeze(0).expand(R, N, 3).reshape(R*N, 3).clone()        # (R*N', 3)            
            seam_limits_single = seam_limits_base.unsqueeze(0).expand(R, N, 2, 3).reshape(R*N, 2, 3).clone()    # (R*N', 2, 3)            
            seam_line_lists.append(seam_line_single)
            seam_tangent_lists.append(seam_tangent_single)
            seam_limits_lists.append(seam_limits_single)
            if i == 0:
                idx_s[i] = 0
            else:
                idx_s[i] = idx_e[i-1].clone()
            idx_e[i] = idx_s[i] + R * N

            # multiple
            seam_line_mul = seam_line_base.unsqueeze(0).unsqueeze(0).expand(R, N, N, 3).reshape(R*N*N, 3).clone()   # (R*N'*N', 3)
            seam_tangent_mul = seam_tangent_base.unsqueeze(0).unsqueeze(0).expand(R, N, N, 3).reshape(R*N*N, 3).clone()   # (R*N'*N', 3)
            seam_limit_mul = seam_limits_base[:, 0].unsqueeze(0).unsqueeze(0).expand(R, N, N, 3).reshape(R*N*N, 3).clone()   # (R*N'*N', 3)
            seam_vertical_mul = seam_vertical_base.unsqueeze(0).unsqueeze(0).expand(R, N, N, 3).reshape(R*N*N, 3).clone()   # (R*N'*N', 3)
            seam_theta_mul = seam_theta_base.unsqueeze(0).unsqueeze(0).expand(R, N, N, L).reshape(R*N*N, L).clone()   # (R*N'*N', L)
            seam_rays_2d_mul = seam_rays_2d_base.unsqueeze(0).unsqueeze(0).expand(R, N, N, L, 2).reshape(R*N*N, L, 2).clone()   # (R*N'*N', L, 2)
            seam_line_mul_lists.append(seam_line_mul)
            seam_tangent_mul_lists.append(seam_tangent_mul)
            seam_limit_mul_lists.append(seam_limit_mul)
            seam_vertical_mul_lists.append(seam_vertical_mul)
            seam_theta_mul_lists.append(seam_theta_mul)
            seam_rays_2d_mul_lists.append(seam_rays_2d_mul)
            if i == 0:
                idx_mul_s[i] = 0
            else:
                idx_mul_s[i] = idx_mul_e[i-1].clone()
            idx_mul_e[i] = idx_mul_s[i] + R * N * N


            R = 1
            # single
            seam_line_single_1 = seam_line_base.unsqueeze(0).expand(R, N, 3).reshape(R*N, 3).clone()              # (R*N', 3)            
            seam_tangent_single_1 = seam_tangent_base.unsqueeze(0).expand(R, N, 3).reshape(R*N, 3).clone()        # (R*N', 3)            
            seam_limits_single_1 = seam_limits_base.unsqueeze(0).expand(R, N, 2, 3).reshape(R*N, 2, 3).clone()    # (R*N', 2, 3)            
            seam_line_lists_1.append(seam_line_single_1)
            seam_tangent_lists_1.append(seam_tangent_single_1)
            seam_limits_lists_1.append(seam_limits_single_1)
            if i == 0:
                idx_s_1[i] = 0
            else:
                idx_s_1[i] = idx_e_1[i-1].clone()
            idx_e_1[i] = idx_s_1[i] + R * N

            # multiple
            seam_line_mul_1 = seam_line_base.unsqueeze(0).unsqueeze(0).expand(R, N, N, 3).reshape(R*N*N, 3).clone()   # (R*N'*N', 3)
            seam_tangent_mul_1 = seam_tangent_base.unsqueeze(0).unsqueeze(0).expand(R, N, N, 3).reshape(R*N*N, 3).clone()   # (R*N'*N', 3)
            seam_limit_mul_1 = seam_limits_base[:, 0].unsqueeze(0).unsqueeze(0).expand(R, N, N, 3).reshape(R*N*N, 3).clone()   # (R*N'*N', 3)
            seam_vertical_mul_1 = seam_vertical_base.unsqueeze(0).unsqueeze(0).expand(R, N, N, 3).reshape(R*N*N, 3).clone()   # (R*N'*N', 3)
            seam_theta_mul_1 = seam_theta_base.unsqueeze(0).unsqueeze(0).expand(R, N, N, L).reshape(R*N*N, L).clone()   # (R*N'*N', L)
            seam_rays_2d_mul_1 = seam_rays_2d_base.unsqueeze(0).unsqueeze(0).expand(R, N, N, L, 2).reshape(R*N*N, L, 2).clone()   # (R*N'*N', L, 2)
            seam_line_mul_lists_1.append(seam_line_mul_1)
            seam_tangent_mul_lists_1.append(seam_tangent_mul_1)
            seam_limit_mul_lists_1.append(seam_limit_mul_1)
            seam_vertical_mul_lists_1.append(seam_vertical_mul_1)
            seam_theta_mul_lists_1.append(seam_theta_mul_1)
            seam_rays_2d_mul_lists_1.append(seam_rays_2d_mul_1)
            if i == 0:
                idx_mul_s_1[i] = 0
            else:
                idx_mul_s_1[i] = idx_mul_e_1[i-1].clone()
            idx_mul_e_1[i] = idx_mul_s_1[i] + R * N * N

        seam_line_lists = torch.cat(seam_line_lists, dim=0)         # (M, 3)
        seam_tangent_lists = torch.cat(seam_tangent_lists, dim=0)   # (M, 3)
        seam_limits_lists = torch.cat(seam_limits_lists, dim=0)     # (M, 2, 3)
        lists_single = (
            seam_line_lists,
            seam_tangent_lists,
            seam_limits_lists
        )
        seam_line_mul_lists = torch.cat(seam_line_mul_lists, dim=0)
        seam_tangent_mul_lists = torch.cat(seam_tangent_mul_lists, dim=0)
        seam_limit_mul_lists = torch.cat(seam_limit_mul_lists, dim=0)
        seam_vertical_mul_lists = torch.cat(seam_vertical_mul_lists, dim=0)
        seam_theta_mul_lists = torch.cat(seam_theta_mul_lists, dim=0)
        seam_rays_2d_mul_lists = torch.cat(seam_rays_2d_mul_lists, dim=0)
        lists_multiple = (
            seam_line_mul_lists,
            seam_tangent_mul_lists,
            seam_limit_mul_lists,
            seam_vertical_mul_lists,
            seam_theta_mul_lists,
            seam_rays_2d_mul_lists
        )
        idx = torch.stack((idx_s, idx_e), dim=1)        # (B, 2)
        idx_mul = torch.stack((idx_mul_s, idx_mul_e), dim=1)        # (B, 2)

        seam_line_lists_1 = torch.cat(seam_line_lists_1, dim=0)         # (M, 3)
        seam_tangent_lists_1 = torch.cat(seam_tangent_lists_1, dim=0)   # (M, 3)
        seam_limits_lists_1 = torch.cat(seam_limits_lists_1, dim=0)     # (M, 2, 3)
        lists_single_1 = (
            seam_line_lists_1,
            seam_tangent_lists_1,
            seam_limits_lists_1
        )
        seam_line_mul_lists_1 = torch.cat(seam_line_mul_lists_1, dim=0)
        seam_tangent_mul_lists_1 = torch.cat(seam_tangent_mul_lists_1, dim=0)
        seam_limit_mul_lists_1 = torch.cat(seam_limit_mul_lists_1, dim=0)
        seam_vertical_mul_lists_1 = torch.cat(seam_vertical_mul_lists_1, dim=0)
        seam_theta_mul_lists_1 = torch.cat(seam_theta_mul_lists_1, dim=0)
        seam_rays_2d_mul_lists_1 = torch.cat(seam_rays_2d_mul_lists_1, dim=0)
        lists_multiple_1 = (
            seam_line_mul_lists_1,
            seam_tangent_mul_lists_1,
            seam_limit_mul_lists_1,
            seam_vertical_mul_lists_1,
            seam_theta_mul_lists_1,
            seam_rays_2d_mul_lists_1
        )
        idx_1 = torch.stack((idx_s_1, idx_e_1), dim=1)        # (B, 2)
        idx_mul_1 = torch.stack((idx_mul_s_1, idx_mul_e_1), dim=1)        # (B, 2)
        
        return lists_single, lists_multiple, idx, idx_mul, lists_single_1, lists_multiple_1, idx_1, idx_mul_1