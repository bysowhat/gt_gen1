import torch
import numpy as np

from . import math_traj as math

class UR12e_t:
    def __init__(self, num_envs, device="cuda"):
        self.num_envs = num_envs
        self.device = device
        camera_to_j6link_pos = torch.tensor((-0.07614382311635252, -0.035416740219402776, 0.16171334688469158), dtype=torch.float64, device=self.device)
        camera_to_j6link_rot = torch.tensor((0.679148984404206, -0.11969930098925956, -0.2194543927388376, -0.6901220934248131), dtype=torch.float64, device=self.device)
        # camera_to_j6link_pos = torch.tensor((-0.07712, -0.032, 0.16814), dtype=torch.float64, device=self.device)
        # camera_to_j6link_rot = torch.tensor((0.679148984404206, -0.11969930098925956, -0.2194543927388376, -0.6901220934248131), dtype=torch.float64, device=self.device)
        self.cam_to_j6_pose = self.transform(camera_to_j6link_pos, camera_to_j6link_rot)
        self.cam_inv = torch.linalg.inv(self.cam_to_j6_pose)
        self.cam_inv = self.cam_inv.unsqueeze(0).repeat(self.num_envs, 1, 1)
        dh_params = [
            [0.0,   0.1807,     0.0,        0.0,],
            [0.0,   0.0,        0.0,        1.5707963267948966],
            [0.0,   0.0,        -0.6127,    0.0],
            [0.0,   0.17415,    -0.57155,   0.0],
            [0.0,   0.11985,    0.0,        1.5707963267948966],
            [0.0,   0.11655,    0.0,        -1.5707963267948966]
        ]   # (theta, d, a, alpha)
        self.dh_params = torch.tensor(dh_params, dtype=torch.float64, device=self.device) 
        self.dh_params = self.dh_params.unsqueeze(0).repeat(self.num_envs, 1, 1)   #(B, 6, 4)
        self.da = self.dh_params[:, :, 2]       #(B, 6)
        self.dd = self.dh_params[:, :, 1]       #(B, 6)
        self.all_idx = torch.arange(num_envs, device=self.device)


    def transform(self, cam_pos, cam_rot):
        R = math.matrix_from_quat(cam_rot)
        T = torch.zeros((4, 4), dtype=torch.float64, device=self.device)
        T[:3, :3] = R[:, :].double()
        T[:3, 3] = cam_pos
        T[3, 3] = 1
        return T


    # IK
    def remainder(self, a, b):
        c = torch.remainder(a, b)
        return torch.where(c >= b/2, c - b, c)


    def solve_fairino_ec(self, cam_pose, env_ids = None):
        if env_ids is None:
            env_ids = self.all_idx
        
        T = cam_pose.double() @ self.cam_inv       # (B, 4, 4)

        solutions = []

        n_x, o_x, a_x, p_x = T[env_ids, 0, 0], T[env_ids, 0, 1], T[env_ids, 0, 2], T[env_ids, 0, 3]    # (B,)
        n_y, o_y, a_y, p_y = T[env_ids, 1, 0], T[env_ids, 1, 1], T[env_ids, 1, 2], T[env_ids, 1, 3]
        n_z, o_z, a_z, p_z = T[env_ids, 2, 0], T[env_ids, 2, 1], T[env_ids, 2, 2], T[env_ids, 2, 3]
        a_3, a_4 = self.da[env_ids, 2], self.da[env_ids, 3]     # (B,)
        d_1, d_2, d_3, d_4, d_5, d_6 = self.dd[env_ids, 0], self.dd[env_ids, 1], self.dd[env_ids, 2], self.dd[env_ids, 3], self.dd[env_ids, 4], self.dd[env_ids, 5]

        j_1s = []
        j_1s.append(2 * torch.atan((-a_x * d_6 + p_x + torch.sqrt(
                a_x ** 2 * d_6 ** 2 - 2 * a_x * d_6 * p_x + a_y ** 2 * d_6 ** 2 - 2 * a_y * d_6 * p_y - d_2 ** 2 - 2 * d_2 * d_3 - 2 * d_2 * d_4 - d_3 ** 2 - 2 * d_3 * d_4 - d_4 ** 2 + p_x ** 2 + p_y ** 2)) / (
                                                   a_y * d_6 + d_2 + d_3 + d_4 - p_y)))    # (B,)
        j_1s.append(-2 * torch.atan((a_x * d_6 - p_x + torch.sqrt(
                a_x ** 2 * d_6 ** 2 - 2 * a_x * d_6 * p_x + a_y ** 2 * d_6 ** 2 - 2 * a_y * d_6 * p_y - d_2 ** 2 - 2 * d_2 * d_3 - 2 * d_2 * d_4 - d_3 ** 2 - 2 * d_3 * d_4 - d_4 ** 2 + p_x ** 2 + p_y ** 2)) / (
                                                  a_y * d_6 + d_2 + d_3 + d_4 - p_y)))

        for j_1 in j_1s:
            j_1 = self.remainder(j_1, 2 * torch.pi)
            j_5s = []
            j_5s.append(-torch.acos(a_x * torch.sin(j_1) - a_y * torch.cos(j_1)))
            j_5s.append(torch.acos(a_x * torch.sin(j_1) - a_y * torch.cos(j_1)))
            
            for j_5 in j_5s:
                j_5 = self.remainder(j_5, 2 * torch.pi)
                j_6 = torch.atan2((-o_x * torch.sin(j_1) + o_y * torch.cos(j_1)) / torch.sin(j_5),
                                 (n_x * torch.sin(j_1) - n_y * torch.cos(j_1)) / torch.sin(j_5))
                j_6 = self.remainder(j_6, 2 * torch.pi)
                J = torch.atan2(-a_z / torch.sin(j_5), -(a_x * torch.cos(j_1) + a_y * torch.sin(j_1)) / torch.sin(j_5))
                M = -d_5 * torch.sin(J) + d_6 * torch.sin(j_5) * torch.cos(J) + p_x * torch.cos(j_1) + p_y * torch.sin(j_1)
                N = -d_1 + d_5 * torch.cos(J) + d_6 * torch.sin(J) * torch.sin(j_5) + p_z
                L = (1 / 2) * (M ** 2 + N ** 2 + a_3 ** 2 - a_4 ** 2) / a_3
                j_2s = []
                j_2s.append(2 * torch.atan((N - torch.sqrt(-L ** 2 + M ** 2 + N ** 2)) / (L + M)))
                j_2s.append(2 * torch.atan((N + torch.sqrt(-L ** 2 + M ** 2 + N ** 2)) / (L + M)))

                for j_2 in j_2s:
                    j_2 = self.remainder(j_2, 2 * torch.pi)
                    # j2 range is [-pi * 3/2, pi / 2] for fairino
                    j_2 = torch.where(j_2 > torch.pi / 2, j_2 - torch.pi * 2, j_2)
                    j_3 = -j_2 + torch.atan2((N - a_3 * torch.sin(j_2)) / a_4, (M - a_3 * torch.cos(j_2)) / a_4)
                    j_3 = self.remainder(j_3, 2 * torch.pi)
                    j_4 = self.remainder(J - j_2 - j_3, 2 * torch.pi)
                    # j4 range is [-pi * 3/2, pi / 2] for fairino
                    j_4 = torch.where(j_4 > torch.pi / 2, j_4 - torch.pi * 2, j_4)
                    joint_states = torch.stack([j_1, j_2, j_3, j_4, j_5, j_6], dim=-1)  # (B, 6)
                    solutions.append(joint_states)
        solutions = torch.stack(solutions, dim=1)

        return solutions
    

    # FK
    def improved_dh_matrix(self, theta, d, a, alpha):
        # torch.sin() and torch.cos() are all based in radians, not in degrees
        """
        theta: shape (B, 6,)
        d: shape (B, 6,)
        a: shape (B, 6,)
        alpha: shape (B, 6,)
        """
        B = theta.shape[0]
        zero = torch.zeros_like(theta, dtype=torch.float, device=self.device)
        one = torch.ones_like(theta, dtype=torch.float, device=self.device)
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        cos_alpha, sin_alpha = torch.cos(alpha), torch.sin(alpha)
        row1 = torch.stack((cos_theta, -sin_theta, zero, a), dim=-1)     # (B, 6, 4)[cos_theta, -sin_theta, 0, a]
        row2 = torch.stack((sin_theta * cos_alpha, cos_theta * cos_alpha, -sin_alpha, -d * sin_alpha), dim=-1)
        row3 = torch.stack((sin_theta * sin_alpha, cos_theta * sin_alpha, cos_alpha, d * cos_alpha), dim=-1)
        row4 = torch.stack((zero, zero, zero, one), dim=-1)
        mat = torch.stack((row1, row2, row3, row4), dim=2)      # (B, 6, 4, 4)
        return mat.double()


    def forward_cam_pose(self, joint_states):
        """
        计算相机的pose,基座标为靠上底座,输出为(pos, quat)
        """
        joint_states = joint_states.to(self.device)
        self.dh_params[:,:,0] = joint_states[:,:]

        T = torch.eye(4).to(self.device).unsqueeze(0).repeat(self.num_envs, 1, 1).double()      # (B, 4, 4)
        T_joints = self.improved_dh_matrix(self.dh_params[:, :, 0], self.dh_params[:, :, 1],
                                           self.dh_params[:, :, 2], self.dh_params[:, :, 3])    # (B, 6, 4, 4)

        final_T = T
        for i in range(6):
            final_T = final_T @ T_joints[:,i, :, :]
        final_T = final_T @ self.cam_to_j6_pose

        return final_T


class CameraFKConfig:
    """关节角 → base 系相机 4×4 位姿，【全部取自 config，无写死值】（决策 D6）。

    机器人本体：走 `config.robot_cfg_path`（curobo `ur12e_full.yml`，URDF-based）的运动学，
      与规划器 / stomp ik_cam 完全同源——curobo yml 本就没有 DH 数字，故"DH 从 config"
      的正确落地是走该机器人模型算 FK，而非另解析一套 DH。
    相机外参：走 `config.camera` 的 `extrinsic_pos` / `extrinsic_quat_wxyz` / `mount_link`。

    相机帧 = T_base_<mount_link> @ T(extrinsic_pos, extrinsic_quat_wxyz)，与
    `gt_gen/sensor.py:camera_pose_from_config` 同一约定。需 curobo + GPU（env_isaaclab）。

    这是采样链路唯一使用的 FK；老 `UR12e_t`（写死 DH + 外参）仅供 legacy `down_sampling.py`，
    采样不再触碰。
    """

    def __init__(self, config, device="cuda"):
        # 延迟 import：curobo 依赖重，且需项目根在 sys.path（gt_gen 可 import）
        from gt_gen.sensor import build_kinematics, load_camera_model, pose_to_T

        self.device = device
        self.kin = build_kinematics(config)               # curobo CudaRobotModel（link_names含 Link6）
        self.cam = load_camera_model(config)              # 扁平相机 dict（含外参）
        self.mount_link = self.cam.get("mount_link", "Link6")
        # 手眼外参 T_link_cam（相对挂载 link），config 单一真源
        T_l6_cam = pose_to_T(self.cam["extrinsic_pos"], self.cam["extrinsic_quat_wxyz"])
        self.T_l6_cam = torch.as_tensor(T_l6_cam, dtype=torch.float64, device=device)  # (4,4)

    def forward_cam_pose(self, joint_states):
        """joint_states: (T,6) → 相机 base 系位姿 (T,4,4) float64（本进程 device）。"""
        qt = torch.as_tensor(joint_states, dtype=torch.float32, device=self.device)
        if qt.dim() == 1:
            qt = qt.unsqueeze(0)
        st = self.kin.get_state(qt)                       # curobo 批量 FK
        lp = st.link_pose[self.mount_link]
        pos = lp.position.to(dtype=torch.float64, device=self.device)      # (T,3)
        quat = lp.quaternion.to(dtype=torch.float64, device=self.device)   # (T,4) wxyz
        R = math.matrix_from_quat(quat)                   # (T,3,3)
        Tn = qt.shape[0]
        T_base_l6 = torch.eye(4, dtype=torch.float64, device=self.device).unsqueeze(0).repeat(Tn, 1, 1)
        T_base_l6[:, :3, :3] = R
        T_base_l6[:, :3, 3] = pos
        return T_base_l6 @ self.T_l6_cam                  # (T,4,4)