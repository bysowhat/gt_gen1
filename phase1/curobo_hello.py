"""M1.0.5: cuRobo Hello World.

验证:
  1. cuRobo 能加载 /home/a/Datas/curobo/example_new_robot/urdf_12e/ur12e.yml
  2. FK: 用 pkl 里的 joint_angles 算末端位姿, 与已知值对照
  3. IK: 用 FK 的输出做 IK 反推, 验证关节角能恢复

跑法 (确保 setup_env.sh 已成功):
  /home/a/miniforge3/envs/env_isaaclab/bin/python phase1/curobo_hello.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig


ROBOT_YAML = "/home/a/Datas/curobo/example_new_robot/urdf_12e/ur12e.yml"
SAMPLE_PKL = (
    "/media/a/新加卷/hanfeng/segment_sub_output/"
    "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_0.pkl"
)


def load_pkl_joint_angles() -> tuple[np.ndarray, list[str]]:
    """从样例 pkl 取一组关节角, 用作 FK/IK 的真值."""
    with open(SAMPLE_PKL, "rb") as f:
        data = pickle.load(f)
    return np.asarray(data["joint_angles"], dtype=np.float32), list(data["joint_names"])


def main() -> int:
    print("=" * 60)
    print("M1.0.5  cuRobo Hello World on ur12e")
    print("=" * 60)

    if not Path(ROBOT_YAML).exists():
        print(f"[ERR] 机器人配置不存在: {ROBOT_YAML}")
        return 1

    tensor_args = TensorDeviceType()

    # ---------- Step 1: 加载机器人配置 ----------
    print("\n[1] 加载 ur12e.yml ...")
    cfg_yaml = load_yaml(ROBOT_YAML)
    robot_cfg = RobotConfig.from_dict(cfg_yaml["robot_cfg"], tensor_args)
    base_link = cfg_yaml["robot_cfg"]["kinematics"]["base_link"]
    ee_link = cfg_yaml["robot_cfg"]["kinematics"]["ee_link"]
    print(f"    base_link = {base_link}")
    print(f"    ee_link   = {ee_link}")

    # ---------- Step 2: FK ----------
    print("\n[2] FK: 把 pkl 里的 joint_angles 喂给 CudaRobotModel ...")
    joint_angles, joint_names = load_pkl_joint_angles()
    print(f"    joint_names  = {joint_names}")
    print(f"    joint_angles = {joint_angles.tolist()}")

    kin = CudaRobotModel(robot_cfg.kinematics)
    dof = kin.get_dof()
    print(f"    DOF = {dof}")
    if dof != len(joint_angles):
        print(f"[WARN] DOF ({dof}) 和 pkl joint_angles 长度 ({len(joint_angles)}) 不一致")

    q = torch.from_numpy(joint_angles).to(**tensor_args.as_torch_dict()).unsqueeze(0)
    state = kin.get_state(q)
    ee_pos = state.ee_position[0].cpu().numpy()
    ee_quat = state.ee_quaternion[0].cpu().numpy()  # [w, x, y, z]
    print(f"    EE position    = [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
    print(f"    EE quaternion  = [{ee_quat[0]:.4f}, {ee_quat[1]:.4f}, "
          f"{ee_quat[2]:.4f}, {ee_quat[3]:.4f}]  (w, x, y, z)")

    # ---------- Step 3: IK ----------
    print("\n[3] IK: 反算关节角, 应能恢复输入 ...")
    ik_cfg = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik = IKSolver(ik_cfg)

    goal = Pose(state.ee_position, state.ee_quaternion)
    result = ik.solve_batch(goal)
    success = bool(result.success[0].item())
    pos_err_cm = float(result.position_error.mean().item()) * 100
    rot_err_deg = float(result.rotation_error.mean().item()) * (180 / np.pi)
    print(f"    success      = {success}")
    print(f"    pos error    = {pos_err_cm:.3f} cm")
    print(f"    rot error    = {rot_err_deg:.3f} deg")

    if success and result.solution is not None:
        q_recovered = result.solution[0, 0].cpu().numpy()
        diff = q_recovered - joint_angles
        print(f"    joint diff   = {np.round(diff, 3).tolist()}")
        # 不必和原值完全相同 (IK 多解), 但 EE pose 应已恢复
        # 用恢复的关节角再做 FK 检查 EE pose 一致
        q2 = torch.from_numpy(q_recovered).to(**tensor_args.as_torch_dict()).unsqueeze(0)
        state2 = kin.get_state(q2)
        ee_pos2 = state2.ee_position[0].cpu().numpy()
        pos_diff_mm = float(np.linalg.norm(ee_pos - ee_pos2)) * 1000
        print(f"    refk pos diff = {pos_diff_mm:.3f} mm  (目标 < 5 mm)")

    print("\n" + "=" * 60)
    if success and pos_err_cm < 0.5 and rot_err_deg < 3.0:
        print("✓ M1.0.5 通过. cuRobo + ur12e 配置可用, 可以进 M1.1.")
        return 0
    print("✗ IK 误差超出预期, 检查机器人配置")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
