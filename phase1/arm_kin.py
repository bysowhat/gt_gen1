"""M1.4 cuRobo IK + 整臂碰撞封装.

主接口 ArmKin:
    fk(theta) → ee 位姿 (base 和 world 两套)
    ik(pos_world, quat_world) → 关节角 (世界系输入)
    update_world(voxel_map)   → 把 occupied 体素喂给 cuRobo 当障碍
    collides(theta)            → 整臂碰撞检测 (vs 当前 world)
    edge_free(theta_a, theta_b) → 两构型间直线插值都不撞
    trajectory_cost(theta_a, theta_b) → 关节空间 L2 距离

约定:
    - 用户面对 world 坐标系 (跟 pkl 里 robot_pose, seam_line 一致)
    - cuRobo 内部用 base 坐标系 (robot URDF 的 base_link)
    - 通过 robot_pose_world (4x4) 在两者间转换
    - 四元数顺序: [w, x, y, z]
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRot

# Compat shim: cuRobo 期望 warp 1.x 的 wp.torch.* 子模块, 但 warp 1.13 把这些
# 函数提到顶层并删了 wp.torch. 这里补一个 SimpleNamespace 让 cuRobo 能跑.
import warp as _wp
if not hasattr(_wp, "torch"):
    import types as _types
    _wp.torch = _types.SimpleNamespace(
        device_from_torch=_wp.device_from_torch,
        device_to_torch=_wp.device_to_torch,
        from_torch=_wp.from_torch,
        to_torch=_wp.to_torch,
        stream_from_torch=_wp.stream_from_torch,
        stream_to_torch=_wp.stream_to_torch,
        dtype_from_torch=_wp.dtype_from_torch,
        dtype_to_torch=_wp.dtype_to_torch,
    )

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

if TYPE_CHECKING:
    from phase1.mapping import VoxelMap


# ---------------------------------------------------------------------- quat utils (numpy)


def _matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 → [w, x, y, z]."""
    q_xyzw = ScipyRot.from_matrix(R).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """[w, x, y, z] → conjugate (单位四元数等于 inverse)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product: [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


# ---------------------------------------------------------------------- ArmKin


class ArmKin:
    """cuRobo wrapper. 一次创建反复用; voxel map 变了调 update_world."""

    def __init__(
        self,
        robot_yml_path: str,
        robot_pose_world: np.ndarray | None = None,
        ik_num_seeds: int = 20,
        ik_pos_threshold: float = 0.005,
        ik_rot_threshold: float = 0.05,
    ):
        self.tensor_args = TensorDeviceType()
        yml = load_yaml(robot_yml_path)
        self._yml = yml
        self.robot_cfg = RobotConfig.from_dict(yml["robot_cfg"], self.tensor_args)
        self.kin = CudaRobotModel(self.robot_cfg.kinematics)
        self.dof = self.kin.get_dof()
        self.joint_names = list(yml["robot_cfg"]["kinematics"]["cspace"]["joint_names"]) \
            if "cspace" in yml["robot_cfg"]["kinematics"] else None

        # world ↔ base 变换 (robot_pose_world: base 在 world 里的位姿)
        self.robot_pose_world = (np.eye(4, dtype=np.float64) if robot_pose_world is None
                                  else np.asarray(robot_pose_world, dtype=np.float64))
        if self.robot_pose_world.shape != (4, 4):
            raise ValueError(f"robot_pose_world must be (4,4), got "
                             f"{self.robot_pose_world.shape}")
        self._R_wb = self.robot_pose_world[:3, :3]            # base → world
        self._t_wb = self.robot_pose_world[:3, 3]
        self._R_bw = self._R_wb.T                              # world → base
        self._t_bw = -self._R_bw @ self._t_wb
        self._q_wb = _matrix_to_quat_wxyz(self._R_wb)         # 四元数版本
        self._q_bw = _quat_conj(self._q_wb)

        # IK solver (无 world 碰撞; collides() 用 RobotWorld 单独检查)
        # use_cuda_graph=False: 允许动态 batch 大小 (M1.7 候选数会变)
        ik_cfg = IKSolverConfig.load_from_robot_config(
            self.robot_cfg, None,
            rotation_threshold=ik_rot_threshold,
            position_threshold=ik_pos_threshold,
            num_seeds=ik_num_seeds,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=False,
        )
        self.ik_solver = IKSolver(ik_cfg)

        # RobotWorld (含碰撞) — 初始空 world
        empty_world = {"cuboid": {}}
        rw_cfg = RobotWorldConfig.load_from_config(
            self.robot_cfg, empty_world,
            collision_activation_distance=0.0,
        )
        self.rw = RobotWorld(rw_cfg)
        self._world: WorldConfig = WorldConfig(cuboid=[])

    # ------------------------------------------------------------------ FK

    def fk(self, theta: np.ndarray | torch.Tensor) -> dict:
        """Forward kinematics.

        theta: (6,) 或 (N, 6)
        return: dict 含 ee_pos_base / ee_quat_base / ee_pos_world / ee_quat_world.
                所有都是 torch tensor on device.
        """
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, **self.tensor_args.as_torch_dict())
        else:
            theta = theta.to(**self.tensor_args.as_torch_dict())
        single = theta.dim() == 1
        if single:
            theta = theta.unsqueeze(0)

        state = self.kin.get_state(theta)
        ee_pos_base = state.ee_position                  # (N, 3)
        ee_quat_base = state.ee_quaternion               # (N, 4) [w,x,y,z]

        # base → world
        R_wb = torch.tensor(self._R_wb, **self.tensor_args.as_torch_dict())
        t_wb = torch.tensor(self._t_wb, **self.tensor_args.as_torch_dict())
        ee_pos_world = (R_wb @ ee_pos_base.T).T + t_wb
        # quat: q_world = q_wb * q_base
        ee_quat_world = self._quat_mul_torch(
            torch.tensor(self._q_wb, **self.tensor_args.as_torch_dict()).unsqueeze(0),
            ee_quat_base,
        )
        return {
            "ee_pos_base": ee_pos_base,
            "ee_quat_base": ee_quat_base,
            "ee_pos_world": ee_pos_world,
            "ee_quat_world": ee_quat_world,
        }

    @staticmethod
    def _quat_mul_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Batch quat mul, [w,x,y,z]. q1, q2 broadcastable to (N, 4)."""
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=-1)

    # ------------------------------------------------------------------ IK

    def ik(
        self,
        ee_pos_world: np.ndarray,
        ee_quat_world: np.ndarray,
    ) -> dict:
        """Inverse kinematics. 输入 world 系目标位姿.

        ee_pos_world:  (3,) 或 (N, 3)
        ee_quat_world: (4,) 或 (N, 4) [w, x, y, z]
        return: dict {
            'success':   (N,) bool tensor (CPU)
            'theta':     (N, dof) tensor on device, 失败的位置填 NaN
            'pos_error': (N,) m
            'rot_error': (N,) rad
        }
        """
        pos = np.atleast_2d(np.asarray(ee_pos_world, dtype=np.float64))
        quat = np.atleast_2d(np.asarray(ee_quat_world, dtype=np.float64))
        if pos.shape[-1] != 3 or quat.shape[-1] != 4:
            raise ValueError(f"ee_pos shape {pos.shape}, ee_quat shape {quat.shape}")
        N = pos.shape[0]
        if quat.shape[0] != N:
            raise ValueError("pos and quat batch sizes mismatch")

        # world → base
        pos_base = (self._R_bw @ pos.T).T + self._t_bw                  # (N, 3)
        quat_base = np.stack([_quat_mul(self._q_bw, q) for q in quat])   # (N, 4)

        pos_t = torch.tensor(np.ascontiguousarray(pos_base),
                             **self.tensor_args.as_torch_dict()).contiguous()
        quat_t = torch.tensor(np.ascontiguousarray(quat_base),
                              **self.tensor_args.as_torch_dict()).contiguous()
        goal = Pose(pos_t, quat_t)
        result = self.ik_solver.solve_batch(goal)

        success = result.success.detach().cpu().reshape(N).contiguous()  # (N,)
        sol = result.solution.detach()
        if sol.dim() == 3:
            sol = sol[:, 0, :]                                            # 取第一个解
        # IK 全失败时 result.solution 形状可能是 (N, 0, dof), 需保护
        if sol.shape[0] != N or sol.dim() != 2 or sol.shape[1] != self.dof:
            sol = torch.full((N, self.dof), float("nan"),
                             **self.tensor_args.as_torch_dict())

        # 失败位置填 NaN
        theta = sol.clone()
        if (~success).any():
            theta[(~success).to(theta.device)] = float("nan")
        return {
            "success": success,
            "theta": theta,
            "pos_error": result.position_error.detach().cpu(),
            "rot_error": result.rotation_error.detach().cpu(),
        }

    # ------------------------------------------------------------------ world / collision

    def update_world(self, voxel_map: "VoxelMap") -> int:
        """把 occupied 体素打成 Cuboid list 喂给 cuRobo. 返回 cuboid 数."""
        from phase1.mapping import STATE_OCCUPIED

        occ_mask = voxel_map.state == STATE_OCCUPIED
        occ_idx = occ_mask.nonzero(as_tuple=False)                       # (M, 3)

        cubes: list[Cuboid] = []
        if len(occ_idx) > 0:
            # 体素中心 (world) → base
            centers_world = voxel_map.index_to_world(occ_idx).cpu().numpy()  # (M, 3)
            centers_base = (self._R_bw @ centers_world.T).T + self._t_bw

            cubes = [
                Cuboid(
                    name=f"vox_{i}",
                    pose=[float(c[0]), float(c[1]), float(c[2]), 1.0, 0.0, 0.0, 0.0],
                    dims=[voxel_map.res, voxel_map.res, voxel_map.res],
                )
                for i, c in enumerate(centers_base)
            ]

        # Workaround: cuRobo 的 _load_collision_model_in_cache 在 max_obb < 1 时
        # 不清旧 cuboid (early return). 总是塞一个远处的 1mm 占位 cuboid 避免.
        cubes.append(Cuboid(
            name="__placeholder_far_far_away__",
            pose=[1000.0, 1000.0, 1000.0, 1.0, 0.0, 0.0, 0.0],
            dims=[0.001, 0.001, 0.001],
        ))
        self._world = WorldConfig(cuboid=cubes)
        self.rw.update_world(self._world)
        return len(cubes) - 1   # 不算占位

    def collides(self, theta: torch.Tensor) -> torch.Tensor:
        """整臂碰撞检测. theta: (N, dof) 或 (dof,). 返回 (N,) bool tensor on device."""
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, **self.tensor_args.as_torch_dict())
        else:
            theta = theta.to(**self.tensor_args.as_torch_dict())
        single = theta.dim() == 1
        if single:
            theta = theta.unsqueeze(0)
        # cuRobo 期望 (batch, dof)
        d_world, d_self = self.rw.get_world_self_collision_distance_from_joints(theta)
        # cuRobo 的 distance: 大于 0 = 碰撞 (penetration), <= 0 = 安全.
        # d_world: (batch, n_links_or_spheres) 或 (batch,) 经过 squeeze
        # 取最严重的 sphere
        if d_world.dim() > 1:
            coll_world = (d_world > 0).any(dim=-1)
        else:
            coll_world = d_world > 0
        if d_self.dim() > 1:
            coll_self = (d_self > 0).any(dim=-1)
        else:
            coll_self = d_self > 0
        return coll_world | coll_self

    def edge_free(
        self,
        theta_a: torch.Tensor | np.ndarray,
        theta_b: torch.Tensor | np.ndarray,
        step_rad: float = 0.05,
    ) -> bool:
        """两构型间直线插值, 全部不撞才算 free. 返回 bool."""
        a = self._to_torch(theta_a)
        b = self._to_torch(theta_b)
        diff = b - a
        max_step = float(diff.abs().max().item())
        n_steps = max(2, int(np.ceil(max_step / step_rad)) + 1)
        ts = torch.linspace(0, 1, n_steps, **self.tensor_args.as_torch_dict())
        intermediates = a.unsqueeze(0) + ts.unsqueeze(1) * diff.unsqueeze(0)  # (n_steps, dof)
        coll = self.collides(intermediates)
        return not bool(coll.any().item())

    def trajectory_cost(
        self,
        theta_a: torch.Tensor | np.ndarray,
        theta_b: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """关节空间 L2 距离. 支持 (dof,) 单对或 (N, dof) 批."""
        a = self._to_torch(theta_a)
        b = self._to_torch(theta_b)
        if a.dim() == 1:
            return (b - a).norm()
        return (b - a).norm(dim=-1)

    def _to_torch(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(**self.tensor_args.as_torch_dict())
        return torch.tensor(x, **self.tensor_args.as_torch_dict())

    # ------------------------------------------------------------------ helpers

    def quat_world_to_base(self, q_world: np.ndarray) -> np.ndarray:
        return _quat_mul(self._q_bw, np.asarray(q_world).reshape(4))

    def quat_base_to_world(self, q_base: np.ndarray) -> np.ndarray:
        return _quat_mul(self._q_wb, np.asarray(q_base).reshape(4))

    def pos_world_to_base(self, pos_world: np.ndarray) -> np.ndarray:
        return self._R_bw @ np.asarray(pos_world).reshape(3) + self._t_bw

    def pos_base_to_world(self, pos_base: np.ndarray) -> np.ndarray:
        return self._R_wb @ np.asarray(pos_base).reshape(3) + self._t_wb


__all__ = ["ArmKin"]
