"""配置加载：合并本项目 configs/default.yaml 与 cuRobo 机器人 cfg。"""
from __future__ import annotations

import os
from dataclasses import dataclass

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "default.yaml")


@dataclass
class Config:
    raw: dict          # 本项目 default.yaml
    robot_cfg: dict    # cuRobo 机器人 cfg（含 robot_cfg 顶层键）

    # ---- 机器人（从 cuRobo cfg 读，避免重复） ----
    @property
    def _kin(self) -> dict:
        return self.robot_cfg["robot_cfg"]["kinematics"]

    @property
    def robot_cfg_path(self) -> str:
        return self.raw["robot"]["cfg_path"]

    @property
    def ee_link(self) -> str:
        return self._kin["ee_link"]

    @property
    def base_link(self) -> str:
        return self._kin["base_link"]

    @property
    def joint_names(self) -> list:
        return self._kin["cspace"]["joint_names"]

    @property
    def retract_config(self) -> list:
        return self._kin["cspace"]["retract_config"]

    @property
    def collision_link_names(self) -> list:
        return self._kin["collision_link_names"]

    # ---- 其它决策 ----
    @property
    def constraint_scope(self) -> str:
        return self.raw["constraint"]["scope"]

    @property
    def max_depth_m(self) -> float:
        return float(self.raw["sensor"]["max_depth_m"])

    @property
    def camera(self) -> dict:
        return self.raw["sensor"]["camera"]

    @property
    def voxel_size_m(self) -> float:
        return float(self.raw["roi"]["voxel_size_m"])

    @property
    def roi_center(self) -> list:
        """ROI 盒中心（base 系，米）——单一 ROI 来源。"""
        return list(self.raw["roi"]["center"])

    @property
    def roi_dims(self) -> list:
        """ROI 盒尺寸（米）——单一 ROI 来源。"""
        return list(self.raw["roi"]["dims"])

    @property
    def roi_expand_m(self) -> float:
        return float(self.raw["roi"].get("expand_m", 0.0))

    @property
    def init_free_dq(self) -> float:
        """初始引导 FREE 空间（关节扫掠法）：retract 各关节活动半幅（rad）。见 docs/initial-free-space.md。"""
        return float(self.raw.get("init_free", {}).get("dq_rad", 0.10))

    @property
    def init_free_cyl_radius(self) -> float:
        """初始引导 FREE 空间（圆柱体法）：圆柱半径（米）。见 docs/initial-free-space.md。"""
        return float(self.raw.get("init_free", {}).get("cyl_radius_m", 0.60))

    @property
    def init_free_cyl_height(self) -> float:
        """初始引导 FREE 空间（圆柱体法）：圆柱高度（米，从 base_link 平面 z=0 往上）。"""
        return float(self.raw.get("init_free", {}).get("cyl_height_m", 1.50))

    # ---- cuRobo IK / 规划 ----
    @property
    def ik_num_seeds(self) -> int:
        return int(self.raw.get("planner", {}).get("ik_num_seeds", 100))

    @property
    def ik_return_seeds(self) -> int:
        return int(self.raw.get("planner", {}).get("ik_return_seeds", 100))

    @property
    def position_threshold(self) -> float:
        return float(self.raw.get("planner", {}).get("position_threshold", 0.05))

    @property
    def rotation_threshold(self) -> float:
        return float(self.raw.get("planner", {}).get("rotation_threshold", 0.5))

    @property
    def drop_collision_links(self) -> list:
        # 空列表/缺省/null 都表示"一个都不 drop"
        return list(self.raw.get("planner", {}).get("drop_collision_links") or [])

    @property
    def params(self) -> dict:
        return self.raw.get("params", {})


def load_config(path: str = DEFAULT_CONFIG) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    with open(raw["robot"]["cfg_path"], "r") as f:
        robot_cfg = yaml.safe_load(f)
    return Config(raw=raw, robot_cfg=robot_cfg)
