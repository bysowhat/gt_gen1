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
        """ROI 盒尺寸（米）——单一 ROI 来源。

        每轴吸附到 voxel_size 的整数倍后返回。原因：cuRobo voxel 碰撞核（CUDA）用
        robust_floor(dim/voxel)+1 现算栅格维度，与 Python 端 get_grid_shape 在【半体素】
        （dim/voxel 落在 x.5）时取整不一致（CUDA round-half-away 保留、Python floor），
        会让该轴的体素索引步长差 1 → 占据沿该轴整体错位/抹开 → 规划起点假碰撞。
        取整数倍可彻底规避（见 docs/step10-precise-obstacle-notes / diag）。
        """
        vs = self.voxel_size_m
        return [int(round(float(d) / vs)) * vs for d in self.raw["roi"]["dims"]]


    @property
    def roi_expand_m(self) -> float:
        return float(self.raw["roi"].get("expand_m", 0.0))

    @property
    def goal_standoff_m(self) -> float:
        """末端目标沿 bisector 后退的 standoff 距离（米）。见 default.yaml goal.standoff_cm。"""
        return float(self.raw.get("goal", {}).get("standoff_cm", 0.0)) / 100.0

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
        """初始引导 FREE 空间（圆柱体法）：圆柱高度（米，从底面 cyl_z_min 往上）。"""
        return float(self.raw.get("init_free", {}).get("cyl_height_m", 1.50))

    @property
    def init_free_cyl_z_min(self) -> float:
        """初始引导 FREE 空间（圆柱体法）：圆柱底面 z（米，base 系）。默认 -0.02 以盖住固定底座
        碰撞球扎到 z<0 的那层体素（实测最低球点 z≈-0.002m，体素中心 -0.01m），避免每段运动假阳性非 FREE。"""
        return float(self.raw.get("init_free", {}).get("cyl_z_min_m", 0.0))

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
    def plan_max_attempts(self) -> int:
        """plan_on_truth / plan_to_pose 的最大规划尝试次数（planner.max_attempts，单一来源）。"""
        return int(self.raw.get("planner", {}).get("max_attempts", 20))

    @property
    def drop_collision_links(self) -> list:
        # 空列表/缺省/null 都表示"一个都不 drop"
        return list(self.raw.get("planner", {}).get("drop_collision_links") or [])

    @property
    def voxel_inflate_voxels(self) -> int:
        """sync_collision_world 把非 FREE 障碍向 FREE 膨胀的体素层数（保守余量，见 default.yaml）。"""
        return int(self.raw.get("planner", {}).get("voxel_inflate_voxels", 1))

    @property
    def num_trajopt_seeds(self) -> int:
        """trajopt 并行优化的轨迹起点数（MotionGenConfig，init 时定）。越大越稳，GPU 并行几乎不加时。

        注：plan_single* 路径下 graph planner 的并行种子数也取此值——cuRobo 把 graph seeds
        硬绑到 trajopt seeds（无独立 num_graph_seeds 旋钮，传了也被忽略），故想加 graph 种子调这个。
        """
        return int(self.raw.get("planner", {}).get("num_trajopt_seeds", 12))

    @property
    def enable_graph(self) -> bool:
        """规划是否先跑 graph planner 找全局可行折线再 trajopt 平滑（窄通道/绕行成功率↑，更慢）。"""
        return bool(self.raw.get("planner", {}).get("enable_graph", True))

    @property
    def time_dilation_factor(self) -> float:
        """轨迹时间放慢系数 ∈(0,1]：<1 把速度/加速度上限按比例缩小，松动力学约束→成功率↑（轨迹更慢）。"""
        return float(self.raw.get("planner", {}).get("time_dilation_factor", 0.5))

    # ---- 规划后端（curobo | stomp，见 gt_gen/stomp_iface.py） ----
    @property
    def planner_backend(self) -> str:
        """obstacle_placement 的「默认轨迹/绕行解」用哪种规划器：curobo(MotionGen) | stomp(stomp_planner)。
        碰撞判定 check_state 不受此影响（恒用 cuRobo handle）。"""
        return str(self.raw.get("planner", {}).get("backend", "curobo")).lower()

    @property
    def stomp_params(self) -> dict:
        """STOMP 旋钮（planner.stomp 段，单一来源；backend=stomp 时透传 stomp_planning_api）。缺省给兜底。"""
        sp = dict(self.raw.get("planner", {}).get("stomp", {}))
        # STOMP 规划核心已 vendoring 到本项目 stomp_planner/（原参考项目 gt_overall 的副本）。
        # 默认指向项目内目录，与机器无关；yaml 写 stomp_planner_dir 或 env STOMP_PLANNER_DIR 可覆盖。
        sp.setdefault("stomp_planner_dir", os.path.join(PROJECT_ROOT, "stomp_planner"))
        sp.setdefault("num_iterations", 120)
        sp.setdefault("num_batch", 8)
        sp.setdefault("num_timesteps", 51)
        sp.setdefault("delta_t", 0.1)
        sp.setdefault("collision_weight", 80.0)
        sp.setdefault("buffer_m", 0.001)
        sp.setdefault("voxel_world", "mesh")     # 兜底=mesh(保持原行为)；yaml 可设 cuboid
        sp.setdefault("local_box_m", 2.0)        # cuboid 模式：base 原点为心、半边长(米)的转换盒
        return sp

    @property
    def params(self) -> dict:
        return self.raw.get("params", {})

    # ---- 障碍物自动放置（见 docs/障碍物位置.md, gt_gen/obstacle_placement.py） ----
    @property
    def obstacle_placement(self) -> dict:
        """obstacle_placement 段（单一来源）。缺省给出与 default.yaml 一致的兜底。"""
        op = dict(self.raw.get("obstacle_placement", {}))
        op.setdefault("max_per_scene", 1)
        op.setdefault("max_attempts", 20)
        op.setdefault("detour_max_attempts", 30)
        op.setdefault("key_links",
                      ["Link2", "Link3", "Link4", "Link5", "Link6", "xiaoyu_accessory_link"])
        op.setdefault("pos_t_window", [0.25, 0.85])
        op.setdefault("goal_clearance_m", 0.25)
        op.setdefault("size_scale_range", [0.6, 1.6])
        op.setdefault("angle_jitter_deg", 30.0)
        op.setdefault("pos_jitter_m", 0.10)
        op.setdefault("detour_min_joint_rad", 0.30)
        op.setdefault("detour_ik_position_threshold", 0.005)
        op.setdefault("detour_ik_rotation_threshold", 0.05)
        op.setdefault("obstacle_types", [])
        op.setdefault("span_clip_m", [0.15, 1.2])
        op.setdefault("tube_r_clip_m", [0.04, 0.20])
        op.setdefault("thickness_range_m", [0.02, 0.05])
        return op



def load_config(path: str = DEFAULT_CONFIG) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    with open(raw["robot"]["cfg_path"], "r") as f:
        robot_cfg = yaml.safe_load(f)
    return Config(raw=raw, robot_cfg=robot_cfg)
