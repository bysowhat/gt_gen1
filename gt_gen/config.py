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
    def voxel_size_m_coarse(self) -> float:
        """粗网格体素边长（米）：覆盖整个 ROI 的探索三态图。见 default.yaml roi.voxel_size_m_coarse。"""
        return float(self.raw["roi"]["voxel_size_m_coarse"])

    @property
    def voxel_size_m_fine(self) -> float:
        """细网格体素边长（米）：焊缝周围 fine 盒的高分辨率子网格。见 default.yaml roi.voxel_size_m_fine。"""
        return float(self.raw["roi"]["voxel_size_m_fine"])

    @property
    def fine_box_margin_m(self) -> float:
        """fine 盒相对焊缝 AABB 的各方向外扩量（米）。见 default.yaml roi.fine_box_margin_cm（方案定 10cm）。"""
        return float(self.raw.get("roi", {}).get("fine_box_margin_cm", 10.0)) / 100.0

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
        vs = self.voxel_size_m_coarse
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

    @property
    def init_free_box_min(self) -> list:
        """初始引导 FREE 空间（立方体法）：AABB 盒下界角点（米，base_link 系）。见 default.yaml init_free.box_min_m。"""
        return list(self.raw.get("init_free", {}).get("box_min_m", [-0.6, -1.3, -0.5]))

    @property
    def init_free_box_max(self) -> list:
        """初始引导 FREE 空间（立方体法）：AABB 盒上界角点（米，base_link 系）。见 default.yaml init_free.box_max_m。"""
        return list(self.raw.get("init_free", {}).get("box_max_m", [2.5, 1.3, 1.7]))

    @property
    def init_free_method_for_init(self) -> str:
        """机械臂初始 FREE 空间方案：'cylinder'=圆柱(cyl_*) | 'box'=立方体(box_*_for_init)。见 default.yaml init_free.method_for_init。"""
        return str(self.raw.get("init_free", {}).get("method_for_init", "cylinder")).lower()

    @property
    def init_free_box_min_for_init(self) -> list:
        """机械臂初始 FREE 空间（立方体法）：AABB 盒下界角点（米，base_link 系）。见 default.yaml init_free.box_min_m_for_init。"""
        return list(self.raw.get("init_free", {}).get("box_min_m_for_init", [-2.5, -1.3, -0.5]))

    @property
    def init_free_box_max_for_init(self) -> list:
        """机械臂初始 FREE 空间（立方体法）：AABB 盒上界角点（米，base_link 系）。见 default.yaml init_free.box_max_m_for_init。"""
        return list(self.raw.get("init_free", {}).get("box_max_m_for_init", [0.6, 1.3, 1.7]))

    @property
    def init_free_swept_path(self) -> str:
        """初始引导 FREE 空间（预存扫掠法）：预计算的整臂扫掠体素中心(.npz)路径。
        相对路径按项目根解析。见 default.yaml init_free.swept_path 与 scripts/viz_joints_open3d.py --save-init-free。"""
        p = self.raw.get("init_free", {}).get("swept_path", "configs/init_free_swept.npz")
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

    # ---- plan_init_pose（lookup 式初始位姿求解的关节角过滤范围） ----
    @property
    def plan_init_ee_xy_range(self) -> list:
        """焊枪末端在 base_link 系 xy 平面到原点距离的 [下限,上限]（米）。
        见 default.yaml plan_init_pose.ee_xy_range_m。"""
        return list(self.raw.get("plan_init_pose", {}).get("ee_xy_range_m", [0.4, 1.0]))

    @property
    def plan_init_ee_z_range(self) -> list:
        """焊枪末端在 base_link 系 z 的 [下限,上限]（米）。见 default.yaml plan_init_pose.ee_z_range_m。"""
        return list(self.raw.get("plan_init_pose", {}).get("ee_z_range_m", [-0.1, 0.1]))

    @property
    def plan_init_n_per_dof(self):
        """lookup 每关节采样档数。见 default.yaml plan_init_pose.n_per_dof。
        · 标量 → 各关节同档（q_table = n^6，旧行为）；
        · 列表 [n0..n5] → link1..link6 各自档数（q_table = ∏ ni），用于按关节调采样精度/显存。"""
        v = self.raw.get("plan_init_pose", {}).get("n_per_dof", 7)
        if isinstance(v, (list, tuple)):
            return [int(x) for x in v]
        return int(v)

    @property
    def plan_init_rot_x_deg(self) -> list:
        """绕 xiaoyu_tip_link 末端局部 +x 轴的旋转采样 [min,max,step]（度，闭区间）。
        见 default.yaml plan_init_pose.rot_x_deg。"""
        return list(self.raw.get("plan_init_pose", {}).get("rot_x_deg", [-30.0, 30.0, 15.0]))

    @property
    def plan_init_rot_y_deg(self) -> list:
        """绕 xiaoyu_tip_link 末端局部 +y 轴的旋转采样 [min,max,step]（度，闭区间）。
        见 default.yaml plan_init_pose.rot_y_deg。"""
        return list(self.raw.get("plan_init_pose", {}).get("rot_y_deg", [-30.0, 30.0, 15.0]))

    @property
    def plan_init_rot_z_deg(self) -> list:
        """绕 xiaoyu_tip_link 末端局部 +z 轴（焊枪自转 roll）的旋转采样 [min,max,step]（度，闭区间）。
        见 default.yaml plan_init_pose.rot_z_deg。"""
        return list(self.raw.get("plan_init_pose", {}).get("rot_z_deg", [0.0, 0.0, 1.0]))

    @property
    def plan_init_collision_tolerance(self) -> float:
        """整臂/retract 碰撞球对工件 ESDF 的允许穿透阈值（米，d<=tol 判 safe）。
        见 default.yaml plan_init_pose.collision_tolerance_m。"""
        return float(self.raw.get("plan_init_pose", {}).get("collision_tolerance_m", 0.03))

    @property
    def plan_init_voxel_size(self) -> float:
        """工件 ESDF 体素大小（米）。见 default.yaml plan_init_pose.voxel_size_m。"""
        return float(self.raw.get("plan_init_pose", {}).get("voxel_size_m", 0.02))

    @property
    def seam_min_length_m(self) -> float:
        """焊缝长度过滤阈值（米）：corrected_p0↔p1 直线距离小于此值的焊缝在 Scene._load_seam 丢弃。
        见 default.yaml seam.min_length_cm。"""
        return float(self.raw.get("seam", {}).get("min_length_cm", 3.0)) / 100.0

    @property
    def plan_init_n_seg(self) -> int:
        """save_seam_pkl 的 seam_line 插值点数。见 default.yaml plan_init_pose.n_seg。"""
        return int(self.raw.get("plan_init_pose", {}).get("n_seg", 20))

    @property
    def plan_init_joint_table_path(self) -> str:
        """precompute_joint_table 结果（q_table/ee_pos/ee_x/ee_rot/link_spheres/retract_spheres 等）
        的存储路径（.pt）。相对路径按项目根解析。见 default.yaml plan_init_pose.joint_table_path。"""
        p = self.raw.get("plan_init_pose", {}).get(
            "joint_table_path", "configs/plan_init_joint_table.pt")
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

    @property
    def plan_init_standoff(self) -> float:
        """初始位姿求解：焊枪头落点沿焊缝角平分线(bisector，远离工件方向)外移的 standoff 距离（米）。
        焊枪头不再落在焊缝中点，而是落在「中点 + standoff·bisector」处（中点退后 standoff 进入自由空间）。
        与 goal.standoff_cm 同义但作用于初始位姿求解。见 default.yaml plan_init_pose.standoff_cm。"""
        return float(self.raw.get("plan_init_pose", {}).get("standoff_cm", 0.0)) / 100.0

    @property
    def plan_init_clearance_inflate(self) -> float:
        """间隙膨胀量（米）：把所有碰撞球半径膨胀此值再判碰，让整臂离工件留间隙；0=关闭。
        d=get_collision_distance 只返回穿透代价(>=0)，不返回负间隙，故不能用负 collision_tolerance。
        见 default.yaml plan_init_pose.clearance_inflate_m。"""
        return float(self.raw.get("plan_init_pose", {}).get("clearance_inflate_m", 0.0))

    @property
    def plan_init_snap_deg(self) -> float:
        """候选朝向 vs 4 种允许朝向：xyz 逐轴旋转误差上限（度），三轴全在内才保留并 snap。见 default.yaml plan_init_pose.snap_deg。"""
        return float(self.raw.get("plan_init_pose", {}).get("snap_deg", 10.0))

    @property
    def plan_init_base_overlap_filter(self) -> bool:
        """是否启用「固定底座(xiaoyu_base_link)碰撞球 vs 工件 在 base-xy 投影相交」过滤
        （相交=机械臂底座压在工件下/工件盖在底座上，丢弃该候选）。true=做该过滤（默认）；false=关闭。
        见 default.yaml plan_init_pose.base_overlap_filter。"""
        return bool(self.raw.get("plan_init_pose", {}).get("base_overlap_filter", True))

    @property
    def plan_init_workpiece_x_min(self) -> float:
        """工件距 base_link 原点【欧氏最近】的那个点，其 base-x 分量的下限（米）：须 > 此值否则丢弃
        （工件离底座 x 向太近/在底座后方）。见 default.yaml plan_init_pose.workpiece_x_min_m。"""
        return float(self.raw.get("plan_init_pose", {}).get("workpiece_x_min_m", 0.3))

    @property
    def plan_init_workpiece_x_voxel(self) -> float:
        """「工件距 base 欧氏最近点 x」过滤的点集体素粒度（米）：把工件表面按此 pitch 体素化，
        用体素中心当稠密点集（与三角形大小解耦，大三角形也不漏采）。与碰撞 ESDF 的 voxel_size_m
        无关、互不影响。见 default.yaml plan_init_pose.workpiece_x_voxel_m。"""
        return float(self.raw.get("plan_init_pose", {}).get("workpiece_x_voxel_m", 0.03))

    @property
    def plan_init_arm_collision_recheck(self) -> bool:
        """snap 后是否复检：候选 R snap 到 90°整倍朝向、重算 t 会改变工件姿态，snap 前的碰撞过滤此时已失效。
        true=对最终 (Rv,t_new) 再查一次「retract 姿态整臂碰撞球 vs 工件 ESDF」，撞则丢弃（判据同 solve：
        含 clearance_inflate、d<=collision_tolerance 判 safe）；false=关闭。
        见 default.yaml plan_init_pose.arm_collision_recheck。"""
        return bool(self.raw.get("plan_init_pose", {}).get("arm_collision_recheck", True))

    # ---- plan_init_pose_kejian（新逻辑：固定朝向 + 平移网格 + STOMP 可达，见 scripts/plan_init_pose_kejian.py） ----
    @property
    def _kejian(self) -> dict:
        return self.raw.get("plan_init_pose_kejian", {})

    @property
    def plan_init_kejian_ee_xy_range(self) -> list:
        """焊缝点在 base_link 系 xy 平面到原点距离的 [下限,上限]（米）。见 plan_init_pose_kejian.ee_xy_range_m。"""
        return list(self._kejian.get("ee_xy_range_m", [0.4, 1.0]))

    @property
    def plan_init_kejian_ee_z_range(self) -> list:
        """焊缝点在 base_link 系 z 的 [下限,上限]（米）。见 plan_init_pose_kejian.ee_z_range_m。"""
        return list(self._kejian.get("ee_z_range_m", [-0.1, 0.1]))

    @property
    def plan_init_kejian_standoff(self) -> float:
        """goal 落点沿 bisector（远离工件方向）外移的 standoff 距离（米）。见 plan_init_pose_kejian.standoff_cm。"""
        return float(self._kejian.get("standoff_cm", 0.0)) / 100.0

    @property
    def plan_init_kejian_xyz_step(self) -> list:
        """工件平移网格步长 [dx,dy,dz]（米，base 系）。见 plan_init_pose_kejian.xyz_step_m。"""
        return list(self._kejian.get("xyz_step_m", [0.15, 0.15, 0.05]))

    @property
    def plan_init_kejian_rot_x_deg(self) -> list:
        """goal 绕末端局部 +x 轴（焊枪自转 roll）采样 [min,max,step]（度）。见 plan_init_pose_kejian.rot_x_deg。"""
        return list(self._kejian.get("rot_x_deg", [-180.0, 180.0, 30.0]))

    @property
    def plan_init_kejian_rot_y_deg(self) -> list:
        """goal 绕末端局部 +y 轴（tilt）采样 [min,max,step]（度）。见 plan_init_pose_kejian.rot_y_deg。"""
        return list(self._kejian.get("rot_y_deg", [-30.0, 30.0, 15.0]))

    @property
    def plan_init_kejian_rot_z_deg(self) -> list:
        """goal 绕末端局部 +z 轴（tilt）采样 [min,max,step]（度）。见 plan_init_pose_kejian.rot_z_deg。"""
        return list(self._kejian.get("rot_z_deg", [-30.0, 30.0, 15.0]))

    @property
    def plan_init_kejian_ik_return_seeds(self) -> int:
        """预筛阶段每个 goal 朝向 IK 取回的候选解数。见 plan_init_pose_kejian.ik_return_seeds。"""
        return int(self._kejian.get("ik_return_seeds", 30))

    @property
    def plan_init_kejian_ik_num_seeds(self) -> int:
        """预筛专用 IK 的并行随机起点数（须 ≥ ik_return_seeds）。见 plan_init_pose_kejian.ik_num_seeds。
        不复用 planner.ik_num_seeds(200)：批量 IK 峰值显存≈ik_batch×num_seeds，200 会 OOM。"""
        return int(self._kejian.get("ik_num_seeds", 60))

    @property
    def plan_init_kejian_ik_batch(self) -> int:
        """预筛批量 IK 的分块大小：每次送多少个 goal 朝向进 solve_batch（峰值显存≈ik_batch×ik_num_seeds）。
        见 plan_init_pose_kejian.ik_batch。"""
        return int(self._kejian.get("ik_batch", 24))

    # ---- plan_init_pose_kejian2（lookup n^6 关节角采样 + 允许朝向 snap + 正反手分类，见 scripts/plan_init_pose_kejian2.py） ----
    @property
    def _kejian2(self) -> dict:
        return self.raw.get("plan_init_pose_kejian2", {})

    @property
    def plan_init_kejian2_ee_xy_range(self) -> list:
        """焊缝点在 base_link 系 xy 平面到原点距离的 [下限,上限]（米）。见 plan_init_pose_kejian2.ee_xy_range_m。"""
        return list(self._kejian2.get("ee_xy_range_m", [0.4, 1.0]))

    @property
    def plan_init_kejian2_ee_z_range(self) -> list:
        """焊缝点在 base_link 系 z 的 [下限,上限]（米）。见 plan_init_pose_kejian2.ee_z_range_m。"""
        return list(self._kejian2.get("ee_z_range_m", [-0.1, 0.1]))

    @property
    def plan_init_kejian2_n_per_dof(self):
        """lookup 每关节采样档数。见 plan_init_pose_kejian2.n_per_dof。
        · 标量 → 各关节同档（q_table = n^6，旧行为）；
        · 列表 [n0..n5] → link1..link6 各自档数（q_table = ∏ ni），用于按关节调采样精度/显存。"""
        v = self._kejian2.get("n_per_dof", 12)
        if isinstance(v, (list, tuple)):
            return [int(x) for x in v]
        return int(v)

    @property
    def plan_init_kejian2_collision_tolerance(self) -> float:
        """整臂/retract 碰撞球对工件 ESDF 的允许穿透阈值（米，d<=tol 判 safe）。见 plan_init_pose_kejian2.collision_tolerance_m。"""
        return float(self._kejian2.get("collision_tolerance_m", 0.0))

    @property
    def plan_init_kejian2_clearance_inflate(self) -> float:
        """间隙膨胀量（米）：把所有碰撞球半径膨胀此值再判碰，让整臂离工件留间隙；0=关闭。
        d=get_collision_distance 只返回穿透代价(>=0)，不返回负间隙，故不能用负 collision_tolerance。
        见 plan_init_pose_kejian2.clearance_inflate_m。"""
        return float(self._kejian2.get("clearance_inflate_m", 0.0))

    @property
    def plan_init_kejian2_voxel_size(self) -> float:
        """工件 ESDF 体素大小（米）。见 plan_init_pose_kejian2.voxel_size_m。"""
        return float(self._kejian2.get("voxel_size_m", 0.01))

    @property
    def plan_init_kejian2_standoff(self) -> float:
        """goal 落点沿 bisector（远离工件方向）外移的 standoff 距离（米）。见 plan_init_pose_kejian2.standoff_cm。"""
        return float(self._kejian2.get("standoff_cm", 0.0)) / 100.0

    @property
    def plan_init_kejian2_rot_x_deg(self) -> list:
        """lookup 绕末端局部 +x 轴（焊枪自转 roll）采样 [min,max,step]（度）。见 plan_init_pose_kejian2.rot_x_deg。"""
        return list(self._kejian2.get("rot_x_deg", [-180.0, 180.0, 30.0]))

    @property
    def plan_init_kejian2_rot_y_deg(self) -> list:
        """lookup 绕末端局部 +y 轴（tilt）采样 [min,max,step]（度）。见 plan_init_pose_kejian2.rot_y_deg。"""
        return list(self._kejian2.get("rot_y_deg", [-30.0, 30.0, 15.0]))

    @property
    def plan_init_kejian2_rot_z_deg(self) -> list:
        """lookup 绕末端局部 +z 轴（tilt）采样 [min,max,step]（度）。见 plan_init_pose_kejian2.rot_z_deg。"""
        return list(self._kejian2.get("rot_z_deg", [-30.0, 30.0, 15.0]))

    @property
    def plan_init_kejian2_snap_deg(self) -> float:
        """候选朝向 vs 4 种允许朝向：xyz 逐轴旋转误差上限（度），三轴全在内才保留并 snap。见 plan_init_pose_kejian2.snap_deg。"""
        return float(self._kejian2.get("snap_deg", 10.0))

    @property
    def plan_init_kejian2_joint_table_path(self) -> str:
        """lookup 关节表缓存路径（.pt）。相对路径按项目根解析。见 plan_init_pose_kejian2.joint_table_path。"""
        p = self._kejian2.get("joint_table_path", "configs/plan_init_kejian2_joint_table.pt")
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

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
        """sync_collision_world 把非 FREE 障碍向 FREE 膨胀的体素层数（保守余量，见 default.yaml）。

        当前约定必须 =0（不膨胀）：膨胀单位是体素层数，多分辨率下同「膨 1 层」在粗/细网格
        物理厚度不同（4cm vs 1cm），语义不一致；且细化已把 GT 闸门的 √3/2·vs 过近似缝收窄，
        膨胀需求随之消失。若要重新启用，先想清楚多分辨率下每张网格各膨几层。
        """
        v = int(self.raw.get("planner", {}).get("voxel_inflate_voxels", 0))
        assert v == 0, (
            f"voxel_inflate_voxels 当前必须为 0，读到 {v}；"
            f"膨胀在多分辨率下语义不一致（粗网格膨 4cm、细网格膨 1cm），已停用")
        return v

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
        """障碍物类型1（走廊自动放障碍）参数段，= obstacle_placement.type1（单一来源）。

        兼容旧结构与外部覆盖：obstacle_placement 顶层若直接写了 type1 参数（如 place_obstacles.py
        往 cfg.raw['obstacle_placement']['key_links'] 写覆盖），这些顶层键会盖过 type1 段里的同名键。
        缺省给出与 default.yaml 一致的兜底。"""
        op_raw = dict(self.raw.get("obstacle_placement", {}))
        op = dict(op_raw.get("type1", {}))
        # 顶层直接写的（非 type* 段）参数覆盖 type1（保留 place_obstacles.py 等旧覆盖路径）
        for k, v in op_raw.items():
            if k not in ("type1", "type2", "type3"):
                op[k] = v
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

    @property
    def obstacle_placement_type2(self) -> dict:
        """障碍物类型2（焊缝旁遮挡板候选）参数段 = obstacle_placement.type2（单一来源）。
        供 Scene.add_obstacle_type2 读取；缺省与 default.yaml 一致（示例见
        scripts/viz_seam_plate_candidates_isaacsim.py）。"""
        p = dict(self.raw.get("obstacle_placement", {}).get("type2", {}))
        p.setdefault("shape", "plate")
        p.setdefault("n_cm", 10.0)
        p.setdefault("width_cm", 30.0)
        p.setdefault("length_pct", 100.0)
        p.setdefault("length_min_cm", 3.0)
        p.setdefault("thickness_cm", 2.0)
        p.setdefault("seed", 0)
        return p

    @property
    def obstacle_placement_type3(self) -> dict:
        """障碍物类型3（把焊缝包住的开口障碍）参数段 = obstacle_placement.type3（单一来源）。
        供 Scene.add_obstacle_type3 读取；缺省与 default.yaml 一致（示例见
        scripts/viz_seam_open_box_isaacsim.py）。"""
        p = dict(self.raw.get("obstacle_placement", {}).get("type3", {}))
        p.setdefault("obstacle", "open_box")
        p.setdefault("dis_cm", [100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        p.setdefault("wall_cm", 2.0)
        return p



def load_config(path: "str | Config | None" = DEFAULT_CONFIG) -> Config:
    """加载配置。path 可为：yaml 路径(str) / 已构造的 Config 实例（原样返回） / None（取默认）。"""
    if isinstance(path, Config):
        return path
    if path is None:
        path = DEFAULT_CONFIG
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    with open(raw["robot"]["cfg_path"], "r") as f:
        robot_cfg = yaml.safe_load(f)
    return Config(raw=raw, robot_cfg=robot_cfg)
