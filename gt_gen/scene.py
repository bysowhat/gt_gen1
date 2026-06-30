"""Scene —— 持有「当前 3D 世界 + 机械臂状态 + 焊缝信息」的状态类（API 重组核心）。

设计原则（见 docs/scene-api.md）：
  · 只做 **API 方向的结构包装**，不改各底层脚本/模块「本身在做的事」。
    例如 plan_init_pose() 只是把 scripts/plan_init_pose.py 的 InitPoseLookupSolver 求解流程
    包成 Scene 的一个方法（输入/输出/算法完全一致），不复制、不改动其几何逻辑。
  · 一个 Scene = 一条焊缝（weld_json 里的第 seam_id 条）。
  · 字段按「静态输入 / 世界状态 / 机械臂状态 / cuRobo 世界（惰性）」分组；本期只实现
    构造 + plan_init_pose，其余字段先占位（None），随后续 API 补充逐步填实。

本期已实现：
  · Scene(cfg, workpiece_obj, weld_json, seam_id=0, ...)           —— 构造
  · Scene.plan_init_pose(...)                                       —— 候选初始位姿求解

后续补充（占位）：apply_init_pose / goal / 障碍 / voxmap / 世界惰性构建 / 规划 / 导出 等。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from gt_gen.config import Config, load_config, PROJECT_ROOT

_SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")


def _load_plan_init_pose():
    """惰性导入 scripts/plan_init_pose.py 模块（不改它、原样复用其求解器与可视化）。

    该脚本顶层只 import 轻量库（argparse/os/numpy…），curobo/torch 都在函数内部惰性导入，
    故模块级 import 代价极小、可安全在 Scene 方法里按需调用。"""
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)
    import plan_init_pose  # noqa: E402  （scripts/plan_init_pose.py）
    return plan_init_pose


@dataclass
class InitPoseCandidate:
    """一个候选初始位姿 = 工件 mesh world ↔ 机械臂 base 的相对 pose (R, t)。

    约定：p_base = R · p_world + t（R,t 即 T_workpiece_in_base 的旋转/平移）。
    它定义一整套自洽的世界布局（工件在 base 系的摆放），**不是机械臂的起步关节角**
    （起步仍恒为 retract）；q 仅是「焊枪头落在该焊缝 standoff 落点」的可达参考构型。

    raw 保留 InitPoseLookupSolver 产出的原始解 dict，供可视化原样复用（见 SceneVisualizer）。
    """
    R: np.ndarray                  # (3,3) 工件→base 旋转
    t: np.ndarray                  # (3,)  工件→base 平移
    q: np.ndarray                  # (dof,) 可达该焊缝的关节角（参考/校验，非起步角）
    ee_pos_in_base: np.ndarray     # 焊枪头落点（= standoff 落枪点）在 base
    mid_in_base: np.ndarray        # 真实焊缝中点在 base（standoff>0 时 ≠ ee_pos_in_base）
    ee_x_in_base: np.ndarray       # 末端局部 +x 在 base
    rot_x_deg: float               # 绕末端局部 x/y/z 轴的扰动角（度）
    rot_y_deg: float
    rot_z_deg: float
    d_link: float                  # 整臂碰撞球对工件 ESDF 的最大穿透（<=tol 判 safe）
    d_retract: float               # retract 碰撞球同上
    align_score: float             # 三轴离 90° 整倍数偏差和（取负，越大越对齐）
    combined_score: float
    raw: dict = field(default=None, repr=False)   # 原始 solver 解（可视化复用）

    @classmethod
    def from_solution(cls, sol: dict) -> "InitPoseCandidate":
        """由 InitPoseLookupSolver.solve_one_weld_lookup 的单条解 dict 构造。"""
        return cls(
            R=np.asarray(sol["R"], dtype=np.float64),
            t=np.asarray(sol["t"], dtype=np.float64),
            q=np.asarray(sol["q"], dtype=np.float64),
            ee_pos_in_base=np.asarray(sol["ee_pos_in_base"], dtype=np.float64),
            mid_in_base=np.asarray(sol["mid_in_base"], dtype=np.float64),
            ee_x_in_base=np.asarray(sol["ee_x_in_base"], dtype=np.float64),
            rot_x_deg=float(sol["rot_x_deg"]),
            rot_y_deg=float(sol["rot_y_deg"]),
            rot_z_deg=float(sol["rot_z_deg"]),
            d_link=float(sol["d_link"]),
            d_retract=float(sol["d_retract"]),
            align_score=float(sol["align_score"]),
            combined_score=float(sol["combined_score"]),
            raw=sol,
        )

    def to_solution(self) -> dict:
        """还原成 InitPoseLookupSolver 解 dict 的形态（供 plan_init_pose 的可视化函数原样吃）。"""
        return self.raw

    @property
    def T_workpiece_in_base(self) -> np.ndarray:
        """4×4 齐次变换 T_workpiece_in_base（p_base = T · p_world）。"""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    @property
    def workpiece_pose7(self) -> np.ndarray:
        """工件在 base 系的 pose7 = [x,y,z, qw,qx,qy,qz]。"""
        pim = _load_plan_init_pose()
        return pim.mat44_to_pose7(self.T_workpiece_in_base)


class Scene:
    """当前世界 + 机械臂状态 + 焊缝信息（一个 Scene 绑定 weld_json 中的第 seam_id 条焊缝）。

    构造示例：
        scene = Scene(
            cfg="configs/default.yaml",
            workpiece_obj="/.../BEAM_..._part_watertight.obj",
            weld_json="/.../BEAM_..._weld_angle3.json",
            seam_id=0)
        cands = scene.plan_init_pose()      # 候选初始位姿（工件↔臂相对 pose）
        scene.init_pose                     # = cands[0]（best）

    cfg 接受 Config 实例 / yaml 路径 / None（None→load_config() 取默认）。
    """

    def __init__(self,
                 cfg,
                 workpiece_obj: str,
                 weld_json: str,
                 seam_id: int,
                 workpiece_pose: Optional[np.ndarray] = None,
                 goal_user: Optional[tuple] = None):
        cfg = load_config(cfg)

        # ===== 静态输入 =====
        self.cfg: Config = cfg
        self.workpiece_obj: str = workpiece_obj      # 工件 mesh（_watertight.obj）
        self.weld_json: str = weld_json              # 焊缝信息文件（_weld_angle3.json）
        self.seam_id: int = int(seam_id)             # 用 weld_json 中的第几条焊缝
        # 焊缝信息：load_welds 解析的 weld dict（p0/p1/mid/bisector/boundary_dirs/raw…）
        self.seams: dict = self._load_seam()
        self._set_cur_seam()
        # 工件在 base 系下的 pose（pose7）；plan_init_pose 选定候选后填入，构造期可由用户给定
        self.workpiece_pose: Optional[np.ndarray] = workpiece_pose
        # 用户【直接输入】的 goal ((x,y,z),(qw,qx,qy,qz))，base 系；不由焊缝计算（后续 API 用）
        self.goal_user: Optional[tuple] = goal_user

        # ===== 世界状态（3D）—— 本期占位，后续 API 填实 =====
        self.init_pose: Optional[InitPoseCandidate] = None   # 当前工件↔臂相对 pose；None=未求解/未应用
        self.init_pose_candidates: List[InitPoseCandidate] = []   # plan_init_pose 产出的全部候选（已排序）
        self.obstacles: list = []            # 已放障碍（ObstacleSpec）——后续
        self.truth_scene = None              # 工件+障碍（base 系）trimesh，raycast 几何源——后续
        self.voxmap = None                   # 三态记忆 ThreeStateVoxelMap——后续

        # ===== 机械臂状态 =====
        # 当前关节角：所有用法起步均为 retract（与初始位姿无关）
        self.cur_cfg: List[float] = [float(v) for v in self.cfg.retract_config]

        # ===== cuRobo 世界（惰性构建，脏标记缓存）—— 本期占位 =====
        self._h_truth = None        # MESH：工件+障碍（真实尺寸）
        self._h_plan = None         # MESH：工件+障碍（外扩 buffer）
        self._world_plan = None     # 与 _h_plan 同源的裸 WorldConfig
        self._h_expl = None         # VOXEL 三态
        self._camera_model = None
        self._solver = None         # InitPoseLookupSolver 句柄（按工件 key 复用查表）
        self._solver_key = None     # (workpiece_obj, n_per_dof)：solver 复用标识
        self._dirty: set = set()    # {"worlds","truth_scene","goal"} 缓存失效标记

    # ------------------------------------------------------------------
    # 焊缝
    # ------------------------------------------------------------------
    def _load_seam(self) -> dict:
        """从 weld_json 读全部焊缝，取第 seam_id 条（按 weld['idx'] 匹配；越界报错）。"""
        if not self.weld_json:
            raise ValueError("Scene 需要 weld_json（_weld_angle3.json）")
        pim = _load_plan_init_pose()
        welds = pim.load_welds(self.weld_json)
        if not welds:
            raise RuntimeError(f"weld_json 无焊缝：{self.weld_json}")
        else:
            return welds

    def _set_cur_seam(self):
        self.seam =  self.seams[self.seam_id]    
    
    # ------------------------------------------------------------------
    # 初始位姿求解（包 scripts/plan_init_pose.py:InitPoseLookupSolver，算法/输入输出完全一致）
    # ------------------------------------------------------------------
    def plan_init_pose(self,
                       n_per_dof: Optional[int] = None,
                       diagnostic: bool = False,
                       rebuild: bool = False) -> List[InitPoseCandidate]:
        """计算「工件 ↔ 机械臂」的候选初始位姿（lookup 式求解，本焊缝 self.seam）。

        等价于 scripts/plan_init_pose.py --solve 对单条焊缝跑一遍：
          ① InitPoseLookupSolver(cfg, obj, tol, voxel, n_per_dof) → precompute_joint_table（与工件无关，
             仅一次；按 (obj, n_per_dof) 缓存复用，rebuild=True 强制重建）；
          ② solve_one_weld_lookup(self.seam, rot_x/y/z)（角度采样、standoff 等全部读 cfg / default.yaml
             plan_init_pose 段，口径与脚本一致）。
        产出全部候选（已按 combined_score 降序 + 真实焊缝中点 x>0 优先）；最优解写入 self.init_pose、
        全部候选写入 self.init_pose_candidates，并据 best 填 self.workpiece_pose。

        参数：
          n_per_dof : 每关节采样档数；None→cfg.plan_init_n_per_dof。
          diagnostic: 透传到 solver，逐 (αx,βy,γz) 打印诊断。
          rebuild   : True 强制重建查表（换工件/换 n_per_dof 时）。

        返回：候选列表（list[InitPoseCandidate]，可能为空=求解失败）。
        """
        if not self.workpiece_obj:
            raise ValueError("plan_init_pose 需要 workpiece_obj（工件 mesh）")
        pim = _load_plan_init_pose()
        cfg = self.cfg
        n_per_dof = int(cfg.plan_init_n_per_dof if n_per_dof is None else n_per_dof)
        rot_x = pim._deg_range(cfg.plan_init_rot_x_deg)
        rot_y = pim._deg_range(cfg.plan_init_rot_y_deg)
        rot_z = pim._deg_range(cfg.plan_init_rot_z_deg)

        # —— 求解器：按 (工件, n_per_dof) 复用查表（同工件多焊缝只建一次） ——
        key = (self.workpiece_obj, n_per_dof)
        if rebuild or self._solver is None or self._solver_key != key:
            self._solver = pim.InitPoseLookupSolver(
                cfg, self.workpiece_obj,
                collision_tolerance=cfg.plan_init_collision_tolerance,
                voxel_size=cfg.plan_init_voxel_size,
                n_per_dof=n_per_dof)
            # 优先读 cfg.plan_init_joint_table_path 的缓存（与工件无关）；rebuild 或无缓存则现算并落盘。
            if rebuild or not self._solver.load_joint_table():
                self._solver.precompute_joint_table()
                self._solver.save_joint_table()
            self._solver_key = key

        sol = self._solver.solve_one_weld_lookup(
            self.seam, rot_x, rot_y, rot_z, diagnostic=diagnostic)

        all_sols = getattr(self._solver, "last_all_solutions", []) or []
        self.init_pose_candidates = [InitPoseCandidate.from_solution(s) for s in all_sols]
        self.init_pose = self.init_pose_candidates[0] if (sol is not None and
                                                          self.init_pose_candidates) else None
        if self.init_pose is not None:
            self.workpiece_pose = self.init_pose.workpiece_pose7
        return self.init_pose_candidates
