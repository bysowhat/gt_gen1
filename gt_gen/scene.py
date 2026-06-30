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
    """一个候选初始位姿（kejian2 逻辑产出）= 工件 mesh world ↔ 机械臂 base 的相对 pose (R, t)
    + 焊枪 goal 位姿 + 该焊缝可达参考关节角 + 正/反手分类。

    约定：p_base = R · p_world + t（R,t 即 T_workpiece_in_base 的旋转/平移）。
    它定义一整套自洽的世界布局（工件在 base 系的摆放），**不是机械臂的起步关节角**
    （起步仍恒为 retract）；q 仅是「焊枪头落在该焊缝 standoff 落点」的可达参考构型。

    raw 保留 scripts/plan_init_pose.py:_kejian2_solve_weld 产出的原始结果 dict，
    供可视化原样复用（见 SceneVisualizer）。
    """
    R: np.ndarray                  # (3,3) 工件→base 旋转（已 snap 到某种允许朝向）
    t: np.ndarray                  # (3,)  工件→base 平移
    q: np.ndarray                  # (dof,) 可达该焊缝的关节角（参考/校验，非起步角）
    goal_pose7: np.ndarray         # 焊枪 goal 位姿 [x,y,z,qw,qx,qy,qz]（base 系，落在 standoff 点）
    bisector_base: np.ndarray      # 焊缝角平分线（背离工件=approach 方向）在 base
    seam_center_base: np.ndarray   # 焊缝中心点在 base
    rot_x_deg: float               # 绕末端局部 x/y/z 轴的扰动角（度）
    rot_y_deg: float
    rot_z_deg: float
    orientation_id: int            # 命中的允许朝向编号（0..3）
    hand: str                      # "forehand"(正手) / "backhand"(反手)
    raw: dict = field(default=None, repr=False)   # 原始 kejian2 结果 dict（可视化复用）

    @classmethod
    def from_kejian2(cls, d: dict) -> "InitPoseCandidate":
        """由 _kejian2_solve_weld 结果（forehand/backhand 列表里的单条 dict）构造。"""
        T = np.asarray(d["T_workpiece_in_base"], dtype=np.float64)
        return cls(
            R=T[:3, :3].copy(),
            t=T[:3, 3].copy(),
            q=np.asarray(d["joint_angles"], dtype=np.float64),
            goal_pose7=np.asarray(d["goal_pose7"], dtype=np.float64),
            bisector_base=np.asarray(d["bisector_base"], dtype=np.float64),
            seam_center_base=np.asarray(d["seam_center_base"], dtype=np.float64),
            rot_x_deg=float(d["rot_x_deg"]),
            rot_y_deg=float(d["rot_y_deg"]),
            rot_z_deg=float(d["rot_z_deg"]),
            orientation_id=int(d["orientation_id"]),
            hand=str(d["hand"]),
            raw=d,
        )

    def to_kejian2(self) -> dict:
        """还原成 _kejian2_solve_weld 结果 dict 的形态（供 kejian2 可视化函数原样吃）。"""
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
        cands = scene.plan_init_pose()      # 候选初始位姿（工件↔臂相对 pose，正反手全返回）
        scene.init_pose_candidates          # = cands（全部候选，无数量上限）

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
        # 工件在 base 系下的 pose（pose7）；构造期可由用户给定（plan_init_pose 不再自动选 best 回填，
        # 候选全在 self.init_pose_candidates，由调用方挑选后自行 apply）
        self.workpiece_pose: Optional[np.ndarray] = workpiece_pose
        # 用户【直接输入】的 goal ((x,y,z),(qw,qx,qy,qz))，base 系；不由焊缝计算（后续 API 用）
        self.goal_user: Optional[tuple] = goal_user

        # ===== 世界状态（3D）—— 本期占位，后续 API 填实 =====
        self.init_pose_candidates: List[InitPoseCandidate] = []   # plan_init_pose 产出的全部候选（正手在前、反手在后，无数量上限）
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
        self._k2ctx = None          # kejian2 工件级求解上下文（solver/ESDF/joint表/允许朝向/底座圆/mesh）
        self._k2ctx_key = None      # workpiece_obj：_k2ctx 复用标识
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
    # 初始位姿求解（包 scripts/plan_init_pose.py 的 kejian2 逻辑，算法/输入输出完全一致）
    # ------------------------------------------------------------------
    def plan_init_pose(self,
                       diagnostic: bool = False,
                       rebuild: bool = False) -> List[InitPoseCandidate]:
        """计算「工件 ↔ 机械臂」的候选初始位姿（kejian2 逻辑，本焊缝 self.seam）。

        与 scripts/plan_init_pose.py 走【完全同一套过滤逻辑】（直接复用其 _kejian2_build_ctx /
        _kejian2_solve_weld，不复制几何）：
          ① 工件级 _kejian2_build_ctx：InitPoseLookupSolver（n^6 关节角采样 + 工件 ESDF 体素化 ②）、
             joint 表（③，已存盘则复用）、4 种允许朝向、固定底座圆 + 工件顶点/三角形——按工件复用、只建一次；
          ② 逐焊缝 _kejian2_solve_weld：lookup 碰撞过滤 → 朝向 snap → 正面过滤 / 焊缝中心 base-x>0 /
             固定底座-工件 base-xy 相交过滤（由 plan_init_pose.base_overlap_filter 开关控制）→ 正反手分类。
        配置全部读 default.yaml 的 plan_init_pose 段（口径与脚本 --solve-kejian2 一致）。

        与脚本【落盘时「每只手最多 15 个」】不同：这里【有多少正反手就返回多少】，不做数量上限挑选。
        全部候选（正手在前、反手在后）写入 self.init_pose_candidates 并返回。

        参数：
          diagnostic: 预留（kejian2 逐焊缝求解暂不细分诊断，当前未使用）。
          rebuild   : True 强制重建工件级 ctx（换工件 / 改 n_per_dof 等缓存失效时）。

        返回：候选列表（list[InitPoseCandidate]，可能为空=求解失败/无合格解）。
        """
        if not self.workpiece_obj:
            raise ValueError("plan_init_pose 需要 workpiece_obj（工件 mesh）")
        pim = _load_plan_init_pose()

        # —— 工件级 ctx：按工件复用（同工件多焊缝只建一次 ②③）。注意 _kejian2_build_ctx 内部用
        #    load_config() 读默认 default.yaml；与 Scene 默认 cfg 同源，口径一致。 ——
        if rebuild or self._k2ctx is None or self._k2ctx_key != self.workpiece_obj:
            self._k2ctx = pim._kejian2_build_ctx(self.workpiece_obj)
            self._k2ctx_key = self.workpiece_obj

        res, _prof = pim._kejian2_solve_weld(self._k2ctx, self.seam)

        # 有多少正反手都返回（不做 15 个上限挑选）：正手在前、反手在后
        cands = list(res.get("forehand", [])) + list(res.get("backhand", []))
        self.init_pose_candidates = [InitPoseCandidate.from_kejian2(d) for d in cands]
        return self.init_pose_candidates
