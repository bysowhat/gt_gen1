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


# ======================================================================
# 障碍物几何（类型2 遮挡板 / 类型3 开口障碍）
# ----------------------------------------------------------------------
# 纯 numpy 几何，镜像两个独立示例脚本的造型逻辑（不含任何 isaacsim/usd 调用；渲染在 scene_viz.py）：
#   · 类型2 遮挡板候选 C1~C4 + 板外形 → scripts/viz_seam_plate_candidates_isaacsim.py
#   · 类型3 开口盒 / 开口圆筒          → scripts/viz_seam_open_box_isaacsim.py
# 造出的原语/网格都在【工件 mesh 系】（与 weld_json 的 corrected_p0/p1/bisector 同框）。
# ======================================================================
_GRAY = [0.62, 0.64, 0.67]          # 钢材灰（RGB 0~1）


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _interp_line(p0, p1, n: int = 20):
    """两端点线性插值成 (n,3) 折线（焊缝 seam_line）。"""
    ts = np.linspace(0.0, 1.0, n)
    return p0[None, :] * (1.0 - ts[:, None]) + p1[None, :] * ts[:, None]


# ---------------------------------------------------------------- 板外形 profile（板局部 宽=u, 长=v 平面）
def _profile_rect(hw, hl):
    """完整矩形：四角 (±宽/2, ±长/2)。"""
    return [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]


def _profile_triangle(hw, hl, apex_sign):
    """等腰三角形：尖端在角平分线 +Y 一侧的整条宽边中点，底边为对侧整条边。"""
    return [(apex_sign * hw, 0.0), (-apex_sign * hw, -hl), (-apex_sign * hw, hl)]


def _profile_trapezoid(hw, hl, rng):
    """从矩形随机倒掉 1~4 个角（每个被选中角沿两邻边各内缩一段）→ 梯形/多边形。"""
    rect = [np.array(p, float) for p in _profile_rect(hw, hl)]
    nc = len(rect)
    chosen = set(rng.sample(range(nc), rng.randint(1, nc)))
    out = []
    for i in range(nc):
        ci = rect[i]
        if i in chosen:
            prev, nxt = rect[(i - 1) % nc], rect[(i + 1) % nc]
            out.append(tuple(ci + rng.uniform(0.2, 0.45) * (prev - ci)))
            out.append(tuple(ci + rng.uniform(0.2, 0.45) * (nxt - ci)))
        else:
            out.append(tuple(ci))
    return out


def _shape_profile(shape, hw, hl, apex_sign, rng):
    if shape == "plate":
        return _profile_rect(hw, hl)
    if shape == "triangle":
        return _profile_triangle(hw, hl, apex_sign)
    if shape == "trapezoid":
        return _profile_trapezoid(hw, hl, rng)
    raise ValueError(f"未知板外形 {shape!r}，可选 plate/triangle/trapezoid")


def _prism_mesh(anchor, R, profile, thickness, color):
    """板局部 profile(宽=u, 长=v) 沿法向 X 挤出 thickness → 薄棱柱三角/多边形网格。

    返回 dict(points=(2n,3), counts=[…面顶点数…], faces=[…顶点索引…], color)。
    镜像 viz_seam_plate_candidates_isaacsim.spawn_prism 的顶点/面拓扑（不含 usd 调用）。
    local(x,u,v) → world = anchor + R @ [x, u, v]（R 列=[法向X, 宽向Y, 长向Z]）。
    """
    nseg = len(profile)
    half = float(thickness) / 2.0
    R = np.asarray(R, float)
    anchor = np.asarray(anchor, float)
    pts = [anchor + R @ np.array([half, float(u), float(v)]) for (u, v) in profile]       # 前环 +half
    pts += [anchor + R @ np.array([-half, float(u), float(v)]) for (u, v) in profile]     # 后环 -half

    counts, faces = [], []
    counts.append(nseg); faces += list(range(nseg))                          # 前盖
    counts.append(nseg); faces += list(range(2 * nseg - 1, nseg - 1, -1))    # 后盖（反序朝外）
    for i in range(nseg):                                                    # 侧面四边形
        j = (i + 1) % nseg
        counts.append(4); faces += [i, j, nseg + j, nseg + i]
    return dict(points=np.asarray(pts, float), counts=counts, faces=faces, color=list(color))


def _cylinder_mesh(p_open, p_close, y_axis, z_axis, radius, color, nseg: int = 48):
    """开口圆筒（圆管杯）→ 光滑网格：侧壁环 + 封底圆盘；开口端不封。

    返回 dict(points, counts, faces, color)。镜像 viz_seam_open_box_isaacsim.spawn_open_cylinder。
    """
    p_open = np.asarray(p_open, float)
    p_close = np.asarray(p_close, float)
    ey = np.asarray(y_axis, float)
    ez = np.asarray(z_axis, float)
    r = float(radius)
    pts = []
    for i in range(nseg):                                # 0..nseg-1：开口环
        th = 2.0 * np.pi * i / nseg
        pts.append(p_open + r * (np.cos(th) * ey + np.sin(th) * ez))
    for i in range(nseg):                                # nseg..2nseg-1：封底环
        th = 2.0 * np.pi * i / nseg
        pts.append(p_close + r * (np.cos(th) * ey + np.sin(th) * ez))
    c_idx = len(pts)
    pts.append(p_close)                                  # 2nseg：封底圆心

    counts, faces = [], []
    for i in range(nseg):                                # 侧壁四边形
        j = (i + 1) % nseg
        counts.append(4); faces += [i, j, nseg + j, nseg + i]
    for i in range(nseg):                                # 封底三角扇
        j = (i + 1) % nseg
        counts.append(3); faces += [c_idx, nseg + j, nseg + i]
    return dict(points=np.asarray(pts, float), counts=counts, faces=faces, color=list(color))


@dataclass
class ObstacleSpec:
    """一个已放置的障碍物（类型2/3）的可视化 + cuRobo 载荷（工件 mesh 系）。

    · otype     : 障碍类型编号（2=遮挡板候选 / 3=开口障碍）。
    · kind      : 具体形状（plate/triangle/trapezoid/open_box/open_cylinder）。
    · prims     : gt_gen.obstacles 的 Box/Tube 原语（open_box→5 块 Box；可转 cuRobo，见 obstacles.to_curobo）。
    · meshes    : [dict(points,counts,faces,color)] 任意棱柱/圆筒网格（遮挡板、open_cylinder 走这里，供直接建 UsdGeom.Mesh）。
    · seam_line : 该障碍对应焊缝折线 (N,3)（可视化画红线）。
    · color     : 缺省渲染色。
    · meta      : 造型时的中间量/参数（诊断用）。
    """
    otype: int
    kind: str
    prims: list = field(default_factory=list)
    meshes: list = field(default_factory=list)
    seam_line: Optional[np.ndarray] = None
    color: List[float] = field(default_factory=lambda: list(_GRAY))
    meta: dict = field(default=None, repr=False)


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

    def _set_cur_seam(self, seam_id):
        self.seam = self.seams[seam_id]    
    
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

    # ------------------------------------------------------------------
    # 障碍物（工件 mesh 系；与 weld_json 的 corrected_p0/p1/bisector 同框）
    # ------------------------------------------------------------------
    def _seam_frame(self, n_seg: int = 20):
        """本焊缝几何 → (mid, t, d1, d2, bis, seam_len, seam_line)，均在工件 mesh 系。

        镜像 viz_seam_*_isaacsim.py 的 seam_frame_world，但数据源改为 Scene 的 weld dict
        （p0_world/p1_world/boundary_dirs/bisector_world，见 plan_init_pose.load_welds）：
          t=焊缝切线(单位)，d1/d2=两面 boundary_dirs，bis=unit(d1+d2)（退化则回退 bisector_world）。
        """
        w = self.seam
        p0 = np.asarray(w["p0_world"], float)
        p1 = np.asarray(w["p1_world"], float)
        mid = np.asarray(w["mid_world"], float)
        t = _unit(p1 - p0)
        bd = np.asarray(w.get("boundary_dirs", np.zeros((2, 3))), float).reshape(2, 3)
        d1, d2 = bd[0], bd[1]
        bis = _unit(d1 + d2)
        if float(np.linalg.norm(bis)) < 1e-9:            # boundary_dirs 缺失/退化 → 回退 bisector
            bis = _unit(np.asarray(w["bisector_world"], float))
        seam_len = float(np.linalg.norm(p1 - p0))
        seam_line = _interp_line(p0, p1, n=n_seg)
        return mid, t, d1, d2, bis, seam_len, seam_line

    def add_obstacle_type2(self, **overrides) -> ObstacleSpec:
        """障碍物类型2：焊缝旁遮挡板（固定候选 **C1**，示例 viz_seam_plate_candidates_isaacsim.py）。

        C1 = 水平板·焊缝正上方对称：∥地面 a 面（板法向=a 面法向 na）、沿壁方向 b_dir 抬高 n_cm、
        板心在焊缝正上方、宽沿 ±对称。参数默认读 cfg.obstacle_placement_type2；可用关键字覆盖
        （shape/n_cm/width_cm/length_pct/length_min_cm/thickness_cm/seed）。产出的 ObstacleSpec
        （以棱柱 mesh 表示，外形由 shape 决定）追加进 self.obstacles 并返回。
        """
        import random
        p = dict(self.cfg.obstacle_placement_type2)
        p.update({k: v for k, v in overrides.items() if v is not None})
        shape = str(p["shape"])

        mid, t, d1, d2, bis, seam_len, seam_line = self._seam_frame()
        a_dir = _unit(d1 - float(np.dot(d1, t)) * t)     # a 面(d1)表面方向，⊥t
        b_dir = _unit(d2 - float(np.dot(d2, t)) * t)     # b 面(d2)表面方向，⊥t
        if float(np.linalg.norm(a_dir)) < 1e-9 or float(np.linalg.norm(b_dir)) < 1e-9:
            raise RuntimeError("焊缝 boundary_dirs 退化，无法造类型2遮挡板（需两面表面方向）")
        na = _unit(np.cross(t, a_dir))                   # a 面法向；取朝 bis 张开侧
        if float(np.dot(na, bis)) < 0:
            na = -na

        n = float(p["n_cm"]) / 100.0
        width = float(p["width_cm"]) / 100.0
        length = max(float(p["length_pct"]) / 100.0 * seam_len, float(p["length_min_cm"]) / 100.0)
        thickness = float(p["thickness_cm"]) / 100.0
        hw, hl = width / 2.0, length / 2.0

        # 候选 C1：板法向=na，板心=焊缝中点沿壁方向 b_dir 抬高 n；长向=切线 t，宽向=t×na
        color = [0.0, 0.85, 0.0]
        anchor = mid + n * b_dir
        x_axis = _unit(na)                               # 板法向 → local X
        z_axis = t                                       # 板长方向 → local Z
        y_axis = _unit(np.cross(z_axis, x_axis))         # 板宽方向 → local Y
        R = np.column_stack([x_axis, y_axis, z_axis])
        apex_sign = 1.0 if float(np.dot(bis, y_axis)) >= 0 else -1.0

        rng = random.Random(int(p.get("seed", 0)) * 100003 + self.seam_id * 101)
        profile = _shape_profile(shape, hw, hl, apex_sign, rng)
        mesh = _prism_mesh(anchor, R, profile, thickness, color)
        spec = ObstacleSpec(
            otype=2, kind=shape, meshes=[mesh], seam_line=seam_line, color=color,
            meta=dict(candidate="C1", anchor=anchor, R=R, width=width,
                      length=length, thickness=thickness))
        self.obstacles.append(spec)
        return spec

    def _box_geom_type3(self, dis_m):
        """据焊缝 + 6 面距离(上下左右前后, 米) 解出开口障碍的盒局部系/尺寸/盒心/三轴投影。

        镜像 viz_seam_open_box_isaacsim._box_geom：盒轴 Z=世界+Z（上）；开口 back(−X) 朝水平角平分线
        bis_h（bis 穿过开口）；front(+X)=−bis_h；Y=Z×X。焊缝折线投到三轴得跨度，配 6 个 dis 唯一
        定出 size 与盒心。open_box 与 open_cylinder 共用。返回 dict。
        """
        from scipy.spatial.transform import Rotation as Rsp
        mid, t, d1, d2, bis, seam_len, pts = self._seam_frame()

        zb = np.array([0.0, 0.0, 1.0])
        bis_h = bis - float(np.dot(bis, zb)) * zb        # 角平分线水平投影
        if float(np.linalg.norm(bis_h)) < 1e-6:          # 退化：bis 近垂直 → 退用 d1 的水平分量
            bis_h = d1 - float(np.dot(d1, zb)) * zb
            if float(np.linalg.norm(bis_h)) < 1e-6:
                bis_h = np.array([1.0, 0.0, 0.0])
        bis_h = _unit(bis_h)

        x_axis = -bis_h                                  # front(+X) 朝工件；back(−X,开口) 朝 +bis_h
        z_axis = zb
        y_axis = _unit(np.cross(z_axis, x_axis))         # 左(+Y)
        Rm = np.column_stack([x_axis, y_axis, z_axis])   # 盒局部→world（列=[X,Y,Z]）

        a = pts @ x_axis                                 # 焊缝沿 X(前后) 投影
        b = pts @ y_axis                                 # 沿 Y(左右)
        c = pts @ z_axis                                 # 沿 Z(上下)
        a0, a1 = float(a.min()), float(a.max())
        b0, b1 = float(b.min()), float(b.max())
        c0, c1 = float(c.min()), float(c.max())

        dt, db, dl, dr, df, dk = [float(v) for v in dis_m]   # 上 下 左 右 前 后（米）
        sz = (c1 - c0) + dt + db
        sy = (b1 - b0) + dl + dr
        sx = (a1 - a0) + df + dk
        cen_c = ((c1 + dt) + (c0 - db)) / 2.0
        cen_b = ((b1 + dl) + (b0 - dr)) / 2.0
        cen_a = ((a1 + df) + (a0 - dk)) / 2.0
        C = cen_a * x_axis + cen_b * y_axis + cen_c * z_axis     # 盒心(world)

        rpy = [float(v) for v in Rsp.from_matrix(Rm).as_euler("xyz", degrees=True)]
        return dict(mid=mid, bis=bis, bis_h=bis_h, seam_len=seam_len, seam_line=pts,
                    x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, Rm=Rm, rpy=rpy,
                    a0=a0, a1=a1, b0=b0, b1=b1, c0=c0, c1=c1,
                    sx=float(sx), sy=float(sy), sz=float(sz),
                    cen_a=cen_a, cen_b=cen_b, cen_c=cen_c, C=C)

    def add_obstacle_type3(self, **overrides) -> ObstacleSpec:
        """障碍物类型3：把本焊缝包住的开口障碍（示例 viz_seam_open_box_isaacsim.py）。

        大小 + 焊缝在障碍内的位置由 6 个面到焊缝的最近距离 dis 唯一解出；开口面=后(back)，朝焊缝角平分线
        水平投影方向。参数默认读 cfg.obstacle_placement_type3；可用关键字覆盖（obstacle/dis_cm/wall_cm）：
          · open_box      → 5 块 Box 原语（可转 cuRobo，见 obstacles.to_curobo）。
          · open_cylinder → 光滑圆筒 mesh（轴=水平角平分线，半径外接盒 Y-Z 截面；纯视觉）。
        产出的 ObstacleSpec 追加进 self.obstacles 并返回。
        """
        p = dict(self.cfg.obstacle_placement_type3)
        p.update({k: v for k, v in overrides.items() if v is not None})
        obstacle = str(p["obstacle"])
        dis_m = [float(v) / 100.0 for v in p["dis_cm"]]
        wall = float(p["wall_cm"]) / 100.0

        g = self._box_geom_type3(dis_m)
        sx, sy, sz = g["sx"], g["sy"], g["sz"]

        if obstacle == "open_cylinder":
            u = g["x_axis"]                              # 圆筒轴（+X=朝工件/封底侧）
            r = 0.5 * max(sy, sz)                        # 外接盒 Y-Z 截面
            C = g["C"]
            p_close = C + (sx / 2.0) * u                 # 封底圆盘心（+X，工件侧）
            p_open = C - (sx / 2.0) * u                  # 开口圈心（−X，+bis_h 侧）
            mesh = _cylinder_mesh(p_open, p_close, g["y_axis"], g["z_axis"], r, _GRAY)
            spec = ObstacleSpec(
                otype=3, kind="open_cylinder", meshes=[mesh], seam_line=g["seam_line"],
                color=list(_GRAY),
                meta=dict(center=C, rpy=g["rpy"], radius=float(r), length=float(sx),
                          z_half=float(r)))
        elif obstacle == "open_box":
            from gt_gen import obstacles as ob
            prims = ob.build("open_box", g["C"].tolist(), anchor_rpy_deg=tuple(g["rpy"]),
                             size=(sx, sy, sz), wall=wall, open_face="back")
            spec = ObstacleSpec(
                otype=3, kind="open_box", prims=prims, seam_line=g["seam_line"],
                color=list(_GRAY),
                meta=dict(center=g["C"], rpy=g["rpy"], size=(sx, sy, sz), z_half=sz / 2.0))
        else:
            raise ValueError(f"未知类型3障碍 {obstacle!r}，可选 open_box/open_cylinder")

        self.obstacles.append(spec)
        return spec
