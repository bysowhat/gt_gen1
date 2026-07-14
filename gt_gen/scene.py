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
from typing import Dict, List, Optional, Tuple

import numpy as np

from gt_gen.config import Config, load_config, PROJECT_ROOT

_SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")


class Ctx(dict):
    """世界上下文容器：dict 语义完全不变（ctx["h_truth"] 照常用），仅重写 __repr__。

    默认 dict 的 repr 会递归 str 化 h_truth / h_expl / truth_scene / vm 等巨型对象，
    VSCode 调试器悬停 / 变量面板会非常卡。这里只打印各键名与其类型，不展开内容。
    """

    def __repr__(self):
        items = ", ".join(f"{k}={type(v).__name__}" for k, v in self.items())
        return f"Ctx({items})"

    __str__ = __repr__


def _load_plan_init_pose():
    """惰性导入 scripts/plan_init_pose.py 模块（不改它、原样复用其求解器与可视化）。

    该脚本顶层只 import 轻量库（argparse/os/numpy…），curobo/torch 都在函数内部惰性导入，
    故模块级 import 代价极小、可安全在 Scene 方法里按需调用。"""
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)
    import plan_init_pose  # noqa: E402  （scripts/plan_init_pose.py）
    return plan_init_pose


def _load_compute_goal_poses2():
    """惰性导入 scripts/compute_goal_poses2.py（观测位姿=goal pose 的去 Isaac 优化器）。

    该模块顶层会：① 把 stomp_planner 加进 sys.path；② 注入 scene_pose 轻量 stub 顶掉
    optimizer_pose 的类型注解 import（避免连带 import isaaclab）；③ import
    ConfigurationPose / OptimizerPose / ScenePose2。故 import 一次即拿到全套 ES 优化器类，
    且不启动 Isaac Sim。torch/curobo/warp 都在 ScenePose2/Optimizer 内部惰性初始化。"""
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)
    import compute_goal_poses2  # noqa: E402  （scripts/compute_goal_poses2.py）
    return compute_goal_poses2


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


def _polygon_mesh_to_trimesh(mesh: dict):
    """棱柱/多边形 mesh dict(points/counts/faces) → 实体 trimesh（多边形面扇形三角化）。

    供把类型2遮挡板（闭合薄棱柱）合并进碰撞 ESDF。返回 None 表示无有效面。
    """
    import trimesh
    pts = np.asarray(mesh["points"], float)
    counts = [int(c) for c in mesh["counts"]]
    idx = [int(i) for i in mesh["faces"]]
    tris, off = [], 0
    for c in counts:
        face = idx[off:off + c]; off += c
        for k in range(1, c - 1):                # 扇形三角化 (f0,fk,fk+1)
            tris.append([face[0], face[k], face[k + 1]])
    if not tris:
        return None
    return trimesh.Trimesh(vertices=pts, faces=np.asarray(tris, np.int64), process=False)


def _box_prim_to_trimesh(prim):
    """Box 原语(pose=[x,y,z,qw,qx,qy,qz], dims) → 实体 trimesh box（watertight）。"""
    import trimesh
    from scipy.spatial.transform import Rotation as Rsp
    pose = np.asarray(prim.pose, float)
    dims = np.asarray(prim.dims, float)
    q = pose[3:7]                                # wxyz → scipy xyzw
    T = np.eye(4)
    T[:3, :3] = Rsp.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = pose[:3]
    return trimesh.creation.box(extents=dims.tolist(), transform=T)


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
        cands = scene.plan_init_pose()      # 当前焊缝候选（工件↔臂相对 pose），按手别分组返回 {"forehand":[...], "backhand":[...]}
        scene.init_pose_candidates          # = {seam_id: {"forehand":[...], "backhand":[...]}}（各组按 xyz 差异降序）

    cfg 接受 Config 实例 / yaml 路径 / None（None→load_config() 取默认）。
    """

    def __init__(self,
                 cfg,
                 workpiece_obj: str,
                 weld_json: str,
                 workpiece_pose: Optional[np.ndarray] = None,
                 goal_user: Optional[tuple] = None):
        cfg = load_config(cfg)

        # ===== 静态输入 =====
        self.cfg: Config = cfg
        self.workpiece_obj: str = workpiece_obj      # 工件 mesh（_watertight.obj）
        self.weld_json: str = weld_json              # 焊缝信息文件（_weld_angle3.json）
        # 焊缝信息：load_welds 解析的 weld dict（p0/p1/mid/bisector/boundary_dirs/raw…）
        self.seams: dict = self._load_seam()
        # 工件在 base 系下的 pose（pose7）；构造期可由用户给定（plan_init_pose 不再自动选 best 回填，
        # 候选全在 self.init_pose_candidates，由调用方挑选后自行 apply）
        self.workpiece_pose: Optional[np.ndarray] = workpiece_pose
        # 用户【直接输入】的 goal ((x,y,z),(qw,qx,qy,qz))，base 系；不由焊缝计算（后续 API 用）
        self.goal_user: Optional[tuple] = goal_user

        # ===== 世界状态（3D）—— 本期占位，后续 API 填实 =====
        # plan_init_pose 产出的候选：按【焊缝】分组 {seam_id: {"forehand":[...], "backhand":[...]}}；
        # 每焊缝下再按手别分组，每组各自按【工件平移 xyz 差异】独立分数降序（差异大的排前面），两组互不混排。
        self.init_pose_candidates: Dict[int, Dict[str, List[InitPoseCandidate]]] = {}
        self.init_pose_candidates_length: Dict[int, Dict[str, int]] = {}
        # {seam_id: [step1,…,step8]}：plan_init_pose 逐步过滤每步【通过】的候选（raw 结果 dict），
        # 供 Open3DSceneVisualizer.show_init_poses_debug(n) 单步可视化。每次 plan_init_pose 覆盖本 seam。
        self.init_pose_debug_steps: Dict[int, List[List[dict]]] = {}
        self.cur_init_pose: Optional[InitPoseCandidate] = None    # 当前选定的 init pose 候选（set_init_pose 设定）——定义工件↔base 摆放
        self.cur_init_hand: Optional[str] = None                  # 当前候选所属手别（"forehand"/"backhand"）
        self.cur_init_index: Optional[int] = None                 # 当前候选在【该手别列表】中的下标
        self.goal_poses: Dict[int, Optional[dict]] = {}   # {seam_id: dict|None} compute_goal_pose 产出的单个观测位姿结果（含 cam_pose 等）；无解=None
        self.trajectories: Dict[int, Dict[Tuple[str, int], list]] = {}   # {seam_id: {(hand, index): list[dict]}} plan_explore_path 产出的
                                                                          # 边走边看轨迹；二级 key=当前 init pose(手别,候选下标)，避免同一 seam 下
                                                                          # 正/反手、不同候选的多条轨迹混进同一个 list
        self.trajectory_goal_poses: Dict[int, Dict[Tuple[str, int], dict]] = {}   # {seam_id: {(hand, index): goal_pose结果}}
                                                                          # 每条成功轨迹对应候选当时的 compute_goal_pose 结果快照，
                                                                          # 供 show_trajectory_isaacsim 按 traj_index 还原正确的视锥（
                                                                          # self.goal_poses 是全局单值，会被后续候选覆盖，不够用）
        self.obstacles: Dict[int, list] = {}   # {seam_id: list[ObstacleSpec]} 已放障碍（按焊缝分组）
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
        self._k2ctx_injected = False  # solver 碰撞世界当前是否已并入障碍（避障态标记）
        self._dirty: set = set()    # {"worlds","truth_scene","goal"} 缓存失效标记

    # ------------------------------------------------------------------
    # 焊缝
    # ------------------------------------------------------------------
    def _load_seam(self) -> dict:
        """从 weld_json 读全部焊缝；丢弃长度(corrected_p0↔p1 直线距离)小于 cfg.seam_min_length_m 的短焊缝。"""
        if not self.weld_json:
            raise ValueError("Scene 需要 weld_json（_weld_angle3.json）")
        pim = _load_plan_init_pose()
        welds = pim.load_welds(self.weld_json)
        if not welds:
            raise RuntimeError(f"weld_json 无焊缝：{self.weld_json}")
        min_len = self.cfg.seam_min_length_m
        kept = [w for w in welds
                if float(np.linalg.norm(np.asarray(w["p1_world"], float)
                                        - np.asarray(w["p0_world"], float))) >= min_len]
        return kept

    def _set_cur_seam(self, seam_id):
        # self._clear_scene()#todo
        self.seam = self.seams[seam_id]
        self.seam_id = seam_id

    def seam_ids_by_length(self) -> List[Tuple[int, float]]:
        """返回全部焊缝 (id, 长度)，按焊缝长度降序排列（最长的在前）。

        id 为 self.seams 下标；长度 = corrected_p0↔p1（p0_world↔p1_world 直线距离，米），
        与 _load_seam 的过滤口径一致。稳定排序（同长保持原相对次序）。
        """
        def _len(w):
            return float(np.linalg.norm(np.asarray(w["p1_world"], float)
                                        - np.asarray(w["p0_world"], float)))
        pairs = [(i, _len(self.seams[i])) for i in range(len(self.seams))]
        return sorted(pairs, key=lambda p: -p[1])
    
    def num_init_pose(self, seam_id=None):
        if seam_id is None:
            seam_id = self.seam_id
        return len(self.init_pose_candidates[seam_id]['forehand']) + len(self.init_pose_candidates[seam_id]['backhand'])
        

    # ------------------------------------------------------------------
    # 初始位姿求解（包 scripts/plan_init_pose.py 的 kejian2 逻辑，算法/输入输出完全一致）
    # ------------------------------------------------------------------
    def plan_init_pose(self,
                       diagnostic: bool = False,
                       rebuild: bool = False,
                       include_obstacles: bool = True,
                       verbose: bool = False) -> Dict[str, List[InitPoseCandidate]]:
        """计算「工件 ↔ 机械臂」的候选初始位姿（kejian2 逻辑，本焊缝 self.seam）。

        与 scripts/plan_init_pose.py 走【完全同一套过滤逻辑】（直接复用其 _kejian2_build_ctx /
        _kejian2_solve_weld，不复制几何）：
          ① 工件级 _kejian2_build_ctx：InitPoseLookupSolver（n^6 关节角采样 + 工件 ESDF 体素化 ②）、
             joint 表（③，已存盘则复用）、4 种允许朝向、固定底座圆 + 工件顶点/三角形——按工件复用、只建一次；
          ② 逐焊缝 _kejian2_solve_weld：lookup 碰撞过滤 → 朝向 snap → 正面过滤 / 焊缝中心 base-x>0 /
             固定底座-工件 base-xy 相交过滤（由 plan_init_pose.base_overlap_filter 开关控制）→ 正反手分类。
        配置全部读 default.yaml 的 plan_init_pose 段（口径与脚本 --solve-kejian2 一致）。

        与脚本【落盘时「每只手最多 15 个」】不同：这里【有多少正反手就返回多少】，不做数量上限挑选。
        全部候选按手别分组写入 self.init_pose_candidates[self.seam_id] = {"forehand":[...], "backhand":[...]}，
        每组各自按【工件平移 xyz 差异】独立分数降序（差异大的排前面），并返回该焊缝的 per-hand dict。

        **避障**：include_obstacles=True（默认）且 self.obstacles 非空时，把已放障碍（类型2遮挡板 /
        类型3 open_box）合并进工件 mesh 一起重算 signed ESDF，覆盖 solver 的碰撞体素——碰撞过滤链路
        （_kejian2_solve_weld 内的 lookup）随之把障碍算进去，一行算法不改（见 _inject_obstacles_into_solver）。
        open_cylinder 为纯视觉、不进碰撞世界。include_obstacles=False 则退回仅工件（若此前注过障碍会自动还原）。

        参数：
          diagnostic       : 预留（kejian2 逐焊缝求解暂不细分诊断，当前未使用）。
          rebuild          : True 强制重建工件级 ctx（换工件 / 改 n_per_dof 等缓存失效时）。
          include_obstacles: True（默认）把 self.obstacles 并入碰撞世界后再求解（避障）。
          verbose          : True 打印 ESDF 并入 / solve profile 计时（诊断噪声，默认 False 静默）。
                             注：逐步过滤候选计数（① lookup→…→⑧ 复检→合格）无条件打印，不受此开关影响。

        返回：候选 dict {"forehand":[...], "backhand":[...]}（各组可能为空=该手别无合格解）。


        过滤步骤：
        ① lookup 候选（绕末端轴 13×5×5 采样） : 4834
        ② 工作空间 + 朝向snap 命中            : 4834 → 446
        ③ 正面过滤(bisector base-z≥0)        : 446 → 317（背面丢 129）
        ④ 焊缝中心 base-x>0                   : 317 → 313（x≤0 丢 4）
        # 废弃 ⑤ 工件+障碍 vs init_free 空间无交集   : 313 → 0（相交丢 313）
        ⑤ 工件+障碍 vs init_free 空间无交集
        ⑥ 底座-工件 XY 投影不相交            : 0 → 0（相交丢 0）
        ⑦ 轻去重(同朝向 + 2cm 同位)          : 0 → 0（重复丢 0）
        ⑧ snap后 retract-工件无碰撞复检      : 0 → 0（碰撞丢                           
        """
        if not self.workpiece_obj:
            raise ValueError("plan_init_pose 需要 workpiece_obj（工件 mesh）")
        pim = _load_plan_init_pose()

        # —— 工件级 ctx：按工件复用（同工件多焊缝只建一次 ②③）。注意 _kejian2_build_ctx 内部用
        #    load_config() 读默认 default.yaml；与 Scene 默认 cfg 同源，口径一致。 ——
        if rebuild or self._k2ctx is None or self._k2ctx_key != self.workpiece_obj:
            self._k2ctx = pim._kejian2_build_ctx(self.workpiece_obj)
            self._k2ctx_key = self.workpiece_obj
            self._k2ctx_injected = False           # 新 ctx 为仅工件世界

        # —— 障碍物并入 / 还原 solver 碰撞世界（不改 _kejian2_solve_weld 的碰撞算法） ——
        solver = self._k2ctx["solver"]
        want_obs = bool(include_obstacles and self.obstacles.get(self.seam_id))
        if want_obs:
            self._inject_obstacles_into_solver(solver, verbose=verbose)   # 每次按当前障碍重算合并 ESDF（覆盖体素）
            self._k2ctx_injected = True
        elif self._k2ctx_injected:                       # 之前注过障碍、这次要干净工件世界 → 还原
            solver._build_robot_world()
            self._k2ctx_injected = False

        # —— ⑤「工件+障碍 vs init_free 起步空间无交集」过滤的障碍点云（piece/mesh 系，与工件同框、同 T） ——
        #   取碰撞实体障碍（_obstacle_solid_trimeshes，与注入 solver 的同源；open_cylinder 纯视觉不计），
        #   按 plan_init_workpiece_x_voxel 体素化成稠密点。每次都重设（含 None），避免上一条焊缝的障碍点残留。
        obs_pts = None
        if want_obs:
            chunks = []
            for tm in self._obstacle_solid_trimeshes():
                p = pim._voxelize_mesh_points(np.asarray(tm.vertices, dtype=np.float64),
                                              np.asarray(tm.faces, dtype=np.int64).reshape(-1, 3),
                                              self.cfg.plan_init_workpiece_x_voxel)
                if p is not None and len(p):
                    chunks.append(np.asarray(p, dtype=np.float64))
            if chunks:
                obs_pts = np.concatenate(chunks, axis=0)
        self._k2ctx["init_free_obstacle_pts"] = obs_pts

        res, _prof, debug_steps = pim._kejian2_solve_weld(self._k2ctx, self.seam, verbose=verbose)
        self.init_pose_debug_steps[self.seam_id] = debug_steps   # 逐步过滤每步通过候选（覆盖本 seam，供 show_init_poses_debug）

        # 正反手分组返回（不做数量上限挑选）：每组各自按工件平移 xyz 差异独立分数降序
        fore = [InitPoseCandidate.from_kejian2(d) for d in res.get("forehand", [])]
        back = [InitPoseCandidate.from_kejian2(d) for d in res.get("backhand", [])]
        
        history_cands = self.init_pose_candidates.get(self.seam_id, {"forehand": [], "backhand": []})
        history_cands["forehand"].extend(self._sort_by_xyz_diversity(fore))
        history_cands["backhand"].extend(self._sort_by_xyz_diversity(back))
        self.init_pose_candidates[self.seam_id] = history_cands
        
        self.init_pose_candidates_length[self.seam_id] = {
            "forehand": len(self.init_pose_candidates[self.seam_id]["forehand"]),
            "backhand": len(self.init_pose_candidates[self.seam_id]["backhand"]),
        }
        return self.init_pose_candidates[self.seam_id]

    @staticmethod
    def _sort_by_xyz_diversity(cands: List["InitPoseCandidate"]) -> List["InitPoseCandidate"]:
        """把同一手别的候选按【工件平移 xyz 差异】独立分数降序排列（差异大的排前面）。

        每个候选的分数 = 它的平移 t 到本组其余所有候选平移的【平均欧氏距离】（米）；分越高
        表示越"离群/铺得开"，排在越前。只看 T_workpiece_in_base 的平移 t（不看旋转、不掺手别，
        因手别已分组）。≤1 个时原样返回。稳定排序（同分保持原相对次序）。
        """
        n = len(cands)
        if n <= 1:
            return list(cands)
        ts = np.stack([np.asarray(c.t, dtype=np.float64).reshape(3) for c in cands])   # (n,3)
        # 两两欧氏距离矩阵 → 每行均值（排除自身：除以 n-1）
        d = np.linalg.norm(ts[:, None, :] - ts[None, :, :], axis=2)                    # (n,n)
        score = d.sum(axis=1) / float(n - 1)
        order = sorted(range(n), key=lambda i: -float(score[i]))                       # 分数降序、稳定
        return [cands[i] for i in order]

    def _obstacle_solid_trimeshes(self):
        """当前 self.obstacles → 可并入碰撞 ESDF 的【实体 trimesh】列表。

        · 类型2 遮挡板（plate/triangle/trapezoid）：闭合薄棱柱 mesh → 三角化实体；
        · 类型3 open_box：5 块 Box 墙各建实体 box（保留开口，机械臂可从开口伸入够焊缝）；
        · open_cylinder：纯视觉、零厚度开口管，不进碰撞世界（跳过）。
        每块 fix_normals 保证外向法线（供 igl 缠绕数按组件求和纠符号）。返回可能为空列表。
        """
        out = []
        for ob in self.obstacles.get(self.seam_id, []):
            if ob.otype == 2:
                for mesh in ob.meshes:
                    tm = _polygon_mesh_to_trimesh(mesh)
                    if tm is not None:
                        out.append(tm)
            elif ob.otype == 3 and ob.kind == "open_box":
                for prim in ob.prims:
                    out.append(_box_prim_to_trimesh(prim))
            # open_cylinder：纯视觉，不注入碰撞世界
        for tm in out:
            try:
                tm.fix_normals()
            except Exception:
                pass
        return out

    def _inject_obstacles_into_solver(self, solver, verbose: bool = True):
        """把当前障碍实体 + 工件 mesh 合并重算 signed ESDF，覆盖 solver 的碰撞体素（避障）。

        镜像 InitPoseLookupSolver._compute_esdf 的体素化 + igl 缠绕数纠符号，但：
          · 世界含【工件 mesh + 障碍实体】（障碍走 cuRobo Mesh 的 vertices/faces，无需临时文件）；
          · bbox 取【工件 ∪ 所有障碍】并集（+4 voxel），否则伸出工件范围的障碍会掉出体素网格；
          · igl sign 用合并后的 (V,F)（多个 watertight 组件缠绕数求和）。
        随后用新 ESDF 重建 VoxelGrid 并 world_voxel_coll.update_voxel_data —— solve_one_weld_lookup
        的碰撞查询（robot_world.get_collision_distance）走的正是这张体素，故算法一行不改即自动避障。
        仅工件（无可注入障碍）时直接返回，不动 solver。
        """
        import gt_gen.compat  # noqa: F401  warp shim，须在 import curobo 前
        import torch
        import trimesh as _trimesh
        from curobo.geom.types import WorldConfig, Cuboid, Mesh as CuMesh, VoxelGrid
        from curobo.geom.sdf.world import CollisionCheckerType, WorldCollisionConfig
        from curobo.geom.sdf.world_mesh import WorldMeshCollision
        from curobo.geom.sdf.world_voxel import WorldVoxelCollision
        from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
        gt_gen.compat.apply_trimesh_shim()

        obs_tms = self._obstacle_solid_trimeshes()
        if not obs_tms:
            if verbose:
                print("[scene] 无可注入碰撞世界的障碍（仅工件或仅 open_cylinder）")
            return

        voxel_size = float(solver.voxel_size)

        # 工件 mesh + 障碍合并 mesh 的顶点/面
        wp = _trimesh.load(solver.obj_fp, force="mesh", process=False)
        wp_V = np.asarray(wp.vertices, np.float64)
        wp_F = np.asarray(wp.faces, np.int64).reshape(-1, 3)
        obs_merged = _trimesh.util.concatenate(obs_tms)
        obs_V = np.asarray(obs_merged.vertices, np.float64)
        obs_F = np.asarray(obs_merged.faces, np.int64).reshape(-1, 3)

        # 并集 bbox（工件 ∪ 障碍）+ 4 voxel 余量
        allV = np.vstack([wp_V, obs_V])
        bbox_min = allV.min(axis=0)
        bbox_max = allV.max(axis=0)
        center = (bbox_min + bbox_max) / 2.0
        size = (bbox_max - bbox_min) + 4.0 * voxel_size

        wp_mesh = CuMesh(name="workpiece", pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                         file_path=solver.obj_fp, scale=[1.0, 1.0, 1.0])
        obs_mesh = CuMesh(name="obstacles", pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                          vertices=obs_V.tolist(), faces=obs_F.tolist())
        world = WorldConfig(mesh=[wp_mesh, obs_mesh])
        coll_cfg = WorldCollisionConfig.load_from_dict(
            {"checker_type": CollisionCheckerType.MESH, "max_distance": 5.0,
             "n_envs": 1, "cache": {"mesh": 2, "obb": 2}},
            world, solver.tensor_args)
        mesh_coll = WorldMeshCollision(coll_cfg)

        bbox_cuboid = Cuboid(
            name="scene_bbox",
            pose=[float(center[0]), float(center[1]), float(center[2]), 1.0, 0.0, 0.0, 0.0],
            dims=size.tolist())
        esdf = mesh_coll.get_esdf_in_bounding_box(bbox_cuboid, voxel_size=voxel_size)

        # igl 广义缠绕数纠 sign（工件 + 障碍合并 V,F；正=内→碰撞）
        try:
            import igl
            xyzr = esdf.create_xyzr_tensor(transform_to_origin=True, tensor_args=solver.tensor_args)
            voxel_centers = xyzr[:, :3].cpu().numpy().astype(np.float64)
            V = np.vstack([wp_V, obs_V])
            F = np.vstack([wp_F, obs_F + len(wp_V)])
            wn = igl.fast_winding_number(V, F, voxel_centers)
            inside_mask = wn > 0.5
            unsigned = esdf.feature_tensor.abs()
            inside_t = torch.from_numpy(inside_mask).to(unsigned.device)
            sign = torch.where(inside_t, torch.ones_like(unsigned), -torch.ones_like(unsigned))
            esdf.feature_tensor = sign * unsigned
            n_in = int(inside_mask.sum()); n_tot = inside_mask.size
            if verbose:
                print(f"[scene] 障碍并入 ESDF：{len(obs_tms)} 块实体，igl inside "
                      f"{n_in}/{n_tot} voxels（{100.0 * n_in / max(n_tot, 1):.1f}%）")
        except ImportError:
            print("[scene][warn] libigl 未安装，障碍 ESDF sign 沿用 cuRobo 默认")
        except Exception as e:
            print(f"[scene][warn] igl winding number 失败: {e}；用 cuRobo 默认 sign")

        # 覆盖 solver 的碰撞体素并【重建】world_voxel_coll + robot_world（并集 bbox 比工件-only 大，
        # update_voxel_data 只能原地更新同尺寸网格，故这里整体重建；镜像 _build_robot_world 的建法）。
        solver._esdf_feature = esdf.feature_tensor.clone()
        solver._esdf_dims = list(esdf.dims)
        solver._esdf_center_world = center.copy()
        new_voxel = VoxelGrid(
            name="workpiece", dims=solver._esdf_dims,
            pose=[float(center[0]), float(center[1]), float(center[2]), 1.0, 0.0, 0.0, 0.0],
            voxel_size=voxel_size, feature_tensor=solver._esdf_feature)
        world_voxel = WorldConfig(voxel=[new_voxel])
        voxel_cfg = WorldCollisionConfig.load_from_dict(
            {"checker_type": CollisionCheckerType.VOXEL, "max_distance": 5.0, "n_envs": 1},
            world_voxel, solver.tensor_args)
        solver.world_voxel_coll = WorldVoxelCollision(voxel_cfg)
        solver.world_voxel_coll.update_voxel_data(new_voxel)
        rwconfig = RobotWorldConfig.load_from_config(
            solver.robot_cfg, None, collision_activation_distance=0.0,
            collision_checker_type=CollisionCheckerType.VOXEL,
            world_collision_checker=solver.world_voxel_coll, tensor_args=solver.tensor_args)
        solver.robot_world = RobotWorld(rwconfig)

    # ------------------------------------------------------------------
    # 当前 init pose + 观测位姿（goal pose）求解
    # ------------------------------------------------------------------
    def set_init_pose(self, hand: str, index: int) -> InitPoseCandidate:
        """把当前焊缝候选 init_pose_candidates[seam_id][hand] 的第 index 个设为【当前 init pose】。

        让 Scene「知道当前工件相对机器人怎么摆」：填 self.cur_init_pose / cur_init_hand /
        cur_init_index，并把 self.workpiece_pose 设为该候选的 base 系 pose7。后续 compute_goal_pose
        （求观测位姿）与 Open3DSceneVisualizer.show_scene_isaacsim（画机械臂/工件/goal）都从这个
        当前 init pose 取工件↔base 相对摆放 T_workpiece_in_base。

        前提：先 plan_init_pose() 求出候选。hand 须为 "forehand"/"backhand"，index 越界报错。
        返回选定的 InitPoseCandidate。
        """
        cands = self.init_pose_candidates.get(self.seam_id, {"forehand": [], "backhand": []})
        if not isinstance(cands, dict) or hand not in cands:
            raise ValueError(f"hand 须为 'forehand'/'backhand'，收到 {hand!r}；"
                             f"可用手别 {list(cands) if isinstance(cands, dict) else '无候选'}")
        lst = cands[hand]
        n = len(lst)
        if n == 0:
            raise RuntimeError(f"手别 {hand!r} 无候选 init pose：请先 plan_init_pose() 或换另一只手")
        if not (-n <= int(index) < n):
            raise IndexError(f"init pose 候选下标越界：hand={hand} index={index}，该手别共 {n} 个候选")
        cand = lst[int(index)]
        self.cur_init_pose = cand
        self.cur_init_hand = hand
        self.cur_init_index = int(index) % n
        self.workpiece_pose = cand.workpiece_pose7          # 工件在 base 系 pose7
        return cand

    def _trajectory_key(self) -> Tuple[str, int]:
        """当前 init pose 派生的 trajectories 二级 key：(手别, 候选下标)。"""
        if self.cur_init_hand is None or self.cur_init_index is None:
            raise RuntimeError("_trajectory_key 需要当前 init pose：请先 set_init_pose(hand, index)")
        return (self.cur_init_hand, self.cur_init_index)

    def add_trajectory_entry(self, entry: dict) -> None:
        """把一条边走边看轨迹 entry 追加进 self.trajectories[seam_id][当前 init pose key]。"""
        key = self._trajectory_key()
        self.trajectories.setdefault(self.seam_id, {}).setdefault(key, []).append(entry)

    @staticmethod
    def merge_trajectory_entries(entries: List[dict]) -> dict:
        """把同一 init pose 下按序规划的多段 entry 合并成 1 条完整轨迹 entry。

        entries 须按规划顺序排列：entries[i]["positions"][0] == entries[i-1]["positions"][-1]
        （后一段从前一段实际到达的末关节角起步，见 compute_pose_and_plan_path），故拼接时每段
        （除第一段）都跳过开头这个重复路点。observe/goal（各 (n,1)，逐行对齐 positions）按同样
        规则拼接：observe 标该行是否被 _observe 拍过，goal 标该行是否是某段的到达目标（一条合并
        轨迹依次到达 B 个观测位姿，故 goal 列可能有多个 1）。
        """
        positions = np.asarray(entries[0]["positions"], float)
        observe = np.asarray(entries[0]["observe"], np.int64)
        goal = np.asarray(entries[0]["goal"], np.int64)
        for e in entries[1:]:
            positions = np.concatenate([positions, np.asarray(e["positions"], float)[1:]], axis=0)
            observe = np.concatenate([observe, np.asarray(e["observe"], np.int64)[1:]], axis=0)
            goal = np.concatenate([goal, np.asarray(e["goal"], np.int64)[1:]], axis=0)
        statuses = [e.get("status") for e in entries]
        status = "reached" if all(s == "reached" for s in statuses) else next(
            s for s in statuses if s != "reached")
        return dict(
            positions=positions, status=status,
            observe=observe, goal=goal,
            goal_index=[e.get("goal_index") for e in entries],
            variant=entries[0].get("variant"),
            cur_joints=entries[0].get("cur_joints"),
            goal_joints=entries[-1].get("goal_joints"),
            goal_source=entries[-1].get("goal_source"),
            info=[e.get("info") for e in entries],
            segments=len(entries),
        )

    # ------------------------------------------------------------------
    # 存盘 / 读盘（数据态；供跨进程「算 goal pose → save → 另进程 load → 可视化」）
    # ------------------------------------------------------------------
    def save(self, path: str) -> str:
        """把当前 Scene 的【数据状态】pickle 存盘，供【另一个干净进程】load 后可视化。

        动机：compute_goal_pose 会 import/初始化 warp(1.13)+curobo，污染本进程；而
        show_scene_isaacsim 的 SimulationApp 必须在【未加载 warp】的干净进程里最先启动，二者
        不能同进程先后跑（见 scripts/demo_scene.py 说明）。故：进程①算 goal pose 后 save；
        进程②（新起，未碰 warp）Scene.load 再 show_scene_isaacsim。

        只存可序列化数据：cfg（Config=纯 dict dataclass）、当前焊缝、候选、当前 init pose、障碍、
        goal_poses（torch 张量转 numpy）、retract 等；**不存** cuRobo/torch 句柄（load 后为惰性 None）。
        """
        import os
        import pickle
        import numpy as np

        def _to_np(v):
            return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)

        state = dict(
            _scene_save_version=1,
            cfg=self.cfg,                              # Config：raw/robot_cfg 皆 dict，可 pickle
            workpiece_obj=self.workpiece_obj,
            weld_json=self.weld_json,
            seam_id=self.seam_id,
            seam=getattr(self, "seam", None),          # 当前焊缝 dict（_set_cur_seam 设的）
            workpiece_pose=self.workpiece_pose,
            goal_user=self.goal_user,
            cur_cfg=list(self.cur_cfg),
            init_pose_candidates=self.init_pose_candidates,   # {seam_id: {"forehand":[...],"backhand":[...]}}（InitPoseCandidate dataclass）
            init_pose_debug_steps=self.init_pose_debug_steps,  # {seam_id: [step1..step8]} 逐步过滤每步通过候选（raw dict）
            cur_init_pose=self.cur_init_pose,
            cur_init_hand=self.cur_init_hand,
            cur_init_index=self.cur_init_index,
            obstacles=self.obstacles,                  # {seam_id: list[ObstacleSpec]}（dataclass；prims=Box、meshes=dict）
            goal_poses={sid: ({k: _to_np(v) for k, v in res.items()} if res else None)
                        for sid, res in self.goal_poses.items()},   # {seam_id: dict|None}
            trajectories={sid: {k: list(v2) for k, v2 in d.items()}
                          for sid, d in self.trajectories.items()},   # {seam_id: {(hand,index): list[dict]}}（positions 等均 numpy，可直接 pickle）
            trajectory_goal_poses={sid: {k: ({kk: _to_np(vv) for kk, vv in res.items()} if res else None)
                                          for k, res in d.items()}
                                   for sid, d in self.trajectory_goal_poses.items()},   # {seam_id: {(hand,index): goal_pose结果}}
        )
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)
        cur_cands = self.init_pose_candidates.get(self.seam_id, {})
        n_f = len(cur_cands.get("forehand", []))
        n_b = len(cur_cands.get("backhand", []))
        cur = ("%s#%d" % (self.cur_init_hand, self.cur_init_index)
               if self.cur_init_index is not None else "未设")
        print(f"[scene] 已保存 → {path}（候选 正手{n_f}/反手{n_b}，"
              f"障碍 {sum(len(v) for v in self.obstacles.values())}，goal_poses {sum(1 for v in self.goal_poses.values() if v)}，"
              f"轨迹 {sum(len(v2) for d in self.trajectories.values() for v2 in d.values())}，"
              f"当前 init pose={cur}）")
        return path

    @classmethod
    def load(cls, path: str) -> "Scene":
        """从 save() 的存盘重建 Scene（数据状态）。cuRobo/torch 句柄不恢复（惰性=None），
        可直接喂 Open3DSceneVisualizer.show_scene_isaacsim 可视化。

        注意：需在【未加载 warp】的干净进程里调用后再启动 SimulationApp（这正是 save/load 的目的）。
        weld_json 路径须仍可读（__init__ 会重读焊缝，随后用存盘的当前焊缝覆盖）。
        """
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        self = cls(cfg=state["cfg"], workpiece_obj=state["workpiece_obj"],
                   weld_json=state["weld_json"])
        if state.get("seam") is not None:
            self.seam = state["seam"]
            self.seam_id = state["seam_id"]
        self.workpiece_pose = state.get("workpiece_pose")
        self.goal_user = state.get("goal_user")
        self.cur_cfg = list(state.get("cur_cfg", self.cur_cfg))
        self.init_pose_candidates = state.get("init_pose_candidates", {}) or {}
        self.init_pose_debug_steps = state.get("init_pose_debug_steps", {}) or {}
        self.cur_init_pose = state.get("cur_init_pose")
        self.cur_init_hand = state.get("cur_init_hand",
                                       getattr(self.cur_init_pose, "hand", None))
        self.cur_init_index = state.get("cur_init_index")
        self.obstacles = state.get("obstacles", {}) or {}
        self.goal_poses = state.get("goal_poses", {}) or {}
        self.trajectories = state.get("trajectories", {}) or {}   # {seam_id: {(hand,index): list[dict]}}
        self.trajectory_goal_poses = state.get("trajectory_goal_poses", {}) or {}   # {seam_id: {(hand,index): goal_pose结果}}
        cur_cands = self.init_pose_candidates.get(self.seam_id, {})
        n_f = len(cur_cands.get("forehand", []))
        n_b = len(cur_cands.get("backhand", []))
        cur = ("%s#%d" % (self.cur_init_hand, self.cur_init_index)
               if self.cur_init_index is not None else "未设")
        print(f"[scene] 已加载 ← {path}（候选 正手{n_f}/反手{n_b}，"
              f"障碍 {sum(len(v) for v in self.obstacles.values())}，goal_poses {sum(1 for v in self.goal_poses.values() if v)}，"
              f"轨迹 {sum(len(v2) for d in self.trajectories.values() for v2 in d.values())}，"
              f"当前 init pose={cur}）")
        return self

    def _seam_data_arrays(self, n_seg: int = None):
        """本焊缝 → 观测位姿优化器要的 (seam_line, seam_tangent, seam_limits)，均在【工件 mesh 系】。

        与 gt_overall per-seam pkl（compute_goal_poses2 的 seam_data）同框同语义：
          · seam_line   (N,3)   焊缝折线（两端点插值）；
          · seam_tangent(N,3)   焊缝切线（单位，直缝→逐点相同）；
          · seam_limits (N,2,3) 两面表面方向 d1/d2（bisector = unit(d1+d2)，与 visionOrientation 口径一致）。
        n_seg 默认取 cfg.plan_init_pose.n_seg（与 save_seam_pkl 一致，缺省 20）。
        """
        if n_seg is None:
            try:
                n_seg = int(self.cfg.raw["plan_init_pose"]["n_seg"])
            except Exception:
                n_seg = 20
        mid, t, d1, d2, bis, seam_len, seam_line = self._seam_frame(n_seg=n_seg)
        N = seam_line.shape[0]
        seam_tangent = np.tile(_unit(t)[None, :], (N, 1))               # (N,3)
        seam_limits = np.tile(np.stack([_unit(d1), _unit(d2)], axis=0)[None], (N, 1, 1))  # (N,2,3)
        return seam_line, seam_tangent, seam_limits

    def fine_box_from_seam(self, margin_m: float = None):
        """当前 init pose 下，本焊缝折线的 base 系 AABB 外扩 margin → (center(3,), dims(3,))。

        供 build_roi_multires_voxmap 的 fine 盒来源：seam_line（_seam_data_arrays，工件 mesh 系）
        经 cur_init_pose.T_workpiece_in_base（p_base=T·p_world）变换到 base 系后取 AABB。
        margin_m 默认读 cfg.fine_box_margin_m（default.yaml roi.fine_box_margin_cm，方案定 10cm）。
        """
        from gt_gen.voxmap import fine_box_from_seam_points
        if margin_m is None:
            margin_m = self.cfg.fine_box_margin_m
        seam_line, _, _ = self._seam_data_arrays()               # 工件 mesh 系 (N,3)
        T = np.asarray(self.cur_init_pose.T_workpiece_in_base, dtype=np.float64)
        pts_base = seam_line @ T[:3, :3].T + T[:3, 3]            # p_base = T · p_world
        return fine_box_from_seam_points(pts_base, margin_m)

    def compute_goal_pose(self,
                          include_obstacles: bool = None,
                          horizontal: int = None,
                          device: str = None,
                          **cfg_overrides) -> list:
        """给定当前 3D 世界（工件 + 障碍）与本焊缝，计算覆盖整条焊缝的【观测位姿序列】(goal pose)。

        忠实复用 scripts/compute_goal_poses2.py 的进化策略优化器（ConfigurationPose / OptimizerPose /
        ScenePose2，一行不改），只把它的 per-seam 主体包成 Scene 方法：
          · seam_line/seam_tangent/seam_limits 由本焊缝 self.seam 造（_seam_data_arrays，工件 mesh 系）；
          · 工件↔机器人相对摆放取【当前 init pose】self.cur_init_pose：piece_pose=identity、
            robot_pose = base 在工件 mesh 系的 pose7 = pose7(inv(T_workpiece_in_base))；
          · include_obstacles=True（默认）把已放障碍（类型2遮挡板 / 类型3 open_box 实体）与工件一起
            update_world 进 ScenePose2 的 cuRobo 碰撞世界（obstacle-aware，open_cylinder 纯视觉跳过），
            使观测位姿的碰撞过滤把障碍算进去；False 则仅工件。

        **参数来源**：默认全部读 configs/default.yaml 的 `compute_goal_pose` 段——
          · 方法级 3 键 include_obstacles / horizontal / device（显式传参 > yaml > 内置默认）；
          · 其余键透传到 ConfigurationPose（凡与其属性同名即 setattr），**cfg_overrides 亦可临时覆盖；
            num_randoms_all / num_envs 依最终 num_batches/num_randoms_* 自动重算。
        ⚠ ES 采样规模（num_batches/num_randoms_new）调太小会触发优化器内部假设崩溃（compute_goal_poses2
          既有行为），默认 8×100 稳定。horizontal：0=水平 / 1=垂直 / 其它=all（全范围）。

        前提：先 set_init_pose(hand, index) 选定当前 init pose（否则报错）。
        返回：dict（含 cam_pose (K,B,7)/joints (K,B,DOF)/start_pts/end_pts/robot_pose_rel），
              同时写入 self.goal_poses[self.seam_id]。无解则返回 None（并写入 None）。

        **输出张量三维语义** `(K, B, DOF)`（以 joints (160,1,6) 为例，见 optimizer_pose.py）：
          · DOF=6：UR12e 6 轴关节角（cam_pose 末维为 7=pos3+quat4）。
          · B：覆盖整条焊缝所需的观测位姿个数。B=1 表示这条焊缝一个视角即可看完；B>1 则是
            必须按顺序访问的多视角序列。
          · K（例中 160）：候选解变体数，**并非** K 个独立最优解，而是「精修快照 × 选出的候选批」：
              1) ES 优化后 selectOutputs() 从 num_batches(默认 8) 个并行批里挑出「已完成且覆盖位姿数
                 等于最少值」的 A0 个候选批（stack 成第一维）；
              2) refine=True 时 refineOutputs() 对这 A0 个候选继续精修 max_iterations_refine(默认 40)
                 步，每 span(默认 1) 步存一次快照，共 num_snapshots 个；
              3) 结尾 torch.cat(..., dim=0) 把快照沿第 0 维拼起来 → K = num_snapshots × A0
                 （例：40 × 4 = 160）。拼接前 list.reverse()，故 dim0 前段是最后（最收敛）的迭代。
          用的时候第一维 K 通常任取其一（如 0）即可，它们都是收敛后的合格解；start_pts/end_pts 也按
          同样的 (K, B) 展开，第一维含义一致。
        """
        if self.cur_init_pose is None:
            raise RuntimeError("compute_goal_pose 需要当前 init pose：请先 set_init_pose(hand, index)")
        import torch

        # —— 参数：configs/default.yaml 的 compute_goal_pose 段（显式传参/**cfg_overrides 可覆盖）——
        sec = dict(self.cfg.raw.get("compute_goal_pose", {}) or {})
        if include_obstacles is None:
            include_obstacles = bool(sec.get("include_obstacles", True))
        if horizontal is None:
            horizontal = int(sec.get("horizontal", 2))
        if device is None:
            device = str(sec.get("device", "cuda"))
        # 障碍膨胀量(米,方法级)：显式 **cfg_overrides > yaml 段 > 0；只作用碰撞世界（见 _inject_obstacles_into_scenepose2）
        obstacle_buffer_m = float(cfg_overrides.get("obstacle_buffer_m",
                                                    sec.get("obstacle_buffer_m", 0.0)) or 0.0)

        cgp = _load_compute_goal_poses2()
        Configuration = cgp.Configuration
        Optimizer = cgp.Optimizer
        ScenePose2 = cgp.Scene

        # —— ES 配置：ConfigurationPose 默认 + 从 yaml 段（及 **cfg_overrides）透传同名字段 ——
        cfg = Configuration()
        cfg.usd_path = ""
        cfg.pc_path = ""
        _method_keys = {"include_obstacles", "horizontal", "device", "obstacle_buffer_m"}
        for k, v in {**sec, **cfg_overrides}.items():
            if k in _method_keys:
                continue
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                print(f"[scene][warn] compute_goal_pose：ConfigurationPose 无字段 {k!r}，忽略")
        # 派生量按最终 num_batches/num_randoms_* 重算（口径同 compute_goal_poses2.main）
        cfg.num_randoms_all = cfg.num_randoms_new + cfg.num_randoms_old + 1
        cfg.num_envs = cfg.num_batches * (cfg.num_randoms_new + cfg.num_randoms_old)

        # —— 焊缝数据（工件 mesh 系）——
        seam_line_np, seam_tangent_np, seam_limits_np = self._seam_data_arrays()
        seam_line = torch.as_tensor(seam_line_np, dtype=torch.float, device=device)
        seam_tangent = torch.as_tensor(seam_tangent_np, dtype=torch.float, device=device)
        seam_limits = torch.as_tensor(seam_limits_np, dtype=torch.float, device=device)

        # —— 工件↔机器人相对摆放：piece 在原点(identity)，robot base = 工件系下 inv(T_workpiece_in_base) ——
        pim = _load_plan_init_pose()
        T = self.cur_init_pose.T_workpiece_in_base
        robot_pose7 = np.asarray(pim.mat44_to_pose7(np.linalg.inv(T)), dtype=np.float64)  # base 在 mesh 系
        piece_pose7 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        robot_pose_t = torch.as_tensor(robot_pose7, dtype=torch.float, device=device)
        piece_pose_t = torch.as_tensor(piece_pose7, dtype=torch.float, device=device)

        scene2 = ScenePose2(cfg, num_envs=cfg.num_envs, device=device,
                            obj_path=self.workpiece_obj, robot_cfg_path=self.cfg.robot_cfg_path)

        want_obs = bool(include_obstacles and self.obstacles.get(self.seam_id))
        # reset：把工件按 piece->base_link 摆进碰撞世界并选关节限位
        robot_pose_rel = scene2.reset(robot_pose_t, int(horizontal), piece_pose_t)
        if want_obs:
            self._inject_obstacles_into_scenepose2(scene2, buffer_m=obstacle_buffer_m)     # 障碍与工件同 pose 一起进碰撞世界（避障；buffer_m>0 膨胀）

        optimizer = Optimizer(cfg=cfg, scene=scene2, device=device)
        optimizer.resetSeamData(seam_line, seam_tangent, seam_limits)
        cam_pose, joints, start_pts, end_pts = optimizer.solve()

        result = None
        if cam_pose is None:
            print("[scene] compute_goal_pose：无观测位姿解")
        else:
            result = {
                "cam_pose": cam_pose.detach().clone(),
                "joints": joints.detach().clone(),
                "start_pts": start_pts.detach().clone(),
                "end_pts": end_pts.detach().clone(),
                "robot_pose_rel": robot_pose_rel.detach().clone(),
            }
            print(f"[scene] compute_goal_pose 成功：cam_pose {tuple(cam_pose.shape)} "
                  f"joints {tuple(joints.shape)}（障碍并入={want_obs}）")
        self.goal_poses[self.seam_id] = result
        return result
    
    def compute_pose_and_plan_path(self, hand, include_obstacles: bool = True,
                                   device: str = None):
        """对某手别的每个候选 init pose，求观测位姿序列后【按序边走边看规划】覆盖整条焊缝的 GT。

        流程（每个候选 init pose）：
          set_init_pose → compute_goal_pose（得【单个】结果，joints 形如 (K,B,DOF)）→
          对选定的 variant（K 维，默认仅第 0 个）：
            · 造一张全新三态体素图 vm（含初始 FREE 冷启动立足之地）；
            · 按 pose 顺序 0..B-1 依次 plan_explore_path：pose[0] 从初始关节角(retract)起步，
              pose[i] 从 pose[i-1]【实际到达的末关节角】起步，且全程共享同一张 vm →
              后一个 pose 继承前一个 pose 边走边看观测到的世界；
            · B 个 pose 全部 reached 才算该序列成功，任一失败即中断该 variant。
        任一 variant 的整条序列成功即返回 True；全部候选/变体都失败返回 False。

        参数：
          hand             : "forehand"/"backhand"。
          include_obstacles: 真值世界是否并入已放障碍（默认 True）。
          device           : cuda/cpu（None→cfg.compute_goal_pose.device）。
          variants         : 要尝试的 K 变体——None(默认)=仅 variant 0；int=仅该变体；"all"=全部 K；
                             可迭代=指定若干。（K 是收敛后近等价快照，通常取 0 即可，全跑很重。）
        """
        cur_cands = self.init_pose_candidates.get(self.seam_id, {"forehand": [], "backhand": []})
        for init_pose_idx in range(len(cur_cands[hand])):
            # 最多尝试2个初始位姿
            if init_pose_idx >= 2:
                continue

            self.set_init_pose(hand, init_pose_idx)
            res = self.compute_goal_pose()
            if res is None:                              # 该 init pose 无观测位姿解
                continue

            # self.save('/media/a/新加卷/tempt/4/scene1.pkl')

            jt = res["joints"]
            joints = jt.detach().cpu().numpy() if hasattr(jt, "detach") else np.asarray(jt)
            K, B = joints.shape[:2]

            # 只使用第1个变体
            variant_list = [0]

            # 静态探索世界（工件+障碍摆放）只随 init pose 变，整批 variant 共用一套（省重型 handle）
            ctx = self._build_explore_world(include_obstacles=include_obstacles, device=device)

            for variant in variant_list:
                vm = self._fresh_explore_voxmap(ctx)     # 每个 variant 重置探索状态，互不串扰
                start = list(self.cur_cfg)               # pose[0] 从初始关节角（retract）起步
                all_reached = True
                seq_entries = []                          # 本序列按 pose 顺序产出的分段 entry，成功后合并成 1 条
                for pose_idx in range(B):                # 按序覆盖 B 个观测位姿
                    entry = self.plan_explore_path(
                        goal_index=pose_idx, variant=variant,
                        cur_joints=start, ctx=ctx, vm=vm)   # 共享 vm ＝ 继承前一个 pose 的观测
                    seq_entries.append(entry)
                    if entry["status"] != "reached":
                        all_reached = False
                        print(f"[scene] 序列中断：init#{init_pose_idx} variant#{variant} "
                              f"pose#{pose_idx}/{B} status={entry['status']}")
                        break
                    start = entry["positions"][-1]        # 下一个 pose 从实际到达的末关节角起步
                if all_reached:
                    merged = self.merge_trajectory_entries(seq_entries)
                    key = self._trajectory_key()
                    self.trajectories.setdefault(self.seam_id, {})[key] = [merged]
                    self.trajectory_goal_poses.setdefault(self.seam_id, {})[key] = res   # 快照当前候选的观测位姿结果，
                                                                                          # 供 show_trajectory_isaacsim 还原对应视锥
                    print(f"[scene] 序列成功：init#{init_pose_idx} variant#{variant} "
                          f"覆盖 B={B} 个观测位姿，已合并为 1 条轨迹（路点={len(merged['positions'])}）")
                    return True
            # 本候选所有 variant 都失败：清掉 plan_explore_path 遗留的分段 entry（未 reached、未合并），
            # 避免污染 self.trajectories 导致 show_trajectory_isaacsim 拍平后取错轨迹
            self.trajectories.get(self.seam_id, {}).pop(self._trajectory_key(), None)
            self.trajectory_goal_poses.get(self.seam_id, {}).pop(self._trajectory_key(), None)
        return False

    def plan_explore_path(self,
                          goal_index: int = 0,
                          cur_joints=None,
                          variant: int = 0,
                          include_obstacles: bool = True,
                          device: str = None,
                          ctx: Ctx = None,
                          vm=None) -> dict:
        """从【当前机械臂关节角】边走边看规划一条到 goal 关节角目标的探索轨迹（GT）。

        忠实复用 scripts/place_obstacles_to_gt2.py 的主体（gt_gen.main_loop.generate_gt，一行不改），
        只把它「读 Scene → 建世界 → 跑主循环」的流程包成 Scene 方法，数据源即本 Scene：
          · h_truth（MESH 真值世界，全知教练）= 工件 mesh + 已放障碍实体（_obstacle_solid_trimeshes），
            都按【当前 init pose】的 workpiece_pose7 摆到 base 系；
          · h_expl（VOXEL 三态探索世界）+ 三态体素图 vm + 初始 FREE 空间（冷启动立足之地，
            按 cfg.init_free_method_for_init 选 box 或圆柱）；
          · truth_scene（base 系 trimesh，raycast 几何源）= 工件 + 障碍 合并（同一 T_workpiece_in_base）；
          · goal —— 【直接用关节角当目标】：goal_joints = joints[variant, goal_index]
            （compute_goal_pose 求得的第 variant 个变体、第 goal_index 个观测位姿对应的可达构型），
            作为 generate_gt 的 goal_cfg 传入——STOMP 后端步①(直达)/步②(P*) 走 plan_joint_single
            直接规划到该关节角，不再解 IK / 换算焊枪位姿。goal_pose 传 None，由 generate_gt 内部
            FK(goal_cfg) 补出供 NBV 用（与关节目标自洽）。
        机械臂从 cur_joints（缺省=self.cur_cfg，通常 retract）起步，只敢走「亲眼看过是空的」区域，
        边走边拍、已知区像水面扩大，直到规划到 goal（reached）或触发主循环终止条件。

        实现上把起点喂给 generate_gt 的办法：起点 cur_joints 作为 generate_gt 的 start_cfg 参数传入
        （缺省时 generate_gt 用 cfg.retract_config）。

        前提：先 set_init_pose(hand, index)（定义工件↔base 摆放）+ compute_goal_pose（求 goal 观测位姿序列）。
        ⚠ 会 import/初始化 warp+curobo，污染本进程；须在【未启动 SimulationApp 的进程】里调用。
          可视化：本方法后 scene.save(path)，另起干净进程 Scene.load 再 show_trajectory_isaacsim。

        参数：
          goal_index       : goal 序列（B 个覆盖观测位姿）里第几个作为终点（支持负索引）。
          cur_joints       : 机械臂当前关节角（rad，list/ndarray，长度=DOF）；None→用 self.cur_cfg。
          variant          : joints 的第几个变体 K（默认 0）。
          include_obstacles: 真值世界是否并入已放障碍（默认 True；open_cylinder 纯视觉不并入）。
          device           : cuda/cpu（None→cfg.compute_goal_pose.device 或 'cuda'）。
          ctx              : 可选【静态探索世界】(_build_explore_world 产出：h_truth/h_expl/truth_scene/
                             cam/world/device)；None→内部按当前 init pose 现建一套（单独调用即用此路径）。
                             同一 init pose 的整条观测位姿序列传同一个 ctx 可省去重复建重型 handle。
          vm               : 可选【三态体素图】；None→内部 _fresh_explore_voxmap(ctx) 造全新的（含初始 FREE）。
                             序列内多个 pose 传【同一张 vm】即可继承前一次规划观测到的世界（见 compute_pose_and_plan_path）。
        返回 dict（同时 append 进 self.trajectories[seam_id][(当前 hand, 当前 index)]）：
          {positions(T,DOF), status, goal_index, variant, cur_joints, goal_joints, goal_source, info}。
          （compute_pose_and_plan_path 序列成功后会把同一 key 下按序产出的多段 entry 合并成 1 条，
          见 Scene.merge_trajectory_entries；单独调用本方法时该 key 下会保留分段的多条 entry。）
        """
        if self.cur_init_pose is None:
            raise RuntimeError("plan_explore_path 需要当前 init pose：请先 set_init_pose(hand, index)")
        if self.goal_poses.get(self.seam_id) is None:
            raise RuntimeError("plan_explore_path 需要 goal pose：请先 compute_goal_pose()")

        from gt_gen.main_loop import generate_gt

        # —— 静态探索世界：缺省现建一套（单独调用即走此路径）；序列驱动时由调用方传同一个 ctx 复用 ——
        if ctx is None:
            ctx = self._build_explore_world(include_obstacles=include_obstacles, device=device)
        h_truth, h_expl = ctx["h_truth"], ctx["h_expl"]
        truth_scene, cam, world = ctx["truth_scene"], ctx["cam"], ctx["world"]

        # —— goal joints：compute_goal_pose 的 joints[variant, goal_index]（形如 (K,B,DOF)） ——
        res = self.goal_poses[self.seam_id]              # 单个 dict（compute_goal_pose 产出）
        jt = res["joints"]
        joints = jt.detach().cpu().numpy() if hasattr(jt, "detach") else np.asarray(jt)
        K, B = joints.shape[:2]
        vi = int(variant)
        gi = int(goal_index)
        if not (-K <= vi < K):
            raise IndexError(f"variant 越界：{variant}，该结果共 {K} 个变体")
        if not (-B <= gi < B):
            raise IndexError(f"goal_index 越界：{goal_index}，该序列共 {B} 个观测位姿")
        goal_joints = np.asarray(joints[vi, gi], float).tolist()

        # —— 起点关节角（缺省=self.cur_cfg）；作为 start_cfg 传入 generate_gt（无需覆盖 retract_config）——
        start = [float(v) for v in (self.cur_cfg if cur_joints is None else cur_joints)]

        # —— 三态体素图：缺省造全新的（含初始 FREE 冷启动立足之地）；序列内传【同一张 vm】即继承前次观测 ——
        if vm is None:
            vm = self._fresh_explore_voxmap(ctx)

        # goal=关节角目标：直接把 goal_joints 交给 generate_gt 的 goal_cfg（STOMP 走 plan_joint_single
        # 直达该关节角，不再解 IK / 换算焊枪位姿）；goal_pose 传 None，由 generate_gt 内部 FK(goal_cfg) 补出供 NBV 用。
        print(f"[scene] goal=关节角目标（观测位姿#{gi}/{B} 变体#{vi}/{K}）goal_joints={np.round(goal_joints, 4)}")
        print("[scene] 开跑 generate_gt 主循环（边走边看）...")

        # world_plan=world：backend=stomp 时步② 规划 P* 需要 MESH 世界（真实尺寸，无 buffer）
        # goal_pose=None + goal_cfg=goal_joints：以关节角为目标（见 place_obstacles_to_gt2）
        GT, status, info = generate_gt(h_truth, h_expl, vm, truth_scene, None,
                                       camera_model=cam, world_plan=world,
                                       goal_cfg=goal_joints, start_cfg=start)

        positions = np.asarray([np.asarray(q, float) for q in GT])
        observe = np.asarray(info.get("observe", []), dtype=np.int64).reshape(-1, 1)
        goal = np.asarray(info.get("goal", []), dtype=np.int64).reshape(-1, 1)
        entry = dict(positions=positions, status=status, goal_index=gi, variant=vi,
                     cur_joints=np.asarray(start, float),
                     goal_joints=np.asarray(goal_joints, float),
                     goal_source="joint_target_K_variant", info=info,
                     observe=observe, goal=goal)
        self.add_trajectory_entry(entry)
        key = self._trajectory_key()
        print(f"[scene] plan_explore_path 完成：status={status} 路点={len(positions)} "
              f"（当前 init pose{key} 第 {len(self.trajectories[self.seam_id][key])} 段）")
        return entry

    def _build_explore_world(self, include_obstacles: bool = True, device: str = None) -> dict:
        """构建【只依赖当前 init pose（工件摆放 + 障碍）】的静态探索世界，供一条观测位姿序列内的
        多次 plan_explore_path 复用（省去重复建重型 handle，也缓解显存）。

        与 plan_explore_path 原内联构建完全一致：工件 mesh + 障碍实体按当前 init pose 的 workpiece_pose7
        摆到 base 系。返回 ctx dict：
          h_truth(MESH 真值 handle) / h_expl(VOXEL 三态 handle) / truth_scene(base 系 trimesh，raycast 源) /
          cam(相机模型) / world(MESH WorldConfig，兼作步② world_plan) / device。
        ⚠ 会 import/初始化 warp+curobo，污染本进程；须在【未启动 SimulationApp 的进程】里调用。
        """
        if self.cur_init_pose is None:
            raise RuntimeError("_build_explore_world 需要当前 init pose：请先 set_init_pose(hand, index)")

        import trimesh as _trimesh
        import gt_gen.compat as _compat  # noqa: F401  warp shim（须在 import curobo 前）
        _compat.apply_trimesh_shim()

        # compute_goal_pose 的 ES 优化器（num_envs≈800）此时已出作用域；显式回收显存，
        # 否则接着建 h_truth+h_expl 两个 MotionGen 易 CUDA OOM（小显存卡尤甚）。
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from gt_gen import curobo_iface as ci
        from gt_gen.sensor import load_camera_model, load_truth_scene
        from curobo.geom.types import WorldConfig, Mesh as CuMesh
        from curobo.geom.sdf.world import CollisionCheckerType

        if device is None:
            device = str(self.cfg.raw.get("compute_goal_pose", {}).get("device", "cuda"))

        # —— base 系摆放：工件 + 障碍实体（顶点在工件 mesh 系）按 workpiece_pose7 一起摆到 base 系 ——
        wp_pose7 = np.asarray(self.cur_init_pose.workpiece_pose7, float).tolist()
        T = np.asarray(self.cur_init_pose.T_workpiece_in_base, float)

        obs_tms = self._obstacle_solid_trimeshes() if include_obstacles else []
        meshes = [CuMesh(name="workpiece", file_path=self.workpiece_obj, pose=wp_pose7)]
        if obs_tms:
            merged = _trimesh.util.concatenate(obs_tms)
            meshes.append(CuMesh(
                name="obstacles",
                vertices=np.asarray(merged.vertices, float).tolist(),
                faces=np.asarray(merged.faces, np.int64).reshape(-1, 3).tolist(),
                pose=wp_pose7))                          # 障碍与工件同 pose → 一并进 base 系
        world = WorldConfig(mesh=meshes)                 # MESH 真值世界（工件+障碍），兼作步② world_plan

        print(f"[scene] _build_explore_world：建 h_truth（MESH，工件 + {len(obs_tms)} 障碍实体）...")
        h_truth = ci.init_curobo(self.cfg, world_model=world,
                                 collision_checker_type=CollisionCheckerType.MESH,
                                 position_threshold=0.05, rotation_threshold=0.5)
        print("[scene] _build_explore_world：建 h_expl（VOXEL 三态）...")
        h_expl = ci.init_curobo(self.cfg)

        # truth_scene（base 系 trimesh）：工件 + 障碍（同一 T 变到 base 系）
        work_mesh = load_truth_scene(self.workpiece_obj, mesh_pose=wp_pose7)
        tms = [work_mesh]
        for tm in obs_tms:
            tmc = tm.copy()
            tmc.apply_transform(T)                       # 工件 mesh 系 → base 系（与工件同一 T）
            tms.append(tmc)
        truth_scene = _trimesh.util.concatenate(tms) if len(tms) > 1 else work_mesh

        cam = load_camera_model(self.cfg)
        return Ctx(h_truth=h_truth, h_expl=h_expl, truth_scene=truth_scene,
                   cam=cam, world=world, device=device)

    def _fresh_explore_voxmap(self, ctx: Ctx):
        """为一条观测位姿序列造一张【全新多分辨率三态体素图 + 初始 FREE（冷启动立足之地）】。

        同一 init pose 的每条序列（每个 variant）各调一次以重置探索状态、互不串扰；序列【内部】的多个
        pose 则共享同一张 vm 累积观测＝继承前一次规划观测到的世界（见 compute_pose_and_plan_path）。
        初始 FREE 按 cfg.init_free_method_for_init 选 box 或圆柱（与 plan_explore_path 原行为一致）。

        多分辨率（方案 缝周多分辨率voxmap）：coarse=整 ROI(vs_coarse) + fine=焊缝 AABB 外扩
        fine_box_margin(vs_fine)；init_free 各方案已支持对 coarse/fine 两子网格各标一遍（_iter_grids）。
        """
        from gt_gen.voxmap import build_roi_multires_voxmap
        from gt_gen.init_free import set_initial_free_cylinder, set_initial_free_box
        h_truth = ctx["h_truth"]
        fine_center, fine_dims = self.fine_box_from_seam()       # 当前 init pose 下焊缝 base 系 AABB 外扩
        vm = build_roi_multires_voxmap(self.cfg, fine_center, fine_dims)
        print(f"[scene] 多分辨率 voxmap：coarse vs={vm.coarse.voxel_size} dims={vm.coarse.shape} + "
              f"fine vs={vm.fine.voxel_size} dims={vm.fine.shape} "
              f"盒中心={np.round(fine_center, 3).tolist()} 尺寸={np.round(fine_dims, 3).tolist()}")
        method = self.cfg.init_free_method_for_init      # cylinder | box（default.yaml init_free.method_for_init）
        if method == "box":
            box_min, box_max = self.cfg.init_free_box_min_for_init, self.cfg.init_free_box_max_for_init
            n_free = set_initial_free_box(h_truth, vm, config=self.cfg, box_min=box_min, box_max=box_max)
            print(f"[scene] 初始 FREE 空间=box {box_min}~{box_max} 体素={n_free}")
        else:
            n_free = set_initial_free_cylinder(h_truth, vm, config=self.cfg)  # 起点末端周围罩 FREE
            print(f"[scene] 初始 FREE 空间=圆柱 体素={n_free}")
        return vm

    def _inject_obstacles_into_scenepose2(self, scene2, buffer_m: float = 0.0):
        """把当前障碍实体（工件 mesh 系）按 piece->base_link 位姿并进 ScenePose2 的碰撞世界，
        并把障碍 mesh（piece 系）并进遮挡 raycast 的 warp mesh。

        buffer_m>0 时【只对碰撞世界】的障碍按 OBB 各维 +2·buffer 膨胀一圈（机械臂避障留安全间隙，
        同 STOMP 绕行 obstacle_buffer_m 语义；工件不膨胀），走 obstacle_placement.inflate_trimesh_obb
        单一来源；遮挡 raycast(visionBlock 视线)仍用【真实尺寸】，避免视线遮挡判定过度保守而更难找位姿。

        两条独立路径都要喂障碍，缺一不可：
          · 碰撞世界 rw（computeCollisionCost）：障碍(可膨胀)与工件同 pose 摆到 base 系，机械臂避障；
          · 遮挡 raycast _wp_mesh（visionBlock）：piece 系(真实尺寸，piece_pos=0，不加 pose)，
            让优化器把“障碍挡住相机→焊缝视线”算进代价，否则会选出视线被障碍遮挡的位姿。
        （_obstacle_solid_trimeshes：类型2遮挡板 + 类型3 open_box；open_cylinder 纯视觉跳过。）不改 scene_pose2.py 主体逻辑。
        """
        import trimesh as _trimesh
        from gt_gen import obstacle_placement as _opl
        obs_tms = self._obstacle_solid_trimeshes()
        if not obs_tms:
            print("[scene] compute_goal_pose：无可注入碰撞的障碍（仅工件或仅 open_cylinder）")
            return
        merged = _trimesh.util.concatenate(obs_tms)                       # 真实尺寸（遮挡 raycast 用）
        # 碰撞世界用的障碍：buffer_m>0 时每块按 OBB 各维 +2·buffer 膨胀（工件不动），否则真实尺寸
        col_tms = [_opl.inflate_trimesh_obb(t, buffer_m) for t in obs_tms] if buffer_m > 0 else obs_tms
        merged_col = _trimesh.util.concatenate(col_tms)
        pose = scene2.robot_base_inv_pose[0].detach().cpu().tolist()    # [x,y,z, qw,qx,qy,qz]
        piece_mesh = scene2._Mesh(name="piece", vertices=scene2._verts_list,
                                  faces=scene2._faces_list, pose=pose)
        obs_mesh = scene2._Mesh(name="obstacles",
                                vertices=np.asarray(merged_col.vertices, float).tolist(),
                                faces=np.asarray(merged_col.faces, np.int64).reshape(-1, 3).tolist(),
                                pose=pose)
        scene2.rw.update_world(scene2._WorldConfig(mesh=[piece_mesh, obs_mesh]))
        # 遮挡视线检测（visionBlock 的 raycast warp mesh）也并入障碍——piece 系、真实尺寸，无需加 pose，
        # 否则优化器眼里障碍“透明”，会选出视线被障碍挡住的观测位姿。
        scene2.set_occluders(np.asarray(merged.vertices, float),
                             np.asarray(merged.faces, np.int64).reshape(-1, 3))
        _tag = f"，碰撞膨胀 OBB +{buffer_m:.3f}m" if buffer_m > 0 else ""
        print(f"[scene] compute_goal_pose：障碍并入碰撞世界 + 遮挡 raycast（{len(obs_tms)} 块实体{_tag}）")

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
            meta=dict(candidate="C1", anchor=anchor, R=R, b_dir=b_dir, n_cm=float(p["n_cm"]),
                      width=width, length=length, thickness=thickness))
        self.obstacles.setdefault(self.seam_id, []).append(spec)
        return spec

    def grow_obstacle_n(self, n_cm: float, spec: Optional[ObstacleSpec] = None) -> ObstacleSpec:
        """把已放的【类型2遮挡板】沿离缝方向 b_dir 再抬高 n_cm（cm）——等价于把创建时的 n_cm 增大 n_cm。

        n_cm 可正可负：正=沿 b_dir 抬远离缝，负=沿 -b_dir 靠近焊缝。直接平移该障碍全部网格顶点
        delta=n_cm/100 米（沿 meta['b_dir']），并同步更新 meta 的 anchor / n_cm。spec 缺省取当前
        焊缝最后放入的障碍。仅适用类型2（meta 含 b_dir/anchor）。
        """
        if spec is None:
            lst = self.obstacles.get(self.seam_id, [])
            if not lst:
                raise RuntimeError(f"焊缝 {self.seam_id} 尚无障碍，无法调整 n_cm")
            spec = lst[-1]
        meta = spec.meta or {}
        if "b_dir" not in meta or "anchor" not in meta:
            raise RuntimeError("该障碍无 b_dir/anchor（仅类型2遮挡板支持增大 n_cm）")
        shift = (float(n_cm) / 100.0) * np.asarray(meta["b_dir"], float)
        for m in spec.meshes:                              # 平移全部网格顶点
            m["points"] = np.asarray(m["points"], float) + shift
        meta["anchor"] = np.asarray(meta["anchor"], float) + shift
        meta["n_cm"] = float(meta.get("n_cm", 0.0)) + float(n_cm)
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

        self.obstacles.setdefault(self.seam_id, []).append(spec)
        return spec
