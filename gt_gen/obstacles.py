"""障碍物 3D 结构生成库（见 docs/障碍物3d结构.md）。

目标：参数化生成工业避障场景里常见的障碍物（钢板/开口盒/圆管/方管框架/门架/夹具/不规则件），
用于在 Isaac Sim 里摆场景、并可塞进 cuRobo 碰撞世界做将来规划。

设计（已与用户确认）：
- 几何只用两种原语：长方体 Box（薄板/方梁/盒壁）与圆柱 Tube（圆管/杆）。倾斜板靠旋转、
  三角架用三根杆近似——不做真三角网格/USD mesh。
- 本模块后端无关：只产 dataclass（纯几何 + wxyz 位姿），不 import curobo / isaac。
  转 cuRobo 见 to_curobo / to_world_config（惰性 import）；喂 Isaac Sim 见 scripts/viz_obstacles_isaacsim.py。
- 位姿一律 [x, y, z, qw, qx, qy, qz]（wxyz，与 cuRobo Obstacle.pose、Isaac core 标量在前约定一致）。
- 每个生成函数在 anchor 局部系搭好结构，再整体变换到 base 系。anchor=(pos, rpy_deg)。

坐标约定（base 系）：+X 朝前(远离机器人)、+Y 左、+Z 上。
  大多数「挡相机视线」的板/框默认法向 +X（相机沿 X 方向看过去时被挡）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

GRAY: List[float] = [0.62, 0.64, 0.67]          # 钢材灰（RGB 0~1）
RPY = Tuple[float, float, float]


# ---------------------------------------------------------------- 原语 dataclass
@dataclass
class Box:
    """长方体。dims=[x,y,z] 边长(米)；pose=[x,y,z,qw,qx,qy,qz] base 系。"""
    name: str
    dims: List[float]
    pose: List[float]
    color: List[float] = field(default_factory=lambda: list(GRAY))


@dataclass
class Tube:
    """圆柱（轴沿局部 +Z，与 cuRobo Cylinder / trimesh.creation.cylinder 一致）。"""
    name: str
    radius: float
    height: float
    pose: List[float]
    color: List[float] = field(default_factory=lambda: list(GRAY))


Prim = Union[Box, Tube]


# ---------------------------------------------------------------- 位姿工具
def euler_to_quat_wxyz(rpy_deg: Sequence[float]) -> List[float]:
    """xyz 内旋欧拉角(度) → 四元数 wxyz。"""
    x, y, z, w = R.from_euler("xyz", list(rpy_deg), degrees=True).as_quat()  # scipy 返回 xyzw
    return [float(w), float(x), float(y), float(z)]


def _R(rpy_deg: Sequence[float]) -> R:
    return R.from_euler("xyz", list(rpy_deg), degrees=True)


def _pose(anchor_pos, anchor_R: R, local_pos, local_R: R) -> List[float]:
    """anchor 局部位姿 → base 系 [x,y,z,qw,qx,qy,qz]。"""
    ap = np.asarray(anchor_pos, float)
    world_pos = ap + anchor_R.apply(np.asarray(local_pos, float))
    x, y, z, w = (anchor_R * local_R).as_quat()
    return [float(world_pos[0]), float(world_pos[1]), float(world_pos[2]),
            float(w), float(x), float(y), float(z)]


def compose(anchor_pos, anchor_rpy_deg: RPY, local_pos, local_rpy_deg: RPY) -> List[float]:
    """便捷版 _pose：局部姿态用 rpy(度) 给。"""
    return _pose(anchor_pos, _R(anchor_rpy_deg), local_pos, _R(local_rpy_deg))


def _align_z_to(direction) -> R:
    """求把局部 +Z 对到给定方向的旋转（用于按两端点摆杆/管）。"""
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, d))
    if c > 1 - 1e-9:
        return R.identity()
    if c < -1 + 1e-9:
        return R.from_rotvec([np.pi, 0.0, 0.0])     # 翻 180°（绕 X）
    axis = np.cross(z, d)
    axis /= np.linalg.norm(axis) + 1e-12
    return R.from_rotvec(axis * np.arccos(np.clip(c, -1.0, 1.0)))


def _rod_between(name, pa, pb, radius, anchor_pos, anchor_R) -> Tube:
    """在 anchor 局部系内，两端点 pa→pb 之间摆一根圆柱（杆/管）。"""
    pa = np.asarray(pa, float)
    pb = np.asarray(pb, float)
    mid = 0.5 * (pa + pb)
    length = float(np.linalg.norm(pb - pa))
    local_R = _align_z_to(pb - pa)
    return Tube(name, float(radius), length, _pose(anchor_pos, anchor_R, mid, local_R))


# ---------------------------------------------------------------- 生成函数（1. 钢板类）
def plate(anchor_pos, anchor_rpy_deg=(0, 0, 0),
          length=0.8, width=0.6, thickness=0.02, tilt_deg=0.0) -> List[Prim]:
    """单块钢板（法向 +X，挡 X 向视线）。tilt_deg 绕 Y 倾斜 → 即「倾斜钢板」。
    dims=[厚, 宽(Y), 高(Z)]，中心在 anchor。"""
    return [Box("plate", [thickness, width, length],
                compose(anchor_pos, anchor_rpy_deg, [0, 0, 0], [0, tilt_deg, 0]))]


def l_bracket(anchor_pos, anchor_rpy_deg=(0, 0, 0),
              length=0.6, width=0.5, thickness=0.02) -> List[Prim]:
    """L 型钢板（└）：一竖板(法向+X，沿+Z) + 一横板(法向+Z，沿+X)，交于 anchor 角。"""
    aR = _R(anchor_rpy_deg)
    vert = Box("l_vertical", [thickness, width, length],
               _pose(anchor_pos, aR, [0, 0, length / 2], R.identity()))
    horiz = Box("l_horizontal", [length, width, thickness],
                _pose(anchor_pos, aR, [length / 2, 0, 0], R.identity()))
    return [vert, horiz]


def u_channel(anchor_pos, anchor_rpy_deg=(0, 0, 0),
              length=0.6, width=0.5, height=0.4, thickness=0.02) -> List[Prim]:
    """U 型钢板槽（开口朝上）：底板 + 左右两侧板。工件可放槽内。"""
    aR = _R(anchor_rpy_deg)
    bottom = Box("u_bottom", [length, width, thickness],
                 _pose(anchor_pos, aR, [0, 0, 0], R.identity()))
    left = Box("u_left", [length, thickness, height],
               _pose(anchor_pos, aR, [0, width / 2, height / 2], R.identity()))
    right = Box("u_right", [length, thickness, height],
                _pose(anchor_pos, aR, [0, -width / 2, height / 2], R.identity()))
    return [bottom, left, right]


# ---------------------------------------------------------------- 生成函数（2. 开口盒/五面体）
def open_box(anchor_pos, anchor_rpy_deg=(0, 0, 0),
             size=(0.6, 0.6, 0.6), wall=0.02, open_face="front") -> List[Prim]:
    """五面开口盒（缺一面的盒子）。open_face ∈ {front(+X), back(-X), left(+Y),
    right(-Y), top(+Z), bottom(-Z)}。盒中心在 anchor。"""
    sx, sy, sz = [float(v) for v in size]
    aR = _R(anchor_rpy_deg)
    faces = {
        "front":  Box("box_front",  [wall, sy, sz], _pose(anchor_pos, aR, [sx / 2, 0, 0], R.identity())),
        "back":   Box("box_back",   [wall, sy, sz], _pose(anchor_pos, aR, [-sx / 2, 0, 0], R.identity())),
        "left":   Box("box_left",   [sx, wall, sz], _pose(anchor_pos, aR, [0, sy / 2, 0], R.identity())),
        "right":  Box("box_right",  [sx, wall, sz], _pose(anchor_pos, aR, [0, -sy / 2, 0], R.identity())),
        "top":    Box("box_top",    [sx, sy, wall], _pose(anchor_pos, aR, [0, 0, sz / 2], R.identity())),
        "bottom": Box("box_bottom", [sx, sy, wall], _pose(anchor_pos, aR, [0, 0, -sz / 2], R.identity())),
    }
    if open_face not in faces:
        raise ValueError(f"open_face 须为 {list(faces)} 之一，收到 {open_face!r}")
    faces.pop(open_face)
    return list(faces.values())


# ---------------------------------------------------------------- 生成函数（3. 圆管类）
_AXIS_RPY = {"x": (0, 90, 0), "y": (90, 0, 0), "z": (0, 0, 0)}     # 把 +Z 轴转到 x/y/z


def pipe(anchor_pos, anchor_rpy_deg=(0, 0, 0),
         length=1.0, radius=0.05, axis="y") -> List[Prim]:
    """单根圆管（横跨视线通道）。axis ∈ {x,y,z} 指定管轴方向。"""
    if axis not in _AXIS_RPY:
        raise ValueError(f"axis 须为 x/y/z，收到 {axis!r}")
    return [Tube("pipe", radius, length,
                 compose(anchor_pos, anchor_rpy_deg, [0, 0, 0], _AXIS_RPY[axis]))]


def parallel_pipes(anchor_pos, anchor_rpy_deg=(0, 0, 0),
                   n=3, length=1.0, radius=0.05, gap=0.2,
                   axis="y", stack="z") -> List[Prim]:
    """多根平行钢管（管束/护栏）：n 根沿 axis 的管，沿 stack 方向以 gap 等距排开。"""
    if axis not in _AXIS_RPY:
        raise ValueError(f"axis 须为 x/y/z，收到 {axis!r}")
    stack_dir = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}.get(stack)
    if stack_dir is None:
        raise ValueError(f"stack 须为 x/y/z，收到 {stack!r}")
    aR = _R(anchor_rpy_deg)
    rpy = _AXIS_RPY[axis]
    out = []
    for i in range(int(n)):
        off = (i - (n - 1) / 2.0) * gap
        lp = [off * d for d in stack_dir]
        out.append(Tube(f"pipe_{i}", radius, length, _pose(anchor_pos, aR, lp, _R(rpy))))
    return out


def crossed_pipes(anchor_pos, anchor_rpy_deg=(0, 0, 0),
                  length=1.0, radius=0.05, cross_deg=90.0) -> List[Prim]:
    """交叉钢管（X 形）：两根管在 Y-Z 平面内夹 cross_deg 交叉。"""
    aR = _R(anchor_rpy_deg)
    half = cross_deg / 2.0
    # 绕 X 转 -90° 使管轴落到 +Y；再 ±half 张开成交叉
    a = Tube("cross_a", radius, length, _pose(anchor_pos, aR, [0, 0, 0], _R([-90 + half, 0, 0])))
    b = Tube("cross_b", radius, length, _pose(anchor_pos, aR, [0, 0, 0], _R([-90 - half, 0, 0])))
    return [a, b]


# ---------------------------------------------------------------- 生成函数（4. 方管/框架）
def box_beam(anchor_pos, anchor_rpy_deg=(0, 0, 0),
             length=1.0, side=0.1, axis="y") -> List[Prim]:
    """方管/方梁（实心方截面梁）。axis 指定梁长方向。"""
    dims = {"x": [length, side, side], "y": [side, length, side],
            "z": [side, side, length]}.get(axis)
    if dims is None:
        raise ValueError(f"axis 须为 x/y/z，收到 {axis!r}")
    return [Box("box_beam", dims, compose(anchor_pos, anchor_rpy_deg, [0, 0, 0], [0, 0, 0]))]


def rect_frame(anchor_pos, anchor_rpy_deg=(0, 0, 0),
               width=0.8, height=0.8, beam=0.08) -> List[Prim]:
    """矩形管框架（Y-Z 平面内一圈方梁，中间留孔——相机可从孔中看）。"""
    aR = _R(anchor_rpy_deg)
    out = [
        Box("frame_top",    [beam, width, beam], _pose(anchor_pos, aR, [0, 0, height / 2], R.identity())),
        Box("frame_bottom", [beam, width, beam], _pose(anchor_pos, aR, [0, 0, -height / 2], R.identity())),
        Box("frame_left",   [beam, beam, height], _pose(anchor_pos, aR, [0, width / 2, 0], R.identity())),
        Box("frame_right",  [beam, beam, height], _pose(anchor_pos, aR, [0, -width / 2, 0], R.identity())),
    ]
    return out


def gantry(anchor_pos, anchor_rpy_deg=(0, 0, 0),
           span=0.8, height=1.0, post=0.08, beam=0.1) -> List[Prim]:
    """门型框架（两立柱 + 一横梁，┌─┐）。立柱从 anchor 平面向上立起。"""
    aR = _R(anchor_rpy_deg)
    out = [
        Box("gantry_post_l", [post, post, height], _pose(anchor_pos, aR, [0, span / 2, height / 2], R.identity())),
        Box("gantry_post_r", [post, post, height], _pose(anchor_pos, aR, [0, -span / 2, height / 2], R.identity())),
        Box("gantry_beam", [beam, span + post, beam], _pose(anchor_pos, aR, [0, 0, height], R.identity())),
    ]
    return out


def braced_frame(anchor_pos, anchor_rpy_deg=(0, 0, 0),
                 width=0.8, height=0.8, beam=0.08, brace=0.06) -> List[Prim]:
    """斜撑框架（矩形框 + 一根对角斜梁，破坏直线路径）。"""
    aR = _R(anchor_rpy_deg)
    out = rect_frame(anchor_pos, anchor_rpy_deg, width, height, beam)
    # 对角斜梁：左下角 → 右上角（Y-Z 平面内）
    pa = np.array([0.0, width / 2, -height / 2])
    pb = np.array([0.0, -width / 2, height / 2])
    diag = _rod_diag_box("frame_brace", pa, pb, brace, anchor_pos, aR)
    out.append(diag)
    return out


def _rod_diag_box(name, pa, pb, side, anchor_pos, anchor_R) -> Box:
    """两端点之间摆一根方梁（用于斜撑），长轴对齐 pa→pb。"""
    pa = np.asarray(pa, float)
    pb = np.asarray(pb, float)
    mid = 0.5 * (pa + pb)
    length = float(np.linalg.norm(pb - pa))
    local_R = _align_z_to(pb - pa)
    return Box(name, [side, side, length], _pose(anchor_pos, anchor_R, mid, local_R))


# ---------------------------------------------------------------- 生成函数（5/7. 三角/台阶）
def tripod(anchor_pos, anchor_rpy_deg=(0, 0, 0),
           height=0.9, base_half=0.35, rod=0.05) -> List[Prim]:
    """三角支架（三根杆组成竖立三角框，Y-Z 平面内 /\\ + 底边）。"""
    aR = _R(anchor_rpy_deg)
    apex = [0.0, 0.0, height / 2]
    left = [0.0, base_half, -height / 2]
    right = [0.0, -base_half, -height / 2]
    return [
        _rod_between("tri_left", apex, left, rod, anchor_pos, aR),
        _rod_between("tri_right", apex, right, rod, anchor_pos, aR),
        _rod_between("tri_base", left, right, rod, anchor_pos, aR),
    ]


def steps(anchor_pos, anchor_rpy_deg=(0, 0, 0),
          n=3, rise=0.15, run=0.25, width=0.6) -> List[Prim]:
    """台阶形障碍（多个长方体叠成楼梯状，遮挡高度逐级变化）。"""
    aR = _R(anchor_rpy_deg)
    out = []
    for i in range(int(n)):
        h = rise * (i + 1)                      # 第 i 级总高（从地面累计）
        out.append(Box(f"step_{i}", [run, width, h],
                       _pose(anchor_pos, aR, [i * run, 0, h / 2], R.identity())))
    return out


# ---------------------------------------------------------------- 生成函数（8. 推荐组合）
def box_with_pipe(anchor_pos, anchor_rpy_deg=(0, 0, 0),
                  size=(0.7, 0.7, 0.7), wall=0.02, open_face="front",
                  pipe_radius=0.05) -> List[Prim]:
    """文档首选组合：五面开口盒 + 一根横管挡在开口前。"""
    sx, sy, sz = [float(v) for v in size]
    aR = _R(anchor_rpy_deg)
    out = open_box(anchor_pos, anchor_rpy_deg, size, wall, open_face)
    # 横管沿 Y，摆在开口面(+X)前一点
    out.append(Tube("combo_pipe", pipe_radius, sy * 1.4,
                     _pose(anchor_pos, aR, [sx / 2 + 0.15, 0, 0], _R(_AXIS_RPY["y"]))))
    return out


def frame_with_brace(anchor_pos, anchor_rpy_deg=(0, 0, 0), **shape) -> List[Prim]:
    """文档组合「方管框架 + 斜撑杆」（= braced_frame 的语义入口）。"""
    return braced_frame(anchor_pos, anchor_rpy_deg, **shape)


# ---------------------------------------------------------------- 注册表 + 入口
REGISTRY = {
    "plate": plate,
    "l_bracket": l_bracket,
    "u_channel": u_channel,
    "open_box": open_box,
    "pipe": pipe,
    "parallel_pipes": parallel_pipes,
    "crossed_pipes": crossed_pipes,
    "box_beam": box_beam,
    "rect_frame": rect_frame,
    "gantry": gantry,
    "braced_frame": braced_frame,
    "tripod": tripod,
    "steps": steps,
    "box_with_pipe": box_with_pipe,
    "frame_with_brace": frame_with_brace,
}


def list_obstacles() -> List[str]:
    return list(REGISTRY)


def build(name: str, anchor_pos, anchor_rpy_deg: RPY = (0, 0, 0), **shape) -> List[Prim]:
    """按名字生成障碍物原语列表。shape 是该障碍的形状参数（见对应函数）。"""
    if name not in REGISTRY:
        raise ValueError(f"未知障碍 {name!r}，可选：{list_obstacles()}")
    return REGISTRY[name](anchor_pos, anchor_rpy_deg, **shape)


# ---------------------------------------------------------------- cuRobo 转换（惰性 import）
def to_curobo(prims: Sequence[Prim]):
    """原语列表 → cuRobo 障碍对象列表（Box→Cuboid、Tube→Cylinder）。名字加序号保证唯一。"""
    import gt_gen.compat  # noqa: F401  warp/trimesh shim，须在 import curobo 前
    gt_gen.compat.apply_trimesh_shim()
    from curobo.geom.types import Cuboid, Cylinder

    out = []
    for i, p in enumerate(prims):
        col = list(p.color) + [1.0] if p.color is not None else None
        nm = f"{p.name}_{i}"
        if isinstance(p, Box):
            out.append(Cuboid(name=nm, dims=list(p.dims), pose=list(p.pose), color=col))
        elif isinstance(p, Tube):
            out.append(Cylinder(name=nm, radius=float(p.radius), height=float(p.height),
                                pose=list(p.pose), color=col))
        else:
            raise TypeError(f"未知原语类型 {type(p)}")
    return out


def to_world_config(prims: Sequence[Prim]):
    """原语列表 → cuRobo WorldConfig（cuboid=[...] / cylinder=[...]），可塞 init_curobo(world_model=)。"""
    import gt_gen.compat  # noqa: F401
    gt_gen.compat.apply_trimesh_shim()
    from curobo.geom.types import WorldConfig

    objs = to_curobo(prims)
    cuboids = [o for o in objs if o.__class__.__name__ == "Cuboid"]
    cylinders = [o for o in objs if o.__class__.__name__ == "Cylinder"]
    return WorldConfig(cuboid=cuboids, cylinder=cylinders)
