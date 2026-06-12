"""在 Isaac Sim 里可视化 gt_gen.obstacles 生成的障碍物（见 docs/障碍物3d结构.md）。

障碍物用 cuRobo 原语（Cuboid/Cylinder）表示 → 这里 spawn 成 VisualCuboid / VisualCylinder
（纯视觉，无碰撞/刚体）。位姿 wxyz 与 obstacles 模块一致。

摆放：仅靠 --anchor x,y,z（+ 可选 --rpy 度）锚定，不从 seam 自动推位置。
--seam 可选：显示工件 USD + 机器人(retract) 做背景，便于判断障碍是否卡在相机—焊缝之间。

运行（带显示器）：
    conda run -n env_isaaclab python scripts/viz_obstacles_isaacsim.py --obstacle open_box --anchor 0.6,0,0.7
    # 覆盖形状参数：
    conda run -n env_isaaclab python scripts/viz_obstacles_isaacsim.py --obstacle plate --params tilt_deg=30,width=0.9
    # 列出所有可用障碍：
    conda run -n env_isaaclab python scripts/viz_obstacles_isaacsim.py --list
    # 一次性平铺可视化全部 15 种（网格排布；控制台打印「编号|类别|障碍|anchor」图例）：
    conda run -n env_isaaclab python scripts/viz_obstacles_isaacsim.py --all
    #   --cols N 控列数(默认5,沿+Y展开)、--pitch M 控间距(默认1.6m,换行沿+X推进)
无显示器自检：加 --headless（spawn 后跑几帧即退，打印每个 prim + VIZ_OBSTACLES_DONE）。
（--all 模式下每种障碍用其默认形状参数，--params/--size 被忽略；类别中文因 USD 名限制只在控制台图例显示，
  prim 路径用 /World/obstacles/g<编号>_c<类别号>_<障碍名>/ 分组，便于在 stage 树/视口里点选辨认。）

================================================================================
障碍目录（共 15 种，对应 docs/障碍物3d结构.md 七大类 + 第 8 节推荐组合）
================================================================================
通用约定：
  - 所有障碍签名为 fn(anchor_pos, anchor_rpy_deg=(0,0,0), **形状参数)。
  - anchor 是结构在 base 系的锚点（中心/角，见各条说明）；--rpy 整体旋转该结构。
  - base 系：+X 朝前(远离机器人)、+Y 左、+Z 上；多数板/框默认法向 +X（挡 X 向视线）。
  - 形状参数用 --params k=v,k=v 覆盖（数值自动转 float；open_face/axis/stack 等留字符串）。
  - 长度单位米，角度单位度。

【1. 钢板/挡板类】
  plate        单块钢板（薄方体，法向 +X，中心在 anchor）。tilt_deg 绕 Y 倾斜=「倾斜钢板」。
               参数: length=0.8 width=0.6 thickness=0.02 tilt_deg=0.0
               dims=[厚, 宽(Y), 高(Z)]。例: --params tilt_deg=30,width=0.9
  l_bracket    L 型钢板（└）：竖板(法向+X,沿+Z) + 横板(法向+Z,沿+X)，交于 anchor 角。
               参数: length=0.6 width=0.5 thickness=0.02
  u_channel    U 型槽（开口朝上）：底板 + 左右两侧板，工件可放槽内。
               参数: length=0.6 width=0.5 height=0.4 thickness=0.02

【2. 多平面开口盒/五面体】
  open_box     五面开口盒（缺一面的盒，盒中心在 anchor）。open_face 决定缺哪面。
               参数: size=(0.6,0.6,0.6) wall=0.02 open_face=front
                     open_face ∈ {front(+X) back(-X) left(+Y) right(-Y) top(+Z) bottom(-Z)}
               注: size 是三元组，--params 传不了，请用独立的 --size x,y,z。
               例: --params open_face=top   或   --size 0.8,0.6,0.5

【3. 圆柱钢管类】
  pipe         单根圆管（横跨视线通道）。axis 指定管轴方向。
               参数: length=1.0 radius=0.05 axis=y    (axis ∈ {x,y,z})
               例: --params axis=x,length=1.2
  parallel_pipes  多根平行钢管（管束/护栏）：n 根沿 axis 的管，沿 stack 以 gap 等距排开。
               参数: n=3 length=1.0 radius=0.05 gap=0.2 axis=y stack=z
               例: --params n=5,axis=z,stack=y,gap=0.15
  crossed_pipes   交叉钢管（X 形）：两根管在 Y-Z 平面内夹 cross_deg 交叉。
               参数: length=1.0 radius=0.05 cross_deg=90.0

【4. 方管/矩形管/框架】
  box_beam     方管/方梁（实心方截面梁）。axis 指定梁长方向。
               参数: length=1.0 side=0.1 axis=y    (axis ∈ {x,y,z})
  rect_frame   矩形管框架（Y-Z 平面一圈方梁，中间留孔，相机可从孔看）。
               参数: width=0.8 height=0.8 beam=0.08
  gantry       门型框架（两立柱 + 一横梁 ┌─┐），立柱从 anchor 平面向上立。
               参数: span=0.8 height=1.0 post=0.08 beam=0.1
               例: --params height=1.2
  braced_frame 斜撑框架（矩形框 + 一根对角斜梁，破坏直线路径）。
               参数: width=0.8 height=0.8 beam=0.08 brace=0.06

【5/7. 三角支架/台阶（不规则件）】
  tripod       三角支架（三根杆组成竖立三角框，Y-Z 平面内 /\ + 底边）。
               参数: height=0.9 base_half=0.35 rod=0.05
  steps        台阶形障碍（多方体叠成楼梯状，遮挡高度逐级变化）。
               参数: n=3 rise=0.15 run=0.25 width=0.6
               例: --params n=4,rise=0.2

【8. 推荐优先组合】
  box_with_pipe   首选组合：五面开口盒 + 一根横管挡在开口前。
               参数: size=(0.7,0.7,0.7) wall=0.02 open_face=front pipe_radius=0.05
  frame_with_brace  组合「方管框架 + 斜撑杆」（= braced_frame 语义入口）。
               参数: 同 braced_frame (width/height/beam/brace)

注：docs 中的「弯管/U 型管、压紧夹具、定位块、V 型支撑块、楔形块、多面折弯板」等未单列
    成器，可用现有原语组合近似（弯管≈crossed/parallel，夹具≈l_bracket+pipe，楔形≈旋转 plate）。
================================================================================
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUROBO_ISAAC = "/home/a/Projects/Github/curobo/examples/isaac_sim"

sys.path.insert(0, ROOT)

# --list 不需要起 Isaac Sim，提前处理
from gt_gen import obstacles as ob  # noqa: E402

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

_ap = argparse.ArgumentParser()
_ap.add_argument("--obstacle", default=None, help="障碍名（见 --list）")
_ap.add_argument("--list", action="store_true", help="列出所有可用障碍后退出")
_ap.add_argument("--anchor", default="0.6,0.0,0.7", help="障碍锚点 x,y,z（base 系，米）")
_ap.add_argument("--rpy", default="0,0,0", help="锚点姿态 roll,pitch,yaw（度）")
_ap.add_argument("--params", default=None, help="形状参数覆盖，如 'tilt_deg=30,width=0.9,open_face=top'")
_ap.add_argument("--size", default=None, help="盒体 size=x,y,z（open_box/box_with_pipe 专用；--params 传不了三元组）")
_ap.add_argument("--all", action="store_true", help="一次性平铺可视化全部 15 种障碍（网格排布，控制台打印类别图例）")
_ap.add_argument("--cols", type=int, default=5, help="--all 网格列数（默认 5，沿 +Y 展开）")
_ap.add_argument("--pitch", type=float, default=1.6, help="--all 网格间距(米，默认 1.6，换行沿 +X 推进)")
_ap.add_argument("--seam", default=None, help="可选 seam pkl：显示工件 USD + 机器人做背景")
_ap.add_argument("--joints", default=None, help="机器人关节角(弧度,逗号分隔);默认 pkl joint_angles")
_ap.add_argument("--headless", action="store_true")
args = _ap.parse_args()

if args.list:
    print("可用障碍（共 %d 种）：" % len(ob.list_obstacles()))
    for n in ob.list_obstacles():
        print("  ", n)
    sys.exit(0)

if not args.obstacle and not args.all:
    _ap.error("需要 --obstacle NAME（或 --all 平铺全部，或 --list 查看可选）")


def parse_vec(s, n):
    v = [float(x) for x in s.split(",")]
    assert len(v) == n, f"期望 {n} 个数，得 {s!r}"
    return v


def parse_params(s):
    """'k=v,k=v' → dict，值按 int→float→str 依次尝试。"""
    out = {}
    if not s:
        return out
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, _, v = kv.partition("=")
        k, v = k.strip(), v.strip()
        for cast in (int, float):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            out[k] = v        # 留作字符串（如 open_face=top, axis=x）
    return out


# 先把障碍生成出来（失败则不必起 Isaac Sim）
anchor = parse_vec(args.anchor, 3)
rpy = parse_vec(args.rpy, 3)

# --all 平铺用的障碍清单：(类别号, 类别中文, 障碍名, 默认形状参数)。
# 对应 docs/障碍物3d结构.md 七大类 + 第 8 节推荐组合；参数为各生成函数默认值（不靠隐式默认）。
ALL_OBSTACLES = [
    (1, "钢板/挡板类",   "plate",            dict(length=0.8, width=0.6, thickness=0.02, tilt_deg=0.0)),
    (1, "钢板/挡板类",   "l_bracket",        dict(length=0.6, width=0.5, thickness=0.02)),
    (1, "钢板/挡板类",   "u_channel",        dict(length=0.6, width=0.5, height=0.4, thickness=0.02)),
    (2, "多平面开口盒",  "open_box",         dict(size=(0.6, 0.6, 0.6), wall=0.02, open_face="front")),
    (3, "圆柱钢管类",    "pipe",             dict(length=1.0, radius=0.05, axis="y")),
    (3, "圆柱钢管类",    "parallel_pipes",   dict(n=3, length=1.0, radius=0.05, gap=0.2, axis="y", stack="z")),
    (3, "圆柱钢管类",    "crossed_pipes",    dict(length=1.0, radius=0.05, cross_deg=90.0)),
    (4, "方管/框架",     "box_beam",         dict(length=1.0, side=0.1, axis="y")),
    (4, "方管/框架",     "rect_frame",       dict(width=0.8, height=0.8, beam=0.08)),
    (4, "方管/框架",     "gantry",           dict(span=0.8, height=1.0, post=0.08, beam=0.1)),
    (4, "方管/框架",     "braced_frame",     dict(width=0.8, height=0.8, beam=0.08, brace=0.06)),
    (5, "三角支架/台阶", "tripod",           dict(height=0.9, base_half=0.35, rod=0.05)),
    (5, "三角支架/台阶", "steps",            dict(n=3, rise=0.15, run=0.25, width=0.6)),
    (8, "推荐优先组合",  "box_with_pipe",    dict(size=(0.7, 0.7, 0.7), wall=0.02, open_face="front", pipe_radius=0.05)),
    (8, "推荐优先组合",  "frame_with_brace", dict(width=0.8, height=0.8, beam=0.08, brace=0.06)),
]


def _grid_anchor(base, idx, cols, pitch):
    """网格平铺：idx 沿列(+Y，居中)展开、换行沿 +X 推进。返回 base 系 anchor。"""
    row, col = divmod(idx, int(cols))
    bx, by, bz = base
    y = by + (col - (cols - 1) / 2.0) * pitch
    x = bx + row * pitch
    return [x, y, bz]


entries = []          # [(group_label, name, anchor, prims), ...]；group_label 为 USD 合法名
if args.all:
    print(f"=== --all：平铺全部 {len(ALL_OBSTACLES)} 种障碍（cols={args.cols} pitch={args.pitch}m）===", flush=True)
    print("网格图例（编号 | 类别 | 障碍 | anchor | 原语数）：", flush=True)
    for idx, (cat, cat_cn, name, shp) in enumerate(ALL_OBSTACLES):
        anc = _grid_anchor(anchor, idx, args.cols, args.pitch)
        prims = ob.build(name, anc, anchor_rpy_deg=tuple(rpy), **shp)
        group = f"g{idx:02d}_c{cat}_{name}"          # USD 名仅许 [A-Za-z0-9_]，类别中文走控制台图例
        entries.append((group, name, anc, prims))
        print(f"  [{idx:02d}] 第{cat}类 {cat_cn:<9} {name:<16} "
              f"anchor={[round(v, 2) for v in anc]} 原语={len(prims)}", flush=True)
else:
    shape = parse_params(args.params)
    if args.size:
        shape["size"] = tuple(parse_vec(args.size, 3))   # 三元组单独走 --size，绕开 --params 的逗号冲突
    prims = ob.build(args.obstacle, anchor, anchor_rpy_deg=tuple(rpy), **shape)
    entries.append((f"g00_c0_{args.obstacle}", args.obstacle, anchor, prims))
    print(f"障碍 '{args.obstacle}'：{len(prims)} 个原语，anchor={anchor} rpy={rpy} 参数={shape}", flush=True)
    for p in prims:
        if isinstance(p, ob.Box):
            print(f"  Box  {p.name:<16} dims={[round(d,3) for d in p.dims]} pos={[round(x,3) for x in p.pose[:3]]}", flush=True)
        else:
            print(f"  Tube {p.name:<16} r={p.radius:.3f} h={p.height:.3f} pos={[round(x,3) for x in p.pose[:3]]}", flush=True)

from omni.isaac.kit import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np  # noqa: E402
import pickle  # noqa: E402
import glob  # noqa: E402

sys.path.insert(0, CUROBO_ISAAC)

from gt_gen import compat  # noqa: F401,E402  warp shim
from gt_gen.config import load_config  # noqa: E402
from curobo.util_file import load_yaml  # noqa: E402

from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.objects import cuboid as _cuboid  # noqa: E402
from omni.isaac.core.objects import cylinder as _cylinder  # noqa: E402
from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa: E402


def spawn_prim(group, k, p):
    """把一个 obstacles 原语 spawn 成 Isaac Sim 纯视觉 prim。group=USD 合法分组名(含类别号+障碍名)。"""
    color = np.asarray(p.color, dtype=float)
    pos = np.asarray(p.pose[:3], dtype=float)
    quat = np.asarray(p.pose[3:7], dtype=float)        # wxyz
    pth = f"/World/obstacles/{group}/prim_{k}"
    if isinstance(p, ob.Box):
        _cuboid.VisualCuboid(prim_path=pth, name=f"obs_{group}_{k}",
                             position=pos, orientation=quat,
                             size=1.0, scale=np.asarray(p.dims, dtype=float), color=color)
    else:
        _cylinder.VisualCylinder(prim_path=pth, name=f"obs_{group}_{k}",
                                 position=pos, orientation=quat,
                                 radius=float(p.radius), height=float(p.height), color=color)


def add_workpiece_and_robot(world):
    """可选：从 --seam 加载工件 USD（基座系，关物理当纯视觉）+ 机器人(retract)。返回 (robot, joint_q) 或 (None, None)。"""
    cfg = load_config()
    robot_cfg = load_yaml(cfg.robot_cfg_path)["robot_cfg"]
    from helper import add_robot_to_scene  # cuRobo 示例自带

    q = np.asarray(cfg.retract_config, float)
    if args.seam:
        d = pickle.load(open(args.seam, "rb"))
        rp = np.asarray(d["robot_pose"][0], float)
        if args.joints:
            q = np.array([float(x) for x in args.joints.split(",")], float)
        elif "joint_angles" in d:
            q = np.asarray(d["joint_angles"], float)
        dd = os.path.dirname(args.seam)
        objs = sorted(glob.glob(os.path.join(dd, "*_watertight.obj"))) or \
            sorted(glob.glob(os.path.join(dd, "*.obj")))
        usd_obj = objs[0].replace("_watertight.obj", ".obj").replace(".obj", ".usd") if objs else None
        if usd_obj and os.path.exists(usd_obj):
            add_reference_to_stage(usd_path=usd_obj, prim_path="/World/workpiece")
            from omni.isaac.core.prims import XFormPrim
            XFormPrim("/World/workpiece").set_world_pose(position=(-rp[:3]).tolist())
            from pxr import Usd, UsdPhysics
            import omni.usd
            stg = omni.usd.get_context().get_stage()
            for pr in Usd.PrimRange(stg.GetPrimAtPath("/World/workpiece")):
                if pr.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(pr).GetCollisionEnabledAttr().Set(False)
                if pr.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(pr).GetRigidBodyEnabledAttr().Set(False)
        else:
            print("warn: 未找到工件 usd（跳过工件显示）:", usd_obj)

    robot, _ = add_robot_to_scene(robot_cfg, world)
    return robot, q, cfg.joint_names


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    compat.apply_trimesh_shim()

    robot, q, joint_names = add_workpiece_and_robot(world)

    for group, name, anc, prims_i in entries:
        for k, p in enumerate(prims_i):
            spawn_prim(group, k, p)

    world.reset()
    if robot is not None:
        if hasattr(robot, "initialize"):
            robot.initialize()
        idx_list = [robot.get_dof_index(j) for j in joint_names]
        robot.set_joint_positions(q, idx_list)

    if args.headless:
        for _ in range(3):
            world.step(render=False)
        n_prim = sum(len(e[3]) for e in entries)
        print(f"已 spawn {n_prim} 个障碍 prim（{len(entries)} 组）。", flush=True)
        print("VIZ_OBSTACLES_DONE", flush=True)
        sys.stdout.flush()
        simulation_app.close()
        return

    print("显示中（关闭窗口结束）…")
    while simulation_app.is_running():
        world.step(render=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
