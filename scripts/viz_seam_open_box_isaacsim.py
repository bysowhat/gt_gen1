"""在 Isaac Sim 里把一个【开口盒 open_box】套在指定焊缝上，并和工件叠加显示（独立脚本，不动业务代码）。

需求：在工件基础上、对指定焊缝，用【1 个 open_box】把焊缝那一段包住（不必包住整个工件，工件可与盒穿模）。
盒大小与「焊缝在盒中的位置」由 6 个面到焊缝的最近距离 dis（上下左右前后，cm）唯一解出。

盒局部坐标约定（与 gt_gen/obstacles.open_box 一致：front=+X / back=−X / left=+Y / right=−Y / top=+Z / bottom=−Z）：
  · 上/下轴 = 世界 ±Z（重力上下）；盒只绕世界 Z 偏航、不翻滚（top/bottom 面恒水平）。
  · 开口面 = 后 back(−X)，朝水平角平分线 bis_h —— 即焊缝角平分线 bis 从焊缝穿出这个开口面（唯一约束）。
    front(+X) = −bis_h（朝工件侧）；左右轴 Y = Z×X。
  · 把焊缝折线投影到三个盒轴得各轴跨度 [min,max]，配对应两面的 dis：
      size  = 跨度 + d_正 + d_负，   盒心沿该轴偏移 = (d_正 − d_负)/2，
    于是 6 个 dis（上下左右前后）唯一定出盒尺寸 + 盒心（焊缝在盒内的位置）。dis 为正=该面在焊缝外侧 dis 处。

dis 顺序固定为 --dis_cm 上 下 左 右 前 后（cm）；开口面恒为「后」。

运行（带显示器）：
    conda run -n env_isaaclab python scripts/viz_seam_open_box_isaacsim.py \
        --seam .../seam_22.pkl --dis_cm 10 10 10 10 10 10 --wall_cm 2
    # --seam 支持多条 / 目录(收 seam_*.pkl) / 通配；每条焊缝各套一个 open_box
无显示器自检：加 --headless（spawn + 跑几帧即退，打印盒尺寸/实测面距 + VIZ_OPEN_BOX_DONE）。
"""
import argparse
import glob
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SEAM = ("/media/a/新加卷/hanfeng/segment_sub_output/"
                "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_22.pkl")

_ap = argparse.ArgumentParser()
_ap.add_argument("--seam", nargs="+", default=[DEFAULT_SEAM],
                 help="一条或多条 seam pkl；也可给目录（收 seam_*.pkl）或通配符")
_ap.add_argument("--dis_cm", nargs=6, type=float, default=[10, 10, 10, 10, 10, 10],
                 metavar=("上", "下", "左", "右", "前", "后"),
                 help="焊缝到盒 6 个面的最近距离(cm)，顺序=上 下 左 右 前 后；后=开口面")
_ap.add_argument("--wall_cm", type=float, default=2.0, help="盒壁厚(cm)")
_ap.add_argument("--headless", action="store_true")
args = _ap.parse_args()


def _unit(v):
    import numpy as np
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def expand_seams(specs):
    """把 --seam 的若干项（pkl/目录/通配）展开成去重后的 pkl 路径列表（保序）。"""
    out = []
    for sp in specs:
        if os.path.isdir(sp):
            out += sorted(glob.glob(os.path.join(sp, "seam_*.pkl")))
        elif any(c in sp for c in "*?["):
            out += sorted(glob.glob(sp))
        else:
            out.append(sp)
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def find_obj(seam_path):
    d = os.path.dirname(seam_path)
    for pat in ("*_watertight.obj", "*.obj"):
        hits = sorted(glob.glob(os.path.join(d, pat)))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no obj in {d}")


def seam_frame_world(d):
    """焊缝几何 → world 系 (mid, t, d1, d2, bis, seam_len)（seam_* 字段本就在 world 系）。"""
    import numpy as np
    line = np.asarray(d["seam_line"], float)
    tang = np.asarray(d["seam_tangent"], float)
    limits = np.asarray(d["seam_limits"], float)
    i = int(d.get("middle", len(line) // 2))
    mid = line[i]
    t = _unit(tang[i])
    d1 = limits[i, 0]
    d2 = limits[i, 1]
    bis = _unit(d1 + d2)
    seam_len = float(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)))
    return mid, t, d1, d2, bis, seam_len


def piece_pose_world(d):
    """工件 world 位姿 [x,y,z,qw,qx,qy,qz]（pkl piece_pose 即 world 系，wxyz）。"""
    import numpy as np
    return np.asarray(d["piece_pose"][0], float)


def seam_polyline_world(d):
    """seam_line（world 系点列 (N,3)）原样返回。"""
    import numpy as np
    return np.asarray(d["seam_line"], float)


def build_open_box(d, dis_cm, wall):
    """据焊缝几何 + 6 个面距离(上下左右前后, 米) 造一个 open_box（open_face=back）。

    盒轴：Z=世界+Z(上)；开口 back(−X) 朝水平角平分线 bis_h（bis 穿过开口）；front(+X)=−bis_h；Y=Z×X。
    焊缝折线投到三轴得跨度，配 6 个 dis 唯一解出 size 与盒心。返回 (prims, info)。
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as Rsp
    from gt_gen import obstacles as ob

    mid, t, d1, d2, bis, seam_len = seam_frame_world(d)
    pts = seam_polyline_world(d)                         # (N,3) world

    zb = np.array([0.0, 0.0, 1.0])
    bis_h = bis - float(np.dot(bis, zb)) * zb            # 角平分线的水平投影
    if float(np.linalg.norm(bis_h)) < 1e-6:              # 退化：bis 近垂直 → 退用 d1 的水平分量
        bis_h = d1 - float(np.dot(d1, zb)) * zb
        if float(np.linalg.norm(bis_h)) < 1e-6:
            bis_h = np.array([1.0, 0.0, 0.0])
    bis_h = _unit(bis_h)

    x_axis = -bis_h                                      # front(+X) 朝工件；back(−X,开口) 朝 +bis_h
    z_axis = zb                                          # 上/下 = 世界 ±Z
    y_axis = _unit(np.cross(z_axis, x_axis))             # 左(+Y)
    Rm = np.column_stack([x_axis, y_axis, z_axis])       # 盒局部→world（列=[X,Y,Z]）

    a = pts @ x_axis                                     # 焊缝沿 X(前后) 投影
    b = pts @ y_axis                                     # 沿 Y(左右)
    c = pts @ z_axis                                     # 沿 Z(上下)
    a0, a1 = float(a.min()), float(a.max())
    b0, b1 = float(b.min()), float(b.max())
    c0, c1 = float(c.min()), float(c.max())

    dt, db, dl, dr, df, dk = [float(v) for v in dis_cm]  # 上 下 左 右 前 后（米）
    sz = (c1 - c0) + dt + db
    sy = (b1 - b0) + dl + dr
    sx = (a1 - a0) + df + dk
    cen_c = ((c1 + dt) + (c0 - db)) / 2.0
    cen_b = ((b1 + dl) + (b0 - dr)) / 2.0
    cen_a = ((a1 + df) + (a0 - dk)) / 2.0
    C = cen_a * x_axis + cen_b * y_axis + cen_c * z_axis  # 盒心(world)

    rpy = [float(v) for v in Rsp.from_matrix(Rm).as_euler("xyz", degrees=True)]
    prims = ob.build("open_box", C.tolist(), anchor_rpy_deg=tuple(rpy),
                     size=(float(sx), float(sy), float(sz)),
                     wall=float(wall), open_face="back")

    achieved = {                                         # 实测各面到焊缝最近距离（应=输入 dis）
        "上": cen_c + sz / 2 - c1, "下": c0 - (cen_c - sz / 2),
        "左": cen_b + sy / 2 - b1, "右": b0 - (cen_b - sy / 2),
        "前": cen_a + sx / 2 - a1, "后": a0 - (cen_a - sx / 2)}
    info = dict(mid=mid, bis=bis, bis_h=bis_h, center=C, rpy=rpy,
                size=(float(sx), float(sy), float(sz)), seam_len=seam_len,
                achieved=achieved)
    return prims, info


# ----------------------------------------------------------------------------
try:
    import isaacsim  # noqa: F401  注册 omni.* 模块路径（必须在 import omni 之前）
except ImportError:
    pass
from omni.isaac.kit import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np  # noqa: E402

sys.path.insert(0, ROOT)
from gt_gen import compat  # noqa: F401,E402  warp shim

from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.objects import cuboid as _cuboid  # noqa: E402
from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa: E402


def spawn_box_prim(path, name, prim, color):
    """open_box 的一面 Box(dims + pose[x,y,z,qw,qx,qy,qz]) → 纯视觉 VisualCuboid。"""
    pos = np.asarray(prim.pose[:3], float)
    quat = np.asarray(prim.pose[3:7], float)            # wxyz
    _cuboid.VisualCuboid(prim_path=path, name=name, position=pos, orientation=quat,
                         size=1.0, scale=np.asarray(prim.dims, float),
                         color=np.asarray(color, float))


def spawn_segment(path, name, p0, p1, color, thick=0.01):
    """两点 p0→p1 间一根细长方体，用作折线段（焊缝由若干段拼成一条红线）。"""
    from scipy.spatial.transform import Rotation as Rsp
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    seg = p1 - p0
    L = float(np.linalg.norm(seg))
    if L < 1e-9:
        return
    d_hat = seg / L
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d_hat); s = float(np.linalg.norm(v)); c = float(np.dot(z, d_hat))
    if s < 1e-9:
        Rm = np.eye(3) if c > 0 else Rsp.from_euler("x", 180, degrees=True).as_matrix()
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rm = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    quat = Rsp.from_matrix(Rm).as_quat()                # xyzw
    pos = (p0 + p1) / 2.0
    _cuboid.VisualCuboid(prim_path=path, name=name, position=pos,
                         orientation=np.r_[quat[3], quat[:3]], size=1.0,
                         scale=np.array([thick, thick, L]), color=np.asarray(color, float))


def spawn_seam_line(prefix, name0, seam_pts, color=(1.0, 0.0, 0.0), thick=0.01):
    """焊缝折线 seam_pts(world 系) → 一串红色细长方体段（prefix 保证 prim 路径唯一）。"""
    for s in range(len(seam_pts) - 1):
        spawn_segment(f"{prefix}/seg{s}", f"{name0}_{s}",
                      seam_pts[s], seam_pts[s + 1], color, thick)


def _spawn_obj_mesh(pth, obj_path):
    """trimesh 读 obj → 在 pth 定义一个灰色 UsdGeom.Mesh（纯视觉，无物理）。"""
    import trimesh
    import omni.usd
    from pxr import UsdGeom, Gf
    tm = trimesh.load(obj_path, force="mesh")
    verts = np.asarray(tm.vertices, float)
    faces = np.asarray(tm.faces, np.int64).reshape(-1, 3)
    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh.Define(stage, pth)
    mesh.CreatePointsAttr([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.72, 0.72)])


def spawn_workpiece(pth, obj_path, ppose):
    """工件摆到它的 world 位姿（关物理当纯视觉）。usd 缺失则用 trimesh 建 Mesh。"""
    usd_obj = obj_path.replace("_watertight.obj", ".usd")
    if not os.path.exists(usd_obj):
        usd_obj = os.path.splitext(obj_path)[0] + ".usd"
    if os.path.exists(usd_obj):
        add_reference_to_stage(usd_path=usd_obj, prim_path=pth)
    else:
        _spawn_obj_mesh(pth, obj_path)
    from omni.isaac.core.prims import XFormPrim
    XFormPrim(pth).set_world_pose(position=np.asarray(ppose[:3], float).tolist(),
                                  orientation=np.asarray(ppose[3:7], float).tolist())
    from pxr import Usd, UsdPhysics
    import omni.usd
    stg = omni.usd.get_context().get_stage()
    for pr in Usd.PrimRange(stg.GetPrimAtPath(pth)):
        if pr.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(pr).GetCollisionEnabledAttr().Set(False)
        if pr.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(pr).GetRigidBodyEnabledAttr().Set(False)


def main():
    seam_paths = expand_seams(args.seam)
    if not seam_paths:
        print("没有匹配到任何 seam pkl:", args.seam)
        simulation_app.close()
        return

    dis_m = [v / 100.0 for v in args.dis_cm]
    wall_m = args.wall_cm / 100.0

    seams = []
    for sp in seam_paths:
        d = pickle.load(open(sp, "rb"))
        prims, info = build_open_box(d, dis_m, wall_m)
        seams.append(dict(
            name=os.path.join(os.path.basename(os.path.dirname(sp)), os.path.basename(sp)),
            obj=find_obj(sp), ppose=piece_pose_world(d),
            seam_pts=seam_polyline_world(d), prims=prims, info=info))

    objs = sorted(set(s["obj"] for s in seams))
    if len(objs) > 1:
        print("警告: 多条 seam 指向不同工件 obj，只显示第 1 个工件，其余焊缝/盒仍按 world 坐标叠加：")
        for o in objs:
            print("   ", o)
    obj0, ppose0 = seams[0]["obj"], seams[0]["ppose"]

    print(f"工件     : {obj0}")
    print(f"dis(cm)  : 上={args.dis_cm[0]} 下={args.dis_cm[1]} 左={args.dis_cm[2]} "
          f"右={args.dis_cm[3]} 前={args.dis_cm[4]} 后(开口)={args.dis_cm[5]}  壁厚={args.wall_cm}cm")
    print(f"焊缝     : 共 {len(seams)} 条；每条套 1 个 open_box（开口面=后，朝角平分线）")
    for s in seams:
        info = s["info"]
        sx, sy, sz = info["size"]
        ach = info["achieved"]
        print(f"  {s['name']:>40s}  盒尺寸(X前后,Y左右,Z上下)="
              f"[{sx:.3f},{sy:.3f},{sz:.3f}]m  盒心={np.round(info['center'], 3)}")
        print(f"      实测面距(m) 上={ach['上']:.3f} 下={ach['下']:.3f} 左={ach['左']:.3f} "
              f"右={ach['右']:.3f} 前={ach['前']:.3f} 后={ach['后']:.3f}（应=输入 dis）")

    world = World(stage_units_in_meters=1.0)

    # 地面置于所有盒底最低处下方 0.3m
    zmin = min(float(s["info"]["center"][2] - s["info"]["size"][2] / 2.0) for s in seams)
    world.scene.add_default_ground_plane(z_position=zmin - 0.3)

    spawn_workpiece("/World/workpiece", obj0, ppose0)

    for si, s in enumerate(seams):
        spawn_seam_line(f"/World/seam/s{si}", f"seam_{si}", s["seam_pts"], color=[1.0, 0.0, 0.0])
        for k, prim in enumerate(s["prims"]):
            spawn_box_prim(f"/World/box/s{si}/face{k}", f"box_{si}_{k}",
                           prim, color=[0.62, 0.64, 0.67])

    world.reset()

    if args.headless:
        for _ in range(3):
            world.step(render=False)
        print(f"已 spawn 1 个工件 + {len(seams)} 条 seam（各含焊缝红线 + open_box 5 面）。")
        print("VIZ_OPEN_BOX_DONE")
        simulation_app.close()
        return

    try:
        from omni.kit.viewport.menubar.lighting.actions import _set_lighting_mode
        _set_lighting_mode("Grey Studio")
    except Exception:
        pass
    print("播放中（关闭窗口结束）。焊缝红线被 open_box 套住，开口面朝角平分线方向。")
    while simulation_app.is_running():
        world.step(render=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
