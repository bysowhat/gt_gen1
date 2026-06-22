"""在 Isaac Sim 里把【多条焊缝】及其遮挡板候选，全部叠加显示在【同一个工件】上（独立脚本，不动业务代码）。

为什么：place_seam_plate.py 加的板「位置/角度」对不对，2D 图说不清。本脚本据每条 seam 的真实几何
（焊缝切线 t、两面表面方向 d1/d2、角平分线 bis）解析地造出【多种候选板】，并把所有 seam 的焊缝线 +
候选板按【真实世界坐标】叠加到同一个工件上，一屏内同时看多条焊缝及其对应障碍，纯视觉、无机器人、无坐标轴。

统一坐标系：seam_line / seam_tangent / seam_limits / piece_pose 本就都在 world 系，故全程用 world 系——
工件只 spawn 一次摆在它真实的 world 位姿上，各 seam 的红线与候选板按世界坐标落到工件各自的位置，天然对齐。
（多条 seam 各自 robot_pose 不同，所以不能再用 base 系；world 系才能把它们摆到同一工件上。）

候选（seam_22 实测：t≈+X，a面 d1=+Y 水平地面、b面 d2=+Z 竖直壁，夹角 90°，bis 朝斜上 +Y/+Z）：
  C1 水平板·焊缝正上方对称  : 板∥地面a(法向=地面法向)，沿壁方向 b_dir 抬高 n_cm，板心在焊缝正上方，宽沿±对称。
  C2 水平板·C型上臂        : 同 C1 朝向，但板心再沿地面方向 a_dir 平移 +宽/2 → 内边贴竖直壁、整块盖在地面臂正上方（标准 ⊏ 上臂）。
  C3 竖直板·∥壁·沿地面横挪 : 板∥壁b(法向=壁法向)，沿地面方向 a_dir 横挪 n_cm，板心与焊缝同高，宽沿竖直对称（= 现 --parallel b）。
  C4 竖直板·C型臂(从地面起): 同 C3 朝向，但板心再沿壁方向 b_dir 平移 +宽/2 → 下边贴地面、从地面向上立起（竖直版 C 臂）。
颜色：C1 绿 / C2 蓝 / C3 橙 / C4 品红；焊缝线红。

运行（带显示器）：
    conda run -n env_isaaclab python scripts/viz_seam_plate_candidates_isaacsim.py \
        --seam .../seam_22.pkl .../seam_5.pkl .../seam_8.pkl \
        --n_cm 10 --width_cm 30 --length_pct 80 --thickness_cm 3
    # --seam 也支持给一个目录（自动收 seam_*.pkl）或通配（如 '.../seam_1*.pkl'）
无显示器自检：加 --headless（spawn + 跑几帧即退，打印焊缝/候选数 + VIZ_CANDIDATES_DONE）。
只看某几个候选：--only C3   或   --only C1,C2
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
_ap.add_argument("--n_cm", type=float, default=10.0, help="离焊缝距离 n(cm)：C1/C2 抬高量、C3/C4 横挪量")
_ap.add_argument("--width_cm", type=float, default=30.0, help="板宽(cm)")
_ap.add_argument("--length_pct", type=float, default=80.0, help="板长占焊缝弧长百分比(%)")
_ap.add_argument("--length_min_cm", type=float, default=10.0,
                 help="板长下限(cm)：实际板长 = max(length_pct%%×焊缝弧长, 此值)")
_ap.add_argument("--thickness_cm", type=float, default=3.0, help="板厚(cm)")
_ap.add_argument("--only", default="", help="只看哪些候选，逗号分隔，如 C1,C2（空=全部）")
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
    """焊缝几何 → world 系 (mid, t, d1, d2, bis, seam_len)（seam_* 字段本就在 world 系，无需变换）。"""
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


def build_candidates(d):
    """据 seam 几何造各候选板 → [(key, desc, color, prims)]。prims 为 world 系 Box（含板心+朝向）。"""
    import numpy as np
    from gt_gen import obstacles as ob
    mid, t, d1, d2, bis, seam_len = seam_frame_world(d)

    a_dir = _unit(d1 - float(np.dot(d1, t)) * t)        # a 面(d1)表面方向，⊥t
    b_dir = _unit(d2 - float(np.dot(d2, t)) * t)        # b 面(d2)表面方向，⊥t
    na = _unit(np.cross(t, a_dir))                      # a 面法向；取朝 bis 张开侧
    if float(np.dot(na, bis)) < 0:
        na = -na
    nb = _unit(np.cross(t, b_dir))                      # b 面法向；取朝 bis 张开侧
    if float(np.dot(nb, bis)) < 0:
        nb = -nb

    n = float(args.n_cm) / 100.0
    width = float(args.width_cm) / 100.0
    length = max(float(args.length_pct) / 100.0 * seam_len, float(args.length_min_cm) / 100.0)
    thickness = float(args.thickness_cm) / 100.0
    from scipy.spatial.transform import Rotation as Rsp

    def make(normal, anchor):
        x_axis = _unit(normal)                          # 板法向 → local X
        z_axis = t                                      # 板长方向 → local Z
        y_axis = _unit(np.cross(z_axis, x_axis))        # 板宽方向 → local Y
        R0 = np.column_stack([x_axis, y_axis, z_axis])
        rpy = [float(v) for v in Rsp.from_matrix(R0).as_euler("xyz", degrees=True)]
        return ob.build("plate", np.asarray(anchor, float).tolist(), anchor_rpy_deg=tuple(rpy),
                        length=length, width=width, thickness=thickness)

    cands = [
        ("C1", "水平板·焊缝正上方对称(∥地面a, 沿壁方向抬高n, 宽±对称)",
         [0.0, 0.85, 0.0], make(na, mid + n * b_dir)),
        ("C2", "水平板·C型上臂(∥地面a, 抬高n + 沿地面平移宽/2, 内边贴壁盖地面臂上方)",
         [0.1, 0.3, 1.0], make(na, mid + n * b_dir + (width / 2.0) * a_dir)),
        ("C3", "竖直板·∥壁b·沿地面横挪n(=现--parallel b, 板心与焊缝同高, 宽竖直对称)",
         [1.0, 0.5, 0.0], make(nb, mid + n * a_dir)),
        ("C4", "竖直板·C型臂(∥壁b, 横挪n + 沿壁平移宽/2, 下边贴地面向上立)",
         [1.0, 0.0, 1.0], make(nb, mid + n * a_dir + (width / 2.0) * b_dir)),
    ]
    info = dict(mid=mid, t=t, a_dir=a_dir, b_dir=b_dir, na=na, nb=nb, bis=bis,
                seam_len=seam_len, width=width, length=length, thickness=thickness)
    return cands, info


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


def spawn_box(path, name, box, color):
    """world 系 Box → 纯视觉 VisualCuboid。box.pose=[x,y,z,qw,qx,qy,qz]。"""
    pos = np.asarray(box.pose[:3], float)
    quat = np.asarray(box.pose[3:7], float)             # wxyz
    _cuboid.VisualCuboid(prim_path=path, name=name, position=pos, orientation=quat,
                         size=1.0, scale=np.asarray(box.dims, float), color=np.asarray(color, float))


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
    only = [s.strip().upper() for s in args.only.split(",") if s.strip()]

    seams = []
    for sp in seam_paths:
        d = pickle.load(open(sp, "rb"))
        cands, info = build_candidates(d)
        if only:
            cands = [c for c in cands if c[0] in only]
        seams.append(dict(
            name=os.path.join(os.path.basename(os.path.dirname(sp)), os.path.basename(sp)),
            obj=find_obj(sp), ppose=piece_pose_world(d),
            seam_pts=seam_polyline_world(d), info=info, cands=cands))

    objs = sorted(set(s["obj"] for s in seams))
    if len(objs) > 1:
        print("警告: 多条 seam 指向不同工件 obj，只会显示第 1 个工件，其余焊缝/板仍按 world 坐标叠加：")
        for o in objs:
            print("   ", o)
    obj0, ppose0 = seams[0]["obj"], seams[0]["ppose"]

    print(f"工件     : {obj0}")
    print(f"焊缝     : 共 {len(seams)} 条；每条候选 {len(seams[0]['cands'])} 种（only={only or '全部'}）")
    for s in seams:
        info = s["info"]
        keys = ",".join(c[0] for c in s["cands"])
        print(f"  {s['name']:>40s}  mid={np.round(info['mid'], 3)} "
              f"焊缝长={info['seam_len']:.3f}m  候选[{keys}]")
    print("颜色     : C1 绿 / C2 蓝 / C3 橙 / C4 品红；焊缝线红")

    world = World(stage_units_in_meters=1.0)

    # 地面置于所有焊缝中点最低处下方 1m
    zmin = min(float(s["info"]["mid"][2]) for s in seams)
    world.scene.add_default_ground_plane(z_position=zmin - 1.0)

    # 只 spawn 一个工件（第 1 条 seam 的），其余焊缝/板按 world 坐标叠加上去
    spawn_workpiece("/World/workpiece", obj0, ppose0)

    for si, s in enumerate(seams):
        spawn_seam_line(f"/World/seam/s{si}", f"seam_{si}", s["seam_pts"], color=[1.0, 0.0, 0.0])
        for key, desc, color, prims in s["cands"]:
            for k, b in enumerate(prims):
                spawn_box(f"/World/cand/s{si}/{key}/box{k}", f"cand_{si}_{key}_{k}", b, color)

    world.reset()

    if args.headless:
        for _ in range(3):
            world.step(render=False)
        print(f"已 spawn 1 个工件 + {len(seams)} 条 seam（各含焊缝线 + 候选板）。")
        print("VIZ_CANDIDATES_DONE")
        simulation_app.close()
        return

    try:
        from omni.kit.viewport.menubar.lighting.actions import _set_lighting_mode
        _set_lighting_mode("Grey Studio")
    except Exception:
        pass
    print("播放中（关闭窗口结束）。多条焊缝及其候选板叠加在同一工件上，颜色见上面图例。")
    while simulation_app.is_running():
        world.step(render=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
