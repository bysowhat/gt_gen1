"""在 Isaac Sim 里【并排】展示焊缝旁遮挡板的几种候选摆法，供肉眼选择（独立脚本，不动业务代码）。

为什么：place_seam_plate.py 加的板「位置/角度」对不对，2D 图说不清。本脚本据某条 seam 的真实几何
（焊缝切线 t、两面表面方向 d1/d2、角平分线 bis）解析地造出【多种候选板】，每种候选放在自己的一条
lane（= 一份 工件 + 红色焊缝线 + 该候选板，沿 base +Y 整体平移 lane_gap），一屏内对比，纯视觉无机器人。

候选（seam_22 实测：t≈+X，a面 d1=+Y 水平地面、b面 d2=+Z 竖直壁，夹角 90°，bis 朝斜上 +Y/+Z）：
  C1 水平板·焊缝正上方对称  : 板∥地面a(法向=地面法向)，沿壁方向 b_dir 抬高 n_cm，板心在焊缝正上方，宽沿±对称。
  C2 水平板·C型上臂        : 同 C1 朝向，但板心再沿地面方向 a_dir 平移 +宽/2 → 内边贴竖直壁、整块盖在地面臂正上方（标准 ⊏ 上臂）。
  C3 竖直板·∥壁·沿地面横挪 : 板∥壁b(法向=壁法向)，沿地面方向 a_dir 横挪 n_cm，板心与焊缝同高，宽沿竖直对称（= 现 --parallel b）。
  C4 竖直板·C型臂(从地面起): 同 C3 朝向，但板心再沿壁方向 b_dir 平移 +宽/2 → 下边贴地面、从地面向上立起（竖直版 C 臂）。
颜色：C1 绿 / C2 蓝 / C3 橙 / C4 品红；焊缝线红；base 轴 X 红 Y 绿 Z 蓝（仅 lane0）。

运行（带显示器）：
    conda run -n env_isaaclab python scripts/viz_seam_plate_candidates_isaacsim.py \
        --seam /media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_22.pkl \
        --n_cm 10 --width_cm 30 --length_pct 80 --thickness_cm 3 --lane_gap 2.0
无显示器自检：加 --headless（spawn + 跑几帧即退，打印候选/lane 数 + VIZ_CANDIDATES_DONE）。
只看某几个：--only C1,C2
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
_ap.add_argument("--seam", default=DEFAULT_SEAM)
_ap.add_argument("--n_cm", type=float, default=10.0, help="离焊缝距离 n(cm)：C1/C2 抬高量、C3/C4 横挪量")
_ap.add_argument("--width_cm", type=float, default=30.0, help="板宽(cm)")
_ap.add_argument("--length_pct", type=float, default=80.0, help="板长占焊缝弧长百分比(%)")
_ap.add_argument("--length_min_cm", type=float, default=10.0,
                 help="板长下限(cm)：实际板长 = max(length_pct%%×焊缝弧长, 此值)")
_ap.add_argument("--thickness_cm", type=float, default=3.0, help="板厚(cm)")
_ap.add_argument("--lane_gap", type=float, default=2.0, help="各候选沿 base +Y 的并排间距(米)")
_ap.add_argument("--only", default="", help="只看哪些候选，逗号分隔，如 C1,C2（空=全部）")
_ap.add_argument("--headless", action="store_true")
args = _ap.parse_args()


def _unit(v):
    import numpy as np
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def find_obj(seam_path):
    d = os.path.dirname(seam_path)
    for pat in ("*_watertight.obj", "*.obj"):
        hits = sorted(glob.glob(os.path.join(d, pat)))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no obj in {d}")


def seam_frame(d):
    """复刻 place_seam_plate.seam_frame：焊缝几何 → base 系 (mid, t, d1, d2, bis, seam_len)。"""
    import numpy as np
    from scipy.spatial.transform import Rotation as Rsp
    line = np.asarray(d["seam_line"], float)
    tang = np.asarray(d["seam_tangent"], float)
    limits = np.asarray(d["seam_limits"], float)
    i = int(d.get("middle", len(line) // 2))
    rp = np.asarray(d["robot_pose"][0], float)
    Rwr = Rsp.from_quat(np.r_[rp[4:7], rp[3]]).as_matrix()
    to_base_p = lambda p: Rwr.T @ (p - rp[:3])
    to_base_v = lambda v: Rwr.T @ v
    mid = to_base_p(line[i])
    t = _unit(to_base_v(tang[i]))
    d1 = to_base_v(limits[i, 0])
    d2 = to_base_v(limits[i, 1])
    bis = _unit(d1 + d2)
    seam_len = float(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)))
    return mid, t, d1, d2, bis, seam_len


def piece_pose_to_robot(d):
    """工件位姿在 base 系：inverse(robot_pose) * piece_pose → [x,y,z,qw,qx,qy,qz]（纯 numpy/scipy）。"""
    import numpy as np
    from scipy.spatial.transform import Rotation as Rsp
    rp = np.asarray(d["robot_pose"][0], float)
    pp = np.asarray(d["piece_pose"][0], float)
    Rr = Rsp.from_quat(np.r_[rp[4:7], rp[3]])
    Rp = Rsp.from_quat(np.r_[pp[4:7], pp[3]])
    pos = Rr.inv().apply(pp[:3] - rp[:3])
    q = (Rr.inv() * Rp).as_quat()                      # xyzw
    return np.r_[pos, q[3], q[:3]].astype(float)       # wxyz


def seam_polyline_base(d):
    """seam_line（world）→ base 系点列 (N,3)：复刻 seam_frame 的世界→base 变换（robot_pose 求逆）。"""
    import numpy as np
    from scipy.spatial.transform import Rotation as Rsp
    line = np.asarray(d["seam_line"], float)
    rp = np.asarray(d["robot_pose"][0], float)
    Rwr = Rsp.from_quat(np.r_[rp[4:7], rp[3]]).as_matrix()
    return np.array([Rwr.T @ (p - rp[:3]) for p in line])


def build_candidates(d):
    """据 seam 几何造各候选板 → [(key, desc, color, prims)]。prims 为 base 系 Box（含板心+朝向）。"""
    import numpy as np
    from gt_gen import obstacles as ob
    mid, t, d1, d2, bis, seam_len = seam_frame(d)

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


def spawn_box(path, name, box, off, color):
    pos = np.asarray(box.pose[:3], float) + off
    quat = np.asarray(box.pose[3:7], float)             # wxyz
    _cuboid.VisualCuboid(prim_path=path, name=name, position=pos, orientation=quat,
                         size=1.0, scale=np.asarray(box.dims, float), color=np.asarray(color, float))


def spawn_axis(path, name, axis_dir, off, color, length=0.4, thick=0.012):
    """从 base 原点出发的一根细长方体表示坐标轴。"""
    from scipy.spatial.transform import Rotation as Rsp
    axis_dir = _unit(axis_dir)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, axis_dir); s = float(np.linalg.norm(v)); c = float(np.dot(z, axis_dir))
    if s < 1e-9:
        Rm = np.eye(3) if c > 0 else Rsp.from_euler("x", 180, degrees=True).as_matrix()
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rm = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    quat = Rsp.from_matrix(Rm).as_quat()                # xyzw
    pos = axis_dir * (length / 2.0) + off
    _cuboid.VisualCuboid(prim_path=path, name=name, position=pos,
                         orientation=np.r_[quat[3], quat[:3]], size=1.0,
                         scale=np.array([thick, thick, length]), color=np.asarray(color, float))


def spawn_segment(path, name, p0, p1, off, color, thick=0.01):
    """两点 p0→p1 间一根细长方体，用作折线段（焊缝由若干段拼成一条红线）。"""
    from scipy.spatial.transform import Rotation as Rsp
    p0 = np.asarray(p0, float) + off
    p1 = np.asarray(p1, float) + off
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


def spawn_seam_line(lane, seam_pts, off, color=(1.0, 0.0, 0.0), thick=0.01):
    """焊缝折线 seam_pts(base 系) → 一串红色细长方体段。"""
    for s in range(len(seam_pts) - 1):
        spawn_segment(f"/World/seam/lane{lane}/seg{s}", f"seam_{lane}_{s}",
                      seam_pts[s], seam_pts[s + 1], off, color, thick)


def main():
    d = pickle.load(open(args.seam, "rb"))
    obj_path = find_obj(args.seam)
    ppose = piece_pose_to_robot(d)
    seam_pts = seam_polyline_base(d)
    cands, info = build_candidates(d)

    only = [s.strip().upper() for s in args.only.split(",") if s.strip()]
    if only:
        cands = [c for c in cands if c[0] in only]
    n_lane = len(cands)
    offsets = [np.array([0.0, i * args.lane_gap, 0.0], float) for i in range(n_lane)]
    mid = info["mid"]

    print("seam     :", args.seam)
    print("obj      :", obj_path)
    print(f"几何     : t={np.round(info['t'],3)} a_dir(地面)={np.round(info['a_dir'],3)} "
          f"b_dir(壁)={np.round(info['b_dir'],3)} bis={np.round(info['bis'],3)}")
    print(f"           焊缝中点 mid={np.round(mid,3)} 焊缝长={info['seam_len']:.3f}m  "
          f"板 长={info['length']:.3f} 宽={info['width']:.3f} 厚={info['thickness']:.3f}m")
    print(f"图例     : 并排 {n_lane} 条 lane（沿 base +Y 间距 {args.lane_gap}m），各 lane 一种候选：")
    for i, (key, desc, color, _) in enumerate(cands):
        print(f"  lane{i} (y={i*args.lane_gap:+.1f}m)  {key}  color={color}  {desc}")

    world = World(stage_units_in_meters=1.0)

    usd_obj = obj_path.replace("_watertight.obj", ".usd")
    if not os.path.exists(usd_obj):
        usd_obj = os.path.splitext(obj_path)[0] + ".usd"
    use_usd = os.path.exists(usd_obj)

    def spawn_obj_mesh(pth):
        import trimesh, omni.usd
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

    def spawn_workpiece(i, off):
        pth = f"/World/workpiece_{i}"
        if use_usd:
            add_reference_to_stage(usd_path=usd_obj, prim_path=pth)
        else:
            spawn_obj_mesh(pth)
        from omni.isaac.core.prims import XFormPrim
        XFormPrim(pth).set_world_pose(position=(ppose[:3] + off).tolist(),
                                      orientation=ppose[3:7].tolist())
        from pxr import Usd, UsdPhysics
        import omni.usd
        stg = omni.usd.get_context().get_stage()
        for pr in Usd.PrimRange(stg.GetPrimAtPath(pth)):
            if pr.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(pr).GetCollisionEnabledAttr().Set(False)
            if pr.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(pr).GetRigidBodyEnabledAttr().Set(False)

    # 地面（参考），置于焊缝下方 1m
    world.scene.add_default_ground_plane(z_position=float(mid[2]) - 1.0)

    for i, (off, (key, desc, color, prims)) in enumerate(zip(offsets, cands)):
        spawn_workpiece(i, off)
        spawn_seam_line(i, seam_pts, off, color=[1.0, 0.0, 0.0])
        for k, b in enumerate(prims):
            spawn_box(f"/World/cand/lane{i}/box{k}", f"cand_{i}_{k}", b, off, color)
        if i == 0:
            spawn_axis(f"/World/axes/x", "axis_x", info["t"], off, [1.0, 0.2, 0.2])   # 实为 t≈焊缝走向
            spawn_axis(f"/World/axes/adir", "axis_a", info["a_dir"], off, [0.2, 1.0, 0.2])
            spawn_axis(f"/World/axes/bdir", "axis_b", info["b_dir"], off, [0.2, 0.2, 1.0])

    world.reset()

    if args.headless:
        for _ in range(3):
            world.step(render=False)
        print(f"已 spawn {n_lane} 条 lane（各 工件+焊缝球+候选板）。")
        print("VIZ_CANDIDATES_DONE")
        simulation_app.close()
        return

    try:
        from omni.kit.viewport.menubar.lighting.actions import _set_lighting_mode
        _set_lighting_mode("Grey Studio")
    except Exception:
        pass
    print("播放中（关闭窗口结束）。lane 顺序见上面图例。")
    while simulation_app.is_running():
        world.step(render=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
