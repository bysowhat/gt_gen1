"""在 Isaac Sim 里【静态】可视化「机械臂(URDF) + cuRobo 碰撞球」于某一组关节角，用来肉眼核实碰撞球覆盖。

纯视觉方案（不走 URDF importer / articulation / 物理，避免 PhysX 惯量报错与 visuals 引用解析问题）：
  · 用轻量 CudaRobotModel 对给定关节角做 FK，得每个 link 在 base 系的位姿；
  · 把每个 link 的 visual mesh（从 URDF 解析）作为 UsdGeom.Mesh 摆到该位姿（含 visual origin）；
  · 碰撞球来自 CudaRobotModel.get_robot_as_spheres(q)（= ur12e_full.yml 里的球，FK 到 base 系），半透明叠加。

输入：一个 URDF（机械臂本体）+ 一个 cuRobo yml（碰撞球/关节定义）+ 一组关节角。静态，不回放。

运行（带显示器）：
    conda run -n env_isaaclab python scripts/viz_robot_spheres_isaacsim.py
    # 指定关节角（弧度，逗号分隔，按 joint_names 顺序）
    conda run -n env_isaaclab python scripts/viz_robot_spheres_isaacsim.py --joints "1.57,-2.0,1.36,-0.96,-1.57,0"
    # 指定别的 urdf / yml
    conda run -n env_isaaclab python scripts/viz_robot_spheres_isaacsim.py \
        --urdf /path/robot_description.urdf --cfg /path/ur12e_full.yml
无显示器自检：加 --headless（建好网格+球、跑几帧即退，打印网格数/球数 + VIZ_ROBOT_SPHERES_DONE）。
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

_NEW = "/home/a/Datas/curobo/example_new_robot/urdf_12e_260626"
_ap = argparse.ArgumentParser()
_ap.add_argument("--cfg", default=None, help="cuRobo robot yml（默认用 config.robot_cfg_path）；含碰撞球/关节定义")
_ap.add_argument("--urdf", default=None, help="机械臂 URDF（默认取 yml 里的 urdf_path）")
_ap.add_argument("--joints", default=None, help="关节角，逗号分隔（弧度，按 joint_names 顺序）；默认 retract")
_ap.add_argument("--opacity", type=float, default=0.35, help="碰撞球透明度 0~1")
_ap.add_argument("--no-spheres", action="store_true", help="只看机械臂本体，不画球")
_ap.add_argument("--headless", action="store_true")
args = _ap.parse_args()

from omni.isaac.kit import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

sys.path.insert(0, ROOT)
from gt_gen import compat  # noqa: F401,E402  warp shim
from gt_gen.config import load_config  # noqa: E402
from curobo.util_file import load_yaml  # noqa: E402
from curobo.types.base import TensorDeviceType  # noqa: E402
from curobo.types.robot import RobotConfig  # noqa: E402
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel  # noqa: E402

from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.objects import sphere as _sphere  # noqa: E402
import omni.usd  # noqa: E402
from pxr import UsdGeom, UsdShade, Gf, Sdf  # noqa: E402


def _f(s):
    return [float(v) for v in str(s).split()] if s is not None else [0, 0, 0]


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def quat_to_R(q):
    w, x, y, z = [float(v) for v in q]
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def T_of(xyz, R):
    m = np.eye(4); m[:3, :3] = R; m[:3, 3] = xyz; return m


def parse_visuals(urdf_path):
    """{link_name: [(abs_mesh_path, T_link_visual)]}（仅 mesh 类型 visual）。"""
    root = ET.parse(urdf_path).getroot()
    out = {}
    for ln in root.findall("link"):
        items = []
        for v in ln.findall("visual"):
            mesh = v.find("geometry/mesh")
            if mesh is None:
                continue
            fn = mesh.get("filename")
            if fn.startswith("file://"):
                fn = fn[7:]
            o = v.find("origin")
            xyz = _f(o.get("xyz")) if o is not None else [0, 0, 0]
            rpy = _f(o.get("rpy")) if o is not None else [0, 0, 0]
            items.append((fn, T_of(xyz, rpy_to_R(*rpy))))
        if items:
            out[ln.get("name")] = items
    return out


def add_mesh_prim(stage, path, tm, T_world, rgb=(0.62, 0.62, 0.62)):
    g = UsdGeom.Mesh.Define(stage, path)
    g.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in tm.vertices])
    g.CreateFaceVertexIndicesAttr([int(i) for i in tm.faces.reshape(-1)])
    g.CreateFaceVertexCountsAttr([3] * len(tm.faces))
    g.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    # USD 行向量约定：M_usd = T_world^T
    UsdGeom.Xformable(g.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(*[float(x) for x in T_world.T.reshape(-1)]))
    return g


def main():
    cfg = load_config()
    cfg_yml = args.cfg or cfg.robot_cfg_path
    rd = load_yaml(cfg_yml)
    kin = rd["robot_cfg"]["kinematics"]
    urdf_path = args.urdf or kin["urdf_path"]
    joint_names = list(kin["cspace"]["joint_names"])
    retract = list(kin["cspace"]["retract_config"])
    q = np.array([float(x) for x in args.joints.split(",")], float) if args.joints else np.array(retract, float)
    print("cfg :", cfg_yml)
    print("urdf:", urdf_path)
    print("关节角:", np.round(q, 4))

    visuals = parse_visuals(urdf_path)

    # 轻量 CudaRobotModel：FK 所有要画的 link + 取球
    ta = TensorDeviceType()
    kin["link_names"] = list(visuals.keys())
    model = CudaRobotModel(RobotConfig.from_dict(rd["robot_cfg"], ta).kinematics)
    qt = torch.as_tensor(q, dtype=torch.float32, device=ta.device).view(1, -1)
    st = model.get_state(qt)
    link_T = {}
    for ln in visuals:
        p = st.link_pose[ln].position[0].detach().cpu().numpy()
        R = quat_to_R(st.link_pose[ln].quaternion[0].detach().cpu().numpy())
        link_T[ln] = T_of(p, R)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()
    # 顶光，确保看得清
    UsdGeom.Xformable  # noqa
    from pxr import UsdLux
    UsdLux.DistantLight.Define(stage, "/World/Light").CreateIntensityAttr(2500.0)

    # 摆机械臂网格
    nmesh = 0
    for ln, items in visuals.items():
        for i, (mp, Tlv) in enumerate(items):
            try:
                tm = trimesh.load(mp, force="mesh")
            except Exception as e:
                print("  [warn] 加载失败", mp, e); continue
            add_mesh_prim(stage, "/World/robot/%s_%d" % (ln.replace("-", "_"), i),
                          tm, link_T[ln] @ Tlv)
            nmesh += 1
    print("机械臂网格数:", nmesh)

    # 画碰撞球（半透明绿）
    nsph = 0
    if not args.no_spheres:
        mtl = UsdShade.Material.Define(stage, "/World/Looks/Sph")
        sh = UsdShade.Shader.Define(stage, "/World/Looks/Sph/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.2, 0.85, 0.3))
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(args.opacity))
        mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        for si, s in enumerate(model.get_robot_as_spheres(qt)[0]):
            pos = np.ravel(s.position)
            if np.any(np.isnan(pos)) or float(s.radius) <= 1e-4:
                continue
            sp = _sphere.VisualSphere(prim_path="/World/spheres/s_%d" % si,
                                      position=pos, radius=float(s.radius))
            UsdShade.MaterialBindingAPI(sp.prim).Bind(mtl)
            nsph += 1
    print("碰撞球数:", nsph)

    world.reset()
    # 写一行自检摘要到文件（Isaac 接管 stdout 后 print 可能不可见）
    try:
        with open("/tmp/viz_robot_check.txt", "w") as _f2:
            _f2.write("mesh=%d sphere=%d joints=%s\n" % (nmesh, nsph, list(np.round(q, 4))))
    except Exception:
        pass
    print("显示中（关闭窗口结束）…" if not args.headless else "headless 自检…")
    frames = 0
    while simulation_app.is_running():
        world.step(render=not args.headless)
        frames += 1
        if args.headless and frames >= 5:
            break

    print("VIZ_ROBOT_SPHERES_DONE")
    simulation_app.close()


if __name__ == "__main__":
    main()
