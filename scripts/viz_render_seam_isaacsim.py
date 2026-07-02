"""【独立脚本】在 Isaac Sim 里可视化：工件 + 焊缝 + 指定初始关节角下的机械臂。

仿照 render/visualize_pointcloud.py（visualize_render_seam_pointcloud）的思路——读
render_seam.py 产出的某个 pose 目录，但这里不再反投影点云，而是把三样东西摆进 Isaac Sim：
    · 工件网格（直接解析 --obj 的 OBJ，按 render_info 的 workpiece_pose7 摆到 base 系）；
    · 焊缝（render_info 的 seam_pose(20,7)，base 系，画成红球 + 黄折线 + 每点蓝色 z 轴朝向）；
    · 机械臂（--robot-usd 载为 Articulation，摆到 --joint-angles 指定的初始关节角）。

本脚本【不依赖本项目任何文件/模块】（不 import gt_gen / render 下任何东西），OBJ 自己解析，
可单独拷出去跑。第三方依赖只用 isaaclab / numpy / torch（跑 Isaac 本就有）。

坐标系：render_info 的 workpiece_pose7、seam_pose 都在机械臂 base(arm) 系；本脚本把机械臂
根节点摆在世界原点（pos=0, rot=I），故 base 系 == 世界系，两者位姿可直接当世界位姿用。
（render 里为约束离地用了 z_lift/仓库下移，那只影响世界 z 的“真实渲染坐标”；base 系视角无关，
这里不需要 z_lift。）

用法（有显示器，弹 GUI）：
    conda run -n env_isaaclab --no-capture-output python scripts/viz_render_seam_isaacsim.py \
        --obj '/media/a/upan/tempt/2/柱_..._part_watertight.obj' \
        --render-dir '/home/a/Downloads/render_out2/m_..._part/m_..._part_seam_40_pose0' \
        --robot-usd '/path/to/robot.usd' \
        --joint-angles '0,-1.57,1.57,0,1.57,0'
    · --joint-angles 缺省时用 render_info 里的 retract_config（渲染时机械臂的姿态）。
    · 无显示器只想自检不弹窗：加 --headless。
"""
import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def parse_args():
    ap = argparse.ArgumentParser(description="Isaac Sim 可视化 工件+焊缝+机械臂(指定关节角)")
    ap.add_argument("--obj", required=True, help="工件 .obj 路径")
    ap.add_argument("--render-dir", required=True,
                    help="render_seam.py 的 pose 目录（含 left/ 或 right/ 的 render_info.npy）")
    ap.add_argument("--robot-usd", required=True, help="机械臂 USD 路径")
    ap.add_argument("--joint-angles", default=None,
                    help="逗号分隔的初始关节角(弧度)，顺序同 --joint-names / render_info 的 joint_names；"
                         "缺省用 render_info 的 retract_config")
    ap.add_argument("--joint-names", default=None,
                    help="逗号分隔的关节名，与 --joint-angles 对应；缺省用 render_info 的 joint_names，"
                         "再缺省则按机械臂内部关节顺序positional赋值")
    ap.add_argument("--seam-radius", type=float, default=0.008, help="焊缝点红球半径(米)")
    ap.add_argument("--no-seam-frames", action="store_true", help="不画每个焊缝点的朝向坐标轴")
    ap.add_argument("--no-ground", action="store_true", help="不加地面（避免遮挡工件）")
    return ap


# ---- AppLauncher 必须在导入其余 isaaclab/pxr 之前启动 ----
from isaaclab.app import AppLauncher  # noqa: E402

_parser = parse_args()
AppLauncher.add_app_launcher_args(_parser)
args_cli = _parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ======================= 启动后再导入 =======================
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402

import omni.usd  # noqa: E402
from pxr import UsdGeom, Gf  # noqa: E402


# ======================= 小工具（纯 numpy / USD，无项目依赖）=======================
def load_meta(render_dir):
    """从 pose 目录任一侧读 render_info.npy（每侧都含完整信息）。"""
    for side in ("left", "right"):
        p = Path(render_dir) / side / "render_info.npy"
        if p.exists():
            return np.load(p, allow_pickle=True).item()
    raise FileNotFoundError(f"{render_dir} 下未找到 left/render_info.npy 或 right/render_info.npy")


def load_obj(path):
    """极简 OBJ 解析：只取顶点 v 和面 f（忽略法线/UV/材质），多边形扇形三角化。

    返回 (verts (N,3) float, tris (M,3) int)。f 支持 'i'、'i/j'、'i/j/k'、负索引。
    """
    verts = []
    tris = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                toks = line.split()[1:]
                idx = []
                for t in toks:
                    s = t.split("/")[0]
                    if not s:
                        continue
                    i = int(s)
                    idx.append(i - 1 if i > 0 else len(verts) + i)   # OBJ 1-based；负=倒数
                for k in range(1, len(idx) - 1):                     # 扇形三角化
                    tris.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(verts, dtype=np.float64), np.asarray(tris, dtype=np.int64)


def quat_wxyz_to_R(q):
    """四元数 (w,x,y,z) -> 3x3 旋转矩阵。"""
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) + 1e-12
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def set_prim_pose(prim_path, pos, quat_wxyz):
    """把 prim 世界位姿设为 (pos, quat_wxyz)。双精度 xformOp，匹配 spawn 出的 prim。"""
    stage = omni.usd.get_context().get_stage()
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xf.ClearXformOpOrder()
    t = xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    t.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    o = xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    w, x, y, z = [float(v) for v in quat_wxyz]
    o.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))


def spawn_mesh(prim_path, verts, tris, pose7, color=(0.6, 0.6, 0.62)):
    """把 (verts, tris) 作 UsdGeom.Mesh 建到 prim_path，并摆到 pose7(base 系)。"""
    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in verts])
    mesh.CreateFaceVertexCountsAttr([3] * len(tris))
    mesh.CreateFaceVertexIndicesAttr([int(i) for tri in tris for i in tri])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    set_prim_pose(prim_path, pose7[:3], pose7[3:7])


def add_sphere(prim_path, center, radius, color):
    stage = omni.usd.get_context().get_stage()
    s = UsdGeom.Sphere.Define(stage, prim_path)
    s.CreateRadiusAttr(float(radius))
    s.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xf = UsdGeom.Xformable(s)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))


def add_polyline(prim_path, points, width, color):
    stage = omni.usd.get_context().get_stage()
    c = UsdGeom.BasisCurves.Define(stage, prim_path)
    c.CreateTypeAttr("linear")
    c.CreateCurveVertexCountsAttr([len(points)])
    c.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
    c.CreateWidthsAttr([float(width)] * len(points))
    c.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    c.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def draw_seam(seam_pose, radius, draw_frames):
    """seam_pose (K,7) base 系：红球标点 + 黄折线连线 + （可选）每点 xyz 朝向短轴。"""
    seam_pose = np.asarray(seam_pose, dtype=np.float64)
    if seam_pose.ndim != 2 or seam_pose.shape[0] == 0:
        print("[viz] render_info 无有效 seam_pose，跳过焊缝绘制")
        return
    pts = seam_pose[:, :3]
    for i, p in enumerate(pts):
        add_sphere(f"/World/seam/pt_{i:03d}", p, radius, (0.9, 0.1, 0.1))
    add_polyline("/World/seam/line", pts, radius * 0.8, (1.0, 0.9, 0.1))
    if draw_frames:
        axlen = radius * 6.0
        cols = [(1, 0, 0), (0, 1, 0), (0.2, 0.4, 1.0)]   # x红 y绿 z蓝
        for i, p7 in enumerate(seam_pose):
            R = quat_wxyz_to_R(p7[3:7])
            for a in range(3):
                add_polyline(f"/World/seam/frame_{i:03d}_ax{a}",
                             [p7[:3], p7[:3] + R[:, a] * axlen], radius * 0.6, cols[a])
    print(f"[viz] 焊缝点 {len(pts)} 个（红球+黄线{'+朝向轴' if draw_frames else ''}）")


def resolve_joints(meta):
    """定初始关节角与关节名。返回 (names 或 None, angles list)。

    优先 --joint-angles / --joint-names；缺省回退 render_info 的 retract_config / joint_names。
    names 为 None 表示后续按机械臂内部关节顺序positional赋值。
    """
    if args_cli.joint_angles is not None:
        angles = [float(x) for x in args_cli.joint_angles.split(",") if x.strip() != ""]
    else:
        rc = meta.get("retract_config")
        if rc is None and meta.get("jointstates") is not None:
            rc = np.asarray(meta["jointstates"]).reshape(-1)
        if rc is None:
            raise RuntimeError("未提供 --joint-angles，且 render_info 无 retract_config/jointstates")
        angles = [float(x) for x in np.asarray(rc).reshape(-1)]
        print("[viz] --joint-angles 缺省，用 render_info 的 retract_config")

    if args_cli.joint_names is not None:
        names = [s.strip() for s in args_cli.joint_names.split(",") if s.strip() != ""]
    else:
        names = list(meta.get("joint_names")) if meta.get("joint_names") is not None else None
    return names, angles


def joint_tensor(robot, names, angles, device, dtype):
    """按机械臂内部关节顺序排好初始关节角，返回 (1, njoints) tensor。"""
    sim_names = list(robot.data.joint_names)
    if names is not None:
        name2val = {n: v for n, v in zip(names, angles)}
        vals = [float(name2val.get(n, 0.0)) for n in sim_names]
        miss = [n for n in sim_names if n not in name2val]
        if miss:
            print(f"[viz][warn] 机械臂关节 {miss} 未在给定关节名中，置 0")
    else:
        if len(angles) != len(sim_names):
            print(f"[viz][warn] 关节角个数 {len(angles)} ≠ 机械臂关节数 {len(sim_names)}，"
                  f"按位截断/补0（顺序={sim_names}）")
        vals = [float(angles[i]) if i < len(angles) else 0.0 for i in range(len(sim_names))]
    return torch.tensor([vals], device=device, dtype=dtype)


def main():
    meta = load_meta(args_cli.render_dir)
    wp_pose7 = np.asarray(meta["workpiece_pose7"], dtype=np.float64).reshape(-1)
    seam_pose = meta.get("seam_pose")
    names, angles = resolve_joints(meta)
    print(f"[viz] 工件 pose7(base)={np.round(wp_pose7, 4)}")
    print(f"[viz] 初始关节角={np.round(angles, 4)}  关节名={names}")

    # ---- 场景：仿真上下文 + 灯 + 地面 ----
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim.set_camera_view([wp_pose7[0] + 2.0, wp_pose7[1] + 2.0, wp_pose7[2] + 1.5],
                        [wp_pose7[0], wp_pose7[1], wp_pose7[2]])
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/Light", light_cfg)
    if not args_cli.no_ground:
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

    # ---- 机械臂：根节点在世界原点（=base 系），初始关节角=retract/给定 ----
    robot_cfg = ArticulationCfg(
        prim_path="/World/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args_cli.robot_usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
    )
    robot = Articulation(cfg=robot_cfg)

    # ---- 工件网格（base 系摆放）----
    verts, tris = load_obj(args_cli.obj)
    print(f"[viz] 工件 OBJ 顶点={len(verts)} 三角面={len(tris)}")
    spawn_mesh("/World/workpiece", verts, tris, wp_pose7)

    # ---- 焊缝 ----
    if seam_pose is not None:
        draw_seam(seam_pose, args_cli.seam_radius, not args_cli.no_seam_frames)

    # ---- 播放并把机械臂摆到初始关节角 ----
    sim.reset()
    device = robot.data.default_root_state.device
    dtype = robot.data.default_root_state.dtype
    q = joint_tensor(robot, names, angles, device, dtype)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    robot.write_data_to_sim()
    sim.play()

    print("[viz] 场景就绪，GUI 交互中（Ctrl+C 或关闭窗口退出）...")
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        sim.step()
        robot.update(dt=sim_dt)


if __name__ == "__main__":
    print(f"PID: {os.getpid()}")
    main()
    simulation_app.close()
