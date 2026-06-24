"""在 Isaac Sim 里可视化 compute_goal_poses.py 算出的【观测位姿序列】（独立脚本，纯视觉无机器人）。

输入三样（与 compute_goal_poses.py 的入参一一对应）：
  --seam-pkl ：per-seam pkl（提供目标焊缝 seam_line 画红线 + piece_pose 摆工件 + 自动找 USD）
  --usd      ：工件 USD（缺省取 seam-pkl 同目录 *_part.usd）
  --save     ：compute_goal_poses.py --save 落盘的 goal poses pkl（list[dict]，含 cam_pose (K,B,7) 等）

展示内容：
  · 目标焊缝   ：seam_line 折线，**红色直线**。
  · 每个 pose  ：在该观测位姿放一个相机（cam_pose 在 piece 系 → 经 piece_pose 变到 world）：
                 - 相机视锥（FOV 截头锥，用 scene_pose 同款 p1..p8，apex=光心、+z 朝焊缝）；
                 - 序列按青→黄渐变上色，便于看覆盖顺序。
  · 真实相机   ：在【每个】观测位姿都额外放一个真实 USD 相机（FOV 取 scene_pose 同款，
                 -z 光轴对齐 frustum 的 +z）。非 headless 下再开第 2 个视口窗口，用其中第
                 --cam-index 个相机出图：一窗看全局俯视、另一窗看该相机的实拍视角。

运行（带显示器）：
  conda run -n env_isaaclab python scripts/viz_goal_poses_isaacsim.py \
    --seam-pkl .../seam_25.pkl --usd .../*_part.usd --save /tmp/goal_poses/goal_poses.pkl
无显示器自检：加 --headless（spawn + 跑几帧即退，打印 pose 数 + VIZ_GOAL_POSES_DONE）。
"""
import argparse
import glob
import os
import pickle

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_ap = argparse.ArgumentParser(description="可视化 compute_goal_poses 的观测位姿序列（相机 + 红色焊缝线）")
_ap.add_argument("--seam-pkl", required=True, help="per-seam pkl（seam_line 红线 + piece_pose + USD 来源）")
_ap.add_argument("--usd", default=None, help="工件 USD；缺省取 seam-pkl 同目录 *_part.usd")
_ap.add_argument("--save", required=True, help="compute_goal_poses.py --save 落盘的 goal poses pkl")
_ap.add_argument("--result-index", type=int, default=0, help="save 里第几个 robot_pose 结果（默认 0）")
_ap.add_argument("--variant", type=int, default=0, help="cam_pose 的第几个变体 K（默认 0）")
_ap.add_argument("--cam-index", type=int, default=0,
                 help="第 2 个视口用第几个观测位姿的真实相机出图（默认 0；每个位姿都会放真实相机）")
_ap.add_argument("--fov-len", type=float, default=1.0,
                 help="视锥缩放：远平面再沿 +z 拉伸的倍数（1=原始 FOV，越大锥越长）")
_ap.add_argument("--headless", action="store_true")
args = _ap.parse_args()


# ----------------------------------------------------------------------------
# 几何工具（纯 numpy / scipy，不依赖 isaac）
# ----------------------------------------------------------------------------
def fov_corners(fov_len=1.0):
    """scene_pose.py 同款相机 FOV 八顶点（相机局部系，+z 朝焊缝）。

    近平面 p1..p4 @ z=0.4；远平面 p5..p8 @ z=0.4*(1+scl_z)。fov_len 把远平面再沿 z 拉长。
    返回 (near(4,3), far(4,3))，顺序 [右下, 左下, 左上, 右上] 与 scene_pose 一致。
    """
    scl = 7.0 / 11.0       # config_pose.ConfigurationPose.scl
    scl_z = 9.5 / 11.0     # config_pose.ConfigurationPose.scl_z
    near = np.array([
        [0.135 * scl, -0.20 * 5 / 6 * scl, 0.4],
        [-0.135 * scl, -0.20 * 5 / 6 * scl, 0.4],
        [-0.135 * scl, 0.20 * 5 / 6 * scl, 0.4],
        [0.135 * scl, 0.20 * 5 / 6 * scl, 0.4],
    ], dtype=float)
    dz = 0.140 * scl_z * scl
    dy = 0.185 * 5 / 6 * scl_z * scl
    z_far = 0.4 * (1 + scl_z)
    far = np.array([
        [0.135 * scl + dz, -0.20 * 5 / 6 * scl - dy, z_far],
        [-0.135 * scl - dz, -0.20 * 5 / 6 * scl - dy, z_far],
        [-0.135 * scl - dz, 0.20 * 5 / 6 * scl + dy, z_far],
        [0.135 * scl + dz, 0.20 * 5 / 6 * scl + dy, z_far],
    ], dtype=float)
    if fov_len != 1.0:                      # 沿 z 把远平面整体往外推，锥更长更醒目
        far = near + (far - near) * float(fov_len)
    return near, far


def compose_pose(piece_pose, cam_pose):
    """cam_pose（piece 局部系, [x,y,z,qw,qx,qy,qz]）→ world，叠加工件 world 位姿 piece_pose（同格式）。

    返回 (pos_w(3,), R_w(3,3))。demo 中 piece_pose=identity 时即 cam_pose 原值。
    """
    from scipy.spatial.transform import Rotation as Rsp
    pp = np.asarray(piece_pose, float)
    cp = np.asarray(cam_pose, float)
    Rp = Rsp.from_quat([pp[4], pp[5], pp[6], pp[3]])     # wxyz → xyzw
    Rc = Rsp.from_quat([cp[4], cp[5], cp[6], cp[3]])
    Rw = Rp * Rc
    pos_w = pp[:3] + Rp.apply(cp[:3])
    return pos_w, Rw.as_matrix()


def find_usd(seam_path):
    d = os.path.dirname(seam_path)
    hits = sorted(glob.glob(os.path.join(d, "*_part.usd")))
    if not hits:
        raise FileNotFoundError(f"未在 {d} 找到 *_part.usd，请用 --usd 指定")
    return hits[0]


# ----------------------------------------------------------------------------
# Isaac Sim（与 viz_seam_open_box_isaacsim.py 同栈：SimulationApp + omni.isaac.core）
# ----------------------------------------------------------------------------
try:
    import isaacsim  # noqa: F401  注册 omni.* 模块路径（必须在 import omni 之前）
except ImportError:
    pass
from omni.isaac.kit import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.objects import cuboid as _cuboid  # noqa: E402
from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa: E402


def spawn_segment(path, name, p0, p1, color, thick=0.005):
    """两点 p0→p1 间一根细长方体（折线段 / 视锥棱）。"""
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


def spawn_seam_line(prefix, seam_pts, color=(1.0, 0.0, 0.0), thick=0.008):
    """焊缝折线 seam_pts(world 系) → 一串红色细长方体段。"""
    for s in range(len(seam_pts) - 1):
        spawn_segment(f"{prefix}/seg{s}", f"seam_{s}", seam_pts[s], seam_pts[s + 1], color, thick)


def spawn_camera(prefix, idx, pos_w, R_w, near, far, color):
    """在一个观测位姿放相机：视锥截头锥（近矩形 + 远矩形 + 4 条侧棱 + 4 条 apex→近角）。"""
    apex = np.asarray(pos_w, float)
    nw = apex + near @ R_w.T                            # (4,3) 近平面角点 world
    fw = apex + far @ R_w.T                             # (4,3) 远平面角点 world
    for k in range(4):                                  # 近矩形
        spawn_segment(f"{prefix}/near{k}", f"c{idx}_near{k}", nw[k], nw[(k + 1) % 4], color, 0.004)
    for k in range(4):                                  # 远矩形
        spawn_segment(f"{prefix}/far{k}", f"c{idx}_far{k}", fw[k], fw[(k + 1) % 4], color, 0.004)
    for k in range(4):                                  # 侧棱（近→远）
        spawn_segment(f"{prefix}/side{k}", f"c{idx}_side{k}", nw[k], fw[k], color, 0.004)
    for k in range(4):                                  # apex（光心）→近角，凸显视锥顶点=相机位置
        spawn_segment(f"{prefix}/apex{k}", f"c{idx}_apex{k}", apex, nw[k], color, 0.004)


def spawn_real_camera(prim_path, pos_w, R_w, near):
    """在一个观测位姿放【真实 USD 相机】并返回其 prim 路径。

    约定：frustum 用 +z 当光轴（朝焊缝），而 USD 相机沿本地 -z 看 → 绕 x 转 180° 把 -z 对到 +z。
    FOV 由近平面半宽高（fov_corners 同款 scene_pose 几何）反推 focal/aperture，使第 2 个视口的
    取景与该位姿的视锥一致。
    """
    from pxr import UsdGeom, Gf
    from scipy.spatial.transform import Rotation as Rsp
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    cam = UsdGeom.Camera.Define(stage, prim_path)

    R_cam = np.asarray(R_w, float) @ Rsp.from_euler("x", 180, degrees=True).as_matrix()
    quat = Rsp.from_matrix(R_cam).as_quat()             # xyzw
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(pos_w[0]), float(pos_w[1]), float(pos_w[2])))
    xf.AddOrientOp().Set(Gf.Quatf(float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])))

    half_x = abs(float(near[0][0]))                     # 0.135*scl
    half_y = abs(float(near[0][1]))                     # 0.20*5/6*scl
    z_near = float(near[0][2])                          # 0.4
    h_aper = 20.955                                     # USD 默认水平光圈(mm)，作基准
    focal = h_aper / (2.0 * (half_x / z_near))          # tan(hFOV/2)=half_x/z_near
    v_aper = 2.0 * focal * (half_y / z_near)            # 保持竖直 FOV 与 frustum 一致
    cam.GetFocalLengthAttr().Set(float(focal))
    cam.GetHorizontalApertureAttr().Set(float(h_aper))
    cam.GetVerticalApertureAttr().Set(float(v_aper))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10.0))
    return prim_path


def spawn_workpiece(pth, usd_path, ppose):
    """工件 USD 摆到它的 world 位姿（关物理当纯视觉）。"""
    add_reference_to_stage(usd_path=usd_path, prim_path=pth)
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


def cam_color(i, n):
    """序列 i/n → 青(0,1,1)→黄(1,1,0) 渐变（都和红色焊缝区分）。"""
    t = 0.0 if n <= 1 else i / (n - 1)
    return [t, 1.0, 1.0 - t]


def main():
    seam_pkl = args.seam_pkl
    if not os.path.isfile(seam_pkl):
        raise FileNotFoundError(f"seam pkl 不存在: {seam_pkl}")
    usd_path = args.usd or find_usd(seam_pkl)
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"工件 USD 不存在: {usd_path}")
    if not os.path.isfile(args.save):
        raise FileNotFoundError(f"goal poses pkl 不存在: {args.save}")

    with open(seam_pkl, "rb") as f:
        seam_data = pickle.load(f)
    seam_pts = np.asarray(seam_data["seam_line"], float)             # (N,3) world
    piece_pose = np.asarray(seam_data["piece_pose"][0], float)       # [x,y,z,qw,qx,qy,qz]

    with open(args.save, "rb") as f:
        results = pickle.load(f)
    if not results:
        print("[viz] save 里没有任何结果，退出。")
        simulation_app.close()
        return
    ri = max(0, min(args.result_index, len(results) - 1))
    cam_pose = np.asarray(results[ri]["cam_pose"], float)            # (K, B, 7)
    K, B = cam_pose.shape[:2]
    vi = max(0, min(args.variant, K - 1))
    seq_cam = cam_pose[vi]                                           # (B, 7) piece 系 wxyz

    near, far = fov_corners(args.fov_len)
    poses_w = [compose_pose(piece_pose, seq_cam[i]) for i in range(B)]  # [(pos(3,), R(3,3))]

    print(f"[viz] seam pkl : {seam_pkl}")
    print(f"[viz] 工件 USD : {usd_path}")
    print(f"[viz] goal pkl : {args.save}（result#{ri}/{len(results)}, 变体#{vi}/{K}）")
    print(f"[viz] 焊缝点数 {len(seam_pts)}（红线）；观测相机 {B} 个（青→黄渐变，每个含真实 USD 相机）")

    world = World(stage_units_in_meters=1.0)
    zmin = min(float(seam_pts[:, 2].min()), min(float(p[2]) for p, _ in poses_w))
    world.scene.add_default_ground_plane(z_position=zmin - 0.3)

    spawn_workpiece("/World/workpiece", usd_path, piece_pose)
    spawn_seam_line("/World/seam", seam_pts, color=[1.0, 0.0, 0.0])
    real_cam_paths = []
    for i, (pos_w, R_w) in enumerate(poses_w):
        spawn_camera(f"/World/cam/c{i}", i, pos_w, R_w, near, far, cam_color(i, B))
        real_cam_paths.append(spawn_real_camera(f"/World/real_cam/c{i}", pos_w, R_w, near))
    ci = max(0, min(args.cam_index, B - 1))             # 第 2 个视口取哪个真实相机

    world.reset()

    # 视口对准焊缝中点
    from omni.isaac.core.utils.viewports import set_camera_view
    tgt = seam_pts.mean(axis=0)
    set_camera_view(eye=[tgt[0] + 1.2, tgt[1] + 1.2, tgt[2] + 0.8], target=list(tgt))

    if args.headless:
        for _ in range(3):
            world.step(render=False)
        print(f"已 spawn 工件 + 焊缝红线 + {B} 个观测相机（含 {B} 个真实 USD 相机）。")
        print("VIZ_GOAL_POSES_DONE")
        simulation_app.close()
        return

    try:
        from omni.kit.viewport.menubar.lighting.actions import _set_lighting_mode
        _set_lighting_mode("Grey Studio")
    except Exception:
        pass

    # 第 2 个视口窗口：用第 ci 个观测位姿的真实相机出图（实拍视角）
    try:
        from omni.kit.viewport.utility import create_viewport_window
        vp_win = create_viewport_window(f"RealCam #{ci}", width=640, height=480)
        vp_win.viewport_api.set_active_camera(real_cam_paths[ci])
        print(f"[viz] 第 2 个视口 = 真实相机 #{ci} → {real_cam_paths[ci]}")
    except Exception as e:
        print(f"[viz] 第 2 个视口创建失败（忽略，仅主视口）：{e}")

    print(f"播放中（关闭窗口结束）。红线=目标焊缝；{B} 个视锥=各观测位姿的相机，+z 朝焊缝；"
          f"第 2 个视口=真实相机 #{ci} 实拍。")
    while simulation_app.is_running():
        world.step(render=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
