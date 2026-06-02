"""可视化：机械臂碰撞球 + 相机视野锥（多关节角），用于确认相机外参/朝向。

相机帧约定（依据参考 env/ik_cam.py）：
    T_base_cam = T_base_Link6 @ T(cam_pos, cam_quat_wxyz)
    光学帧 OpenCV：+Z 朝前（视线）、+X 右、+Y 下；内参 fx,fy,cx,cy。

用法：
    # 交互查看（在你本机有显示器时，逐个关节角弹窗，关掉看下一个）
    conda run -n env_isaaclab python scripts/viz_arm_camera.py --mode show
    # 离屏渲染存图（无显示器/给我看）
    conda run -n env_isaaclab python scripts/viz_arm_camera.py --mode save --out /tmp/armcam
    # 仅自检（构建几何、不渲染）
    conda run -n env_isaaclab python scripts/viz_arm_camera.py --mode check
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def quat_wxyz_to_R(q):
    w, x, y, z = [float(v) for v in q]
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def pose_to_T(pos, quat_wxyz):
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_R(quat_wxyz)
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def build_model(cfg):
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.util_file import load_yaml

    d = load_yaml(cfg.robot_cfg_path)
    d["robot_cfg"]["kinematics"]["link_names"] = ["Link6"]  # 让 FK 返回 Link6 位姿
    ta = TensorDeviceType()
    rc = RobotConfig.from_dict(d["robot_cfg"], ta)
    return CudaRobotModel(rc.kinematics), ta


def fk(model, ta, q):
    import torch
    qt = torch.tensor([q], dtype=torch.float32, device=ta.device)
    st = model.get_state(qt)
    spheres = st.link_spheres_tensor[0].detach().cpu().numpy()  # (N,4) xyz+r (base frame)
    l6 = st.link_pose["Link6"]
    l6_pos = l6.position[0].detach().cpu().numpy()
    l6_quat = l6.quaternion[0].detach().cpu().numpy()           # wxyz (cuRobo)
    return spheres, l6_pos, l6_quat


def camera_frustum_lines(T_base_cam, cam, z_far=0.6):
    fx, fy = cam["intrinsics"]["fx"], cam["intrinsics"]["fy"]
    cx, cy = cam["intrinsics"]["cx"], cam["intrinsics"]["cy"]
    W, H = cam["width"], cam["height"]
    corners_px = [(0, 0), (W, 0), (W, H), (0, H)]
    pts = [np.array([0.0, 0.0, 0.0])]  # apex
    for (u, v) in corners_px:
        x = (u - cx) / fx * z_far
        y = (v - cy) / fy * z_far
        pts.append(np.array([x, y, z_far]))         # OpenCV: +Z forward
    pts = np.array(pts)
    pts_h = (T_base_cam @ np.hstack([pts, np.ones((5, 1))]).T).T[:, :3]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    return pts_h, lines


def build_scene(model, ta, cfg, q):
    import open3d as o3d
    cam = cfg.camera
    spheres, l6_pos, l6_quat = fk(model, ta, q)
    T_base_l6 = pose_to_T(l6_pos, l6_quat)
    T_l6_cam = pose_to_T(cam["extrinsic_pos"], cam["extrinsic_quat_wxyz"])
    T_base_cam = T_base_l6 @ T_l6_cam

    geoms = []
    # 机械臂碰撞球
    for s in spheres:
        r = float(s[3])
        if r <= 1e-4:
            continue
        m = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=8)
        m.translate(s[:3])
        m.paint_uniform_color([0.3, 0.6, 0.95])
        m.compute_vertex_normals()
        geoms.append(m)
    # 相机视野锥
    pts, lines = camera_frustum_lines(T_base_cam, cam, z_far=0.6)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.paint_uniform_color([0.9, 0.1, 0.1])
    geoms.append(ls)
    # 相机坐标轴（X红/Y绿/Z蓝；Z=视线方向）
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
    cam_frame.transform(T_base_cam)
    geoms.append(cam_frame)
    # 基座坐标轴
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25))
    return geoms, T_base_cam


def default_configs(retract):
    r = list(retract)
    def mod(i, d):
        c = list(r); c[i] += d; return c
    return [
        ("retract", r),
        ("j1+0.6", mod(0, 0.6)),
        ("j1-0.6", mod(0, -0.6)),
        ("j5+0.8", mod(4, 0.8)),
        ("j2+0.5,j3-0.5", [r[0], r[1] + 0.5, r[2] - 0.5, r[3], r[4], r[5]]),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["show", "save", "check"], default="show")
    ap.add_argument("--out", default="/tmp/armcam")
    args = ap.parse_args()

    from gt_gen import compat  # noqa: F401  (warp shim, 须在 import curobo 前)
    from gt_gen.config import load_config

    cfg = load_config()
    model, ta = build_model(cfg)
    configs = default_configs(cfg.retract_config)

    if args.mode == "check":
        import open3d as o3d  # noqa: F401
        for name, q in configs:
            geoms, T = build_scene(model, ta, cfg, q)
            campos = T[:3, 3]
            zfwd = T[:3, 2]  # 相机视线方向(+Z)
            print(f"[{name}] 几何数={len(geoms)} 相机位置={campos.round(3)} 视线(+Z)={zfwd.round(3)}")
        print("CHECK_OK")
        return

    if args.mode == "save":
        import open3d as o3d
        import open3d.visualization.rendering as rendering
        os.makedirs(args.out, exist_ok=True)
        W, H = 1280, 960
        renderer = rendering.OffscreenRenderer(W, H)
        renderer.scene.set_background([1, 1, 1, 1])
        for name, q in configs:
            for gid in list(renderer.scene.geometry_names if hasattr(renderer.scene, "geometry_names") else []):
                renderer.scene.remove_geometry(gid)
            renderer.scene.clear_geometry()
            geoms, T = build_scene(model, ta, cfg, q)
            for i, g in enumerate(geoms):
                mat = rendering.MaterialRecord()
                mat.shader = "unlitLine" if isinstance(g, o3d.geometry.LineSet) else "defaultLit"
                mat.line_width = 3.0
                renderer.scene.add_geometry(f"g{i}", g, mat)
            # 取景：看向碰撞球中心
            sph, _, _ = fk(model, ta, q)
            ctr = sph[sph[:, 3] > 1e-4][:, :3].mean(axis=0)
            for vname, eye in [("front", ctr + np.array([1.6, -1.6, 1.0])),
                               ("side",  ctr + np.array([0.05, -2.0, 0.4]))]:
                renderer.setup_camera(60.0, ctr, eye, np.array([0, 0, 1.0]))
                img = renderer.render_to_image()
                p = os.path.join(args.out, f"{name}_{vname}.png")
                o3d.io.write_image(p, img)
                print("saved", p)
        print("SAVE_OK")
        return

    # show: 交互逐个弹窗
    import open3d as o3d
    for name, q in configs:
        geoms, T = build_scene(model, ta, cfg, q)
        print(f"显示 [{name}]，关闭窗口看下一个 …")
        o3d.visualization.draw_geometries(geoms, window_name=f"arm+camera: {name}", width=1280, height=960)


if __name__ == "__main__":
    main()
