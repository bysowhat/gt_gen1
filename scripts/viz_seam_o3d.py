"""Open3D 离屏渲染：把规划好的轨迹（/tmp/seam_traj.npz）逐帧画出来——
机械臂碰撞球 + 工件 mesh（基座系）。用于快速确认轨迹合理。

运行：conda run -n env_isaaclab python scripts/viz_seam_o3d.py --traj /tmp/seam_traj.npz --out /tmp/seamviz
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="/tmp/seam_traj.npz")
    ap.add_argument("--out", default="/tmp/seamviz")
    ap.add_argument("--frames", type=int, default=4)
    args = ap.parse_args()

    from gt_gen import compat  # noqa: F401
    from gt_gen.config import load_config
    import open3d as o3d
    import open3d.visualization.rendering as rendering
    import trimesh
    from viz_arm_camera import build_model, fk  # 复用 Step-1 验证里的 FK/球提取

    data = np.load(args.traj, allow_pickle=True)
    positions = data["positions"]            # (T,6)
    robot_pos = np.asarray(data["robot_pose"], dtype=float)[:3]
    obj_path = str(data["obj_path"])
    print("轨迹点数:", positions.shape, " obj:", obj_path)

    cfg = load_config()
    model, ta = build_model(cfg)

    # 工件 mesh -> 基座系（减 robot_pos）
    tm = trimesh.load(obj_path, force="mesh")
    tm.apply_translation(-robot_pos)
    work = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(tm.vertices)),
        o3d.utility.Vector3iVector(np.asarray(tm.faces)),
    )
    work.paint_uniform_color([0.6, 0.6, 0.6])
    work.compute_vertex_normals()
    wctr = np.asarray(tm.vertices).mean(axis=0)

    os.makedirs(args.out, exist_ok=True)
    W, H = 1280, 960
    renderer = rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background([1, 1, 1, 1])

    idxs = np.linspace(0, len(positions) - 1, args.frames).astype(int)
    for fi, ti in enumerate(idxs):
        renderer.scene.clear_geometry()
        # 工件
        mw = rendering.MaterialRecord(); mw.shader = "defaultLit"
        renderer.scene.add_geometry("work", work, mw)
        # 机械臂碰撞球
        spheres, _, _ = fk(model, ta, positions[ti].tolist())
        merged = o3d.geometry.TriangleMesh()
        for s in spheres:
            r = float(s[3])
            if r <= 1e-4:
                continue
            m = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=6)
            m.translate(s[:3]); merged += m
        merged.paint_uniform_color([0.2, 0.55, 0.95]); merged.compute_vertex_normals()
        ma = rendering.MaterialRecord(); ma.shader = "defaultLit"
        renderer.scene.add_geometry("arm", merged, ma)
        # 基座坐标轴
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        mf = rendering.MaterialRecord(); mf.shader = "defaultLit"
        renderer.scene.add_geometry("frame", frame, mf)
        # 取景
        ctr = (wctr + spheres[spheres[:, 3] > 1e-4][:, :3].mean(0)) / 2
        eye = ctr + np.array([2.2, -2.2, 1.2])
        renderer.setup_camera(55.0, ctr, eye, np.array([0, 0, 1.0]))
        img = renderer.render_to_image()
        p = os.path.join(args.out, f"frame_{fi}_t{ti}.png")
        o3d.io.write_image(p, img)
        print("saved", p)
    print("VIZ_OK")


if __name__ == "__main__":
    main()
