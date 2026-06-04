"""Step 6 可视化验证（open3d）：给定一段轨迹(npz)，画出
  ① 工件 mesh(灰)  ② 机械臂初始位姿的碰撞球(绿)  ③ 整段轨迹整臂【横扫后的体素占用】(橙,半透明实心)。
看清"机械臂扫过哪些格子"——这正是 motion_stays_in_free 判定所用的扫掠体积。

运行：
  conda run -n env_isaaclab python scripts/viz_swept_o3d.py --traj /tmp/seam_traj.npz --mode save --out /tmp/swept
  conda run -n env_isaaclab python scripts/viz_swept_o3d.py --traj /tmp/seam_traj.npz --mode show
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CELL_COLOR = [0.95, 0.45, 0.10]     # 橙：扫掠占用格
ARM_COLOR = [0.20, 0.72, 0.32]      # 绿：初始碰撞球


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="/tmp/seam_traj.npz")
    ap.add_argument("--mode", choices=["show", "save"], default="save")
    ap.add_argument("--out", default="/tmp/swept")
    ap.add_argument("--stride", type=int, default=1, help="每多少步取1步计算横扫占用（--index 未给时用）")
    ap.add_argument("--index", default=None,
                    help="指定帧：逗号分隔与切片，如 '0,20,40' 或 '0:100:10'（负索引可用）。给了则忽略 stride")
    args = ap.parse_args()

    from gt_gen.config import load_config
    from gt_gen import curobo_iface as ci
    from gt_gen import sensor
    from gt_gen.voxmap import build_roi_voxmap
    from gt_gen.swept import fk_spheres_batch, voxelize_spheres

    data = np.load(args.traj, allow_pickle=True)
    positions = data["positions"]                       # (T,6)
    obj_path = str(data["obj_path"])
    piece_pose = np.asarray(data["piece_pose_to_robot"], float).tolist()
    print("轨迹点数:", positions.shape, " obj:", os.path.basename(obj_path))

    cfg = load_config()
    h = ci.init_curobo(cfg)
    vm = build_roi_voxmap(cfg)

    # 工件 → base 系
    scene = sensor.load_truth_scene(obj_path, mesh_pose=piece_pose)

    # 选帧：--index 优先（逗号/切片），否则按 stride；都含末点
    n = len(positions)
    if args.index:
        sel = set()
        for tok in args.index.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if ":" in tok:
                p = (tok.split(":") + ["", "", ""])[:3]
                a = int(p[0]) if p[0] else 0
                b = int(p[1]) if p[1] else n
                st = int(p[2]) if p[2] else 1
                sel.update(range(a, b, st))
            else:
                sel.add(int(tok))
        sel = sorted({(i % n) for i in sel if -n <= i < n})
    else:
        sel = sorted(set(range(0, n, max(1, args.stride))) | {n - 1})
    pos_used = positions[sel]
    spheres_all = fk_spheres_batch(h, pos_used).reshape(-1, 4)
    cells_idx = voxelize_spheres(vm, spheres_all)
    cell_ctrs = vm.voxel_to_world(cells_idx)
    # 初始位姿碰撞球
    sph0 = fk_spheres_batch(h, positions[:1])[0]
    print(f"选 {len(pos_used)}/{n} 帧 {sel if len(sel)<=20 else str(sel[:20])+'...'} → "
          f"横扫占用 {cell_ctrs.shape[0]} 格  | 初始碰撞球 {int((sph0[:,3]>1e-4).sum())} 个")

    import open3d as o3d
    vs = vm.voxel_size

    work = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(scene.vertices)),
                                     o3d.utility.Vector3iVector(np.asarray(scene.faces)))
    work.paint_uniform_color([0.72, 0.72, 0.72]); work.compute_vertex_normals()

    # 横扫占用格：半透明实心立方
    cells_mesh = o3d.geometry.TriangleMesh()
    for c in cell_ctrs:
        b = o3d.geometry.TriangleMesh.create_box(vs, vs, vs); b.translate(c - vs / 2); cells_mesh += b
    if len(cells_mesh.vertices):
        cells_mesh.paint_uniform_color(CELL_COLOR); cells_mesh.compute_vertex_normals()

    # 初始碰撞球
    arm0 = o3d.geometry.TriangleMesh()
    for s in sph0:
        r = float(s[3])
        if r <= 1e-4:
            continue
        m = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=8); m.translate(s[:3]); arm0 += m
    arm0.paint_uniform_color(ARM_COLOR); arm0.compute_vertex_normals()

    geoms = [("work", work, "lit", None),
             ("swept", cells_mesh, "fill", CELL_COLOR + [0.45]),
             ("arm0", arm0, "lit", None),
             ("base", o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3), "lit", None)]

    ctr = cell_ctrs.mean(0) if cell_ctrs.shape[0] else sph0[:, :3].mean(0)

    if args.mode == "save":
        import open3d.visualization.rendering as rendering
        os.makedirs(args.out, exist_ok=True)
        rd = rendering.OffscreenRenderer(1400, 1000); rd.scene.set_background([1, 1, 1, 1])
        for name, g, kind, rgba in geoms:
            mat = rendering.MaterialRecord()
            if kind == "fill":
                mat.shader = "defaultLitTransparency"; mat.base_color = rgba
            else:
                mat.shader = "defaultLit"
            rd.scene.add_geometry(name, g, mat)
        for vn, eye in [("v0", ctr + np.array([1.4, -1.4, 1.0])),
                        ("v1", ctr + np.array([0.05, -1.9, 0.7]))]:
            rd.setup_camera(55.0, ctr.tolist(), eye.tolist(), [0, 0, 1.0])
            p = os.path.join(args.out, f"swept_{vn}.png"); o3d.io.write_image(p, rd.render_to_image()); print("saved", p)
        print("SAVE_OK")
    else:
        o3d.visualization.draw_geometries([g for _, g, _, _ in geoms],
                                          window_name="工件(灰)+初始臂(绿)+横扫占用格(橙)",
                                          width=1400, height=1000)


if __name__ == "__main__":
    main()
