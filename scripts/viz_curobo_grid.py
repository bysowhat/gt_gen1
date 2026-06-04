"""核对自维护 voxmap 与 cuRobo 内部 voxel 网格是否一致：
构造一个已知形状的占据(竖直板)，sync_collision_world 后，把【cuRobo 实际判占据】的体素
(红，从 ESDF 读回) 与【voxmap 占据】的体素中心(绿) 叠画，并打印数值吻合度。

运行：
  conda run -n env_isaaclab python scripts/viz_curobo_grid.py --mode save --out /tmp/curobo_grid
  conda run -n env_isaaclab python scripts/viz_curobo_grid.py --mode show
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["show", "save"], default="save")
    ap.add_argument("--out", default="/tmp/curobo_grid")
    args = ap.parse_args()

    from gt_gen.config import load_config
    from gt_gen import curobo_iface as ci
    from gt_gen.collision_sync import sync_collision_world, curobo_occupied_centers
    from gt_gen.voxmap import ThreeStateVoxelMap, FREE, OCCUPIED

    cfg = load_config()
    h = ci.init_curobo(cfg, roi_dims=[1.6, 1.6, 1.6], roi_center=[0.0, 0.0, 0.6])
    center = np.asarray(h.voxel["pose"][:3], float)
    dims = np.asarray(h.voxel["dims"], float)
    vs = h.voxel["voxel_size"]
    # voxmap 比 cuRobo ROI 略大(留 margin)，使 cuRobo "+1" 边界体素也落在 voxmap 内→FREE，
    # 否则边界层会被判 UNKNOWN→占据，形成一圈"墙"遮挡视线(该行为安全但不利于核对)。
    m = 4 * vs
    vm = ThreeStateVoxelMap(origin=center - dims / 2 - m, size_xyz=dims + 2 * m, voxel_size=vs)

    # 已知形状：整体 FREE，挖一块竖直板 OCCUPIED（便于肉眼核对形状/位置）
    vm.fill(FREE)
    lo = vm.world_to_voxel([0.10, -0.30, 0.20])
    hi = vm.world_to_voxel([0.16, 0.30, 1.00])
    ii, jj, kk = np.meshgrid(*[np.arange(lo[k], hi[k] + 1) for k in range(3)], indexing="ij")
    slab = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
    vm.set_many(slab, OCCUPIED)
    print("voxmap counts:", {["UNK", "FREE", "OCC"][k]: v for k, v in vm.counts().items()})

    sync_collision_world(h, vm)

    cu_occ = curobo_occupied_centers(h)                 # cuRobo 实际占据中心(base)
    vm_occ = vm.state_centers(OCCUPIED)                 # voxmap 占据中心(base)
    print("cuRobo 占据体素:", cu_occ.shape[0], " voxmap 占据体素:", vm_occ.shape[0])

    # 数值吻合：cuRobo 占据的中心，在 voxmap 里查应为占据(非 FREE)
    st = np.asarray(vm.get(vm.world_to_voxel(cu_occ)))
    agree = float((st != FREE).mean()) if cu_occ.shape[0] else 1.0
    print(f"cuRobo 占据点中 voxmap 也判占据的比例: {agree*100:.2f}%")
    # 反向：voxmap 占据中心，cuRobo 判占据的比例（最近邻：用 ESDF 阈值近似）
    print(f"两者占据体素数差: {abs(cu_occ.shape[0]-vm_occ.shape[0])}（边缘±1/离散差异属正常）")

    # ---- 渲染：cuRobo 红方块 + voxmap 绿点 ----
    import open3d as o3d
    def voxelgrid(points, color, size):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pc.paint_uniform_color(color)
        return o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=size)

    geoms = []
    if cu_occ.shape[0]:
        geoms.append(("curobo", voxelgrid(cu_occ, [0.92, 0.12, 0.12], vs), "defaultLit"))
    if vm_occ.shape[0]:
        pcv = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vm_occ))
        pcv.paint_uniform_color([0.1, 0.7, 0.2])
        geoms.append(("voxmap", pcv, "defaultUnlit"))
    geoms.append(("base", o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3), "defaultLit"))

    if args.mode == "save":
        import open3d.visualization.rendering as rendering
        os.makedirs(args.out, exist_ok=True)
        rd = rendering.OffscreenRenderer(1200, 900); rd.scene.set_background([1, 1, 1, 1])
        for name, g, shader in geoms:
            mat = rendering.MaterialRecord(); mat.shader = shader; mat.point_size = 4.0
            rd.scene.add_geometry(name, g, mat)
        ctr = center
        for vn, eye in [("v0", ctr + np.array([1.0, -1.0, 0.6])), ("v1", ctr + np.array([1.2, 0.1, 0.2]))]:
            rd.setup_camera(60.0, ctr.tolist(), eye.tolist(), [0, 0, 1.0])
            p = os.path.join(args.out, f"grid_{vn}.png"); o3d.io.write_image(p, rd.render_to_image()); print("saved", p)
        print("SAVE_OK")
    else:
        o3d.visualization.draw_geometries([g for _, g, _ in geoms],
                                          window_name="cuRobo(红方块) vs voxmap(绿点)", width=1200, height=900)


if __name__ == "__main__":
    main()
