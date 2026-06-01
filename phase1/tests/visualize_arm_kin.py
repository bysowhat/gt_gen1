"""M1.4 视觉验证: 在 Open3D 里显示机械臂 (碰撞球) + 体素障碍 + 工件 + 焊缝.

合格标准:
    - 机械臂的碰撞球串成 UR12 形状, 末端在工件附近
    - 障碍体素 (红方块) 在工件表面
    - 如果某构型碰撞, 出红色"COLLISION" 提示
    - 修改 --theta 看不同关节角下的姿态

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/visualize_arm_kin.py \
        --obj "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part.obj" \
        --pkl "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_0.pkl"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import open3d as o3d
import torch

from phase1.arm_kin import ArmKin
from phase1.depth_source import CameraIntrinsics, RaycastSim
from phase1.mapping import STATE_OCCUPIED, VoxelMap
from phase1.tests._viz_helpers import load_seam_data, look_at, make_seam_geoms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--robot-yml",
        default="/home/a/Datas/curobo/example_new_robot/urdf_12e/ur12e.yml")
    parser.add_argument("--theta", type=float, nargs=6, default=None,
                        help="6 关节角. 默认用 pkl 里 joint_angles.")
    parser.add_argument("--build-map", action="store_true",
                        help="先用 RaycastSim 建一遍 voxel map (= M1.2 流程)")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    # ---- 加载 ----
    seam = load_seam_data(args.pkl)
    if args.theta is not None:
        theta_arr = np.array(args.theta)
    else:
        theta_arr = seam.joint_angles
    print(f"[seam] {seam.label}")
    print(f"[theta] {theta_arr.round(3).tolist()}")

    kin = ArmKin(robot_yml_path=args.robot_yml, robot_pose_world=seam.robot_pose)
    print(f"[arm] dof={kin.dof}, base_in_world={seam.robot_pose[:3, 3].round(3)}")

    # ---- 建 voxel map (可选) ----
    K = CameraIntrinsics.from_fov(86.0, 57.0, 240, 320)
    sim = RaycastSim(mesh_paths=[args.obj], intrinsics=K)
    bounds = np.stack([sim.mesh.bounds[0] - 0.6, sim.mesh.bounds[1] + 0.6])
    vm = VoxelMap(bounds=bounds, resolution=0.03, device="cuda" if torch.cuda.is_available() else "cpu",
                  max_range=2.0)
    if args.build_map:
        for i in range(8):
            ang = 2 * np.pi * i / 8
            cp = seam.p_weld + np.array([0.5 * np.cos(ang), 0.5 * np.sin(ang), 0.3])
            d = sim.render(look_at(cp, seam.p_weld))
            vm.integrate(d, K, look_at(cp, seam.p_weld))
        n_obs = kin.update_world(vm)
        counts = vm.num_voxels_by_state()
        print(f"[map] occupied={counts['occupied']}, fed {n_obs} cuboids to cuRobo")
    else:
        # 空 world
        kin.update_world(vm)
        print("[map] empty (use --build-map to build from 8 views)")

    # ---- FK + collision ----
    theta = torch.tensor(theta_arr, **kin.tensor_args.as_torch_dict())
    fk = kin.fk(theta)
    coll = kin.collides(theta)
    is_coll = bool(coll.item() if coll.dim() == 0 else coll[0].item())
    ee_pos = fk["ee_pos_world"][0].cpu().numpy()
    print(f"[fk] EE world = {ee_pos.round(3)}")
    print(f"[collision] {'⚠ COLLISION' if is_coll else '✓ free'}")

    # ---- 提取机械臂的碰撞球 (in base frame) ----
    state = kin.kin.get_state(theta.unsqueeze(0))
    spheres = state.link_spheres_tensor[0].cpu().numpy()  # (n_spheres, 4) [x,y,z,r]
    # base → world
    R_wb = seam.robot_pose[:3, :3]
    t_wb = seam.robot_pose[:3, 3]
    centers_world = (R_wb @ spheres[:, :3].T).T + t_wb
    radii = spheres[:, 3]
    print(f"[arm_spheres] {len(centers_world)} 个 collision spheres")

    # ---- Open3D ----
    geoms = []

    # 工件灰
    mesh = o3d.io.read_triangle_mesh(args.obj)
    mesh.compute_vertex_normals(); mesh.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.append(mesh)

    # 占据体素 (红立方体)
    state_cpu = vm.state.cpu().numpy()
    occ_idx = np.argwhere(state_cpu == STATE_OCCUPIED)
    if len(occ_idx) > 0:
        occ_pts = bounds[0] + (occ_idx + 0.5) * vm.res
        pcd_occ = o3d.geometry.PointCloud()
        pcd_occ.points = o3d.utility.Vector3dVector(occ_pts)
        pcd_occ.paint_uniform_color([1.0, 0.1, 0.1])
        geoms.append(pcd_occ)

    # 机械臂球 (碰撞用 = 红, 不碰撞 = 蓝)
    arm_color = (1.0, 0.2, 0.2) if is_coll else (0.2, 0.5, 1.0)
    for c, r in zip(centers_world, radii):
        if r <= 0:
            continue
        s = o3d.geometry.TriangleMesh.create_sphere(radius=float(r), resolution=10)
        s.translate(c); s.compute_vertex_normals()
        s.paint_uniform_color(list(arm_color))
        geoms.append(s)

    # 焊缝
    geoms.extend(make_seam_geoms(seam))

    # 机械臂底座坐标轴
    axes_base = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=seam.robot_pose[:3, 3])
    geoms.append(axes_base)

    # EE 位置
    ee_sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=14)
    ee_sph.translate(ee_pos); ee_sph.compute_vertex_normals()
    ee_sph.paint_uniform_color([0.0, 1.0, 1.0])  # 青色 = EE
    geoms.append(ee_sph)

    print()
    print("Legend:")
    print("  灰 mesh        — 工件")
    print("  品红圆柱       — 焊缝")
    print(f"  机械臂球       — {'红 = 碰撞' if is_coll else '蓝 = free'}")
    print("  青球           — EE 末端")
    print("  红方块点云     — voxel map 中的 occupied (障碍)")
    print("  RGB 轴         — 机械臂底座坐标系")

    title = (f"M1.4 ArmKin  |  {seam.label}  |  "
             f"{'⚠ 碰撞' if is_coll else '✓ free'}  |  "
             f"{len(occ_idx)} obstacles")
    if not args.no_window:
        o3d.visualization.draw_geometries(
            geoms, window_name=title, width=1280, height=720)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
