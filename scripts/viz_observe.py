"""Step 4 真实场景可视化：3 个不同位姿(xyz+yaw 各异、均观测工件)的机械臂 + 工件，
区分【被观测到】(红色 OCCUPIED 贴在工件表面) 与【未观测到】(裸露的灰色 mesh)。

做法：绕工件质心、不同方位角/俯仰/距离生成"看向工件"的相机位姿 → 反推末端位姿 → IK
求可达且确实看到工件的关节角；3 个位姿各 observe_and_update 累积进同一 voxmap；渲染。

运行：
  conda run -n env_isaaclab python scripts/viz_observe.py --mode save --out /tmp/observe
  conda run -n env_isaaclab python scripts/viz_observe.py --mode show     # 本机有显示器
"""
import argparse
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_SEAM = ("/media/a/新加卷/hanfeng/segment_sub_output/"
                "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_0.pkl")

ARM_COLORS = [[0.20, 0.45, 0.95], [0.95, 0.62, 0.10], [0.60, 0.20, 0.80]]


def look_at_T(cam_pos, target, world_up=(0, 0, 1)):
    """OpenCV 光学帧 look-at：+Z 指向 target、+Y 朝下。返回 4x4 (base 系相机位姿)。"""
    z = np.asarray(target, float) - np.asarray(cam_pos, float)
    z /= np.linalg.norm(z)
    up = np.asarray(world_up, float)
    if abs(np.dot(up, z)) > 0.95:                 # 近平行 → 换参考
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)   # 右
    y = np.cross(z, x)                            # 下
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = cam_pos
    return T


def mesh_pose_base(robot_pose, piece_pose):
    """工件相对 base 的 pose [x,y,z,qw,qx,qy,qz]（= inv(T_w_robot)∘T_w_piece）。"""
    import torch
    from curobo.types.math import Pose

    def _p(p7):
        return Pose(position=torch.tensor([p7[:3]], dtype=torch.float32, device="cuda"),
                    quaternion=torch.tensor([p7[3:7]], dtype=torch.float32, device="cuda"))
    T = _p(robot_pose).inverse().multiply(_p(piece_pose))
    return T.get_pose_vector()[0].detach().cpu().numpy().tolist()


def fk_spheres(handle, q):
    import torch
    st = handle.mg.kinematics.get_state(torch.tensor([list(q)], dtype=torch.float32, device="cuda"))
    return st.link_spheres_tensor[0].detach().cpu().numpy()   # (N,4) xyz+r, base 系


def frustum_lineset(Tcam, cm, color, z_far=0.8):
    """相机视野锥线框（apex + 4 远角），base 系。z_far=锥体可视深度(米)。"""
    import open3d as o3d
    fx, fy, cx, cy = cm["fx"], cm["fy"], cm["cx"], cm["cy"]
    W, H = cm["width"], cm["height"]
    pts = [np.zeros(3)]
    for u, v in [(0, 0), (W, 0), (W, H), (0, H)]:
        pts.append([(u - cx) / fx * z_far, (v - cy) / fy * z_far, z_far])  # OpenCV +Z 朝前
    pts = (Tcam @ np.c_[np.array(pts), np.ones(5)].T).T[:, :3]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts),
                              o3d.utility.Vector2iVector(lines))
    ls.paint_uniform_color(color)
    return ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", default=DEFAULT_SEAM)
    ap.add_argument("--mode", choices=["show", "save"], default="save")
    ap.add_argument("--out", default="/tmp/observe")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--draw_free", action="store_true", help="同时画绿色 FREE 体素")
    ap.add_argument("--seed", type=int, default=0, help="选不同的一组分开位姿")
    args = ap.parse_args()

    from gt_gen.config import load_config
    from gt_gen import curobo_iface as ci
    from gt_gen import sensor
    from gt_gen.mapping import observe_and_update
    from gt_gen.voxmap import ThreeStateVoxelMap, OCCUPIED, FREE
    from curobo.geom.types import WorldConfig, Mesh
    from curobo.geom.sdf.world import CollisionCheckerType
    from scipy.spatial.transform import Rotation as Rsp

    cfg = load_config()
    cm = sensor.load_camera_model(cfg)
    d = pickle.load(open(args.seam, "rb"))
    robot_pose = np.asarray(d["robot_pose"][0], float)
    piece_pose = np.asarray(d["piece_pose"][0], float)

    # 工件 obj → base 系
    obj_path = None
    import glob
    for pat in ("*_watertight.obj", "*.obj"):
        hits = sorted(glob.glob(os.path.join(os.path.dirname(args.seam), pat)))
        if hits:
            obj_path = hits[0]; break
    mp = mesh_pose_base(robot_pose, piece_pose)
    scene = sensor.load_truth_scene(obj_path, mesh_pose=mp)
    bb_min, bb_max = scene.bounds
    print("工件 obj:", obj_path)
    print("工件 bbox(base):", np.round(bb_min, 2), np.round(bb_max, 2))

    # 焊缝中点(base) = 实际作业/观测目标（工件质心对长梁太高太远，不可达）
    Rwr = Rsp.from_quat(np.r_[robot_pose[4:7], robot_pose[3]]).as_matrix()
    line = np.asarray(d["seam_line"], float)
    mid = int(d.get("middle", len(line) // 2))
    target = Rwr.T @ (line[mid] - robot_pose[:3])
    print("焊缝中点(base):", np.round(target, 3))

    # cuRobo handle（工件当障碍，焊枪排除；用于 FK；含 IK 备用）
    world = WorldConfig(mesh=[Mesh(name="workpiece", file_path=obj_path, pose=mp)])
    print("\n初始化 cuRobo（含工件 MESH）...")
    h = ci.init_curobo(cfg, world_model=world, collision_checker_type=CollisionCheckerType.MESH,
                       drop_collision_links=["xiaoyu_accessory_link"])

    # 大范围扰动 base-yaw 等 → 让三臂在空间上完全分开（仅验证观测，不要求可达焊缝）
    base_q = np.asarray(d["joint_angles"], float)
    print("\n搜索观测到工件的位姿（大范围扰动使三臂分开）...")
    cands = []
    for dj0 in np.linspace(-1.0, 1.0, 11):        # base yaw 大范围 → 臂扫开
        for dj1 in (-0.35, 0.0, 0.35):            # 肩 → 前后/高低
            for dj4 in (-0.25, 0.25):             # 腕 → 视线
                q = base_q.copy(); q[0] += dj0; q[1] += dj1; q[4] += dj4
                Tcam = sensor.camera_pose_from_config(h, q, cm)
                _, occ = sensor.raycast_observe(Tcam, cm, scene, cm["max_depth"], pixel_stride=24)
                if occ.shape[0] >= 15:            # 确实看到工件
                    # 用相机(工作端)位置做分散度量：固定底座下半截必然重叠，按相机端散开才有意义
                    cands.append({"cen": Tcam[:3, 3].copy(), "q": q, "Tcam": Tcam,
                                  "occ": int(occ.shape[0])})
    print(f"  命中候选 {len(cands)} 个")
    if len(cands) < 3:
        print("候选不足 3 个，放宽阈值/换 seam 重试"); return

    # 最远点采样：选相机端两两离得最开的 3 个；--seed 选不同起点 → 不同的一组关节角
    pos = np.array([c["cen"] for c in cands])
    rank = np.argsort(-np.linalg.norm(pos - pos.mean(0), axis=1))   # 离质心由远到近
    idx = [int(rank[args.seed % len(cands)])]
    while len(idx) < 3:
        dmin = np.min(np.linalg.norm(pos[:, None, :] - pos[idx], axis=2), axis=1)
        dmin[idx] = -1.0
        idx.append(int(np.argmax(dmin)))
    chosen = [cands[i] for i in idx]
    d01 = np.linalg.norm(chosen[0]["cen"] - chosen[1]["cen"])
    d02 = np.linalg.norm(chosen[0]["cen"] - chosen[2]["cen"])
    d12 = np.linalg.norm(chosen[1]["cen"] - chosen[2]["cen"])
    for i, c in enumerate(chosen):
        print(f"  位姿{i}: 相机端={np.round(c['cen'],3)} occ={c['occ']}")
    print(f"  相机端间距: {d01:.2f} / {d02:.2f} / {d12:.2f} m（固定底座，下半截必然重叠）")

    # voxmap 只覆盖焊缝周边；3 位姿累积观测。每个位姿的命中点单独留存(按相机颜色绘制)
    from gt_gen.mapping import commit_observation
    half = 0.9
    vm = ThreeStateVoxelMap(origin=target - half, size_xyz=(2 * half,) * 3, voxel_size=cfg.voxel_size_m)
    print("\nvoxmap:", vm.shape, " 累积观测 3 个位姿 ...")
    zfars = []
    for c in chosen:
        free, occ = sensor.raycast_observe(c["Tcam"], cm, scene, cm["max_depth"], pixel_stride=args.stride)
        r = commit_observation(vm, free, occ)
        c["occ_pts"] = occ
        cp = c["Tcam"][:3, 3]
        zfars.append(float(np.median(np.linalg.norm(occ - cp, axis=1))) if occ.shape[0] else 0.8)
        print("  写入:", r)
    cc = vm.counts()
    print("counts:", {["UNKNOWN", "FREE", "OCCUPIED"][k]: v for k, v in cc.items()})

    # ---- 渲染 ----
    import open3d as o3d
    geoms = []
    # 工件 mesh（浅灰，未观测处即裸露此色）
    work = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(scene.vertices)),
                                     o3d.utility.Vector3iVector(np.asarray(scene.faces)))
    work.paint_uniform_color([0.78, 0.78, 0.78]); work.compute_vertex_normals()
    geoms.append(("work", work, "defaultLit"))
    if args.draw_free:
        fp = vm.state_centers(FREE)
        if fp.shape[0] > 40000:
            fp = fp[np.linspace(0, fp.shape[0] - 1, 40000).astype(int)]
        pcf = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fp))
        pcf.paint_uniform_color([0.55, 0.85, 0.6])
        geoms.append(("free", pcf, "defaultUnlit"))
    # 3 个位姿：机械臂碰撞球 + 视野锥 + 被该相机观测到的表面点，三者同色
    for i, c in enumerate(chosen):
        op = c["occ_pts"]
        if op.shape[0]:
            pco = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(op))
            pco.paint_uniform_color(ARM_COLORS[i])
            geoms.append((f"occ{i}", pco, "defaultUnlit"))
            # 命中点 ↔ 相机光心 连线（同色，下采样避免糊）
            cam_c = c["Tcam"][:3, 3]
            sub = op if op.shape[0] <= 60 else op[np.linspace(0, op.shape[0] - 1, 60).astype(int)]
            pts = np.vstack([cam_c, sub])
            lines = [[0, k + 1] for k in range(sub.shape[0])]
            ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts),
                                      o3d.utility.Vector2iVector(lines))
            ls.paint_uniform_color(ARM_COLORS[i])
            geoms.append((f"rays{i}", ls, "unlitLine"))
        sph = fk_spheres(h, c["q"])
        merged = o3d.geometry.TriangleMesh()
        for s in sph:
            r = float(s[3])
            if r <= 1e-4:
                continue
            m = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=6); m.translate(s[:3])
            merged += m
        merged.paint_uniform_color(ARM_COLORS[i]); merged.compute_vertex_normals()
        geoms.append((f"arm{i}", merged, "defaultLit"))
        geoms.append((f"frust{i}", frustum_lineset(c["Tcam"], cm, ARM_COLORS[i], z_far=zfars[i]),
                      "unlitLine"))
    geoms.append(("base", o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3), "defaultLit"))

    if args.mode == "save":
        import open3d.visualization.rendering as rendering
        os.makedirs(args.out, exist_ok=True)
        W, H = 1400, 1000
        rd = rendering.OffscreenRenderer(W, H)
        rd.scene.set_background([1, 1, 1, 1])
        for name, g, shader in geoms:
            mat = rendering.MaterialRecord(); mat.shader = shader
            mat.point_size = 11.0 if name.startswith("occ") else 6.0
            mat.line_width = 4.0
            rd.scene.add_geometry(name, g, mat)
        ctr = target
        for vname, eye in [("v0", ctr + np.array([1.3, -1.3, 1.0])),
                           ("v1", ctr + np.array([-1.3, -1.3, 0.9])),
                           ("v2", ctr + np.array([0.05, -1.8, 0.5]))]:
            rd.setup_camera(60.0, ctr, eye, np.array([0, 0, 1.0]))
            p = os.path.join(args.out, f"observe_{vname}.png")
            o3d.io.write_image(p, rd.render_to_image()); print("saved", p)
        print("SAVE_OK")
    else:
        o3d.visualization.draw_geometries([g for _, g, _ in geoms],
                                          window_name="observe: 3 poses + workpiece",
                                          width=1400, height=1000)


if __name__ == "__main__":
    main()
