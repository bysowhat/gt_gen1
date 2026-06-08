"""Step 8 验证：候选视点生成。参考 scripts/verify_step7.py，每个 verify_* 验证 candidates.py 的一个函数。

被测函数（candidates.py，逐个补充验证）：
  - verify_cluster_centroids       → cluster_centroids（对 B 聚类取簇心 T）✅
  - verify_standoff_poses_looking_at → standoff_poses_looking_at（每个 T 撒相机位姿）  [待写]
  - verify_look_at_pose            → look_at_pose（眼睛 p 对准 T 的相机位姿）        [待写]
  - verify_generate_candidates     → generate_candidates（眼在手 IK + 看向 + 可达过滤）[待写]

共享场景【冷启动】(build_scene)：初始引导 FREE 空间用【圆柱体法】set_initial_free_cylinder
（参数取自 configs/default.yaml 的 init_free 段）罩住 retract 整臂；当前构型 = retract。
真值上规划 P*，沿 P* 向前扫到圆柱边界停下(reach_pt)，取前方阻塞段 B。

运行：conda run -n env_isaaclab python scripts/verify_step8.py [--viz]
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

SEAM = ("/media/a/新加卷/hanfeng/segment_sub_output/"
        "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_22.pkl")

# 各簇配色（每类 1 种颜色）
PALETTE = [[0.90, 0.10, 0.10], [0.10, 0.55, 0.95], [0.95, 0.65, 0.10],
           [0.20, 0.75, 0.30], [0.65, 0.20, 0.85], [0.10, 0.75, 0.75]]


def build_scene(args):
    """真值 handle + P* + 冷启动 voxmap（初始引导 FREE 用圆柱体法，参数取自 default.yaml）。返回上下文 dict。"""
    from gt_gen.config import load_config
    from gt_gen import curobo_iface as ci
    from gt_gen.voxmap import build_roi_voxmap
    from gt_gen.sensor import load_camera_model, load_truth_scene
    from gt_gen.init_free import set_initial_free_cylinder
    from gt_gen.reach_b import compute_reach_pt
    from plan_seam import seam_ee_pose
    from curobo.geom.types import WorldConfig, Mesh
    from curobo.geom.sdf.world import CollisionCheckerType
    import torch
    from curobo.types.math import Pose

    cfg = load_config()
    d = pickle.load(open(args.seam, "rb"))
    robot_pose = np.asarray(d["robot_pose"][0], float)
    piece_pose = np.asarray(d["piece_pose"][0], float)
    goal_pose = seam_ee_pose(d)
    obj = sorted(glob.glob(os.path.dirname(args.seam) + "/*_watertight.obj"))[0]

    def _p(p7):
        return Pose(position=torch.tensor(np.array([p7[:3]]), dtype=torch.float32, device="cuda"),
                    quaternion=torch.tensor(np.array([p7[3:7]]), dtype=torch.float32, device="cuda"))
    mp = _p(robot_pose).inverse().multiply(_p(piece_pose)).get_pose_vector()[0].cpu().numpy().tolist()

    world = WorldConfig(mesh=[Mesh(name="workpiece", file_path=obj, pose=mp)])
    print("初始化真值 cuRobo（含工件 MESH）...")
    h = ci.init_curobo(cfg, world_model=world, collision_checker_type=CollisionCheckerType.MESH)

    metric = ci.free_pose_metric(h, free_rot=(0,))
    P = ci.plan_on_truth(h, cfg.retract_config, goal_pose, max_attempts=20, pose_cost_metric=metric)
    assert P is not None, "真值上 P* 规划失败"
    print(f"P* 路点数: {P.shape}")

    cam = load_camera_model(cfg)
    scene = load_truth_scene(obj, mesh_pose=mp)

    # 冷启动：初始引导 FREE 空间 = 圆柱体法（default.yaml: cyl_radius_m / cyl_height_m），罩住 retract 整臂
    vm = build_roi_voxmap(cfg)
    n = set_initial_free_cylinder(h, vm, config=cfg)
    print(f"初始圆柱 FREE 体素={n}  (R={cfg.init_free_cyl_radius}m h={cfg.init_free_cyl_height}m, 取自 default.yaml)")

    cur_cfg = list(cfg.retract_config)                        # 冷启动当前构型 = retract
    reach_idx = compute_reach_pt(h, vm, P)
    print(f"reach_idx={reach_idx}/{len(P)-1}（从 retract 沿 P* 扫到圆柱边界停下，cur_cfg=retract）")
    return dict(cfg=cfg, h=h, P=P, vm=vm, cam=cam, scene=scene, obj=obj, mp=mp,
                cur_cfg=cur_cfg, reach_idx=reach_idx, goal_pose=goal_pose)


# ============ open3d 可视化小工具（各 verify_* 共用） ============

def _cells_mesh(vm, centers):
    """把一组体素中心画成实心小立方体（合并为一个 mesh）。"""
    import open3d as o3d
    vs = vm.voxel_size
    m = o3d.geometry.TriangleMesh()
    for c in np.asarray(centers):
        b = o3d.geometry.TriangleMesh.create_box(vs, vs, vs); b.translate(c - vs / 2); m += b
    if len(m.vertices):
        m.compute_vertex_normals()
    return m


def _arm_mesh(handle, q):
    """构型 q 的整臂碰撞球（绿）。"""
    import open3d as o3d
    from gt_gen.swept import fk_spheres_batch
    arm = o3d.geometry.TriangleMesh()
    for s in fk_spheres_batch(handle, [q])[0]:
        if float(s[3]) <= 1e-4:
            continue
        b = o3d.geometry.TriangleMesh.create_sphere(radius=float(s[3]), resolution=8)
        b.translate(s[:3]); arm += b
    arm.paint_uniform_color([0.20, 0.72, 0.32]); arm.compute_vertex_normals()
    return arm


def _work_mesh(scene):
    """工件 trimesh → open3d（灰）。"""
    import open3d as o3d
    w = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(scene.vertices)),
                                  o3d.utility.Vector3iVector(np.asarray(scene.faces)))
    w.paint_uniform_color([0.72, 0.72, 0.72]); w.compute_vertex_normals()
    return w


def _draw(geoms, title):
    """geoms: [(name, geometry, kind, rgba)]，kind ∈ {lit, fill(半透明), line}。"""
    import open3d as o3d
    from open3d.visualization import rendering
    items = []
    for name, g, kind, rgba in geoms:
        mat = rendering.MaterialRecord()
        if kind == "fill":
            mat.shader = "defaultLitTransparency"; mat.base_color = rgba
        elif kind == "line":
            mat.shader = "unlitLine"; mat.line_width = 2.0
        else:
            mat.shader = "defaultLit"
        items.append({"name": name, "geometry": g, "material": mat})
    print(f"  打开 open3d 窗口: {title}（关闭后继续）...")
    o3d.visualization.draw(items, title=title, width=1400, height=1000, bg_color=(1.0, 1.0, 1.0, 1.0))


def _roi_and_base(vm):
    """ROI 包围盒线框（灰）+ base 坐标轴。"""
    import open3d as o3d
    aabb = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(vm.origin, vm.upper))
    aabb.paint_uniform_color([0.6, 0.6, 0.6])
    base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    return [("roi", aabb, "line", None), ("base", base, "lit", None)]


def _lines(segs, color):
    """segs: [(p0,p1), ...] → 一个上色 LineSet。"""
    import open3d as o3d
    pts = np.asarray([p for s in segs for p in s], float)
    idx = np.arange(len(pts)).reshape(-1, 2)
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(idx))
    ls.paint_uniform_color(color)
    return ls


def _ball(center, r, color):
    import open3d as o3d
    s = o3d.geometry.TriangleMesh.create_sphere(radius=r); s.translate(np.asarray(center, float))
    s.paint_uniform_color(color); s.compute_vertex_normals()
    return s


def _ortho_basis(axis):
    """返回与 axis 正交的两个单位向量 e1,e2（构成垂直 axis 的平面基）。"""
    a = np.asarray(axis, float); a = a / (np.linalg.norm(a) + 1e-12)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, tmp); e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(a, e1)
    return e1, e2


def _circle(center, axis, r, color, n=72):
    """以 center 为心、在垂直 axis 的平面内、半径 r 的圆环 LineSet（= 半球的边界大圆/赤道）。"""
    import open3d as o3d
    e1, e2 = _ortho_basis(axis)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = center + r * (np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2))
    idx = np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1)
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(idx))
    ls.paint_uniform_color(color)
    return ls


def _hemisphere_surface(center, axis, r, rgba, resolution=20):
    """以 center 为心、半径 r、朝 axis 那一侧的半球曲面（半透明 fill mesh）。"""
    import open3d as o3d
    a = np.asarray(axis, float); a = a / (np.linalg.norm(a) + 1e-12)
    sph = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=resolution)
    V = np.asarray(sph.vertices); Tr = np.asarray(sph.triangles)
    keep = (V[Tr].mean(axis=1) @ a) >= 0.0                     # 只留 axis 那半边的三角面
    sph.triangles = o3d.utility.Vector3iVector(Tr[keep])
    sph.remove_unreferenced_vertices()
    sph.translate(np.asarray(center, float)); sph.compute_vertex_normals()
    return sph


def _fov_frustum(M, cam, depth):
    """由相机位姿 M(4x4) + 内参 + 远面深度 depth，算针孔相机视锥。返回 (棱线 LineSet, 视锥面 mesh)。

    四个像素角 (0,0)(W,0)(W,H)(0,H) 的光学射线 d=[(u-cx)/fx,(v-cy)/fy,1]，
    远角 = p + R·(depth·d)（z=1 → 远面是 z=depth 的矩形）；apex=相机中心 p。
    """
    import open3d as o3d
    R = M[:3, :3]; p = M[:3, 3]
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    W, H = cam["width"], cam["height"]
    far = []
    for u, v in [(0, 0), (W, 0), (W, H), (0, H)]:
        d = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        far.append(p + R @ (depth * d))
    pts = np.vstack([p[None], np.asarray(far)])                # 0=apex, 1..4=远面四角
    edges = np.array([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)])
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(edges))
    ls.paint_uniform_color([0.6, 0.2, 0.85])
    tris = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 2, 3], [1, 3, 4]])
    cone = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(pts), o3d.utility.Vector3iVector(tris))
    cone.compute_vertex_normals()
    return ls, cone


# ============ 每个 verify_* 验证 candidates.py 的一个函数 ============

def verify_cluster_centroids(ctx, viz):
    """验证 cluster_centroids：对阻塞段 B 聚类取簇心 T。

    判据：① 1 ≤ 簇数 ≤ max_clusters(=3)；② 每个簇心落在 B 的包围盒内；
         ③ 每个 B 点都被某簇心代表（最近簇心指派无空簇）。
    可视化（--viz）：整网格(ROI 框 + 圆柱 FREE 半透明) + 机械臂(retract) + 工件 + B 按簇着色(每类 1 色) + 簇心(大球)。
    """
    from gt_gen.reach_b import compute_blocking_B
    from gt_gen.candidates import cluster_centroids

    h, vm, P = ctx["h"], ctx["vm"], ctx["P"]
    nbv = ctx["cfg"].params["nbv"]
    k = nbv["k_lookahead"]
    cl = nbv.get("cluster", {})                              # 聚类参数全取自 default.yaml
    max_clusters = int(cl.get("max", 3))
    link_dist = float(cl.get("link_dist_m", 0.15))
    method = str(cl.get("method", "single"))

    print("\n== verify cluster_centroids ==")
    print(f"  聚类参数(default.yaml): max={max_clusters} link_dist={link_dist}m method={method}")
    B = compute_blocking_B(h, vm, P, ctx["reach_idx"], k)
    assert B.shape[0] > 0, "B 为空——换更靠前的 reach 状态再测"
    Bw = vm.voxel_to_world(B)
    lo, hi = Bw.min(0), Bw.max(0)

    targets = cluster_centroids(Bw, max_clusters=max_clusters, link_dist=link_dist, method=method)
    # 每个 B 点指派到最近簇心（用于上色 + 验证无空簇）
    dist = np.linalg.norm(Bw[:, None, :] - targets[None, :, :], axis=2)   # (M,k)
    assign = dist.argmin(axis=1)

    print(f"  B={B.shape[0]} 格  →  簇心 {targets.shape[0]} 个：")
    for j, t in enumerate(targets):
        cnt = int((assign == j).sum())
        print(f"    T{j}={np.round(t, 3)}  簇内点数={cnt}")
        assert np.all(t >= lo - vm.voxel_size) and np.all(t <= hi + vm.voxel_size), "簇心越出 B 包围盒"
        assert cnt > 0, f"簇 {j} 为空（不应出现）"
    assert 1 <= targets.shape[0] <= max_clusters, "簇数应在 [1, max_clusters]"
    print(f"  ✓ 簇数 {targets.shape[0]}∈[1,{max_clusters}]、簇心都在 B 包围盒内、无空簇")

    if viz:
        geoms = [("arm", _arm_mesh(h, ctx["cur_cfg"]), "lit", None),
                 ("work", _work_mesh(ctx["scene"]), "lit", None)]
        fc = vm.state_centers(1)                              # FREE=1：整块初始圆柱
        if fc.shape[0]:
            geoms.append(("free", _cells_mesh(vm, fc), "fill", [0.20, 0.45, 0.95, 0.18]))
        for j, t in enumerate(targets):
            col = PALETTE[j % len(PALETTE)]
            cm = _cells_mesh(vm, Bw[assign == j]); cm.paint_uniform_color(col)
            geoms.append((f"cluster{j}", cm, "lit", None))
            import open3d as o3d
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.05); s.translate(t)
            s.paint_uniform_color([min(1.0, col[0] + 0.3), min(1.0, col[1] + 0.3), min(1.0, col[2] + 0.3)])
            s.compute_vertex_normals()
            geoms.append((f"centroid{j}", s, "lit", None))
        geoms += _roi_and_base(vm)
        _draw(geoms, f"step8 cluster_centroids: B={B.shape[0]}格 → {targets.shape[0]}簇(每色1类,大球=簇心)")


def verify_standoff_poses_looking_at(ctx, viz):
    """验证 standoff_poses_looking_at：对目标点 T 撒朝向 T、站位在 FREE 的相机位姿。

    判据：① 至少 1 个位姿；② 每个位姿眼睛 p 在 FREE 体素；③ 光轴(+Z)精确对准 T(夹角≈0)；
         ④ 视线 p→T 不被 OCCUPIED 挡；⑤ 站距 |p-T| 命中 standoff_d 某档；⑥ 旋转部分正交 det=1。
    可视化（--viz）：按【步骤】连开 5 个窗口，逐步叠加：
       步骤0 场景(工件/圆柱FREE/B橙/目标T品红) → 步骤1 半球轴 axis(黄线 T→flange原点) →
       步骤2 绕 axis 撒 n 个方向 u(灰线) → 步骤3 眼睛 p=T+d·u(绿=在FREE保留/红=出圆柱弃) →
       步骤4 保留的相机位姿(每个相机坐标轴 + 红线=视线 p→T)。
    """
    import open3d as o3d
    from gt_gen.reach_b import compute_blocking_B
    from gt_gen.voxmap import FREE
    from gt_gen import candidates as C

    h, vm, cam = ctx["h"], ctx["vm"], ctx["cam"]
    nbv = ctx["cfg"].params["nbv"]
    standoff_d = tuple(nbv["standoff_d_m"]); n_view_dirs = int(nbv["n_view_dirs"])

    print("\n== verify standoff_poses_looking_at ==")
    print(f"  参数(default.yaml): standoff_d_m={standoff_d}  n_view_dirs={n_view_dirs}")
    B = compute_blocking_B(h, vm, ctx["P"], ctx["reach_idx"], nbv["k_lookahead"])
    T = C.cluster_centroids(vm.voxel_to_world(B))[0]          # 取第一个簇心当目标点
    anchor = C.flange_origin(h, ctx["cur_cfg"])               # 半球轴锚点：当前构型 flange 原点
    print(f"  目标点 T={np.round(T, 3)}  flange锚点={np.round(anchor, 3)}")

    poses = C.standoff_poses_looking_at(T, vm, cam, anchor=anchor,
                                        standoff_d=standoff_d, n_view_dirs=n_view_dirs)
    print(f"  → 生成相机位姿 {len(poses)} 个")
    assert len(poses) >= 1, "无 standoff 位姿（自由区太小/视线全被挡）"
    for M in poses:
        p = M[:3, 3]; axis = M[:3, 2]
        assert int(vm.get(vm.world_to_voxel(p))) == FREE, "眼睛 p 不在 FREE"
        look = C._normalize(T - p)
        assert float(np.dot(axis, look)) > 1 - 1e-6, "光轴未对准 T"
        assert C._sight_clear(vm, p, T), "视线被 OCCUPIED 挡"
        dmin = min(abs(np.linalg.norm(p - T) - dd) for dd in standoff_d)
        assert dmin < 1e-6, "站距不在 standoff 档位"
        R = M[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-6) and abs(np.linalg.det(R) - 1) < 1e-6, "旋转非正交"
    print(f"  ✓ {len(poses)} 个位姿：眼在FREE、光轴对准T、视线通、站距合档、旋转正交")

    if not viz:
        return

    # —— 公共场景几何（每步窗口都含）；b_fill=True 时 B 用半透明 ——
    def base_geoms(b_fill=False):
        g = [("arm", _arm_mesh(h, ctx["cur_cfg"]), "lit", None),
             ("work", _work_mesh(ctx["scene"]), "lit", None)]
        fc = vm.state_centers(FREE)
        if fc.shape[0]:
            g.append(("free", _cells_mesh(vm, fc), "fill", [0.20, 0.45, 0.95, 0.15]))
        bm = _cells_mesh(vm, vm.voxel_to_world(B))
        if b_fill:
            g.append(("B", bm, "fill", [1.0, 0.55, 0.0, 0.30]))        # 橙 半透明
        else:
            bm.paint_uniform_color([1.0, 0.55, 0.0])
            g.append(("B", bm, "lit", None))                          # 橙 不透明
        g.append(("T", _ball(T, 0.05, [0.9, 0.1, 0.9]), "lit", None))   # 目标点 品红
        g += _roi_and_base(vm)
        return g

    # 步骤0：场景
    _draw(base_geoms(), "step8 standoff 步骤0 场景: 工件(灰) 圆柱FREE(蓝) B(橙) 目标T(品红球)")

    # 步骤1：半球轴 axis = (flange-T)/|..|（黄线 T → flange 原点）；本步 B 用半透明
    axis = C._normalize(anchor - T)
    g = base_geoms(b_fill=True) + [("axis", _lines([(T, anchor)], [0.95, 0.85, 0.0]), "line", None),
                        ("flange", _ball(anchor, 0.04, [0.95, 0.85, 0.0]), "lit", None)]
    _draw(g, f"step8 standoff 步骤1 半球轴 axis=(flange-T)/|·|={np.round(axis,2)} (黄线 T→flange原点)")

    # 步骤1b：半球边界——以 T 为心、朝 axis 一侧、半径取最大站距档的半球曲面(半透明青) + 赤道环(青线)，
    #         看清"相机就撒在这个半球壳上"。
    r_hemi = float(max(standoff_d))
    g = base_geoms(b_fill=True) + [
        ("axis", _lines([(T, anchor)], [0.95, 0.85, 0.0]), "line", None),
        ("flange", _ball(anchor, 0.04, [0.95, 0.85, 0.0]), "lit", None),
        ("hemi", _hemisphere_surface(T, axis, r_hemi, [0.0, 0.8, 0.8, 0.22]), "fill", [0.0, 0.8, 0.8, 0.22]),
        ("equator", _circle(T, axis, r_hemi, [0.0, 0.6, 0.6]), "line", None)]
    _draw(g, f"step8 standoff 步骤1b 半球边界(青壳 r={r_hemi}m, 青环=赤道) 朝axis一侧(flange侧), 相机撒在此半球壳上")

    # 步骤2：绕 axis 撒 n 个方向 u（灰线，从 T 出发，长 = 最大站距档，只为伸到最远那档眼睛处）
    dirs = C._hemisphere_dirs(axis, n_view_dirs)
    segs = [(T, T + r_hemi * u) for u in dirs]
    g = base_geoms() + [("axis", _lines([(T, anchor)], [0.95, 0.85, 0.0]), "line", None),
                        ("dirs", _lines(segs, [0.35, 0.35, 0.35]), "line", None)]
    _draw(g, f"step8 standoff 步骤2 绕axis撒{n_view_dirs}个方向u (灰线长={r_hemi}m=最大站距档, 半球内)")

    # 步骤3：眼睛 p=T+d·u；绿=在FREE(保留)、红=出圆柱/非FREE(弃)
    g = base_geoms()
    nkeep = 0
    for d in standoff_d:
        for u in dirs:
            p = T + d * u
            free = int(vm.get(vm.world_to_voxel(p))) == FREE and C._sight_clear(vm, p, T)
            nkeep += free
            g.append((f"p_{d}_{id(u)}", _ball(p, 0.022, [0.1, 0.8, 0.2] if free else [0.85, 0.1, 0.1]),
                      "lit", None))
    _draw(g, f"step8 standoff 步骤3 眼睛p=T+d·u: 绿=在FREE保留({nkeep}) 红=出圆柱弃")

    # 步骤4：保留的相机位姿（相机坐标轴 + 红线 视线 p→T）
    g = base_geoms()
    for M in poses:
        fr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10); fr.transform(M)
        g.append((f"cam_{id(M)}", fr, "lit", None))
        g.append((f"ray_{id(M)}", _lines([(M[:3, 3], T)], [0.9, 0.1, 0.1]), "line", None))
    _draw(g, f"step8 standoff 步骤4 保留{len(poses)}个相机位姿(轴=相机帧, 红线=视线p→T)")


def verify_look_at_pose(ctx, viz):
    """验证 look_at_pose：给定眼睛 p 与目标 T 构造相机位姿，并由位姿+内参算出/可视化相机视野(FOV)。

    判据：① 平移=p；② 光轴(+Z)精确对准 T(夹角≈0)；③ 旋转正交 det=1；④ 右轴 X 水平(⊥世界上方)、
         下轴 Y 朝下；⑤ 与 standoff 输出一致(standoff 确用 look_at_pose 构造)；
         ⑥ 由位姿+内参把 T 投影回像素，落在画幅内(T 在 FOV 中心)；⑦ 光轴竖直的退化情形旋转仍正交。
    可视化（--viz）：对【每个可行 standoff 位姿】各弹【一个窗口】(一起显示太乱)，每窗画 相机坐标轴 +
         由内参算出的视锥(紫色棱线+半透明锥面，远面过 T) + 视线 p→T(红)，叠加 工件/圆柱FREE/B/目标T。
    """
    import open3d as o3d
    from gt_gen.reach_b import compute_blocking_B
    from gt_gen.voxmap import FREE
    from gt_gen import candidates as C

    h, vm, cam = ctx["h"], ctx["vm"], ctx["cam"]
    nbv = ctx["cfg"].params["nbv"]
    up = np.array([0.0, 0.0, 1.0])

    print("\n== verify look_at_pose ==")
    B = compute_blocking_B(h, vm, ctx["P"], ctx["reach_idx"], nbv["k_lookahead"])
    T = C.cluster_centroids(vm.voxel_to_world(B))[0]
    anchor = C.flange_origin(h, ctx["cur_cfg"])
    poses = C.standoff_poses_looking_at(T, vm, cam, anchor=anchor,
                                        standoff_d=tuple(nbv["standoff_d_m"]), n_view_dirs=int(nbv["n_view_dirs"]))
    assert poses, "无 standoff 位姿"
    p = poses[0][:3, 3]                                       # 取一个真实站在 FREE 的相机眼睛
    M = C.look_at_pose(p, T)
    print(f"  eye p={np.round(p, 3)}  T={np.round(T, 3)}")

    assert np.allclose(M[:3, 3], p), "平移应=p"
    axis = M[:3, 2]; look = C._normalize(T - p)
    ang = float(np.degrees(np.arccos(np.clip(axis @ look, -1, 1))))
    assert ang < 1e-4, f"光轴未对准 T，夹角 {ang}"
    R = M[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6) and abs(np.linalg.det(R) - 1) < 1e-6, "旋转非正交"
    assert abs(M[:3, 0] @ up) < 1e-6, "右轴 X 应水平(⊥世界上方)"
    assert M[:3, 1] @ up <= 1e-6, "下轴 Y 应朝下"
    assert np.allclose(M, poses[0], atol=1e-6), "standoff 应当用 look_at_pose 构造"

    # 由位姿+内参把 T 投影回像素：T 在光轴上 → 应落在画幅中心 (cx,cy)
    Xc = R.T @ (T - p)
    u_px = cam["fx"] * Xc[0] / Xc[2] + cam["cx"]
    v_px = cam["fy"] * Xc[1] / Xc[2] + cam["cy"]
    assert 0 <= u_px <= cam["width"] and 0 <= v_px <= cam["height"], "T 不在 FOV 内"
    print(f"  ✓ 平移=p、光轴对准T({ang:.1e}°)、正交、X水平Y朝下、与standoff一致、"
          f"T投影像素=({u_px:.0f},{v_px:.0f})∈[{cam['width']}x{cam['height']}]")

    # 退化：光轴竖直(p 正下方看 T 正上方)，up 与光轴平行 → 应走 fallback 仍正交
    Md = C.look_at_pose(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    Rd = Md[:3, :3]
    assert np.allclose(Rd.T @ Rd, np.eye(3), atol=1e-6) and abs(np.linalg.det(Rd) - 1) < 1e-6, "退化情形旋转非正交"
    print("  ✓ 退化(光轴竖直)情形旋转仍正交(走 up=[1,0,0] fallback)")

    if not viz:
        return

    # 逐个窗口、每窗显示 1 个可行 standoff 位姿（一起显示太乱）：相机坐标轴 + FOV视锥 + 视线 p→T(红)
    N = len(poses)
    for i, Mi in enumerate(poses):
        pi = Mi[:3, 3]
        depth = float(np.linalg.norm(T - pi))                # 远面取到 T
        edges, cone = _fov_frustum(Mi, cam, depth)
        g = [("arm", _arm_mesh(h, ctx["cur_cfg"]), "lit", None),
             ("work", _work_mesh(ctx["scene"]), "lit", None)]
        fc = vm.state_centers(FREE)
        if fc.shape[0]:
            g.append(("free", _cells_mesh(vm, fc), "fill", [0.20, 0.45, 0.95, 0.10]))
        bm = _cells_mesh(vm, vm.voxel_to_world(B)); bm.paint_uniform_color([1.0, 0.55, 0.0])
        g.append(("B", bm, "lit", None))
        g.append(("T", _ball(T, 0.04, [0.9, 0.1, 0.9]), "lit", None))
        fr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12); fr.transform(Mi)
        g.append(("cam", fr, "lit", None))
        g.append(("fov_cone", cone, "fill", [0.6, 0.2, 0.85, 0.22]))
        g.append(("fov_edges", edges, "line", None))
        g.append(("ray", _lines([(pi, T)], [0.9, 0.1, 0.1]), "line", None))
        g += _roi_and_base(vm)
        _draw(g, f"step8 look_at_pose 位姿{i+1}/{N}: 相机@{np.round(pi,2)} FOV视锥(紫,远面过T,深{depth:.2f}m) 视线→T(红)")


def verify_generate_candidates(ctx, viz):
    """端到端验证 generate_candidates：cluster → standoff → 眼在手 IK → 看向校验 + 保守可达过滤 → 候选构型。

    判据：① ≥1 个候选；对每个候选 —— ② 光轴对准 T ≤ max_look_deg；③ 从当前构型经自由区可达
         (motion_stays_in_free 整臂扫掠 ⊆ FREE)；④ check_state 可行(关节限位+自碰撞)；
         ⑤ 相机眼睛在 FREE；⑥ target 落在 B 包围盒内、IK 位置误差小。
    可视化（--viz）：对【每个候选】各弹【一个窗口】，画 该候选构型下的整臂(绿，真摆成看 B 的姿态) +
         相机坐标轴 + FOV视锥(远面过 T) + 视线 p→T(红) + 工件/圆柱FREE/B/目标T。
    """
    import open3d as o3d
    from gt_gen.reach_b import compute_blocking_B
    from gt_gen.voxmap import FREE
    from gt_gen.swept import motion_stays_in_free
    from gt_gen import curobo_iface as ci
    from gt_gen import candidates as C

    h, vm, cam = ctx["h"], ctx["vm"], ctx["cam"]
    nbv = ctx["cfg"].params["nbv"]
    cur_cfg = ctx["cur_cfg"]
    max_look_deg = float(nbv["max_look_deg"])                # 看向阈值取自 default.yaml

    print("\n== verify generate_candidates ==")
    print(f"  max_look_deg(default.yaml)={max_look_deg}°")
    B = compute_blocking_B(h, vm, ctx["P"], ctx["reach_idx"], nbv["k_lookahead"])
    Bw = vm.voxel_to_world(B); lo, hi = Bw.min(0), Bw.max(0)

    cands = C.generate_candidates(h, vm, B, cam, cur_cfg, max_look_deg=max_look_deg)
    print(f"  B={B.shape[0]} 格 → 通过全部过滤的候选 {len(cands)} 个：")
    assert len(cands) >= 1, "无可达候选——主循环此时应转『就近揭示』兜底（§6）"
    for c in cands:
        ok, nnf = motion_stays_in_free(h, vm, cur_cfg, c.config)
        feas, _ = ci.check_state(h, c.config)
        eye = c.cam_pose[:3, 3]
        eye_free = int(vm.get(vm.world_to_voxel(eye))) == FREE
        print(f"    look={c.look_err_deg:4.1f}° ik_err={c.ik_pos_err*1000:4.1f}mm "
              f"可达={ok}(非FREE={nnf}) 眼在FREE={eye_free} 自碰撞OK={feas} T={np.round(c.target,2)}")
        assert c.look_err_deg <= max_look_deg, "光轴未对准 T"
        assert ok, "候选从当前构型不可达（应已被过滤）"
        assert feas, "候选自碰撞/越限（IK 应已保证）"
        assert eye_free, "相机眼睛不在 FREE"
        assert np.all(c.target >= lo - vm.voxel_size) and np.all(c.target <= hi + vm.voxel_size), "T 越出 B 包围盒"
    print(f"  ✓ {len(cands)} 个候选：朝B、可达(扫掠⊆FREE)、自碰撞OK、眼在FREE、T在B包围盒内")

    if not viz:
        return

    N = len(cands)
    for i, c in enumerate(cands):
        eye = c.cam_pose[:3, 3]
        depth = float(np.linalg.norm(c.target - eye))
        edges, cone = _fov_frustum(c.cam_pose, cam, depth)
        g = [("arm", _arm_mesh(h, c.config), "lit", None),     # 候选构型下的整臂（真摆成看 B 的姿态）
             ("work", _work_mesh(ctx["scene"]), "lit", None)]
        fc = vm.state_centers(FREE)
        if fc.shape[0]:
            g.append(("free", _cells_mesh(vm, fc), "fill", [0.20, 0.45, 0.95, 0.10]))
        bm = _cells_mesh(vm, vm.voxel_to_world(B)); bm.paint_uniform_color([1.0, 0.55, 0.0])
        g.append(("B", bm, "lit", None))
        g.append(("T", _ball(c.target, 0.04, [0.9, 0.1, 0.9]), "lit", None))
        fr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12); fr.transform(c.cam_pose)
        g.append(("cam", fr, "lit", None))
        g.append(("fov_cone", cone, "fill", [0.6, 0.2, 0.85, 0.22]))
        g.append(("fov_edges", edges, "line", None))
        g.append(("ray", _lines([(eye, c.target)], [0.9, 0.1, 0.1]), "line", None))
        g += _roi_and_base(vm)
        _draw(g, f"step8 generate_candidates 候选{i+1}/{N}: 整臂(绿)看B look={c.look_err_deg:.1f}° "
                 f"ik_err={c.ik_pos_err*1000:.1f}mm FOV(紫,远面过T)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", default=SEAM)
    ap.add_argument("--viz", action="store_true", help="弹 open3d 交互窗口（需显示器，默认关）")
    args = ap.parse_args()

    ctx = build_scene(args)

    # —— 逐个验证 candidates.py 的函数（后续按需取消注释/补写）——
    # verify_cluster_centroids(ctx, args.viz)
    verify_cluster_centroids(ctx, False)
    # verify_standoff_poses_looking_at(ctx, args.viz)
    verify_standoff_poses_looking_at(ctx, False)
    verify_look_at_pose(ctx, False)
    # verify_generate_candidates(ctx, args.viz)
    verify_generate_candidates(ctx, args.viz)

    print("\nVERIFY_STEP8_OK [cluster_centroids, standoff_poses_looking_at, look_at_pose, generate_candidates]")


if __name__ == "__main__":
    main()
