"""标定"尽量小的初始引导 FREE 空间"——并排评测各 init_free 方案，定其参数。

这是【离线一次性 / 复核】工具，不在运行时调用。它回答一个问题：
gt_gen/init_free 里各"初始 FREE 空间"方案的参数该取多少，才能让机械臂在探索起步时动得了、
同时空间尽量小？做法：对每个方案扫一组参数，用该 FREE 空间对每条 tmp/plan_seam 轨迹跑
compute_reach_pt，统计起步成功率 / 能走多远 / blob 体素数。机器人、retract 或轨迹集变化后，
重跑本脚本复核即可。

【扩展】每个方案 = 一个 calibrate_*(h, cfg, Ps) 函数，返回 (方案名, 默认参数下的 blob 体素中心)。
要新增方案：写一个这样的函数，并把它加进 METHODS 列表即可，main 会自动评测 + 安全核查。

运行：
  conda run -n env_isaaclab python scripts/calibrate_init_free.py
  conda run -n env_isaaclab python scripts/calibrate_init_free.py --check-workpiece --n-workpiece 20

字段读取约定同 scripts/plan_seam.py / scripts/viz_seam_isaacsim.py：
  npz: positions(T,6), retract(6), piece_pose_to_robot(=mesh_pose,[x,y,z,qw,qx,qy,qz]), obj_path
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_TRAJ_DIR = os.path.join(ROOT, "tmp", "plan_seam")
# DQ_GRID = (0.05, 0.15, 0.30)            # 方法①：关节活动半幅(rad) 扫描
# CYL_RADIUS_GRID = (0.40, 0.50, 0.60)                # 方法②：圆柱半径 R(m) 扫描
# CYL_HEIGHT_GRID = (1.4, 1.60, 1.8)                # 方法②：圆柱高度 h(m，从 base_link z=0 往上) 扫描


DQ_GRID = (0.2)            # 方法①：关节活动半幅(rad) 扫描
CYL_RADIUS_GRID = (0.60)                # 方法②：圆柱半径 R(m) 扫描
CYL_HEIGHT_GRID = (1.8)                # 方法②：圆柱高度 h(m，从 base_link z=0 往上) 扫描
VIZ = False                              # 由 --viz 置位：每组参数弹 open3d 窗口（需显示器）
WORKS = []                               # 由 main 填充：[(name, obj_path, mesh_pose), ...]，可视化时逐个工件叠加


def viz_blob(handle, vm, q, title):
    """open3d 交互窗口：初始 FREE blob(蓝半透明) + 构型 q 整臂碰撞球(绿) + 工件(灰) + ROI 线框 + base 轴。

    若 WORKS 非空：对其中【每个工件】各弹一个窗口（同一 blob+整臂，叠加该工件），并在标题/stdout
    报出 blob 到该工件表面的最近距离，逐个看完为止；blob 中触碰工件(<1体素)的格子标红。
    若 WORKS 为空：只弹一个不含工件的窗口。
    （交互窗口，可旋转缩放；关闭当前窗口后自动进入下一个工件。需本机有显示器。）
    """
    import open3d as o3d
    from open3d.visualization import rendering
    from gt_gen.voxmap import FREE
    from gt_gen.swept import fk_spheres_batch
    from gt_gen.sensor import load_truth_scene
    from scipy.spatial.transform import Rotation as Rsp

    vs = vm.voxel_size

    def cells_mesh(centers):
        m = o3d.geometry.TriangleMesh()
        for c in centers:
            b = o3d.geometry.TriangleMesh.create_box(vs, vs, vs); b.translate(c - vs / 2); m += b
        if len(m.vertices):
            m.compute_vertex_normals()
        return m

    def seam_tube(pts, radius=0.006):
        """把焊缝折线 pts(N,3) 画成一串细圆柱(红色 tube)。"""
        pts = np.asarray(pts, float)
        m = o3d.geometry.TriangleMesh()
        z = np.array([0.0, 0.0, 1.0])
        for a, b in zip(pts[:-1], pts[1:]):
            v = b - a; L = float(np.linalg.norm(v))
            if L < 1e-6:
                continue
            cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=L, resolution=8)
            d = v / L
            ax = np.cross(z, d); s = float(np.linalg.norm(ax))
            if s > 1e-9:
                cyl.rotate(Rsp.from_rotvec(ax / s * np.arccos(np.clip(np.dot(z, d), -1, 1))).as_matrix(),
                           center=(0, 0, 0))
            elif d[2] < 0:
                cyl.rotate(Rsp.from_rotvec([np.pi, 0, 0]).as_matrix(), center=(0, 0, 0))
            cyl.translate((a + b) / 2); m += cyl
        if len(m.vertices):
            m.paint_uniform_color([0.90, 0.05, 0.05]); m.compute_vertex_normals()
        return m

    # —— 整臂碰撞球(绿)：所有窗口共用 ——
    sph = fk_spheres_batch(handle, [q])[0]
    arm = o3d.geometry.TriangleMesh()
    for s in sph:
        r = float(s[3])
        if r <= 1e-4:
            continue
        b = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=8); b.translate(s[:3]); arm += b
    arm.paint_uniform_color([0.20, 0.72, 0.32]); arm.compute_vertex_normals()

    fc = vm.state_centers(FREE)                          # blob 体素中心(base 系)，距离/上色都用它
    aabb = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(vm.origin, vm.upper)); aabb.paint_uniform_color([0.6, 0.6, 0.6])
    base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    def draw(geoms, win_title):
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
        print(f"    打开 open3d 窗口: {win_title}（关闭后继续）...")
        o3d.visualization.draw(items, title=win_title, width=1400, height=1000,
                               bg_color=(1.0, 1.0, 1.0, 1.0))

    # —— 无工件：单窗口（保留旧行为）——
    if not WORKS:
        geoms = [("arm", arm, "lit", None)]
        if fc.shape[0]:
            geoms.append(("free", cells_mesh(fc), "fill", [0.20, 0.45, 0.95, 0.30]))
        geoms += [("roi", aabb, "line", None), ("base", base, "lit", None)]
        draw(geoms, title)
        return

    # —— 逐个工件：各弹一窗，报最近距离 ——
    N = len(WORKS)
    for i, (name, obj, mesh_pose, seam) in enumerate(WORKS):
        scene = load_truth_scene(obj, mesh_pose=mesh_pose)
        if fc.shape[0]:
            _, dist, _ = scene.nearest.on_surface(fc)     # 每个 blob 体素中心到工件表面的最近距离
            dmin = float(dist.min()); ntouch = int((dist < vs).sum())
            near = dist < vs                              # 触碰工件的格子
        else:
            dmin = float("nan"); ntouch = 0; near = np.zeros(0, bool)
        print(f"    工件 {i+1}/{N} {name}: blob 到工件最近 {dmin:.3f} m  触碰格(<{vs}m)={ntouch}"
              f"{'' if seam is not None else '  [无焊缝pkl]'}")

        geoms = [("arm", arm, "lit", None)]
        if fc.shape[0]:
            far_c = fc[~near]; near_c = fc[near]
            if far_c.shape[0]:
                geoms.append(("free", cells_mesh(far_c), "fill", [0.20, 0.45, 0.95, 0.30]))  # 蓝半透明
            if near_c.shape[0]:
                tm = cells_mesh(near_c); tm.paint_uniform_color([0.95, 0.10, 0.10])           # 触碰=红
                geoms.append(("touch", tm, "lit", None))
        work = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(scene.vertices)),
                                         o3d.utility.Vector3iVector(np.asarray(scene.faces)))
        work.paint_uniform_color([0.72, 0.72, 0.72]); work.compute_vertex_normals()
        geoms.append(("work", work, "lit", None))
        if seam is not None and len(seam) >= 2:           # 目标焊缝：红色细圆柱
            geoms.append(("seam", seam_tube(seam), "lit", None))
        geoms += [("roi", aabb, "line", None), ("base", base, "lit", None)]
        draw(geoms, f"{title} | 工件{i+1}/{N} {name} 最近={dmin:.3f}m 触碰={ntouch}")


def link6_pos(handle, q):
    import torch
    st = handle.mg.kinematics.get_state(
        torch.tensor([list(q)], dtype=torch.float32, device="cuda"))
    return st.link_pose["Link6"].position[0].detach().cpu().numpy()


def eval_reach(handle, vm, Ps):
    """对每条轨迹用给定 voxmap 跑 compute_reach_pt，返回 (reach_idx[], EE行进[m])。"""
    from gt_gen.reach_b import compute_reach_pt
    ris, travels = [], []
    for P in Ps:
        ri = compute_reach_pt(handle, vm, P)
        ris.append(ri)
        travels.append(float(np.linalg.norm(link6_pos(handle, P[ri]) - link6_pos(handle, P[0]))))
    return np.array(ris), np.array(travels)


def _row(label, n_free, ris, travels):
    """统一打印一行评测结果（label 携带各方案自己的前导列）。"""
    print(f"  {label} | {n_free:7d} | "
          f"{ris.min():3d}/{int(np.median(ris)):3d}/{ris.max():3d} | "
          f"{np.median(travels):.3f}/{travels.max():.3f} | {(ris >= 1).mean() * 100:3.0f}%")


# ============ 各 init_free 方案的标定函数（统一签名 (h, cfg, Ps) -> (名, 默认blob体素中心)）============

def calibrate_swept(h, cfg, Ps):
    """方法① set_initial_free_space：retract 各关节 ±dq 整臂扫掠并集标 FREE，扫 dq 取最小可起步幅度。"""
    from gt_gen.voxmap import build_roi_voxmap
    from gt_gen.init_free import set_initial_free_space

    print("\n==== 方法① set_initial_free_space（关节 ±dq 扫掠）====")
    print("  dq(rad) | blob体素 | reach min/med/max | EE行进 med/max | reach>=1")
    default_cells = None
    for dq in np.atleast_1d(DQ_GRID):
        dq = float(dq)
        vm = build_roi_voxmap(cfg)
        n, cells = set_initial_free_space(h, vm, config=cfg, dq=dq, return_cells=True)
        ris, travels = eval_reach(h, vm, Ps)
        _row(f"dq={dq:4.2f}", n, ris, travels)
        if VIZ:
            viz_blob(h, vm, cfg.retract_config,
                     f"方法① set_initial_free_space  dq={dq:.2f}rad  blob={n}格  起步率={(ris >= 1).mean() * 100:.0f}%")
        if abs(dq - cfg.init_free_dq) < 1e-9:
            default_cells = vm.voxel_to_world(cells)
    return "set_initial_free_space", default_cells


def calibrate_cylinder(h, cfg, Ps):
    """方法② set_initial_free_cylinder：竖立在 base_link 平面、直接给定半径/高度的圆柱整块标 FREE。

    轴过 base 原点(x=y=0)、底面 z=0(base_link)、半径 R、高 h(往上)。
    对 CYL_RADIUS_GRID × CYL_HEIGHT_GRID 做二维扫描，取最小可起步的 (R,h)。
    """
    from gt_gen.voxmap import build_roi_voxmap
    from gt_gen.init_free import set_initial_free_cylinder, base_cylinder_bounds

    r_arm, h_arm = base_cylinder_bounds(h, cfg.retract_config)   # 罩住整臂的最小尺寸（参考下限）
    print("\n==== 方法② set_initial_free_cylinder（base_link 平面竖直圆柱，直接给 R/h）====")
    print(f"  参考：罩住整臂最小 R≥{r_arm:.2f}m  h≥{h_arm:.2f}m（小于此值会切到臂）")
    print("  R(m) h(m) | blob体素 | reach min/med/max | EE行进 med/max | reach>=1")
    default_cells = None
    for R in np.atleast_1d(CYL_RADIUS_GRID):
        R = float(R)
        for hh in np.atleast_1d(CYL_HEIGHT_GRID):
            hh = float(hh)
            vm = build_roi_voxmap(cfg)
            n, cells = set_initial_free_cylinder(h, vm, config=cfg, radius=R, height=hh,
                                                 return_cells=True)
            ris, travels = eval_reach(h, vm, Ps)
            _row(f"R={R:.2f} h={hh:.2f}", n, ris, travels)
            if VIZ:
                viz_blob(h, vm, cfg.retract_config,
                         f"方法② set_initial_free_cylinder  R={R:.2f}m h={hh:.2f}m  blob={n}格  起步率={(ris >= 1).mean() * 100:.0f}%")
            if abs(R - cfg.init_free_cyl_radius) < 1e-9 and abs(hh - cfg.init_free_cyl_height) < 1e-9:
                default_cells = vm.voxel_to_world(cells)
    return "set_initial_free_cylinder", default_cells


# 注册所有方案；新增方案 = 写一个 calibrate_*(h,cfg,Ps) 并加到这里。
METHODS = [calibrate_cylinder, calibrate_swept]


def check_workpiece_clear(cells_world, traj_files, n_sample, voxel_size):
    """安全性核查：blob 体素中心是否贴近某工件表面（< 1 个体素即视为"触碰"）。

    退回 retract 是安全 home，理论上 blob 不应与任何工件相交；这里抽样核查。
    """
    from gt_gen.sensor import load_truth_scene
    import trimesh  # noqa: F401  (load_truth_scene 内部用)

    files = traj_files[:: max(1, len(traj_files) // n_sample)][:n_sample]
    worst = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        obj = str(d["obj_path"])
        if not os.path.exists(obj):
            print(f"  [skip] obj 不存在: {obj}")
            continue
        mp = np.asarray(d["piece_pose_to_robot"], float)
        scene = load_truth_scene(obj, mesh_pose=mp)
        # 体素中心到工件表面的最近距离
        _, dist, _ = scene.nearest.on_surface(cells_world)
        near = int((dist < voxel_size).sum())
        worst.append((near, float(dist.min()), os.path.basename(f)))
    worst.sort(reverse=True)
    print(f"  抽样 {len(worst)} 个工件；'触碰'= 体素中心到工件面 < {voxel_size} m")
    for near, dmin, name in worst[:5]:
        print(f"    {name}: 触碰体素={near}  最近距离={dmin:.3f} m")
    total_touch = sum(w[0] for w in worst)
    print(f"  合计触碰体素={total_touch}（应为 0：blob 落在工件外才安全标 FREE）")


def seam_line_base(npz_path, obj_path):
    """从 npz 名 + obj 同目录定位 seam_*.pkl，读 seam_line(世界系) 换算到 base 系。

    npz 名形如 <part>__seam_NN.npz；seam pkl = dirname(obj)/seam_NN.pkl。找不到则返回 None。
    换算同 plan_seam.seam_ee_pose：base = (line - robot_pos) @ R(robot_quat)。
    """
    import pickle
    from scipy.spatial.transform import Rotation as Rsp

    tag = os.path.basename(npz_path)[:-4].split("__")[-1]    # 'seam_NN'
    pkl = os.path.join(os.path.dirname(obj_path), tag + ".pkl")
    if not os.path.exists(pkl):
        return None
    d = pickle.load(open(pkl, "rb"))
    line = np.asarray(d["seam_line"], float)                 # (N,3) world
    rp = np.asarray(d["robot_pose"][0], float)               # [x,y,z,qw,qx,qy,qz]
    Rwr = Rsp.from_quat(np.r_[rp[4:7], rp[3]]).as_matrix()   # wxyz -> xyzw
    return (line - rp[:3]) @ Rwr                             # (N,3) base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", default=DEFAULT_TRAJ_DIR)
    ap.add_argument("--check-workpiece", action="store_true",
                    help="对各方案默认参数 blob 抽样核查是否贴近工件表面（默认关，较慢）")
    ap.add_argument("--n-workpiece", type=int, default=20)
    ap.add_argument("--viz", action="store_true",
                    help="每组参数弹 open3d 窗口看 blob+整臂（标题含方法+参数；需显示器，默认关）")
    args = ap.parse_args()

    global VIZ, WORKS
    VIZ = args.viz
    from gt_gen.config import load_config
    from gt_gen import curobo_iface as ci
    from gt_gen.voxmap import build_roi_voxmap

    cfg = load_config()
    h = ci.init_curobo(cfg)                       # 无 world → 全自由 VOXEL 世界 + 运动学
    retract = np.asarray(cfg.retract_config, float)

    fs = sorted(glob.glob(os.path.join(args.traj_dir, "*.npz")))
    if not fs:
        print(f"未找到轨迹: {args.traj_dir}/*.npz")
        return
    Ps = [np.asarray(np.load(f, allow_pickle=True)["positions"], float) for f in fs]
    print(f"retract: {np.round(retract, 4)}")
    print(f"n traj : {len(Ps)}  (来自 {args.traj_dir})")

    # 可视化时逐个叠加的工件（用 --n-workpiece 控制数量；设大于轨迹数即看全部）
    if VIZ:
        sub = fs[:: max(1, len(fs) // args.n_workpiece)][:args.n_workpiece]
        WORKS = []
        for f in sub:
            d = np.load(f, allow_pickle=True)
            obj = str(d["obj_path"])
            WORKS.append((os.path.basename(f), obj,
                          np.asarray(d["piece_pose_to_robot"], float),
                          seam_line_base(f, obj)))           # 焊缝线(base 系) 或 None
        print(f"可视化将对每组参数逐个叠加 {len(WORKS)} 个工件（--n-workpiece 控制数量）")

    # 基线：全 UNKNOWN（应全 0 → 起步不了）
    vm0 = build_roi_voxmap(cfg)
    ri_base, _ = eval_reach(h, vm0, Ps)
    print(f"\n全 UNKNOWN 基线 reach_idx: min/med/max = "
          f"{ri_base.min()}/{int(np.median(ri_base))}/{ri_base.max()}  (期望全 0)")

    # 逐方案评测
    results = [method(h, cfg, Ps) for method in METHODS]

    # 各方案默认参数下的 blob 安全性核查
    if args.check_workpiece:
        for name, cells_world in results:
            if cells_world is None:
                continue
            print(f"\n== 工件安全性核查：{name} 默认参数 blob ==")
            check_workpiece_clear(cells_world, fs, args.n_workpiece, cfg.voxel_size_m_coarse)

    print(f"\n默认参数：init_free.dq_rad={cfg.init_free_dq}  "
          f"cyl_radius_m={cfg.init_free_cyl_radius}  cyl_height_m={cfg.init_free_cyl_height}"
          f"（configs/default.yaml）")
    print("CALIB_DONE")


if __name__ == "__main__":
    main()
