"""用 Open3D 可视化 UR12e【初始位姿 + 4 组关节角】之间两两规划轨迹的【合并扫掠空间】（独立脚本）。

共 5 个关节角（初始 retract + 4 组用户给定），对全部 C(5,2)=10 个配对各规划一条无碰撞轨迹
（gt_gen.curobo_iface.plan_to_config，全自由 VOXEL 世界），逐条算整臂扫掠体素
（gt_gen.swept：沿插值轨迹 FK 碰撞球 → 体素化，剔除固定底座球），再把 10 段扫掠体素并起来，
作为一整片半透明占用格可视化。base 处画坐标系，每个关节角画末端(ee)小坐标系。

  · 初始位姿 = configs/default.yaml 引用的 ur12e.yml:retract_config（Phase 1 固定起点）。
  · 其余 4 组 = 下方 GIVEN_JOINTS（顺序同 ur12e 关节）。
  · 规划失败的配对：退回关节空间线性插值（密采样）算扫掠，并打印告警。

运行（env_isaaclab，唯一装了 open3d 且能跑 cuRobo 的环境）：
  /home/a/miniforge3/envs/env_isaaclab/bin/python scripts/viz_joints_open3d.py --mode show
无显示器存图：--mode save --out /tmp/swept_joints
无显示器自检：--headless（建 handle + 规划 + 算各段/合并扫掠格数 + 打印 ee 位姿，不渲染）。
叠加 5 个关节角的整臂碰撞球：加 --show-arms（灰=初始, 红/蓝/绿/橙=4 组）。
"""
import argparse
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 待可视化的 4 组关节角（用户给定，顺序同 ur12e 关节）
GIVEN_JOINTS = [
    [2.1617229,  -1.54621942,  0.41668922, -1.11872395, -1.28860027, -0.94842607],
    [2.01191783, -1.22037132,  0.57176382, -1.67799201, -1.38148195, -1.04241163],
    [0.64489609, -1.4321359,   0.38564521, -0.48228438, -1.17660886, -2.59106523],
    [0.6936698,  -1.21555643,  0.54380876, -0.89780887, -0.78737957, -2.5436681],
]

# 关节角颜色：初始位姿灰，4 组分别 红/蓝/绿/橙（仅 --show-arms 时画）
GRAY = [0.60, 0.60, 0.60]
COLORS = [
    [0.90, 0.20, 0.20],   # 红
    [0.20, 0.55, 0.95],   # 蓝
    [0.20, 0.75, 0.30],   # 绿
    [0.95, 0.70, 0.15],   # 橙
]
NAMES = ["init", "组0", "组1", "组2", "组3"]
SWEPT_COLOR = [0.95, 0.45, 0.10]      # 合并扫掠体素：橙，半透明实心


def quat_wxyz_to_matrix(q):
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


# ---- 轨迹规划 + 扫掠体素 ------------------------------------------------------
def plan_traj(ci, handle, qa, qb):
    """规划 qa→qb 的无碰撞轨迹，返回 (traj(T,dof), ok)。
    规划失败 → 退回关节空间密集线性插值（ok=False，便于打印告警）。"""
    from gt_gen.swept import _interp_count
    res = ci.plan_to_config(handle, list(qa), list(qb))
    if res is not None and bool(res.success.item()):
        traj = res.get_interpolated_plan().position.detach().cpu().numpy()
        return traj, True
    # 退回线性插值（密到每球单步位移 ≤ 一个体素，不漏格）
    qa = np.asarray(qa, float); qb = np.asarray(qb, float)
    K = _interp_count(handle, qa, qb, handle.voxel["voxel_size"])
    ts = np.linspace(0.0, 1.0, K)
    return qa[None] + ts[:, None] * (qb - qa)[None], False


def swept_cells_for_traj(handle, vm, traj):
    """一条轨迹 (T,dof) 的整臂扫掠体素下标 (M,3)：沿轨迹 FK 碰撞球→体素化，剔除固定底座球。"""
    from gt_gen.swept import fk_spheres_batch, voxelize_spheres, _moving_sphere_mask
    sph = fk_spheres_batch(handle, traj)                  # (T,S,4)
    mask = _moving_sphere_mask(handle)                    # 剔除固定底座球（None=不剔除）
    if mask is not None:
        sph = sph[:, mask, :]
    return voxelize_spheres(vm, sph.reshape(-1, 4))       # (M,3)


def fk_spheres(handle, q):
    """(6,) 关节角 → (S,4) 整臂碰撞球（world 系），并返回 ee (pos(3,), quat_wxyz(4,))。"""
    import torch
    qt = torch.tensor([list(q)], dtype=torch.float32, device="cuda")
    st = handle.mg.kinematics.get_state(qt)
    sph = st.link_spheres_tensor[0].detach().cpu().numpy()
    return sph, st.ee_position[0].detach().cpu().numpy(), st.ee_quaternion[0].detach().cpu().numpy()


# ---- Open3D 几何 ------------------------------------------------------------
def make_frame(pos, quat_wxyz, size=0.1):
    import open3d as o3d
    f = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    f.rotate(quat_wxyz_to_matrix(quat_wxyz), center=(0, 0, 0))
    f.translate(pos)
    return f


def cells_mesh(vm, cells):
    """体素下标 (M,3) → 合并成一个半透明实心立方体 mesh。"""
    import open3d as o3d
    vs = vm.voxel_size
    m = o3d.geometry.TriangleMesh()
    for c in vm.voxel_to_world(cells):
        b = o3d.geometry.TriangleMesh.create_box(vs, vs, vs); b.translate(c - vs / 2); m += b
    if len(m.vertices):
        m.compute_vertex_normals()
    return m


def arm_mesh(handle, q, color):
    import open3d as o3d
    sph, _, _ = fk_spheres(handle, q)
    m = o3d.geometry.TriangleMesh()
    for s in sph:
        r = float(s[3])
        if r <= 1e-4 or not np.isfinite(s[:3]).all():
            continue
        b = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=8); b.translate(s[:3]); m += b
    if len(m.vertices):
        m.paint_uniform_color(color); m.compute_vertex_normals()
    return m


def render(geoms, mode, out, title, ctr):
    """geoms: [(name, geometry, kind, rgba)]，kind ∈ {lit, fill(半透明), line}。"""
    import open3d as o3d
    from open3d.visualization import rendering
    if mode == "save":
        os.makedirs(out, exist_ok=True)
        rd = rendering.OffscreenRenderer(1400, 1000); rd.scene.set_background([1, 1, 1, 1])
        for name, g, kind, rgba in geoms:
            mat = rendering.MaterialRecord()
            if kind == "fill":
                mat.shader = "defaultLitTransparency"; mat.base_color = rgba
            elif kind == "line":
                mat.shader = "unlitLine"; mat.line_width = 2.0
            else:
                mat.shader = "defaultLit"
            rd.scene.add_geometry(name, g, mat)
        for vn, eye in [("v0", ctr + np.array([1.4, -1.4, 1.0])),
                        ("v1", ctr + np.array([0.05, -1.9, 0.7]))]:
            rd.setup_camera(55.0, ctr.tolist(), eye.tolist(), [0, 0, 1.0])
            p = os.path.join(out, f"swept_{vn}.png")
            o3d.io.write_image(p, rd.render_to_image()); print("saved", p)
        print("SAVE_OK")
    else:
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
        o3d.visualization.draw(items, title=title, width=1400, height=1000,
                               bg_color=(1.0, 1.0, 1.0, 1.0))


def main():
    ap = argparse.ArgumentParser(
        description="Open3D 可视化 初始位姿 + 4 组关节角 两两规划轨迹的合并扫掠空间（cuRobo）")
    ap.add_argument("--mode", choices=["show", "save"], default="show",
                    help="show=开窗交互；save=离屏存 PNG 到 --out")
    ap.add_argument("--out", default="/tmp/swept_joints", help="--mode save 的输出目录")
    ap.add_argument("--headless", action="store_true",
                    help="只建 handle + 规划 + 算各段/合并扫掠格数 + 打印 ee，不渲染")
    ap.add_argument("--show-arms", action="store_true",
                    help="叠加 5 个关节角的整臂碰撞球（灰=初始, 红/蓝/绿/橙=4 组）")
    ap.add_argument("--save-init-free", nargs="?", const="", default=None, metavar="PATH",
                    help="把合并扫掠空间存为 init_free 预存扫掠文件(.npz: centers+voxel_size)。"
                         "不带参数则存到 config.init_free_swept_path；带路径则存到该路径")
    args = ap.parse_args()

    # warp/trimesh shim 必须在 import curobo 前
    from gt_gen import compat  # noqa: F401
    from gt_gen import curobo_iface as ci
    from gt_gen.config import load_config
    from gt_gen.voxmap import build_roi_voxmap

    cfg = load_config()
    retract = list(map(float, cfg.retract_config))

    print("[viz] 构建 cuRobo handle（首次 warmup 略慢）…")
    handle = ci.init_curobo(cfg)
    vm = build_roi_voxmap(cfg)

    # 5 个关节角：第 0 个是初始位姿(retract)，其余为给定关节角
    configs = [retract] + [list(map(float, j)) for j in GIVEN_JOINTS]

    print(f"[viz] 5 个关节角的 ee 位姿：")
    for name, q in zip(NAMES, configs):
        _, ee_pos, ee_quat = fk_spheres(handle, q)
        print(f"  {name:<6s} joints={np.round(q, 4).tolist()}")
        print(f"  {'':6s} ee_pos={np.round(ee_pos, 4).tolist()} "
              f"ee_quat(wxyz)={np.round(ee_quat, 4).tolist()}")

    # 全部两两配对 C(5,2)=10：各规划一条轨迹 → 算扫掠体素 → 合并
    pairs = list(itertools.combinations(range(len(configs)), 2))
    print(f"\n[viz] 规划 {len(pairs)} 个两两配对的轨迹并算扫掠体素：")
    merged = []
    n_fail = 0
    for a, b in pairs:
        traj, ok = plan_traj(ci, handle, configs[a], configs[b])
        cells = swept_cells_for_traj(handle, vm, traj)
        merged.append(cells)
        if not ok:
            n_fail += 1
        flag = "OK  " if ok else "FAIL→线性"
        print(f"  {NAMES[a]:>4s}→{NAMES[b]:<4s} [{flag}] 轨迹{traj.shape[0]:>4d}点  扫掠 {cells.shape[0]:>5d} 格")

    merged_cells = (np.unique(np.vstack(merged), axis=0)
                    if any(c.shape[0] for c in merged) else np.empty((0, 3), np.int64))
    print(f"\n[viz] 合并扫掠空间：{merged_cells.shape[0]} 格"
          f"（voxel_size={vm.voxel_size}m，规划失败 {n_fail}/{len(pairs)} 段走线性插值）")

    # 存盘为 init_free 预存扫掠文件（centers=体素中心 base 系米 + voxel_size），供
    # gt_gen.init_free.set_initial_free_swept 加载，免每次重算（见 default.yaml init_free.swept_path）。
    if args.save_init_free is not None:
        path = args.save_init_free or cfg.init_free_swept_path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        centers = (vm.voxel_to_world(merged_cells).astype(np.float32)
                   if merged_cells.shape[0] else np.empty((0, 3), np.float32))
        np.savez_compressed(path, centers=centers, voxel_size=np.float32(vm.voxel_size))
        print(f"[viz] 已存 init_free 预存扫掠文件：{path}（{centers.shape[0]} 格中心）")

    if args.headless:
        print("VIZ_JOINTS_DONE")
        return

    import open3d as o3d
    geoms = [("base", o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25), "lit", None)]
    # ROI 包围盒线框
    aabb = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(vm.origin, vm.upper))
    aabb.paint_uniform_color([0.6, 0.6, 0.6])
    geoms.append(("roi", aabb, "line", None))

    # 合并扫掠空间（半透明橙）
    if merged_cells.shape[0]:
        sm = cells_mesh(vm, merged_cells); sm.paint_uniform_color(SWEPT_COLOR)
        geoms.append(("swept", sm, "fill", SWEPT_COLOR + [0.55]))

    # 每个关节角的 ee 坐标系（+可选整臂碰撞球）
    for i, (name, q) in enumerate(zip(NAMES, configs)):
        _, ee_pos, ee_quat = fk_spheres(handle, q)
        geoms.append((f"ee_{name}", make_frame(ee_pos, ee_quat, size=0.12), "lit", None))
        if args.show_arms:
            color = GRAY if i == 0 else COLORS[i - 1]
            geoms.append((f"arm_{name}", arm_mesh(handle, q, color), "lit", None))

    ctr = (vm.voxel_to_world(merged_cells).mean(0) if merged_cells.shape[0]
           else np.asarray(vm.origin) + 0.5 * (np.asarray(vm.upper) - np.asarray(vm.origin)))
    title = "5 关节角两两轨迹 合并扫掠空间（橙=扫掠占用格, 坐标系=各 ee）"
    render(geoms, args.mode, args.out, title, ctr)


if __name__ == "__main__":
    main()
