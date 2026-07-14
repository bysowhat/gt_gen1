"""可视化 ROI 网格(线框,只画边) + 初始关节角下的整臂碰撞球，看清机械臂与网格相对关系。
- voxmap 盒 与 cuRobo voxel 世界盒 用不同颜色的线框(不填充)；
- 碰撞球半透明，便于透视看到盒子边。

运行：
  conda run -n env_isaaclab python scripts/viz_roi_grids.py --mode save --out /tmp/roi_grids
  conda run -n env_isaaclab python scripts/viz_roi_grids.py --mode show
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VOXMAP_COLOR = [0.10, 0.35, 0.95]      # 蓝：自己维护的 voxmap 盒
CUROBO_COLOR = [0.95, 0.45, 0.10]      # 橙：cuRobo voxel 世界盒
ARM_COLOR = [0.20, 0.72, 0.32]         # 绿：碰撞球


def aabb_wire(lo, hi, color):
    import open3d as o3d
    aabb = o3d.geometry.AxisAlignedBoundingBox(np.asarray(lo, float), np.asarray(hi, float))
    ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
    ls.paint_uniform_color(color)
    return ls


def fk_spheres(handle, q):
    import torch
    st = handle.mg.kinematics.get_state(torch.tensor([list(q)], dtype=torch.float32, device="cuda"))
    return st.link_spheres_tensor[0].detach().cpu().numpy()


# 立方体 8 角(±h) 与 12 条边
_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7),      # z 向
          (0, 2), (1, 3), (4, 6), (5, 7),      # y 向
          (0, 4), (1, 5), (2, 6), (3, 7)]      # x 向


def arm_cell_centers(origin, voxel, spheres):
    """把整臂碰撞球体素化到 (origin,voxel) 网格，返回被占据格子的中心 (M,3)。"""
    origin = np.asarray(origin, float)
    occ = set()
    for s in spheres:
        c = s[:3]; r = float(s[3])
        if r <= 1e-4:
            continue
        lo = np.floor((c - r - origin) / voxel).astype(int)
        hi = np.floor((c + r - origin) / voxel).astype(int)
        ii, jj, kk = np.meshgrid(*[np.arange(lo[k], hi[k] + 1) for k in range(3)], indexing="ij")
        idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], 1)
        ctr = origin + (idx + 0.5) * voxel
        for t in map(tuple, idx[np.linalg.norm(ctr - c, axis=1) <= r]):
            occ.add(t)
    idx = np.array(sorted(occ)) if occ else np.empty((0, 3), int)
    return origin + (idx + 0.5) * voxel


def cells_wire(centers, voxel, color):
    """把一堆格子中心画成线框立方体(只画边)。返回 o3d LineSet。"""
    import open3d as o3d
    h = voxel / 2.0
    pts = (centers[:, None, :] + _CORNERS[None] * h).reshape(-1, 3)
    base = (np.arange(centers.shape[0]) * 8)[:, None]
    lines = (np.array(_EDGES)[None] + base[:, :, None]).reshape(-1, 2)
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts),
                              o3d.utility.Vector2iVector(lines))
    ls.paint_uniform_color(color)
    return ls


def cells_solid(centers, voxel, color):
    """把一堆格子中心画成【实心】立方体(不画边)。返回合并的 TriangleMesh。"""
    import open3d as o3d
    merged = o3d.geometry.TriangleMesh()
    for c in centers:
        b = o3d.geometry.TriangleMesh.create_box(voxel, voxel, voxel)
        b.translate(np.asarray(c, float) - voxel / 2.0)
        merged += b
    if len(merged.vertices):
        merged.paint_uniform_color(color); merged.compute_vertex_normals()
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["show", "save"], default="save")
    ap.add_argument("--out", default="/tmp/roi_grids")
    ap.add_argument("--random", type=int, default=0,
                    help=">0：随机生成这么多【相邻】占据格(连通块)，sync 到 cuRobo 后对比(蓝=voxmap 橙=cuRobo)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from gt_gen.config import load_config
    from gt_gen import curobo_iface as ci
    from gt_gen.collision_sync import _world_centers, sync_collision_world, curobo_occupied_centers
    from gt_gen.voxmap import build_roi_voxmap, FREE, OCCUPIED

    cfg = load_config()
    h = ci.init_curobo(cfg)                      # cuRobo voxel 世界用 config ROI
    vm = build_roi_voxmap(cfg)                   # voxmap 用同一 ROI(外扩1体素)
    vs = cfg.voxel_size_m_coarse

    vm_lo, vm_hi = vm.origin, vm.upper
    vg = h.mg.world_coll_checker.get_voxel_grid(h.voxel["name"])
    c = _world_centers(h, vg)
    cu_origin = c.min(0) - vs / 2
    cu_lo, cu_hi = c.min(0) - vs / 2, c.max(0) + vs / 2

    retract = cfg.retract_config
    sph = fk_spheres(h, retract)

    if args.random > 0:
        # 随机生成一个【连通】占据块(随机游走加邻格) → sync → cuRobo；对比两者占据格
        rng = np.random.default_rng(args.seed)
        vm.fill(FREE)
        start = tuple(int(x) for x in vm.world_to_voxel(rng.uniform(vm_lo + 0.6, vm_hi - 0.6)))
        cells = {start}; clist = [start]
        steps = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        guard = 0
        while len(cells) < args.random and guard < args.random * 50:
            guard += 1
            base = clist[rng.integers(len(clist))]
            nb = tuple(int(base[k] + steps[rng.integers(6)][k]) for k in range(3))
            if nb not in cells:
                cells.add(nb); clist.append(nb)
        idx = np.array(sorted(cells))
        idx = idx[vm.in_bounds(idx)]
        vm.set_many(idx, OCCUPIED)
        sync_collision_world(h, vm)
        vm_cells = vm.state_centers(OCCUPIED)
        cu_cells = curobo_occupied_centers(h)
        st = np.asarray(vm.get(vm.world_to_voxel(cu_cells)))
        agree = float((st != FREE).mean()) * 100 if cu_cells.shape[0] else 100.0
        print(f"随机连通块 {args.random} 格 → voxmap 占据 {vm_cells.shape[0]} / cuRobo 占据 {cu_cells.shape[0]}"
              f"  吻合 {agree:.2f}%  差 {abs(vm_cells.shape[0]-cu_cells.shape[0])}")
    else:
        # 缺省：整臂占据的 4cm 格子(两套网格各自体素化)
        vm_cells = arm_cell_centers(vm.origin, vs, sph)
        cu_cells = arm_cell_centers(cu_origin, vs, sph)
        print(f"碰撞球 {int((sph[:,3]>1e-4).sum())} 个 → 占据格子 voxmap {vm_cells.shape[0]} / cuRobo {cu_cells.shape[0]}")
    print(f"voxel={vs}m  voxmap 盒(蓝): {np.round(vm_lo,2)}→{np.round(vm_hi,2)} 共 {vm.num_voxels} 格")

    import open3d as o3d
    # 元组: (name, geom, kind, rgba)  kind: line=线框 / fill=半透明实心 / frame=坐标轴
    geoms = [
        ("roi_voxmap", aabb_wire(vm_lo, vm_hi, [0.6, 0.7, 0.95]), "line", None),   # ROI 外框(线)
        ("roi_curobo", aabb_wire(cu_lo, cu_hi, [0.95, 0.8, 0.6]), "line", None),
        ("cells_voxmap", cells_solid(vm_cells, vs, VOXMAP_COLOR), "fill", VOXMAP_COLOR + [0.6]),
        ("cells_curobo", cells_solid(cu_cells, vs, CUROBO_COLOR), "fill", CUROBO_COLOR + [0.6]),
    ]
    merged = o3d.geometry.TriangleMesh()
    for s in sph:
        r = float(s[3])
        if r <= 1e-4:
            continue
        m = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=8); m.translate(s[:3])
        merged += m
    merged.paint_uniform_color(ARM_COLOR); merged.compute_vertex_normals()
    geoms.append(("arm", merged, "fill", ARM_COLOR + [0.55]))
    geoms.append(("base", o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3), "frame", None))

    # 取景中心：random 模式对准占据块，否则对准机械臂
    ctr = vm_cells.mean(0) if (args.random > 0 and vm_cells.shape[0]) else sph[sph[:, 3] > 1e-4][:, :3].mean(0)
    span = 1.0 if args.random > 0 else 0.7

    if args.mode == "save":
        import open3d.visualization.rendering as rendering
        os.makedirs(args.out, exist_ok=True)
        rd = rendering.OffscreenRenderer(1400, 1000); rd.scene.set_background([1, 1, 1, 1])
        for name, g, kind, rgba in geoms:
            mat = rendering.MaterialRecord()
            if kind == "fill":                                    # 半透明实心(占据格/碰撞球)
                mat.shader = "defaultLitTransparency"; mat.base_color = rgba
            elif kind == "line":                                  # ROI 外框线
                mat.shader = "unlitLine"; mat.line_width = 2.0
            else:
                mat.shader = "defaultLit"
            rd.scene.add_geometry(name, g, mat)
        for vn, eye in [("near", ctr + np.array([span, -span, span * 0.7])),
                        ("far", ctr + np.array([span * 1.8, -span * 1.8, span * 1.3]))]:
            rd.setup_camera(55.0, ctr.tolist(), eye.tolist(), [0, 0, 1.0])
            p = os.path.join(args.out, f"roi_{vn}.png"); o3d.io.write_image(p, rd.render_to_image()); print("saved", p)
        print("SAVE_OK")
    else:
        o3d.visualization.draw_geometries([g for _, g, _, _ in geoms],
                                          window_name="ROI 占据格(蓝=voxmap 橙=cuRobo, 半透明) + 碰撞球(绿)",
                                          width=1400, height=1000)


if __name__ == "__main__":
    main()
