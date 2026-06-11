"""原型对照（独立脚本，不改仓库现有代码）：单次观测的 FREE/OCC 体积——
  A) 现状 trimesh 逐像素 raycast_observe(pixel_stride=16)
  B) warp 逐体素视锥雕刻（视锥内每个体素与相机中心连线，wp.mesh_query_ray 判遮挡）

目的：验证 warp 方案能把单次观测的 FREE 从「稀疏细管」变「实心锥」，从而扛得住 sync 的
inflate=1 障碍膨胀（见会话讨论）。指标：FREE/OCC 体素数、实心 FREE（26 邻接全 FREE 的内点，
≈ 膨胀后仍留存的 FREE）、耗时。

运行：
  conda run -n env_isaaclab python scripts/proto_warp_carve.py            # 仅打印对照
  conda run -n env_isaaclab python scripts/proto_warp_carve.py --viz      # 再弹 open3d 对比窗口
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

SEAM = ("/media/a/新加卷/hanfeng/segment_sub_output/"
        "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_22.pkl")


def _interior_free(grid, FREE):
    """26 邻接全 FREE 的 FREE 内点数（≈ 扛得住 sync inflate=1 障碍膨胀的 FREE）。"""
    from scipy import ndimage
    free = (grid == FREE)
    st = ndimage.generate_binary_structure(3, 3)
    return int(ndimage.binary_erosion(free, structure=st, border_value=0).sum())


# ----------------- 两种观测实现 -----------------

def observe_trimesh(vm, M, cam, scene, max_depth, pixel_stride=16):
    """A) 现状：trimesh 逐像素 raycast，沿射线标 FREE、命中标 OCC。"""
    from gt_gen.sensor import raycast_observe
    from gt_gen.mapping import commit_observation
    free_pts, occ_pts = raycast_observe(M, cam, scene, max_depth, pixel_stride=pixel_stride)
    commit_observation(vm, free_pts, occ_pts)
    return {"free_pts": len(free_pts), "occ_pts": len(occ_pts)}


def _build_wp_mesh(scene):
    import warp as wp
    verts = np.asarray(scene.vertices, dtype=np.float32)
    faces = np.asarray(scene.faces, dtype=np.int32).reshape(-1)
    return wp.Mesh(points=wp.array(verts, dtype=wp.vec3, device="cuda"),
                   indices=wp.array(faces, dtype=wp.int32, device="cuda"))


def _carve_kernel():
    import warp as wp

    @wp.kernel
    def carve(mesh: wp.uint64, cam_o: wp.vec3,
              centers: wp.array(dtype=wp.vec3), ranges: wp.array(dtype=wp.float32),
              max_t: wp.float32, half_vox: wp.float32, out: wp.array(dtype=wp.int32)):
        tid = wp.tid()
        c = centers[tid]
        r = ranges[tid]
        dn = wp.normalize(c - cam_o)
        query = wp.mesh_query_ray(mesh, cam_o, dn, max_t)
        if query.result:
            th = query.t
            if r < th - half_vox:
                out[tid] = 1            # FREE：表面前方、无遮挡
            elif r <= th + half_vox:
                out[tid] = 2            # OCCUPIED：命中表面那一层
            else:
                out[tid] = 0            # 被挡 → UNKNOWN
        else:
            out[tid] = 1                # 该方向 max_depth 内无命中 → FREE
    return carve


def observe_warp_carve(vm, M, cam, scene, max_depth, mesh, kernel, centers_all, idx_all):
    """B) warp 视锥雕刻：选视锥内体素，逐体素连线 mesh_query_ray 判 FREE/OCC/遮挡。

    centers_all/idx_all：ROI 全体素中心 + 下标（循环外缓存，ROI 不变）。
    """
    import warp as wp
    from gt_gen.voxmap import FREE, OCCUPIED

    org = M[:3, 3]; R = M[:3, :3]
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    W, H = cam["width"], cam["height"]
    vs = vm.voxel_size
    near = vs

    Xc = (centers_all - org) @ R                                  # = R^T (C-org)
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * Xc[:, 0] / z + cx
        v = fy * Xc[:, 1] / z + cy
    rr = np.linalg.norm(centers_all - org, axis=1)
    infr = ((z > near) & (z <= max_depth) & (u >= 0) & (u <= W) &
            (v >= 0) & (v <= H) & (rr <= max_depth))             # 视锥内
    idx_f = idx_all[infr]

    cw = wp.array(centers_all[infr].astype(np.float32), dtype=wp.vec3, device="cuda")
    rw = wp.array(rr[infr].astype(np.float32), dtype=wp.float32, device="cuda")
    ow = wp.zeros(idx_f.shape[0], dtype=wp.int32, device="cuda")
    wp.launch(kernel, dim=idx_f.shape[0],
              inputs=[mesh.id, wp.vec3(float(org[0]), float(org[1]), float(org[2])),
                      cw, rw, float(max_depth), float(0.5 * vs), ow], device="cuda")
    wp.synchronize()
    out = ow.numpy()

    vm.set_many(idx_f[out == 1], FREE)
    vm.set_many(idx_f[out == 2], OCCUPIED)                        # occ 后写覆盖（与 commit 一致）
    return {"infrustum": int(idx_f.shape[0]), "total": int(idx_all.shape[0])}


# ----------------- 可视化（参考 main_loop._debug_viz_curobo 的切片风格） -----------------

def _show(title, vm, ctx, M, every_n_layers=2):
    """画一个 voxmap：FREE(蓝半透明) + OCCUPIED(红) 沿 z【每 every_n_layers 层取 1 层】水平切片
    （同 _debug_viz_curobo，控制渲染量、又看清沿高度的形态），叠加工件(灰) + retract 整臂(绿)
    + 相机视锥(紫) + ROI 框/base 轴。
    """
    from verify_step8 import _arm_mesh, _work_mesh, _cells_mesh, _draw, _roi_and_base, _fov_frustum
    from gt_gen.voxmap import FREE, OCCUPIED

    cfg, h_truth, scene, cam = ctx["cfg"], ctx["h_truth"], ctx["scene"], ctx["cam"]
    vs = vm.voxel_size
    n_layers = max(1, int(every_n_layers))
    z_origin = float(vm.origin[2])

    def slab(centers):
        """沿 z 每 n_layers 层抽 1 层（同 _debug_viz_curobo）。"""
        if centers.shape[0] == 0:
            return centers
        layer = np.rint((centers[:, 2] - z_origin) / vs).astype(int)
        return centers[layer % n_layers == 0]

    geoms = [("work", _work_mesh(scene), "lit", None),
             ("arm", _arm_mesh(h_truth, list(cfg.retract_config)), "lit", None)]
    nf_all = int((vm.grid == FREE).sum())
    no_all = int((vm.grid == OCCUPIED).sum())
    fc = slab(vm.state_centers(FREE))
    n_fshow = int(fc.shape[0])
    if n_fshow:
        geoms.append(("free", _cells_mesh(vm, fc), "fill", [0.10, 0.45, 0.95, 0.35]))
    oc = slab(vm.state_centers(OCCUPIED))
    if oc.shape[0]:
        om = _cells_mesh(vm, oc); om.paint_uniform_color([0.92, 0.12, 0.12])
        geoms.append(("occ", om, "lit", None))
    edges, cone = _fov_frustum(M, cam, cfg.max_depth_m)
    geoms.append(("fov_cone", cone, "fill", [0.6, 0.2, 0.85, 0.10]))
    geoms.append(("fov_edges", edges, "line", None))
    geoms += _roi_and_base(vm)
    print(f"  打开 open3d: {title}  FREE={nf_all} OCC={no_all}（每{n_layers}层取1层显示FREE {n_fshow}格）...")
    _draw(geoms, f"{title}: FREE={nf_all}格(蓝,每{n_layers}层1层显示{n_fshow}) OCC={no_all}格(红) 绿=整臂 紫=视锥")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", default=SEAM)
    ap.add_argument("--viz", action="store_true", help="弹 open3d 对比窗口（A 稀疏 / B 实心）")
    ap.add_argument("--pixel-stride", type=int, default=16)
    args = ap.parse_args()

    class A:
        seam = args.seam
        viz = False
    from verify_step10 import build_scene_step10
    from gt_gen.sensor import camera_pose_from_config
    from gt_gen.voxmap import build_roi_voxmap, FREE, OCCUPIED

    ctx = build_scene_step10(A())
    cfg, h_truth, cam, scene = ctx["cfg"], ctx["h_truth"], ctx["cam"], ctx["scene"]
    q = list(cfg.retract_config)
    M = camera_pose_from_config(h_truth, q, cam)
    max_depth, vs = cfg.max_depth_m, cfg.voxel_size_m
    print(f"\nROI shape={build_roi_voxmap(cfg).shape}  voxel={vs}  max_depth={max_depth}")
    print(f"cam_org={np.round(M[:3,3],3)}  cam_axis(+Z)={np.round(M[:3,2],3)}")

    # ---------- A) trimesh ----------
    vmA = build_roi_voxmap(cfg)
    t0 = time.perf_counter()
    iA = observe_trimesh(vmA, M, cam, scene, max_depth, pixel_stride=args.pixel_stride)
    tA = time.perf_counter() - t0
    cA = vmA.counts()
    print(f"\n[A trimesh stride{args.pixel_stride}] free_pts={iA['free_pts']} occ_pts={iA['occ_pts']}  "
          f"FREE={cA[FREE]} OCC={cA[OCCUPIED]}  实心FREE={_interior_free(vmA.grid, FREE)}  用时={tA*1000:.0f}ms")

    # ---------- B) warp ----------
    import warp as wp
    wp.init()
    mesh = _build_wp_mesh(scene)
    kernel = _carve_kernel()
    # ROI 全体素中心 + 下标（循环外缓存一次）
    sh = vmA.shape
    gi, gj, gk = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]), np.arange(sh[2]), indexing="ij")
    idx_all = np.stack([gi.ravel(), gj.ravel(), gk.ravel()], axis=1)
    centers_all = vmA.voxel_to_world(idx_all)

    vmB = build_roi_voxmap(cfg)
    t0 = time.perf_counter()
    iB = observe_warp_carve(vmB, M, cam, scene, max_depth, mesh, kernel, centers_all, idx_all)
    tB1 = time.perf_counter() - t0
    # 稳态再跑一次（kernel 已编译）
    vmB2 = build_roi_voxmap(cfg)
    t0 = time.perf_counter()
    observe_warp_carve(vmB2, M, cam, scene, max_depth, mesh, kernel, centers_all, idx_all)
    tB2 = time.perf_counter() - t0
    cB = vmB.counts()
    print(f"[B warp carve]        视锥内={iB['infrustum']}/{iB['total']}  "
          f"FREE={cB[FREE]} OCC={cB[OCCUPIED]}  实心FREE={_interior_free(vmB.grid, FREE)}")
    print(f"                      首跑={tB1*1000:.0f}ms(含JIT) 稳态={tB2*1000:.0f}ms")

    print(f"\n对照 实心FREE：A={_interior_free(vmA.grid, FREE)}  B={_interior_free(vmB.grid, FREE)}  "
          f"（B 远大 → 视锥实心、扛得住 inflate=1）")

    if args.viz:
        _show("A trimesh(稀疏细管)", vmA, ctx, M)
        _show("B warp carve(实心锥)", vmB, ctx, M)

    print("\nPROTO_WARP_CARVE_DONE")


if __name__ == "__main__":
    main()
