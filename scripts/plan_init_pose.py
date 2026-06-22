"""规划机械臂初始位姿（起步姿态）相关工具（独立脚本，不动业务代码）。

第一步：用 open3d 可视化【机械臂某构型下的整臂碰撞球】+【init_free 起步引导空间（立方体盒）】，
肉眼核对碰撞球是否整只落在 init 盒内（盒太小→球冒出盒外那层体素恒 UNKNOWN，起步会假阳性非 FREE）。

  · 碰撞球：在构型 q（默认 = retract 固定安全 home）下，对全部 collision_link_names 做 FK，
    把各 link 的 collision_spheres 变换到 base 系，按真实半径画成红色线框球。
  · init 盒：base_link 系下的轴对齐长方体 [box_min, box_max]（configs/default.yaml: init_free.box_min_m /
    box_max_m），画成青色线框。盒中心落在 base_link。
  · base 坐标系：原点处一个小三轴坐标架，便于判读朝向。

运行（本机 conda，需要显示器）：
    conda run -n env_isaaclab --no-capture-output python scripts/plan_init_pose.py
可选：--solid 把碰撞球画成实心球、--q j0 j1 ... 指定构型（缺省用 retract）。
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def init_space_geometries(cfg, q=None, solid_spheres=False):
    """构造 open3d 几何列表：构型 q 下整臂碰撞球 + init_free 立方体盒（线框）+ base 坐标架。

    参数：
      cfg           : Config（load_config()）。
      q             : 构型（rad，长度=关节数）；None 时取 cfg.retract_config（起步固定 home）。
      solid_spheres : True 把碰撞球画成实心球；False（默认）画成红色线框球。

    返回：(geoms, stat)。geoms 为 open3d 几何列表；stat 为 dict（球数 / 盒尺寸等，便于打印）。
    """
    import open3d as o3d
    from gt_gen.obstacle_placement import compute_link_sweep

    if q is None:
        q = cfg.retract_config
    q = [float(v) for v in q]

    geoms = []

    # —— 整臂碰撞球：FK 到 base 系，按真实半径逐球画 ——
    per_wp, _ = compute_link_sweep(cfg, [q], cfg.collision_link_names)
    n_sph = 0
    for ln, s in per_wp.items():
        for c in np.asarray(s, float)[0]:                    # (S,4)：取唯一路点
            cx, cy, cz, r = (float(v) for v in c)
            if r <= 1e-4:
                continue
            ball = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=8)
            ball.translate((cx, cy, cz))
            if solid_spheres:
                ball.compute_vertex_normals()
                ball.paint_uniform_color([0.85, 0.1, 0.1])
                geoms.append(ball)
            else:
                ls = o3d.geometry.LineSet.create_from_triangle_mesh(ball)
                ls.paint_uniform_color([0.85, 0.1, 0.1])
                geoms.append(ls)
            n_sph += 1

    # —— init_free 立方体盒：base 系轴对齐 [box_min, box_max]，青色线框 ——
    lo = np.asarray(cfg.init_free_box_min, float)
    hi = np.asarray(cfg.init_free_box_max, float)
    aabb = o3d.geometry.AxisAlignedBoundingBox(lo.tolist(), hi.tolist())
    aabb.color = (0.0, 0.75, 0.75)
    geoms.append(aabb)

    # —— base 坐标架（原点，0.3m）——
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=(0.0, 0.0, 0.0))
    geoms.append(frame)

    stat = dict(n_spheres=n_sph, box_min=lo.tolist(), box_max=hi.tolist(),
                box_size=(hi - lo).tolist(), box_center=((lo + hi) / 2.0).tolist())
    return geoms, stat


def show_init_space(cfg, q=None, solid_spheres=False):
    """开窗显示「整臂碰撞球 + init_free 立方体盒」（关闭窗口结束）。"""
    import open3d as o3d
    geoms, stat = init_space_geometries(cfg, q=q, solid_spheres=solid_spheres)
    print(f"碰撞球   : {stat['n_spheres']} 个（红色线框）")
    print(f"init 盒  : min={np.round(stat['box_min'], 3)} max={np.round(stat['box_max'], 3)} "
          f"尺寸={np.round(stat['box_size'], 3)}m 中心={np.round(stat['box_center'], 3)}（青色线框）")
    print("显示中（关闭窗口结束）…")
    o3d.visualization.draw_geometries(geoms, window_name="init pose: 碰撞球 + init_free box")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", nargs="+", type=float, default=None,
                    help="构型(rad)；缺省用 cfg.retract_config")
    ap.add_argument("--solid", action="store_true", help="碰撞球画实心球（默认红色线框）")
    args = ap.parse_args()

    from gt_gen.config import load_config
    cfg = load_config()
    show_init_space(cfg, q=args.q, solid_spheres=args.solid)


if __name__ == "__main__":
    main()
