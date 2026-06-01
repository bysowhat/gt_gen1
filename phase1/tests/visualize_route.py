"""可视化 batch_eval.py 保存的单条焊缝规划路线 (.npz).

灰 mesh | 品红焊缝(绿=start,红=end) | 橙 P_weld | 黑 起点 | 红折线+蓝点 EE路线
| 黄/紫 终点(成功黄/失败紫) | 青线 wedge中线 | 紫小球 探索图节点

跑法:
  /home/a/miniforge3/envs/env_isaaclab/bin/python phase1/tests/visualize_route.py \\
      tmp/batch_eval/routes/BEAM_..._part__seam_24.npz
  # 不带参数 = 列出所有路线 + 成功率
"""
from __future__ import annotations

import sys
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

ROUTEDIR = Path("/home/a/Projects/Github/gt_gen_hanfeng/tmp/batch_eval/routes")


def _ray(o, d, color, length, o3d):
    d = np.asarray(d, float); d = d / (np.linalg.norm(d) + 1e-12)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.vstack([o, o + d * length]))
    ls.lines = o3d.utility.Vector2iVector([[0, 1]])
    ls.colors = o3d.utility.Vector3dVector([color])
    return ls


def visualize(npz_path: str, no_window: bool = False):
    import open3d as o3d
    from phase1.observation_check import wedge_bisector
    d = np.load(npz_path, allow_pickle=True)
    part = str(d["part"]); seam = str(d["seam"]); obj = str(d["obj"])
    success = bool(d["success"]); ee = d["ee_path"]; pw = d["p_weld"]
    sline = d["seam_line"]; slim = d["seam_limits"]; mid = int(d["p_weld_idx"])
    tan = d["seam_tangent"][mid]
    print(f"{part}/{seam}: success={success} switch@{int(d['switch_at'])} "
          f"cyc={int(d['cycles'])} observed={d['observed'].tolist()[:6]} "
          f"fail={str(d['fail_reason'])}")
    print(f"EE 路线 {len(ee)} 步; 起点 d2P="
          f"{np.linalg.norm(ee[0]-pw):.2f} 终点 d2P={np.linalg.norm(ee[-1]-pw):.2f}"
          if len(ee) else "无路线")

    geoms = []
    mesh = o3d.io.read_triangle_mesh(obj)
    mesh.compute_vertex_normals(); mesh.paint_uniform_color([0.72, 0.72, 0.72])
    geoms.append(mesh)

    # 焊缝圆柱线 + 起终点球
    for i in range(len(sline) - 1):
        seg = sline[i+1] - sline[i]; L = np.linalg.norm(seg)
        if L < 1e-6: continue
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.004, height=L)
        z = np.array([0, 0, 1.0]); ax = np.cross(z, seg/L); s = np.linalg.norm(ax)
        if s > 1e-6:
            ang = np.arccos(np.clip(z@(seg/L), -1, 1))
            cyl.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(ax/s*ang), center=(0,0,0))
        cyl.translate((sline[i]+sline[i+1])/2); cyl.paint_uniform_color([1.0,0.0,1.0])
        cyl.compute_vertex_normals(); geoms.append(cyl)
    for p, c in [(sline[0], [0.1,1.0,0.1]), (sline[-1], [1.0,0.1,0.1]), (pw, [1.0,0.55,0.0])]:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.012); sph.translate(p)
        sph.paint_uniform_color(c); sph.compute_vertex_normals(); geoms.append(sph)

    if len(ee):
        # 起点黑
        s0 = o3d.geometry.TriangleMesh.create_sphere(radius=0.05); s0.translate(ee[0])
        s0.paint_uniform_color([0,0,0]); s0.compute_vertex_normals(); geoms.append(s0)
        # 路线红折线 + 蓝点
        if len(ee) >= 2:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(ee)
            ls.lines = o3d.utility.Vector2iVector([[i,i+1] for i in range(len(ee)-1)])
            ls.colors = o3d.utility.Vector3dVector([[1,0.2,0.2]]*(len(ee)-1))
            geoms.append(ls)
            for p in ee[1:-1]:
                sp = o3d.geometry.TriangleMesh.create_sphere(radius=0.018); sp.translate(p)
                sp.paint_uniform_color([0.2,0.5,1.0]); sp.compute_vertex_normals(); geoms.append(sp)
        # 终点 黄(成功)/紫(失败)
        se = o3d.geometry.TriangleMesh.create_sphere(radius=0.05); se.translate(ee[-1])
        se.paint_uniform_color([1,1,0] if success else [0.5,0,0.5])
        se.compute_vertex_normals(); geoms.append(se)

    # 每个路点画相机坐标系 (从保存的 4×4 位姿), 直观看完整路径的位置+朝向
    ee_poses = d["ee_poses"] if "ee_poses" in d.files else np.zeros((0,4,4))
    for T in ee_poses:
        fr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
        fr.transform(T); geoms.append(fr)
    # 终点观测视锥 (相机 z=光轴), 直观看"看向哪"
    if len(ee_poses) and "obs_d" in d.files:
        T = ee_poses[-1]; dmax = float(d["obs_d"][1])
        fov = d["cam_fov"] if "cam_fov" in d.files else np.array([86.0,57.0])
        import math
        hh = math.tan(math.radians(fov[0]/2))*dmax
        vh = math.tan(math.radians(fov[1]/2))*dmax
        corners_cam = np.array([[0,0,0],[ hh, vh,dmax],[-hh, vh,dmax],
                                [-hh,-vh,dmax],[ hh,-vh,dmax]])
        cw = (T[:3,:3]@corners_cam.T).T + T[:3,3]
        fr = o3d.geometry.LineSet()
        fr.points = o3d.utility.Vector3dVector(cw)
        fr.lines = o3d.utility.Vector2iVector([[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]])
        fr.colors = o3d.utility.Vector3dVector([[1,1,0] if success else [0.5,0,0.5]]*8)
        geoms.append(fr)

    # wedge 中线 (青)
    bis = wedge_bisector(slim[mid], tangent=tan)
    if bis is not None:
        geoms.append(_ray(pw, bis, [0,0.9,0.9], 0.5, o3d))
        sl = slim[mid]
        geoms.append(_ray(pw, sl[0], [0.6,0.6,0.6], 0.4, o3d))
        geoms.append(_ray(pw, sl[1], [0.6,0.6,0.6], 0.4, o3d))

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=pw))
    if no_window:
        return

    # 用 O3DVisualizer (GUI) 才能加 3D 文字标注每步步数
    n_ee = len(ee)
    try:
        gui = o3d.visualization.gui.Application.instance
        gui.initialize()
        vis = o3d.visualization.O3DVisualizer(
            f"{part}/{seam}  {'OK' if success else 'FAIL'}  "
            f"({n_ee}步, switch@{int(d['switch_at'])})", 1280, 720)
        vis.show_settings = True
        for i, g in enumerate(geoms):
            vis.add_geometry(f"g{i}", g)
        # 每个路点标步数 (0=起点, 末=终点)
        for i, p in enumerate(ee):
            if i == 0:
                txt = "0 起点"
            elif i == n_ee - 1:
                txt = f"{i} {'观测' if success else '终止'}"
            else:
                txt = str(i)
            vis.add_3d_label(p + np.array([0, 0, 0.03]), txt)
        vis.reset_camera_to_default()
        gui.add_window(vis)
        gui.run()
    except Exception as e:
        print(f"[警告] O3DVisualizer 标注模式失败 ({e}), 退回无标注 draw_geometries")
        o3d.visualization.draw_geometries(
            geoms, window_name=f"{part}/{seam} {'OK' if success else 'FAIL'}",
            width=1280, height=720)


def list_all():
    files = sorted(glob.glob(str(ROUTEDIR / "*.npz")))
    if not files:
        print("还没有路线结果. 先跑 batch_eval.py"); return
    n_ok = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        ok = bool(d["success"]); n_ok += ok
        print(f'  {"OK " if ok else "FAIL"} {Path(f).stem}  switch@{int(d["switch_at"])}')
    print(f"\n成功率 {n_ok}/{len(files)} = {100*n_ok/len(files):.0f}%")
    print("可视化某条: python phase1/tests/visualize_route.py <上面某个 .npz 路径>")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--no-window"]
    nw = "--no-window" in sys.argv
    if not args:
        list_all()
    else:
        p = args[0]
        if not p.endswith(".npz"):
            p = str(ROUTEDIR / f"{p}.npz")
        visualize(p, no_window=nw)
