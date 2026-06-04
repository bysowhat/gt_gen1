"""可视化辅助（穿插在各步验证中使用）。"""
from __future__ import annotations

import numpy as np


def _voxmap_geometries(voxmap, draw_free=True, draw_unknown=False):
    """把三态体素图转成 open3d 几何：一个彩色 VoxelGrid（真立方体）+ 基座坐标轴。

    OCCUPIED 红、FREE 绿、UNKNOWN 灰。UNKNOWN 默认不画（整图会糊满）。
    """
    import open3d as o3d
    from gt_gen.voxmap import UNKNOWN, FREE, OCCUPIED

    layers = [(OCCUPIED, [0.9, 0.15, 0.15])]
    if draw_free:
        layers.append((FREE, [0.2, 0.7, 0.25]))
    if draw_unknown:
        layers.append((UNKNOWN, [0.6, 0.6, 0.6]))

    pts, cols = [], []
    for state, color in layers:
        p = voxmap.state_centers(state)
        if p.shape[0]:
            pts.append(p)
            cols.append(np.tile(color, (p.shape[0], 1)))

    geoms = []
    if pts:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts)))
        pc.colors = o3d.utility.Vector3dVector(np.concatenate(cols))
        # 用体素中心点云生成 VoxelGrid → 每个非空体素显示为一个实心立方体
        vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=voxmap.voxel_size)
        geoms.append(vg)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))
    return geoms


def show_voxmap(voxmap, draw_free=True, draw_unknown=False, save=None,
                window_name="voxmap (R=OCCUPIED G=FREE)", width=1280, height=960):
    """交互式 3D 显示三态体素图（open3d 窗口，可鼠标旋转/缩放/平移）。

    OCCUPIED 红、FREE 绿、UNKNOWN 灰(默认不画)。
    save=路径 时改为离屏渲染存 png（无显示器场景），否则弹交互窗口。
    """
    import open3d as o3d
    geoms = _voxmap_geometries(voxmap, draw_free=draw_free, draw_unknown=draw_unknown)

    if save is not None:
        import open3d.visualization.rendering as rendering
        renderer = rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([1, 1, 1, 1])
        for i, g in enumerate(geoms):
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit" if isinstance(g, o3d.geometry.TriangleMesh) else "defaultUnlit"
            renderer.scene.add_geometry(f"g{i}", g, mat)
        ctr = voxmap.center
        renderer.setup_camera(55.0, ctr, ctr + np.array([1.8, -1.8, 1.2]), np.array([0, 0, 1.0]))
        o3d.io.write_image(save, renderer.render_to_image())
        return save

    o3d.visualization.draw_geometries(geoms, window_name=window_name,
                                      width=width, height=height)
    return None


def show_rays(camera_pose, free_points, occ_points, truth_scene=None,
              max_free=20000, save=None, window_name="raycast (G=free R=occ)",
              width=1280, height=960):
    """交互式 3D 显示一次 raycast：真值 mesh(灰) + free 点(绿) + occ 点(红) + 相机坐标轴。

    free 点很多时随机下采样到 max_free。save=路径 则离屏存 png，否则弹交互窗口。
    """
    import open3d as o3d

    geoms = []
    if truth_scene is not None:
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(truth_scene.vertices)),
            o3d.utility.Vector3iVector(np.asarray(truth_scene.faces)))
        m.paint_uniform_color([0.7, 0.7, 0.7])
        m.compute_vertex_normals()
        geoms.append(m)

    fp = np.asarray(free_points)
    if fp.shape[0] > max_free:
        sel = np.linspace(0, fp.shape[0] - 1, max_free).astype(int)
        fp = fp[sel]
    if fp.shape[0]:
        pcf = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fp))
        pcf.paint_uniform_color([0.2, 0.7, 0.25])
        geoms.append(pcf)

    op = np.asarray(occ_points)
    if op.shape[0]:
        pco = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(op))
        pco.paint_uniform_color([0.9, 0.15, 0.15])
        geoms.append(pco)

    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    cam_frame.transform(np.asarray(camera_pose))
    geoms.append(cam_frame)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))  # base

    if save is not None:
        import open3d.visualization.rendering as rendering
        renderer = rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([1, 1, 1, 1])
        for i, g in enumerate(geoms):
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit" if isinstance(g, o3d.geometry.TriangleMesh) else "defaultUnlit"
            mat.point_size = 4.0
            renderer.scene.add_geometry(f"g{i}", g, mat)
        ctr = np.asarray(camera_pose)[:3, 3]
        renderer.setup_camera(60.0, ctr + np.array([0, 0, 0.6]), ctr + np.array([1.2, -1.2, 0.4]),
                              np.array([0, 0, 1.0]))
        o3d.io.write_image(save, renderer.render_to_image())
        return save

    o3d.visualization.draw_geometries(geoms, window_name=window_name, width=width, height=height)
    return None


def show_candidates(B, candidates, **kw):
    raise NotImplementedError("viz")
