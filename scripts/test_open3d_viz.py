"""测试 open3d 在 VNC 虚拟 DISPLAY 上的可视化(方案一)。

用法:
    # 1. 先确认 VNC 的 DISPLAY 号(浏览器 VNC 终端里 echo $DISPLAY,一般是 :1)
    # 2. 让本进程用该 DISPLAY:
    export DISPLAY=:1
    # 若 GLX 报错,强制软件渲染:
    export LIBGL_ALWAYS_SOFTWARE=1
    python scripts/test_open3d_viz.py

跑通后,浏览器里的 VNC 桌面(ip:4000)应弹出一个窗口,里面有坐标轴、
一个红色球和一堆随机彩色点。能用鼠标转动即成功。
"""

import os
import sys

import numpy as np
import open3d as o3d


def main() -> int:
    display = os.environ.get("DISPLAY")
    print(f"[test] DISPLAY = {display!r}")
    print(f"[test] LIBGL_ALWAYS_SOFTWARE = {os.environ.get('LIBGL_ALWAYS_SOFTWARE')!r}")
    print(f"[test] open3d version = {o3d.__version__}")

    if not display:
        print(
            "[test] 警告:DISPLAY 未设置。请先 `export DISPLAY=:1`(用 VNC 桌面里 "
            "echo $DISPLAY 得到的实际值),否则开窗会失败。",
            file=sys.stderr,
        )

    # 坐标轴
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    # 一个红色球
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
    sphere.translate((1.0, 0.0, 0.0))
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    sphere.compute_vertex_normals()

    # 一团随机彩色点云
    pts = np.random.default_rng(0).uniform(-1.0, 1.0, size=(3000, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector((pts + 1.0) / 2.0)

    print("[test] 正在打开可视化窗口……关闭窗口即结束。")
    o3d.visualization.draw_geometries(
        [axis, sphere, pcd],
        window_name="open3d VNC test",
        width=1024,
        height=768,
    )
    print("[test] 窗口已关闭,测试结束。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
