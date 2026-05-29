"""M1.3 2D 视觉验证: 每个候选 viewpoint 渲染**真实 RGB 照片**, 文件名说明合不合格.

每张照片:
    Open3D OffscreenRenderer 渲的工件 + 焊缝场景 (RGB)
        灰色 mesh    工件
        品红粗线段   焊缝主体
        绿色大球     焊缝起点
        红色大球     焊缝终点
        橙色球       P_weld

文件名 (中文, 列所有失败原因):
    valid:    00合格_010_距0.42米_角53度.png
    distance: 01不合格_001_距太近0.21米.png
    multi:    01不合格_023_距太远0.79米_视线被挡_角度太小12度.png

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/visualize_observation_2d.py \
        --obj "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part.obj" \
        --pkl "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_0.pkl"

默认 PNG 存到项目下 tmp/m1_3_observations/ (gitignored).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 找系统装的 CJK 字体, 用真实字体名 (不是搜索关键词)
def _find_cjk_font():
    """按优先级返回第一个找到的 CJK 字体. 优先 Noto/思源/微软雅黑 (含 ASCII)."""
    candidates = [f.name for f in fm.fontManager.ttflist]
    # 优先级: 字体名含这些关键词, 越靠前越先选
    priority = ["noto sans cjk", "source han", "yahei", "simhei", "wenquanyi",
                "han sans", "ming", "song", "fallback"]
    for kw in priority:
        for name in candidates:
            if kw in name.lower():
                return name
    return None

_cjk = _find_cjk_font()
if _cjk:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [_cjk] + list(plt.rcParams["font.sans-serif"])
    plt.rcParams["axes.unicode_minus"] = False
    print(f"[fonts] CJK = {_cjk}")
else:
    print("[fonts] WARNING: 没找到 CJK 字体, 中文会显示为方块")

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import torch

from phase1.depth_source import CameraIntrinsics, RaycastSim
from phase1.mapping import VoxelMap
from phase1.observation_check import (
    Viewpoint,
    check_distance,
    check_in_frustum,
    check_incidence,
    check_line_of_sight,
)
from phase1.tests._viz_helpers import load_seam_data, look_at


# ---------------------------------------------------------------------- 场景搭建


def build_offscreen_scene(
    obj_path: str, seam, width: int, height: int,
) -> rendering.OffscreenRenderer:
    """初始化 OffscreenRenderer, 加进 mesh + 焊缝可视化."""
    r = rendering.OffscreenRenderer(width, height)

    # 背景 (浅灰白)
    r.scene.set_background([0.92, 0.93, 0.95, 1.0])

    # 主光: 一束 directional + 环境光
    r.scene.scene.set_indirect_light_intensity(30000)
    r.scene.scene.set_sun_light(
        direction=[-0.4, -0.6, -0.8], color=[1.0, 1.0, 1.0], intensity=80000,
    )
    r.scene.scene.enable_sun_light(True)

    # ---- 1. 工件 mesh (灰色) ----
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()
    mat_mesh = rendering.MaterialRecord()
    mat_mesh.shader = "defaultLit"
    mat_mesh.base_color = [0.65, 0.65, 0.68, 1.0]
    r.scene.add_geometry("mesh", mesh, mat_mesh)

    # ---- 2. 焊缝: 一条品红圆柱沿采样点串起来 (放在焊缝表面) ----
    pts = seam.seam_line.astype(np.float64)
    # 圆柱半径根据焊缝长度自适应; 最小 3mm, 长焊缝可达 1cm
    seg_radius = float(np.clip(seam.seam_length * 0.25, 0.003, 0.010))
    for i in range(len(pts) - 1):
        cyl = _make_cylinder_between(pts[i], pts[i + 1], radius=seg_radius)
        if cyl is None:
            continue
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.base_color = [1.0, 0.05, 0.55, 1.0]    # 品红
        r.scene.add_geometry(f"seam_seg_{i}", cyl, mat)

    # 端点用小球封口, 让圆柱不至于看起来截断
    end_radius = seg_radius * 1.1
    def _add_endcap(name, center, color):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=end_radius, resolution=14)
        s.translate(center); s.compute_vertex_normals()
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.base_color = [*color, 1.0]
        r.scene.add_geometry(name, s, mat)
    _add_endcap("seam_cap_start", pts[0], (1.0, 0.05, 0.55))
    _add_endcap("seam_cap_end", pts[-1], (1.0, 0.05, 0.55))

    return r


def _make_cylinder_between(p0: np.ndarray, p1: np.ndarray, radius: float):
    """两点之间一段圆柱, 用作粗线段."""
    p0 = np.asarray(p0, dtype=np.float64); p1 = np.asarray(p1, dtype=np.float64)
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return None
    cyl = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=length, resolution=12, split=1,
    )
    cyl.compute_vertex_normals()
    # 默认 cylinder 沿 +z 高 length, 中心在原点
    z = np.array([0.0, 0.0, 1.0])
    axis = direction / length
    if abs(np.dot(z, axis)) > 0.999:
        # 平行 z, 不需要旋转 (或反向)
        if axis[2] < 0:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([np.pi, 0, 0]))
            cyl.rotate(R, center=(0, 0, 0))
    else:
        rot_axis = np.cross(z, axis); rot_axis /= np.linalg.norm(rot_axis)
        ang = float(np.arccos(np.clip(np.dot(z, axis), -1, 1)))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * ang)
        cyl.rotate(R, center=(0, 0, 0))
    cyl.translate((p0 + p1) / 2.0)
    return cyl


# ---------------------------------------------------------------------- 渲染


def render_one_rgb(
    renderer: rendering.OffscreenRenderer,
    intrin: CameraIntrinsics,
    camera_pose: np.ndarray,
) -> np.ndarray:
    """一帧 RGB. camera_pose: (4,4) world中相机位姿 (OpenCV: x右 y下 z前)."""
    K_mat = np.array([
        [intrin.fx, 0, intrin.cx],
        [0, intrin.fy, intrin.cy],
        [0, 0, 1],
    ], dtype=np.float64)
    # Open3D setup_camera 需要 extrinsic = world → camera
    extrinsic = np.linalg.inv(camera_pose).astype(np.float64)
    renderer.setup_camera(K_mat, extrinsic, intrin.width, intrin.height)
    img = renderer.render_to_image()
    return np.asarray(img)


def save_with_title(
    rgb: np.ndarray, out_path: Path, title: str, title_color: str,
    cam_pos_text: str,
) -> None:
    """加标题 + 相机位置, 保存."""
    fig, ax = plt.subplots(figsize=(rgb.shape[1] / 100, rgb.shape[0] / 100 + 0.6),
                           dpi=100)
    ax.imshow(rgb)
    ax.set_title(title, color=title_color, fontweight="bold", fontsize=11)
    ax.text(0.99, 0.02, cam_pos_text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="white",
            bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"))
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--pkl", required=True)
    _default_out = (Path(__file__).resolve().parent.parent.parent
                    / "tmp" / "m1_3_observations")
    parser.add_argument("--output-dir", default=str(_default_out))
    parser.add_argument("--n-candidates", type=int, default=50)
    parser.add_argument("--radius-min", type=float, default=0.15)
    parser.add_argument("--radius-max", type=float, default=0.85)
    parser.add_argument("--d-min", type=float, default=0.30)
    parser.add_argument("--d-max", type=float, default=0.60)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--mode", choices=["strict", "optimistic"], default="optimistic")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    K = CameraIntrinsics.from_fov(86.0, 57.0, args.height, args.width)

    # M1.1 RaycastSim 仅用来建 voxel map (M1.3 LOS 检查用), 不用来出图
    sim = RaycastSim(mesh_paths=[args.obj], intrinsics=K, device=device)
    print(f"[load] {sim.num_triangles} triangles")

    seam = load_seam_data(args.pkl)
    print(f"[seam] {seam.label}")

    # 预建 voxel map (8 视角)
    bounds = np.stack([sim.mesh.bounds[0] - 0.6, sim.mesh.bounds[1] + 0.6])
    vm = VoxelMap(bounds=bounds, resolution=0.02, device=device, max_range=2.0)
    for i in range(8):
        ang = 2 * np.pi * i / 8
        cp = seam.p_weld + np.array([0.5 * np.cos(ang), 0.5 * np.sin(ang), 0.3])
        d = sim.render(look_at(cp, seam.p_weld))
        vm.integrate(d, K, look_at(cp, seam.p_weld))

    # ---- Offscreen RGB 渲染器 (一次创建反复用) ----
    print("[init] creating Open3D OffscreenRenderer...")
    t0 = time.time()
    renderer = build_offscreen_scene(args.obj, seam, args.width, args.height)
    print(f"  done in {time.time() - t0:.2f} s")

    # 候选位置: 一半合规距离 (确保有 valid), 一半全范围
    half = args.n_candidates // 2
    rest = args.n_candidates - half
    radii = np.concatenate([
        rng.uniform(args.d_min, args.d_max, half),
        rng.uniform(args.radius_min, args.radius_max, rest),
    ])
    rng.shuffle(radii)
    u01 = rng.uniform(0, 1, args.n_candidates)
    v01 = rng.uniform(0, 1, args.n_candidates)
    theta = np.arccos(1 - u01)
    phi = 2 * np.pi * v01
    offsets = np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], axis=1) * radii[:, None]
    positions = seam.p_weld + offsets

    out_dir = Path(args.output_dir)
    if args.clear and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_dir}")

    PREFIX = {
        None:           "00valid",
        "distance":     "01invalid_distance",
        "frustum":      "02invalid_frustum",
        "line_of_sight":"03invalid_los",
        "incidence":    "04invalid_incidence",
    }

    counter = Counter()
    t_render = 0.0
    for i, pos in enumerate(positions):
        cam_dir = (seam.p_weld - pos)
        cam_dir /= np.linalg.norm(cam_dir)
        vp = Viewpoint(pos=pos, dir=cam_dir, up=np.array([0, 0, 1.0]))
        cam_pose = look_at(pos, seam.p_weld)

        # 跑全部 4 条检查 (不 short-circuit, 收集所有失败)
        dist_ok, dist = check_distance(vp, seam.p_weld, args.d_min, args.d_max)
        frustum_ok = check_in_frustum(vp, seam.p_weld, K)
        los_ok = check_line_of_sight(
            vp, seam.p_weld, vm,
            unknown_as_block=(args.mode == "strict"),
        )
        inc_ok, inc_angle = check_incidence(
            vp, seam.p_weld, seam.tangent, 30.0, 90.0,
        )

        # 收集失败原因 (中文, 详细)
        fail_reasons_zh = []
        fail_tags = []          # 文件名用的简短标签
        if not dist_ok:
            if dist < args.d_min:
                fail_reasons_zh.append(f"距离太近({dist:.3f}米, <{args.d_min}米)")
                fail_tags.append(f"距太近{dist:.2f}米")
            elif dist > args.d_max:
                fail_reasons_zh.append(f"距离太远({dist:.3f}米, >{args.d_max}米)")
                fail_tags.append(f"距太远{dist:.2f}米")
            else:
                fail_reasons_zh.append(f"距离不合规({dist:.3f}米)")
                fail_tags.append(f"距异常{dist:.2f}米")
        if not frustum_ok:
            fail_reasons_zh.append("焊缝中点不在相机视锥内")
            fail_tags.append("视锥外")
        if not los_ok:
            fail_reasons_zh.append("视线被遮挡(中间有障碍或穿过未知区)")
            fail_tags.append("视线被挡")
        if not inc_ok:
            if inc_angle < 30.0:
                fail_reasons_zh.append(f"入射角太小({inc_angle:.1f}度, <30度)")
                fail_tags.append(f"角度太小{inc_angle:.0f}度")
            else:
                fail_reasons_zh.append(f"入射角不合规({inc_angle:.1f}度)")
                fail_tags.append(f"角度异常{inc_angle:.0f}度")

        is_valid = len(fail_reasons_zh) == 0
        first_fail = None if is_valid else (
            "distance" if not dist_ok else
            "frustum" if not frustum_ok else
            "line_of_sight" if not los_ok else
            "incidence"
        )
        counter[first_fail] += 1

        # 渲 RGB
        t1 = time.time()
        rgb = render_one_rgb(renderer, K, cam_pose)
        t_render += time.time() - t1

        # 标题 (中文)
        if is_valid:
            title = (f"候选 #{i:03d}  ✓ 合格    "
                     f"距离 = {dist:.3f}米   入射角 = {inc_angle:.1f}度")
            color = "darkgreen"
        else:
            n = len(fail_reasons_zh)
            reasons_str = "  /  ".join(fail_reasons_zh)
            title = (f"候选 #{i:03d}  ✗ 不合格 ({n} 条)\n{reasons_str}")
            color = "darkred"
        cam_text = f"相机世界坐标: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"

        # 文件名 (中文, 列所有失败 tag)
        if is_valid:
            fname = f"00合格_{i:03d}_距{dist:.2f}米_角{inc_angle:.0f}度.png"
        else:
            tags_str = "_".join(fail_tags)
            fname = f"01不合格_{i:03d}_{tags_str}.png"

        save_with_title(rgb, out_dir / fname, title, color, cam_text)

        if (i + 1) % 10 == 0 or i == len(positions) - 1:
            print(f"  rendered {i+1}/{len(positions)}  "
                  f"(avg {t_render/(i+1)*1000:.0f} ms/frame)")

    print()
    print(f"[stats]  mode = {args.mode}  (按首次失败的条目分类)")
    for reason in [None, "distance", "frustum", "line_of_sight", "incidence"]:
        n = counter.get(reason, 0)
        zh = {None: "✓ 合格", "distance": "✗ 距离", "frustum": "✗ 视锥",
              "line_of_sight": "✗ 视线", "incidence": "✗ 入射角"}[reason]
        print(f"  {zh:<10s} {n}")

    print()
    print(f"[done] {len(positions)} RGB PNGs to {out_dir}")
    print(f"  xdg-open {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
