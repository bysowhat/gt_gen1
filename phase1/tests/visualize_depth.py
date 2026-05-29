"""M1.1 视觉验证: 用 BEAM 工件渲染一张深度图, 保存 PNG.

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/visualize_depth.py \
        --obj "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part.obj" \
        --pkl "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_0.pkl" \
        --output /tmp/raycast_depth.png

期望: PNG 里能看到 BEAM 工件的轮廓 (灰度/伪彩深度).
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

# 确保能 from phase1.xxx import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from phase1.depth_source import CameraIntrinsics, RaycastSim


def pose7_to_mat4(pose7) -> np.ndarray:
    """[x,y,z, qw,qx,qy,qz] (7,) → (4,4). 四元数顺序 [w, x, y, z]."""
    p = np.asarray(pose7).reshape(-1)
    pos = p[:3]
    qw, qx, qy, qz = p[3:7]
    R = np.array(
        [[1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
         [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
         [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def look_at(cam_pos: np.ndarray, target: np.ndarray, up_hint=np.array([0, 0, 1])) -> np.ndarray:
    """构造相机位姿矩阵, z 轴指向 target.
    OpenCV/Isaac Sim 约定: x 右, y 下, z 前.
    """
    z_axis = target - cam_pos
    z_axis = z_axis / np.linalg.norm(z_axis)
    # up 和 z 平行时换一个
    if abs(np.dot(z_axis, up_hint)) > 0.99:
        up_hint = np.array([0, 1, 0]) if abs(z_axis[1]) < 0.99 else np.array([1, 0, 0])
    x_axis = np.cross(up_hint, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cam_pos
    return T


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--output", default="/tmp/raycast_depth.png")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--cam-offset-z", type=float, default=0.5,
                        help="相机放在 P_weld 上方多少米")
    args = parser.parse_args()

    print(f"[load] mesh {args.obj}")
    K = CameraIntrinsics.from_fov(86.0, 57.0, args.height, args.width)
    sim = RaycastSim(mesh_paths=[args.obj], intrinsics=K)
    print(f"  triangles: {sim.num_triangles}")
    print(f"  intrinsics: fx={K.fx:.1f}, fy={K.fy:.1f}, cx={K.cx:.1f}, cy={K.cy:.1f}")
    print(f"  image: {K.width}x{K.height}")

    with open(args.pkl, "rb") as f:
        pkl = pickle.load(f)
    seam_line = np.asarray(pkl["seam_line"])
    seam_mid = seam_line[len(seam_line) // 2]
    print(f"[scene] seam mid point (world) = {seam_mid.round(3)}")

    cam_pos = seam_mid + np.array([0.3, 0.3, args.cam_offset_z])  # 斜上方
    cam_pose = look_at(cam_pos, seam_mid)
    print(f"[camera] world pos = {cam_pos.round(3)}, looking at seam_mid")

    sim.render(cam_pose)  # warmup
    t0 = time.time()
    depth = sim.render(cam_pose)
    elapsed_ms = (time.time() - t0) * 1000

    depth_np = depth.cpu().numpy()
    valid = depth_np > 0
    n_hit = int(valid.sum())
    n_total = depth_np.size
    if n_hit:
        d_mean = float(depth_np[valid].mean())
        d_min = float(depth_np[valid].min())
        d_max = float(depth_np[valid].max())
    else:
        d_mean = d_min = d_max = 0.0
    print(
        f"[output] hits = {n_hit}/{n_total}, "
        f"depth = [{d_min:.3f}, {d_max:.3f}], mean = {d_mean:.3f} m"
    )
    print(f"[time] {elapsed_ms:.1f} ms/frame")

    # 出图
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    if n_hit:
        ax.imshow(np.where(valid, depth_np, np.nan), cmap="viridis", vmin=d_min, vmax=d_max)
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(cmap="viridis", norm=Normalize(vmin=d_min, vmax=d_max))
        plt.colorbar(sm, ax=ax, label="depth (m)")
    else:
        ax.imshow(depth_np, cmap="viridis")
    ax.set_title(
        f"RaycastSim depth\n"
        f"cam world {cam_pos.round(3)} → seam_mid {seam_mid.round(3)}\n"
        f"hits {n_hit}/{n_total}, {elapsed_ms:.1f} ms"
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=100, bbox_inches="tight")
    print(f"[saved] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
