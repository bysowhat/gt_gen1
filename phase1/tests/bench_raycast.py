"""M1.1 性能 benchmark.

跑法:
    /home/a/miniforge3/envs/env_isaaclab/bin/python \
        /home/a/Projects/Github/gt_gen_hanfeng/phase1/tests/bench_raycast.py \
        --obj "/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part.obj"

期望:
    720p (1280×720) 单帧 < 15 ms; rays/sec > 50 M.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from phase1.depth_source import CameraIntrinsics, RaycastSim


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--n", type=int, default=20, help="frames to render")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()

    K = CameraIntrinsics.from_fov(86.0, 57.0, args.height, args.width)
    sim = RaycastSim(mesh_paths=[args.obj], intrinsics=K)
    print(f"[load] {sim.num_triangles} triangles")
    print(f"[image] {K.width}x{K.height}  ({K.width*K.height} rays/frame)")

    # 一个能命中 mesh 的 pose: 站在 mesh 上方 (粗略估)
    bbox = sim.mesh.bounds  # (2, 3)
    center = bbox.mean(axis=0)
    extent = (bbox[1] - bbox[0]).max()
    cam_pos = center + np.array([0, 0, extent * 1.5])
    z_axis = center - cam_pos; z_axis /= np.linalg.norm(z_axis)
    up = np.array([0, 1, 0])  # 用 y 当 up_hint
    x_axis = np.cross(up, z_axis); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    pose[:3, 3] = cam_pos

    sim.render(pose)  # warmup

    t0 = time.time()
    for _ in range(args.n):
        sim.render(pose)
    elapsed_s = (time.time() - t0) / args.n
    n_rays = K.height * K.width
    print(f"[bench] {elapsed_s * 1000:.2f} ms/frame, "
          f"{n_rays / elapsed_s / 1e6:.1f} M rays/sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
