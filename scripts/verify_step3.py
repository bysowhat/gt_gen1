"""Step 3 验证：传感器模拟（raycast 真值场景）。

合成可预测的场景核对 raycast：相机在 base 原点、沿 +Z 看一面墙(z=D 平面)。
- 大墙全命中：所有 occ 点 z≈D、所有 free 点 z<D、都在 max_depth 内；
- max_depth<D：无命中，全 free 且不超过 max_depth；
- 小墙部分命中：occ 数 ∈ (0, 射线数)，漏检射线 free 到 max_depth。
另：camera_pose_from_config 的 FK+手眼外参链路（相机相对 Link6 偏移量核对）。

运行：conda run -n env_isaaclab python scripts/verify_step3.py [--viz]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def wall(D, extents_xy=4.0, thick=0.05):
    """z=D 处一面垂直于视线的墙(thin slab)，近面在 z=D-thick/2。"""
    import trimesh
    m = trimesh.creation.box(extents=(extents_xy, extents_xy, thick))
    m.apply_translation([0.0, 0.0, D])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", action="store_true")
    args = ap.parse_args()

    from gt_gen.config import load_config
    from gt_gen import sensor

    cfg = load_config()
    cm = sensor.load_camera_model(cfg)
    print("== camera_model ==")
    print(f"fx={cm['fx']} fy={cm['fy']} cx={cm['cx']} cy={cm['cy']} "
          f"{cm['width']}x{cm['height']} max_depth={cm['max_depth']}")

    eye = np.eye(4)            # 相机=base 系，+Z 视线
    stride = 32
    D = 1.0
    near_face = D - 0.025      # slab 近面

    # 1) 大墙全命中
    print("\n== 1) 大墙全命中 ==")
    w_big = wall(D, extents_xy=6.0)
    free, occ = sensor.raycast_observe(eye, cm, w_big, cm["max_depth"], pixel_stride=stride)
    nray = sensor.pixel_ray_dirs(cm, stride).shape[0]
    print(f"射线数={nray} free={free.shape[0]} occ={occ.shape[0]}")
    assert occ.shape[0] == nray, "墙够大，每条射线都应命中"
    assert np.all(np.abs(occ[:, 2] - near_face) < 1e-3), f"occ 应贴近面 z={near_face}"
    assert free.shape[0] > 0 and np.all(free[:, 2] < near_face), "free 应都在命中点之前"
    # 所有点都在 max_depth 范围内
    assert np.all(np.linalg.norm(occ, axis=1) <= cm["max_depth"] + 1e-6)
    assert np.all(np.linalg.norm(free, axis=1) <= cm["max_depth"] + 1e-6)
    print("occ z 范围:", np.round([occ[:, 2].min(), occ[:, 2].max()], 4),
          " free z 上界:", round(float(free[:, 2].max()), 4))

    # 2) max_depth < D：截断，无命中
    print("\n== 2) max_depth 截断 ==")
    free2, occ2 = sensor.raycast_observe(eye, cm, w_big, max_depth=0.5, pixel_stride=stride)
    print(f"free={free2.shape[0]} occ={occ2.shape[0]}")
    assert occ2.shape[0] == 0, "墙在 0.975m，max_depth=0.5 应无命中"
    assert free2.shape[0] > 0 and free2[:, 2].max() <= 0.5 + 1e-6, "free 不超过 max_depth"

    # 3) 小墙部分命中
    print("\n== 3) 小墙部分命中 ==")
    w_small = wall(D, extents_xy=0.5)
    free3, occ3 = sensor.raycast_observe(eye, cm, w_small, cm["max_depth"], pixel_stride=stride)
    print(f"射线数={nray} free={free3.shape[0]} occ={occ3.shape[0]}")
    assert 0 < occ3.shape[0] < nray, "应只有中心一束射线命中小墙"
    assert np.all(np.abs(occ3[:, 2] - near_face) < 1e-3)
    # 漏检射线 free 应延伸到接近 max_depth（远超近面）
    assert free3[:, 2].max() > near_face, "漏检射线应 free 到远处"

    # 4) camera_pose_from_config：FK + 手眼外参
    print("\n== 4) camera_pose_from_config（FK+外参）==")
    kin = sensor.build_kinematics(cfg)
    q = cfg.retract_config
    T = sensor.camera_pose_from_config(kin, q, cm)
    R, p = T[:3, :3], T[:3, 3]
    l6_pos, _ = sensor.link6_pose(kin, q)
    print("相机位置(base):", np.round(p, 4), " 视线 +Z:", np.round(T[:3, 2], 4))
    print("Link6 位置(base):", np.round(l6_pos, 4))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5), "旋转应正交"
    assert abs(np.linalg.det(R) - 1.0) < 1e-5, "右手系 det=1"
    off = np.linalg.norm(p - l6_pos)
    exp_off = np.linalg.norm(cm["extrinsic_pos"])
    print(f"相机相对 Link6 偏移 |Δ|={off:.4f}  外参模长={exp_off:.4f}")
    assert abs(off - exp_off) < 1e-4, "相机应距 Link6 恰为外参平移模长"

    if args.viz:
        from gt_gen import viz
        print("\n打开 open3d：大墙全命中场景（灰=墙 绿=free 红=occ 轴=相机/base）")
        viz.show_rays(eye, free, occ, truth_scene=w_big)

    print("\nSTEP3_OK")


if __name__ == "__main__":
    main()
