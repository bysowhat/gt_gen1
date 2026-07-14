"""Step 2 验证：三态体素地图数据结构（建图 / 坐标互转 / set-get / 掩码）。

运行：conda run -n env_isaaclab python scripts/verify_step2.py [--viz]
（纯 numpy，不依赖 cuRobo；--viz 需要 open3d 离屏渲染）
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", action="store_true", help="open3d 离屏渲染到 /tmp/voxmap.png")
    args = ap.parse_args()

    from gt_gen.config import load_config
    from gt_gen.voxmap import (
        ThreeStateVoxelMap, build_roi_voxmap, UNKNOWN, FREE, OCCUPIED,
    )

    # 1) 直接构造一个小图：origin=(-1,-1,0), 边长 2x2x1 m, 2cm 体素
    vm = ThreeStateVoxelMap(origin=(-1.0, -1.0, 0.0), size_xyz=(2.0, 2.0, 1.0), voxel_size=0.02)
    print("== 建图 ==")
    print("shape:", vm.shape, " num_voxels:", vm.num_voxels)
    print("origin:", vm.origin, " center:", np.round(vm.center, 4), " upper:", np.round(vm.upper, 4))
    assert vm.shape == (100, 100, 50), vm.shape
    assert np.allclose(vm.center, [0.0, 0.0, 0.5]), vm.center
    assert vm.counts()[UNKNOWN] == vm.num_voxels, "初始应全 UNKNOWN"

    # 2) 坐标互转
    print("\n== 坐标互转 ==")
    assert np.array_equal(vm.world_to_voxel(vm.origin), [0, 0, 0]), "origin 应落在体素 0"
    # voxel_to_world 返回中心 → 再转回原下标（round-trip）
    rng = np.random.default_rng(0)
    samples = rng.uniform(vm.origin + 1e-3, vm.upper - 1e-3, size=(2000, 3))
    idx = vm.world_to_voxel(samples)
    ctr = vm.voxel_to_world(idx)
    assert np.all(np.abs(samples - ctr) <= vm.voxel_size / 2 + 1e-9), "点应落在所属体素内"
    assert np.array_equal(vm.world_to_voxel(ctr), idx), "体素中心 round-trip 应一致"
    print("2000 随机点 round-trip OK（点∈所属体素、中心↔下标一致）")

    # 3) in_bounds / 越界
    print("\n== 边界 ==")
    assert bool(vm.in_bounds([0, 0, 0])) and bool(vm.in_bounds([99, 99, 49]))
    assert not bool(vm.in_bounds([100, 0, 0])) and not bool(vm.in_bounds([-1, 0, 0]))
    assert vm.get([100, 100, 100]) == UNKNOWN, "越界 get 应返回 UNKNOWN"
    n_set = vm.set_many([[200, 0, 0], [5, 5, 5]], OCCUPIED)
    assert n_set == 1, "越界 set 应被跳过，仅 1 个生效"
    print("越界 get->UNKNOWN、越界 set 跳过 OK")

    # 4) set / get（单个 + 批量 + 世界坐标）
    print("\n== set/get ==")
    assert vm.get([5, 5, 5]) == OCCUPIED
    free_idx = np.array([[10, 10, 10], [11, 10, 10], [12, 10, 10]])
    vm.set_many(free_idx, FREE)
    assert np.array_equal(vm.get(free_idx), [FREE, FREE, FREE])
    # 用世界坐标设置：取某体素中心点
    p = vm.voxel_to_world([20, 30, 40])
    vm.set_world(p, OCCUPIED)
    assert vm.get([20, 30, 40]) == OCCUPIED
    print("单个/批量/世界坐标 set-get 一致 OK")

    # 5) 非自由掩码 + 统计
    print("\n== non_free_mask / counts ==")
    c = vm.counts()
    mask = vm.non_free_mask()
    print("counts:", {["UNKNOWN", "FREE", "OCCUPIED"][k]: v for k, v in c.items()})
    assert mask.sum() == c[UNKNOWN] + c[OCCUPIED], "非 FREE = UNKNOWN ∪ OCCUPIED"
    assert mask.shape == vm.shape
    # state_centers 形状
    occ_pts = vm.state_centers(OCCUPIED)
    assert occ_pts.shape == (c[OCCUPIED], 3)
    print("non_free_mask 与 counts 自洽 OK")

    # 6) build_roi_voxmap（单一 ROI 来源：config.roi 的 center/dims/voxel_size_m）
    print("\n== build_roi_voxmap ==")
    cfg = load_config()
    roi = build_roi_voxmap(cfg)                        # 默认外扩 1 体素
    dims = np.asarray(cfg.roi_dims); vs = cfg.voxel_size_m_coarse
    exp = tuple(int(round((dims[k] + 2 * vs) / vs)) for k in range(3))
    print(f"roi center={cfg.roi_center} dims={cfg.roi_dims} voxel={vs} -> "
          f"shape={roi.shape} (~{roi.num_voxels/1e6:.2f}M voxels)")
    assert roi.shape == exp, (roi.shape, exp)
    assert np.allclose(roi.center, cfg.roi_center, atol=vs), (roi.center, cfg.roi_center)
    assert roi.counts()[UNKNOWN] == roi.num_voxels

    # 7) 可视化（可选）：交互式 3D 窗口（可旋转/缩放）
    if args.viz:
        from gt_gen import viz
        print("打开 open3d 交互窗口（红=OCCUPIED 绿=FREE）；关闭窗口后继续...")
        viz.show_voxmap(vm)

    print("\nSTEP2_OK")


if __name__ == "__main__":
    main()
