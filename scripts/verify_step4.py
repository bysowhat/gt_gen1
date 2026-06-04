"""Step 4 验证：观测更新 observe_and_update（raycast→体素化→合并进 voxmap）。

场景：相机在 base 原点沿 +Z，z=1.0 处一面墙；voxmap 覆盖相机前方区域。
- 单次观测：occ 体素贴墙(z≈0.975)、free 体素在墙前、墙后仍 UNKNOWN；
- 累积 + 粘滞：再观测一次「墙已撤走(移到 max_depth 外)」，
  墙后原 UNKNOWN 变 FREE、free/总量单增，而原 OCCUPIED 体素不被降级。

运行：conda run -n env_isaaclab python scripts/verify_step4.py [--viz]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def wall(D, extents_xy=6.0, thick=0.05):
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
    from gt_gen.mapping import observe_and_update
    from gt_gen.voxmap import ThreeStateVoxelMap, UNKNOWN, FREE, OCCUPIED

    cfg = load_config()
    cm = sensor.load_camera_model(cfg)
    eye = np.eye(4)                      # 相机=base，+Z 视线
    D = 1.0
    near_face = D - 0.025
    stride = 8

    vm = ThreeStateVoxelMap(origin=(-0.6, -0.6, -0.1), size_xyz=(1.2, 1.2, 1.7), voxel_size=0.02)
    print("voxmap shape:", vm.shape, " 初始 counts:", _names(vm.counts()))
    assert vm.counts()[UNKNOWN] == vm.num_voxels

    # 1) 单次观测：z=1.0 大墙
    print("\n== 1) 单次观测（墙 @ z=1.0）==")
    r1 = observe_and_update(vm, eye, cm, wall(D), cm["max_depth"], pixel_stride=stride)
    c1 = vm.counts()
    print("写入:", r1, " counts:", _names(c1))
    assert c1[OCCUPIED] > 0 and c1[FREE] > 0
    occ_ctr = vm.state_centers(OCCUPIED)
    free_ctr = vm.state_centers(FREE)
    print("occ z 范围:", np.round([occ_ctr[:, 2].min(), occ_ctr[:, 2].max()], 4),
          " free z 上界:", round(float(free_ctr[:, 2].max()), 4))
    assert np.all((occ_ctr[:, 2] > near_face - 0.03) & (occ_ctr[:, 2] < near_face + 0.03)), "occ 应贴墙近面"
    assert free_ctr[:, 2].max() < near_face, "free 应都在墙前"
    # 墙后 (z=1.3) 应仍 UNKNOWN（被墙挡住没观测到）
    behind = vm.world_to_voxel([0.0, 0.0, 1.3])
    assert vm.get(behind) == UNKNOWN, "墙后体素应仍 UNKNOWN"
    # 记一个 occ 体素，后面验证粘滞
    occ_idx = vm.world_to_voxel(occ_ctr[len(occ_ctr) // 2])
    assert vm.get(occ_idx) == OCCUPIED
    print("墙后 z=1.3 体素: UNKNOWN ✓")

    # 2) 累积 + 粘滞：墙撤走（移到 max_depth 外），重新观测
    print("\n== 2) 再观测（墙撤走→自由空间）+ 粘滞校验 ==")
    r2 = observe_and_update(vm, eye, cm, wall(10.0), cm["max_depth"], pixel_stride=stride)
    c2 = vm.counts()
    print("写入:", r2, " counts:", _names(c2))
    assert c2[OCCUPIED] == c1[OCCUPIED], "粘滞：OCCUPIED 不应被 free 降级"
    assert vm.get(occ_idx) == OCCUPIED, "原 occ 体素仍 OCCUPIED"
    assert c2[FREE] > c1[FREE], "墙撤走后应揭开更多 FREE"
    assert c2[UNKNOWN] < c1[UNKNOWN], "UNKNOWN 应单调减少"
    assert vm.get(behind) == FREE, "墙后 z=1.3 现应被揭为 FREE"
    print("OCCUPIED 未降级 ✓  墙后 z=1.3 现为 FREE ✓  UNKNOWN 单减 ✓")

    if args.viz:
        from gt_gen import viz
        print("\n打开 open3d：观测后的体素图（红=OCC 绿=FREE）")
        viz.show_voxmap(vm)

    print("\nSTEP4_OK")


def _names(c):
    return {["UNKNOWN", "FREE", "OCCUPIED"][k]: v for k, v in c.items()}


if __name__ == "__main__":
    main()
