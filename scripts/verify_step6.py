"""Step 6 验证：整臂扫掠体积 + motion_stays_in_free。

- voxmap 全 UNKNOWN：q0→q1 运动【被拒】（扫掠体积伸进未知）；
- 把该段扫掠体积标 FREE：同一运动【通过】（全在自由区）；
- 在扫掠体积里放一个 OCCUPIED 体素：又【被拒】（保守，碰到障碍）。

运行：conda run -n env_isaaclab python scripts/verify_step6.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from gt_gen.config import load_config
    from gt_gen import curobo_iface as ci
    from gt_gen.voxmap import build_roi_voxmap, FREE, OCCUPIED, UNKNOWN
    from gt_gen.swept import swept_volume, motion_stays_in_free

    cfg = load_config()
    h = ci.init_curobo(cfg)
    vm = build_roi_voxmap(cfg)
    assert vm.counts()[UNKNOWN] == vm.num_voxels

    q0 = list(cfg.retract_config)
    q1 = list(q0); q1[0] += 0.5; q1[1] += 0.3        # 整臂扫一段

    cells = swept_volume(h, vm, q0, q1)
    print("扫掠体积体素数:", cells.shape[0])
    assert cells.shape[0] > 0

    # 1) 全 UNKNOWN → 被拒
    ok, n = motion_stays_in_free(h, vm, q0, q1)
    print(f"全 UNKNOWN: ok={ok} 非FREE={n}")
    assert not ok and n == cells.shape[0], "未知区里整段都应判非自由"

    # 2) 扫掠体积标 FREE → 通过
    vm.set_many(cells, FREE)
    ok, n = motion_stays_in_free(h, vm, q0, q1)
    print(f"扫掠区已FREE: ok={ok} 非FREE={n}")
    assert ok and n == 0, "全在自由区应通过"

    # 3) 扫掠体积内放一个 OCCUPIED → 被拒
    vm.set_many(cells[cells.shape[0] // 2], OCCUPIED)
    ok, n = motion_stays_in_free(h, vm, q0, q1)
    print(f"扫掠区含1个OCCUPIED: ok={ok} 非FREE={n}")
    assert not ok and n >= 1, "碰到障碍应被拒"

    # 4) 不动的运动(q0→q0)在自由区应通过；细分辨率不漏体素
    vm2 = build_roi_voxmap(cfg)
    c0 = swept_volume(h, vm2, q0, q0)
    vm2.set_many(c0, FREE)
    ok, n = motion_stays_in_free(h, vm2, q0, q0)
    print(f"零位移运动(自身FREE): ok={ok} 非FREE={n}")
    assert ok and n == 0

    print("\nSTEP6_OK")


if __name__ == "__main__":
    main()
