"""Step 5 验证：voxmap → cuRobo 碰撞世界同步（未知=障碍）。

- 全 UNKNOWN 同步后：retract 构型判【碰撞】（未知=障碍，悲观）；
- 把 retract 整臂所在体素标 FREE 再同步：同构型变【无碰撞】；
- 另一构型 q2 仍伸在未知区 → 仍【碰撞】；把 q2 所在体素标 FREE 再同步 → q2 变【无碰撞】。

运行：conda run -n env_isaaclab python scripts/verify_step5.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fk_spheres(handle, q):
    import torch
    st = handle.mg.kinematics.get_state(torch.tensor([list(q)], dtype=torch.float32, device="cuda"))
    return st.link_spheres_tensor[0].detach().cpu().numpy()   # (N,4) xyz+r base


def carve_free(voxmap, spheres, margin):
    """把整臂碰撞球(含 margin)覆盖的体素标 FREE（模拟"这块已观测为自由"）。"""
    from gt_gen.voxmap import FREE
    n = 0
    for s in spheres:
        c = s[:3]; r = float(s[3])
        if r <= 1e-4:
            continue
        R = r + margin
        lo = voxmap.world_to_voxel(c - R); hi = voxmap.world_to_voxel(c + R)
        rs = [np.arange(lo[k], hi[k] + 1) for k in range(3)]
        ii, jj, kk = np.meshgrid(*rs, indexing="ij")
        idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
        keep = np.linalg.norm(voxmap.voxel_to_world(idx) - c, axis=1) <= R
        n += voxmap.set_many(idx[keep], FREE)
    return n


def main():
    from gt_gen.config import load_config
    from gt_gen import curobo_iface as ci
    from gt_gen.collision_sync import sync_collision_world
    from gt_gen.voxmap import ThreeStateVoxelMap, UNKNOWN

    cfg = load_config()
    print("== init_curobo（VOXEL 世界，小 ROI 便于快测）==")
    h = ci.init_curobo(cfg, roi_dims=[2.0, 2.0, 2.0], roi_center=[0.0, 0.0, 0.6])
    center = np.asarray(h.voxel["pose"][:3], float)
    dims = np.asarray(h.voxel["dims"], float)
    vs = h.voxel["voxel_size"]
    print("voxel 世界 dims:", dims, " center:", center, " voxel:", vs)

    # 与 cuRobo voxel 世界共范围的 voxmap（初始全 UNKNOWN）
    vm = ThreeStateVoxelMap(origin=center - dims / 2, size_xyz=dims, voxel_size=vs)
    assert vm.counts()[UNKNOWN] == vm.num_voxels

    retract = cfg.retract_config
    q2 = list(retract); q2[0] += 1.2; q2[1] -= 0.3      # 摆到另一片区域

    # 1) 全 UNKNOWN → 同步 → 两个构型都应判碰撞
    print("\n== 1) 全 UNKNOWN 同步（未知=障碍）==")
    r = sync_collision_world(h, vm)
    print("  同步:", r)
    f_ret, c_ret = ci.check_state(h, retract)
    f_q2, c_q2 = ci.check_state(h, q2)
    print(f"  retract feasible={f_ret} (constraint={c_ret:.3f})")
    print(f"  q2      feasible={f_q2} (constraint={c_q2:.3f})")
    assert not f_ret and not f_q2, "全未知时任何构型都应判碰撞"

    # 2) 标 FREE 掉 retract 整臂所在体素 → 同步 → retract 无碰撞，q2 仍碰撞
    print("\n== 2) 观测 retract 周边为 FREE 后同步 ==")
    n = carve_free(vm, fk_spheres(h, retract), margin=0.06)
    print("  置 FREE 体素数:", n, " counts:", {["UNK", "FREE", "OCC"][k]: v for k, v in vm.counts().items()})
    sync_collision_world(h, vm)
    f_ret, c_ret = ci.check_state(h, retract)
    f_q2, c_q2 = ci.check_state(h, q2)
    print(f"  retract feasible={f_ret} (constraint={c_ret:.3f})")
    print(f"  q2      feasible={f_q2} (constraint={c_q2:.3f})")
    assert f_ret, "retract 区域已 FREE，应无碰撞"
    assert not f_q2, "q2 仍在未知区，应仍碰撞"

    # 3) 再把 q2 区域标 FREE → q2 也无碰撞
    print("\n== 3) 再观测 q2 周边为 FREE ==")
    carve_free(vm, fk_spheres(h, q2), margin=0.06)
    sync_collision_world(h, vm)
    f_q2, c_q2 = ci.check_state(h, q2)
    print(f"  q2 feasible={f_q2} (constraint={c_q2:.3f})")
    assert f_q2, "q2 区域已 FREE，应无碰撞"

    print("\nSTEP5_OK")


if __name__ == "__main__":
    main()
