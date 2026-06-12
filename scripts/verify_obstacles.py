"""障碍物生成库单元校验（无需 Isaac Sim）。

遍历 gt_gen.obstacles.REGISTRY 每个障碍：
  - build(...) 产非空原语列表；
  - 每个 Box 的 dims>0、Tube 的 radius/height>0；
  - pose 长度 7 且四元数近单位；
  - to_curobo / to_world_config 能成功构建（验证 cuRobo 兼容）。
打印每种障碍的原语个数，全过则打印 VERIFY_OBSTACLES_OK。

运行：conda run -n env_isaaclab python scripts/verify_obstacles.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gt_gen import obstacles as ob  # noqa: E402


def _check_pose(pose):
    assert len(pose) == 7, f"pose 长度应为 7，得 {len(pose)}"
    q = np.asarray(pose[3:7], float)
    n = float(np.linalg.norm(q))
    assert abs(n - 1.0) < 1e-4, f"四元数非单位：|q|={n:.5f}"


def _check_prims(name, prims):
    assert len(prims) > 0, f"{name} 产出空列表"
    for p in prims:
        _check_pose(p.pose)
        if isinstance(p, ob.Box):
            d = np.asarray(p.dims, float)
            assert d.shape == (3,) and np.all(d > 0), f"{name}.{p.name} dims 非正：{p.dims}"
        elif isinstance(p, ob.Tube):
            assert p.radius > 0 and p.height > 0, \
                f"{name}.{p.name} radius/height 非正：r={p.radius} h={p.height}"
        else:
            raise TypeError(f"{name} 含未知原语类型 {type(p)}")


def main():
    anchor = [0.6, 0.0, 0.7]
    names = ob.list_obstacles()
    print(f"共 {len(names)} 种障碍：{names}\n")

    n_box = n_tube = 0
    for name in names:
        prims = ob.build(name, anchor, anchor_rpy_deg=(0, 0, 0))
        _check_prims(name, prims)
        nb = sum(isinstance(p, ob.Box) for p in prims)
        nt = sum(isinstance(p, ob.Tube) for p in prims)
        n_box += nb
        n_tube += nt
        # cuRobo 兼容：转换不报错
        wc = ob.to_world_config(prims)
        nc = len(wc.cuboid or [])
        ncy = len(wc.cylinder or [])
        assert nc == nb and ncy == nt, \
            f"{name} WorldConfig 数目不符：cuboid {nc}/{nb} cylinder {ncy}/{nt}"
        print(f"  ✓ {name:<16} 原语={len(prims):>2}  (Box={nb} Tube={nt})  → WorldConfig OK")

    # 抽查带姿态参数的变体也能构建
    for kw in (dict(name="plate", tilt_deg=30.0),
               dict(name="open_box", open_face="top"),
               dict(name="pipe", axis="x"),
               dict(name="parallel_pipes", n=5, axis="z", stack="y")):
        nm = kw.pop("name")
        prims = ob.build(nm, anchor, anchor_rpy_deg=(0, 0, 0), **kw)
        _check_prims(nm, prims)
        ob.to_world_config(prims)
    print("\n  ✓ 参数变体（tilt/open_face/axis/stack）均构建通过")

    print(f"\n合计原语：Box={n_box} Tube={n_tube}")
    print("\nVERIFY_OBSTACLES_OK [全部障碍生成合法 + cuRobo WorldConfig 转换通过]")


if __name__ == "__main__":
    main()
