"""在 Isaac Sim 里一屏【平铺】展示所有障碍物类型及其碰撞(膨胀)版本的对比。

不读 npz：直接遍历 gt_gen.obstacles 全部类型（list_obstacles），每种用其默认形状生成原语，
再用 gt_gen.obstacle_placement.inflate_prims 按给定 buffer 膨胀（与 build_world 同一来源）：
  Box 每边 +2·buffer、Tube 半径 +buffer/高 +2·buffer。
平铺：每种类型占一个网格格子；格子里【原始(橙) 在上(+Y)、碰撞膨胀版(红) 在下(−Y)】挨着不重合，
全部类型排成 ncol 列的网格，一次可视化即可看到所有类型胖了多少。纯视觉 prim、无物理。

脚本只输入 buffer：
    conda run -n env_isaaclab python scripts/viz_obstacle_buffer_isaacsim.py --buffer 0.05
无显示器自检：加 --headless（spawn + 跑几帧即退，打印每种类型膨胀前后尺寸 + VIZ_BUFFER_DONE）。
"""
import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

_ap = argparse.ArgumentParser()
_ap.add_argument("--buffer", type=float, default=0.05, help="膨胀量(米)：障碍碰撞版相对真实尺寸外扩一圈")
_ap.add_argument("--headless", action="store_true")
args = _ap.parse_args()

from omni.isaac.kit import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np  # noqa: E402

sys.path.insert(0, ROOT)

from gt_gen import compat  # noqa: F401,E402  warp shim
from gt_gen import obstacles as ob  # noqa: E402
from gt_gen import obstacle_placement as opl  # noqa: E402

from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.objects import cuboid as _cuboid  # noqa: E402
from omni.isaac.core.objects import cylinder as _cylinder  # noqa: E402

ORANGE = np.array([0.95, 0.55, 0.15])   # 原始（真实尺寸）
RED = np.array([0.85, 0.10, 0.10])      # 碰撞膨胀版
NCOL = 5                                # 网格列数（15 种 → 3 行）
GAP = 0.30                              # 间隙（米）：格内上下 / 格间
Z_FLOOR = 0.20                          # 障碍底部离地高度（米），避免穿地面


def spawn_prim(prim, tag, color, offset):
    """Box/Tube → 纯视觉 prim（指定颜色），整体平移 offset。pose=[x,y,z,qw,qx,qy,qz]。"""
    pos = np.asarray(prim.pose[:3], float) + np.asarray(offset, float)
    quat = np.asarray(prim.pose[3:7], float)            # wxyz
    pth = f"/World/{tag}"
    if isinstance(prim, ob.Box):
        _cuboid.VisualCuboid(prim_path=pth, name=tag, position=pos, orientation=quat,
                             size=1.0, scale=np.asarray(prim.dims, float), color=color)
    else:
        _cylinder.VisualCylinder(prim_path=pth, name=tag, position=pos, orientation=quat,
                                 radius=float(prim.radius), height=float(prim.height), color=color)


def _bound(prims):
    """整组障碍的保守 AABB：返回 (中心(3), 最大轴跨度, 下界(3))。
    每个 prim 用包围球(Box 半对角线 / Tube hypot)当作立方体外估，足够用于平铺布局间距。"""
    los, his = [], []
    for p in prims:
        c = np.asarray(p.pose[:3], float)
        if isinstance(p, ob.Box):
            r = 0.5 * float(np.linalg.norm(np.asarray(p.dims, float)))
        else:
            r = float(math.hypot(float(p.radius), 0.5 * float(p.height)))
        los.append(c - r)
        his.append(c + r)
    lo = np.min(los, axis=0)
    hi = np.max(his, axis=0)
    return (lo + hi) / 2.0, float(np.max(hi - lo)), lo


def main():
    buffer_m = float(args.buffer)
    types = ob.list_obstacles()

    # 先全建一遍，求全局最大跨度 → 统一网格间距
    built = []
    dmax = 0.0
    for name in types:
        prims = ob.build(name, [0.0, 0.0, 0.0])
        inflated = opl.inflate_prims(prims, buffer_m)
        ctr, _, _ = _bound(prims)
        _, dia_inf, lo_inf = _bound(inflated)
        dmax = max(dmax, dia_inf)
        built.append((name, prims, inflated, ctr, lo_inf))

    pair_dy = dmax + GAP                     # 格内：原始 → 膨胀版 的下移量
    pitch_x = dmax + GAP                     # 列间距
    pitch_row = 2.0 * dmax + 2.0 * GAP       # 行间距（容下上下一对 + 间隙）

    print(f"buffer={buffer_m:.3f}m  类型={len(types)} 种  网格={NCOL}列  单元跨度≈{dmax:.2f}m")

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    for i, (name, prims, inflated, ctr, lo_inf) in enumerate(built):
        col, row = i % NCOL, i // NCOL
        cell_x = col * pitch_x
        cell_y = -row * pitch_row
        lift = Z_FLOOR - float(lo_inf[2])    # 用膨胀版下界抬起，红/橙都不穿地
        base_off = np.array([cell_x - ctr[0], cell_y - ctr[1], lift])
        red_off = base_off + np.array([0.0, -pair_dy, 0.0])

        for k, p in enumerate(prims):        # 原始（橙），在上(+Y)
            spawn_prim(p, f"orig_{i:02d}_{name}_{k}", ORANGE, base_off)
        for k, q in enumerate(inflated):     # 膨胀版（红），在下(−Y)
            spawn_prim(q, f"infl_{i:02d}_{name}_{k}", RED, red_off)

        sample = prims[0]
        if isinstance(sample, ob.Box):
            sz = f"box dims≈{[round(v,3) for v in sample.dims]}(+{2*buffer_m:.2f}/边)"
        else:
            sz = f"tube r≈{sample.radius:.3f}(+{buffer_m:.2f}) h≈{sample.height:.3f}(+{2*buffer_m:.2f})"
        print(f"  [{i:02d}] 行{row}列{col} x={cell_x:6.2f} {name:<18} 原语{len(prims)}个  {sz}")

    world.reset()

    if args.headless:
        for _ in range(3):
            world.step(render=False)
        print(f"已 spawn {len(types)} 种类型 × (橙=原始 / 红=膨胀)。")
        print("VIZ_BUFFER_DONE")
        simulation_app.close()
        return

    print("橙=原始(真实尺寸,在上)，红=碰撞膨胀版(+buffer,在下)。关闭窗口结束。")
    while simulation_app.is_running():
        world.step(render=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
