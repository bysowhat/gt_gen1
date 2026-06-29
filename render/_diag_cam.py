"""诊断：找出左右目相机点（Link6 局部系）落在机械臂哪个网格的包围盒内。"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np
from pxr import Usd, UsdGeom, Gf

LEFT = np.array([0.05018328, 0.09963676, 0.14782703])
RIGHT = np.array([-0.06923272, 0.09421033, 0.14022987])

stage = Usd.Stage.Open(args.usd)
link6 = next(p for p in stage.Traverse() if p.GetName() == "Link6")
xc = UsdGeom.XformCache(Usd.TimeCode.Default())
bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)

T_world_link6 = xc.GetLocalToWorldTransform(link6)
T_link6_world = T_world_link6.GetInverse()


def in_link6(pt_world):
    p = T_link6_world.Transform(Gf.Vec3d(*pt_world))
    return np.array([p[0], p[1], p[2]])


print(f"{'mesh':50s} {'min(L6)':28s} {'max(L6)':28s}  L  R")
for prim in Usd.PrimRange(link6):
    if prim.GetTypeName() != "Mesh":
        continue
    rng = bb.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    # 8 角点转 Link6 系求 AABB
    corners = []
    for xx in (mn[0], mx[0]):
        for yy in (mn[1], mx[1]):
            for zz in (mn[2], mx[2]):
                corners.append(in_link6([xx, yy, zz]))
    corners = np.array(corners)
    lo, hi = corners.min(0), corners.max(0)

    def inside(pt):
        return bool(np.all(pt >= lo - 1e-4) and np.all(pt <= hi + 1e-4))

    name = prim.GetPath().pathString.split("Link6/")[-1][:50]
    print(f"{name:50s} {np.round(lo,3)} {np.round(hi,3)}  "
          f"{'X' if inside(LEFT) else '.'}  {'X' if inside(RIGHT) else '.'}")

app.close()
