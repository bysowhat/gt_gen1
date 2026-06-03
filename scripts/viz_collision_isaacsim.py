"""Isaac Sim 可视化：输入关节角，只画出与工件【相交】的碰撞球（红色），无碰撞则不画。

碰撞按 robot yml 的 collision_spheres 逐颗算（FK 到基座系 -> 与工件 mesh 求带符号距离，
球心入网格深度 > -半径即相交）。机器人在原点，工件 USD 平移 -robot_pos 摆到基座系。

运行（带显示器）：
    # 默认用 seam pkl 的 joint_angles
    conda run -n env_isaaclab python scripts/viz_collision_isaacsim.py --seam <seam_x.pkl>
    # 指定关节角（弧度，按 joint_names 顺序）
    conda run -n env_isaaclab python scripts/viz_collision_isaacsim.py --seam <..> --joints "j1,...,j6"
无显示器自检：加 --headless（跑一帧即退，打印碰撞球数）。
"""
import argparse
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUROBO_ISAAC = "/home/a/Projects/Github/curobo/examples/isaac_sim"

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

_ap = argparse.ArgumentParser()
_ap.add_argument("--seam", required=True, help="seam pkl（取工件 obj + robot_pose + 默认目标角）")
_ap.add_argument("--joints", default=None, help="关节角(弧度,逗号分隔);默认用 pkl joint_angles")
_ap.add_argument("--headless", action="store_true")
args = _ap.parse_args()

from omni.isaac.kit import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np  # noqa: E402

sys.path.insert(0, ROOT)
sys.path.insert(0, CUROBO_ISAAC)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from gt_gen import compat  # noqa: F401,E402  warp shim
from gt_gen.config import load_config  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

from curobo.types.base import TensorDeviceType  # noqa: E402
from curobo.types.robot import RobotConfig  # noqa: E402
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel  # noqa: E402
from curobo.util_file import load_yaml  # noqa: E402

from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.objects import sphere  # noqa: E402
from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa: E402
from helper import add_robot_to_scene  # noqa: E402

from check_collision_spheres import quat_wxyz_to_R  # noqa: E402
import glob  # noqa: E402


def find_obj(seam_path):
    dd = os.path.dirname(seam_path)
    for pat in ("*_watertight.obj", "*.obj"):
        hits = sorted(glob.glob(os.path.join(dd, pat)))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no obj in {dd}")


def red_material(stage, opacity=0.6):
    from pxr import UsdShade, Sdf
    p = Sdf.Path("/World/Looks/CollisionRed")
    mtl = UsdShade.Material.Define(stage, p)
    sh = UsdShade.Shader.Define(stage, p.AppendChild("Shader"))
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.95, 0.1, 0.1))
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set((0.4, 0.0, 0.0))
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mtl


def main():
    cfg = load_config()
    d = pickle.load(open(args.seam, "rb"))
    rp = np.asarray(d["robot_pose"][0], float)
    q = (np.array([float(x) for x in args.joints.split(",")], float)
         if args.joints else np.asarray(d["joint_angles"], float))
    print("关节角:", np.round(q, 4))

    # 工件 mesh -> 基座系，建邻近查询
    obj_path = find_obj(args.seam)
    tm = trimesh.load(obj_path, force="mesh"); tm.apply_translation(-rp[:3])
    pq = trimesh.proximity.ProximityQuery(tm)

    # 注册所有碰撞 link 做 FK
    rd = load_yaml(cfg.robot_cfg_path)
    kin = rd["robot_cfg"]["kinematics"]
    coll_links = list(kin["collision_link_names"])
    spheres_def = kin["collision_spheres"]
    kin["link_names"] = coll_links
    ta = TensorDeviceType()
    model = CudaRobotModel(RobotConfig.from_dict(rd["robot_cfg"], ta).kinematics)

    # 算出相交的球（基座系世界坐标 + 半径）
    st = model.get_state(torch.tensor([q], dtype=torch.float32, device=ta.device))
    hits = []
    for ln in coll_links:
        if ln not in spheres_def:
            continue
        pos = st.link_pose[ln].position[0].detach().cpu().numpy()
        R = quat_wxyz_to_R(st.link_pose[ln].quaternion[0].detach().cpu().numpy())
        centers, radii, idxs = [], [], []
        for i, s in enumerate(spheres_def[ln]):
            r = float(s["radius"])
            if r <= 1e-4:
                continue
            centers.append(R @ np.asarray(s["center"], float) + pos); radii.append(r); idxs.append(i)
        if not centers:
            continue
        centers = np.array(centers); radii = np.array(radii)
        pen = pq.signed_distance(centers) + radii      # >0 相交
        for j in np.where(pen > 0)[0]:
            hits.append((ln, idxs[j], centers[j], float(radii[j]), float(pen[j])))
    print(f"碰撞球数: {len(hits)}")
    for ln, i, c, r, pe in hits:
        print(f"  {ln}#{i} 穿透={pe:.4f}m r={r:.3f} 球心={np.round(c,3)}")

    # ---- 场景 ----
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # 工件 USD（基座系：平移 -robot_pos），关物理当纯视觉
    usd_obj = obj_path.replace("_watertight.obj", ".usd")
    if os.path.exists(usd_obj):
        add_reference_to_stage(usd_path=usd_obj, prim_path="/World/workpiece")
        from omni.isaac.core.prims import XFormPrim
        XFormPrim("/World/workpiece").set_world_pose(position=(-rp[:3]).tolist())
        from pxr import Usd, UsdPhysics
        import omni.usd
        stg = omni.usd.get_context().get_stage()
        for p in Usd.PrimRange(stg.GetPrimAtPath("/World/workpiece")):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Set(False)
    else:
        print("warn: 未找到工件 usd:", usd_obj)

    robot_cfg = load_yaml(cfg.robot_cfg_path)["robot_cfg"]
    robot, _ = add_robot_to_scene(robot_cfg, world)

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    mtl = red_material(stage)

    # 只画相交的球
    from pxr import UsdShade
    for k, (ln, i, c, r, pe) in enumerate(hits):
        sp = sphere.VisualSphere(prim_path=f"/collision/sph_{k}",
                                 position=np.ravel(c), radius=r)
        UsdShade.MaterialBindingAPI(sp.prim).Bind(mtl)

    world.reset()
    if hasattr(robot, "initialize"):
        robot.initialize()
    idx_list = [robot.get_dof_index(j) for j in cfg.joint_names]
    robot.set_joint_positions(q, idx_list)

    if args.headless:
        for _ in range(3):
            world.step(render=False)
        print("VIZ_COLLISION_DONE")
        simulation_app.close()
        return

    print("显示中（关闭窗口结束）…" if hits else "无碰撞，未画球（关闭窗口结束）…")
    while simulation_app.is_running():
        world.step(render=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
