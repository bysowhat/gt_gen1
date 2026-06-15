"""在 Isaac Sim 里可视化「自动放置障碍物」的一个场景（place_obstacles.py 产出的 npz）。

显示：工件 USD（基座系）+ 机器人 + 放置的障碍物（灰色纯视觉 VisualCuboid/VisualCylinder），
并回放默认轨迹(--which default，应撞障碍) 或 绕行轨迹(--which detour，应绕开)。

运行（带显示器）：
    conda run -n env_isaaclab python scripts/viz_placed_obstacle_isaacsim.py \
        --out_dir /tmp/placed_obstacles --index 0 --which detour
    # 或直接指定场景文件：--scene /tmp/placed_obstacles/scene_00_Link3_plate.npz
无显示器自检：加 --headless（spawn + 跑几帧即退，打印障碍数/轨迹点数 + VIZ_PLACED_DONE）。
"""
import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUROBO_ISAAC = "/home/a/Projects/Github/curobo/examples/isaac_sim"

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

_ap = argparse.ArgumentParser()
_ap.add_argument("--scene", default=None, help="场景 npz 路径（优先）")
_ap.add_argument("--out_dir", default="/tmp/placed_obstacles", help="场景目录（配 --index）")
_ap.add_argument("--index", type=int, default=0, help="--out_dir 内第几个场景（按文件名排序）")
_ap.add_argument("--which", choices=["default", "detour"], default="detour", help="回放哪条轨迹")
_ap.add_argument("--headless", action="store_true")
_ap.add_argument("--fps", type=int, default=30)
args = _ap.parse_args()

scene_path = args.scene
if scene_path is None:
    hits = sorted(glob.glob(os.path.join(args.out_dir, "*.npz")))
    if not hits:
        raise FileNotFoundError(f"{args.out_dir} 下没有场景 npz")
    scene_path = hits[max(0, min(args.index, len(hits) - 1))]

from omni.isaac.kit import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np  # noqa: E402

sys.path.insert(0, ROOT)
sys.path.insert(0, CUROBO_ISAAC)

from gt_gen import compat  # noqa: F401,E402  warp shim
from gt_gen.config import load_config  # noqa: E402
from curobo.util_file import load_yaml  # noqa: E402

from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.objects import cuboid as _cuboid  # noqa: E402
from omni.isaac.core.objects import cylinder as _cylinder  # noqa: E402
from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa: E402
from helper import add_robot_to_scene  # noqa: E402


def spawn_prim_dict(k, p):
    """序列化的障碍 dict → 纯视觉 prim（灰色）。pose=[x,y,z,qw,qx,qy,qz]。"""
    pos = np.asarray(p["pose"][:3], float)
    quat = np.asarray(p["pose"][3:7], float)            # wxyz
    color = np.asarray(p["color"], float)
    pth = f"/World/obstacles/prim_{k}"
    if p["kind"] == "box":
        _cuboid.VisualCuboid(prim_path=pth, name=f"obs_{k}", position=pos, orientation=quat,
                             size=1.0, scale=np.asarray(p["dims"], float), color=color)
    else:
        _cylinder.VisualCylinder(prim_path=pth, name=f"obs_{k}", position=pos, orientation=quat,
                                 radius=float(p["radius"]), height=float(p["height"]), color=color)


def main():
    data = np.load(scene_path, allow_pickle=True)
    positions = data["positions"] if args.which == "default" else data["detour_positions"]
    positions = np.concatenate([positions, np.tile(positions[-1][None], (50, 1))], axis=0)
    joint_names = [str(x) for x in data["joint_names"]]
    piece_pose_to_robot = np.asarray(data["piece_pose_to_robot"], float)
    obj_path = str(data["obj_path"])
    prims = list(data["obstacle_prims"])
    link = str(data["link"]); otype = str(data["otype"])
    print(f"场景: {os.path.basename(scene_path)}  link={link} otype={otype} 障碍原语={len(prims)}")
    print(f"回放 {args.which} 轨迹，点数: {positions.shape}")

    cfg = load_config()
    compat.apply_trimesh_shim()
    robot_cfg = load_yaml(cfg.robot_cfg_path)["robot_cfg"]

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # 工件 USD（基座系），关物理当纯视觉
    usd_obj = obj_path.replace("_watertight.obj", ".usd")
    if os.path.exists(usd_obj):
        add_reference_to_stage(usd_path=usd_obj, prim_path="/World/workpiece")
        from omni.isaac.core.prims import XFormPrim
        XFormPrim("/World/workpiece").set_world_pose(
            position=(piece_pose_to_robot[:3]).tolist(),
            orientation=(piece_pose_to_robot[3:7]).tolist())
        from pxr import Usd, UsdPhysics
        import omni.usd
        stg = omni.usd.get_context().get_stage()
        for pr in Usd.PrimRange(stg.GetPrimAtPath("/World/workpiece")):
            if pr.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(pr).GetCollisionEnabledAttr().Set(False)
            if pr.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(pr).GetRigidBodyEnabledAttr().Set(False)
    else:
        print("warn: 未找到工件 usd:", usd_obj)

    for k, p in enumerate(prims):
        spawn_prim_dict(k, p)

    robot, _ = add_robot_to_scene(robot_cfg, world)

    world.reset()
    if hasattr(robot, "initialize"):
        robot.initialize()
    idx_list = [robot.get_dof_index(j) for j in joint_names]
    robot.set_joint_positions(positions[0], idx_list)

    if args.headless:
        for _ in range(3):
            world.step(render=False)
        print(f"已 spawn {len(prims)} 个障碍 prim。")
        print("VIZ_PLACED_DONE")
        simulation_app.close()
        return

    i = hold = 0
    print(f"开始播放 {args.which}（关闭窗口结束）…")
    while simulation_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            continue
        if i < len(positions):
            robot.set_joint_positions(positions[i], idx_list)
            i += 1
        else:
            hold += 1
            if hold > args.fps * 2:
                i = hold = 0
    simulation_app.close()


if __name__ == "__main__":
    main()
