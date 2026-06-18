"""在 Isaac Sim 里可视化「自动放置障碍物」的一个场景（place_obstacles.py 产出的 npz）。

显示：工件 USD（基座系）+ 机器人 + 放置的障碍物（灰色纯视觉 VisualCuboid/VisualCylinder），
并回放默认轨迹(--which default，应撞障碍) 或 绕行轨迹(--which detour，应绕开)。

多绕行解并排：--which detour 且 npz 含 detour_positions_all 有 ≥2 条候选时，沿 base 系 +Y 方向
按 --lane_gap 间距并排放【多套】(工件+障碍+机器人)，每套整体平移、回放各自那一条绕行轨迹，
一屏内对比所有候选。--detour_index ≥0 时退回只看该一条。

运行（带显示器）：
    conda run -n env_isaaclab python scripts/viz_placed_obstacle_isaacsim.py \
        --out_dir /tmp/placed_obstacles --index 0 --which detour
    # 或直接指定场景文件：--scene /tmp/placed_obstacles/scene_00_Link3_plate.npz
无显示器自检：加 --headless（spawn + 跑几帧即退，打印套数/障碍数/轨迹点数 + VIZ_PLACED_DONE）。
"""
import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# curobo 的 isaac_sim 示例目录（含 helper.add_robot_to_scene）。逐机器不同：
# 本地默认走下面这个；服务器用环境变量 CUROBO_ISAAC 覆盖（见 scripts/bash/*.sh）。
CUROBO_ISAAC = os.environ.get("CUROBO_ISAAC",
                              "/home/a/Projects/Github/curobo/examples/isaac_sim")

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

_ap = argparse.ArgumentParser()
_ap.add_argument("--scene", default=None, help="场景 npz 路径（优先）")
_ap.add_argument("--out_dir", default="/tmp/placed_obstacles", help="场景目录（配 --index）")
_ap.add_argument("--index", type=int, default=0, help="--out_dir 内第几个场景（按文件名排序）")
_ap.add_argument("--which", choices=["default", "detour"], default="detour", help="回放哪条轨迹")
_ap.add_argument("--detour_index", type=int, default=-1,
                 help="--which detour 时：<0(默认)=所有绕行候选并排可视化；≥0=只看第该条")
_ap.add_argument("--lane_gap", type=float, default=3.0,
                 help="多套并排时沿 base +Y 的整体平移间距（米）")
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


def _pad(positions):
    """轨迹首尾各补 30 帧静止，便于看清起止姿态。"""
    positions = np.asarray(positions, float)
    positions = np.concatenate([np.tile(positions[0][None], (30, 1)), positions], axis=0)
    positions = np.concatenate([positions, np.tile(positions[-1][None], (30, 1))], axis=0)
    return positions


def _temp_usd_dest(robot_cfg):
    """复刻 helper.add_robot_to_scene(ISAAC_SIM_45 分支) 写出的临时 USD 路径。
    首套导入后该文件即存在，后续套用 AddReference 引用它（避免重复跑 URDF importer 崩溃）。"""
    from curobo.util_file import (get_assets_path, get_filename,
                                  get_path_of_dir, join_path)
    k = robot_cfg["kinematics"]
    ap = k.get("external_asset_path") or get_assets_path()
    full = join_path(ap, k["urdf_path"])
    return join_path(get_path_of_dir(full),
                     get_filename(full, remove_extension=True) + "_temp.usd")


def spawn_prim_dict(lane, k, p, off):
    """序列化的障碍 dict → 纯视觉 prim（灰色）。pose=[x,y,z,qw,qx,qy,qz]；off=该套整体平移。"""
    pos = np.asarray(p["pose"][:3], float) + off
    quat = np.asarray(p["pose"][3:7], float)            # wxyz
    color = np.asarray(p["color"], float)
    pth = f"/World/obstacles/lane{lane}/prim_{k}"
    if p["kind"] == "box":
        _cuboid.VisualCuboid(prim_path=pth, name=f"obs_{lane}_{k}", position=pos, orientation=quat,
                             size=1.0, scale=np.asarray(p["dims"], float), color=color)
    else:
        _cylinder.VisualCylinder(prim_path=pth, name=f"obs_{lane}_{k}", position=pos,
                                 orientation=quat, radius=float(p["radius"]),
                                 height=float(p["height"]), color=color)


def main():
    data = np.load(scene_path, allow_pickle=True)

    # ---- 选出要回放的轨迹列表（lanes）：default=单条；detour=按 detour_index 取单条或全部 ----
    if args.which == "default":
        lanes = [np.asarray(data["positions"], float)]
    elif "detour_positions_all" in data.files:
        alld = [np.asarray(t, float) for t in data["detour_positions_all"]]
        if args.detour_index >= 0:
            di = min(args.detour_index, len(alld) - 1)
            lanes = [alld[di]]
            print(f"绕行候选 {len(alld)} 条，只回放第 {di} 条")
        else:
            lanes = alld
            print(f"绕行候选 {len(alld)} 条，全部并排可视化（间距 {args.lane_gap} m）")
    else:                                               # 旧 npz：仅单条 detour_positions
        lanes = [np.asarray(data["detour_positions"], float)]

    lanes = [_pad(l) for l in lanes]
    n_lane = len(lanes)
    offsets = [np.array([0.0, i * args.lane_gap, 0.0], float) for i in range(n_lane)]

    joint_names = [str(x) for x in data["joint_names"]]
    piece_pose_to_robot = np.asarray(data["piece_pose_to_robot"], float)
    obj_path = str(data["obj_path"])
    prims = list(data["obstacle_prims"])
    link = str(data["link"]); otype = str(data["otype"])
    print(f"场景: {os.path.basename(scene_path)}  link={link} otype={otype} "
          f"障碍原语={len(prims)}  套数={n_lane}")
    print(f"回放 {args.which} 轨迹，各套点数: {[int(l.shape[0]) for l in lanes]}")

    cfg = load_config()
    compat.apply_trimesh_shim()
    robot_cfg = load_yaml(cfg.robot_cfg_path)["robot_cfg"]

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    usd_obj = obj_path.replace("_watertight.obj", ".usd")

    def spawn_workpiece(i, off):
        """第 i 套工件 USD（基座系 + 整体平移 off），关物理当纯视觉。"""
        if not os.path.exists(usd_obj):
            print("warn: 未找到工件 usd:", usd_obj)
            return
        pth = f"/World/workpiece_{i}"
        add_reference_to_stage(usd_path=usd_obj, prim_path=pth)
        from omni.isaac.core.prims import XFormPrim
        XFormPrim(pth).set_world_pose(
            position=(piece_pose_to_robot[:3] + off).tolist(),
            orientation=(piece_pose_to_robot[3:7]).tolist())
        from pxr import Usd, UsdPhysics
        import omni.usd
        stg = omni.usd.get_context().get_stage()
        for pr in Usd.PrimRange(stg.GetPrimAtPath(pth)):
            if pr.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(pr).GetCollisionEnabledAttr().Set(False)
            if pr.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(pr).GetRigidBodyEnabledAttr().Set(False)

    base_link_name = robot_cfg["kinematics"]["base_link"]
    dest_path = _temp_usd_dest(robot_cfg)

    def add_robot_ref(i, off):
        """额外套：引用首套导入生成的临时 USD（不再重复跑 URDF importer，否则 stage
        default prim 失效 → UsdExpiredPrimAccessError 崩溃），底座按 off 平移。"""
        from omni.isaac.core.robots import Robot
        from curobo.util.usd_helper import set_prim_transform
        stage = world.scene.stage
        pth = f"/World/robot_lane{i}"
        stage.OverridePrim(pth).GetReferences().AddReference(dest_path)
        set_prim_transform(stage.GetPrimAtPath(pth),
                           [float(off[0]), float(off[1]), float(off[2]), 1.0, 0.0, 0.0, 0.0])
        return world.scene.add(Robot(prim_path=f"{pth}/{base_link_name}", name=f"robot_{i}"))

    # ---- 逐套 spawn：工件 + 障碍 + 机器人（机器人底座按 off 平移）----
    # 第 0 套走 add_robot_to_scene（跑一次 URDF importer，生成临时 USD）；其余套引用该 USD。
    robots = []                                         # [(robot, idx_list, positions)]
    for i, off in enumerate(offsets):
        spawn_workpiece(i, off)
        for k, p in enumerate(prims):
            spawn_prim_dict(i, k, p, off)
        if i == 0 or not os.path.exists(dest_path):
            robot, _ = add_robot_to_scene(
                robot_cfg, world, robot_name=f"robot_{i}",
                position=np.array([0.0, off[1], 0.0]))
        else:
            robot = add_robot_ref(i, off)
        robots.append([robot, None, lanes[i]])

    world.reset()
    for entry in robots:
        robot = entry[0]
        if hasattr(robot, "initialize"):
            robot.initialize()
        entry[1] = [robot.get_dof_index(j) for j in joint_names]
        robot.set_joint_positions(entry[2][0], entry[1])

    if args.headless:
        for _ in range(3):
            world.step(render=False)
        print(f"已 spawn {n_lane} 套 × {len(prims)} 个障碍 prim。")
        print("VIZ_PLACED_DONE")
        simulation_app.close()
        return

    maxlen = max(int(p.shape[0]) for _, _, p in robots)
    i = hold = 0
    print(f"开始播放 {args.which}（{n_lane} 套并排，关闭窗口结束）…")
    while simulation_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            continue
        if i < maxlen:
            for robot, idx_list, pos in robots:
                j = min(i, int(pos.shape[0]) - 1)
                robot.set_joint_positions(pos[j], idx_list)
            i += 1
        else:
            hold += 1
            if hold > args.fps * 2:
                i = hold = 0
    simulation_app.close()


if __name__ == "__main__":
    main()
