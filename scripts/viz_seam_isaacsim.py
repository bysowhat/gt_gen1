"""在 Isaac Sim 里播放规划好的轨迹（/tmp/seam_traj.npz）。

复用 cuRobo 自带示例的 add_robot_to_scene（与你的 Isaac Sim 版本对齐）。
工件用 part 目录的 .usd 引用进来，平移 -robot_pos 放到机器人基座系（机器人在原点）。

运行（本机带显示器）：
    conda run -n env_isaaclab python scripts/viz_seam_isaacsim.py --traj /tmp/seam_traj.npz
无显示器自检：加 --headless（不渲染窗口，仅跑通流程）。
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUROBO_ISAAC = "/home/a/Projects/Github/curobo/examples/isaac_sim"

# ---- SimulationApp 必须最先创建 ----
try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

_ap = argparse.ArgumentParser()
_ap.add_argument("--traj", default="/tmp/seam_traj.npz")
_ap.add_argument("--headless", action="store_true")
_ap.add_argument("--fps", type=int, default=30)
args = _ap.parse_args()

from omni.isaac.kit import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

# ---- 之后再导入其余 ----
import numpy as np  # noqa: E402

sys.path.insert(0, ROOT)
sys.path.insert(0, CUROBO_ISAAC)

from gt_gen import compat  # noqa: F401,E402  warp shim
from gt_gen.config import load_config  # noqa: E402
from curobo.util_file import load_yaml  # noqa: E402

from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa: E402
from helper import add_robot_to_scene  # noqa: E402  cuRobo 示例自带


def main():
    data = np.load(args.traj, allow_pickle=True)
    positions = data["positions"]                       # (T,6)
    positions = np.concatenate([positions, np.tile(positions[-1][None], (50,1))], axis=0)
    joint_names = [str(x) for x in data["joint_names"]]
    piece_pose_to_robot = np.asarray(data["piece_pose_to_robot"], dtype=float)
    obj_path = str(data["obj_path"])
    seam_mid = np.asarray(data["seam_mid"], float) if "seam_mid" in data else None
    seam_bisector = np.asarray(data["seam_bisector"], float) if "seam_bisector" in data else None
    print("轨迹点数:", positions.shape, " 关节:", joint_names)

    cfg = load_config()
    compat.apply_trimesh_shim()
    robot_cfg = load_yaml(cfg.robot_cfg_path)["robot_cfg"]

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # 工件：引用 part 目录的 .usd，平移 -robot_pos（基座系；机器人在原点）
    usd_obj = obj_path.replace("_watertight.obj", ".usd")
    if os.path.exists(usd_obj):
        add_reference_to_stage(usd_path=usd_obj, prim_path="/World/workpiece")
        try:
            from omni.isaac.core.prims import XFormPrim
            XFormPrim("/World/workpiece").set_world_pose(
                position=(piece_pose_to_robot[:3]).tolist(),
                orientation=(piece_pose_to_robot[3:7]).tolist())
        except Exception as e:
            print("warn: 设置工件位姿失败:", e)
        # 工件当纯视觉：关碰撞（机械臂直接穿过）+ 关刚体（不被推/不掉落，位姿恒定）
        try:
            from pxr import Usd, UsdPhysics
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            wp_prim = stage.GetPrimAtPath("/World/workpiece")
            for p in Usd.PrimRange(wp_prim):
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
                if p.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Set(False)
        except Exception as e:
            print("warn: 禁用工件物理失败:", e)
    else:
        print("warn: 未找到工件 usd:", usd_obj, "（跳过工件显示）")

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, world)

    # 焊缝中点处两面方向相加的方向（base 系，机器人在原点）：用 debug_draw 画红色箭头线
    draw = None
    if seam_mid is not None and seam_bisector is not None:
        try:
            try:
                from omni.isaac.debug_draw import _debug_draw
            except ImportError:
                from isaacsim.util.debug_draw import _debug_draw
            draw = _debug_draw.acquire_debug_draw_interface()
        except Exception as e:
            print("warn: debug_draw 不可用，跳过方向显示:", e)

    _L = 0.15  # 箭头长度(m)
    def _draw_tangent():
        if draw is None:
            return
        t = seam_bisector / (np.linalg.norm(seam_bisector) + 1e-9)
        p0 = seam_mid
        p1 = seam_mid + _L * t
        # 箭头两翼：在切向两侧各回折一段
        ref = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        side = np.cross(t, ref); side /= (np.linalg.norm(side) + 1e-9)
        h = 0.3 * _L
        w0 = p1 - h * t + h * side
        w1 = p1 - h * t - h * side
        col = (1.0, 0.1, 0.1, 1.0)
        starts = [p0.tolist(), p1.tolist(), p1.tolist()]
        ends = [p1.tolist(), w0.tolist(), w1.tolist()]
        draw.clear_lines()
        draw.draw_lines(starts, ends, [col] * 3, [4] * 3)

    world.reset()
    robot.initialize() if hasattr(robot, "initialize") else None
    idx_list = [robot.get_dof_index(j) for j in joint_names]

    # 先摆到起点
    robot.set_joint_positions(positions[0], idx_list)

    i = 0
    hold = 0
    print("开始播放（关闭窗口结束）…")
    while simulation_app.is_running():
        world.step(render=not args.headless)
        if not world.is_playing():
            continue
        _draw_tangent()
        if i < len(positions):
            robot.set_joint_positions(positions[i], idx_list)
            i += 1
        else:
            hold += 1
            if hold > args.fps * 2:     # 终点停 2 秒后循环
                i, hold = 0, 0
        if args.headless and i >= len(positions):
            break                        # 自检模式跑完即退

    print("VIZ_ISAAC_DONE")
    simulation_app.close()


if __name__ == "__main__":
    main()
