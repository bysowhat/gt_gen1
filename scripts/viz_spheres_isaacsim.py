"""在 Isaac Sim 里叠加显示「机械臂 USD + cuRobo 碰撞球」，用来确认碰撞模型。

碰撞球来自 motion_gen.kinematics.get_robot_as_spheres(q)（与规划用的完全一致）。
球做成半透明，能透过去看到机械臂本体。

运行（带显示器）：
    # 静态：默认摆到 retract_config
    conda run -n env_isaaclab python scripts/viz_spheres_isaacsim.py
    # 指定关节角（弧度，逗号分隔，按 joint_names 顺序）
    conda run -n env_isaaclab python scripts/viz_spheres_isaacsim.py --joints "-0.78,-2.1,1.67,-0.76,1.92,0"
    # 播放一条已规划轨迹（球随之更新）
    conda run -n env_isaaclab python scripts/viz_spheres_isaacsim.py --traj /tmp/seam_traj.npz
无显示器自检：加 --headless（仅跑通流程，跑完即退）。
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
_ap.add_argument("--traj", default=None, help="可选：规划好的 npz，播放并更新球")
_ap.add_argument("--joints", default=None, help="可选：静态关节角，逗号分隔（弧度）")
_ap.add_argument("--opacity", type=float, default=0.35, help="碰撞球透明度 0~1")
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
from gt_gen import curobo_iface as ci  # noqa: E402

from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.objects import sphere  # noqa: E402
from helper import add_robot_to_scene  # noqa: E402  cuRobo 示例自带


def make_transparent_material(stage, opacity):
    """建一个半透明绿色材质，绑给所有碰撞球（RTX 下可靠出透明）。"""
    from pxr import UsdShade, Sdf
    mtl_path = Sdf.Path("/World/Looks/SphereGlass")
    mtl = UsdShade.Material.Define(stage, mtl_path)
    shader = UsdShade.Shader.Define(stage, mtl_path.AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.2, 0.85, 0.3))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    mtl.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mtl


def main():
    cfg = load_config()

    # 关节构型：traj > joints > retract
    positions = None
    if args.traj:
        data = np.load(args.traj, allow_pickle=True)
        positions = data["positions"]                    # (T, dof)
        q0 = positions[0]
    elif args.joints:
        q0 = np.array([float(x) for x in args.joints.split(",")], dtype=float)
    else:
        q0 = np.array(cfg.retract_config, dtype=float)
    print("初始关节角:", np.round(q0, 4))

    # cuRobo（全自由世界即可，这里只取 kinematics 算球）
    h = ci.init_curobo(cfg)
    joint_names = list(cfg.joint_names)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    robot_cfg = h.config.robot_cfg["robot_cfg"] if hasattr(h.config, "robot_cfg") else None
    from curobo.util_file import load_yaml
    robot_cfg = load_yaml(cfg.robot_cfg_path)["robot_cfg"]
    robot, robot_prim_path = add_robot_to_scene(robot_cfg, world)

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    mtl = make_transparent_material(stage, args.opacity)

    world.reset()
    if hasattr(robot, "initialize"):
        robot.initialize()
    idx_list = [robot.get_dof_index(j) for j in joint_names]
    robot.set_joint_positions(q0, idx_list)

    def spheres_for(q):
        qt = h.ta.to_device(list(q)).view(1, -1)
        return h.mg.kinematics.get_robot_as_spheres(qt)[0]

    sph_prims = []

    def update_spheres(q):
        from pxr import UsdShade
        sph_list = spheres_for(q)
        if not sph_prims:                       # 第一次：建 prim
            for si, s in enumerate(sph_list):
                if np.isnan(s.position[0]) or float(s.radius) <= 1e-4:
                    sph_prims.append(None)
                    continue
                sp = sphere.VisualSphere(
                    prim_path=f"/curobo/robot_sphere_{si}",
                    position=np.ravel(s.position),
                    radius=float(s.radius),
                )
                UsdShade.MaterialBindingAPI(sp.prim).Bind(mtl)
                sph_prims.append(sp)
        else:                                   # 之后：只更新位姿/半径
            for si, s in enumerate(sph_list):
                sp = sph_prims[si]
                if sp is None or np.isnan(s.position[0]):
                    continue
                sp.set_world_pose(position=np.ravel(s.position))
                sp.set_radius(float(s.radius))

    update_spheres(q0)
    print(f"碰撞球数: {sum(p is not None for p in sph_prims)}")

    i = 0
    print("显示中（关闭窗口结束）…")
    while simulation_app.is_running():
        world.step(render=not args.headless)
        if not world.is_playing():
            continue
        if positions is not None:
            q = positions[i % len(positions)]
            robot.set_joint_positions(q, idx_list)
            update_spheres(q)
            i += 1
            if args.headless and i >= len(positions):
                break
        elif args.headless:
            break                                # 静态自检：跑一帧即退

    print("VIZ_SPHERES_DONE")
    simulation_app.close()


if __name__ == "__main__":
    main()
