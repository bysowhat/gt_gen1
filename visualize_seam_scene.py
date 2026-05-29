"""Visualize a static weld-seam scene in Isaac Sim.

Inputs:
  --usd : workpiece USD (e.g. .../BEAM_xxx_part.usd)
  --pkl : seam pickle containing keys
            seam_line, seam_tangent, seam_limits,
            robot_pose (1x7), piece_pose (1x7),
            joint_angles (6,), joint_names (6 strings)

The scene contains:
  - the workpiece, transformed by piece_pose
  - the UR12e/xiaoyu robot, transformed by robot_pose, with joint_angles set
  - red spheres along seam_line
  - a green sphere at the middle seam point

Pose layout: [x, y, z, qw, qx, qy, qz].
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", required=True, help="Workpiece USD path.")
parser.add_argument("--pkl", required=True, help="Seam pkl path.")
parser.add_argument(
    "--robot_usd",
    default="/home/a/Datas/curobo/example_new_robot/urdf_12e/ur12e.usd",
    help="Robot USD path.",
)
parser.add_argument("--seam_radius", type=float, default=0.01)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch app FIRST, before importing omni / pxr.
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import pickle

import numpy as np
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage


def _set_xform(stage, prim_path: str, pose: np.ndarray) -> None:
    """Set translate + orient (quat is [w,x,y,z]) on an Xformable prim."""
    prim = stage.GetPrimAtPath(prim_path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2])))
    quat = Gf.Quatd(float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6]))
    xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(quat)


def _add_seam_markers(stage, seam_line: np.ndarray, middle_idx: int, radius: float) -> None:
    UsdGeom.Xform.Define(stage, "/World/Seam")
    for i, p in enumerate(seam_line):
        path = f"/World/Seam/p_{i:02d}"
        sph = UsdGeom.Sphere.Define(stage, path)
        sph.CreateRadiusAttr(radius)
        UsdGeom.Xformable(sph).AddTranslateOp().Set(
            Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))
        )
        color = (0.1, 1.0, 0.1) if i == middle_idx else (1.0, 0.1, 0.1)
        sph.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def main() -> None:
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    robot_pose = np.asarray(data["robot_pose"]).reshape(-1)
    piece_pose = np.asarray(data["piece_pose"]).reshape(-1)
    seam_line = np.asarray(data["seam_line"])
    joint_angles = np.asarray(data["joint_angles"], dtype=np.float32)
    joint_names = list(data["joint_names"])
    middle_idx = int(data.get("middle", len(seam_line) // 2))

    print(f"[seam] {len(seam_line)} points, middle={middle_idx}")
    print(f"[piece_pose] {piece_pose}")
    print(f"[robot_pose] {robot_pose}")
    print(f"[joint_names] {joint_names}")
    print(f"[joint_angles] {joint_angles}")

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # Lighting + ground.
    light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
    light.CreateIntensityAttr(2500.0)
    light.CreateAngleAttr(2.0)

    # Workpiece.
    piece_path = "/World/Workpiece"
    add_reference_to_stage(usd_path=args.usd, prim_path=piece_path)
    _set_xform(stage, piece_path, piece_pose)

    # Robot.
    robot_path = "/World/Robot"
    add_reference_to_stage(usd_path=args.robot_usd, prim_path=robot_path)
    _set_xform(stage, robot_path, robot_pose)

    # Seam markers.
    _add_seam_markers(stage, seam_line, middle_idx, radius=args.seam_radius)

    # Articulation handle for joint setting.
    robot = SingleArticulation(prim_path=robot_path, name="robot")
    world.scene.add(robot)
    world.reset()

    # Map provided joint_names onto the articulation's DOF order.
    dof_names = robot.dof_names
    positions = robot.get_joint_positions().copy()
    for name, angle in zip(joint_names, joint_angles):
        if name in dof_names:
            positions[dof_names.index(name)] = float(angle)
        else:
            print(f"[warn] joint '{name}' not in articulation DOFs: {dof_names}")
    robot.set_joint_positions(positions)
    # Re-apply each frame would fight physics; we just render so the pose holds.

    print("[ready] static scene loaded. Close the window to exit.")
    while simulation_app.is_running():
        # Render only — no physics step, so the robot stays where we placed it.
        simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()
