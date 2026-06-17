"""把 place_obstacles.py 产出的「直接规划好的绕行轨迹」改造成「边看边走」的探索 GT。

place_obstacles.py 存的 npz 里 detour/默认轨迹都是【全知】一把规划出来的（已知工件+障碍）。
本脚本读同一个 npz，但改用 verify_step10 的主循环 generate_gt 重新生成轨迹——机械臂只敢走
「亲眼看过是空的」区域，边走边拍、已知区像水面扩大直到淹没目标。与 verify_step10 唯一的关键
区别：真值世界里的障碍 = 工件 **+ place_obstacles 放的障碍物**（不再只有工件）。

四个世界（同 main_loop 文档）：
  - h_truth   : MESH 真值世界，含【工件 mesh + 障碍原语转的 mesh】——全知教练（P*/NBV/候选避障）；
  - h_expl    : VOXEL 纯三态探索世界（无 mesh）——机械臂真正规划/执行，GT 来自这里；
  - truth_scene: base 系 trimesh，含【工件 + 障碍】——raycast 几何源（看到障碍才会标 OCCUPIED）；
  - vm        : 三态体素图 + 初始 FREE 圆柱（冷启动立足之地）。

goal 不再从 seam pkl 重算：直接 FK npz 里默认轨迹的终点（那就是 place_obstacles 用的 standoff 目标，
且一定可达），自洽、无需再带 --seam。

产出 npz 字段与 place_obstacles 对齐（额外把 GT 同时写进 positions / detour_positions），
可直接喂 scripts/viz_placed_obstacle_isaacsim.py 回放——工件 + 障碍 + 这条边看边走 GT 一起显示。

运行（本机 conda，无需 Isaac）：
    conda run -n env_isaaclab --no-capture-output python -u scripts/place_obstacles_to_gt.py \
        --out_dir /tmp/placed_obstacles --index 0 --out /tmp/placed_gt/scene_00_gt.npz
    # 或直接指定场景：--scene /tmp/placed_obstacles/scene_00_Link2_plate.npz
可视化：
    conda run -n env_isaaclab python scripts/viz_placed_obstacle_isaacsim.py \
        --scene /tmp/placed_gt/scene_00_gt.npz --which default
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _deserialize_prims(arr):
    """place_obstacles 存的障碍 dict 列表 → Box/Tube 原语列表（与 _serialize_prims 互逆）。"""
    from gt_gen import obstacles as ob
    prims = []
    for raw in arr:
        p = dict(raw)
        if p["kind"] == "box":
            prims.append(ob.Box(name=str(p["name"]), dims=list(map(float, p["dims"])),
                                pose=list(map(float, p["pose"])), color=list(map(float, p["color"]))))
        else:
            prims.append(ob.Tube(name=str(p["name"]), radius=float(p["radius"]),
                                 height=float(p["height"]), pose=list(map(float, p["pose"])),
                                 color=list(map(float, p["color"]))))
    return prims


def _prim_to_trimesh(p):
    """单个 Box/Tube 原语 → base 系 trimesh.Trimesh（pose=[x,y,z,qw,qx,qy,qz]）。

    Box  → create_box(extents=dims)（居中于原点）；Tube → create_cylinder(轴沿局部 +Z，居中），
    与 obstacles.py 的几何约定一致；再按 pose 的旋转(wxyz)+平移整体变换到 base 系。
    """
    import trimesh
    from gt_gen import obstacles as ob
    from scipy.spatial.transform import Rotation as R

    if isinstance(p, ob.Box):
        m = trimesh.creation.box(extents=list(map(float, p.dims)))
    else:                                                       # Tube：轴沿 +Z
        m = trimesh.creation.cylinder(radius=float(p.radius), height=float(p.height))
    qw, qx, qy, qz = p.pose[3:7]
    T = np.eye(4)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()       # wxyz -> xyzw
    T[:3, 3] = np.asarray(p.pose[:3], float)
    m.apply_transform(T)
    return m


def build_scene(args):
    """读场景 npz → 建 h_truth(MESH 工件+障碍) + h_expl(VOXEL) + truth_scene(工件+障碍) + vm + cam + goal。"""
    from gt_gen.config import load_config
    from gt_gen import curobo_iface as ci
    from gt_gen import obstacle_placement as opl
    from gt_gen.voxmap import build_roi_voxmap
    from gt_gen.sensor import load_camera_model, load_truth_scene
    from gt_gen.init_free import set_initial_free_cylinder
    from curobo.geom.types import Mesh
    from curobo.geom.sdf.world import CollisionCheckerType
    import trimesh

    data = np.load(args.scene, allow_pickle=True)
    positions = np.asarray(data["positions"], float)            # 默认轨迹（终点即 standoff 目标）
    # 预规划绕行轨迹：place_obstacles 已在「障碍外扩 buffer」世界里成功规划好、且验证过可行/绕开。
    # 第一轮主循环直接拿它当 P*（不重新规划）——绕开「同空间这里 plan 却随机失败」。无则退回默认轨迹。
    p_star_init = (np.asarray(data["detour_positions"], float)
                   if "detour_positions" in data.files else positions)
    joint_names = [str(x) for x in data["joint_names"]]
    mesh_pose = np.asarray(data["piece_pose_to_robot"], float).tolist()
    obj_path = str(data["obj_path"])
    prims = _deserialize_prims(list(data["obstacle_prims"]))
    link = str(data["link"]); otype = str(data["otype"])
    print(f"场景: {os.path.basename(args.scene)}  link={link} otype={otype} 障碍原语={len(prims)}")

    cfg = load_config()

    # h_truth：MESH 世界 = 工件 mesh + 障碍原语转 mesh（全知教练，含 place_obstacles 放的障碍）
    workpiece = Mesh(name="workpiece", file_path=obj_path, pose=mesh_pose)
    world = opl.build_world(workpiece, prims, buffer_m=0.0)      # 真值=真实尺寸，不加 buffer
    print(f"初始化 h_truth（MESH，工件 + {len(prims)} 障碍原语）...")
    h_truth = ci.init_curobo(cfg, world_model=world, collision_checker_type=CollisionCheckerType.MESH,
                             position_threshold=0.05, rotation_threshold=0.5)
    # h_truth_plan：仅给 P* 规划用——【障碍外扩 buffer、工件不变】（同 place_obstacles 的 world_inflated），
    # 使 P* 与真实障碍留间隙；buffer 从 cfg.obstacle_placement.obstacle_buffer_m 读，0 则与 h_truth 等价。
    buf = float(cfg.obstacle_placement.get("obstacle_buffer_m", 0.0))
    if buf > 0.0 and prims:
        world_inflated = opl.build_world(workpiece, prims, buffer_m=buf)   # 工件不加 buffer，只障碍外扩
        print(f"初始化 h_truth_plan（MESH，障碍外扩 buffer={buf}m，工件不变，仅供 P* 规划）...")
        h_truth_plan = ci.init_curobo(cfg, world_model=world_inflated,
                                      collision_checker_type=CollisionCheckerType.MESH,
                                      position_threshold=0.05, rotation_threshold=0.5)
    else:
        h_truth_plan = None
        print(f"buffer={buf}m（≤0 或无障碍）→ P* 规划退回用 h_truth（无 buffer）")
    print("初始化 h_expl（VOXEL，纯三态，无 mesh）...")
    h_expl = ci.init_curobo(cfg)

    # truth_scene：工件 trimesh + 障碍 trimesh（raycast 看得到障碍才会把它标 OCCUPIED）
    work_mesh = load_truth_scene(obj_path, mesh_pose=mesh_pose)
    scene = trimesh.util.concatenate([work_mesh] + [_prim_to_trimesh(p) for p in prims])
    print(f"truth_scene 顶点={len(scene.vertices)} 面={len(scene.faces)}（工件+障碍合并）")

    # goal_pose：FK 默认轨迹终点（= place_obstacles 用的 standoff 目标，可达且自洽）
    eep, eeq, _ = ci.fk(h_truth, positions[-1].tolist())
    goal_pose = (eep.tolist(), eeq.tolist())
    print(f"goal=FK(默认轨迹终点)  pos={np.round(eep, 3)}")

    cam = load_camera_model(cfg)
    vm = build_roi_voxmap(cfg)
    n = set_initial_free_cylinder(h_truth, vm, config=cfg)
    print(f"初始圆柱 FREE 体素={n} (R={cfg.init_free_cyl_radius}m h={cfg.init_free_cyl_height}m)")

    return dict(cfg=cfg, h_truth=h_truth, h_expl=h_expl, h_truth_plan=h_truth_plan, vm=vm, cam=cam,
                scene=scene, goal_pose=goal_pose, data=data, joint_names=joint_names, mesh_pose=mesh_pose,
                obj_path=obj_path, prims_raw=list(data["obstacle_prims"]), link=link, otype=otype,
                p_star_init=p_star_init)


def run_generate_gt(ctx):
    """跑 generate_gt 主循环产边看边走 GT，打印 status / 路点数 / 进展。"""
    from gt_gen.main_loop import generate_gt

    print("\n== generate_gt 主循环（边看边走，真值含工件+障碍） ==")
    GT, status, info = generate_gt(ctx["h_truth"], ctx["h_expl"], ctx["vm"], ctx["scene"],
                                   ctx["goal_pose"], camera_model=ctx["cam"],
                                   h_truth_plan=ctx.get("h_truth_plan"),
                                   p_star_init=ctx.get("p_star_init"))
    print(f"  status={status}  GT 路点={len(GT)}  轮数={info['rounds']}  P*长={info['P_len']}")
    print(f"  status 轨迹={info['status_seq']}")
    print(f"  |B| 轨迹={info['n_B']}")
    if status != "reached":
        print(f"  warn: 主循环未到达目标（status={status}）——仍存盘已走出的部分 GT 供检查")
    return GT, status, info


def save_gt(ctx, GT, out):
    """存 npz：GT 同时写进 positions / detour_positions，并带上工件+障碍信息，
    字段与 place_obstacles 输出对齐 → 可直接喂 viz_placed_obstacle_isaacsim.py（工件+障碍+GT 一起显示）。"""
    data = ctx["data"]
    positions = np.asarray([np.asarray(q, float) for q in GT])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(
        out,
        positions=positions,                                    # 边看边走 GT（--which default 回放它）
        detour_positions=positions,                             # 同一条 GT（--which detour 也回放它）
        detour_positions_all=np.array([positions], dtype=object),
        n_detour=1,
        joint_names=np.array(ctx["joint_names"]),
        retract=np.asarray(data["retract"], float) if "retract" in data.files
        else np.asarray(ctx["cfg"].retract_config, float),
        obstacle_prims=np.array(ctx["prims_raw"], dtype=object),
        anchor=np.asarray(data["anchor"], float) if "anchor" in data.files else np.zeros(3),
        link=ctx["link"], otype=ctx["otype"],
        piece_pose_to_robot=np.asarray(ctx["mesh_pose"], float),
        obj_path=ctx["obj_path"],
        robot_usd=str(data["robot_usd"]) if "robot_usd" in data.files else "",
        seam_mid=np.asarray(data["seam_mid"], float) if "seam_mid" in data.files else np.zeros(3),
        seam_bisector=np.asarray(data["seam_bisector"], float) if "seam_bisector" in data.files
        else np.zeros(3),
        dt=float(data["dt"]) if "dt" in data.files else 0.02,
    )
    print(f"  ✓ GT 已存: {out}  (positions={positions.shape})")
    print(f"    可视化: conda run -n env_isaaclab python scripts/viz_placed_obstacle_isaacsim.py "
          f"--scene {out} --which default")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None, help="场景 npz 路径（优先）")
    ap.add_argument("--out_dir", default="/tmp/placed_obstacles", help="场景目录（配 --index）")
    ap.add_argument("--index", type=int, default=0, help="--out_dir 内第几个场景（按文件名排序）")
    ap.add_argument("--out", default=None, help="GT 存盘路径（npz）；缺省按场景名放 /tmp/placed_gt/")
    args = ap.parse_args()

    if args.scene is None:
        hits = sorted(glob.glob(os.path.join(args.out_dir, "*.npz")))
        if not hits:
            raise FileNotFoundError(f"{args.out_dir} 下没有场景 npz")
        args.scene = hits[max(0, min(args.index, len(hits) - 1))]
    if args.out is None:
        base = os.path.splitext(os.path.basename(args.scene))[0]
        args.out = os.path.join("/tmp/placed_gt", f"{base}_gt.npz")

    ctx = build_scene(args)
    GT, status, _ = run_generate_gt(ctx)
    save_gt(ctx, GT, args.out)

    print(f"\nPLACE_OBSTACLES_TO_GT_{'OK' if status == 'reached' else 'PARTIAL'} "
          f"[status={status}, 路点={len(GT)}]")


if __name__ == "__main__":
    main()
