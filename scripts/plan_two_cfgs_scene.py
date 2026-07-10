"""在【Scene pkl 的真实碰撞世界】(工件 + 已放障碍) 上规划 cur_cfg → target_cfg。

融合两脚本：
  · 数据源同 scripts/place_obstacles_to_gt2.py —— 输入 --scene 读 Scene pkl，复用其
    build_worlds 由 Scene 重建机械臂 / 工件 / 障碍信息，拿到真实碰撞世界：
        h_truth (MESH，工件 + 障碍实体，按 workpiece_pose7 摆到 base 系)  —— curobo 后端在此规划；
        world   (同一批 mesh 的 WorldConfig)                              —— stomp 后端直接吃这个 MESH 世界。
  · 规划流程同 scripts/plan_two_cfgs.py —— 给定两个关节角，check_state 核两端 feasible、
    explain_endpoints 区分失败成因，再按 default.yaml 的 planner.backend 分派 curobo / stomp
    规划，成功则子进程拉起 Isaac Sim 回放这条轨迹。

与 plan_two_cfgs.py 的唯一区别是碰撞世界来源：那里是 h_expl(VOXEL 三态 + 初始 FREE 圆柱)，
这里是 h_truth(MESH，含真实工件 + 障碍)。两端 feasible 是必要条件，plan_to_config 要求整条
插值路径都无碰撞。

两个关节角均由文件顶部常量 CUR_CFG / TARGET_CFG 指定（按需改）。
传 --target-from-scene 可改用 scene 里 goal pose 的关节角当【终点】（需先 compute_goal_pose）。

运行（本机 conda，规划进程不得先启动 SimulationApp）：
    conda run -n env_isaaclab --no-capture-output python -u scripts/plan_two_cfgs_scene.py \
        --scene '/media/a/新加卷/tempt/4/scene1.pkl'
  规划成功后自动拉起 Isaac Sim 回放（同文件子进程，--viz-only）：把轨迹追加进 scene 另存 pkl，
  子进程 Scene.load 后复用 scene_viz.show_trajectory_isaacsim —— 同屏显示工件 + 障碍 + 机械臂沿轨迹运动；
  --no-viz 关可视化、--headless 无头自检、--max-attempts N 调 curobo 尝试次数、
  --no-obstacles 真值世界不并障碍(仅工件)。
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))   # 复用 place_obstacles_to_gt2.build_worlds

DEFAULT_SCENE = "/media/a/新加卷/tempt/4/scene1.pkl"

# 两个关节角均由用户在此指定（按需改）。--target-from-scene 时改用 scene goal pose 的关节角覆盖 TARGET_CFG。
# CUR_CFG = [1.5707824230194092, -2.617993878, 1.3613484541522425,-1.021018, -1.5707963267948966, 3.14159]
CUR_CFG = [1.5707824230194092, -2.617993878, 1.3613484541522425, -1.021018, -1.5707963267948966, 3.14159]
TARGET_CFG = [1.2000235319137573, -0.8123235702514648, 1.0760598182678223, -1.2892558574676514, -1.382904291152954, -0.7342060804367065]


def _parse_args():
    ap = argparse.ArgumentParser(description="在 Scene pkl 真实碰撞世界上规划两个关节角之间的路径")
    ap.add_argument("--scene", default=DEFAULT_SCENE, help="Scene pkl（demo_scene.demo_main 存）")
    ap.add_argument("--no-obstacles", action="store_true", help="真值世界不并入障碍（仅工件）")
    ap.add_argument("--target-from-scene", action="store_true",
                    help="目标改用 scene goal pose 的关节角（需先 compute_goal_pose）")
    ap.add_argument("--variant", type=int, default=0, help="--target-from-scene 时 joints 的变体 K")
    ap.add_argument("--goal-index", type=int, default=0, help="--target-from-scene 时观测位姿的第几个合格解")
    ap.add_argument("--no-viz", action="store_true", help="规划成功也不拉起 Isaac Sim 可视化")
    ap.add_argument("--headless", action="store_true", help="可视化用无头模式（跑通即退，自检用）")
    ap.add_argument("--fps", type=int, default=30, help="可视化回放帧率")
    ap.add_argument("--scene-out", default="/tmp/plan_two_cfgs_scene_viz.pkl",
                    help="把规划轨迹追加进 scene 后另存的 pkl（供子进程回放，含工件+障碍）")
    ap.add_argument("--max-attempts", type=int, default=200, help="plan_to_config 最大规划尝试次数")
    # 内部用：子进程回放入口。SimulationApp 必须在导入 cuRobo 之前最先建，而规划进程已加载 cuRobo，
    # 无法同进程再开 Isaac；故规划成功后用此 flag 重入本文件起一个干净子进程。
    ap.add_argument("--viz-only", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args()


# ============================ 模式一：Isaac Sim 回放（子进程入口）============================
def run_viz(scene_pkl, headless, fps):
    """回放 scene_pkl 里最新一条轨迹，【同屏显示工件 + 障碍 + 机械臂】。

    本函数被 `--viz-only` 子进程调用，是该进程做的【第一件事】：直接复用 scene_viz 的
    show_trajectory_isaacsim（内部先建 SimulationApp 再导入 torch/cuRobo，顺序不可换），故本进程
    在此之前【绝不能】导入 cuRobo/torch。工件按 workpiece_pose7、障碍按 T_workpiece_in_base 摆到
    base 系，机械臂沿轨迹逐帧运动 —— 与 place_obstacles_to_gt2 回放同一套渲染。
    """
    from gt_gen.scene import Scene
    from gt_gen.scene_viz import Open3DSceneVisualizer

    scene = Scene.load(scene_pkl)
    Open3DSceneVisualizer(scene).show_trajectory_isaacsim(traj_index=-1, headless=headless, fps=fps)
    print("VIZ_TWO_CFGS_SCENE_DONE")


def _launch_viz_subprocess(scene, traj, cur_cfg, target_cfg, scene_out, headless, fps):
    """把规划轨迹作为一条 trajectory 追加进 scene 并另存 pkl，再用同环境解释器重入本文件的
    --viz-only 子进程回放（子进程 Scene.load 后 show_trajectory_isaacsim，含工件 + 障碍）。"""
    import numpy as np
    positions = np.asarray(traj, float)
    entry = dict(positions=positions, status="reached",
                 cur_joints=np.asarray(cur_cfg, float), goal_joints=np.asarray(target_cfg, float),
                 goal_source="plan_two_cfgs_scene", goal_index=0, variant=0)
    scene.add_trajectory_entry(entry)
    scene.save(scene_out)
    print(f"轨迹已追加进 scene 并存盘：{scene_out}（{len(positions)} 点）")

    cmd = [sys.executable, os.path.abspath(__file__), "--viz-only",
           "--scene-out", scene_out, "--fps", str(fps)]
    if headless:
        cmd.append("--headless")
    print("拉起 Isaac Sim 可视化（子进程）：", " ".join(cmd))
    subprocess.run(cmd, check=False)


# ============================ 模式二：规划（默认入口）============================
def run_plan(args):
    import numpy as np
    from gt_gen import curobo_iface as ci
    from gt_gen.config import load_config
    from gt_gen.scene import Scene
    # 复用 place_obstacles_to_gt2 的世界重建（工件 + 障碍 → h_truth(MESH) / world(WorldConfig)）
    from place_obstacles_to_gt2 import build_worlds, goal_joints_from_scene

    scene = Scene.load(args.scene)
    if scene.cur_init_pose is None:
        raise RuntimeError("pkl 未设当前 init pose（需 set_init_pose 后再 save）")

    W = build_worlds(scene, include_obstacles=not args.no_obstacles)
    cfg = load_config()                               # 从 default.yaml 读，不用 scene 里存的 cfg
    h_truth = W["h_truth"]                            # MESH：工件 + 障碍，真实碰撞世界
    world = W["world"]                                # 同一批 mesh 的 WorldConfig（供 stomp）

    # 两个关节角均由文件顶部常量指定：起点=CUR_CFG，终点=TARGET_CFG，
    # 或 --target-from-scene 改用 goal pose 的关节角当终点
    cur_cfg = list(CUR_CFG)
    if args.target_from_scene:
        if scene.goal_poses.get(scene.seam_id) is None:
            raise RuntimeError("pkl 无 goal pose（--target-from-scene 需先 compute_goal_pose）")
        target_cfg, dbg = goal_joints_from_scene(scene, args.variant, args.goal_index)
        print(f"[scene] 目标=goal pose 关节角#变体{dbg['variant']}/{dbg['K']} 位姿{dbg['goal_index']}/{dbg['B']}")
    else:
        target_cfg = list(TARGET_CFG)
    print(f"[scene] cur_cfg(CUR_CFG)   ={np.round(cur_cfg, 4)}")
    print(f"[scene] target_cfg          ={np.round(target_cfg, 4)}\n")

    # 1) 分别核两端构型 feasible（碰撞判定恒用 cuRobo MESH handle=h_truth，与规划后端无关）
    cf, cc = ci.check_state(h_truth, cur_cfg)
    tf, tc = ci.check_state(h_truth, target_cfg)
    print(f"check_state cur_cfg   : feasible={cf} constraint={cc:.4f}")
    print(f"check_state target_cfg: feasible={tf} constraint={tc:.4f}")
    print(f"explain_endpoints     : {ci.explain_endpoints(h_truth, cur_cfg, target_cfg)}\n")

    # 2) 规划 cur → target：后端由 default.yaml 的 planner.backend 决定（curobo | stomp），
    #    与 main_loop._move_to 同一套分派。两后端均产出 (T,dof) numpy 轨迹，复用同一回放。
    backend = cfg.planner_backend
    print(f"规划后端：{backend}（planner.backend）")
    if backend == "stomp":
        from gt_gen import stomp_iface as si
        from curobo.geom.sdf.world import CollisionCheckerType
        # world 是 MESH WorldConfig（工件 + 障碍），STOMP 直接吃 MESH 检查器
        traj = si.plan_joint_single(cfg, world=world, cur_cfg=cur_cfg, target_cfg=target_cfg,
                                    checker_type=CollisionCheckerType.MESH)
        ok = traj is not None
    else:
        res = ci.plan_to_config(h_truth, cur_cfg, target_cfg, max_attempts=args.max_attempts)
        ok = res is not None and bool(res.success.item())
        traj = res.get_interpolated_plan().position.detach().cpu().numpy() if ok else None
        if not ok:
            status = getattr(res, "status", None) if res is not None else "res=None"
            print(f"plan_to_config 失败：status={status}")

    if ok:
        print(f"规划成功：轨迹 {traj.shape[0]} 点")
        print("PLAN_TWO_CFGS_SCENE_OK [reached]")
        # 规划成功 → 追加轨迹进 scene 后子进程回放（工件 + 障碍 + 机械臂同屏；--no-viz 可关）
        if not args.no_viz:
            _launch_viz_subprocess(scene, traj, cur_cfg, target_cfg, args.scene_out,
                                   args.headless, args.fps)
    else:
        print(f"规划失败（backend={backend}）")
        print(f"  成因诊断：{ci.explain_endpoints(h_truth, cur_cfg, target_cfg)}")
        print("PLAN_TWO_CFGS_SCENE_FAIL")
        # 规划失败也回放：轨迹退化为 CUR_CFG 单点（同屏看工件 + 障碍 + 机械臂当前构型；--no-viz 可关）
        if not args.no_viz:
            _launch_viz_subprocess(scene, [cur_cfg], cur_cfg, target_cfg, args.scene_out,
                                   args.headless, args.fps)


if __name__ == "__main__":
    _args = _parse_args()
    if _args.viz_only:                # 子进程：只回放（不碰 cuRobo 规划）
        run_viz(_args.scene_out, _args.headless, _args.fps)
    else:                             # 默认：规划 → 成功则拉起可视化
        run_plan(_args)
