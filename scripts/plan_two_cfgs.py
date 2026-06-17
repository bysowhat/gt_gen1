"""独立验证：在 h_expl(VOXEL 纯三态探索世界) 上规划 cur_cfg → target_cfg。

复刻 main_loop._move_to 第 53 行那次规划的环境（与 verify_step10 同一套建场流程）：
  - 建 h_expl（VOXEL 全自由）；
  - 建三态 voxmap，罩初始 FREE 圆柱（参数取自 default.yaml 的 init_free 段）；
  - sync_collision_world：把 voxmap 的「非 FREE」(此刻=圆柱外全 UNKNOWN) 灌成 cuRobo 障碍场，
    即 UNKNOWN 当障碍——这正是机械臂起步时 h_expl 的真实碰撞世界；
  - check_state 分别核两端构型是否 feasible，explain_endpoints 区分失败成因；
  - plan_to_config 规划 cur_cfg → target_cfg，打印成败 + 轨迹点数。

两个构型据称「都在 init 空间中」（= 都落在初始 FREE 圆柱内）。本脚本据此核实：
两端 feasible 是必要条件，但 plan_to_config 要求【整条插值路径】都无碰撞，故两端合法
仍可能因中段摆出圆柱外(UNKNOWN=障碍)而失败——结论以 explain_endpoints + 规划结果为准。

============================ 实测结论（用本脚本 + diag_plan_two_cfgs.py 跑出）============================

【先验假设错了】最初以为是「中段摆出 init 圆柱、UNKNOWN=障碍」导致失败。把圆柱半径从
0.6 放大到 2.8（几乎整个 ROI 都 FREE，746438 格）后【依旧 TRAJOPT_FAIL】；再做对照实验：
在【完全空的世界】（fresh h_expl，根本不 sync 任何障碍）规划，默认旋钮【仍然失败】。
→ 失败纯在【轨迹优化(trajopt)】，与 init_free / 碰撞无关。所以放大圆柱没用，可把半径改回 0.6。

【对照实验表】（CUR→TGT，两端在全自由世界都 feasible、constraint=0）：
  世界      enable_graph  time_dilation  max_attempts | 结果
  全自由        true          0.5          (yaml默认)  | ❌ TRAJOPT_FAIL
  全自由        true          0.5             50       | ❌ TRAJOPT_FAIL
  全自由        true          0.25            50       | ❌ FINETUNE_TRAJOPT_FAIL
  全自由        true          0.10            80       | ✅ 839 点
  全自由        false         0.5             50       | ✅ 158 点

【真正根因】大幅关节运动 + 当前 planner 旋钮的组合：
  - 运动幅度：J0 基座 +107°、J5 腕部 −187°、J4 −61°。单次要走这么大，trajopt 在
    time_dilation=0.5(较快)下满足不了速度/加速度约束。
  - default.yaml 恰好是 enable_graph=true + time_dilation_factor=0.5，【正是上表失败的那组】。
  - 反直觉点：enable_graph=false 反而成功。自由空间里 graph(PRM) 给的折线种子绕来绕去，
    trajopt 难在限速下平滑它；无 graph 时 trajopt 拿干净的线性插值种子，大幅自由运动很好解。

【两个可行修法（各有取舍）】
  1. 调小 time_dilation_factor（推荐，安全）：如 0.1。放慢轨迹、松动力学约束 → 即使 graph
     开着也能成（表中 0.10 行）。代价：GT 轨迹更慢、点更多(839)。保留 enable_graph=true，
     不伤害「放障碍后窄通道绕行」场景。
  2. enable_graph=false：此对成功且轨迹短(158 点)。但【不建议全局关】——graph 是专为放障碍
     后的绕行/窄通道开的(见 default.yaml 注释)，全局关会拉低那些场景成功率。
  建议：不动 enable_graph，把 time_dilation_factor 降到 0.1~0.2，或在 _move_to 那次
  plan_to_config 上单独传更小的 time_dilation_factor。

运行：conda run -n env_isaaclab --no-capture-output python -u scripts/plan_two_cfgs.py
      规划成功后【自动拉起 Isaac Sim 回放该轨迹】（同一文件、子进程模式：--viz-only）；
      --no-viz 关可视化、--headless 无头自检、--max-attempts N 调尝试次数。
诊断脚本：scripts/diag_plan_two_cfgs.py（全自由世界 + 旋钮扫扫，复现上表）
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                       # 让 gt_gen 可导入（规划/回放两模式都需要）
# curobo 的 isaac_sim 示例目录（含 helper.add_robot_to_scene）。逐机器不同：
# 本地默认走下面这个；服务器用环境变量 CUROBO_ISAAC 覆盖。
CUROBO_ISAAC = os.environ.get("CUROBO_ISAAC",
                              "/home/a/Projects/Github/curobo/examples/isaac_sim")

CUR_CFG = [1.5707824230194092, -2.0071660480894984, 1.3613484541522425,
           -0.9599629205516357, -1.570770565663473, 0.0]
TARGET_CFG = [3.442432165145874, -2.472576379776001, 1.4129290580749512,
              -0.4071854054927826, -2.634472131729126, -3.2598252296447754]


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-viz", action="store_true", help="规划成功也不拉起 Isaac Sim 可视化")
    ap.add_argument("--headless", action="store_true", help="可视化用无头模式（跑通即退，自检用）")
    ap.add_argument("--fps", type=int, default=30, help="可视化回放帧率")
    ap.add_argument("--traj-out", default="/tmp/plan_two_cfgs.npz", help="成功轨迹落盘路径（供回放）")
    ap.add_argument("--max-attempts", type=int, default=200, help="plan_to_config 最大规划尝试次数")
    # 内部用：子进程回放入口。Isaac Sim 的 SimulationApp 必须在导入 cuRobo 之前最先建，而规划
    # 进程已加载 cuRobo，无法同进程再开 Isaac；故规划成功后用此 flag 重入本文件起一个干净子进程。
    ap.add_argument("--viz-only", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args()


# ============================ 模式一：Isaac Sim 回放（子进程入口）============================
def run_viz(traj_path, headless, fps):
    """纯机器人回放规划好的关节轨迹（无工件/障碍）。

    本函数被 `--viz-only` 子进程调用，是该进程做的【第一件事】：先建 SimulationApp，再导入
    其余 isaac/cuRobo 模块（顺序不可换——SimulationApp 必须早于 torch/cuRobo）。
    """
    try:
        import isaacsim  # noqa: F401
    except ImportError:
        pass
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": headless})

    import numpy as np
    sys.path.insert(0, ROOT)
    sys.path.insert(0, CUROBO_ISAAC)

    from gt_gen import compat  # noqa: F401  warp shim
    from gt_gen.config import load_config
    from curobo.util_file import load_yaml
    from omni.isaac.core import World
    from helper import add_robot_to_scene  # cuRobo 示例自带

    data = np.load(traj_path, allow_pickle=True)
    positions = np.asarray(data["positions"], dtype=float)            # (T,6)
    positions = np.concatenate([positions, np.tile(positions[-1][None], (50, 1))], axis=0)
    joint_names = [str(x) for x in data["joint_names"]]
    print("轨迹点数:", positions.shape, " 关节:", joint_names)

    cfg = load_config()
    compat.apply_trimesh_shim()
    robot_cfg = load_yaml(cfg.robot_cfg_path)["robot_cfg"]

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    robot, _ = add_robot_to_scene(robot_cfg, world)

    world.reset()
    robot.initialize() if hasattr(robot, "initialize") else None
    idx_list = [robot.get_dof_index(j) for j in joint_names]
    robot.set_joint_positions(positions[0], idx_list)                  # 先摆到起点 cur_cfg

    i = 0
    hold = 0
    print("开始播放（关闭窗口结束）…")
    while simulation_app.is_running():
        world.step(render=not headless)
        if not world.is_playing():
            continue
        if i < len(positions):
            robot.set_joint_positions(positions[i], idx_list)
            i += 1
        else:
            hold += 1
            if hold > fps * 2:           # 终点停 2 秒后循环
                i, hold = 0, 0
        if headless and i >= len(positions):
            break                         # 自检模式跑完即退
    print("VIZ_TWO_CFGS_DONE")
    simulation_app.close()


def _launch_viz_subprocess(traj, joint_names, traj_out, headless, fps):
    """把插值轨迹存 npz，再用同环境解释器重入本文件的 --viz-only 子进程回放。"""
    import numpy as np
    np.savez(traj_out, positions=np.asarray(traj, dtype=float),
             joint_names=np.asarray(joint_names, dtype=object))
    print(f"轨迹已存：{traj_out}（{len(traj)} 点）")

    cmd = [sys.executable, os.path.abspath(__file__), "--viz-only",
           "--traj-out", traj_out, "--fps", str(fps)]
    if headless:
        cmd.append("--headless")
    print("拉起 Isaac Sim 可视化（子进程）：", " ".join(cmd))
    subprocess.run(cmd, check=False)


# ============================ 模式二：规划（默认入口）============================
def run_plan(args):
    from gt_gen.config import load_config
    from gt_gen import curobo_iface as ci
    from gt_gen.voxmap import build_roi_voxmap, FREE  # noqa: F401
    from gt_gen.init_free import set_initial_free_cylinder
    from gt_gen.collision_sync import sync_collision_world
    from curobo.util_file import load_yaml

    cfg = load_config()

    print("初始化 h_expl（VOXEL，纯三态，无 mesh）...")
    h_expl = ci.init_curobo(cfg)                      # 默认 VOXEL 全自由

    vm = build_roi_voxmap(cfg)

    # n = set_initial_free_cylinder(h_expl, vm, config=cfg)   # 圆柱法不做 FK，handle 仅占位
    n = set_initial_free_cylinder(h_expl, vm, config=cfg, radius=10.6, height=11.8, z_min=-10.02)   # 圆柱法不做 FK，handle 仅占位
    print(f"初始圆柱 FREE 体素={n} (R={cfg.init_free_cyl_radius}m h={cfg.init_free_cyl_height}m "
          f"z_min={cfg.init_free_cyl_z_min}m)")

    # sync：把「非 FREE」(圆柱外全 UNKNOWN) 灌成 cuRobo 障碍 → 机械臂起步时的真实碰撞世界
    sync_collision_world(h_expl, vm)
    print(f"sync 完成（voxel_inflate_voxels={cfg.voxel_inflate_voxels}）；UNKNOWN 已当障碍\n")

    # 1) 分别核两端构型 feasible（必要条件）
    cf, cc = ci.check_state(h_expl, CUR_CFG)
    tf, tc = ci.check_state(h_expl, TARGET_CFG)
    print(f"check_state cur_cfg   : feasible={cf} constraint={cc:.4f}")
    print(f"check_state target_cfg: feasible={tf} constraint={tc:.4f}")
    print(f"explain_endpoints     : {ci.explain_endpoints(h_expl, CUR_CFG, TARGET_CFG)}\n")

    # 2) 规划 cur → target（与 _move_to 第 53 行一致）
    res = ci.plan_to_config(h_expl, CUR_CFG, TARGET_CFG, max_attempts=200, enable_graph=False)
    ok = res is not None and bool(res.success.item())
    if ok:
        traj = res.get_interpolated_plan().position.detach().cpu().numpy()
        print(f"plan_to_config 成功：轨迹 {traj.shape[0]} 点")
        print("PLAN_TWO_CFGS_OK [reached]")
        # 规划成功 → 调 Isaac Sim 可视化这条轨迹（纯机器人回放；--no-viz 可关）
        if not args.no_viz:
            joint_names = load_yaml(cfg.robot_cfg_path)["robot_cfg"]["kinematics"]["cspace"]["joint_names"]
            _launch_viz_subprocess(traj, joint_names, args.traj_out, args.headless, args.fps)
    else:
        status = getattr(res, "status", None) if res is not None else "res=None"
        print(f"plan_to_config 失败：status={status}")
        print(f"  成因诊断：{ci.explain_endpoints(h_expl, CUR_CFG, TARGET_CFG)}")
        print("PLAN_TWO_CFGS_FAIL")


if __name__ == "__main__":
    _args = _parse_args()
    if _args.viz_only:                # 子进程：只回放（不碰 cuRobo 规划）
        run_viz(_args.traj_out, _args.headless, _args.fps)
    else:                             # 默认：规划 → 成功则拉起可视化
        run_plan(_args)
