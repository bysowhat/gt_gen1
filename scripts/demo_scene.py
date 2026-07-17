"""demo：Scene API 用法示例，每个用法各封成一个函数，main 只做参数解析 + 调用。

  · demo_init_pose(args)       —— 原始：求候选初始位姿 + open3d 可视化
        （等价 scripts/plan_init_pose.py --solve-kejian2 <seam-id> [--viz]，需 GPU/curobo）
  · demo_obstacle_type2(args)  —— 给焊缝加【障碍物类型2(固定 C1 遮挡板)】+ isaacsim 可视化当前 3D 场景
        （等价 scripts/viz_seam_plate_candidates_isaacsim.py）
  · demo_obstacle_type3(args)  —— 给焊缝加【障碍物类型3(开口障碍)】+ isaacsim 可视化当前 3D 场景
        （等价 scripts/viz_seam_open_box_isaacsim.py）

每个函数各自新建 Scene、互相独立。注意几点不要在【同一进程】里先后跑：
  · demo_init_pose 会初始化 curobo/torch；
  · demo_obstacle_type2 / type3 各要最先启动 isaacsim SimulationApp（一个进程只能有一个）。
默认只跑 demo_obstacle_type2，想看别的把 main 里的调用换一下即可。

运行（类型2 可视化，带显示器）：
    conda run -n env_isaaclab python scripts/demo_scene.py --seam-id 1
无显示器自检：加 --headless（spawn + 跑几帧即退，打印障碍数 + VIZ_SCENE_DONE）。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_OBJ = ("/media/a/新加卷/hanfeng/segment/A3Changfang/"
               "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part_watertight.obj")
DEFAULT_WELD_JSON = ("/media/a/新加卷/hanfeng/segment/A3Changfang/"
                     "BEAM_1aEEYa00Ed5Z4sE34qDJKu_weld_angle3.json")

# PROFILEMAIN=1 时，用 sys.settrace 逐行统计【本文件】每行代码耗时（含该行调用的其它模块耗时）。
_PROFILE_MAIN = os.environ.get("PROFILEMAIN") == "1"


class _LineProfiler:
    """逐行行级计时器（仅统计 target_file 内的帧）。

    原理：settrace 全局钩子对每个新帧的 'call' 事件被调用——只对 target_file 的帧返回本地
    tracer，其它文件返回 None（不产生逐行事件）。故某行若调用了别的模块（如 scene.compute_*），
    其耗时会在【控制权回到本文件下一行】时，被归到发起调用的那一行——正是「每行耗时」的语义。
    每个帧独立维护 (上一行号, 上一时刻)，正确处理本文件内的函数嵌套（如 _make_scene）。
    """

    def __init__(self, target_file):
        self.target_abs = os.path.abspath(target_file)
        self.stats = {}          # lineno -> [hits, total_seconds]
        self._state = {}         # frame -> [last_lineno, last_perf_counter]
        self._file_cache = {}    # co_filename -> bool(是否本文件)

    def _trace(self, frame, event, arg):
        if event == "call":
            fn = frame.f_code.co_filename
            matched = self._file_cache.get(fn)
            if matched is None:
                matched = (os.path.abspath(fn) == self.target_abs)
                self._file_cache[fn] = matched
            if not matched:
                return None                      # 非本文件：不逐行跟踪（其耗时归调用行）
            self._state[frame] = [None, time.perf_counter()]
            return self._trace
        if event == "line":
            now = time.perf_counter()
            st = self._state.get(frame)
            if st is not None:
                if st[0] is not None:
                    rec = self.stats.setdefault(st[0], [0, 0.0])
                    rec[0] += 1
                    rec[1] += now - st[1]
                st[0] = frame.f_lineno
                st[1] = now
            return self._trace
        if event == "return":
            now = time.perf_counter()
            st = self._state.pop(frame, None)
            if st is not None and st[0] is not None:
                rec = self.stats.setdefault(st[0], [0, 0.0])
                rec[0] += 1
                rec[1] += now - st[1]
            return self._trace
        return self._trace

    def __enter__(self):
        sys.settrace(self._trace)
        return self

    def __exit__(self, *exc):
        sys.settrace(None)
        self.report()
        return False

    def report(self):
        try:
            with open(self.target_abs, "r", encoding="utf-8") as f:
                src = f.readlines()
        except Exception:
            src = []
        total = sum(t for _, t in self.stats.values())
        print("\n" + "=" * 78)
        print(f"[PROFILEMAIN] 逐行耗时 {os.path.basename(self.target_abs)}"
              f"（总计 {total:.3f}s，仅列命中行）")
        print("=" * 78)
        print(f"{'行号':>5} | {'命中':>5} | {'耗时(s)':>10} | {'占比':>6} | 源码")
        print("-" * 78)
        for ln in sorted(self.stats):
            hits, t = self.stats[ln]
            pct = (100.0 * t / total) if total > 0 else 0.0
            code = src[ln - 1].rstrip("\n") if 0 < ln <= len(src) else ""
            print(f"{ln:>5} | {hits:>5} | {t:>10.4f} | {pct:>5.1f}% | {code.strip()}")
        print("-" * 78)
        top = sorted(self.stats.items(), key=lambda kv: -kv[1][1])[:10]
        print("[PROFILEMAIN] 耗时 Top 10 行：")
        for ln, (hits, t) in top:
            code = src[ln - 1].rstrip("\n") if 0 < ln <= len(src) else ""
            print(f"  L{ln:<4} {t:>9.4f}s  {code.strip()}")
        print("=" * 78 + "\n")


def _make_scene(args):
    from gt_gen.scene import Scene
    scene = Scene(cfg="configs/default.yaml",
                 workpiece_obj=args.obj,
                 weld_json=args.weld_json)
    return scene

def demo_init_pose(args):
    """原始示例：求候选初始位姿（需 GPU/curobo），可选 open3d 逐个可视化。"""
    from gt_gen.scene_viz import Open3DSceneVisualizer

    scene = _make_scene(args)
    scene._set_cur_seam(9)

    scene.plan_init_pose()

    cands = scene.init_pose_candidates.get(scene.seam_id, {"forehand": [], "backhand": []})
    n_fore, n_back = len(cands["forehand"]), len(cands["backhand"])
    if n_fore + n_back == 0:
        print("[demo] 求解失败：无候选初始位姿")
        return
    print(f"[demo] 候选初始位姿 {n_fore + n_back} 个（正手 {n_fore} / 反手 {n_back}）")

    # Open3DSceneVisualizer(scene).show_init_poses()


def demo_obstacle_type2(args):
    """障碍物类型2示例：焊缝旁遮挡板（固定 C1，参数读 obstacle_placement.type2）+ isaacsim 可视化。"""
    from gt_gen.scene_viz import Open3DSceneVisualizer

    scene = _make_scene(args)
    scene._set_cur_seam(9)

    spec2 = scene.add_obstacle_type2()
    print(f"[demo] 类型2 遮挡板 1 个：{spec2.meta['candidate']}({spec2.kind})，启动 isaacsim 可视化…")
    Open3DSceneVisualizer(scene).show_scene_isaacsim(headless=args.headless)


def demo_obstacle_type3(args):
    """障碍物类型3示例：把焊缝包住的开口障碍（参数读 obstacle_placement.type3）+ isaacsim 可视化。"""
    from gt_gen.scene_viz import Open3DSceneVisualizer

    scene = _make_scene(args)
    scene._set_cur_seam(9)

    spec3 = scene.add_obstacle_type3()
    print(f"[demo] 类型3 开口障碍 1 个：{spec3.kind}，启动 isaacsim 可视化…")
    Open3DSceneVisualizer(scene).show_scene_isaacsim(headless=args.headless)

def demo_init_poses(args):
    """多候选初始位姿【同屏铺网格】isaacsim 可视化（两进程模式）。

    进程①（curobo，会污染 warp）：求候选 + 存盘——
        scene = _make_scene(args); scene._set_cur_seam(9)
        scene.plan_init_pose(); scene.save('/media/a/新加卷/tempt/4/scene_init.pkl')
    进程②（本函数，干净进程）：load + 铺网格可视化（每格 工件 + 机械臂摆到候选 q）。
    """
    from gt_gen.scene import Scene
    from gt_gen.scene_viz import Open3DSceneVisualizer

    scene = Scene.load('/media/a/新加卷/tempt/4/scene_init.pkl')
    Open3DSceneVisualizer(scene).show_init_poses_isaacsim(
        hand=args.hand, top_n=args.top_n, spacing=args.spacing, headless=args.headless)


def obstacle_type2_demo_main(args):
    import random
    from gt_gen.scene_viz import Open3DSceneVisualizer


    # from gt_gen.repro_plan_joint import replay
    # seg = replay("plan_joint_case.pkl")


    scene = _make_scene(args)
    scene._set_cur_seam(106)#
    # if random.random() < 0.7:  # 70% 概率添加遮挡板
    #     scene.add_obstacle_type2()
    scene.add_obstacle_type2()
    scene.plan_init_pose_fast(include_obstacles=False, diagnostic=True)
    scene.save('/media/a/新加卷/tempt/4/scene1.pkl')

    # # scene.plan_init_pose()
    # # scene.save('/media/a/新加卷/tempt/4/scene1.pkl')
    # num_init_pose = scene.num_init_pose()
    # if num_init_pose == 0:
    #     # 放宽障碍物条件，不考虑障碍物碰撞
    #     scene.plan_init_pose(include_obstacles=False)
    # scene.save('/media/a/新加卷/tempt/4/scene1.pkl')
    # # if num_init_pose == 0:
    # #     # raise NotImplementedError()
    # #     # 继续放宽障碍物条件，将障碍物挪到更远一点的位置
    # #     # scene.grow_obstacle_n(n_cm, spec=None)
    # #     print(1)
    # # scene.save('/media/a/新加卷/tempt/4/scene1.pkl')

    # # scene.seam_ids_by_length()  169,88,164,106
    # fflag = scene.compute_pose_and_plan_path(hand="forehand")
    # scene.save('/media/a/新加卷/tempt/4/scene1.pkl')
    # bflag = scene.compute_pose_and_plan_path(hand="backhand")
    # scene.save('/media/a/新加卷/tempt/4/scene2.pkl')
    # print(1)


    from gt_gen.scene import Scene
    scene = Scene.load('/media/a/新加卷/tempt/4/scene1.pkl')
    # Open3DSceneVisualizer(scene).show_scene_isaacsim(headless=args.headless, goal_arm_index=[0,1])
    # Open3DSceneVisualizer(scene).show_init_poses()
    # Open3DSceneVisualizer(scene).show_joint_table_ee()

    # # Open3DSceneVisualizer(scene).show_init_poses_debug(4, sort_by_seam_x=True)
    # # Open3DSceneVisualizer(scene).show_goal_pose_collision("backhand", 0)
    # # Open3DSceneVisualizer(scene).show_goal_pose(hand="backhand", variant=0, goal_index=1)
    Open3DSceneVisualizer(scene).show_init_pose_prefilter(stage=2)#stage=1/2/3


    # Open3DSceneVisualizer(scene).show_seam(106)
    # # Open3DSceneVisualizer(scene).show_seam_all_isaacsim()
    # Open3DSceneVisualizer(scene).show_trajectory_isaacsim(traj_index=0, headless=args.headless)


def obstacle_type1_demo_main(args):
    '''
        先规划出一条轨迹
        在扫掠空间中添加障碍物type1
        重新规划1条轨迹
    '''
    import random
    from gt_gen.scene import Scene
    from gt_gen.scene_viz import Open3DSceneVisualizer

    # # from gt_gen.repro_plan_joint import replay
    # # seg = replay("plan_joint_case.pkl")

    scene = _make_scene(args)
    scene._set_cur_seam(58)#
    scene.plan_init_pose_fast(verbose=True)
    scene.save('/media/a/新加卷/tempt/4/scene1.pkl')


    # # scene.seam_ids_by_length()  89,39,58,92
    # scene = Scene.load('/media/a/新加卷/tempt/4/scene1.pkl')
    fflag = scene.compute_pose_and_plan_path(hand="forehand")
    scene.save('/media/a/新加卷/tempt/4/scene1.pkl')
    scene = Scene.load('/media/a/新加卷/tempt/4/scene1.pkl')
    bflag = scene.add_obstacle_type1(link='Link3', 
                             hand='forehand', 
                             index=1, 
                             entry_index=0,
                             otype='plate')
    print(1)
    scene.save('/media/a/新加卷/tempt/4/scene2.pkl')

    # # # # scene = Scene.load('/media/a/新加卷/tempt/4/scene2.pkl')
    # fflag = scene.compute_pose_and_plan_path(hand="forehand")
    # scene.save('/media/a/新加卷/tempt/4/scene2.pkl')

    # print(1)


    # scene = Scene.load('/media/a/新加卷/tempt/4/scene2.pkl')
    # # Open3DSceneVisualizer(scene).show_scene_isaacsim(headless=args.headless, goal_arm_index=[0,1])
    # # Open3DSceneVisualizer(scene).show_init_poses()
    # # Open3DSceneVisualizer(scene).show_joint_table_ee()

    # # # Open3DSceneVisualizer(scene).show_init_poses_debug(4, sort_by_seam_x=True)
    # # # Open3DSceneVisualizer(scene).show_goal_pose_collision("backhand", 0)
    # # # Open3DSceneVisualizer(scene).show_goal_pose(hand="backhand", variant=0, goal_index=1)
    # # Open3DSceneVisualizer(scene).show_init_pose_prefilter(stage=2)#stage=1/2/3


    # # Open3DSceneVisualizer(scene).show_seam(106)
    # # # Open3DSceneVisualizer(scene).show_seam_all_isaacsim()
    # Open3DSceneVisualizer(scene).show_trajectory_isaacsim(traj_index=0, headless=args.headless)

def main():
    ap = argparse.ArgumentParser(description="Scene API demo：初始位姿求解 / 障碍物类型2 / 障碍物类型3 + isaacsim 可视化")
    ap.add_argument("--obj", default=DEFAULT_OBJ, help="工件 mesh（_part.obj / _watertight.obj）")
    ap.add_argument("--weld-json", default=DEFAULT_WELD_JSON, help="焊缝 _weld_angle3.json")
    ap.add_argument("--headless", action="store_true", help="无显示器自检：spawn 后跑几帧即退")
    ap.add_argument("--goal-index", type=int, default=None,
                    help="再画一条到达第几个 goal 观测位姿的机械臂，并打印其三分碰撞（自碰撞/工件/障碍）")
    ap.add_argument("--hand", default=None, choices=["forehand", "backhand"],
                    help="demo_init_poses：只铺一只手的候选（缺省正手+反手都铺）")
    ap.add_argument("--top-n", type=int, default=3,
                    help="demo_init_poses：每只手取前 N 个候选（默认 3）")
    ap.add_argument("--spacing", type=float, default=2.5,
                    help="demo_init_poses：网格格心节距（米，默认 2.5；1m 会重叠）")
    args = ap.parse_args()

    # 默认跑障碍物类型2；想看别的换成下面对应调用（勿与本调用同进程先后跑，见模块 docstring）
    # demo_obstacle_type2(args)
    # demo_obstacle_type3(args)
    # demo_init_pose(args)
    # demo_init_poses(args)   # 多候选初始位姿同屏铺网格（先另进程 plan_init_pose + save）

    demo = obstacle_type1_demo_main
    # demo = obstacle_type2_demo_main
    if _PROFILE_MAIN:
        with _LineProfiler(__file__):       # PROFILEMAIN=1：逐行计时（含各行调用的耗时）
            demo(args)
    else:
        demo(args)

if __name__ == "__main__":
    main()
