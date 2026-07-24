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

def _make_oa_scene(args):
    from gt_gen.observe_scene import ObserveAnythingScene
    scene = ObserveAnythingScene(cfg="configs/default.yaml",
                                usd_path=args.usd,
                                welds_json=args.weld_json)
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

    spec2 = scene.add_obstacle_type2(hand="forehand")   # 纯 viz 演示：随便挂一只手别桶（viz 画并集）
    print(f"[demo] 类型2 遮挡板 1 个：{spec2.meta['candidate']}({spec2.kind})，启动 isaacsim 可视化…")
    Open3DSceneVisualizer(scene).show_scene_isaacsim(headless=args.headless)


def demo_obstacle_type3(args):
    """障碍物类型3示例：把焊缝包住的开口障碍（参数读 obstacle_placement.type3）+ isaacsim 可视化。"""
    from gt_gen.scene_viz import Open3DSceneVisualizer

    scene = _make_scene(args)
    scene._set_cur_seam(9)

    spec3 = scene.add_obstacle_type3(hand="forehand")   # 纯 viz 演示：随便挂一只手别桶（viz 画并集）
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
    '''
        对【所有焊缝】、每缝【正手 + 反手】各跑一个生产轨迹任务（类型2 遮挡板）：
          加 1 个类型2 遮挡板（按焊缝几何放置，不依赖已有轨迹）→ 规划带障碍轨迹。
        类型2 障碍无需先规划无障碍轨迹即可放置，故不做无障碍预规划。
        单个 Scene 全程复用、循环内不 save/load；正/反手障碍分桶隔离；
        每缝【完成后】存 1 个 per-seam pkl（文件名含焊缝号，便于断点续跑跳过）。
    '''
    scene = _make_scene(args)
    out_root = getattr(args, "out_root", None)
    force = getattr(args, "force", False)

    for seam_id in range(len(scene.seams)):
        out_path = _out_path(scene, "type2", seam_id=seam_id, out_root=out_root)
        if not force and os.path.exists(out_path):
            print(f"[demo] seam {seam_id}：已存在 {out_path}，跳过")
            continue
        try:
            scene._set_cur_seam(seam_id)
            # 候选初始位姿：hand/障碍无关（放宽不避障），每缝算 1 次即可
            scene.plan_init_pose_fast(include_obstacles=False, verbose=False)
            for hand in ("forehand", "backhand"):
                scene.obstacles.get(seam_id, {}).pop(hand, None)   # 隔离本手别桶
                # 1) 加 1 个类型2 遮挡板，挂到本手别桶
                scene.add_obstacle_type2(hand=hand)
                # 2) 规划带障碍轨迹（只避开本手别桶障碍）
                if scene.compute_pose_and_plan_path(hand, max_stomp_try=1):
                    print(f"[demo] seam {seam_id} {hand}：带障碍轨迹成功")
                else:
                    print(f"[demo] seam {seam_id} {hand}：带障碍轨迹失败")
            scene.save(out_path)          # 仅在本缝正常处理完后存盘（异常则不存 → 续跑重试）
        except Exception as e:
            import traceback
            print(f"[demo] seam {seam_id} 异常跳过: {e}")
            traceback.print_exc()

    print(f"[demo] type2 全部焊缝完成 → {out_root or scene.cfg.output_root}")


def _out_path(scene, tag: str, seam_id: int = None, out_root: str = None) -> str:
    """结果落盘路径：<root>/<工件名>_<tag>[_seam<id>]_all.pkl（工件名区分不同工件）。

    out_root 给定则覆盖 cfg.output_root；seam_id 给定则文件名含焊缝号（per-seam pkl，
    里面仍是整场景快照——含其它已处理焊缝也无妨，命名区分即可断点续跑跳过本缝）。
    """
    if not scene.workpiece_obj:
        obj_stem = os.path.splitext(os.path.basename(scene.usd_path))[0]
    else:   
        obj_stem = os.path.splitext(os.path.basename(scene.workpiece_obj))[0]
    root = out_root or scene.cfg.output_root
    os.makedirs(root, exist_ok=True)
    if seam_id is not None:
        return os.path.join(root, f"{obj_stem}_{tag}_seam{seam_id}.pkl")
    return os.path.join(root, f"{obj_stem}_{tag}_all.pkl")


def obstacle_type1_demo_main(args):
    '''
        对【所有焊缝】、每缝【正手 + 反手】各跑一个生产轨迹任务：
          先规划一条不带障碍物的轨迹 → 在其扫掠空间随机加 1 个障碍物 type1 → 带障碍重新规划。
        单个 Scene 对象全程复用（curobo 句柄跨缝复用：h_truth 只 update_world、h_expl/cam 全程复用），
        循环内不 save/load。正/反手障碍按手别桶隔离，互不影响。每跑完 1 条焊缝存盘一次（覆盖）。
    '''
    '''
       DEBUG tools
           scene.seam_ids_by_length()
            scene.num_init_pose()
            from gt_gen.scene import Scene                                                                    
            scene = Scene.load('/media/a/新加卷/tempt/4/scene1.pkl')                                          
            Open3DSceneVisualizer(scene).show_scene_isaacsim(headless=args.headless, goal_arm_index=[0,1])  
            Open3DSceneVisualizer(scene).show_init_poses()                                                  
            Open3DSceneVisualizer(scene).show_seam(106)                                                     
            Open3DSceneVisualizer(scene).show_seam_all_isaacsim()                                         
            Open3DSceneVisualizer(scene).show_trajectory_isaacsim(traj_index=0, headless=args.headless)   
    '''
    scene = _make_scene(args)
    out_root = getattr(args, "out_root", None)
    force = getattr(args, "force", False)

    for seam_id in range(len(scene.seams)):
        out_path = _out_path(scene, "type1", seam_id=seam_id, out_root=out_root)
        if not force and os.path.exists(out_path):
            print(f"[demo] seam {seam_id}：已存在 {out_path}，跳过")
            continue
        try:
            scene._set_cur_seam(seam_id)
            scene.plan_init_pose_fast(verbose=True)     # 复用 _fastctx，产正/反手候选
            for hand in ("forehand", "backhand"):
                # 隔离：清掉本手别桶（防重跑残留；正反手本就分桶，互不干扰）
                scene.obstacles.get(seam_id, {}).pop(hand, None)

                # 1) 无障碍轨迹（此时本手别桶为空 → compute_pose_and_plan_path 不避障）
                if not scene.compute_pose_and_plan_path(hand, max_stomp_try=1):
                    print(f"[demo] seam {seam_id} {hand}：无障碍轨迹失败，跳过该手")
                    continue
                key1 = scene._trajectory_key()          # 步骤1 命中的 init pose，步骤3 须复用

                # 2) 基于刚成功的轨迹（当前 init pose key）随机加 1 个 type1 障碍
                placed = False
                for link_n in ("Link3", "Link4", "Link5", "xiaoyu_accessory_link"):
                    if scene.add_obstacle_type1(link=link_n, hand=None, index=None,
                                                entry_index=-1):
                        placed = True
                        break

                scene.trajectories.get(seam_id, {}).pop(key1, None)
                scene.trajectory_goal_poses.get(seam_id, {}).pop(key1, None)

                if not placed:
                    print(f"[demo] seam {seam_id} {hand}：4 个 link 均放不下障碍，跳过带障碍重规划")
                    # 无障碍轨迹仅用于放障碍，不能留作 GT
                    continue

                # 3) 带障碍重规划（只避开本手别桶的障碍；必须复用步骤1 的 init pose）
                if scene.compute_pose_and_plan_path(hand, max_stomp_try=1, init_pose_idx=key1[1]):
                    print(f"[demo] seam {seam_id} {hand}：带障碍轨迹成功")
                else:
                    print(f"[demo] seam {seam_id} {hand}：带障碍轨迹失败")
            scene.save(out_path)          # 仅在本缝正常处理完后存盘（异常则不存 → 续跑重试）
        except Exception as e:
            import traceback
            print(f"[demo] seam {seam_id} 异常跳过: {e}")
            traceback.print_exc()

    print(f"[demo] type1 全部焊缝完成 → {out_root or scene.cfg.output_root}")
    

def obstacle_type3_demo_main(args):
    '''
        Obersever Anything.
        forehand： 初始关节角下焊缝能被看到
        backhand： 初始关节角下焊缝完全看不到

        buffer_m: 0.001->0.1

        debug:
            Open3DSceneVisualizer(scene).show_observe_init_poses_debug(n=1, stride=1)
    '''
    scene = _make_oa_scene(args)
    out_root = getattr(args, "out_root", None)
    force = getattr(args, "force", False)

    for seam_id in range(len(scene.seams)):
        out_path = _out_path(scene, "type3", seam_id=seam_id, out_root=out_root)
        if not force and os.path.exists(out_path):
            print(f"[demo] seam {seam_id}：已存在 {out_path}，跳过")
            continue
        try:
            scene._set_cur_seam(seam_id)
            # 候选初始位姿：hand/障碍无关（放宽不避障），每缝算 1 次即可
            scene.plan_init_pose_fast(verbose=True, debug=False)
            # from gt_gen.scene_viz import Open3DSceneVisualizer
            # Open3DSceneVisualizer(scene).show_init_poses()
            # Open3DSceneVisualizer(scene).show_observe_init_poses_debug(3)
            for hand in ("forehand", "backhand"):
                if scene.compute_pose_and_plan_path(hand, extra_stomp_try=1, max_goal_pose=2, use_nm=False, pl_limit_deg=170, max_init_pose=20):
                    print(f"[demo] seam {seam_id} {hand}：轨迹成功")
                else:
                    print(f"[demo] seam {seam_id} {hand}：轨迹失败")
            scene.save(out_path)          # 仅在本缝正常处理完后存盘（异常则不存 → 续跑重试）
        except Exception as e:
            import traceback
            print(f"[demo] seam {seam_id} 异常跳过: {e}")
            traceback.print_exc()

    print(f"[demo] type2 全部焊缝完成 → {out_root or scene.cfg.output_root}")



def viz(args):
    from gt_gen.scene import Scene
    from gt_gen.scene_viz import Open3DSceneVisualizer


    scene = Scene.load(args.pkl)
    scene.summarize_trajectories()
    time.sleep(3)
    # Open3DSceneVisualizer(scene).show_trajectory_isaacsim(seam_id=0,hand="forehand")
    if args.ds:
        fps = 3
    else:
        fps = 30
    Open3DSceneVisualizer(scene).show_trajectory_isaacsim(seam_id=args.pkl_seamid,hand=args.pkl_hand,ds=args.ds,fps=fps)


def viz_ob(args):
    from gt_gen.observe_scene import ObserveAnythingScene
    from gt_gen.scene_viz import ObserveSceneVisualizer

    scene = ObserveAnythingScene.load(args.pkl)
    scene.summarize_trajectories()
    # ObserveSceneVisualizer(scene).show_scene_isaacsim()
    time.sleep(3)
    if args.ds:
        fps = 3
    else:
        fps = 30
    ObserveSceneVisualizer(scene).show_trajectory_isaacsim(seam_id=args.pkl_seamid,hand=args.pkl_hand,ds=args.ds,fps=fps)


def main():
    ap = argparse.ArgumentParser(description="Scene API demo：初始位姿求解 / 障碍物类型2 / 障碍物类型3 + isaacsim 可视化")
    ap.add_argument("--obj", default=DEFAULT_OBJ, help="工件 mesh（_part.obj / _watertight.obj）")
    ap.add_argument("--usd", help="场景usd")
    ap.add_argument("--weld-json", default=DEFAULT_WELD_JSON, help="焊缝 _weld_angle3.json")
    ap.add_argument("--pkl", help="保存轨迹的pkl文件")
    ap.add_argument("--pkl-seamid", type=int, help="保存轨迹的pkl文件")
    ap.add_argument("--pkl-hand", help="保存轨迹的pkl文件")
    ap.add_argument("--ds", action="store_true",
                    help="viz：回放【关键帧采样后】轨迹（scene.sampled_trajectories，需先跑 scripts/traj_downsample.py）")
    ap.add_argument("--task", default="viz", choices=["type1", "type2", "type3", "viz", "vizob"],
                    help="批处理任务：type1=障碍类型1 / type2=障碍类型2 / type3=ObserveAnything / viz=本地可视化调试（默认）")
    ap.add_argument("--out-root", default=None,
                    help="覆盖 cfg.output_root 的结果落盘根目录（per-seam pkl 存这里）")
    ap.add_argument("--force", action="store_true",
                    help="忽略已存在的 per-seam pkl，强制重跑该焊缝")
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

    # 默认 --task viz：保持本地可视化调试行为不变（勿与 curobo demo 同进程先后跑，见模块 docstring）。
    # 批处理由 shell（scripts/bash/v1/）传 --task type1/type2/type3 + --out-root 调用。
    if args.task == "type1":
        demo = obstacle_type1_demo_main  
    elif args.task == "type2":
        demo = obstacle_type2_demo_main
    elif args.task == "type3":
        demo = obstacle_type3_demo_main
    elif args.task == "viz":
        viz(args)
    elif args.task == "vizob":
        viz_ob(args)
    else:
        raise ValueError()
        
    if _PROFILE_MAIN:
        with _LineProfiler(__file__):       # PROFILEMAIN=1：逐行计时（含各行调用的耗时）
            demo(args)
    else:
        demo(args)
    return
    
    # --task viz（默认）：本地可视化调试入口
    # demo_obstacle_type2(args)
    # demo_obstacle_type3(args)
    # demo_init_pose(args)
    # demo_init_poses(args)   # 多候选初始位姿同屏铺网格（先另进程 plan_init_pose + save）
    # viz('/media/a/新加卷/tempt/5/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part_watertight_type1_all.pkl')

if __name__ == "__main__":
    main()
