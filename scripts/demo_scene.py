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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_OBJ = ("/media/a/新加卷/hanfeng/segment/A3Changfang/"
               "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part_watertight.obj")
DEFAULT_WELD_JSON = ("/media/a/新加卷/hanfeng/segment/A3Changfang/"
                     "BEAM_1aEEYa00Ed5Z4sE34qDJKu_weld_angle3.json")


def _make_scene(args):
    from gt_gen.scene import Scene
    scene = Scene(cfg="configs/default.yaml",
                 workpiece_obj=args.obj,
                 weld_json=args.weld_json,
                 seam_id=args.seam_id)
    return scene

def demo_init_pose(args):
    """原始示例：求候选初始位姿（需 GPU/curobo），可选 open3d 逐个可视化。"""
    from gt_gen.scene_viz import Open3DSceneVisualizer

    scene = _make_scene(args)
    scene._set_cur_seam(9)

    scene.plan_init_pose()

    if not scene.init_pose_candidates:
        print("[demo] 求解失败：无候选初始位姿")
        return
    n_fore = sum(1 for c in scene.init_pose_candidates if c.hand == "forehand")
    n_back = sum(1 for c in scene.init_pose_candidates if c.hand == "backhand")
    print(f"[demo] 候选初始位姿 {len(scene.init_pose_candidates)} 个（正手 {n_fore} / 反手 {n_back}）")

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

def demo_main(args):
    from gt_gen.scene_viz import Open3DSceneVisualizer

    scene = _make_scene(args)
    scene._set_cur_seam(9)
    scene.add_obstacle_type2()
    scene.plan_init_pose()


    if not scene.init_pose_candidates:
        print("[demo] 求解失败：无候选初始位姿")
        return
    n_fore = sum(1 for c in scene.init_pose_candidates if c.hand == "forehand")
    n_back = sum(1 for c in scene.init_pose_candidates if c.hand == "backhand")
    print(f"[demo] 候选初始位姿 {len(scene.init_pose_candidates)} 个（正手 {n_fore} / 反手 {n_back}）")
    
    Open3DSceneVisualizer(scene).show_init_poses()


def main():
    ap = argparse.ArgumentParser(description="Scene API demo：初始位姿求解 / 障碍物类型2 / 障碍物类型3 + isaacsim 可视化")
    ap.add_argument("--obj", default=DEFAULT_OBJ, help="工件 mesh（_part.obj / _watertight.obj）")
    ap.add_argument("--weld-json", default=DEFAULT_WELD_JSON, help="焊缝 _weld_angle3.json")
    ap.add_argument("--seam-id", type=int, default=1, help="用 weld_json 中的第几条焊缝")
    ap.add_argument("--headless", action="store_true", help="无显示器自检：spawn 后跑几帧即退")
    args = ap.parse_args()

    # 默认跑障碍物类型2；想看别的换成下面对应调用（勿与本调用同进程先后跑，见模块 docstring）
    # demo_obstacle_type2(args)
    # demo_obstacle_type3(args)
    # demo_init_pose(args)

    demo_main(args)

if __name__ == "__main__":
    main()
