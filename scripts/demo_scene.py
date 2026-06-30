"""demo：用 Scene API 求候选初始位姿并(可选)open3d 可视化。

薄壳示例——展示 gt_gen.scene.Scene + gt_gen.scene_viz.Open3DSceneVisualizer 的用法，
等价于 scripts/plan_init_pose.py --solve --welds <seam-id> [--viz]，但走新 API：

    scene = Scene(cfg, workpiece_obj, weld_json, seam_id)
    scene.plan_init_pose()                          # 求候选初始位姿（需 GPU）
    Open3DSceneVisualizer(scene).show_init_poses()  # 逐个可视化（需显示器，按 C 翻页）

运行（求解，无窗口）：
    conda run -n env_isaaclab --no-capture-output python -u scripts/demo_scene_init_pose.py
带可视化（需显示器）：加 --viz
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_OBJ = ("/media/a/新加卷/hanfeng/segment/A3Changfang/"
               "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part_watertight.obj")
DEFAULT_WELD_JSON = ("/media/a/新加卷/hanfeng/segment/A3Changfang/"
                     "BEAM_1aEEYa00Ed5Z4sE34qDJKu_weld_angle3.json")


def main():
    ap = argparse.ArgumentParser(description="Scene API demo：候选初始位姿求解 + open3d 可视化")
    ap.add_argument("--obj", default=DEFAULT_OBJ, help="工件 mesh（_part.obj / _watertight.obj）")
    ap.add_argument("--weld-json", default=DEFAULT_WELD_JSON, help="焊缝 _weld_angle3.json")
    ap.add_argument("--seam-id", type=int, default=0, help="用 weld_json 中的第几条焊缝")
    args = ap.parse_args()

    from gt_gen.scene import Scene
    from gt_gen.scene_viz import Open3DSceneVisualizer

    scene = Scene(cfg="configs/default.yaml",
                  workpiece_obj=args.obj,
                  weld_json=args.weld_json,
                  seam_id=args.seam_id)

    scene.plan_init_pose()

    if not scene.init_pose_candidates:
        print("[demo] 求解失败：无候选初始位姿")
        return
    n_fore = sum(1 for c in scene.init_pose_candidates if c.hand == "forehand")
    n_back = sum(1 for c in scene.init_pose_candidates if c.hand == "backhand")
    print(f"[demo] 候选初始位姿 {len(scene.init_pose_candidates)} 个（正手 {n_fore} / 反手 {n_back}）")

    Open3DSceneVisualizer(scene).show_init_poses()


if __name__ == "__main__":
    main()
