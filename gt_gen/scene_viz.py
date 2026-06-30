"""SceneVisualizer —— 用 open3d / isaacsim 可视化 Scene 的各种信息（API 重组）。

设计：基类 SceneVisualizer 持有一个 Scene；两个后端子类：
  · Open3DSceneVisualizer  —— 用 open3d 在本机开窗可视化（需显示器，env_isaaclab 装了 open3d）。
  · IsaacSimSceneVisualizer —— 用 isaacsim/isaaclab 可视化（后续补充）。

与 Scene 一样，只做 **API 方向的结构包装**：本期 open3d 的「逐个可视化候选初始位姿」直接复用
scripts/plan_init_pose.py 的 show_lookup_solutions（整臂碰撞球 + init_free 盒 + 工件网格 + 焊缝线
+ standoff 落点），不改其渲染逻辑。

本期已实现：
  · Open3DSceneVisualizer.show_init_poses()  —— 逐个看 Scene 的候选初始位姿（按 C 切下一个）

后续补充（占位）：3D 世界 / 机械臂 / 焊缝 / 已观测区域 等的 open3d & isaacsim 可视化。
"""
from __future__ import annotations

from gt_gen.scene import Scene, _load_plan_init_pose


class SceneVisualizer:
    """Scene 可视化基类：持有 Scene，具体渲染由后端子类实现。"""

    def __init__(self, scene: Scene):
        if not isinstance(scene, Scene):
            raise TypeError(f"SceneVisualizer 需要 Scene 实例，收到 {type(scene)}")
        self.scene = scene


class Open3DSceneVisualizer(SceneVisualizer):
    """open3d 后端可视化（本机开窗）。"""

    def show_init_poses(self, stride: int = 1):
        """逐个可视化该 Scene 焊缝的【候选初始位姿】（工件相对机械臂的摆放）。

        前提：先调用 Scene.plan_init_pose() 求出候选。可视化复用 scripts/plan_init_pose.py 的
        _show_kejian2_results（正手→反手依次开窗：按 (R,t) 摆放的工件网格 + 焊缝/中点 + standoff 落点
        + 蓝色 bisector 正反手判据轴）；同一窗口按 **C 键**切到下一个候选。

        stride：每隔几个候选抽 1 个看（默认 1=逐个看全部）。
        """
        scene = self.scene
        if not scene.init_pose_candidates:
            raise RuntimeError(
                "无候选初始位姿可视化：请先调用 Scene.plan_init_pose()（且求解成功）")
        pim = _load_plan_init_pose()
        res = {"forehand": [c.raw for c in scene.init_pose_candidates if c.hand == "forehand"],
               "backhand": [c.raw for c in scene.init_pose_candidates if c.hand == "backhand"]}
        pim._show_kejian2_results(scene.cfg, scene.workpiece_obj, scene.seam, res, stride=stride)


class IsaacSimSceneVisualizer(SceneVisualizer):
    """isaacsim/isaaclab 后端可视化（后续补充）。"""

    def show_init_poses(self):
        raise NotImplementedError("IsaacSimSceneVisualizer.show_init_poses 待补充")
