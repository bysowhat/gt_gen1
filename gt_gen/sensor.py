"""Step 3: 传感器模拟（对真值 obj mesh 做 raycast）。

见 docs/gt-generation-curobo-implementation.md §4.1、privileged-nbv.md §4.5 ②/⑥。
相机内参/外参从 sensor.camera_usd（左目）读取；深度上限 sensor.max_depth_m。
"""
from __future__ import annotations


def load_camera_model(config):
    """从 camera_usd 读取左目相机内参、FOV、手眼外参（相机相对 ee_link）。"""
    raise NotImplementedError("Step 3")


def camera_pose_from_config(handle, cfg, camera_model):
    """由关节构型 + 手眼外参，求相机在世界系的位姿。"""
    raise NotImplementedError("Step 3")


def raycast_observe(camera_pose, camera_model, truth_scene, max_depth):
    """从相机位姿向真值场景投射光线。

    返回：(free_voxels, occ_voxels) —— 视线穿过的体素 / 命中点体素。
    """
    raise NotImplementedError("Step 3")
