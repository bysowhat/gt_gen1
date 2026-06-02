"""Step 0 验证：环境 + 机器人配置 + cuRobo MotionGen 初始化/warmup。

运行：  conda run -n env_isaaclab python scripts/verify_step0.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import torch

    from gt_gen import compat  # 修 warp 1.13 的 wp.torch 命名空间（须在 import curobo 前）

    import curobo

    print("== 环境 ==")
    print("warp.torch shim:", "已生效" if compat.applied else "未生效")
    print("curobo:", getattr(curobo, "__version__", "?"))
    print("torch :", torch.__version__, "| cuda:", torch.cuda.is_available())

    from gt_gen.config import load_config

    cfg = load_config()
    print("\n== 决策/配置 ==")
    print("约束范围   :", cfg.constraint_scope)
    print("机器人cfg  :", cfg.robot_cfg_path)
    print("base/ee    :", cfg.base_link, "->", cfg.ee_link)
    print("关节       :", cfg.joint_names)
    print("retract    :", [round(x, 4) for x in cfg.retract_config])
    print("碰撞link数 :", len(cfg.collision_link_names))
    print("体素分辨率 :", cfg.voxel_size_m, "m | ROI外扩:", cfg.roi_expand_m, "m")
    print("深度上限   :", cfg.max_depth_m, "m")

    print("\n== cuRobo MotionGen 初始化（空世界）+ warmup ==")
    from curobo.types.base import TensorDeviceType
    from curobo.geom.types import WorldConfig, Cuboid
    from curobo.util_file import load_yaml
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    tensor_args = TensorDeviceType()
    robot_cfg = load_yaml(cfg.robot_cfg_path)["robot_cfg"]
    # 占位障碍（远处小立方体）：仅为通过 warmup 的碰撞检查；真实运行用 voxel 世界
    world = WorldConfig(
        cuboid=[Cuboid(name="dummy", pose=[5.0, 5.0, 5.0, 1, 0, 0, 0], dims=[0.05, 0.05, 0.05])]
    )

    mg_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world,
        tensor_args,
        interpolation_dt=0.02,
    )
    mg = MotionGen(mg_cfg)
    mg.warmup(warmup_js_trajopt=False)
    print("MotionGen warmup 成功 ✓")

    print("\nSTEP0_OK")


if __name__ == "__main__":
    main()
