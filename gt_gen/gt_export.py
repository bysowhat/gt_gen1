"""Step 11: GT 导出（沿用现有 traj.pkl 格式，schema 落地时对齐）。

见 docs/gt-generation-curobo-implementation.md §7。
参考现有格式：gt.legacy_example（/home/a/Datas/curobo/example_new_robot/robot/traj.pkl）。
"""
from __future__ import annotations


def export_gt(trajectory, meta, out_path):
    """把关节轨迹 + 元信息（起点/目标/成功标志/时间戳）按现有格式序列化。"""
    raise NotImplementedError("Step 11")
