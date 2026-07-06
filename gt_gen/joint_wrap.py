"""旋转关节目标角的 2π 分支归一化（避免规划器为到达等价姿态而绕整圈）。

问题：UR 腕关节限位很宽（约 ±2π），同一物理姿态在关节空间有相差 2π 的多支等价角。
若目标构型给了「远离起点」的那一支（如起点 J6=+π、目标 J6=−3.3656≈−π−2π），规划器会老老实实
把该关节转近一整圈，表现为焊枪绕自身轴自转 360°。而 −3.3656+2π=2.9176 与起点仅差 13°，是同一
末端位姿。

做法：对每个旋转关节，把 goal[j] 平移 2π·k 到「落在限位内、且离 start[j] 最近」的等价分支。
±2π 平移不改变旋转关节的物理位姿（只是换个圈数表示），故不影响任何碰撞/可见性/IK 等价性判定。
限位跨度 < 2π 的关节天然只有一支等价角 → 该关节自动无操作。
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def wrap_goal_near_start(start, goal, lower, upper, *, verbose=True, tag="wrap"):
    """把 goal 各旋转关节归一到「限位内、离 start 最近」的 ±2π 等价分支。

    start/goal : 关节角序列（长度=dof）
    lower/upper: 关节下/上限（长度=dof）
    verbose    : 发生平移时打印一行 `[tag] Jk: g -> g' (省 xxx°)`
    返回：归一后的 goal（python list, dof）。都不在限位内时保留原值（绝不越界）。
    """
    start = np.asarray(start, float)
    goal = np.asarray(goal, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    out = goal.copy()
    for j in range(len(goal)):
        best = float(goal[j])
        best_d = abs(float(goal[j]) - float(start[j]))
        k0 = int(round((float(start[j]) - float(goal[j])) / TWO_PI))
        for kk in (k0 - 1, k0, k0 + 1):
            cand = float(goal[j]) + kk * TWO_PI
            if lower[j] - 1e-9 <= cand <= upper[j] + 1e-9 and abs(cand - float(start[j])) < best_d:
                best, best_d = cand, abs(cand - float(start[j]))
        if verbose and abs(best - float(goal[j])) > 1e-9:
            saved = abs(float(goal[j]) - float(start[j])) - abs(best - float(start[j]))
            print(f"[{tag}] J{j + 1}: {float(goal[j]):.4f} -> {best:.4f} "
                  f"(省 {np.degrees(saved):.0f}°)")
        out[j] = best
    return out.tolist()
