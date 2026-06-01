# Phase 1 里程碑总览

每个里程碑独立可验证。**实现顺序按依赖链**；M1.1–M1.6 之间相对独立，可并行；M1.7 之后串行。

| ID | 模块 | 文档 | 实现文件 | 状态 | 估时 |
|----|------|------|---------|------|------|
| M1.0 | 环境配置 | — | `setup_env.sh` + `verify_env.py` | ✓ | 0.5 天 |
| M1.0.5 | cuRobo Hello World | — | `curobo_hello.py` | ✓ | 0.5 天 |
| M1.1 | RaycastSim 模拟深度图 | [M1.1_RaycastSim.md](M1.1_RaycastSim.md) | `depth_source.py` | ☐ | 2 天 |
| M1.2 | PyTorch 体素栅格 | [M1.2_VoxelMap.md](M1.2_VoxelMap.md) | `mapping.py` | ☐ | 2 天 |
| M1.3 | 合格观测检查 | [M1.3_ObservationCheck.md](M1.3_ObservationCheck.md) | `observation_check.py` | ☐ | 1 天 |
| M1.4 | cuRobo 封装 | [M1.4_CuRoboWrapper.md](M1.4_CuRoboWrapper.md) | `arm_kin.py` | ☐ | 3 天 |
| M1.5 | 全局图 + Dijkstra | [M1.5_Graph.md](M1.5_Graph.md) | `graph.py` | ☐ | 2 天 |
| M1.6 | 增益 + 目标偏置 | [M1.6_Gain.md](M1.6_Gain.md) | `gain.py` | ☐ | 1 天 |
| M1.7 | 单 cycle 探索 | [M1.7_SingleCycle.md](M1.7_SingleCycle.md) | `exploration.py` (核心) + `viz.py` | ✓ | 3 天 |
| M1.8 | 多 cycle + 卡住回退 | [M1.8_MultiCycle.md](M1.8_MultiCycle.md) | `exploration.py` (扩展) | ✓ | 2 天 |
| M1.9 | 切 Isaac Sim 真渲染 | [M1.9_IsaacSimDepth.md](M1.9_IsaacSimDepth.md) | `depth_source.py` (扩展) | ☐ | 2 天 |
| M1.10 | 端到端 + 视频 | [M1.10_EndToEnd.md](M1.10_EndToEnd.md) | `run_phase1.py` | ☐ | 1 天 |

**总计 21 天 ≈ 4 周**。

## 依赖关系图

```text
M1.0 ─┬─ M1.0.5 ──┐
      │           │
      ├─ M1.1 ────┼─┐
      │           │ │
      ├─ M1.2 ────┤ │  ← M1.2 需要 M1.1 (建图喂深度)
      │           │ │
      ├─ M1.3 ────┘ │  ← M1.3 需要 M1.2 (合格观测查体素)
      │             │
      ├─ M1.4 ──────┤  ← M1.4 需要 M1.0.5
      │             │
      ├─ M1.5 ──────┘  ← M1.5 独立
      │
      └─ M1.6 ←─────── M1.2 + M1.4

        M1.1+M1.2+M1.3+M1.4+M1.5+M1.6 → M1.7 → M1.8 → M1.9 → M1.10
```

## 验证方法

每个里程碑文档末尾都给出：

1. **自动测试命令**（pytest）
2. **期望输出**（具体行）
3. **视觉验证**（如适用，输出 PNG / Isaac Sim 截图）
4. **手动验收清单**

## 通用约定

### 坐标系

```text
world frame:    pkl 里的 robot_pose, piece_pose, seam_line 都在这个系
                seam_line 在 (24.025, 0.157, 11.432) 附近, 单位米

robot base:     由 robot_pose 决定 ((24.4, 0.5, 9.6) 附近)
                cuRobo 的 FK 输出在这个系

camera frame:   z 朝前, x 右, y 下 (OpenCV/Isaac Sim 标准)
```

### Pose 表示

```text
所有 SE(3) 用 4×4 齐次矩阵 (numpy float64 或 torch float32)
四元数用 [w, x, y, z] (跟 pkl 一致)
转换函数集中在 phase1/utils/transforms.py
```

### Tensor 设备

```text
所有大数据 (depth, voxel grid, ray bundle) 默认在 'cuda:0'
小数据 (pose, joint angles) 在 CPU 也可
单元测试都在 CPU 跑 (用 'cuda' if available)
```

### 样例数据

```text
工件 USD:       /media/a/新加卷/hanfeng/segment_sub_output/
                    BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/
                    BEAM_1aEEYa00Ed5Z4sE34qDJKu_part.usd
工件 OBJ:       同目录下 .obj 文件
焊缝 pkl:        同目录下 seam_0.pkl
机器人 USD:      /home/a/Datas/curobo/example_new_robot/urdf_12e/ur12e.usd
机器人 yml:      /home/a/Datas/curobo/example_new_robot/urdf_12e/ur12e.yml
```
