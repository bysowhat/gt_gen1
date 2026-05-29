# Phase 1 — Exploration

焊缝在线主动巡检的探索阶段实现。详细设计见上一级目录的 [焊缝在线主动巡检.md](../焊缝在线主动巡检.md)。

## 环境

```text
conda env: env_isaaclab
python:    /home/a/miniforge3/envs/env_isaaclab/bin/python
GPU:       NVIDIA GPU + CUDA 12.x (推荐 12.4+)
基础包:     torch 2.7+cu128, isaacsim
```

## 安装

按以下顺序执行（**首次安装大约 10-15 分钟**，cuRobo 和 nvblox 都要编译 CUDA kernel）：

```bash
cd /home/a/Projects/Github/gt_gen_hanfeng/phase1
bash setup_env.sh
```

完成后跑环境验证：

```bash
/home/a/miniforge3/envs/env_isaaclab/bin/python verify_env.py
```

预期输出：所有项 ✓。如果某项 ✗，按提示信息处理。

## 文件结构（逐步填充）

| 文件 | 里程碑 | 状态 |
|------|--------|------|
| `setup_env.sh` | M1.0 | ☑ |
| `verify_env.py` | M1.0 | ☑ |
| `curobo_hello.py` | M1.0.5 | ☑ |
| `depth_source.py` | M1.1 | ☑ |
| `mapping.py` | M1.2 | ☑ |
| `observation_check.py` | M1.3 | ☐ |
| `arm_kin.py` | M1.4 | ☐ |
| `graph.py` | M1.5 | ☐ |
| `gain.py` | M1.6 | ☐ |
| `exploration.py` | M1.7 | ☐ |
| `viz.py` | M1.7+ | ☐ |
| `config.py` | M1.7 | ☐ |
| `run_phase1.py` | M1.10 | ☐ |
