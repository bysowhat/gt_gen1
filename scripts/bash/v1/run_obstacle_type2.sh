#!/bin/bash
# 障碍类型2 批处理入口：多 GPU 并行计算 filenames.txt 全部工件【所有焊缝】的 type2 轨迹。
# 所有参数在本文件显式设定，传给公共核心 _run_obstacle.sh 执行。
# 各参数含义见 _run_obstacle.sh 顶部注释。改参数只改本文件即可。
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==== 参数（可被命令行同名环境变量覆盖，否则用这里的值）====
export TASK=type2                                                          # 任务类型
export OUT_ROOT="${OUT_ROOT:-/kpfs_dataset_ssd/dataset/render_baiyu/obstacle_type2}"  # 结果落盘根目录
export FILELIST="${FILELIST:-/kpfs_dataset_ssd/dataset/render_baiyu/filenames.txt}"   # 工件清单（每行 obj<TAB>weld_json）
export PY="${PY:-/workspace/isaaclab/_isaac_sim/python.sh}"                 # python 解释器
export NUM_GPUS="${NUM_GPUS:-}"                                            # GPU 张数（留空=nvidia-smi 自动探测；2→8 无需改）
export TIMEOUT="${TIMEOUT:-7200}"                                         # 单工件超时秒（2 小时；0=不限）
export START="${START:-0}"                                                # 从第 START 个工件起
export LIMIT="${LIMIT:-0}"                                                # 最多处理 LIMIT 个（0=不限）
export FORCE="${FORCE:-0}"                                                # 1=忽略已存在结果/标记，强制重跑

exec "$HERE/_run_obstacle.sh"
