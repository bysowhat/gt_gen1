#!/bin/bash
# 障碍类型3 批处理入口：
# ./scripts/bash/v1/run_obstacle_type3.sh 0 /kpfs_dataset_ssd/dataset/render_baiyu/obstacle_type3/usd/warehouse.usdz /kpfs_dataset_ssd/dataset/render_baiyu/obstacle_type3/jsons/warehouse.json /kpfs_dataset_ssd/dataset/render_baiyu/obstacle_type3/traj/warehouse

# 接收脚本输入参数
CUDA_DEVICES=$1
USD_FILE=$2
JSON_FILE=$3
OUT_ROOT=$4

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES /workspace/isaaclab/_isaac_sim/python.sh scripts/find_surface_welds.py \
    --usd "$USD_FILE" \
    --out "$JSON_FILE" \
    --min 0.15 \
    --max 1.2 \
    --angle-min 30 \
    --angle-max 150 \
    --similar-len-tol 0.05 \
    --similar-dir-tol 0.05 \
    --similar-dist-tol 1.0

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES /workspace/isaaclab/_isaac_sim/python.sh scripts/demo_scene.py \
    --task type3 \
    --usd "$USD_FILE" \
    --weld-json "$JSON_FILE" \
    --out-root  "$OUT_ROOT"
