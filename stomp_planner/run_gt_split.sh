#!/bin/bash

INPUT_ROOT="/kpfs_dataset_ssd/render/segment_output"
OUTPUT_ROOT="/kpfs_dataset/dataset/base_data/data_2/output"
DONE_FOLDER_ROOT="/kpfs_dataset/dataset/base_data/data_2/completed"
NUM_GPUS=8
PER_GPU_JOBS=2

cd /kpfs_dataset/dataset/simulation/gt/gt_overall

echo "=== 开始 Pose 阶段 ==="
/root/miniconda3/envs/mapanything/bin/python3.11 -u main_controller_sp_pose.py \
    --input_root      "$INPUT_ROOT" \
    --output_root     "$OUTPUT_ROOT" \
    --num_gpus        "$NUM_GPUS" \
    --per_gpu_jobs    "$PER_GPU_JOBS"

# echo "=== Pose 阶段完成，开始 Traj 阶段 ==="
# /root/miniconda3/envs/mapanything/bin/python3.11 -u main_controller_sp_traj.py \
#     --input_root      "$INPUT_ROOT" \
#     --output_root     "$OUTPUT_ROOT" \
#     --done_folder_root "$DONE_FOLDER_ROOT" \
#     --num_gpus        "$NUM_GPUS" \
#     --per_gpu_jobs    "$PER_GPU_JOBS"

echo "=== 全部完成 ==="
