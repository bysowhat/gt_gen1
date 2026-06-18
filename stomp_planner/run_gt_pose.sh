#!/bin/bash

# INPUT_ROOT="/kpfs_dataset_ssd/render/segment_output"
# OUTPUT_ROOT="/kpfs_dataset/dataset/base_data/data_2/output"
# LOGGER_FOLDER="/kpfs_dataset/dataset/base_data/data_2/logs_pose"

INPUT_ROOT="/kpfs_dataset_ssd/render/segment_output"
OUTPUT_ROOT="/kpfs_dataset/dataset/base_data/data_2/output_25"
LOGGER_FOLDER="/kpfs_dataset/dataset/simulation/gt/gt_overall/logger/pose_8_2_m25"
REFERENCE_ROOT="/kpfs_dataset/dataset/base_data/data_2/output"

NUM_GPUS=8
PER_GPU_JOBS=2
POSE_TIMEOUT_MIN=3

cd /kpfs_dataset/dataset/simulation/gt/gt_overall

/root/miniconda3/envs/mapanything/bin/python3.11 -u main_controller_sp_pose.py \
    --input_root       "$INPUT_ROOT" \
    --output_root      "$OUTPUT_ROOT" \
    --logger_folder    "$LOGGER_FOLDER" \
    --reference_root   "$REFERENCE_ROOT" \
    --num_gpus         "$NUM_GPUS" \
    --per_gpu_jobs     "$PER_GPU_JOBS" \
    --pose_timeout_min "$POSE_TIMEOUT_MIN"
