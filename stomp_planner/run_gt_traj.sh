#!/bin/bash

INPUT_ROOT="/kpfs_dataset_ssd/render/segment_output"
OUTPUT_ROOT="/kpfs_dataset/dataset/base_data/data_2/output"
DONE_FOLDER_ROOT="/kpfs_dataset/dataset/base_data/data_2/done"
LOGGER_FOLDER="/kpfs_dataset/dataset/simulation/gt/gt_overall/logger/traj_8_2_4"

NUM_GPUS=8
PER_GPU_JOBS=2
TRAJ_TIMEOUT_MIN=3

cd /kpfs_dataset/dataset/simulation/gt/gt_overall

/root/miniconda3/envs/mapanything/bin/python3.11 -u main_controller_sp_traj.py \
    --input_root       "$INPUT_ROOT" \
    --output_root      "$OUTPUT_ROOT" \
    --done_folder_root "$DONE_FOLDER_ROOT" \
    --logger_folder    "$LOGGER_FOLDER" \
    --num_gpus         "$NUM_GPUS" \
    --per_gpu_jobs     "$PER_GPU_JOBS" \
    --traj_timeout_min "$TRAJ_TIMEOUT_MIN"
