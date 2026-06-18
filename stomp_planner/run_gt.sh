#!/bin/bash

# INPUT_ROOT="/kpfs_dataset_ssd/render/segment_output"
# OUTPUT_ROOT="/kpfs_dataset/dataset/base_data/data_2/output"
# DONE_FOLDER_ROOT="/kpfs_dataset/dataset/base_data/data_2/completed"
# LOGGER_FOLDER="/kpfs_dataset/dataset/base_data/data_2/logs"

INPUT_ROOT="/kpfs_dataset/dataset/simulation/gt/test_data/input"
OUTPUT_ROOT="/kpfs_dataset/dataset/simulation/gt/test_data/output"
DONE_FOLDER_ROOT="/kpfs_dataset/dataset/simulation/gt/test_data/completed"
LOGGER_FOLDER="/kpfs_dataset/dataset/simulation/gt/gt_overall/logger/test_2_1"

NUM_GPUS=2
PER_GPU_JOBS=1

cd /kpfs_dataset/dataset/simulation/gt/gt_overall

/root/miniconda3/envs/mapanything/bin/python3.11 -u main_controller_sp.py \
    --input_root       "$INPUT_ROOT" \
    --output_root      "$OUTPUT_ROOT" \
    --done_folder_root "$DONE_FOLDER_ROOT" \
    --logger_folder    "$LOGGER_FOLDER" \
    --num_gpus         "$NUM_GPUS" \
    --per_gpu_jobs     "$PER_GPU_JOBS"
