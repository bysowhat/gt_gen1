#!/usr/bin/env bash
#
# 多卡调度器：将 initpose_full/data 下所有 (npy, obj) 任务分配到 8 张 GPU 上执行。
#
# 原理：
#   每个任务启动一个 render_seam.py 子进程，设置环境变量 RENDER_DONE_FILE 指向
#   唯一的哨兵文件（/tmp/render_seam_done/<stem>_<seam_idx>.done）。
#   render_seam.py 完成全部 pose 渲染后会写入该哨兵文件。
#   调度器每 5 秒轮询一次哨兵文件，一旦出现就 kill 对应进程，把该卡标记为空闲，
#   然后从队列取下一个任务分配给该卡。
#
# 用法：
#   bash scripts/bash/kejian/render_seam_scheduler.sh [--dry-run] [--max-envs N]

set -euo pipefail

# ===== 可调参数 =====
NPY_DIR="/kpfs_dataset_ssd/dataset/render_kejian/initpose_full/data"
OBJ_DIR="/kpfs_dataset_ssd/dataset/render_kejian/segment_output_sub"
OUT_DIR="/kpfs_dataset_ssd/dataset/render_kejian/render_outs"
DONE_TMPDIR="/tmp/render_seam_done"
NUM_GPUS=8
POLL_INTERVAL=5          # 轮询间隔（秒）
MAX_ENVS=3               # --max-envs 默认值
PYTHON_BIN="/workspace/isaaclab/_isaac_sim/python.sh"
RENDER_SCRIPT="render/render_seam.py"
DRY_RUN=false

# ===== 解析参数 =====
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --max-envs) MAX_ENVS="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ===== 确保临时目录存在 =====
mkdir -p "$DONE_TMPDIR"

# ===== 收集全部任务 =====
echo "===== 收集任务列表 ====="
tasks=()          # 每个元素: "stem|npy_path|obj_path|seam_idx"
task_count=0

for stem_dir in "$NPY_DIR"/*/; do
    stem=$(basename "$stem_dir")
    obj_path="$OBJ_DIR/${stem}_part/${stem}_part_watertight.obj"

    if [[ ! -f "$obj_path" ]]; then
        echo "  [跳过] obj 不存在: $obj_path"
        continue
    fi

    for npy in "$stem_dir"/seam_*.npy; do
        [[ -f "$npy" ]] || continue
        npy_name=$(basename "$npy" .npy)   # 如 seam_47
        tasks+=("${stem}|${npy}|${obj_path}|${npy_name}")
        task_count=$((task_count + 1))
    done
done

echo "共收集到 $task_count 个任务"
echo ""

if [[ $task_count -eq 0 ]]; then
    echo "没有待执行的任务，退出。"
    exit 0
fi

if $DRY_RUN; then
    echo "===== [DRY RUN] 打印任务列表（前 20 个）====="
    for ((i=0; i<task_count && i<20; i++)); do
        IFS='|' read -r stem npy_path obj_path seam_idx <<< "${tasks[$i]}"
        echo "  [$i] stem=$stem  seam=$seam_idx  npy=$npy_path  obj=$obj_path"
    done
    if [[ $task_count -gt 20 ]]; then
        echo "  ... 还有 $((task_count - 20)) 个任务未显示"
    fi
    echo ""
    echo "===== 命令预览（前 3 个）====="
    for ((i=0; i<task_count && i<3; i++)); do
        IFS='|' read -r stem npy_path obj_path seam_idx <<< "${tasks[$i]}"
        done_file="$DONE_TMPDIR/${stem}__${seam_idx}.done"
        echo "RENDER_DONE_FILE=$done_file \\"
        echo "  $PYTHON_BIN $RENDER_SCRIPT --obj '$obj_path' --seam-npy '$npy_path' --out '$OUT_DIR' --max-envs=$MAX_ENVS --headless"
        echo ""
    done
    exit 0
fi

# ===== 全局状态 =====
declare -A gpu_pid       # gpu_id -> 子进程 PID
declare -A gpu_done_file # gpu_id -> 对应哨兵文件路径
declare -A gpu_task_idx  # gpu_id -> 任务在 tasks 数组中的索引
next_task=0               # 下一个待分配任务的索引
finished=0                # 已完成任务数

# ===== 清理函数 =====
cleanup() {
    echo ""
    echo "===== 收到中断信号，清理子进程 ====="
    for gpu_id in "${!gpu_pid[@]}"; do
        pid="${gpu_pid[$gpu_id]}"
        if kill -0 "$pid" 2>/dev/null; then
            echo "  杀死 GPU $gpu_id 上的进程 PID=$pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    # 清理哨兵文件
    rm -f "$DONE_TMPDIR"/*.done
    echo "已清理。"
    exit 1
}
trap cleanup SIGINT SIGTERM

# ===== 启动一个任务到指定 GPU =====
launch_task() {
    local gpu_id="$1"
    local task_str="$2"
    local task_idx="$3"

    IFS='|' read -r stem npy_path obj_path seam_idx <<< "$task_str"

    local done_file="$DONE_TMPDIR/${stem}__${seam_idx}.done"
    # 先删掉可能残留的旧哨兵
    rm -f "$done_file"

    local log_file="$OUT_DIR/logs/${stem}__${seam_idx}.log"
    mkdir -p "$(dirname "$log_file")" "$OUT_DIR"

    echo "  [GPU $gpu_id] 启动: $stem / $seam_idx"
    echo "    npy: $npy_path"
    echo "    obj: $obj_path"
    echo "    log: $log_file"

    # 在后台启动 render_seam.py，通过 CUDA_VISIBLE_DEVICES 绑定 GPU
    # RENDER_DONE_FILE 是哨兵文件，render_seam.py 完成后会写入
    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        export RENDER_DONE_FILE="$done_file"
        exec "$PYTHON_BIN" "$RENDER_SCRIPT" \
            --obj "$obj_path" \
            --seam-npy "$npy_path" \
            --out "$OUT_DIR" \
            --max-envs="$MAX_ENVS" \
            --headless \
            > "$log_file" 2>&1
    ) &
    local pid=$!

    gpu_pid[$gpu_id]=$pid
    gpu_done_file[$gpu_id]="$done_file"
    gpu_task_idx[$gpu_id]="$task_idx"

    echo "    PID=$pid  (GPU $gpu_id)"
}

# ===== 标记某 GPU 完成当前任务 =====
finish_gpu_task() {
    local gpu_id="$1"
    local pid="${gpu_pid[$gpu_id]}"
    local done_file="${gpu_done_file[$gpu_id]}"
    local task_str="${tasks[${gpu_task_idx[$gpu_id]}]}"

    IFS='|' read -r stem npy_path obj_path seam_idx <<< "$task_str"

    # 检查哨兵文件内容
    if [[ -f "$done_file" ]]; then
        echo "  [GPU $gpu_id] 哨兵文件已写入，任务完成: $stem / $seam_idx"
        cat "$done_file" | while read line; do echo "    $line"; done
    fi

    # 强杀进程（simulation_app.close() 会卡住）
    if kill -0 "$pid" 2>/dev/null; then
        echo "  [GPU $gpu_id] 强杀挂起进程 PID=$pid"
        kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi

    # 清理哨兵
    rm -f "$done_file"

    # 重置该卡状态
    unset gpu_pid[$gpu_id]
    unset gpu_done_file[$gpu_id]
    unset gpu_task_idx[$gpu_id]

    finished=$((finished + 1))
    echo "  [GPU $gpu_id] 已释放（进度: $finished / $task_count）"
}

# ===== 主循环 =====
echo "===== 开始调度 ($(date)) ====="
echo "GPU 数量: $NUM_GPUS"
echo "总任务数: $task_count"
echo "输出目录: $OUT_DIR"
echo "哨兵目录: $DONE_TMPDIR"
echo "轮询间隔: ${POLL_INTERVAL}s"
echo ""

while [[ $finished -lt $task_count ]]; do
    # 1. 检查哪些 GPU 空闲，分配新任务
    for ((gpu_id=0; gpu_id<NUM_GPUS; gpu_id++)); do
        if [[ -z "${gpu_pid[$gpu_id]+x}" ]] && [[ $next_task -lt $task_count ]]; then
            # 该卡空闲且有剩余任务
            launch_task "$gpu_id" "${tasks[$next_task]}" "$next_task"
            next_task=$((next_task + 1))
        fi
    done

    # 2. 检查已运行的 GPU 是否完成（哨兵文件出现）
    for gpu_id in "${!gpu_pid[@]}"; do
        done_file="${gpu_done_file[$gpu_id]}"
        if [[ -f "$done_file" ]]; then
            finish_gpu_task "$gpu_id"
        fi
    done

    # 3. 如果全部完成了，退出循环
    if [[ $finished -ge $task_count ]]; then
        break
    fi

    # 4. 如果所有卡都忙，等待后重试
    if [[ ${#gpu_pid[@]} -ge $NUM_GPUS ]] || [[ $next_task -ge $task_count ]]; then
        sleep "$POLL_INTERVAL"
    fi
done

# ===== 收尾 =====
# 清理可能残留的哨兵文件
rm -f "$DONE_TMPDIR"/*.done

echo ""
echo "===== 全部完成！($(date)) ====="
echo "总任务数: $task_count，全部完成"
echo "输出目录: $OUT_DIR"
