#!/usr/bin/env bash
#
# 多卡调度器：将 obstacle_type2.ds 目录下每个 pkl 分配到多张 GPU 上执行。
#
# 原理：
#   任务单元 = 一个 pkl 文件。每个任务启动一个 render_trajectory.py 子进程，传 --pkl 指向
#   该 pkl，--all 一次性渲染其内全部有采样结果的成功轨迹（与 render_seam 的 --max-seams-per-proc
#   分批不同：这里无 cap、无重排，一个 pkl 一次跑到底）。
#
#   render_trajectory.py 渲完整个 pkl 后会在标准输出末尾打印 "RENDER_TRAJECTORY_DONE"，之后
#   调用 simulation_app.close()——该调用常卡住不返回、进程迟迟不退且占着显存。故不能靠"进程
#   退出"判完成，改为：把子进程输出重定向到日志文件，调度器每 POLL_INTERVAL 秒 grep 该日志，
#   一旦出现 "RENDER_TRAJECTORY_DONE" 即认定该 pkl 渲完，主动按进程组强杀释放显存。
#   （等价于 render_seam 靠哨兵文件强杀，只是这里用日志完成标记代替哨兵，好处是不用改 Python。）
#
#   崩溃/超时(无完成标记)判失败；重跑本脚本时靠 render_trajectory.py 的 pkl 内 per-traj _DONE_
#   文件跳过已渲完的轨迹，增量补回。
#
# 用法：
# cd /kpfs_dataset_ssd/dataset/baiyu/code/gt_gen_hanfeng
# bash scripts/bash/v1/render_trajectory_scheduler.sh [--task-timeout N] [--dry-run]

set -euo pipefail

# ===== 可调参数 =====
DS_DIR="/kpfs_dataset_ssd/dataset/render_baiyu/obstacle_type1.ds"
OUT_DIR="/kpfs_dataset_ssd/dataset/render_baiyu/obstacle_type1_render"
POLL_INTERVAL=5          # 轮询间隔（秒）
TASK_TIMEOUT=8000        # 单个任务(一个 pkl，--all 渲其全部轨迹)超时（秒）
MAX_ENVS=2               # --max-envs 固定值
PYTHON_BIN="/workspace/isaaclab/_isaac_sim/python.sh"
RENDER_SCRIPT="render/render_trajectory.py"
DONE_MARKER="RENDER_TRAJECTORY_DONE"   # 完成标记（render_trajectory.py 末尾打印）
DRY_RUN=false

# ===== 解析参数 =====
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --task-timeout) TASK_TIMEOUT="$2"; shift 2 ;;
        --max-envs) MAX_ENVS="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ===== 自动检测 GPU 数量（取不到则报错停止）=====
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "错误：找不到 nvidia-smi，无法检测 GPU 数量，退出。" >&2
    exit 1
fi
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
if [[ -z "$NUM_GPUS" || "$NUM_GPUS" -lt 1 ]]; then
    echo "错误：未检测到可用 GPU（nvidia-smi -L 返回 0），退出。" >&2
    exit 1
fi
echo "检测到 GPU 数量: $NUM_GPUS"

# ===== 收集全部任务（平铺遍历 DS_DIR 下的 *.pkl）=====
echo "===== 收集任务列表 ====="
if [[ ! -d "$DS_DIR" ]]; then
    echo "错误：pkl 目录不存在: $DS_DIR" >&2
    exit 1
fi

tasks=()          # 每个元素: pkl 绝对路径（一个 pkl 一个任务，进程内 --all 渲其全部轨迹）
task_count=0

shopt -s nullglob
pkl_list=("$DS_DIR"/*.pkl)
shopt -u nullglob
for pkl_path in "${pkl_list[@]}"; do
    [[ -f "$pkl_path" ]] || continue
    tasks+=("$pkl_path")
    task_count=$((task_count + 1))
done

echo "共收集到 $task_count 个 pkl 任务"
echo ""

if [[ $task_count -eq 0 ]]; then
    echo "没有待执行的任务，退出。"
    exit 0
fi

if $DRY_RUN; then
    echo "===== [DRY RUN] 打印任务列表（前 20 个）====="
    for ((i=0; i<task_count && i<20; i++)); do
        echo "  [$i] $(basename "${tasks[$i]}")"
    done
    if [[ $task_count -gt 20 ]]; then
        echo "  ... 还有 $((task_count - 20)) 个任务未显示"
    fi
    echo ""
    echo "===== 命令预览（前 3 个）====="
    for ((i=0; i<task_count && i<3; i++)); do
        echo "  $PYTHON_BIN $RENDER_SCRIPT --pkl '${tasks[$i]}' --out '$OUT_DIR' --max-envs $MAX_ENVS --all --headless"
        echo ""
    done
    exit 0
fi

# ===== 全局状态 =====
declare -A gpu_pid        # gpu_id -> 子进程 PID
declare -A gpu_log_file   # gpu_id -> 对应日志文件路径
declare -A gpu_task_idx   # gpu_id -> 任务在 tasks 数组中的索引
declare -A gpu_start_time # gpu_id -> 任务启动时间戳 (epoch seconds)
next_task=0               # 下一个待分配任务的索引
finished=0                # 已完成 pkl 数
failed=0                  # 失败/超时任务数
total_tasks=$task_count   # pkl 总数

# ===== 清理函数 =====
cleanup() {
    echo ""
    echo "===== 收到中断信号，清理子进程 ====="
    for gpu_id in "${!gpu_pid[@]}"; do
        pid="${gpu_pid[$gpu_id]}"
        if kill -0 "$pid" 2>/dev/null; then
            echo "  杀死 GPU $gpu_id 上的进程组 PID=$pid"
            kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    echo "已清理。"
    exit 1
}
trap cleanup SIGINT SIGTERM

# ===== 启动一个任务到指定 GPU =====
launch_task() {
    local gpu_id="$1"
    local pkl_path="$2"
    local task_idx="$3"

    local stem; stem="$(basename "$pkl_path" .pkl)"
    local log_file="$OUT_DIR/logs/${stem}.log"
    mkdir -p "$(dirname "$log_file")" "$OUT_DIR"

    echo "  [GPU $gpu_id] 启动 pkl: $stem"
    echo "    pkl: $pkl_path"
    echo "    log: $log_file"

    # 后台启动 render_trajectory.py，通过 CUDA_VISIBLE_DEVICES 绑定 GPU。
    # 用 setsid 让 python.sh 成为新进程组组长，其 fork 出的 Isaac Sim 进程与它同组，
    # 后面按"进程组"杀，避免只杀外壳、留下 python_exe 孤儿占显存 -> 后续任务 OOM。
    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        exec setsid "$PYTHON_BIN" "$RENDER_SCRIPT" \
            --pkl "$pkl_path" \
            --out "$OUT_DIR" \
            --max-envs "$MAX_ENVS" \
            --all \
            --headless \
            > "$log_file" 2>&1
    ) &
    local pid=$!

    gpu_pid[$gpu_id]=$pid
    gpu_log_file[$gpu_id]="$log_file"
    gpu_task_idx[$gpu_id]="$task_idx"
    gpu_start_time[$gpu_id]=$(date +%s)

    echo "    PID=$pid  (GPU $gpu_id)"
}

# ===== 标记某 GPU 完成当前任务（日志已出现 DONE 标记）=====
finish_gpu_task() {
    local gpu_id="$1"
    local pid="${gpu_pid[$gpu_id]}"
    local task_str="${tasks[${gpu_task_idx[$gpu_id]}]}"
    local stem; stem="$(basename "$task_str" .pkl)"

    echo "  [GPU $gpu_id] 检测到完成标记: $stem"

    # 强杀进程组（simulation_app.close() 会卡住），连同 Isaac Sim 子进程一起干掉释放显存
    if kill -0 "$pid" 2>/dev/null; then
        echo "  [GPU $gpu_id] 强杀挂起进程组 PID=$pid"
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi

    unset gpu_pid[$gpu_id]
    unset gpu_log_file[$gpu_id]
    unset gpu_task_idx[$gpu_id]
    unset gpu_start_time[$gpu_id]

    finished=$((finished + 1))
    echo "  [GPU $gpu_id] pkl 完成并释放（完成: $finished / $total_tasks, 失败: $failed）"
}

# ===== 强行终止某个卡上的任务（超时或进程已死）=====
kill_gpu_task() {
    local gpu_id="$1"
    local reason="$2"
    local pid="${gpu_pid[$gpu_id]}"
    local log_file="${gpu_log_file[$gpu_id]}"
    local task_str="${tasks[${gpu_task_idx[$gpu_id]}]}"
    local stem; stem="$(basename "$task_str" .pkl)"
    local elapsed=$(( $(date +%s) - ${gpu_start_time[$gpu_id]} ))

    echo "  [GPU $gpu_id] ${reason}: $stem (已运行 ${elapsed}s)"

    # 进程意外退出（崩溃）时，把日志尾部打出来便于直接看到崩因
    if [[ "$reason" == *意外退出* ]]; then
        if [[ -f "$log_file" ]]; then
            echo "  [GPU $gpu_id] ---- 日志尾部 ($log_file) ----"
            tail -n 20 "$log_file" | while read -r line || [[ -n "$line" ]]; do echo "    $line"; done
            echo "  [GPU $gpu_id] ---- 日志结束 ----"
        else
            echo "  [GPU $gpu_id] 日志文件不存在: $log_file"
        fi
    fi

    # 强杀进程组：即使外壳 python.sh 已退出，其 fork 的 Isaac Sim 子进程也可能还占显存，
    # 进程组仍存在，按组杀才能把孤儿一并回收。无条件尝试。
    echo "  [GPU $gpu_id] 强杀进程组 PID=$pid"
    kill -KILL -- "-$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true

    unset gpu_pid[$gpu_id]
    unset gpu_log_file[$gpu_id]
    unset gpu_task_idx[$gpu_id]
    unset gpu_start_time[$gpu_id]

    failed=$((failed + 1))
    echo "  [GPU $gpu_id] 已释放（成功: $finished, 失败/超时: $failed）"
}

# ===== 主循环 =====
echo "===== 开始调度 ($(date)) ====="
echo "GPU 数量: $NUM_GPUS"
echo "pkl 任务数: $task_count"
echo "pkl 目录: $DS_DIR"
echo "输出目录: $OUT_DIR"
echo "单任务超时: ${TASK_TIMEOUT}s"
echo "轮询间隔: ${POLL_INTERVAL}s"
echo ""

while true; do
    now_ts=$(date +%s)

    # 1. 检查哪些 GPU 空闲，分配新任务
    for ((gpu_id=0; gpu_id<NUM_GPUS; gpu_id++)); do
        if [[ -z "${gpu_pid[$gpu_id]+x}" ]] && [[ $next_task -lt $task_count ]]; then
            launch_task "$gpu_id" "${tasks[$next_task]}" "$next_task"
            next_task=$((next_task + 1))
        fi
    done

    # 2. 检查已运行的 GPU 是否完成（日志出现 DONE 标记）
    for gpu_id in "${!gpu_pid[@]}"; do
        log_file="${gpu_log_file[$gpu_id]}"
        if [[ -f "$log_file" ]] && grep -q "$DONE_MARKER" "$log_file"; then
            finish_gpu_task "$gpu_id"
        fi
    done

    # 3. 检测超时任务（超过 TASK_TIMEOUT 还没出现完成标记）
    for gpu_id in "${!gpu_pid[@]}"; do
        elapsed=$(( now_ts - ${gpu_start_time[$gpu_id]} ))
        if [[ $elapsed -ge $TASK_TIMEOUT ]]; then
            kill_gpu_task "$gpu_id" "任务超时 ($TASK_TIMEOUT s)"
        fi
    done

    # 4. 检测进程是否已死（崩溃退出但日志无完成标记）
    for gpu_id in "${!gpu_pid[@]}"; do
        pid="${gpu_pid[$gpu_id]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            kill_gpu_task "$gpu_id" "进程已意外退出"
        fi
    done

    # 5. 终止条件：所有任务已分配且无 GPU 在跑
    if [[ $next_task -ge $task_count ]] && [[ ${#gpu_pid[@]} -eq 0 ]]; then
        break
    fi

    # 6. 所有卡都忙或已无待分配任务 → 等待后重试；否则短暂 sleep 防抖
    if [[ ${#gpu_pid[@]} -ge $NUM_GPUS ]] || [[ $next_task -ge $task_count ]]; then
        sleep "$POLL_INTERVAL"
    else
        sleep 1
    fi
done

# ===== 收尾 =====
echo ""
echo "===== 全部完成！($(date)) ====="
echo "pkl 总数: $total_tasks"
echo "成功: $finished"
echo "失败/超时: $failed"
echo "输出目录: $OUT_DIR"
if [[ $failed -gt 0 ]]; then
    echo "⚠️  有 $failed 个 pkl 任务未完成（崩溃/超时），请检查对应日志；重跑本脚本会靠 pkl 内 per-traj _DONE_ 增量补回。"
fi
