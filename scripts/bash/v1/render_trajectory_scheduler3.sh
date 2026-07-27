#!/usr/bin/env bash
#
# 多卡调度器（type3 / ObserveAnything 本地版）：将 type3 下采样目录下每个 pkl 分配到多张 GPU 上执行。
#
# 与 render_trajectory_scheduler2.sh（type2）逻辑完全一致，只改三处 + 本地 python：
#   · DS_DIR      → type3 下采样 pkl 目录（ObserveAnythingScene 产物）
#   · RENDER_SCRIPT → render/render_trajectory_observe.py（整场景 USD 直接引用渲染）
#   · OUT_DIR     → type3 渲染输出目录
#   · python      → 本地 conda env_isaaclab（不用远程 /workspace/.../python.sh）
#
# 原理（同 scheduler2）：
#   任务单元 = 一个 pkl。每任务起一个 render_trajectory_observe.py 子进程，--all 一次渲其全部
#   有采样结果的成功轨迹。子进程渲完在 stdout 末尾打印 "RENDER_TRAJECTORY_DONE"，之后
#   simulation_app.close() 常卡住占显存 → 不靠进程退出判完成，改为轮询 grep 日志出现 DONE 标记后
#   按进程组强杀释放显存。崩溃/超时(无标记)判失败；重跑靠 pkl 内 per-traj _DONE_ 增量补回。
#
# ⚠️ 显存：整场景（如 full_warehouse）+ 多 env 很吃显存（本地 11.6G 卡易 OOM）。默认 MAX_ENVS=1，
#    先单 env 跑通再上调。
#
# 用法：
# cd /home/a/Projects/Github/gt_gen_hanfeng
# bash scripts/bash/v1/render_trajectory_scheduler3.sh [--task-timeout N] [--max-envs N] [--dry-run]

set -euo pipefail

# ===== 可调参数 =====
DS_DIR="/media/a/新加卷/tempt/7_ds"                  # type3 下采样 pkl 目录
OUT_DIR="/media/a/新加卷/tempt/7_render"             # type3 渲染输出目录
POLL_INTERVAL=5          # 轮询间隔（秒）
TASK_TIMEOUT=8000        # 单个任务(一个 pkl，--all 渲其全部轨迹)超时（秒）
MAX_ENVS=1               # --max-envs 固定值（整场景很吃显存，默认 1）
# 本地 conda env_isaaclab（数组形式：--no-capture-output 保证日志实时刷出，供 grep DONE）
PYTHON_CMD=(conda run --no-capture-output -n env_isaaclab python)
RENDER_SCRIPT="render/render_trajectory_observe.py"
DONE_MARKER="RENDER_TRAJECTORY_DONE"   # 完成标记（render_trajectory_observe.py 末尾打印）
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
        echo "  ${PYTHON_CMD[*]} $RENDER_SCRIPT --pkl '${tasks[$i]}' --out '$OUT_DIR' --max-envs $MAX_ENVS --all --headless"
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

    # 后台启动 render_trajectory_observe.py，通过 CUDA_VISIBLE_DEVICES 绑定 GPU。
    # 用 setsid 让 conda/python 成为新进程组组长，其 fork 出的 Isaac Sim 进程与它同组，
    # 后面按"进程组"杀，避免只杀外壳、留下 python_exe 孤儿占显存 -> 后续任务 OOM。
    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        exec setsid "${PYTHON_CMD[@]}" "$RENDER_SCRIPT" \
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

    # 强杀进程组：即使外壳已退出，其 fork 的 Isaac Sim 子进程也可能还占显存，
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
echo "MAX_ENVS: $MAX_ENVS"
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
