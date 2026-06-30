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
# cd /kpfs_dataset_ssd/dataset/baiyu/code/gt_gen_hanfeng
# bash scripts/bash/kejian/render_seam_scheduler.sh

set -euo pipefail

# ===== 可调参数 =====
NPY_DIR="/kpfs_dataset_ssd/dataset/render_kejian/initpose_full/data"
OBJ_DIR="/kpfs_dataset_ssd/dataset/render_kejian/segment_output_sub"
OUT_DIR="/kpfs_dataset_ssd/dataset/render_kejian/render_outs"
DONE_TMPDIR="/tmp/render_seam_done"
NUM_GPUS=8
POLL_INTERVAL=5          # 轮询间隔（秒）
TASK_TIMEOUT=3000        # 单个任务超时（秒）
MAX_ENVS=2               # --max-envs 默认值
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
    stem="${stem_dir%/}"; stem="${stem##*/}"   # 取目录名，代替 basename
    obj_path="$OBJ_DIR/${stem}_part/${stem}_part_watertight.obj"

    if [[ ! -f "$obj_path" ]]; then
        echo "  [跳过] obj 不存在: $obj_path"
        continue
    fi

    for npy in "$stem_dir"seam_*.npy; do
        [[ -f "$npy" ]] || continue
        npy_name="${npy##*/}"; npy_name="${npy_name%.npy}"   # 如 seam_47，代替 basename
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
declare -A gpu_start_time # gpu_id -> 任务启动时间戳 (epoch seconds)
next_task=0               # 下一个待分配任务的索引
finished=0                # 已完成任务数
failed=0                  # 失败/超时任务数

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
    #
    # 用 setsid 让 python.sh 成为新进程组的组长，这样它 fork 出来的真正的
    # Isaac Sim 进程 (python_exe) 与它同组。后面杀任务时按"进程组"杀，
    # 否则只杀掉 python.sh 外壳、留下 python_exe 孤儿继续占着显存 ->
    # 几轮之后每张卡显存被僵尸进程占满 -> 新任务 OOM 卡到超时。
    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        export RENDER_DONE_FILE="$done_file"
        exec setsid "$PYTHON_BIN" "$RENDER_SCRIPT" \
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
    gpu_start_time[$gpu_id]=$(date +%s)

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
        cat "$done_file" | while read -r line || [[ -n "$line" ]]; do echo "    $line"; done
    fi

    # 强杀进程（simulation_app.close() 会卡住）
    # 按进程组杀，连同 python.sh fork 出来的 Isaac Sim 子进程一起干掉，释放显存
    if kill -0 "$pid" 2>/dev/null; then
        echo "  [GPU $gpu_id] 强杀挂起进程组 PID=$pid"
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi

    # 清理哨兵
    rm -f "$done_file"

    # 重置该卡状态
    unset gpu_pid[$gpu_id]
    unset gpu_done_file[$gpu_id]
    unset gpu_task_idx[$gpu_id]
    unset gpu_start_time[$gpu_id]

    finished=$((finished + 1))
    echo "  [GPU $gpu_id] 已释放（进度: $finished / $task_count）"
}

# ===== 强行终止某个卡上的任务（超时或进程已死）=====
kill_gpu_task() {
    local gpu_id="$1"
    local reason="$2"
    local pid="${gpu_pid[$gpu_id]}"
    local done_file="${gpu_done_file[$gpu_id]}"
    local task_str="${tasks[${gpu_task_idx[$gpu_id]}]}"

    IFS='|' read -r stem npy_path obj_path seam_idx <<< "$task_str"
    local elapsed=$(( $(date +%s) - ${gpu_start_time[$gpu_id]} ))

    echo "  [GPU $gpu_id] ${reason}: $stem / $seam_idx (已运行 ${elapsed}s)"

    # 进程意外退出（崩溃）时，把对应日志尾部打出来，便于直接看到崩因
    if [[ "$reason" == *意外退出* ]]; then
        local log_file="$OUT_DIR/logs/${stem}__${seam_idx}.log"
        if [[ -f "$log_file" ]]; then
            echo "  [GPU $gpu_id] ---- 日志尾部 ($log_file) ----"
            tail -n 20 "$log_file" | while read -r line || [[ -n "$line" ]]; do echo "    $line"; done
            echo "  [GPU $gpu_id] ---- 日志结束 ----"
        else
            echo "  [GPU $gpu_id] 日志文件不存在: $log_file"
        fi
    fi

    # 强杀进程组：即使外壳 python.sh 已退出（意外退出场景），它 fork 的
    # Isaac Sim 子进程也可能还活着占着显存，进程组仍存在，按组杀才能把孤儿
    # 一并回收。无条件尝试，避免漏掉占显存的僵尸进程。
    echo "  [GPU $gpu_id] 强杀进程组 PID=$pid"
    kill -KILL -- "-$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true

    # 清理
    rm -f "$done_file"
    unset gpu_pid[$gpu_id]
    unset gpu_done_file[$gpu_id]
    unset gpu_task_idx[$gpu_id]
    unset gpu_start_time[$gpu_id]

    failed=$((failed + 1))
    echo "  [GPU $gpu_id] 已释放（成功: $finished, 失败/超时: $failed）"
}

# ===== 主循环 =====
echo "===== 开始调度 ($(date)) ====="
echo "GPU 数量: $NUM_GPUS"
echo "总任务数: $task_count"
echo "输出目录: $OUT_DIR"
echo "哨兵目录: $DONE_TMPDIR"
echo "轮询间隔: ${POLL_INTERVAL}s"
echo ""

while [[ $((finished + failed)) -lt $task_count ]]; do
    now_ts=$(date +%s)

    # 1. 检查哪些 GPU 空闲，分配新任务
    for ((gpu_id=0; gpu_id<NUM_GPUS; gpu_id++)); do
        if [[ -z "${gpu_pid[$gpu_id]+x}" ]] && [[ $next_task -lt $task_count ]]; then
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

    # 3. 检测超时任务（超过 TASK_TIMEOUT 还没有哨兵）
    for gpu_id in "${!gpu_pid[@]}"; do
        elapsed=$(( now_ts - ${gpu_start_time[$gpu_id]} ))
        if [[ $elapsed -ge $TASK_TIMEOUT ]]; then
            kill_gpu_task "$gpu_id" "任务超时 ($TASK_TIMEOUT s)"
        fi
    done

    # 4. 检测进程是否已死（崩溃退出但没写哨兵）
    for gpu_id in "${!gpu_pid[@]}"; do
        pid="${gpu_pid[$gpu_id]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            kill_gpu_task "$gpu_id" "进程已意外退出"
        fi
    done

    # 5. 如果全部完成了，退出循环
    if [[ $((finished + failed)) -ge $task_count ]]; then
        break
    fi

    # 6. 如果所有卡都忙或已无待分配任务，等待后重试
    if [[ ${#gpu_pid[@]} -ge $NUM_GPUS ]] || [[ $next_task -ge $task_count ]]; then
        sleep "$POLL_INTERVAL"
    else
        # 有空闲卡且有任务但分配不上？可能是刚释放的，短暂 sleep 防抖
        sleep 1
    fi
done

# ===== 收尾 =====
# 清理可能残留的哨兵文件
rm -f "$DONE_TMPDIR"/*.done

echo ""
echo "===== 全部完成！($(date)) ====="
echo "总任务数: $task_count"
echo "成功: $finished"
echo "失败/超时: $failed"
echo "输出目录: $OUT_DIR"
if [[ $failed -gt 0 ]]; then
    echo "⚠️  有 $failed 个任务未完成，请检查对应日志。"
fi
