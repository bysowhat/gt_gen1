#!/usr/bin/env bash
#
# 多卡调度器：将 initpose_full/data 下每个工件(stem)分配到 8 张 GPU 上执行。
#
# 原理：
#   任务单元 = 一个工件(stem)。每个任务启动一个 render_seam.py 子进程，传 --seam-dir
#   指向该工件目录，进程内一次性渲染其下全部 seam_*.npy（最多 --max-seams-per-proc 条），
#   从而把 Isaac Sim 启动 / 建场景 / 材质编译等与 seam 无关的固定开销摊销到多条 seam 上。
#   进程设环境变量 RENDER_DONE_FILE 指向唯一哨兵（/tmp/render_seam_done/<stem>.done），
#   渲完后写入哨兵，内含：
#     complete=true|false  remaining=<未完成 seam 数>  done_this_run=<本进程渲染条数>
#   调度器每 5 秒轮询哨兵：出现即 kill 该进程释放显存；再据 complete 标志——
#     complete=true  → 整工件完成，销账；
#     complete=false → 因 cap 命中仍有 seam 未渲，把该工件重排回队列续跑（靠 render_seam.py
#                      的 skip-on-resume 跳过已完成的 seam）。
#   崩溃/超时(无完整哨兵)判失败，靠重跑本脚本时 skip-on-resume 增量补回。
#
# 用法：
# cd /kpfs_dataset_ssd/dataset/baiyu/code/gt_gen_hanfeng
# bash scripts/bash/kejian/render_seam_scheduler.sh [--max-envs N] [--max-seams-per-proc N] [--dry-run]

set -euo pipefail

# ===== 可调参数 =====
NPY_DIR="/kpfs_dataset_ssd/dataset/render_kejian/initpose_full/data"
OBJ_DIR="/kpfs_dataset_ssd/dataset/render_kejian/segment_output_sub"
OUT_DIR="/kpfs_dataset/dataset/render_kejian/render_outs"
DONE_TMPDIR="/tmp/render_seam_done"
NUM_GPUS=8
POLL_INTERVAL=5          # 轮询间隔（秒）
TASK_TIMEOUT=8000        # 单个任务(一个工件、最多 MAX_SEAMS_PER_PROC 条 seam)超时（秒）
MAX_ENVS=2               # --max-envs 默认值
MAX_SEAMS_PER_PROC=8     # 单进程最多渲染多少条 seam 即退出（剩余由调度器重排续跑；0=不限）
PYTHON_BIN="/workspace/isaaclab/_isaac_sim/python.sh"
RENDER_SCRIPT="render/render_seam.py"
DRY_RUN=false

# ===== 解析参数 =====
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --max-envs) MAX_ENVS="$2"; shift 2 ;;
        --max-seams-per-proc) MAX_SEAMS_PER_PROC="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ===== 确保临时目录存在 =====
mkdir -p "$DONE_TMPDIR"

# ===== 收集全部任务 =====
echo "===== 收集任务列表 ====="
tasks=()          # 每个元素: "stem|stem_dir|obj_path"（一个工件一个任务，进程内渲其全部 seam）
task_count=0

for stem_dir in "$NPY_DIR"/*/; do
    stem="${stem_dir%/}"; stem="${stem##*/}"   # 取目录名，代替 basename
    obj_path="$OBJ_DIR/${stem}_part/${stem}_part_watertight.obj"

    if [[ ! -f "$obj_path" ]]; then
        echo "  [跳过] obj 不存在: $obj_path"
        continue
    fi

    # 该工件目录下至少要有一条 seam_*.npy 才算一个任务
    shopt -s nullglob
    seam_list=("$stem_dir"seam_*.npy)
    shopt -u nullglob
    if [[ ${#seam_list[@]} -eq 0 ]]; then
        echo "  [跳过] 无 seam_*.npy: $stem_dir"
        continue
    fi

    tasks+=("${stem}|${stem_dir%/}|${obj_path}")
    task_count=$((task_count + 1))
done

echo "共收集到 $task_count 个工件任务"
echo ""

if [[ $task_count -eq 0 ]]; then
    echo "没有待执行的任务，退出。"
    exit 0
fi

if $DRY_RUN; then
    echo "===== [DRY RUN] 打印任务列表（前 20 个）====="
    for ((i=0; i<task_count && i<20; i++)); do
        IFS='|' read -r stem stem_dir obj_path <<< "${tasks[$i]}"
        echo "  [$i] stem=$stem  dir=$stem_dir  obj=$obj_path"
    done
    if [[ $task_count -gt 20 ]]; then
        echo "  ... 还有 $((task_count - 20)) 个任务未显示"
    fi
    echo ""
    echo "===== 命令预览（前 3 个）====="
    for ((i=0; i<task_count && i<3; i++)); do
        IFS='|' read -r stem stem_dir obj_path <<< "${tasks[$i]}"
        done_file="$DONE_TMPDIR/${stem}.done"
        echo "RENDER_DONE_FILE=$done_file \\"
        echo "  $PYTHON_BIN $RENDER_SCRIPT --obj '$obj_path' --seam-dir '$stem_dir' --out '$OUT_DIR' --max-envs=$MAX_ENVS --max-seams-per-proc=$MAX_SEAMS_PER_PROC --headless"
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
finished=0                # 已完成工件数（整 stem 全部 seam 渲完）
failed=0                  # 失败/超时任务数
requeued=0                # 因 cap 命中被重排续跑的次数
total_stems=$task_count   # 工件总数（tasks 会因续跑动态增长，用它做进度分母）

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

    IFS='|' read -r stem stem_dir obj_path <<< "$task_str"

    local done_file="$DONE_TMPDIR/${stem}.done"
    # 先删掉可能残留的旧哨兵
    rm -f "$done_file"

    local log_file="$OUT_DIR/logs/${stem}.log"
    mkdir -p "$(dirname "$log_file")" "$OUT_DIR"

    echo "  [GPU $gpu_id] 启动工件: $stem"
    echo "    dir: $stem_dir"
    echo "    obj: $obj_path"
    echo "    log: $log_file"

    # 在后台启动 render_seam.py，通过 CUDA_VISIBLE_DEVICES 绑定 GPU。
    # 进程内渲染该工件 --seam-dir 下全部 seam（最多 --max-seams-per-proc 条），跑完写
    # RENDER_DONE_FILE 哨兵（内含 complete 标志）。
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
            --seam-dir "$stem_dir" \
            --out "$OUT_DIR" \
            --max-envs="$MAX_ENVS" \
            --max-seams-per-proc="$MAX_SEAMS_PER_PROC" \
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
    local task_idx="${gpu_task_idx[$gpu_id]}"
    local task_str="${tasks[$task_idx]}"

    IFS='|' read -r stem stem_dir obj_path <<< "$task_str"

    # 读哨兵内容里的 complete 标志：true=整工件全部 seam 渲完；false=因 cap 命中仍有剩余
    local complete="true"
    local remaining="0"
    if [[ -f "$done_file" ]]; then
        complete="$(grep -m1 '^complete=' "$done_file" | cut -d= -f2)"
        remaining="$(grep -m1 '^remaining=' "$done_file" | cut -d= -f2)"
        [[ -z "$complete" ]] && complete="true"   # 兜底（旧格式无该字段视为完成）
        echo "  [GPU $gpu_id] 哨兵已写入: $stem (complete=$complete, remaining=$remaining)"
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

    if [[ "$complete" == "true" ]]; then
        finished=$((finished + 1))
        echo "  [GPU $gpu_id] 工件完成并释放（完成: $finished / $total_stems, 续跑重排: $requeued, 失败: $failed）"
    else
        # 因 cap 命中、仍有 seam 未渲 → 把该工件任务追加回队列续跑（下一轮某卡接手，
        # 靠 skip-on-resume 跳过已完成的 seam）。每次续跑必然有进展，故收敛。
        tasks+=("$task_str")
        requeued=$((requeued + 1))
        echo "  [GPU $gpu_id] 工件未完（剩 $remaining 条 seam）→ 重排续跑（续跑重排累计: $requeued）"
    fi
}

# ===== 强行终止某个卡上的任务（超时或进程已死）=====
kill_gpu_task() {
    local gpu_id="$1"
    local reason="$2"
    local pid="${gpu_pid[$gpu_id]}"
    local done_file="${gpu_done_file[$gpu_id]}"
    local task_str="${tasks[${gpu_task_idx[$gpu_id]}]}"

    IFS='|' read -r stem stem_dir obj_path <<< "$task_str"
    local elapsed=$(( $(date +%s) - ${gpu_start_time[$gpu_id]} ))

    echo "  [GPU $gpu_id] ${reason}: $stem (已运行 ${elapsed}s)"

    # 进程意外退出（崩溃）时，把对应日志尾部打出来，便于直接看到崩因
    if [[ "$reason" == *意外退出* ]]; then
        local log_file="$OUT_DIR/logs/${stem}.log"
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
echo "工件任务数: $task_count"
echo "单进程 seam 上限: $MAX_SEAMS_PER_PROC"
echo "输出目录: $OUT_DIR"
echo "哨兵目录: $DONE_TMPDIR"
echo "轮询间隔: ${POLL_INTERVAL}s"
echo ""

while true; do
    now_ts=$(date +%s)

    # 1. 检查哪些 GPU 空闲，分配新任务（用动态队列长度 ${#tasks[@]} 做上界，含续跑重排进来的）
    for ((gpu_id=0; gpu_id<NUM_GPUS; gpu_id++)); do
        if [[ -z "${gpu_pid[$gpu_id]+x}" ]] && [[ $next_task -lt ${#tasks[@]} ]]; then
            launch_task "$gpu_id" "${tasks[$next_task]}" "$next_task"
            next_task=$((next_task + 1))
        fi
    done

    # 2. 检查已运行的 GPU 是否完成（哨兵文件出现）；finish 内部据 complete 标志决定销账或重排
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

    # 5. 终止条件：队列已分配完（含续跑重排）且无 GPU 在跑
    if [[ $next_task -ge ${#tasks[@]} ]] && [[ ${#gpu_pid[@]} -eq 0 ]]; then
        break
    fi

    # 6. 如果所有卡都忙或已无待分配任务，等待后重试
    if [[ ${#gpu_pid[@]} -ge $NUM_GPUS ]] || [[ $next_task -ge ${#tasks[@]} ]]; then
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
echo "工件总数: $total_stems"
echo "成功(整工件完成): $finished"
echo "续跑重排次数: $requeued"
echo "失败/超时: $failed"
echo "输出目录: $OUT_DIR"
if [[ $failed -gt 0 ]]; then
    echo "⚠️  有 $failed 个工件任务未完成（崩溃/超时），请检查对应日志；重跑本脚本会靠 skip-on-resume 增量补回。"
fi
