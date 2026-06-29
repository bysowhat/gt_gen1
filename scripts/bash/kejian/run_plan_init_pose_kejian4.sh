#!/usr/bin/env bash
# 批量跑 plan_init_pose_kejian2（--all-seams）：对 segment_output_sub 下所有工件的全部焊缝求初始位姿。
#
# 数据布局（远程，/kpfs_dataset 在远程机器本地、ssh debug 可达）：
#   · 工件 mesh : $SUB_DIR/<stem>_part/<stem>_part_watertight.obj
#   · 焊缝 json : $SUB_DIR/<stem>_part/<stem>_weld_angle3.json（已随 obj 放在【同一目录】下，
#                 不再去 SEG_DIR 递归找；kejian2 的 load_welds 只认这种 json，
#                 不认子文件夹里的 seam_*.pkl —— 那是旧管线输出、缺 bisector，无法直接喂给 kejian2）
#   任务 = 每个 watertight obj + 它同目录下的 <stem>_weld_angle3.json（两者俱在才跑）。
#
# 输出 / 日志（每工件按 <stem> 单独建子项，天然不重名）：
#   · 结果 : $DATA_DIR/<stem>/seam_<idx>.npy        （kejian2 --all-seams --out 逐焊缝落盘）
#   · 求解摘要 : $LOGS_DIR/<stem>.summary.txt        （kejian2 --log：各 seam 合格/已存条数、成功率）
#   · 运行日志 : $LOGS_DIR/<stem>.log                （该工件 python 的 stdout+stderr）
#   · 完成标记 : $DATA_DIR/<stem>/.done              （rc=0 后写；重跑时据此跳过 → 可断点续跑）
#
# 并行：严格「每进程独占一卡」。GPUS 列出要用的卡号（默认自动探测全部可见 GPU）；
#       N=卡数，487 个工件按下标对 N 取模分到各卡，每卡一个 worker 串行跑自己那一份，
#       worker 内每工件设 CUDA_VISIBLE_DEVICES=<该卡> 独占。本节点只有 1 卡时即串行。
#
# 用法（在远程跑；本机无 /kpfs 挂载）：
#   ssh debug 'bash /kpfs_dataset/dataset/baiyu/code/render/gt_gen_hanfeng/scripts/bash/run_plan_init_pose_kejian2.sh'
#   # 覆盖参数示例：
#   GPUS=0,1,2,3 bash .../run_plan_init_pose_kejian2.sh          # 4 卡并行
#   LIMIT=2 bash .../run_plan_init_pose_kejian2.sh                # 只跑前 2 个工件（冒烟测试）
#   FILTER_SHORT=1 bash .../run_plan_init_pose_kejian2.sh         # 丢弃 <3cm 短焊缝
#   PER_WP_TIMEOUT=1800 bash .../run_plan_init_pose_kejian2.sh    # 单工件超时兜底（秒）

set -uo pipefail   # 故意不加 -e：单个工件失败要继续跑下一个，不中断整批

# ---- 路径/参数（环境变量可覆盖）----
SUB_DIR="${SUB_DIR:-/kpfs_dataset_ssd/dataset/render_kejian/segment_output_subs/part_4}"
DATA_DIR="${DATA_DIR:-/kpfs_dataset_ssd/dataset/render_kejian/initpose_full/data}"
LOGS_DIR="${LOGS_DIR:-/kpfs_dataset_ssd/dataset/render_kejian/initpose_full/logs}"
PY="${PY:-/workspace/isaaclab/_isaac_sim/python.sh}"                          # 远程 isaac python（含 torch+cuda）
CUROBO_SRC="${CUROBO_SRC:-/kpfs_dataset/dataset/baiyu/code/render/curobo/src}" # curobo 源码（加进 PYTHONPATH）
FILTER_SHORT="${FILTER_SHORT:-1}"     # 1=加 --filter-short（丢弃 <3cm 短焊缝，不求解、记入 summary）；默认开，设 0 关闭
LIMIT="${LIMIT:-0}"                   # >0：只跑前 N 个工件（按 stem 排序，调试用）
PER_WP_TIMEOUT="${PER_WP_TIMEOUT:-0}" # >0：单工件超时秒数（兜底卡死，需要 timeout 命令）；0=不限

# GPU 列表：默认自动探测全部可见卡；严格每进程独占一卡
if [[ -z "${GPUS:-}" ]]; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)"
    [[ -z "$GPUS" ]] && GPUS="0"
fi
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
# N=${#GPU_ARR[@]}
N=1

# 仓库根：本脚本在 scripts/bash/ 下
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPT="$REPO/scripts/plan_init_pose_kejian2.py"
export PYTHONPATH="$CUROBO_SRC${PYTHONPATH:+:$PYTHONPATH}"

# ---- 校验 ----
[[ -x "$PY" ]]        || { echo "[run] python 不可执行: $PY" >&2; exit 1; }
[[ -f "$SCRIPT" ]]    || { echo "[run] 找不到 plan_init_pose_kejian2.py: $SCRIPT" >&2; exit 1; }
[[ -d "$SUB_DIR" ]]   || { echo "[run] 工件目录不存在: $SUB_DIR" >&2; exit 1; }
[[ -d "$CUROBO_SRC" ]]|| { echo "[run] curobo 源码目录不存在: $CUROBO_SRC" >&2; exit 1; }
mkdir -p "$DATA_DIR" "$LOGS_DIR"

# ---- 建任务表：遍历 SUB_DIR 下的 watertight obj，取同目录下的 <stem>_weld_angle3.json ----
# weld json 现已随 obj 放在同一个 <stem>_part 目录里 → 只需对 $SUB_DIR 做【一次】find 遍历，
# 每个 obj 由 dirname + <stem> 直接拼出同目录 json 路径（单次 stat），无需再递归 SEG_DIR。
# 注：数据集可能正被并发生成 → obj 数随时间增长，本表为启动时刻快照；重跑会据 .done 跳过已完成者，
#     可安全增量补跑后来新增的工件。终端长时间无新行 ≠ 卡死，多半在扫盘 / load 关节表。
echo "[run] (1/2) 扫描 $SUB_DIR 下的 *_part_watertight.obj（唯一的一次目录遍历，网络盘较慢请稍候）..."
mapfile -t ALL_OBJ < <(find "$SUB_DIR" -name "*_part_watertight.obj" 2>/dev/null | sort)
echo "[run] (2/2) 找到 ${#ALL_OBJ[@]} 个 obj，逐个取同目录下的 <stem>_weld_angle3.json ..."
JOBS=()
matched=0
missing=0
scanned=0
for obj in "${ALL_OBJ[@]}"; do
    d="$(dirname "$obj")"
    st="$(basename "$obj" _part_watertight.obj)"   # <stem>_part_watertight.obj -> <stem>
    jf="$d/${st}_weld_angle3.json"                  # 同目录 weld json，单次 stat
    if [[ -f "$jf" ]]; then
        JOBS+=("$(printf '%s\t%s\t%s' "$st" "$jf" "$obj")")
        matched=$((matched + 1))
    else
        missing=$((missing + 1))
    fi
    scanned=$((scanned + 1))
    # 每查 200 个 obj 报一次进度，避免大目录下长时间静默
    (( scanned % 200 == 0 )) && echo "[run]       已检查 $scanned/${#ALL_OBJ[@]}，命中 $matched ..."
done
# 按 stem 排序，保证 worker 分片可复现
if [[ ${#JOBS[@]} -gt 0 ]]; then
    mapfile -t JOBS < <(printf '%s\n' "${JOBS[@]}" | sort)
fi
echo "[run] (2/2) 匹配完成：$matched 个 obj 有同目录 weld json（缺 json $missing 个，共 ${#ALL_OBJ[@]} 个 obj）"
total=${#JOBS[@]}
if [[ "$total" -eq 0 ]]; then
    echo "[run] 在 $SUB_DIR 下没找到任何「obj + 同目录 weld json」的工件，无可跑任务" >&2
    exit 1
fi
if [[ "$LIMIT" -gt 0 && "$total" -gt "$LIMIT" ]]; then
    JOBS=("${JOBS[@]:0:$LIMIT}")
    total=$LIMIT
fi

echo "[run] 工件目录   : $SUB_DIR"
echo "[run] 焊缝 json   : $SUB_DIR/<stem>_part/<stem>_weld_angle3.json（与 obj 同目录）"
echo "[run] 结果目录   : $DATA_DIR"
echo "[run] 日志目录   : $LOGS_DIR"
echo "[run] python     : $PY"
echo "[run] PYTHONPATH : $CUROBO_SRC"
echo "[run] GPU 卡      : $GPUS  （N=$N，每进程独占一卡）"
echo "[run] filter_short: $FILTER_SHORT   per_wp_timeout: ${PER_WP_TIMEOUT}s   limit: $LIMIT"
echo "[run] 共 $total 个工件待处理"

# ---- 单工件求解 ----
run_one() {
    local gpu="$1" idx="$2" stem="$3" jf="$4" obj="$5"
    local outdir="$DATA_DIR/$stem"
    local log="$LOGS_DIR/$stem.log"
    local summary="$LOGS_DIR/$stem.summary.txt"

    if [[ -f "$outdir/.done" ]]; then
        echo "[gpu$gpu][$idx/$total] $stem 已完成，跳过"
        return 0
    fi
    mkdir -p "$outdir"
    : > "$log"

    local extra=()
    [[ "$FILTER_SHORT" == "1" ]] && extra+=(--filter-short)

    # 启动前先报「开始」：首个工件要 load 1.85GB 关节表 + 求解，python 输出全进 $log，
    # 终端这段时间无新行属正常。想看实时细节： tail -f "$log"
    local t0=$SECONDS
    echo "[gpu$gpu][$idx/$total] $stem 开始（首个工件需 load 关节表，稍慢；tail -f $log 看实时）"

    # 注意：--all-seams 时 --out 是【目录】，每条焊缝存 <out>/seam_<idx>.npy；--log 写求解摘要 txt。
    local runner=(env "CUDA_VISIBLE_DEVICES=$gpu" PYTHONUNBUFFERED=1 "$PY" -u "$SCRIPT"
        --solve-kejian2 --all-seams
        --obj "$obj" --weld-json "$jf"
        --out "$outdir" --log "$summary" "${extra[@]}")

    if [[ "$PER_WP_TIMEOUT" -gt 0 ]]; then
        timeout "$PER_WP_TIMEOUT" "${runner[@]}" >"$log" 2>&1
    else
        "${runner[@]}" >"$log" 2>&1
    fi
    local rc=$?
    local dt=$((SECONDS - t0))

    if [[ $rc -eq 0 ]]; then
        touch "$outdir/.done"
        echo "[gpu$gpu][$idx/$total] $stem 完成 (${dt}s) -> $outdir"
    else
        echo "[gpu$gpu][$idx/$total] $stem 失败 rc=$rc (${dt}s)（详见 $log）" >&2
    fi
}

# ---- 每卡一个 worker：处理下标 ≡ w (mod N) 的工件，串行独占该卡 ----
worker() {
    local w="$1"
    local gpu="${GPU_ARR[$w]}"
    local i
    for ((i = w; i < total; i += N)); do
        IFS=$'\t' read -r stem jf obj <<< "${JOBS[$i]}"
        run_one "$gpu" "$((i + 1))" "$stem" "$jf" "$obj"
    done
}

echo "[run] 启动 $N 个 worker（每卡一个，独占）开始求解 $total 个工件 ..."
for ((w = 0; w < N; w++)); do
    worker "$w" &
done
wait

# ---- 汇总 ----
done_cnt=0
for j in "${JOBS[@]}"; do
    IFS=$'\t' read -r stem _ _ <<< "$j"
    [[ -f "$DATA_DIR/$stem/.done" ]] && done_cnt=$((done_cnt + 1))
done
fail_cnt=$((total - done_cnt))
echo ""
echo "[run] 全部结束：完成 $done_cnt / 失败 $fail_cnt（共 $total）"
echo "[run] 结果目录 $DATA_DIR ；日志目录 $LOGS_DIR"
[[ "$fail_cnt" -gt 0 ]] && exit 1
exit 0
