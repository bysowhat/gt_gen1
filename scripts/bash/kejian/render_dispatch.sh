#!/usr/bin/env bash
# 多 GPU 并行渲染调度器：把「每个工件 × 每条焊缝」的渲染作业分摊到本节点所有 GPU 上同时跑。
#
# 范式与 run_plan_init_pose_kejian0.sh 一致（每进程独占一卡、.done 标记断点续跑），唯一区别：
# 渲染进程跑完不会自退（卡在 isaac 的 simulation_app.close()），所以不靠「等进程返回 rc」，
# 而是靠 render_seam.py 写的【完成哨兵文件】——render_seam.sh 轮询到哨兵即 SIGKILL 整个进程
# 组并返回 0，本调度器据其返回码 touch .done，然后让该卡去跑下一个作业。
#
# 数据布局（远程；/kpfs* 仅远程机器本地、ssh debug 可达）：
#   · 工件 mesh   : $SUB_DIR/<stem>_part/<stem>_part_watertight.obj
#   · 焊缝初始位姿: $INITPOSE_DIR/data/<stem>/seam_<idx>.npy（含 workpiece_pose7）
#   作业 = 每个 watertight obj 的每一条 seam_<idx>.npy（两者俱在才跑）。
#
# 输出 / 状态：
#   · 渲染结果 : $OUT/<obj_ascii_stem>/seam_<idx>/pose_*/{left,right}_{rgb.png,depth.exr}+meta.npy
#   · 完成哨兵 : render_seam.py 在结果目录写 _DONE；并写本调度器指定的唯一哨兵
#                $STATE_DIR/<stem>__seam_<idx>.sentinel（worker 轮询用）
#   · 续跑标记 : $STATE_DIR/<stem>__seam_<idx>.done（rc=0 后写；重跑据此跳过 → 断点续跑）
#   · 单作业日志: $STATE_DIR/<stem>__seam_<idx>.log
#
# 用法（在远程跑）：
#   ssh debug 'bash /kpfs_dataset/.../scripts/bash/kejian/render_dispatch.sh'
#   GPUS=0,1,2,3 bash .../render_dispatch.sh        # 指定 4 卡
#   LIMIT=4 bash .../render_dispatch.sh              # 只跑前 4 个作业（冒烟测试）
#   STEMS='柱_aaa 柱_bbb' bash .../render_dispatch.sh # 只跑这些工件（空格分隔；默认全扫）
#   SEAM_IDXS='40 41' bash .../render_dispatch.sh     # 只跑这些焊缝号（默认该工件下全部 seam_*.npy）
#   OUT=/path MAX_ENVS=4 bash .../render_dispatch.sh  # 透传渲染参数给 render_seam.sh

set -uo pipefail   # 故意不加 -e：单作业失败要继续跑下一个，不中断整批

# ---- 路径/参数（环境变量可覆盖；与 render_seam.sh 默认保持一致）----
SUB_DIR="${SUB_DIR:-/kpfs_dataset_ssd/dataset/render_kejian/segment_output_subs/part_0}"
INITPOSE_DIR="${INITPOSE_DIR:-/kpfs_dataset_ssd/dataset/render_kejian/initpose_full}"
OUT="${OUT:-/kpfs_dataset_ssd/dataset/render_kejian/render_out}"
STATE_DIR="${STATE_DIR:-$OUT/.dispatch_state}"

# 透传给 render_seam.sh 的渲染参数（这里不解释，详见 render_seam.sh）
export PY="${PY:-/workspace/isaaclab/_isaac_sim/python.sh}"
export ROBOT_CFG="${ROBOT_CFG:-/kpfs_dataset/dataset/baiyu/code/render/urdf_12e_260626_remote/ur12e_full.yml}"
export SUB_DIR INITPOSE_DIR OUT
export MAX_ENVS="${MAX_ENVS:-2}"
export SPACING="${SPACING:-40.0}"
export SETTLE_STEPS="${SETTLE_STEPS:-12}"
export HEADLESS="${HEADLESS:-1}"
export FORCE_CONVERT="${FORCE_CONVERT:-0}"
export GRACE_SEC="${GRACE_SEC:-3}"   # 哨兵命中后落盘宽限秒数（透传给 render_seam.sh）

# 作业筛选（调试用）
STEMS="${STEMS:-}"                   # 空=扫 $SUB_DIR 下全部工件；否则只跑列出的 stem（空格分隔）
SEAM_IDXS="${SEAM_IDXS:-}"           # 空=每工件跑其 data/<stem>/ 下全部 seam_*.npy；否则只跑列出的号
LIMIT="${LIMIT:-0}"                  # >0：只跑前 N 个作业（按排序）

DATA_DIR="$INITPOSE_DIR/data"

# 仓库根：本脚本在 scripts/bash/kejian/ 下
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDER_SH="$SCRIPT_DIR/render_seam.sh"

# ---- GPU 列表：默认自动探测全部可见卡；严格每进程独占一卡 ----
if [[ -z "${GPUS:-}" ]]; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)"
    [[ -z "$GPUS" ]] && GPUS="0"
fi
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}

# ---- 校验 ----
[[ -f "$RENDER_SH" ]] || { echo "[dispatch] 找不到 render_seam.sh: $RENDER_SH" >&2; exit 1; }
[[ -x "$PY" ]]        || { echo "[dispatch] python 不可执行: $PY" >&2; exit 1; }
[[ -d "$SUB_DIR" ]]   || { echo "[dispatch] 工件目录不存在: $SUB_DIR" >&2; exit 1; }
[[ -d "$DATA_DIR" ]]  || { echo "[dispatch] 焊缝位姿目录不存在: $DATA_DIR" >&2; exit 1; }
mkdir -p "$OUT" "$STATE_DIR"

# ---- 建作业表：(stem, seam_idx) ----
echo "[dispatch] (1/2) 扫描 $SUB_DIR 下的 *_part_watertight.obj ..."
if [[ -n "$STEMS" ]]; then
    read -r -a STEM_ARR <<< "$STEMS"
else
    mapfile -t ALL_OBJ < <(find "$SUB_DIR" -name "*_part_watertight.obj" 2>/dev/null | sort)
    STEM_ARR=()
    for obj in "${ALL_OBJ[@]}"; do
        STEM_ARR+=("$(basename "$obj" _part_watertight.obj)")
    done
fi

echo "[dispatch] (2/2) 为 ${#STEM_ARR[@]} 个工件枚举焊缝 seam_*.npy ..."
JOBS=()   # 每行 "stem<TAB>seam_idx"
for st in "${STEM_ARR[@]}"; do
    [[ -f "$SUB_DIR/${st}_part/${st}_part_watertight.obj" ]] || continue   # obj 必须在
    if [[ -n "$SEAM_IDXS" ]]; then
        for idx in $SEAM_IDXS; do
            [[ -f "$DATA_DIR/$st/seam_${idx}.npy" ]] && JOBS+=("$(printf '%s\t%s' "$st" "$idx")")
        done
    else
        [[ -d "$DATA_DIR/$st" ]] || continue
        for sf in "$DATA_DIR/$st"/seam_*.npy; do
            [[ -e "$sf" ]] || continue   # 无匹配时 glob 原样返回，-e 过滤掉
            local_idx="$(basename "$sf" .npy)"; local_idx="${local_idx#seam_}"
            JOBS+=("$(printf '%s\t%s' "$st" "$local_idx")")
        done
    fi
done

# 排序保证 worker 分片可复现
if [[ ${#JOBS[@]} -gt 0 ]]; then
    mapfile -t JOBS < <(printf '%s\n' "${JOBS[@]}" | sort)
fi
total=${#JOBS[@]}
[[ "$total" -eq 0 ]] && { echo "[dispatch] 没有可跑的 (工件,焊缝) 作业" >&2; exit 1; }
if [[ "$LIMIT" -gt 0 && "$total" -gt "$LIMIT" ]]; then
    JOBS=("${JOBS[@]:0:$LIMIT}"); total=$LIMIT
fi

echo "[dispatch] 工件目录 : $SUB_DIR"
echo "[dispatch] 焊缝目录 : $DATA_DIR"
echo "[dispatch] 输出目录 : $OUT"
echo "[dispatch] 状态目录 : $STATE_DIR"
echo "[dispatch] GPU 卡   : $GPUS （N=$N，每进程独占一卡）"
echo "[dispatch] 渲染参数 : MAX_ENVS=$MAX_ENVS SPACING=$SPACING SETTLE_STEPS=$SETTLE_STEPS HEADLESS=$HEADLESS"
echo "[dispatch] 共 $total 个 (工件,焊缝) 作业待渲染"

# ---- 单作业：调 render_seam.sh（它负责哨兵→强杀），rc=0 后 touch .done ----
run_one() {
    local gpu="$1" idx="$2" stem="$3" seam="$4"
    local tag="${stem}__seam_${seam}"
    local done_mark="$STATE_DIR/$tag.done"
    local sentinel="$STATE_DIR/$tag.sentinel"
    local log="$STATE_DIR/$tag.log"

    if [[ -f "$done_mark" ]]; then
        echo "[gpu$gpu][$idx/$total] $tag 已完成，跳过"
        return 0
    fi
    rm -f "$sentinel"
    local t0=$SECONDS
    echo "[gpu$gpu][$idx/$total] $tag 开始（tail -f $log 看实时）"

    # render_seam.sh 内部：python 入独立进程组 → 轮询 RENDER_DONE_FILE 哨兵 → 强杀 → 返回 0
    CUDA_VISIBLE_DEVICES="$gpu" \
    STEM="$stem" SEAM_IDX="$seam" \
    RENDER_DONE_FILE="$sentinel" \
        bash "$RENDER_SH" >"$log" 2>&1
    local rc=$?
    local dt=$((SECONDS - t0))

    if [[ $rc -eq 0 && -f "$sentinel" ]]; then
        touch "$done_mark"
        echo "[gpu$gpu][$idx/$total] $tag 完成 (${dt}s)"
    else
        echo "[gpu$gpu][$idx/$total] $tag 失败 rc=$rc (${dt}s)（详见 $log）" >&2
    fi
}

# ---- 每卡一个 worker：处理下标 ≡ w (mod N) 的作业，串行独占该卡 ----
worker() {
    local w="$1"
    local gpu="${GPU_ARR[$w]}"
    local i
    for ((i = w; i < total; i += N)); do
        IFS=$'\t' read -r stem seam <<< "${JOBS[$i]}"
        run_one "$gpu" "$((i + 1))" "$stem" "$seam"
    done
}

echo "[dispatch] 启动 $N 个 worker（每卡一个，独占）渲染 $total 个作业 ..."
for ((w = 0; w < N; w++)); do
    worker "$w" &
done
wait

# ---- 汇总 ----
done_cnt=0
for j in "${JOBS[@]}"; do
    IFS=$'\t' read -r stem seam <<< "$j"
    [[ -f "$STATE_DIR/${stem}__seam_${seam}.done" ]] && done_cnt=$((done_cnt + 1))
done
fail_cnt=$((total - done_cnt))
echo ""
echo "[dispatch] 全部结束：完成 $done_cnt / 失败 $fail_cnt（共 $total）"
echo "[dispatch] 结果目录 $OUT ；状态/日志目录 $STATE_DIR"
[[ "$fail_cnt" -gt 0 ]] && exit 1
exit 0
