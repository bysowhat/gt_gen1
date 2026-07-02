#!/usr/bin/env bash
# 批量「平放工件 + 拍照」：对 segment_output_subs 下所有工件跑 scripts/viz_lay_flat_batch.py，
# 每个工件出 2 张图（flat 摆平 / orig 未旋转对照），肉眼核对平放是否正确。
#
# 数据布局（远程，/kpfs_dataset_ssd 在远程机器本地、ssh debug 可达）：
#   · 工件 mesh : $SUB_DIR/<stem>_part/<stem>_part_watertight.obj
#   · 焊缝 json : $SUB_DIR/<stem>_part/<stem>_weld_angle3.json（与 obj 同目录同前缀）
#     lay_flat 会【自动】在 obj 同目录按此命名找 json：
#       - 最长焊缝 > 工件最长轴一半 → 让最长那批焊缝落在同一水平面来选滚转；
#       - 否则（含无 json）→ 退回「落地质心最低」。
#   注：viz_lay_flat_batch.py 自己递归遍历 $SUB_DIR 下的所有 *_part_watertight.obj，
#       本脚本只需单进程调用一次（离屏渲染走 open3d OffscreenRenderer/EGL，无需显示器）。
#
# 输出：
#   · flat 图 : $OUT_DIR/flat/<sub>__<stem>_flat.png
#   · orig 图 : $OUT_DIR/orig/<sub>__<stem>_orig.png
#   · 运行日志: $OUT_DIR/run.log（python 的 stdout+stderr）
#
# 用法（在远程跑；本机无 /kpfs 挂载）：
#   ssh debug 'bash /kpfs_dataset/dataset/baiyu/code/render/gt_gen_hanfeng/scripts/bash/kejian/run_viz_lay_flat_batch.sh'
#   # 覆盖参数示例：
#   SUB_DIR=/path/to/part_1 OUT_DIR=/tmp/lf_1 bash .../run_viz_lay_flat_batch.sh
#   ONLY_FLAT=1 bash .../run_viz_lay_flat_batch.sh          # 只出 flat 图，不出 orig 对照
#   AZIM=45 ELEV=30 DIST_FACTOR=2.2 bash .../run_viz_lay_flat_batch.sh

set -uo pipefail

# ---- 路径/参数（环境变量可覆盖）----
SUB_DIR="${SUB_DIR:-/kpfs_dataset_ssd/dataset/render_kejian/segment_output_subs/part_0}"
OUT_DIR="${OUT_DIR:-/kpfs_dataset_ssd/dataset/tempt/1/part_0}"
PY="${PY:-/workspace/isaaclab/_isaac_sim/python.sh}"    # 远程 isaac python（含 numpy/trimesh/open3d）
AZIM="${AZIM:-45}"                # 相机方位角(度)
ELEV="${ELEV:-30}"                # 相机仰角(度)
DIST_FACTOR="${DIST_FACTOR:-2.0}" # 相机距离系数
WIDTH="${WIDTH:-1280}"            # 图片宽(px)
HEIGHT="${HEIGHT:-960}"           # 图片高(px)
SEAM_RATIO="${SEAM_RATIO:-0.9}"   # 「差不多长」阈值：长度 ≥ ratio·最长焊缝 的那批参与定平面
ONLY_FLAT="${ONLY_FLAT:-0}"       # 1=只出 flat 图，不出 orig 对照

# 仓库根：本脚本在 scripts/bash/kejian/ 下
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SCRIPT="$REPO/scripts/viz_lay_flat_batch.py"

# ---- 校验 ----
[[ -x "$PY" ]]      || { echo "[run] python 不可执行: $PY" >&2; exit 1; }
[[ -f "$SCRIPT" ]]  || { echo "[run] 找不到 viz_lay_flat_batch.py: $SCRIPT" >&2; exit 1; }
[[ -d "$SUB_DIR" ]] || { echo "[run] 工件目录不存在: $SUB_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR"

echo "[run] 工件目录 : $SUB_DIR"
echo "[run] 输出目录 : $OUT_DIR   （flat/ 与 orig/ 两子目录）"
echo "[run] python   : $PY"
echo "[run] 脚本     : $SCRIPT"
echo "[run] 相机     : azim=$AZIM elev=$ELEV dist_factor=$DIST_FACTOR  图=${WIDTH}x${HEIGHT}"
echo "[run] seam_ratio: $SEAM_RATIO   only_flat: $ONLY_FLAT"

# ---- 组装参数并执行（单进程，脚本内部自遍历所有 obj）----
extra=()
[[ "$ONLY_FLAT" == "1" ]] && extra+=(--only-flat)

log="$OUT_DIR/run.log"
echo "[run] 开始渲染，实时日志： tail -f $log"
t0=$SECONDS
PYTHONUNBUFFERED=1 "$PY" -u "$SCRIPT" "$SUB_DIR" \
    --out-dir "$OUT_DIR" \
    --azim "$AZIM" --elev "$ELEV" --dist-factor "$DIST_FACTOR" \
    --width "$WIDTH" --height "$HEIGHT" \
    --seam-ratio "$SEAM_RATIO" "${extra[@]}" >"$log" 2>&1
rc=$?
dt=$((SECONDS - t0))

if [[ $rc -eq 0 ]]; then
    n_flat=$(find "$OUT_DIR/flat" -name '*_flat.png' 2>/dev/null | wc -l)
    echo "[run] 完成 (${dt}s) rc=0  flat 图 $n_flat 张 -> $OUT_DIR/flat"
else
    echo "[run] 失败 rc=$rc (${dt}s)（详见 $log）" >&2
fi
exit $rc
