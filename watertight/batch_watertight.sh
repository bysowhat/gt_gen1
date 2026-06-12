#!/usr/bin/env bash
# 批量把 *_part.obj 转为 *_part_watertight.obj（并行调用 make_watertight.py）。
#
# 用法：
#   bash watertight/batch_watertight.sh                          # 默认目录、10 路并行
#   bash watertight/batch_watertight.sh <根目录>                 # 指定根目录
#   bash watertight/batch_watertight.sh <根目录> <并行数>        # 指定并行数
#   FORCE=1 bash watertight/batch_watertight.sh ...              # 覆盖已存在的 _watertight.obj
#   PY=/isaac-sim/python.sh bash watertight/batch_watertight.sh ...  # 服务器（无 conda）用 Isaac Sim python
#
# 说明：
#   - 递归找根目录下所有 *_part.obj（已生成的 *_part_watertight.obj 不匹配，天然排除）。
#   - 用 xargs -0 -P 多进程，每进程一次处理 CHUNK 个文件（摊薄 python/trimesh 导入开销）。
#   - make_watertight.py 默认跳过已存在的 _watertight.obj，可安全中断后重跑续传；FORCE=1 才覆盖。
#   - 解释器由 PY 指定：本地默认 `conda run -n env_isaaclab python`；服务器无 conda 时传
#     PY=/isaac-sim/python.sh（Isaac Sim 自带 python，需含 trimesh + manifold3d）。
set -euo pipefail

ROOT="${1:-/media/a/新加卷/hanfeng/1/segment_output_sub}"
JOBS="${2:-10}"
CHUNK="${CHUNK:-20}"
ENV="${ENV:-env_isaaclab}"
PY="${PY:-conda run --no-capture-output -n ${ENV} python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MW="$SCRIPT_DIR/make_watertight.py"

FORCE_FLAG=""
[ "${FORCE:-0}" = "1" ] && FORCE_FLAG="--force"

[ -d "$ROOT" ] || { echo "目录不存在: $ROOT" >&2; exit 1; }
[ -f "$MW" ]   || { echo "找不到 make_watertight.py: $MW" >&2; exit 1; }

TOTAL=$(find "$ROOT" -type f -name '*_part.obj' | wc -l)
echo "根目录 : $ROOT"
echo "待处理 : $TOTAL 个 *_part.obj"
echo "并行数 : $JOBS   每批: $CHUNK   覆盖: ${FORCE:-0}"
echo "解释器 : $PY"
echo "开始时间: $(date '+%F %T')"
echo

# 每个 worker 收到 CHUNK 个文件，单进程内导入一次 trimesh/manifold 后逐个处理。
find "$ROOT" -type f -name '*_part.obj' -print0 \
  | xargs -0 -P "$JOBS" -n "$CHUNK" \
      $PY "$MW" $FORCE_FLAG

echo
DONE=$(find "$ROOT" -type f -name '*_part_watertight.obj' | wc -l)
echo "结束时间: $(date '+%F %T')"
echo "已生成 : $DONE / $TOTAL 个 *_part_watertight.obj"
