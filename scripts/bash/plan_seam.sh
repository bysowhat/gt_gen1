#!/bin/bash
set -u

# 新数据结构：ROOT 下有多个工件文件夹，每个文件夹内含 1 个 .obj 和多个 seam_*.pkl。
# plan_seam.py 会在 seam 同目录里自动找 obj，所以 seam 必须就地传入（不要搬动）。
# 输出统一存进项目的 tmp，用「工件名__seam名」命名，保证不同文件夹的同名 seam 不会重名。
ROOT="/media/a/新加卷/hanfeng/1/segment_output_sub"
OUT_DIR="tmp/plan_seam"
MAX_ATTEMPTS=200

mkdir -p "$OUT_DIR"

for part_dir in "$ROOT"/*/; do
    part=$(basename "$part_dir")
    for seam in "$part_dir"*.pkl; do
        [ -e "$seam" ] || continue   # 该文件夹没有 pkl 时跳过
        sname=$(basename "$seam" .pkl)
        out="$OUT_DIR/${part}__${sname}.npz"
        if [ -f "$out" ]; then
            echo "=== Skipping ${part}__${sname} (already exists) ==="
            continue
        fi
        echo "=== Processing ${part}__${sname} ==="
        python scripts/plan_seam.py --seam "$seam" --max_attempts "$MAX_ATTEMPTS" --out "$out"
    done
done
