#!/bin/bash
set -u

SEAM_DIR="/media/a/新加卷/hanfeng/segment_sub_output/BEAM_1aEEYa00Ed5Z4sE34qDJKu_part2"
OUT_DIR="tmp/plan_seam"
MAX_ATTEMPTS=200

mkdir -p "$OUT_DIR"

for seam in "$SEAM_DIR"/*.pkl; do
    name=$(basename "$seam" .pkl)
    out="$OUT_DIR/$name.npz"
    if [ -f "$out" ]; then
        echo "=== Skipping $name (already exists) ==="
        continue
    fi
    echo "=== Processing $name ==="
    python scripts/plan_seam.py --seam "$seam" --max_attempts "$MAX_ATTEMPTS" --out "$out"
done
