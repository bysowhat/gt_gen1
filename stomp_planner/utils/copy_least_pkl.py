"""
扫描 source_dir 下各子文件夹的 pkl 文件数量，
取数量最少的 48 个子文件夹，整体复制到 target_dir。
"""

import os
import shutil

source_dir = "/kpfs_dataset_ssd/render/segment_output/"
target_dir = "/kpfs_dataset/dataset/simulation/gt/test_data/input/"  # 按需修改


def count_pkl(folder: str) -> int:
    return sum(1 for f in os.listdir(folder) if f.endswith(".pkl"))


def main():
    entries = [
        e for e in os.scandir(source_dir)
        if e.is_dir()
    ]
    if not entries:
        print(f"[错误] {source_dir} 下没有子文件夹")
        return

    counts = []
    for e in entries:
        n = count_pkl(e.path)
        if n < 5:
            print(f"  [跳过] {e.name}: {n} pkl")
            continue
        counts.append((n, e.name, e.path))
        print(f"  {e.name}: {n} pkl")

    counts.sort(key=lambda x: x[0])
    least48 = counts[:48]

    print(f"\n共 {len(counts)} 个子文件夹，pkl 最少的 48 个：")
    for n, name, _ in least48:
        print(f"  {name}: {n} pkl")

    os.makedirs(target_dir, exist_ok=True)
    for n, name, src_path in least48:
        dst_path = os.path.join(target_dir, name)
        if os.path.exists(dst_path):
            print(f"[跳过] {name} 已存在于目标目录")
            continue
        print(f"复制 {name} ({n} pkl) -> {dst_path}")
        shutil.copytree(src_path, dst_path)

    print(f"\n完成，共复制 {len(least48)} 个文件夹到 {target_dir}")


if __name__ == "__main__":
    main()
