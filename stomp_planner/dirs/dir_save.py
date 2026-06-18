import os

A = "/DATA/baiyu/20251125/input"
save_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "A_folders.txt")

# 获取 A 下所有文件夹名称
A_folders = [d for d in os.listdir(A) if os.path.isdir(os.path.join(A, d))]

# 保存到本地
with open(save_file, "w") as f:
    for folder in A_folders:
        f.write(folder + "\n")

print(f"已将 A 下的 {len(A_folders)} 个文件夹名称保存到：{save_file}")