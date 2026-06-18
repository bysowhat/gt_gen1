import os
import shutil

A_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "A_folders.txt")
B = "/DATA/baiyu/20251125/input"

# 从文件读取A的文件夹名称
with open(A_file, "r") as f:
    A_folders = set([line.strip() for line in f.readlines() if line.strip()])

# 获取 B 下所有文件夹名称
B_folders = set([d for d in os.listdir(B) if os.path.isdir(os.path.join(B, d))])

# 根据 A_folders 判断是否删除 B 下的文件夹
for folder in B_folders:
    if folder not in A_folders:
        path = os.path.join(B, folder)
        print("删除：", path)
        # ⚠️ 生产环境建议测试完后再取消注释
        # shutil.rmtree(path)
    # else:
    #     print("保留：", folder)
