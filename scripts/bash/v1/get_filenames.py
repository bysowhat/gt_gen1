import os
import random

SEGMENT_ROOT = '/kpfs_dataset/dataset/baiyu/dataset_v2/debug/segment'
SEED = 42
MAX_PER_PROJECT = 50


def build_json_to_project(segment_root):
    """建立 {weld_angle3.json 文件名: 项目名} 的反向索引。"""
    json2proj = {}
    for project in os.listdir(segment_root):
        proj_dir = os.path.join(segment_root, project)
        if not os.path.isdir(proj_dir):
            continue
        for fn in os.listdir(proj_dir):
            if fn.endswith('_weld_angle3.json'):
                json2proj[fn] = project
    return json2proj


def main(root):
    folders = os.listdir(root)
    txt_fp = '/kpfs_dataset_ssd/dataset/render_baiyu/filenames.txt'

    json2proj = build_json_to_project(SEGMENT_ROOT)

    # 按项目分组收集通过校验的 folder
    proj2items = {}
    unmatched = []

    for folder in folders:
        obj_fp = os.path.join(root, folder, folder + '_watertight.obj')
        json_name = folder.replace('_part', '_weld_angle3.json')
        json_fp = os.path.join(root, folder, json_name)
        if not (os.path.exists(obj_fp) and os.path.exists(json_fp)):
            continue

        project = json2proj.get(json_name)
        if project is None:
            unmatched.append(folder)
            continue
        proj2items.setdefault(project, []).append((obj_fp, json_fp))

    if unmatched:
        print('[WARN] %d 个 folder 未匹配到项目:' % len(unmatched))
        for f in unmatched:
            print('  ', f)

    # 每个项目内部用固定 seed 打乱顺序，并截断到最多 MAX_PER_PROJECT 个
    rng = random.Random(SEED)
    projects = sorted(proj2items.keys())
    for project in projects:
        rng.shuffle(proj2items[project])
        proj2items[project] = proj2items[project][:MAX_PER_PROJECT]

    # 按项目名排序轮转写入：第一轮各项目取第 1 个，第二轮取第 2 个……
    # 直到所有（截断后）符合条件的 (obj_fp, json_fp) 都被写入
    total = 0
    max_len = max((len(v) for v in proj2items.values()), default=0)
    with open(txt_fp, 'w') as f:
        for i in range(max_len):
            for project in projects:
                items = proj2items[project]
                if i < len(items):
                    obj_fp, json_fp = items[i]
                    f.write('%s\t%s\n' % (obj_fp, json_fp))
                    total += 1

    print('共 %d 个项目，%d 个工件（每项目最多 %d），已写入 %s'
          % (len(projects), total, MAX_PER_PROJECT, txt_fp))


if __name__ == '__main__':
    root = '/kpfs_dataset_ssd/dataset/render_kejian/segment_output_sub'
    main(root)
