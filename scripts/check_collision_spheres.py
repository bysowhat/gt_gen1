"""确认 yml 里哪些碰撞球与工件相交——起点(retract)和目标位姿各查一次。

按 robot yml 的 collision_spheres（每个 link 命名的球），用 FK 把每颗球变换到基座系，
和工件 mesh 求带符号距离，球心到表面距 < 半径即判相交，逐颗报告(link + 第几颗 + 穿透深度)。

运行：
    conda run -n env_isaaclab python scripts/check_collision_spheres.py \
        --seam /media/a/新加卷/.../seam_74.pkl
    # 目标构型默认用 pkl 的 joint_angles；也可显式给：--joints "j1,...,j6"
"""
import argparse
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

DEFAULT_SEAM = ("/media/a/新加卷/hanfeng/segment_sub_output/"
                "BEAM_1aEEYa00Ed5Z4sE34qDJKu_part/seam_74.pkl")


def quat_wxyz_to_R(q):
    w, x, y, z = [float(v) for v in q]
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", default=DEFAULT_SEAM)
    ap.add_argument("--joints", default=None, help="目标关节角(弧度,逗号分隔);默认用 pkl joint_angles")
    args = ap.parse_args()

    from gt_gen import compat  # noqa: F401  warp shim
    from gt_gen.config import load_config
    import torch
    import trimesh
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.util_file import load_yaml
    from plan_seam import find_obj

    cfg = load_config()
    d = pickle.load(open(args.seam, "rb"))
    rp = np.asarray(d["robot_pose"][0], float)

    # 工件 mesh -> 基座系（减 robot 平移；旋转一般还要按 robot 四元数，这里 identity）
    obj_path = find_obj(args.seam)
    tm = trimesh.load(obj_path, force="mesh")
    tm.apply_translation(-rp[:3])
    pq = trimesh.proximity.ProximityQuery(tm)
    print("seam :", args.seam)
    print("obj  :", obj_path, " 顶点:", len(tm.vertices))

    # 注册所有碰撞 link 做 FK
    rd = load_yaml(cfg.robot_cfg_path)
    kin = rd["robot_cfg"]["kinematics"]
    coll_links = list(kin["collision_link_names"])
    spheres_def = kin["collision_spheres"]
    kin["link_names"] = coll_links
    ta = TensorDeviceType()
    rc = RobotConfig.from_dict(rd["robot_cfg"], ta)
    model = CudaRobotModel(rc.kinematics)

    retract = np.array(cfg.retract_config, float)
    target = (np.array([float(x) for x in args.joints.split(",")], float)
              if args.joints else np.asarray(d["joint_angles"], float))

    def link_poses(q):
        st = model.get_state(torch.tensor([q], dtype=torch.float32, device=ta.device))
        return {ln: (st.link_pose[ln].position[0].detach().cpu().numpy(),
                     st.link_pose[ln].quaternion[0].detach().cpu().numpy())
                for ln in coll_links}

    def check(name, q):
        lp = link_poses(q)
        hits = []
        n_total = 0
        for ln in coll_links:
            if ln not in spheres_def:
                continue
            pos, quat = lp[ln]
            R = quat_wxyz_to_R(quat)
            centers, radii = [], []
            for s in spheres_def[ln]:
                r = float(s["radius"])
                if r <= 1e-4:
                    continue
                centers.append(R @ np.asarray(s["center"], float) + pos)
                radii.append(r)
            if not centers:
                continue
            centers = np.array(centers); radii = np.array(radii)
            n_total += len(centers)
            sd = pq.signed_distance(centers)        # >0 在网格内部
            pen = sd + radii                         # >0 表示相交(近似穿透深度)
            for i in np.where(pen > 0)[0]:
                hits.append((ln, int(i), float(pen[i]), float(radii[i]), centers[i]))
        hits.sort(key=lambda x: -x[2])
        print(f"\n[{name}] 总球数={n_total} 碰撞球={len(hits)}")
        # 按 link 汇总
        from collections import Counter
        c = Counter(h[0] for h in hits)
        for ln, k in c.items():
            print(f"    {ln}: {k} 颗碰撞")
        # 明细 top 10
        for ln, i, pen, r, ctr in hits[:10]:
            print(f"      {ln}#{i}  穿透={pen:.4f}m r={r:.3f} 球心={np.round(ctr,3)}")
        return hits

    check("起点 retract", retract)
    check("目标 joint_angles", target)


if __name__ == "__main__":
    main()
