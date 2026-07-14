"""plan_joint_single 的输入落盘 / 复现工具。

用途：把 main_loop.py 里
    seg = si.plan_joint_single(cfg, _w1, cur_cfg, goal_cfg, checker_type=_ck1)
这一次调用的【全部输入】存成一个 curobo 无关的 .pkl，之后在任意进程（有 curobo/STOMP 的机器）
读回来，一模一样地再跑一次，用于单独 debug 这条规划。

保存内容（全是 numpy / python 基础类型，不 pickle curobo 对象，跨版本稳）：
  - stomp_params(dict) 与 robot_cfg_path      —— plan_joint_single 真正用到的 cfg 字段
  - world 拆成 meshes / cuboids 的顶点面/尺寸位姿 —— WorldConfig 的可移植表示
  - cur_cfg / target_cfg                       —— 起点/目标关节角
  - checker_type 的名字（"MESH"/"PRIMITIVE"）

用法
----
① 落盘（在 main_loop.py:813 那句之前插一行）：
    from gt_gen.repro_plan_joint import dump_inputs
    dump_inputs("plan_joint_case.pkl", cfg, _w1, cur_cfg, goal_cfg, _ck1)

② 复现（另起进程）：
    python -m gt_gen.repro_plan_joint plan_joint_case.pkl
  或代码里：
    from gt_gen.repro_plan_joint import replay
    seg = replay("plan_joint_case.pkl")
"""
from __future__ import annotations

import pickle
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np


def _checker_name(checker_type: Any) -> Optional[str]:
    """CollisionCheckerType 枚举 -> 名字("MESH"/"PRIMITIVE"...)；None 原样返回。"""
    if checker_type is None:
        return None
    return getattr(checker_type, "name", str(checker_type))


def _world_to_plain(world: Any) -> dict:
    """curobo WorldConfig -> {'meshes':[...], 'cuboids':[...]}（只取几何，不含 tensor）。"""
    meshes = []
    for m in (getattr(world, "mesh", None) or []):
        meshes.append({
            "name": getattr(m, "name", None),
            "vertices": np.asarray(m.vertices, float).tolist(),
            "faces": np.asarray(m.faces, np.int64).tolist(),
            "pose": list(getattr(m, "pose", [0, 0, 0, 1, 0, 0, 0])),
        })
    cuboids = []
    for c in (getattr(world, "cuboid", None) or []):
        cuboids.append({
            "name": getattr(c, "name", None),
            "dims": list(c.dims),
            "pose": list(c.pose),
        })
    return {"meshes": meshes, "cuboids": cuboids}


def dump_inputs(path: str, cfg, world, cur_cfg, target_cfg, checker_type) -> str:
    """把 plan_joint_single 的一次调用输入存到 path（.pkl）。返回 path。"""
    payload = {
        "stomp_params": dict(cfg.stomp_params),
        "robot_cfg_path": cfg.robot_cfg_path,
        "world": _world_to_plain(world),
        "cur_cfg": np.asarray(cur_cfg, float).tolist(),
        "target_cfg": np.asarray(target_cfg, float).tolist(),
        "checker_type": _checker_name(checker_type),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    w = payload["world"]
    print(f"[repro] 已保存 -> {path}  (meshes={len(w['meshes'])} cuboids={len(w['cuboids'])} "
          f"checker={payload['checker_type']})")
    return path


def _plain_to_world(w: dict):
    """{'meshes','cuboids'} -> curobo WorldConfig（在有 curobo 的进程里调用）。"""
    import gt_gen.compat  # noqa: F401  trimesh shim（mesh 版需要）
    gt_gen.compat.apply_trimesh_shim()
    from curobo.geom.types import WorldConfig, Mesh, Cuboid
    meshes = [Mesh(name=m["name"], vertices=m["vertices"], faces=m["faces"], pose=m["pose"])
              for m in w.get("meshes", [])]
    cuboids = [Cuboid(name=c["name"], dims=c["dims"], pose=c["pose"])
               for c in w.get("cuboids", [])]
    return WorldConfig(mesh=meshes, cuboid=cuboids)


def _checker_from_name(name: Optional[str]):
    """名字 -> CollisionCheckerType 枚举；None 原样返回（让 API 自动推断）。"""
    if name is None:
        return None
    from curobo.geom.sdf.world import CollisionCheckerType
    return CollisionCheckerType[name]


def replay(path: str) -> Optional[np.ndarray]:
    """读回 path，重建 world/cfg 并再跑一次 plan_joint_single。返回 (T,dof) ndarray 或 None。"""
    with open(path, "rb") as f:
        payload = pickle.load(f)

    # 轻量 cfg：plan_joint_single 只用 .stomp_params 与 .robot_cfg_path
    cfg = SimpleNamespace(stomp_params=payload["stomp_params"],
                          robot_cfg_path=payload["robot_cfg_path"])
    world = _plain_to_world(payload["world"])
    checker = _checker_from_name(payload["checker_type"])

    from gt_gen import stomp_iface as si
    seg = si.plan_joint_single(cfg, world, payload["cur_cfg"], payload["target_cfg"],
                               checker_type=checker)
    print(f"[repro] plan_joint_single -> {'None' if seg is None else f'ok shape={seg.shape}'}")
    return seg


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python -m gt_gen.repro_plan_joint <case.pkl>")
        raise SystemExit(2)
    replay(sys.argv[1])
