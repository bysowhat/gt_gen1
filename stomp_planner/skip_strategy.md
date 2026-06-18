# GT 流水线跳过与完成策略

本文档描述 `main_controller_sp_new.py` 中各阶段的幂等性保障机制。

## 总体思路

流水线分两个阶段（Pose → Traj），每个阶段在启动前先检测是否已完成，避免重复计算。
子进程使用 SIGKILL 终止（因 `python.sh` 包装器需要），通过 `.done` 哨兵文件判断是否成功。

---

## Step 1：输入文件夹过滤

**位置**：`GTGenerator.run()` 扫描 `input_root`

**规则**：
- 跳过：文件夹中没有 `.pkl` 文件 → 打印"跳过空文件夹"
- 跳过：文件夹中没有 `.usd` 文件 → 打印"文件夹不完整"
- 通过：同时包含 `.pkl` 和 `.usd` → 加入待处理队列

**实现要点**：仅检查文件名后缀，找到即 `break`，不加载文件内容，避免 4000+ 文件夹扫描耗时过长。

---

## Step 2：Pose 阶段跳过逻辑

**位置**：`optimize_pose.py` → `OptimizePose.run()`

**规则**：
```python
pose_dir = os.path.join(output_folder, sub_folder_pose)
if os.path.exists(pose_dir):
    print(f"[SKIP] pose 已存在: {output_folder}")
    return
```

- 若 `output/<folder>/pose/` 已存在 → 直接返回，不重跑
- 否则正常执行 ES 优化，结果写入 `pose/`

---

## Step 3：Traj 阶段跳过与清理逻辑

**位置**：`optimize_traj.py` → `OptimizeTraj.run()`

**规则**：
1. 若 `output/<folder>/pose/` **不存在** → `shutil.rmtree(output_folder)` 清除输出文件夹，返回（前置条件缺失，清理残留）
2. 若 `output/<folder>/traj/` **已存在** → 直接返回，跳过
3. 否则正常执行 STOMP 轨迹优化，结果写入 `traj/`

---

## Step 4：哨兵文件 + SIGKILL 完成机制

**背景**：子进程使用 `/workspace/isaaclab/_isaac_sim/python.sh` 启动，该包装器在进程被 SIGKILL 时返回非零退出码并打印错误，无法用退出码判断成功与否。

**方案**：在 `optimize_pose.py` / `optimize_traj.py` 的 `__main__` 末尾：

```python
# 正常完成后写 sentinel 文件
open(os.path.join(args_cli.output_folder, ".done"), "w").close()
# 强制终止进程（AppLauncher 不能 close() 后安全退出）
os.kill(os.getpid(), signal.SIGKILL)
```

---

## Step 5：Controller 中的完成检测

**位置**：`GTGenerator._worker()` 等待子进程结束后：

```python
done_file = os.path.join(self.output_root, folder_name, ".done")
if ret != 0 and not os.path.exists(done_file):
    print(f"[GPU{gpu_id}] !!! {stage} 失败(ret={ret}): {folder_name}")
    continue          # 失败，跳过后续处理
if os.path.exists(done_file):
    os.remove(done_file)   # 清理 sentinel
print(f"[GPU{gpu_id}] <<< 完成 {stage.upper()}: {folder_name}")
```

**判断逻辑**：

| `ret` | `.done` 存在 | 结果 |
|-------|-------------|------|
| 0     | 任意        | 成功 |
| 非零  | 是          | 成功（SIGKILL 预期行为）|
| 非零  | 否          | 失败 |

---

## Step 6：完成后移动文件夹

Traj 阶段成功后，将输入文件夹从 `input_root` 移动到 `done_folder_root`：

```python
src = os.path.join(self.input_root, folder_name)
dst = os.path.join(self.done_folder_root, folder_name)
shutil.move(src, dst)
```

这样下次扫描 `input_root` 时该文件夹不再出现，天然避免重复处理。

---

## 幂等性总结

| 场景 | 行为 |
|------|------|
| Pose 已完成，Traj 未完成 | 跳过 Pose，直接运行 Traj |
| Pose 和 Traj 都已完成 | 两阶段均跳过，文件夹已在 done_folder_root |
| Traj 输出存在但 Pose 缺失 | 清除 output/\<folder\>，重新全流程 |
| 中途崩溃无 .done 文件 | 视为失败，下次重跑该文件夹 |
| 中途崩溃有 .done 文件 | 视为成功，正常推进 |
