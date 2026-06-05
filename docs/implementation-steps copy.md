# GT 生成实现步骤（逐步追踪）

- [✅已完成] Step 0  — 决策锁定 + 环境 + 代码骨架
- [✅已完成] Step 1  — cuRobo 基础封装（IK + 规划，空世界）
- [✅已完成] Step 2  — 三态体素地图数据结构
- [✅已完成] Step 3  — 传感器模拟（raycast 真值场景）
- [✅已完成] Step 4  — 观测更新 `observe_and_update`
- [✅已完成] Step 5  — voxmap → cuRobo 碰撞世界同步（未知=障碍）
- [✅已完成] Step 6  — 整臂扫掠体积 + `motion_stays_in_free`
- [✅已完成] Step 7  — `reach_pt` + 阻塞段 `B` 计算
- [⬜未完成] Step 8  — 候选视点生成
- [⬜未完成] Step 9  — 特权 NBV 打分与选择
- [⬜未完成] Step 10 — 主循环编排（①~⑦）+ 冷启动 + 卡住处理
- [⬜未完成] Step 11 — GT 导出
- [⬜未完成] Step 12 — 批量产 GT + 鲁棒性
