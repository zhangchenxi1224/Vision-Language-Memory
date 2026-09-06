# R11_new 执行接管与起点审计

日期：2026-09-06。用户在本任务明确回复“是”，授权接管代码修改、测试、Git 提交和目标实例训练。

## 执行归属

- 新执行任务：`01a0565e-3fdd-70c0-ab58-58dbdc26b7e1`。
- 原执行任务：`01a04a9e-2388-7841-a35c-3254313844bb`；接管前工具核验为 idle，最近轮次已 completed，未产生新的代码修改或训练证据。已通知其后续只读，避免双重写入。
- 接续工作区：`C:/Users/Expedition/r11-new-sigma-audit-fix-20260905`。
- 接管基线：`398cf72d7d18033f2fceb1a7dd1b7a5ccf5420e7`，分支 `codex/r11-new-bridge-distance-20260905`。
- 保留并完成接管时已有的 trainer、core 和 teacher-matched-init JSON 三项未提交改动；不覆盖旧运行根，不删除旧失败结果。
- 唯一目标实例：`vlm-r3-h200x2-live-20260717`。接管准备时通过 Inspire 执行 `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`，无 GPU 计算进程输出；这只是资源快照，不代表任何实验通过。

## 规范与科学状态

用户指定方案仍为 `D:/2026WorkExperience/VisonLearnableMemory/reports/r11-new-frozen-dreamlite-oracle-training-plan-20260904.md`。已完整阅读，并核验执行工作区的同名文件与其正文一致，仅 CRLF/LF 换行存在差异。

| 阶段 | 接管时已有证据 | 未完成要求 |
| --- | --- | --- |
| Phase 0 / 原 Phase 1A | 工程 8/8 有效，query-level gate 6/8；目标 01、07 失败 | 同一新 solver、同一提交和预检下达到固定 8/8；不能拼接不同协议成功项 |
| Bridge distance / post128 cosine | 已交付的 raw step256 距离门和 Reader 门均失败；secondary schedule 诊断通过 | 仅说明该优化日程改善不足，不能声称科学成功 |
| Teacher-matched initialization | 接管时只有配置与部分实现，没有新 forward 或训练结果 | 实现接线、负例测试、独立聚合、提交、fresh-root GPU 预检与诊断 |
| Phase 2 | 固定名单哈希、64/128 选择公式已锁定 | 批量 bank 构建与全成员 gate 尚未形成完整执行链路；Phase 1A 通过后才激活 |
| Phase 3A / 3B | 目标、信息边界和推进顺序已定义 | 专用训练入口、配置、控制器、测试与真实运行；不得用旧 R11/R12 代替 |

## 本轮最小诊断

只改变 x_T 初始化，保留完整冻结四步 DreamLite、固定 teacher、MSE 目标、256 Adam updates、原 post128 cosine 学习率、数据与所有主门。初始化使用实际 scheduler sigma 在 FP32 反解，再按原 sampler compute dtype 和 `mul→add` 顺序重建。M0 仍是该初始化经过完整四步后的未优化 endpoint。

先修复并测试 sigma、teacher std、MSE/NRMSE 元数据的独立绑定，以及每个 checkpoint 当前 x_T 到起始混合状态的重建。该初始化使用已知答案相关 teacher，所有结果均仅为诊断：`formal_success=false`、`phase2_allowed=false`。

## 时间与交付边界

规划时钟不重置，仍从 `2026-09-05T07:18:59.891307+00:00` 起算。原约 30 小时全闭环窗口已不可实现，不能以降门槛、过滤样本或缩至少于 64 条补救。待统一 Phase 1A 8/8 真正通过后，再按固定公式报告 bank 规模及修订 ETA。

接管不是训练成功。每轮仍须分别交付配置/代码提交、原始 receipts、checkpoint、日志、环境、清单哈希、机器可读结果、Markdown 报告和下一项决策。
