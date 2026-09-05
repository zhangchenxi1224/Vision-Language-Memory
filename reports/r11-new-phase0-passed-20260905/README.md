# R11_new Phase 0 技术预检通过

技术预检通过；Phase 1A query-level reachability 尚未评估，科学成功为 false。

- 训练提交：`2cde77ece6f020ab8c747d7c73e19dac4d8fba1b`。
- 根：`/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-new/r11-new-phase1a-2cde77e-20260905-round02/phase0-preflight-target-00`。
- Controller：`technical_completed=true`，全部 26 个 execution_checks 为 true，child exit code 0。
- 起止 UTC：`2026-09-05T07:35:17.899223+00:00` 至 `2026-09-05T07:36:05.963814+00:00`。
- backward 1 次，optimizer steps 0；完整四步冻结 DreamLite，五点 trajectory，仅 FP32 x_T 可训练。
- Reader CE：14.354171752929688；梯度 L2：2.1787609734177065；非零梯度元素比例：0.999603271484375。
- 模型全冻结、snapshot 未变、condition 重算与所有 checkpoint 校验通过。

## 纠错验证

首轮因 sigma 浮点实测值与名义值的精确等值检查不一致而技术失败。修复复用了首轮前向即有的数值容差。此次与首轮的 loss、梯度范数、非零梯度比例、sigma 实测列表精确相同；x_T、endpoint 和全部五个 trajectory 张量的 canonical SHA-256 也全部精确相同。图片 SHA-256 均为 `aa3aa73276d70dec0b1fdc257b6175142e8e53ce4a4c6ad9e4605efae105d3a9`。

这说明校验修复未改变实际模型计算。两轮终态均保留，未改写失败结果。

## 交付与下一步

`phase0-passed-original.tar.gz` 保存完整控制器/trainer manifest、两层 inventory、checkpoint、condition、图片、环境、原始日志、阶段报告与终态；额外保存 program-clock 和修复前后 parity audit。

- 归档 bytes：2141509。
- 归档 SHA-256：`f506396eef903976da9ab38d15a1749bfeb3ddbccd5b7079f80d44137d8f58fd`。
- 两层 inventory 共 33 条文件哈希已在远端独立复算通过。
- 独立审计摘要：`PREFLIGHT_AUDIT.json`。

下一步只授权进入既定 Phase 1A target 0 的正式 256 步诊断；target 0 技术有效后才继续其余七个。必须八个目标全部通过原有 gate 才进入 Phase 2。正式训练 source commit 继续锁定为 `2cde77e`；报告提交不改变训练快照。

30 小时规划起点保持首次预检 `2026-09-05T07:18:59.891307+00:00`，包括工程失败、诊断、修复和重跑。
