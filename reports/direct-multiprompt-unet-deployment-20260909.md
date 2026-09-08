# 多问法 Direct 自动接续 U-Net：2026-09-09 实际部署

北京时间 00:28:56 现场验证：两个接续进程存活，均为 `waiting_for_oracle`；新一轮 Direct 已完成30/96，U-Net 本轮尚未开始更新。

## 执行位置与输入

- Oracle：`dl-base-h200x4-20260907`，源码固定 `46cd36b1eb421471dafa7865fbab4dfe02336336`。
- Oracle 输出：`P/runs/direct-multiprompt-eos/46cd36b-20260909-r01`。
- U-Net：`vlm-r11-trust-h200x4-20260907-r3`，容器 hostname `vlm-r11-trust-h200x4-20260907-r3--1529e224dfc5-wj6s2jn4v3`。
- 接续及训练源码固定 `f5c9c9ef1604f6b2d0466e73a80acab692cb763a`，独立 checkout 为 `P/repos/latent-bank-unet-multiprompt-20260909`；没有修改仍运行的 oracle 或旧 Frozen 接续源码。

这里的输入仍为同一道 ambient 题的96个不同起点。原问及paraphrase_1、paraphrase_2以86/85/85次轮换训练；paraphrase_3、paraphrase_4仅评测。本次新增审计验证256条真实训练问法记录、manifest和评测行标记，不把多问法结果误记为单问法结果。

## 两个已启动进程

| seed | PID | GPU组 | 独立输出 |
|---|---:|---|---|
| 20260908 | 867115 | 0,1 | `P/runs/oracle-to-unet/f5c9c9e-direct-multiprompt-trust-seed20260908` |
| 20260909 | 867116 | 2,3 | `P/runs/oracle-to-unet/f5c9c9e-direct-multiprompt-trust-seed20260909` |

每个目录已有 config、dispatch、status、resource-lease 和 sidecar.log。两个进程在CPU上等待，当前trust四卡各1MiB、利用率0%；这与尚未开始U-Net训练一致。

96条轨迹及campaign/lane完成记录全部齐全后，自动校验原始产物、分别封存成功latent集合、分析几何，再分别训练U-Net LoRA各512步并完成训练前后的新噪声答题评测。仍以原问step256 raw exact match决定入库，不使用留出问法筛选目标，不复用第一轮单问法teacher或adapter。

原有资源执行上限保持北京时间09-09 10:38:18；这是任务执行预算，不是声称的平台自动停止时间。若oracle失败、产物不全、无正确终点或GPU被其它任务占用，接续会明确报错或等待，不把这些状态算作训练成功。

## 已完成验证

- 本地相关回归：33 passed。
- trust实例上CPU读取新oracle前两条完整真实轨迹，验证257点、11个checkpoint、EOS、模型/数据/源码绑定、15格原始评测，以及三训练/两留出问法记录，全部通过。此预检没有执行优化更新、没有以两个样本生成正式bank。
- `P/runs/oracle-to-unet/f5c9c9e-direct-multiprompt-preflight/preflight.json` 的 SHA256：`ee66470af58a186085e4ec70e2723b63a316eaf0febb9c71ea5ed90c2611ab6f`。
- 相同目录的 `ready.json` 保存两套配置与资源绑定SHA，`dispatch_verification.json` 保存两个PID存活、30/96进度、U-Net尚未训练的现场证据。

第一轮单问法U-Net的两个seed均已经完成512步，但新生成图片原问均为0/8。此次部署执行成功不代表科学结果成功；新一轮结果必须以各自的unet/result.json与原始generation记录为准。

`P = /inspire/ssd/project/exploration-topic/czxs26210936`。
