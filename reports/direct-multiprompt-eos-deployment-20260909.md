# Direct 三问法轮换训练：启智部署

北京时间 2026-09-09 00:01:21，在 `vlm-r11-trust-h200x4-20260907-r3` 实际启动两条训练 lane，GPU 0/1 和 2/3 各一条。部署前已确认旧 Direct 和两次 U-Net 训练完成、四卡无计算进程。

- 训练源码：`46cd36b1eb421471dafa7865fbab4dfe02336336`。
- GitHub 分支：`codex/direct-multiprompt-eos-20260908`。
- 远端独立源码：`P/repos/direct-multiprompt-eos-20260908`。
- 远端结果：`P/runs/direct-multiprompt-eos/46cd36b-20260909-r01`。
- supervisor PID：686405；lane PID：686476、686477。
- 配置 SHA256：`4f47fdbd2a43cf31b6b0c264cd2223a8790b7dbb784e8f1d98b240a6608760f1d`。
- 任务预算到北京时间 04:01:21，原 launcher 在完整轨迹边界预留至少15分钟暂停；这不是平台到期时间，也未修改平台自动停止设置。

## 实际验证

16 项本地 CPU 检查通过。测试真实执行 256 次优化循环，以小型可微 Reader 替代真实大模型，验证86/85/85次轮换、保留问法禁入梯度，以及旧协议产物或错误步级问法记录被拒绝。

00:02:57 的真实 GPU 检查：Gaussian seed0 到179步，Uniform seed0 到177步；两者均通过初始 loss/gradient 逐位重复性检查，初始 latent SHA 与原基线对应 run 一致。前6次更新实际依次使用 q0/q1/q2/q0/q1/q2。Gaussian 初始 loss `13.129262924194336`，梯度 SHA `c7418ef0cfd6ce5ee12324b0a99cd555b333ae82a5e457678e687bc53c3d395b`，与原基线相同。

00:03:56 已核实首两条均完整结束，各保存257个latent、11个checkpoint和15条终点评测：

| 起点 | 256步耗时 | 三个训练问法 | 两个保留问法 |
|---|---:|---:|---:|
| Gaussian seed0 scale1 | 73.28秒 | 3/3 | 2/2 |
| Uniform seed0 scale1 | 74.38秒 | 3/3 | 2/2 |

两条在五种问法下均原始输出 `ambient`。原基线 Gaussian seed0 在q4输出 `ambient synthwave`，本次该同起点案例修复；此局部结果不代表96条全量改善已成立。下一对 Sphere/Rademacher seed0 已分别到124/119步，四条已启动run的初始latent SHA均与各自基线一致。

真实GPU与逐run验证快照保存在输出根目录 `deployment_verification.json`。这是部署验证，不是96条最终科学结果。完整结果以新目录的逐run `terminal.json`、`generations.jsonl` 和最终 `summary.json` 为准。

## 固定协议

96个起点与原基线逐项相同；原模型、数据、初始化、答案/EOS损失和生成方式保留。每次256步，仅按 q0/q1/q2 轮换训练，q3/q4只在固定终点读出。详细约束见 [实验协议](direct-multiprompt-eos-protocol-20260908.md)。

保存所有257个 latent、11个原问法checkpoint评测和15个终点评测。汇总分别报告原问法、三个训练问法、两个保留问法及五问法整体表现。本轮独立结果不会覆盖旧96条或修改旧U-Net输入库/自动接续进程。

`P = /inspire/ssd/project/exploration-topic/czxs26210936`。
