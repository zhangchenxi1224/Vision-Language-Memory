# R11_new Phase 0 首轮失败与最小工程诊断

本轮技术预检未通过；query-level reachability 未评估，科学成功为 false。正式 256 步训练尚未启动。

## 固定证据

- 训练提交：`a6cba6d508ecd864ba07d391733a546d2937f980`。
- 实例：`vlm-r3-h200x2-live-20260717`，物理节点 `qb-prod-gpu1889`。
- 根：`/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-new/r11-new-phase1a-a6cba6d-20260905/phase0-preflight-target-00`。
- Controller 起止：`2026-09-05T07:18:59.891307+00:00` 至 `2026-09-05T07:20:07.801151+00:00`。
- child exit code：1；technical_completed：false；互斥锁正常释放。
- 一次完整 forward/backward，optimizer steps：0，checkpoint：step 0。

| 原始观察值 | 数值 |
| --- | ---: |
| Reader CE（仅用于技术预检） | 14.354171752929688 |
| x_T 梯度 L2 范数 | 2.1787609734177065 |
| x_T 非零梯度元素比例 | 0.999603271484375 |
| trajectory 点数 / denoiser 步数 | 5 / 4 |
| 所有模型冻结 / snapshot 未变 | true / true |
| 唯一 trainable | x_T_fp32 |

## 从原始失败到原因

名义 sigma 是 `[0.5, 0.375, 0.25, 0.125]`。scheduler 实测并落盘的值为 `[0.4999999701976776, 0.375, 0.25, 0.1249999925494194]`，最大绝对偏差 `2.9802322387695312e-08`。

首次运行前的 sampler 使用 `torch.allclose(rtol=2e-6, atol=2e-6)`，trainer 的 forward 使用 `math.isclose(rel_tol=2e-6, abs_tol=2e-6)`。两者接受该浮点结果；checkpoint 与后续 technical 检查却要求与名义列表精确相等。因此前向数值契约和产物校验发生内部不一致，导致 checkpoint 验证失败并级联为 trainer/controller 的技术失败。

最小判别实验只加载原始 step 0 checkpoint 和冻结 scheduler 配置，在 CPU 重新运行四个 timestep 的构造，不加载模型、不运行模型 forward、不执行 backward 或 optimizer step。所得序列与原始保存值精确相同，严格十进制等值为 false，既有 forward 容差判定为 true。这证实失败来自浮点记录校验不一致。

修复仅复用原有 forward 容差，并保持 payload 与 record 的实测值精确相同。不会更改 sampler 计算，也不会将旧失败目录改判为通过；正式训练仍须等待新提交的新预检通过。

## 原始交付

原始归档 `phase0-failure-original.tar.gz` 包括两层 terminal/inventory、manifest、环境、日志、step 0 checkpoint、condition 与图片。远端归档位于上述 controller 根的父目录。

- bytes：`2141153`。
- SHA-256：`c543a24e3ad5f6f9902d35e241c3fd04b6efeb99513ecbcb212742832c1b4fbf`。
- 机器可读最小诊断见 `scheduler-roundtrip-diagnostic.json`。

## 可解释范围

本轮得到了一次有限非零的全链路 x_T 梯度观察；它尚不是预检工程通过，更不能证明 oracle 可达、共享 writer 可学习或 Picture Memory 成功。

修复联合回归：core、trainer、controller、aggregator 共 `52 passed in 60.47s`；Ruff 与 `git diff --check` 通过。归档两层 inventory 的 33 条文件哈希独立复算全部一致，旧技术失败终态被完整保留。回归覆盖实际 FP32 sigma 接受、越容差/NaN/错长度拒绝、raw 值保持，以及两份容差内但彼此不一致的 artifact 仍被拒绝。

首次预检启动时间也是本轮 30 小时规划的起点。诊断、修复与重跑时间均计入已消耗时间。
