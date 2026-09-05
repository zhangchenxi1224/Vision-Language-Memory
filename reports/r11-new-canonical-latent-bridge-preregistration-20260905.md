# R11_new canonical-latent bridge-distance 预注册

> 日期：2026-09-05
> 状态：在任何 bridge optimizer 输出产生前冻结
> 父提交：`e2e60126285486d42386b4cd670b419334dd06db`
> 配置：`configs/experiments/r11_new_canonical_latent_bridge_target01.json`
> 配置 SHA-256：`ab00453511cb43265a3e3d2af6aa11c8d0aa2cf6e9ca5baf44c8695d4a8bcde0`

## 1. 当前证据与唯一问题

R11_new Phase 1A 的完整冻结 DreamLite 路径在工程上 8/8 有效，但 query-level gate 仅 6/8；Target 1、7 的 CE 均明显下降且四个固定排列全部改善，accuracy 仍为 0。严格 8/8 门未过，因此 Phase 2 保持阻塞。

canonical R11 已证明冻结 VAE latent 空间里存在 Reader 可读 code，但它绕过 DreamLite。现在必须区分：

1. 已知可读 latent 是否能由完整冻结 DreamLite 路径在锁定预算内逼近；
2. 还是路径可达，但 Reader choice-CE 的优化 landscape 没有稳定跨过正确 argmax。

本实验只回答第一个问题的经验版，不证明数学上的全局可达性。

## 2. 为什么固定 Target 1

- 选择规则在结果前固定为 Phase1A 失败目标中的最小数字索引，因此选择 Target 1，而不是根据 teacher 难度或预期成功率挑选。
- Target 1 segment：`r5-f1-392d41fd097d069c42218e0a`。
- Target 7 暂不并行运行，避免看到 Target 1 结果后选择性解释；它只在本轮决策完成后按新预注册处理。

## 3. 单一改变因素

| 项目 | Phase1A | 本 bridge 实验 |
| --- | --- | --- |
| 可训练对象 | `x_T_fp32` | 不变 |
| DreamLite 路径 | 完整冻结四步 | 不变 |
| source/event/condition | Target 1 固定输入 | 不变 |
| 初始化 | seed 0 event-noise | 不变且 step-0 hash 必须相同 |
| optimizer | Adam, LR 0.05, WD 0 | 不变 |
| 预算 | 256 steps | 不变 |
| clipping | 无 | 不变 |
| checkpoints | 0/64/128/192/256 | 不变 |
| primary endpoint | raw step 256 | 不变 |
| **训练 objective** | Reader choice CE | **改为 endpoint-to-teacher FP32 mean MSE** |

目标函数：

```text
L_bridge = mean((Phi_DL(x_T; blank_source_latent, event_text) - z_R11*)^2)
```

Reader 不参与训练梯度，只在 teacher、M0 和 raw step-256 endpoint 上做固定审计。这个改变是 objective 本身的必要组成，不额外更改模型、输入、预算或评测口径。

## 4. 不可变输入绑定

Phase1A parity：

- initial `x_T_fp32`：`c970092e2afca24ededea1aec2892bd6bd54ba0dd2193522dab22af10ac1d991`；
- initial `z_t_fp32`：`11c7216fe2a70f0caa314d182b2c176b4f50c78f2d081f5aa0271184e5c8e659`；
- blank source RGB：`a3b784da71eaa113fb4d9d71502a7a3526ba0d41e2d42ed96fe79111ca3dba65`；
- source latent：`719e92867b60546b21b281cfc633ab782c8ce2274bfb41c6b3cee6d673e74eaa`；
- event text：`f170a7e2dfe0070fbd160c09d29dbcf897ddbf5f75929a3ee4af84cf627965bb`；
- condition prompt/mask：`473bd457...a60` / `4f941a46...ea9`；
- DreamLite/Reader snapshot manifests：`1bcf41b1...3159` / `159a504d...27c`。

canonical R11 teacher：

- source commit：`f4a018f3c4eef453fff3367b049ea732332e8c37`；
- file SHA-256：`d359291de63bb5232325b2e7a9294ff3d861287c06e63da2ab6ebe42eab036b9`；
- schema/key：`vision_memory.r11-vae-latent-endpoint.v1` / `latent_fp32`；
- tensor SHA-256：`6857afeffd37124bb196ab7c6607580c57c950d72d760ca6b49f8cc00bdef3f1`；
- shape/dtype：`[1,4,128,128]`, FP32, finite；
- population std：`0.6546660662`；
- 历史四排列 readout：4/4 correct，mean CE `0.000176956004`。

## 5. 先过技术预检

technical-preflight 执行一次完整四步 forward 和一次 MSE backward，不执行 optimizer step，也不计算 bridge 科学结果。它必须同时验证：

- teacher 文件、tensor、shape、dtype 与 hash；
- Target 1 的 source、condition、initial xT、initial z 全部 parity hash；
- 只有 `x_T_fp32` 可训练；
- 每次恰好四个 DreamLite steps；
- xT gradient finite 且 nonzero；
- DreamLite/Reader 参数全部冻结且无 gradient；
- teacher 经当前冻结 VAE/Reader 重放仍是同一 query 的四个固定 reverse-cyclic 排列全部正确，mean CE 不高于 `0.001`。

teacher replay 是输入有效性技术门，不是 bridge 结果。若失败，formal 不启动。

## 6. Formal 指标与预注册门槛

每一步保存 MSE、RMSE、L2 distance、相对 M0 的 MSE/L2 比例、以 teacher population std 归一化的 RMSE、xT gradient/update，以及完整四步 trajectory stats。

Primary endpoint 固定为 raw step 256，禁止用中间最佳 checkpoint 救场。强 bridge pass 必须同时满足：

1. exact technical gate；
2. teacher replay gate；
3. `MSE_256 / MSE_0 <= 0.01`，即至少消除 99% 初始 MSE；
4. `L2_256 / L2_0 <= 0.10`，即至少收缩 90% 初始欧氏距离；
5. `RMSE_256 / std(z_R11*) <= 0.10`，避免只因初始距离极大而得到虚假的相对改善；
6. raw step-256 endpoint 在同一 query 的四个固定 reverse-cyclic 排列上 accuracy 等于 1.0。

0.01/0.10/0.10 门槛在任何 bridge 训练输出前固定。四个排列只是同一 query 的稳健性排列，不当作四个独立统计样本。

## 7. 结果决策表

| 距离门 | Reader transfer | 结论与下一步 |
| --- | --- | --- |
| PASS | PASS | 路径在锁定预算内经验上到达可读 teacher 邻域；下一轮优先把 QA objective/optimization 作为单因素原因 |
| PASS | FAIL | 欧氏近邻不足以保证 Reader transfer；先做 teacher-neighborhood 稳健性诊断 |
| FAIL | PASS | 未精确逼近 teacher，但 dense guidance 找到可读 endpoint；仍优先定位 QA objective |
| FAIL | FAIL | 只说明该已知 teacher 在当前 solver/预算下未到达；下一轮只改变 budget、scheduler 或 initialization 中一个，不宣称数学不可达 |

无论哪种结果，`formal_success=false`，Phase 2 继续阻塞。

## 8. 运行与交付顺序

1. 提交并推送本预注册；
2. 实现 trainer/controller/independent aggregator 与 fail-closed tests；
3. 推送实现 commit；
4. 在 Inspire 创建该 commit 的 clean detached checkout；
5. fresh root 跑 technical-preflight；
6. 预检通过后 fresh root 跑 Target 1 formal 256 steps；
7. 独立从 raw receipts、checkpoints、teacher 与 reader rows 复算；
8. 提交报告、配置、日志、五个 checkpoints、metrics、图片、原始 rows、环境、inventories 与 SHA。

## 9. 禁止越界

- 不改变 Phase1A 原结果或门槛；
- 不事后降低 bridge 门槛或选 best checkpoint；
- 不把 human-unreadable image 当成功/失败判据；
- 不把一次失败解释为所有可读 latent 在数学上不可达；
- 不声称 state-level memory、shared writer、ID/OOD、长期递归或 Picture Memory 科学成功。
