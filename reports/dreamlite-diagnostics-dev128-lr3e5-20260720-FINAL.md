# DreamLite 全量训练后诊断与低学习率实验报告

日期：2026-07-20（Asia/Shanghai）  
原正式实验：`dreamlite-fullscale-formal-v1-seed0-20260720-10bde56`  
诊断实验：`dreamlite-diagnostics-dev128-20260720-10bde56`

## 结论先行

**当前不应继续扩大 DreamLite 的训练步数或 seed 数量。** 256-step checkpoint sweep 没有证明相对初始化发生学习改善；`lr=3e-5` 的 64-step 对照也没有通过相同的学习门。模型确实依赖写入后的 memory state，但这一优势在未训练初始化时已经存在，因此不能把它解释成训练学到的新能力。

核心判断如下：

| 问题 | 结果 | 判断 |
|---|---:|---|
| `lr=1e-4` 训练到 step 256 是否优于初始化 | ΔCE `+0.003608`，95% CI `[-0.427757, +0.418330]` | **未通过学习门** |
| step 256 是否依赖 memory state | standard−reset ΔCE `-10.287912`，95% CI `[-12.816189, -7.797490]` | **通过 memory 门，但初始化也通过** |
| 巨大梯度来自哪里 | one-step 中 LoRA-B 占 pre-clip 平方范数 `95.97%`；64-step 中位数 `99.86%` | **定位为 factor 级长期集中** |
| `3e-5` 是否控制实际更新 | update norm 与 update/weight 均缩为 `1e-4` 的 `30.000005%` | **通过更新收缩门** |
| `3e-5` 是否解决梯度异常 | raw norm 仍很大，64/64 steps 全部触发 clip=1 | **没有解决** |
| `3e-5` 是否优于 `1e-4` | ΔCE `-0.195628`，95% CI `[-0.610139, +0.187255]` | **不确定**：既未证明优越，也未证明 `+0.01` 非劣；同时不能称为更差 |
| 是否支持继续全量扩展 | checkpoint learning gate=false；低 LR overall gate=false | **暂不支持** |

这里最重要的统计边界是：`lr3e5 − lr1e4` 的 CE 点估计为负，方向上有利于低学习率；但置信区间同时跨过 `0` 和预注册非劣界值 `+0.01`。因此结果是 `inconclusive_not_noninferior`，不是“低 LR 更差”。

## 实验与统计契约

- 算力：Inspire 实例 `vlm-dreamlite-full-h200x2-20260720`，2×H200 141 GB，节点 `qb-prod-gpu1930`。
- 原训练代码：commit `10bde565d30d119a68e8460757d979b1c35e1b8f`。
- 诊断代码：commit `94bfc6c532392d7c1304b42dea2ab2b867ce7345`；新增 opt-in optimizer diagnostics，默认训练行为不变。
- DreamLite nested revision：`a6e20c8cc94027f37dd7c5a81b0b3b472aa18409`。
- 固定 dev：128 episodes、32 semantic groups、332 base queries；每个 query 使用 reverse-cyclic4，共 1,328 条 prediction/condition。
- dev 在 color、meal、music、style 四个 topic 上各 32 episodes；dev SHA256 为 `42691e61b9fdfa4068f3cab63d77889b38e25b5978de99ea89d99bed77763deb`。
- 所有差值均为“右侧方法减左侧方法”；CE 越低越好，accuracy、correct margin 越高越好。
- 95% CI 使用 semantic-group paired cluster bootstrap，32 clusters、10,000 次；checkpoint 的四个 CE 比较另做 Holm 校正。
- sweep、两条 optimizer probe、64-step short arm、matched eval v2 均为 `status=succeeded / passed=true / exit_code=0`，且产物 SHA 严格校验通过。

## Checkpoint sweep：未观察到可重复学习改善

### 绝对指标

| checkpoint / condition | Accuracy | Listwise CE mean | Correct margin mean | View consistency |
|---|---:|---:|---:|---:|
| init / standard | 15.1355% | 12.057505 | -10.992422 | 77.1084% |
| step 64 / standard | 14.6837% | 12.313948 | -11.194638 | 76.8072% |
| step 128 / standard | 14.8343% | 12.269273 | -11.234427 | 79.5181% |
| step 192 / standard | 13.9307% | 12.649623 | -11.589926 | 80.4217% |
| step 256 / standard | 14.0060% | 12.061112 | -10.911615 | 78.3133% |
| init / reset | 10.8434% | 22.349025 | -20.149451 | 100.0000% |
| step 256 / reset | 10.8434% | 22.349025 | -20.149451 | 100.0000% |

### 相对初始化的配对差值

| 比较 | ΔCE [95% CI] | Holm-adjusted p | ΔAccuracy [95% CI] | ΔMargin [95% CI] |
|---|---:|---:|---:|---:|
| init → step 64 | `+0.256443 [-0.253037, +0.735613]` | 0.9432 | `-0.004518 [-0.017405, +0.007622]` | `-0.202217 [-0.662519, +0.275770]` |
| init → step 128 | `+0.211768 [-0.293378, +0.762864]` | 0.9432 | `-0.003012 [-0.013235, +0.007530]` | `-0.242006 [-0.809313, +0.282927]` |
| init → step 192 | `+0.592119 [+0.018788, +1.284180]` | 0.1624 | `-0.012048 [-0.029070, +0.002435]` | `-0.597504 [-1.231094, -0.086650]` |
| init → step 256 | `+0.003608 [-0.427757, +0.418330]` | 0.9784 | `-0.011295 [-0.028965, +0.000744]` | `+0.080806 [-0.347814, +0.539696]` |

step 192 在未经多重校正的 CE 与 margin 上表现更差；Holm 后不拒绝四个 CE 零假设。step 256 几乎回到初始化的 CE，但 accuracy 仍低约 1.13 个百分点，三个学习门条件没有同时成立。分析器结论为：

```text
learning_evidence_passed = false
step_256_standard_beats_reset = true
continue_training_supported = false
```

### Memory control 的含义

| 比较（standard − reset） | ΔCE [95% CI] | ΔAccuracy [95% CI] | ΔMargin [95% CI] |
|---|---:|---:|---:|
| init | `-10.291520 [-12.830245, -7.784987]` | `+0.042922 [+0.007714, +0.081019]` | `+9.157029 [+6.784348, +11.718449]` |
| step 256 | `-10.287912 [-12.816189, -7.797490]` | `+0.031627 [+0.003085, +0.062500]` | `+9.237835 [+6.896616, +11.745021]` |

step 256 的 standard 明确优于 reset，说明输出确实使用 memory state；但 init 的效应几乎同样大，而且 init-reset 与 step256-reset 的绝对指标完全相同。它证明的是**机制依赖**，不是**训练增益**。

## One-step probes：低 LR 只缩小步幅，未改变梯度

两条 probe 都从原 `checkpoint-000256.pt`（SHA256 `3b648eccf9df05c7df1da35626c271d5805b4cd7c378e10f44650b20afd61880`）开始，使用同一 8 episodes、fresh AdamW、gradient accumulation=8，只改变 LR 和输出路径。

| 指标 | `lr=1e-4` | `lr=3e-5` | 结论 |
|---|---:|---:|---|
| Loss / loss hex | 8.740582 / `0x1.17b2d9e000000p+3` | 完全相同 | 配对成立 |
| Raw gradient SHA256 | `2b179b54…a070e` | 完全相同 | 配对成立 |
| Clipped gradient SHA256 | `9481c964…acd0d` | 完全相同 | 配对成立 |
| Global pre-clip norm | 21,647.892091 | 21,647.892091 | 极大且相同 |
| Global post-clip norm | 0.999999940 | 0.999999940 | clip 正常执行 |
| Clipping factor | 0.000046194 | 0.000046194 | 约 99.9954% 范数被裁掉 |
| Actual update norm | 0.127459477 | 0.038237850 | 低 LR / 高 LR = 0.30000005 |
| Global update/weight | 0.007048361 | 0.002114509 | 低 LR / 高 LR = 0.30000005 |
| State-gradient audit | passed | passed | final/intermediate/query 路径均连通 |

所有 probe gates 均通过：lineage、artifact SHA、paired-gradient identity、diagnostic integrity、post-clip bound、state-gradient connectivity 和 lower-LR contraction。

### 梯度归因

| 轴 | 主要分量 | Pre-clip 平方范数占比 |
|---|---|---:|
| Factor | LoRA-B | 95.9686% |
| Factor | LoRA-A | 4.0314% |
| Projection | `to_out` + `to_v` | 83.1280% |
| Stage | `mid_block` | 46.4982% |
| Cross 最大项 | `mid_block|to_out|lora_B` | 21.1467% |
| Cross 前 6 项 | 全为 LoRA-B 的 `to_out/to_v` | 80.5159% |

这不是单个 block 的偶发尖峰，而是 LoRA-B × out/v 路径的系统性集中。fresh AdamW 更新后，LoRA-B 的 update/weight 为 `5.6603%`（`1e-4`）和 `1.6981%`（`3e-5`），均约为 LoRA-A 的 9.31 倍。state audit 还显示 `first_intermediate_state` 最大梯度范数为 `1829.19`，远高于 `final_state=19.85` 和 `query_image=12.01`，应优先排查 recurrent memory 的中间状态路径及其 loss 尺度。

## `lr=3e-5` 64-step short arm

short arm 从头训练 64 optimizer steps（512 episode presentations），墙钟时间 3,727.93 秒；执行与诊断完整性通过。

| 训练统计 | `lr=1e-4` steps 1–64 | `lr=3e-5` steps 1–64 |
|---|---:|---:|
| Loss mean / median | 14.538544 / 14.295135 | 14.480566 / 14.250830 |
| Loss OLS slope / Pearson r | +0.006815 / +0.04169 | -0.001902 / -0.01207 |
| 首 16 → 末 16 loss mean | 14.497950 → 14.619259 | 14.803664 → 14.449649 |
| Raw grad median / p95 / max | 18,555.79 / 420,428.74 / 1,039,446.31 | 12,075.22 / 302,205.04 / 510,998.81 |
| Clip rate | 64/64 | 64/64 |
| Update norm median / p95 | 未记录 | 0.010742 / 0.019173 |
| Update/weight median / p95 | 未记录 | 0.0006007 / 0.0010722 |

低 LR 的描述性 loss 略低，但斜率接近零；它降低了 raw-gradient 的尾部规模，却没有消除 100% clipping。64 个 step 中 LoRA-B pre-clip squared-share 的中位数仍为 `99.8563%`，进一步确认 one-step 定位不是偶然。

训练器内部 dev loss `12.477196` 仅是健康检查；它与原正式训练的 step、dev SHA、评价口径不同，因此未用于性能结论。

## 同口径 matched eval v2

四个比较臂使用完全相同的 128 episodes、332 base queries、4 views/query 和 32 semantic-group clusters。

| Method / condition | Accuracy | Listwise CE | Correct margin | View consistency |
|---|---:|---:|---:|---:|
| init / standard | 15.1355% | 12.057505 | -10.992422 | 77.1084% |
| `1e-4` step 64 / standard | 14.6837% | 12.313948 | -11.194638 | 76.8072% |
| `3e-5` step 64 / standard | 14.0813% | 12.118320 | -11.055871 | 77.7108% |
| `3e-5` step 64 / reset | 10.8434% | 22.349025 | -20.149451 | 100.0000% |

| 比较 | ΔCE [95% CI] | ΔAccuracy [95% CI] | ΔMargin [95% CI] | Gate |
|---|---:|---:|---:|---|
| init → `3e-5` step 64 | `+0.060816 [-0.456553, +0.559859]` | `-0.010542 [-0.031609, +0.006329]` | `-0.063449 [-0.581708, +0.469692]` | learning=false |
| `1e-4` → `3e-5` step 64 | `-0.195628 [-0.610139, +0.187255]` | `-0.006024 [-0.021084, +0.007716]` | `+0.138767 [-0.249106, +0.574955]` | superiority=false；noninferiority=false；inferiority=false |
| `3e-5` reset → standard | `-10.230704 [-12.740110, -7.760410]` | `+0.032380 [+0.003765, +0.060295]` | `+9.093580 [+6.726628, +11.595692]` | memory=true |

`3e-5` 的 CE 点估计优于 `1e-4`，但 CI 上界 `+0.187255` 既不小于 0，也不小于非劣界值 `+0.01`；CI 下界又明显小于 `+0.01`，所以也不能证明其劣于 `1e-4` 超过该界值。最终分类为 **不确定且尚未证明非劣**，overall gate=false。

## 下一步建议

1. **先停止全量扩展。** 不再立即跑 5,000 presentations 或增加 seed；当前主要限制不是样本量，而是训练没有通过相对初始化的学习门。
2. **做 16–32 step 的机制微诊断。** 按 `update depth × subtype × loss component` 记录 loss 与梯度，重点检查 `first_intermediate_state`；加入只用于诊断的 detach/scale control，确认 recurrent path 是否造成量级放大。
3. **单变量测试 LoRA-A/B 优化。** 保持 `lr=3e-5`、数据顺序和全局配置不变，单独尝试 LoRA-B 的较小 LR multiplier 或 factor-wise pre-clip；不要同时改 loss、LR 和 clip。目标是降低 LoRA-B 对 pre-clip 方向的近 100% 支配，同时保持 state-gradient connectivity。
4. **候选方案先跑 64/128-step 固定 gate。** 必须同时满足：相对 init 的 CE CI 上界 < 0、accuracy Δ≥0、margin Δ>0；standard-vs-reset memory 门通过；无非有限梯度和 lineage/SHA 异常。
5. **只有 gate 通过后再扩大。** 先用第二 seed 复核，再进入完整 presentations；`3e-5` 目前可以作为较安全的诊断 LR，但证据不足以把它宣布为默认最优 LR。

## 失败契约记录：matched eval v1

第一次 matched runner `run_lr3e5_step64_matched_eval_94bfc6c.sh` 在 GPU 模型加载前 fail closed：预检查错误地把原始 JSONL 中显式 `type=query` turns 期望为 332，实际为 256。stage 记录为 `status=failed / passed=false / exit_code=1`，运行时间约 0.22 秒，没有产生可用于分析的模型预测。

该失败没有被覆盖或删除。v1 runner SHA256 为 `17b0070db39d72dbc47ee8faf6521e785a1ee7797b17de9455fd89796cf05374`，v1 terminal SHA256 为 `64c3717b031588e74370592495c8a425b294c458488f76be0dd294b9f83f4222`。v2 将 raw-turn 断言修正为 256，并使用新的 `-v2` stage/output；evaluator 展开后的 332 base queries、1,328 rows/condition 合同保持不变。v2 成功结束，未复用 v1 的部分产物。

## 可复现指纹

### 代码、数据与模型

| 工件 | SHA256 / commit |
|---|---|
| 原正式训练代码 | `10bde565d30d119a68e8460757d979b1c35e1b8f` |
| 诊断代码 | `94bfc6c532392d7c1304b42dea2ab2b867ce7345` |
| DreamLite nested revision | `a6e20c8cc94027f37dd7c5a81b0b3b472aa18409` |
| formal-v1 train | `24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184` |
| formal-v1 dev | `8b167df38022a631d4e631d3c0d66e9fca74171f4224fec436030d6650047303` |
| stratified dev-128 | `42691e61b9fdfa4068f3cab63d77889b38e25b5978de99ea89d99bed77763deb` |
| DreamLite snapshot manifest | `1bcf41b170c4b4a806bac6701cbdf4fabd5c3c53fa67415d065ab95ce2703159` |
| Qwen Reader snapshot manifest | `159a504daaae6dc412535978f087150a0eb8e50164afd70a8a17f83906f1127c` |
| passed preflight v3 | `3a321c685bd14bea3d8a37cd65dba4f27feb7707a21213f540de6e1451a6a405` |

### Runner 与分析工件

| 工件 | SHA256 |
|---|---|
| checkpoint sweep runner | `cc2fa2a855ecbf179ad9fe543f0a5a9ea5fbe7cdd0e688c6460da62d5a6ed0bd` |
| checkpoint analyzer | `b40869c1c8303768e47797408cdb2e1caf62a47c60aacc0b76bd6c8ff4d1f7c0` |
| checkpoint analysis | `7110f8a03fd2711e6ce193cfc3f0a6b6e6e124d5b0e46cb374a22629dbac87d5` |
| probe runner `1e-4` v2 | `8bf38aa27b89776d8efbdc47c9e8f4a82cb5e7a378ab92d40894dd5cab68a97c` |
| probe runner `3e-5` v2 | `df9e3bfa7b4c0300eea165e9cf8bda57d809995059b0de4a6d55c19f52cdb8ea` |
| probe validator v2 | `8af4eafb3d6b7ad0c9e201a0c9dc7d9a2192df923d90bd384fee187442eca5fd` |
| probe comparison | `1da671654d5a98a785e2dcd8d53b965663b8b1da99cc547872bdfaddf250aee2` |
| `3e-5` short-arm runner | `7e279990188f0542bc05fae0ad054670046b921574368f00af050d9668b7a491` |
| `3e-5` short-arm checkpoint 64 | `d84a75c52d3239bfe15e925d8d3c7147bbb9047505103b91f9bf9c58cde60a2f` |
| short-arm training analysis | `69e12669d843373ce7e0f2394d189a1d8d54da4dd7a670f972d6dd8e4058924f` |
| matched runner v1（失败记录） | `17b0070db39d72dbc47ee8faf6521e785a1ee7797b17de9455fd89796cf05374` |
| matched runner v2 | `41bcc45ddad063db85c13164dfa7a06b2fd694afc3b62c40f1f758e02b67a7c5` |
| matched analyzer v2 | `b76b8ff3d32e5ebcdc81a5fcee9feaea121092b5da61d81041a6e9aa627c77e4` |
| corrected matched analysis v2 | `65e57707dcc785351b4896c353635a122431614eba556194c3dc4c6d30c2f7d7` |

## 证据路径

远端 source of truth：

```text
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r3/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r3/dreamlite-diagnostics-dev128-20260720-10bde56
```

本地证据根目录：

```text
D:\2026WorkExperience\VisonLearnableMemory\runs\formal_reports\dreamlite-diagnostics-dev128-20260720-10bde56
```

本报告直接复核的本地分析工件：

- [Checkpoint sweep analysis](../runs/formal_reports/dreamlite-diagnostics-dev128-20260720-10bde56/report_inputs/checkpoint-sweep-analysis.json)
- [Optimizer probe comparison](../runs/formal_reports/dreamlite-diagnostics-dev128-20260720-10bde56/report_inputs/optimizer_probe_comparison.json)
- [64-step training analysis](../runs/formal_reports/dreamlite-diagnostics-dev128-20260720-10bde56/report_inputs/lr3e5_short_training_analysis.json)
- [Corrected matched analysis v2](../runs/formal_reports/dreamlite-diagnostics-dev128-20260720-10bde56/report_inputs/matched-analysis-v2.json)
- [Matched v1 failed terminal](../runs/formal_reports/dreamlite-diagnostics-dev128-20260720-10bde56/report_inputs/matched-v1-terminal.json)
- [Matched v1 stderr](../runs/formal_reports/dreamlite-diagnostics-dev128-20260720-10bde56/report_inputs/matched-v1-stderr.log)

原正式训练报告保持不变：

- [dreamlite-fullscale-formal-v1-seed0-20260720-10bde56.md](dreamlite-fullscale-formal-v1-seed0-20260720-10bde56.md)
