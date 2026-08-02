# DreamLite 限时全量级训练实验报告

## 1. 结论摘要

| 判定项 | 结果 | 说明 |
|---|---:|---|
| 训练执行 | **通过** | 进程正常退出，`exit_code=0`，256 个 optimizer steps 全部完成 |
| 3–4 小时时间预算 | **通过** | 训练器计时 3:07:14；终端墙钟 3:07:36 |
| 全量级机制 | **通过** | 使用完整 DreamLite + Qwen Reader、1024 分辨率、4-step DreamLite、whole-episode BPTT 和正式数据源 |
| 原始正式矩阵 | **未完成，符合本次缩减约束** | 单 seed、2048/5000 个训练 episode（40.96%），未执行多 seed、完整 epoch 或 ID/OOD 矩阵 |
| 学习趋势 | **弱 / 不确定** | 首末 32 步 loss 均值仅下降 2.13%，中位数反而上升 1.99%，没有清晰收敛趋势 |
| 科学测评结论 | **不可判定** | 仅在末步测了固定 8 个 dev episode × 4 个 choice views；没有 step-0 dev baseline，也没有独立 ID/OOD scientific gate |

本次实验在工程层面成功：完整训练链路、正式数据、检查点、指标和可复现信息均正常产出，并将总时间控制在 3–4 小时内。但现有证据不足以证明 DreamLite 已经学到稳定有效的更新策略。最明显的优化风险是：256/256 个更新的裁剪前梯度范数都高于阈值 1.0，最后一步达到 `27,106,712`，训练处于持续梯度裁剪状态。

## 2. 实验目标与范围

目标是跳过耗时的分层测评，直接在正式数据源上运行 DreamLite 完整训练机制，观察 loss、梯度、吞吐和显存，并把总实验时长限制为 3–4 小时。

本报告中的“限时全量级”指：

- 完整模型与完整训练机制，不降低 DreamLite 分辨率或 denoising steps；
- 从正式 5000 条 train pool 全量加载并确定性 shuffle，不截取有序前缀；
- 用最大 optimizer steps 限制时间，因此实际消费 2048 条、占正式 train pool 的 40.96%；
- 只跑 seed 0 和一个 QA-only 主臂；
- 仅做一次末端小规模 dev 观测，不复现原三种子、两轮训练和完整 ID/OOD 矩阵。

因此，本次结果适合回答“完整 DreamLite 训练能否在 3–4 小时稳定跑通、loss 是否表现出初步学习迹象”，不适合替代正式统计实验。

## 3. 运行环境与可复现信息

| 项目 | 值 |
|---|---|
| Inspire 实例 | `vlm-dreamlite-full-h200x2-20260720` |
| 节点 / 算力组 | `qb-prod-gpu1930` / `开发区-H200-3号机房` |
| GPU | 2 × NVIDIA H200 SXM 141 GiB |
| CPU / RAM / SHM | 40 CPU / 400 GiB / 128 GiB |
| 镜像 | `ngc-pytorch:25.02-cuda12.8.0-py3` |
| Python / PyTorch / CUDA | 3.12.3 / 2.7.0a0+nv25.02 / 12.8 |
| 代码提交 | `10bde565d30d119a68e8460757d979b1c35e1b8f`，worktree clean |
| 运行 ID | `dreamlite-fullscale-formal-v1-seed0-20260720-10bde56` |
| 启动时间（北京时间） | 2026-07-20 02:35:37 |
| 完成时间（北京时间） | 2026-07-20 05:43:13 |
| 终端状态 | `succeeded`, `passed=true`, `exit_code=0` |

严格确定性模式已启用，包括 deterministic algorithms、关闭 TF32、固定 Python/CUDA 随机源并使用 math SDPA。两张 GPU 分工为 DreamLite `cuda:0`、Qwen Reader `cuda:1`；当前 trainer 不使用 DDP，因此增加 GPU 数量不会加速单训练臂。

### 数据与模型指纹

| 工件 | SHA256 / revision |
|---|---|
| formal-v1 train（5000） | `24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184` |
| formal-v1 dev（500） | `8b167df38022a631d4e631d3c0d66e9fca74171f4224fec436030d6650047303` |
| formal-v1 manifest | `089beefa00b9b78149b3d7f4bd40cf802dc2c92a3757c04f30d9534bbdc51215` |
| DreamLite revision | `6695c3f4be230f0493fa5dbf78be3bc4d3bb2ab4` |
| DreamLite snapshot manifest | `1bcf41b170c4b4a806bac6701cbdf4fabd5c3c53fa67415d065ab95ce2703159` |
| Qwen Reader revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Qwen snapshot manifest | `159a504daaae6dc412535978f087150a0eb8e50164afd70a8a17f83906f1127c` |
| preflight | `056b39959c8ae8d5cc9a8006e3b03d49f3cb1f59ef4b31984d5a42e28781dfc9` |
| launcher configuration | `bfe78ee4529f34d25b61c574da33e5077d7080db3776115295c034e6c1e1286a` |

## 4. 训练配置

| 参数 | 设置 |
|---|---|
| regime / objective | `qa_only` / `qa` |
| Reader loss | `listwise-choice` |
| choice schedule | `cyclic4`；dev 使用 reverse cyclic4 四视图 |
| recurrence | `direct_latent`，不 detach events |
| curriculum / noop policy | `full` / `update` |
| initial state | 1024×1024 中性灰 blank fixture |
| seed / adapter seed | 0 / 0 |
| LoRA rank / 可训练参数 | 4 / 1,644,544 |
| optimizer steps | 256 |
| gradient accumulation | 8 episodes / step |
| 实际训练 episode | 2048，正式 train pool 的 40.96% |
| learning rate / weight decay | `1e-4` / `0.01` |
| gradient clip | global norm `1.0` |
| checkpoint | 每 8 步，共 32 个周期检查点，另有 `best.pt` 和 `last.pt` |
| dev | 仅 step 256；8 episodes × 4 views = 32 次视图评估 |

## 5. 主要结果

| 指标 | 结果 |
|---|---:|
| optimizer steps | 256 |
| 训练 episode presentations | 2048 |
| 最后一步 train loss | 14.4677 |
| 唯一一次 / 最优 dev loss | 10.0845 |
| 全程 train loss 均值 / 中位数 / 标准差 | 14.3718 / 14.2466 / 3.0215 |
| loss OLS 斜率 / Pearson r | -0.002179 每步 / -0.0534 |
| 首 32 步 loss 均值 | 14.2365 |
| 末 32 步 loss 均值 | 13.9335 |
| 首末均值变化 | -2.13% |
| 首 32 步 loss 中位数 | 13.7783 |
| 末 32 步 loss 中位数 | 14.0529 |
| 首末中位数变化 | +1.99% |
| 裁剪前梯度中位数 | 17,250.46 |
| 裁剪触发率 | 256/256 = 100% |
| 裁剪前梯度最大值 | 27,106,712（step 256） |
| 训练器耗时 | 11,233.80 秒 = 3:07:14 |
| 纯训练累计耗时 | 11,179.35 秒 |
| 平均 optimizer step | 43.67 秒 |
| 平均单 episode | 5.46 秒 |
| 吞吐 | 0.1832 episode/s |
| 峰值显存 | cuda:0 21.16 GiB；cuda:1 21.39 GiB |

### 每 32 步窗口统计

| optimizer steps | loss mean | loss median | loss min | loss max | raw grad median | raw grad max |
|---|---:|---:|---:|---:|---:|---:|
| 1–32 | 14.2365 | 13.7783 | 9.7063 | 21.0319 | 11,839 | 1,039,446 |
| 33–64 | 14.8406 | 15.0022 | 9.4939 | 21.5287 | 20,667 | 749,078 |
| 65–96 | 14.1675 | 13.3347 | 8.1111 | 21.4458 | 9,534 | 983,802 |
| 97–128 | 15.2154 | 15.0145 | 8.1654 | 22.1397 | 13,023 | 186,763 |
| 129–160 | 14.4144 | 14.6444 | 8.3367 | 18.8850 | 18,325 | 316,645 |
| 161–192 | 13.8126 | 13.8080 | 7.6901 | 20.8284 | 14,500 | 2,908,102 |
| 193–224 | 14.3536 | 14.2071 | 7.9737 | 24.7281 | 34,983 | 309,537 |
| 225–256 | 13.9335 | 14.0529 | 9.6166 | 19.2445 | 19,809 | 27,106,712 |

![训练 loss 曲线](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/figures/loss_total.png)

![梯度范数与裁剪](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/figures/gradient_norm_and_clip.png)

![显存与吞吐](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/figures/memory_throughput.png)

## 6. 结果解释

### 6.1 工程结论

完整 DreamLite 训练在 2×H200 上稳定执行了 256 个更新，没有 OOM、NaN/Inf、进程崩溃或检查点中断。3–4 小时预算可支持约 2048 个正式 episode presentations，且 32 个周期检查点均成功保存。因此，当前代码和环境足以支持后续定向实验。

峰值显存只有约 21.4 GiB/卡，远低于 H200 的 141 GiB。这里说明显存不是当前限制，但本次没有采集 SM 利用率，不能仅凭显存数据判断算力利用率。严格确定性 math kernels、4-step DreamLite 和逐 episode Reader 前向共同决定了当前约 0.183 episode/s 的吞吐。

### 6.2 学习结论

loss 的 EMA 在早期下降后长期围绕 14 高噪声振荡。首末窗口均值下降 2.13%，但中位数上升 1.99%；各 32 步窗口也没有单调改善。全程 OLS 斜率只有 -0.002179/step，Pearson `r=-0.0534`，说明 step 与 loss 几乎没有稳定线性关系。因此，现阶段只能认为存在很弱的均值变化，不能称为收敛。

`10.0845` 是 step 256 唯一一次 dev loss，也因此被程序记为 `best_dev_loss`。由于没有相同 dev slice 上的 step-0 baseline，不能据此计算泛化改进；8/500 的 dev 覆盖率也只有 1.6%。

梯度是更强的风险信号：所有更新都触发 global-norm clip，最后一步的裁剪前范数比总体中位数高约 1571 倍。step 256 的 train loss 本身并非离群点，说明裁剪至少抑制了即时 loss 爆炸；这不等于已经发生数值发散。但持续裁剪和重尾梯度表明优化过程由 clip 主导，当前 `lr=1e-4`、loss 尺度或梯度传播路径值得进一步诊断。

## 7. 局限

- 没有 step-0 dev baseline；无法量化训练前后改善。
- 只在 step 256 评估一次，无法判断最佳点是否更早出现。
- dev 只使用固定 8/500 episodes，尚未覆盖 ID/OOD test。
- 只有 seed 0，不能给出方差或稳定性结论。
- 正式 train pool 只消费 2048/5000；没有完成完整 epoch，更没有原正式矩阵的两轮训练。
- `state_gradient_audit` 本次按限时方案关闭；已有 raw global gradient norm，但没有逐状态张量的梯度审计。
- 未采集 GPU SM 利用率、功耗和 kernel profile；只能报告端到端吞吐与峰值显存。
- 启动器没有生成独立 stage-evidence/scientific-gate 文件，所以自动报告能证明 terminal 与日志哈希一致，但不会把本次标为“strict scientific complete”。

## 8. 建议的最小后续实验

1. **先做 checkpoint-only 评估，不立即重跑长训练。** 在完全相同的固定 dev slice 上测未训练初始化，以及 steps 64、128、192、256 四个检查点。这样能低成本补出真正的学习曲线和 step-0 baseline。
2. **诊断持续裁剪。** 增加 post-clip norm、update/weight ratio 和按模块梯度统计；先用 32–64 步短跑比较 `lr=1e-4` 与 `3e-5`。不建议在原因未明时直接放大 clip 阈值。
3. **只有在 checkpoint dev 曲线显示改善时再扩量。** 届时优先扩到完整 5000 presentations 或增加一个 seed，而不是马上恢复整个多臂分层矩阵。
4. **补 scientific gate。** 固定 dev/ID/OOD slice、明确 baseline 与容差，并把 gate 输出绑定到 terminal，之后自动报告即可通过 strict-complete 校验。

## 9. 工件与审计入口

- [自动生成的单文件 HTML 报告](../runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/report.html)
- [自动生成的 Markdown 报告](../runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/report.md)
- [训练曲线 CSV](../runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/metrics/training_curve.csv)
- [机器可读报告摘要](../runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/metrics/report_summary.json)
- [完整 SHA256 清单](../runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/artifacts.sha256)
- [原始 metrics.jsonl 副本](../runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/source/metrics.jsonl)
- [terminal.json 副本](../runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/provenance/terminal.json)
- [训练失效诊断可视化 HTML](../runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/diagnostic_report.html)
- [训练失效诊断 Markdown](../runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/diagnostic_report.md)
- [训练失效诊断机器摘要](../runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/diagnostics/diagnostic_summary.json)
- [训练失效诊断 SHA256 清单](../runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/diagnostic_artifacts.sha256)

原始自动报告基线共 20 个文件，仍由 `artifacts.sha256` 逐一校验且未被改写。新增诊断数据、图表和报告使用独立的 `diagnostic_artifacts.sha256`，避免改变基线证据含义。远端原始运行目录为：

```text
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r3/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56
```

自动证据包保存在该目录的 `report-v1/`；本地镜像保存在 `runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/`。

## 10. 训练失效诊断补充

### 10.1 分析单位与证据边界

已按训练代码中的 `episode_order(seed=0, epoch=0)` 重建实际 shuffle 顺序，并将前 2048 条 episode 与 256 条 train metrics 精确对齐。所有 metrics 的 `episode_cursor` 均等于 `optimizer_step × 8`，正式 train 文件 SHA256 与 manifest 完全一致，5000 条 episode 的 schema、反事实链接、clean/NOOP 链接和 mixed-delayed probe 结构检查均通过。

需要特别强调：原 run 每 8 条 episode 才记录一次 group mean loss 和一次累积后的 raw global gradient norm。因此，本节以 **8-episode 梯度累积组**为独立统计单位。旧日志不能恢复逐 episode 梯度，也不能恢复各 LoRA block 的梯度；异常审计只能先锁定候选组，不能把组级梯度虚假分摊给组内样本。

实际消费数据的路由构成为：

| 项目 | 分布 |
|---|---|
| updater calls / episode | 2 次：1019；3 次：1029 |
| reader queries / episode | 2 次：1015；3 次：1033 |
| mixed / separate | 1033 / 1015 |
| topic | color 360；drink 377；material 361；meal 301；music 316；style 333 |
| 实际事件序列 | `set→set` 497；`set→overwrite` 395；`set→noop→overwrite` 377；`set→set→noop` 276；`set→noop→set` 251；`set→clear` 127；`set→noop→clear` 125 |
| delayed probe | 2048/2048 均存在，因而没有 non-delayed 对照 |

### 10.2 事件类型 × update 次数 × 梯度

以 `log10(raw global gradient norm)` 为因变量、256 个累积组为独立样本，得到以下探索性关联。置信区间由固定 seed 2026 的 10,000 次 group bootstrap 得到；`Holm p` 同时校正 21 个事件、主题、loss 和阶段特征。

| 组级特征 | Spearman ρ | 95% bootstrap CI | raw p | Holm p |
|---|---:|---:|---:|---:|
| 平均 updater calls | +0.134 | [0.008, 0.256] | 0.0316 | 0.6004 |
| mixed episode 数 | -0.026 | [-0.147, 0.097] | 0.6813 | 1.0000 |
| SET 事件数 | +0.138 | [0.014, 0.255] | 0.0272 | 0.5429 |
| OVERWRITE 事件数 | -0.065 | [-0.188, 0.056] | 0.2989 | 1.0000 |
| CLEAR 事件数 | -0.108 | [-0.237, 0.019] | 0.0834 | 1.0000 |
| NOOP 事件数 | +0.134 | [0.010, 0.256] | 0.0316 | 0.6004 |
| group loss | +0.102 | [-0.016, 0.215] | 0.1032 | 1.0000 |
| optimizer step | +0.150 | [0.031, 0.268] | 0.0165 | 0.3458 |

没有组成特征在多重校正后达到 0.05。SET 是所有 episode 的初始状态建立事件，无法形成“无 SET”对照；NOOP 数、3-update episode 数、turn 数和 distractor 数由数据生成规则结构性绑定，因此不能用这份旧日志把它们分别解释成独立原因。

![事件/更新组成与梯度](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/figures/diagnostic_event_update_gradient.png)

组 loss 与梯度仅弱相关。最高 loss 的 step 200 为 24.7281，但 raw gradient 只有 22,549.7；最高梯度的 step 256 loss 为 14.4677，并不是 loss 异常组。这反驳了“最难样本组自然产生最大梯度”的简单解释。

![loss 与梯度](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/figures/diagnostic_loss_vs_gradient.png)

### 10.3 训练阶段

| optimizer steps | loss mean | loss median | raw grad median | raw grad P90 | raw grad max |
|---|---:|---:|---:|---:|---:|
| 1–64 | 14.5385 | 14.2951 | 18,556 | 99,787 | 1,039,446 |
| 65–128 | 14.6914 | 14.4727 | 10,311 | 117,323 | 983,802 |
| 129–192 | 14.1135 | 14.2942 | 16,821 | 114,662 | 2,908,102 |
| 193–256 | 14.1436 | 14.1633 | 24,709 | 268,440 | 27,106,712 |

四阶段梯度分布的 Kruskal-Wallis `p=0.03285`，loss 的对应值为 `p=0.79132`。`log10(raw gradient)` 的 Theil-Sen 每步斜率为 `+0.001387`，95% CI `[+0.000266, +0.002560]`。梯度不是逐阶段单调上升，但末四分位的中位数、P90 和极端尾部均明显抬升，而 loss 没有同步恶化，说明参数轨迹或反向传播尺度比数据难度更值得优先排查。

![分阶段梯度分布](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/figures/diagnostic_gradient_by_phase.png)

### 10.4 哪些任务实际在学习

使用 256 个独立组拟合条件线性 loss 趋势，并控制事件序列、mixed 比例和 choice rotation；以下数值表示模型估计的 step 1→256 loss 变化：

| topic | 建模 loss 变化 | 95% group-bootstrap CI | 可判定学习 |
|---|---:|---:|---:|
| color | +2.872 | [-4.517, +9.674] | 否 |
| drink | +2.587 | [-3.741, +9.054] | 否 |
| material | -0.322 | [-6.162, +5.420] | 否 |
| meal | -1.990 | [-10.760, +6.307] | 否 |
| music | -1.374 | [-7.501, +5.138] | 否 |
| style | -5.362 | [-12.470, +2.571] | 否 |

style 的点估计方向最好，`set→set` 事件序列也有 `-3.983` 的点估计，但两者 CI 均跨 0。结合缺少固定 dev checkpoint accuracy/margin，当前不能声称任何主题或事件任务已经稳定学会。

![主题与事件序列学习趋势](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/figures/diagnostic_category_learning_trends.png)

### 10.5 异常样本候选审计

共有 3 个累积组超过一百万 raw gradient：step 256（27,106,712）、step 179（2,908,102）和 step 11（1,039,446）。step 256 比第二大值高 9.32 倍，其推导裁剪系数只有 `3.689×10^-8`。

step 256 的 8 条候选如下：

| 位置 | episode ID | topic | 实际事件序列 | updates | queries | mixed |
|---:|---|---|---|---:|---:|---:|
| 1 | `r3-train-semantic-000103-s0-noop` | style | `set→noop→overwrite` | 3 | 2 | 0 |
| 2 | `r3-train-semantic-000262-s0-clean` | drink | `set→overwrite` | 2 | 2 | 0 |
| 3 | `r3-train-semantic-000139-s1-noop` | meal | `set→noop→overwrite` | 3 | 2 | 0 |
| 4 | `r3-train-semantic-000199-s1-clean` | color | `set→overwrite` | 2 | 2 | 0 |
| 5 | `r3-train-semantic-000831-s0-clean` | material | `set→set` | 2 | 3 | 1 |
| 6 | `r3-train-semantic-001046-s0-clean` | drink | `set→overwrite` | 2 | 3 | 1 |
| 7 | `r3-train-semantic-000463-s0-clean` | drink | `set→set` | 2 | 2 | 0 |
| 8 | `r3-train-semantic-001006-s0-noop` | material | `set→set→noop` | 3 | 2 | 0 |

该组平均 2.375 次 updater call，只有 2/8 mixed，没有 CLEAR；它不是长 episode 或 mixed episode 特别集中的组。完整的最高梯度 20 组、最高 loss 20 组及各自 160 条候选分别保存在：

- [最高梯度 20 组](../runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/diagnostics/top20_gradient_groups.csv)
- [最高梯度组的逐条候选](../runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/diagnostics/top20_gradient_group_candidates.csv)
- [最高 loss 20 组](../runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/diagnostics/top20_loss_groups.csv)
- [最高 loss 组的逐条候选](../runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/diagnostics/top20_loss_group_candidates.csv)

### 10.6 三个问题的当前答案

1. **哪类样本导致千万级梯度？** 只能锁定 step 256 的 8 条候选，不能在旧日志中定位单条。该组事件和长度组成普通，暂无证据把根因归给 CLEAR、NOOP、mixed 或长 episode。
2. **哪些任务实际上已经在学习？** 尚无可靠类别。style 和 `set→set` 只有方向性信号，置信区间均跨 0；没有固定 dev 基线就不能升级为学习结论。
3. **问题来自梯度尺度、数据难度还是多轮状态更新？** 当前证据优先支持“优化轨迹或梯度传播尺度”工作假设：loss-gradient 关联弱、极端组不长、梯度尾部在后期加重。多轮更新只有弱正关联且 Holm 校正后不显著；具体模块根因仍未测得。

### 10.7 尚未由本次旧日志补齐的项目

- **逐 episode 梯度与前 20 episode**：必须对异常组候选逐条 `zero_grad → backward → record → discard`；旧日志无法逆推。
- **按模型模块梯度**：正式 run 没有记录 `down/mid/up × q/k/v/out × LoRA A/B` 的 pre/post-clip norm，也没有实际 AdamW update/weight ratio。
- **固定 dev checkpoint 曲线**：当前仍只有 step 256 的单一 dev loss；init、64、128、192、256 的 CE、accuracy、margin 必须从检查点补测。
- **因果评测**：blank、reset、matched state-swap、state shuffle 尚无完整终态结果，不能判断模型是否已经使用视觉状态。

因此，这一版报告已经完成了现有数据允许的最细粒度离线归因，并把千万级梯度缩小到可复查的 8 条候选；要继续收敛到单条样本和单个 LoRA block，必须使用新增梯度探针，不能继续从同一份聚合日志外推。
