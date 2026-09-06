# R11_new：Source-only initialization 最小诊断预注册

日期：2026-09-07。协议：`R11-New-Canonical-Latent-Bridge-Source-Only-Init-Target01`。

本文在任何本轮完整模型前向或优化结果产生前锁定。它是主计划失败分支及 teacher-matched 初始化预注册第 8、10 节下的下一项最小诊断，不是 Phase 1A 通过声明或 Phase 2 启动许可。主计划仍为 `reports/r11-new-frozen-dreamlite-oracle-training-plan-20260904.md`；旧结果、配置和报告保持只读。

## 1. 唯一问题和唯一变化

本轮问：在固定 target、完整冻结四步 DreamLite、同一 teacher MSE 目标和同一优化预算下，不把逐样本 teacher 注入初始化，仅以输入 source latent 初始化 x_T，能否改善相对 Gaussian 父臂的 endpoint 恢复？

唯一变化是 `x_T` 初始化：

```python
# On the actual locked DreamLite compute device, before the first full forward:
source_latents_fp32 = source_latents.detach().float()
x_T_init_fp32 = source_latents_fp32.clone()
```

initializer 的内容输入只能是 `source_latents`。其 API 不得接收 teacher、query、answer、choices、target index 或 sample ID；固定 dtype/device/scheduler 元数据用于技术核验而非提供语义内容。不得使用 teacher 的均值、标准差、投影、梯度、优化结果或缓存变相初始化。

Teacher 仍在初始化以外用于固定 dense MSE 监督、teacher replay 和距离评测；因此只可称“初始化答案无关”，不能称“整个求解器不依赖 teacher”或“共享 writer 已学会”。本轮始终锁定：

```text
initialization_teacher_assisted = false
optimization_teacher_supervised = true
answer_independent_writer_usable = false
formal_success = false
phase2_allowed = false
```

这也不是训练 learned initializer。没有新增共享可训练模型、额外数据或额外训练目标。

## 2. 不可变配置和已闭合证据

配置：`configs/experiments/r11_new_canonical_latent_bridge_target01_source_only_init.json`。

| 锁定项 | SHA256 / commit |
| --- | --- |
| 本轮 JSON 文件 SHA256 | `2bdf0e0b45e2c0b22d4c04711c0be11c5e9b94fc30d7ee70ed1aa3cb219b1702` |
| 本轮 JSON canonical SHA256 | `cabaed92d8f80871f8a83511dd107b81672ba1b7fb33844a4241d2c0350ed1fe` |
| Gaussian/post128-cosine 训练 commit | `16318e005b496a16b7712ad4ff3cea50e2be34fa` |
| Gaussian 交付 commit | `398cf72d7d18033f2fceb1a7dd1b7a5ccf5420e7` |
| Gaussian comparison SHA256 | `00aff581a78b988658fc868158131e5a61d441fcf90e639e2c268284b1f3adfe` |
| Gaussian RAW_ARTIFACTS SHA256 | `143de73292240812e148aed0acefe936d4de01e6d7eb1b83c7fef05b629d8e49` |
| Gaussian config SHA256 | `c9794f5197f6c62f2f84af3cf0db9aee0ff225b4649004967f52d1424a247dc5` |
| Teacher-matched 训练 commit | `5bffcb9a1605c7cdef77df1126daca6f9578bdda` |
| Teacher-matched 交付 commit | `8b904d82d17ff36fb44c80e08696ed767445f38e` |
| Teacher-matched comparison SHA256 | `f66650cd6f66cbdf54c0694e8c55aa18c6ad3497ad487f8f7cb620f107d1d8ed` |
| Teacher-matched RAW_ARTIFACTS SHA256 | `2af2d9a0b6124e450a7b3658352bd1c6e58d7cede2ddcec384064e1dae9d909a` |

| 已完成臂（raw256） | 绝对 endpoint MSE | Reader mean CE | Reader | 主门 |
| --- | ---: | ---: | ---: | --- |
| Gaussian/post128-cosine | 0.09553645551204681 | 25.536474171257463 | 0/4 | 距离、Reader 均失败 |
| Teacher-matched | 0.06570044904947281 | 10.97591445709591 | 0/4 | 距离、Reader 均失败 |

两臂技术门与 teacher replay 均通过。Teacher-matched 的二级初始化改善通过，但不挽救主要失败。它只支持在这个 target 上初始化会改变结果，不证明初始化是唯一或主导根因，也未分离 teacher 内容与起点几何的贡献。

**唯一二级对照始终是 Gaussian/post128-cosine。** 新 JSON 的 `teacher_matched_reference` 仅用于原样描述和证据绑定，不得替代 Gaussian 对照、提供新阈值或成为择优标准。

## 3. 保留的 target、数据和因果锚点

固定 target1：`r5-f1-392d41fd097d069c42218e0a`。它是原 Phase 1A 失败成员中的预注册最低序号；target7 暂缓，不按结果选样本。

固定 Phase 1A 来源根：

```text
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-new/r11-new-phase1a-2cde77e-20260905-round02/target-01-retry01
```

| 固定件 | SHA256 |
| --- | --- |
| target1 父 manifest | `cd5b740f1f60b32bfb3b8ccf8ba2cfe84bb4ea9c650649810c07d9d9b3972184` |
| train | `24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184` |
| dev | `8b167df38022a631d4e631d3c0d66e9fca74171f4224fec436030d6650047303` |
| 固定八成员 payload | `6198beb3a3758fd7df912c6956bc05eac0ace8603708f37147826c65a4d61845` |
| source RGB | `a3b784da71eaa113fb4d9d71502a7a3526ba0d41e2d42ed96fe79111ca3dba65` |
| source FP32 latent | `719e92867b60546b21b281cfc633ab782c8ce2274bfb41c6b3cee6d673e74eaa` |
| event text | `f170a7e2dfe0070fbd160c09d29dbcf897ddbf5f75929a3ee4af84cf627965bb` |
| condition embeds | `473bd457d6fff070a71b119a19d950b8d094cfaf6f126ceb817330eb01263a60` |
| condition mask | `4f941a468150ea22f64ac4f7304e9a94a3dd1c721d07dd7f8ebd10185fbe2ea9` |
| DreamLite snapshot manifest | `1bcf41b170c4b4a806bac6701cbdf4fabd5c3c53fa67415d065ab95ce2703159` |
| Reader snapshot manifest | `159a504daaae6dc412535978f087150a0eb8e50164afd70a8a17f83906f1127c` |

Canonical teacher 仍为 `.../vision-language-memory-r11/r11-vae-latent-f4a018f-20260831/target-01/run/endpoint_raw.pt`，完整绝对路径在 JSON 中锁定。文件 SHA256 `d359291de63bb5232325b2e7a9294ff3d861287c06e63da2ab6ebe42eab036b9`，tensor SHA256 `6857afeffd37124bb196ab7c6607580c57c950d72d760ca6b49f8cc00bdef3f1`，FP32 shape `[1,4,128,128]`，population std `0.6546660661697388`。历史 teacher 4/4 不替代本轮 replay。

## 4. 实际数值语义和初始化证据

标称 effective sigma 保持 `[0.5,0.375,0.25,0.125]`。实际 sigma 必须来自锁定 scheduler setup；对标称值继续沿用 relative/absolute 各 `2e-6` 的已有容差。原始 metadata/payload/receipt 之间同一实测序列仍精确相等，不 round，不替换 nominal 数字。

```python
source_compute = source_latents_fp32.to(compute_dtype)
h0 = source_compute.mul(1 - actual_sigma).add(
    x_T_init_fp32.to(compute_dtype), alpha=actual_sigma
)
```

生产 `compute_dtype=torch.bfloat16`、`compute_device_type=cuda`。即使 source 与 x_T 相同，也不能假设浮点原生 mul-add 结果必然逐位等于 source；不能把实际 h0 替换成 source，也不能改成代数等价的其他运算次序。

首次完整四步前向前，保存 `source_only_initialization.pt`，schema 为 `vision_memory.r11-new-source-only-initialization.v1`。三个 tensor 字段严格为：

- `source_latents_fp32`；
- `x_T_init_fp32`；
- `reconstructed_start_state_compute`。

保存 formula、reconstruction operator、nominal/actual effective sigmas、actual sigma0、parameter/compute dtype、实际设备类型、shape、`initial_x_t_equals_source`、`initialization_teacher_assisted=false`、每 tensor SHA256，以及原始文件字节数/SHA256。具体字段在 JSON 的 `metadata_must_store` 锁定。初始化 artifact 不保存 teacher tensor 或其统计量。

独立核验不能信任布尔自声明：必须验证 source 固定哈希、重新取得 FP32 source、精确重建 x_T 与原生 h0，再与实际 trajectory point0 逐位绑定。生产 CUDA artifact 必须在 CUDA 上复算；缺少 CUDA 时拒绝核验，不回退 CPU、不放宽逐位合同。

旧 teacher-matched 的“h0 对 teacher NRMSE <=0.01”是旧初始化定义的技术检查，本轮明确不使用。替换为“初始 x_T 精确等于 source，且 h0 满足原生重构”的干预身份检查；**最终 endpoint 距离门完全不变**。

只有 step0 的 x_T 必须等于固定 source。后续 checkpoint 已优化，必须用各自当前 x_T、固定 source、实际 sigma 和锁定 backend 重建其 h0；不能要求 step64/128/192/256 仍等于 source。训练无 source 附近 trust region。

## 5. 优化、M0 和评测不变项

```text
loss = mean((full_four_step_endpoint.float() - teacher_fp32)^2)
trainable = x_T_fp32 only
optimizer = Adam; weight_decay = 0; clipping = none
optimizer calls = 256; full DreamLite denoising steps per forward = 4
checkpoint steps = [0,64,128,192,256]; primary endpoint = raw_step_256
u=1..128: lr=0.05
u=129..256: lr=0.025*(1+cos(pi*(u-128)/128))
global seed=0; strict determinism=true; DreamLite=cuda:0; Reader=cuda:1
```

第 256 次仍执行 forward/backward/Adam.step；学习率为零，因此参数差允许为零，但梯度必须 finite/nonzero，完整 Adam step counter 必须为 256。各 checkpoint 保存完整 Adam state，不以摘要 hash 替代数值核验。

本轮 M0 是新初始化下**未经优化但已完整运行四步 DreamLite 的 endpoint**，不是 source/h0/teacher，也不是任何旧臂 M0。M0 MSE 必须有限且大于零；不加 epsilon、不替换分母。本轮主要比值使用本轮 M0；跨臂比较只用相同 teacher 下的绝对 endpoint MSE 和 Reader CE。

Reader 不参与本轮优化。固定 reverse-cyclic 四视图的原始 ordered logits、target index、permutation 必须保存并逐行复算 CE/正确性。四视图是同一 target 的一致性检查，不是四个独立样本。保持 M0/endpoint 的 normal/reset 评测，不把 reset 更换为另一种 source。

## 6. Preflight、技术门和主要诊断门

先完成本地实现、单元测试、source-only import/fail-closed 检查，再用新提交执行固定 target1 technical-preflight：一次完整四步前向、一次反向、零 optimizer 更新。source/teacher/condition/model/hash 初始化和实际 h0 必须有效，只有 x_T FP32 trainable，梯度 finite/nonzero，全部模型参数冻结且无梯度。

Teacher replay 必须四个固定 reverse-cyclic 视图全部答对且 mean CE <=0.001。Preflight 的 `bridge_result_evaluated=false`；它不是诊断或科学成功。

正式技术门要求 256 条连续真实 receipts、每条四步和实际 LR 可复算、无裁剪、模型冻结、完整固定 checkpoint 集合与 tensor/PNG/trajectory/full Adam/condition/source/teacher 哈希绑定。初始化技术字段为 `source_only_initialization_artifact_valid`。Receipt 是更新前损失，checkpoint 是对应更新次数后的 endpoint；报告不得错位。

仅在技术门和本轮 teacher replay 有效后，raw256 距离门同时要求：

| 指标 | 固定门槛 |
| --- | ---: |
| endpoint MSE / 本轮完整四步 M0 MSE | <=0.01 |
| endpoint L2 / 本轮 M0 L2 | <=0.1 |
| endpoint RMSE / teacher population std | <=0.1 |

Reader transfer 要求 raw256 四视图全部正确。Mean CE 原样报告，但不新增 endpoint CE 主门。中间 checkpoint、best loss 和二级改善不得挽救主要失败。

任何技术门或 teacher replay 失败都不进入科学诊断决策树；保留无效证据并先修工程。本轮即使两主门通过，也只能称“source-only 初始化下 teacher-MSE bridge 诊断通过”。

## 7. 有界假设、二级审计与决策树

本轮可区分的操作性假设限定在此 target、此 teacher-MSE 目标、此 256 调用 solver：

- H1：source-only 初始 x_T 相比 Gaussian 父臂，使 raw256 绝对 MSE 与 Reader CE 两项均严格改善。
- H0：这两项没有同时严格改善。

这不是显著性检验，也不能把一次 H1 成立升级为关于全部初始化的定理。Teacher 仍提供稠密标签；H1 不证明 teacher 内容完全不重要，H0 不证明所有答案无关起点均无效。

二级审计仅在技术门、teacher replay 通过且距离/Reader 两主门都失败时参与解释：

```text
raw256 MSE < 0.09553645551204681
raw256 mean Reader CE < 25.536474171257463
```

两者都须严格小于；等值不算。NaN、Inf、负数、布尔值或非数值指标拒绝。Teacher-matched 参考臂数值只描述，不用于此判断。

| 距离门 | Reader 门 | 二级审计 | 本轮解释与下一候选 |
| --- | --- | --- | --- |
| 通过 | 通过 | 不参与 | 稠密监督下从答案无关起点恢复成立；另行预注册原 QA 目标的统一 solver 验证。 |
| 通过 | 失败 | 不参与 | 保留主要分支优先；另行测试已到达 teacher 邻域的 Reader 鲁棒性。 |
| 失败 | 通过 | 不参与 | 可读 endpoint 已找到但未满足距离合同；另行预注册原 QA 目标。 |
| 失败 | 失败 | 两项严格改善 | 此答案无关起点有帮助但不足；另行隔离一个 conditioning 干预。 |
| 失败 | 失败 | 未同时改善 | 仅本次 source-only 干预不足；另行隔离一个 conditioning 干预，不声称不可达。 |

下一候选均需要新预注册、新提交与 fresh root；本配置不授权追加训练、更换 target、预算、目标、初始化或门槛。

## 8. 回到 Phase 1A 仍缺的证据

本轮优化器仍可从 canonical teacher 的 dense loss 得到答案相关梯度。因此即使 initializer 不接收 teacher，也还没有证明“仅凭原 QA 监督即可找到答案无关初始化的可读 endpoint”。

返回主线必须另行预注册不依赖逐样本 canonical teacher 的统一 QA solver，在同一固定八成员、相同主计划 endpoint 和因果口径下完整验证。不得沿用旧六个成功样本，只换两个失败样本后拼成 8/8。当前 Phase 1A 仍为 6/8，失败成员仍为 1、7。

只有统一 solver 重新满足主计划 Phase 1A 8/8 门，才可讨论固定 64/128 的 Phase 2 MVP。之后共享 writer 的摊销可学习性还需 Phase 3A/3B；本轮不证明 state-level memory、held-out 泛化或长期 recurrence。

## 9. 执行与交付约束

只在指定 `vlm-r3-h200x2-live-20260717` 的 `/inspire/ssd/` 全新目录执行；保留旧结果不覆盖。实现提交先测试、推送，再独立 clean detached checkout；suite lock、防重复进程、固定环境和数据/模型 hash 检查保持。

每次 preflight/formal 保存配置与 commit、manifest、环境、初始化/condition artifact、真实 receipts、原始日志、固定 checkpoint/full Adam/trajectory/PNG、teacher replay 与 Reader 原始 rows、文件 SHA 清单、机器可读结果和 Markdown 报告。独立 aggregator 重建技术门和所有主要/二级结论，不信 trainer 自声明。

每轮先完成核验与 GitHub 交付再开展下一轮。本文不预填尚未发生的本轮训练 commit、运行根或结果。代码、JSON、本文冲突时停止阶段并报告，不擅自改科学定义。

原 30 小时规划时钟仍起于 `2026-09-05T07:18:59.891307+00:00`；不得重新起算、缩减最低 64 条或降低 solver/科学门追赶时间。实际累计耗时与修订 ETA 在真实运行报告中披露。
