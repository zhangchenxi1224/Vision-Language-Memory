# R11_new：Teacher-matched initialization 最小诊断预注册

日期：2026-09-06。协议：`R11-New-Canonical-Latent-Bridge-Teacher-Matched-Init-Target01`。

本文在本轮任何 teacher-matched 初始化前向或优化结果产生之前冻结。它是主计划失败分支下的一次最小诊断，不是新的 Phase 1A 成功声明，也不是 Phase 2 启动许可。

## 1. 唯一问题、唯一变量和结论边界

本轮只问：在相同 target、相同 canonical teacher、完整冻结的四步 DreamLite、相同稠密 latent MSE 目标和相同 256 次 Adam 调用下，把初始 flow state 设置在 teacher 附近，能否改善最终 endpoint 的经验恢复结果？

唯一改变的是 `x_T` 初始化：从事件键决定的 seed-0 标准高斯，改为从固定 teacher 与固定 source 反解出的确定性 FP32 初始化。优化器学习率曲线、调用次数、模型、数据、条件、teacher、主门和评测口径均不改变。

这里明确使用了答案相关的 canonical teacher。因此：

- 无论结果如何，`formal_success = false`，`phase2_allowed = false`。
- 若主门通过，只说明该 target 和该 solver 下，teacher 辅助初始化可以经验性地恢复可读 endpoint；不证明答案无关的 writer 已学会。
- `x_T` 在后续优化中没有局部 trust-region 约束。初始化接近 teacher，不意味着全程局部，也不能据此声称“局部可控性已证明”或“初始化是主导根因”。
- 若失败，只说明本次干预不足；不能证明数学不可达、模型没有容量或全部初始化都失败。
- 四个固定排列视图属于同一 target 的一致性评测，不是四个独立统计样本；本轮只有一个诊断 target。

主计划仍为 `reports/r11-new-frozen-dreamlite-oracle-training-plan-20260904.md`；本文及下述不可变 JSON 只定义这一次初始化诊断。不得借本分支改变主计划科学门槛或数据划分。若文档、JSON 与实现冲突，停止本阶段并报告。

## 2. 不可变配置及父证据

本轮配置：`configs/experiments/r11_new_canonical_latent_bridge_target01_teacher_matched_init.json`。

| 锁定项 | SHA256 / commit |
| --- | --- |
| 本轮 JSON 文件 SHA256 | `3168134145c8e715c9d134d00f25e2152c589d75730474c323dee1be2d13cf2f` |
| 本轮 JSON canonical SHA256 | `ecd84a874f282da48db5498e2db663d7b4f37b0bb4dd2f4f1500d1a4924ca3a7` |
| 父 post128-cosine 训练 commit | `16318e005b496a16b7712ad4ff3cea50e2be34fa` |
| 父交付 commit | `398cf72d7d18033f2fceb1a7dd1b7a5ccf5420e7` |
| 父 comparison SHA256 | `00aff581a78b988658fc868158131e5a61d441fcf90e639e2c268284b1f3adfe` |
| 父 RAW_ARTIFACTS SHA256 | `143de73292240812e148aed0acefe936d4de01e6d7eb1b83c7fef05b629d8e49` |
| 父配置 SHA256 | `c9794f5197f6c62f2f84af3cf0db9aee0ff225b4649004967f52d1424a247dc5` |

父 post128-cosine 结果根：

```text
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-new/r11-new-bridge-post128-cosine-16318e0-20260906-round01
```

父实验工程门和 teacher replay 通过，但距离门与 Reader 门均失败。其 raw step256：

| 指标 | 父实验实际值 |
| --- | ---: |
| endpoint MSE | 0.09553645551204681 |
| endpoint MSE / 父 M0 MSE | 0.7696052787944336 |
| step128 MSE / 父 M0 MSE | 0.7737632777116902 |
| endpoint Reader accuracy | 0/4 |
| endpoint Reader mean CE | 25.536474171257463 |

父实验的二级 schedule 审计通过，不曾挽救主要失败。本轮保留该学习率曲线，不把二级诊断通过描述为成功训练。

## 3. 固定 target、数据、teacher 和模型

固定 target 为 Phase 1A 两个失败成员中预先选择的最低序号 `target_index = 1`，segment 为 `r5-f1-392d41fd097d069c42218e0a`。target 7 继续暂缓；不能根据本轮结果重新挑 target。

Phase 1A 父来源：

```text
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-new/r11-new-phase1a-2cde77e-20260905-round02/target-01-retry01
```

这个路径由 core 的 `BRIDGE_PHASE1A_SOURCE_ROOT` 保留。新 JSON 未重复旧 `exact_parity_bindings` section，并不意味着这些未干预的因果锚点可以解除。

| 固定项 | 值 |
| --- | --- |
| Phase 1A 训练 commit | `2cde77ece6f020ab8c747d7c73e19dac4d8fba1b` |
| Phase 1A 修正交付 commit | `e2e60126285486d42386b4cd670b419334dd06db` |
| Phase 1A comparison SHA256 | `f3764a83ee8bc31cc7895fd33d494d4ea3be0c187c8cc9b0f6a85a4cf257843d` |
| Phase 1A RAW_ARTIFACTS SHA256 | `c1e36f92766c98f245917a9009dc9061a07729842dbef6c7b97ad18889edfbba` |
| 固定八成员 selected segments SHA256 | `6198beb3a3758fd7df912c6956bc05eac0ace8603708f37147826c65a4d61845` |
| target-01 父 manifest SHA256 | `cd5b740f1f60b32bfb3b8ccf8ba2cfe84bb4ea9c650649810c07d9d9b3972184` |
| train SHA256 | `24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184` |
| dev SHA256 | `8b167df38022a631d4e631d3c0d66e9fca74171f4224fec436030d6650047303` |
| blank source RGB SHA256 | `a3b784da71eaa113fb4d9d71502a7a3526ba0d41e2d42ed96fe79111ca3dba65` |
| source latent FP32 SHA256 | `719e92867b60546b21b281cfc633ab782c8ce2274bfb41c6b3cee6d673e74eaa` |
| event text SHA256 | `f170a7e2dfe0070fbd160c09d29dbcf897ddbf5f75929a3ee4af84cf627965bb` |
| condition prompt embeds SHA256 | `473bd457d6fff070a71b119a19d950b8d094cfaf6f126ceb817330eb01263a60` |
| condition attention mask SHA256 | `4f941a468150ea22f64ac4f7304e9a94a3dd1c721d07dd7f8ebd10185fbe2ea9` |
| DreamLite snapshot manifest SHA256 | `1bcf41b170c4b4a806bac6701cbdf4fabd5c3c53fa67415d065ab95ce2703159` |
| Reader snapshot manifest SHA256 | `159a504daaae6dc412535978f087150a0eb8e50164afd70a8a17f83906f1127c` |

固定 canonical teacher：

```text
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11/r11-vae-latent-f4a018f-20260831/target-01/run/endpoint_raw.pt
```

| Teacher 属性 | 值 |
| --- | --- |
| 文件 SHA256 | `d359291de63bb5232325b2e7a9294ff3d861287c06e63da2ab6ebe42eab036b9` |
| Tensor SHA256 | `6857afeffd37124bb196ab7c6607580c57c950d72d760ca6b49f8cc00bdef3f1` |
| shape / dtype | `[1, 4, 128, 128]` / `torch.float32` |
| population std | `0.6546660661697388` |
| 历史固定四视图 accuracy / mean CE | `1.0` / `0.000176956004` |

历史正确率不能替代本轮 teacher replay。Teacher 来自 canonical R11（直接优化最终 VAE latent），不是 R11_new（优化 initial latent 并运行完整 DreamLite）的 Phase 1A 成果。

## 4. 初始化的精确计算语义

使用纯文本代码公式，避免 Markdown 数学渲染依赖。

设 `source_latents_fp32` 为锁定 source latent，`teacher_fp32` 为锁定 canonical teacher。初始标称 sigma 为 `0.5`，但计算必须使用锁定 DreamLite scheduler setup 返回的第一个实际 effective sigma：

```python
actual_sigma = actual_effective_sigmas[0]
x_T_init_fp32 = (
    teacher_fp32 - source_latents_fp32.mul(1 - actual_sigma)
).div(actual_sigma)
```

反解发生在 FP32。不能先把 teacher 转 BF16 再反解，不能将 nominal `0.5` 硬编码进数值计算，也不能用 `round` 改写原始 sigma 或落盘张量。

标称 effective schedule 始终是 `[0.5, 0.375, 0.25, 0.125]`。实际浮点值可能是 `[0.4999999701976776, 0.375, 0.25, 0.1249999925494194]`。采用已经存在的 scheduler 合同：runtime 实测值对标称值使用 relative/absolute 各 `2e-6` 的容差；长度、有限性必须有效。配置和 manifest 的理想声明不改；同一原始 artifact 的 record 与 tensor payload 之间仍精确绑定，不用容差掩盖原始证据差异。

Sampler 起点必须按原生运算次序重建：

```python
source_compute = source_latents_fp32.to(compute_dtype)
start_compute = source_compute.mul(1 - actual_sigma).add(
    x_T_init_fp32.to(compute_dtype), alpha=actual_sigma
)
```

`compute_dtype` 保持父 DreamLite updater 的实际 dtype（本链路为 BF16），不切换精度。不能把上式替换成 `source + sigma * (x_T - source)`；实数代数等价不保证 BF16 舍入和逐位结果相同。

`x_T_init_fp32` 是唯一可训练变量。完整冻结 DreamLite 仍从上述 start state 跑完四个 denoising steps，然后得到最终 `z_t`。不允许绕过 DreamLite、直接返回 teacher，或把 teacher 作为最终 latent 注入解码器来伪造桥接通过。

有限精度含义：我们期望重建的起点接近 teacher，不声称逐位相等。初始 NRMSE 的固定计算为：

```text
start_RMSE = sqrt(mean((start_compute.float() - teacher_fp32)^2))
start_NRMSE = start_RMSE / 0.6546660661697388
preflight 要求：start_NRMSE <= 0.01
```

这里将实际 compute-dtype 起点转换回 FP32 与固定 FP32 teacher 比较，包含 BF16 量化与运算舍入误差。阈值在结果前锁定；超门必须停止，不得事后改阈值。

设备也属于数值复算合同。执行任务在 H200 上使用不含模型前向的纯张量玩具检查，观察到同一 FP32 反解在 CPU/CUDA 间的最大差为 `4.768e-7`，BF16 起点 mul→add 的最大差为 `0.015625`。这是跨设备数值语义的工程证据，不是 teacher-matched DreamLite 实验结果，也不是放宽科学门槛的理由。仅相同 dtype 和运算表达式不足以保证跨设备逐位一致。

训练计算保持不变；trainer 和独立 aggregator 的数值复算必须使用 artifact 绑定的实际 `compute_device_type`。生产产物绑定 CUDA，CPU 仅用于对应 CPU 单元测试；在实际计算设备上完成 FP32 反解和 compute-dtype 重构后，再移回 CPU 做 tensor SHA / `torch.equal` 比较。核验 CUDA 产物却缺少 CUDA 时，必须明确拒绝核验，不能静默改用 CPU、放宽逐位绑定，或把复算设备差异解释为科学失败。这不新增另一种初始化，也不改变 JSON 的 dtype、运算顺序或数值门槛。

## 5. 初始化证据和每 checkpoint 起点绑定

首次完整 DreamLite 前向之前，必须保存初始化 artifact，并至少保存：

- `source_latents_fp32`、`teacher_fp32`、`x_T_init_fp32`、`reconstructed_start_state_compute`。
- 实际 effective sigma schedule、第一个 actual sigma、parameter dtype、compute dtype、shape、运算顺序。
- 文件字节数与 SHA256、各原始 tensor 的 SHA256、独立复算的初始 NRMSE。

控制器和独立 aggregator 不能只相信 artifact 内部的自我声明。必须从锁定的 source、teacher、actual sigma 重新计算 FP32 反解和 mul→add 重构，再与原始 artifact 及真正前向的 trajectory point 0 精确绑定。首次优化前失败即关闭本轮。

`trajectory_point0_binding_valid_every_checkpoint` 的含义是：每个 checkpoint 均用该 checkpoint 自己的当前 `x_T_fp32`、固定 source 和同一实际 sigma 重构，并与该 checkpoint 的 trajectory point 0 对上。

只有初始 checkpoint 0 必须等于初始化 artifact 的 `x_T_init_fp32` / 起点，并满足 teacher 接近阈值。64、128、192、256 的 `x_T` 已经更新，不能要求它们与最初 xT 相等，也不能额外要求它们的起点始终靠近 teacher。这不是新增局部约束。

旧 Gaussian xT、旧初始 endpoint z_t 和旧 step128 优化前缀不再要求相同；这些正是初始化干预的下游结果。Source、event、condition、model、data、teacher 的固定 hash 和父 manifest 身份仍必须相同。

## 6. 保留的训练合同与新的 M0

训练目标不变：完整冻结 DreamLite 输出 `z_t` 与固定 `teacher_fp32` 的逐元素 FP32 MSE 均值。Reader 只用于固定 checkpoint 的评测，不进入本轮优化目标。

```text
loss = mean((z_t.float() - teacher_fp32)^2)
trainable = x_T_fp32 only
optimizer = Adam
weight_decay = 0
gradient_clipping = none
optimizer calls = 256
diffusion steps per training forward = 4
checkpoint steps = [0, 64, 128, 192, 256]
primary endpoint = raw_step_256
global determinism seed = 0
strict determinism = true
DreamLite device = cuda:0
Reader device = cuda:1
```

学习率完全继承父 post128-cosine。`u` 表示从 1 开始编号的 Adam 更新调用，而不是该调用前已有的更新数：

```text
u = 1..128:   lr(u) = 0.05
u = 129..256: lr(u) = 0.025 * (1 + cos(pi * (u - 128) / 128))
lr(256) = 0
```

第 256 次仍执行前向、反向和 `Adam.step()`；梯度必须有限且非零，而 xT 参数差可以为零。Checkpoint 的完整 Adam 状态必须可复算其 hash，并且实际 step counter 与 checkpoint 步数一致（step0 为尚未建立的空 optimizer state）。自洽 hash 不能替代计数检查。不得把相同零学习率的空参数更新夸大为增加有效优化预算。

本轮 M0 重新定义为：从新的 teacher-matched `x_T_init_fp32` 出发，在没有任何 Adam 更新时，完整运行四步 DreamLite 后的 endpoint。

```text
new_M0_z = FrozenDreamLite_four_steps(x_T_init_fp32, fixed_source, fixed_event)
new_M0_MSE = mean((new_M0_z.float() - teacher_fp32)^2)
endpoint_MSE_ratio = raw_step256_MSE / new_M0_MSE
```

M0 不是 trajectory point 0、不是 teacher 自身，也不是父 Gaussian 初始化的 M0。M0 必须有限且严格大于零；否则比值合同不成立，停止并报告，不自行加 epsilon 或更换归一化口径。

因此，本轮主门中的 MSE/M0 比值与父实验各自使用各自 M0，不能把两个比值直接当作固定分母的绝对改善。二级比较专门使用相同 teacher 和维度下的绝对 endpoint MSE 与 Reader CE。

## 7. 阶段门：preflight、技术门、诊断主门

### 7.1 先做 preflight，禁止直接启动 256 步

在新结果目录中，完成实现、定向 CPU 测试和 fail-closed 测试后，再执行单 target preflight：0 次 optimizer 更新、1 次完整四步前向、1 次反向。

要求 teacher 文件/tensor、source 和初始化 artifact 独立复算通过；初始 point0 NRMSE 不超过 `0.01`；仅 FP32 xT 可训练；xT 梯度有限且非零；所有模型参数冻结且无梯度；条件和模型 snapshot hash 不变。Teacher 必须在固定 reverse-cyclic 四视图下全部答对，mean CE 不超过 `0.001`。

Preflight 只判定工程路径可执行，`bridge_result_evaluated = false`。不得拿 preflight 的 teacher replay 或初始 loss 宣称 endpoint 诊断通过。

### 7.2 正式诊断技术门

必须有恰好 256 条连续 optimizer receipts，每条对应实际四步 DreamLite 和一次 Adam 调用，逐行复算学习率；梯度有限且非零；无裁剪；仅 xT 更新；模型 snapshot 不变。初始化 artifact、每 checkpoint 起点、条件、teacher、checkpoint 文件与 tensor/PNG/full Adam 状态均须独立绑定。

Checkpoint 集合严格等于 `[0, 64, 128, 192, 256]`。Receipt 是更新前训练损失，raw checkpoint 是对应次数更新后的实际 endpoint；报告和图表必须标明前/后坐标，不能错一位。

任何技术门或 teacher replay 失败，先报告工程/证据失败；不得将其归类为主要科学假设被否定。

### 7.3 仅 raw step256 的主要距离门与 Reader 门

在技术门及 teacher replay 都通过之后，距离门同时要求：

| 固定指标 | 通过条件 |
| --- | ---: |
| endpoint MSE / 新 M0 MSE | <= 0.01 |
| endpoint L2 distance / 新 M0 L2 distance | <= 0.1 |
| endpoint RMSE / teacher population std | <= 0.1 |

Reader transfer 门要求 raw step256 的固定 reverse-cyclic 四视图全部正确，accuracy 等于 `1.0`。其 mean CE 原样报告，但不新增或替换 endpoint CE 科学阈值。

四视图的 CE 与正确性必须从保存的原始有序四选项 logits、target index 和锁定排列逐行复算。中间 checkpoint、best train loss、偶然正确视图不能挽救 raw step256 主门失败。

这两个主门即使同时通过，也只称“本轮 teacher-assisted bridge 诊断通过”；始终不称完整 R11_new 训练成功。

## 8. 二级初始化审计与固定决策树

二级审计只有在技术门和 teacher replay 通过、且主要距离门和 Reader 门都失败时才有资格参与解释。必须同时满足以下两个严格改善：

```text
raw step256 absolute MSE < 0.09553645551204681
raw step256 Reader mean CE < 25.536474171257463
```

等于父值不算改善；只有 MSE 或只有 CE 改善不算二级通过。NaN、Inf、负数或非数值的指标必须拒绝。二级审计始终不是成功门，也不能修改主门状态。

| 距离门 | Reader 门 | 二级审计 | 本轮结论与下一项候选最小实验 |
| --- | --- | --- | --- |
| 通过 | 通过 | 不参与 | 该 teacher 辅助初始化下经验恢复成立；另行预注册答案无关的 learned initializer / inverse writer。 |
| 通过 | 失败 | 不参与 | 保持主要分支优先；另行测试已到达 teacher 邻域中的 Reader 鲁棒性。 |
| 失败 | 通过 | 不参与 | 保持主要分支优先；优先回到原始 QA 目标，另行预注册。 |
| 失败 | 失败 | 两项均严格改善 | Teacher 辅助改善 endpoint MSE 和 CE，但仍不足；另行隔离答案无关初始化或 conditioning 干预。 |
| 失败 | 失败 | 未同时改善 | Teacher-state matching 未解决桥接；另行预注册 reachable-teacher control，以区分优化失败与 teacher/flow 不匹配，不宣称不可达。 |

技术门或 replay 无效时不进入此表。下一项是需要新预注册的候选方向，不是本配置内追加实验授权。不得看到结果后更换初始化公式、预算、目标、threshold 或 target。

## 9. 保存、独立核验与部署约束

只在指定实例 `vlm-r3-h200x2-live-20260717`、SSD 结果目录和全新不覆盖的输出根运行。实现 commit 必须先推送，再使用 clean detached checkout；suite lock、配置文件 hash 和 canonical hash 均要校验。历史结果保持只读。

每次 preflight 与正式诊断保存：

- 原始 JSON 配置、代码 commit、manifest、父证据绑定、环境/软件版本、设备和模型快照信息。
- 初始化原始 artifact、condition artifact、teacher replay 原始 logits/rows。
- 原始 receipts、逐步日志、checkpoint tensor/trajectory/PNG、完整 Adam state、原始文件哈希清单。
- 机器可读结果、主要门与二级门、`formal_success=false`、`phase2_allowed=false`、明确失败原因和决策。
- Markdown 报告及独立 aggregator 的逐行/逐 checkpoint 复算结果。报告不得只读取 trainer 声明的布尔门。

每轮完成独立核验后，由主执行任务按用户授权提交和推送交付，记录实际 SHA。文档编写时不预填尚不存在的运行根、训练 commit 或实验结果；这些由锁定后的真实运行 manifest 填入。

## 10. 如何回到主线

当前 Phase 1A 的固定八成员结果仍是 6/8，失败成员仍为 1 和 7。本诊断不改变这组结果，也不能把 canonical R11 的 8/8 或 teacher 辅助的 target-01 恢复拼入新 Phase 1A。

若本轮证据支持新的求解策略，先为不依赖逐样本 canonical teacher 的统一 solver 另行预注册；再按同一 solver、同一固定八成员、同一主计划评测与因果控制完整验证。不能给成功的六个样本沿用旧 solver、只给两个失败样本换 solver 后拼接成 8/8。

只有重新达到主计划当前生效的 Phase 1A 门，才讨论已授权的 Phase 2 MVP 子集（64 或 128，按预算预先固定），以及 Phase 3 共享 writer 的可学习性。Phase 1B 暂缓和 Phase 2 不全量的用户修订，不等于解除 Phase 1A 门，也不允许将本 teacher-assisted 诊断直接升格为主线通过。

本轮最强可交付结论始终是一项可复现、因果变量受控的初始化诊断，而不是模型已经学会生成答案无关 memory state。
