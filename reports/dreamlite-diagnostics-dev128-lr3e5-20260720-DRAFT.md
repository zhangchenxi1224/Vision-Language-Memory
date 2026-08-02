# DRAFT — DreamLite checkpoint、梯度与低学习率诊断报告

> 状态：**待实验结果回填，不可作为最终结论引用**  
> 固定日期：2026-07-20（Asia/Shanghai）  
> 原正式训练：`dreamlite-fullscale-formal-v1-seed0-20260720-10bde56`  
> 本轮诊断根目录：`dreamlite-diagnostics-dev128-20260720-10bde56`

本骨架在结果产生前固定报告字段、比较方向和判定规则。所有写作中的 `[待回填]` 都必须来自原始 JSON/JSONL、terminal 或 SHA256 清单；不得根据日志片段、文件大小或预期行为补写数值。

## 1. 最终结论摘要（待回填）

| 问题 | 预注册证据 | 结果 | 判定 |
|---|---|---:|---|
| 256 步是否相对初始化产生可重复的学习改善？ | 固定 dev-128、四视图、semantic-group paired bootstrap | `[待回填]` | `[通过 / 未通过]` |
| step 256 的效果是否依赖写入后的 memory state？ | step256 `standard - reset` 对照 | `[待回填]` | `[通过 / 未通过]` |
| 巨大 raw gradient 来自哪些 LoRA 模块？ | 两档 one-step probe 的 pre/post-clip 分组范数 | `[待回填]` | `[定位 / 未定位]` |
| `3e-5` 是否真正缩小参数更新？ | fresh-Adam 配对 probe 的实际 update/weight ratio | `[待回填]` | `[通过 / 未通过]` |
| `3e-5` 的 64 步学习效果是否优于 `1e-4`？ | 相同 dev-128 外部四视图评估与配对 bootstrap | `[待回填]` | `[优于 / 不确定 / 更差]` |
| 是否支持继续扩大 DreamLite 训练？ | checkpoint 学习门、memory 门、低 LR 选择门联合 | `[待回填]` | `[继续 / 先修优化 / 停止扩量]` |

### 已知正式训练锚点

| 指标 | 已锁定结果 |
|---|---:|
| optimizer steps / episode presentations | 256 / 2048 |
| 训练耗时 | 3:07:14（trainer） |
| train loss：首 32 步均值 → 末 32 步均值 | 14.2365 → 13.9335（-2.13%） |
| train loss：首 32 步中位数 → 末 32 步中位数 | 13.7783 → 14.0529（+1.99%） |
| raw gradient clip rate | 256/256 = 100% |
| raw gradient 中位数 / 最大值 | 17,250.46 / 27,106,712 |
| 原训练唯一 dev loss（仅 8 episodes，不作为本轮 gate） | 10.08446 |

## 2. 不可变实验设计与指纹

### 2.1 代码、数据与模型

| 工件 | 路径或标识 | SHA256 / commit |
|---|---|---|
| 原正式训练代码 | 主 worktree | `10bde565d30d119a68e8460757d979b1c35e1b8f` |
| 诊断代码 | `Vision-Language-Memory-diagnostic-94bfc6c` | `94bfc6c532392d7c1304b42dea2ab2b867ce7345` |
| DreamLite nested revision | `third_party/DreamLite` | `a6e20c8cc94027f37dd7c5a81b0b3b472aa18409` |
| formal-v1 train（5000） | `data/.../formal-v1/train.jsonl` | `24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184` |
| formal-v1 dev（500） | `data/.../formal-v1/dev.jsonl` | `8b167df38022a631d4e631d3c0d66e9fca74171f4224fec436030d6650047303` |
| stratified dev-128 | `inputs/dev-stratified-128.jsonl` | `42691e61b9fdfa4068f3cab63d77889b38e25b5978de99ea89d99bed77763deb` |
| dev-128 manifest | 本地 `tools/dev-stratified-128.manifest.json` | `34a332f5a9b78149ad12377c8e81239abd52338fb382661035c3ec6c888219a0` |
| 32 个 semantic groups ID 集 | manifest 字段 | `50ee92130ab047fb3b17fcf07015f8e538c0d0602e73f3f5dafcf44b1e526198` |
| DreamLite snapshot manifest | 模型快照绑定 | `1bcf41b170c4b4a806bac6701cbdf4fabd5c3c53fa67415d065ab95ce2703159` |
| Qwen Reader snapshot manifest | 模型快照绑定 | `159a504daaae6dc412535978f087150a0eb8e50164afd70a8a17f83906f1127c` |
| 原训练 metrics | 本地证据包 `source/metrics.jsonl` | `6531cbe0c42232cbb8e97b1deec8f418129a39e60e2f55b808f204fb04feb40e` |

### 2.2 原训练 checkpoints

| checkpoint | SHA256 |
|---|---|
| step 64 | `9c5883565d9ad6705d9841d313468339cf1b29f35e9fff3953c0b2fec9163f8b` |
| step 128 | `75a1055e6cc792a893bee9cfc1d07ee954cf6ee6a25531edbc707b38f524307e` |
| step 192 | `845fe1d0dec5cfb56831050f85da2c7c4eaf630a00c8da8c31be83b43e6c5bb7` |
| step 256 | `3b648eccf9df05c7df1da35626c271d5805b4cd7c378e10f44650b20afd61880` |

### 2.3 固定 dev-128 设计

- 128 episodes，32 semantic groups，4 个 topic 各 8 groups / 32 episodes。
- 每个 condition 预期 332 个 base queries；每个 query 使用 reverse-cyclic4 的 4 个 choice views，因此每个 condition 预期 1,328 条 prediction records。
- checkpoint sweep 共 7 个 method-condition 组合，完整时预期 9,296 条 prediction records。
- 所有统计先将 4 个 views 聚合回 base query，再按 `semantic_group_id` 做 paired cluster bootstrap；不得把 4 个 views 当作独立样本。
- bootstrap 固定为 10,000 次，seed 2026。四个 `init → checkpoint` 的 listwise CE p 值使用 Holm 校正。

### 2.4 脚本与启动证据

| 工件 | SHA256 |
|---|---|
| `run_checkpoint_sweep.sh` | `cc2fa2a855ecbf179ad9fe543f0a5a9ea5fbe7cdd0e688c6460da62d5a6ed0bd` |
| `analyze_checkpoint_sweep.py` | `b40869c1c8303768e47797408cdb2e1caf62a47c60aacc0b76bd6c8ff4d1f7c0` |
| `run_lr3e5_short64_fromscratch_94bfc6c.sh` | `7e279990188f0542bc05fae0ad054670046b921574368f00af050d9668b7a491` |
| `run_lr3e5_step64_matched_eval_94bfc6c.sh`（v1；raw query 预检失败，原样保留） | `17b0070db39d72dbc47ee8faf6521e785a1ee7797b17de9455fd89796cf05374` |
| `run_lr3e5_step64_matched_eval_94bfc6c-v2.sh` | `41bcc45ddad063db85c13164dfa7a06b2fd694afc3b62c40f1f758e02b67a7c5` |
| checkpoint-sweep launcher configuration | `daaa51d73d07e79288830d7e18b72af3b60467a3cf54f5cb0f83e30d960f5e7c` |
| 诊断 preflight v3 | `[待回填；只接受 passed=true 的 v3 与其 .sha256]` |

## 3. Checkpoint sweep

### 3.1 覆盖与完整性

| method | condition | prediction records | base queries | episodes | semantic groups | 输入文件 SHA256 | evaluator report SHA256 |
|---|---|---:|---:|---:|---:|---|---|
| `init_seed0` | standard | `[待回填；预期 1328]` | `[预期 332]` | `[预期 128]` | `[预期 32]` | `[待回填]` | `[待回填]` |
| `init_seed0` | reset | `[待回填；预期 1328]` | `[预期 332]` | `[预期 128]` | `[预期 32]` | `[同 init 文件或待回填]` | `[待回填]` |
| `step_064` | standard | `[待回填；预期 1328]` | `[预期 332]` | `[预期 128]` | `[预期 32]` | `[待回填]` | `[待回填]` |
| `step_128` | standard | `[待回填；预期 1328]` | `[预期 332]` | `[预期 128]` | `[预期 32]` | `[待回填]` | `[待回填]` |
| `step_192` | standard | `[待回填；预期 1328]` | `[预期 332]` | `[预期 128]` | `[预期 32]` | `[待回填]` | `[待回填]` |
| `step_256` | standard | `[待回填；预期 1328]` | `[预期 332]` | `[预期 128]` | `[预期 32]` | `[待回填]` | `[待回填]` |
| `step_256` | reset | `[待回填；预期 1328]` | `[预期 332]` | `[预期 128]` | `[预期 32]` | `[同 step256 文件或待回填]` | `[待回填]` |

完整性 gate：7 个组合都达到固定计数；每个 base query 的 view index 必须恰为 `[0,1,2,3]`；所有 pair key 完全一致；stage terminal 为 `succeeded / passed=true / exit_code=0`；`artifacts.sha256` 全部校验通过。任一失败时，不计算或引用科学 gate。

### 3.2 各 checkpoint 绝对指标

| method:condition | accuracy ↑ | listwise CE mean ↓ | listwise CE median ↓ | correct margin mean ↑ | view consistency ↑ |
|---|---:|---:|---:|---:|---:|
| `init_seed0:standard` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| `init_seed0:reset` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| `step_064:standard` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| `step_128:standard` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| `step_192:standard` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| `step_256:standard` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| `step_256:reset` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |

另附三个分层表：`topic`、`probe_role`、`subtype`。每个 stratum 必须报告 queries、accuracy、listwise CE、correct margin；小样本 stratum 只用于定位，不独立作显著性结论。

### 3.3 初始化到 checkpoint 的配对差值

差值方向固定为 `checkpoint - init_seed0`。因此 CE 为负、accuracy 为正、margin 为正表示改善。

| comparison | Δ CE [95% CI] | raw p | Holm-adjusted p / reject | Δ accuracy [95% CI] | Δ margin [95% CI] | Δ view consistency [95% CI] |
|---|---:|---:|---|---:|---:|---:|
| init → step 64 | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| init → step 128 | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| init → step 192 | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| init → step 256 | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |

### 3.4 Reset controls 与固定 gate

`init reset → standard` 与 `step256 reset → standard` 的差值方向都为 `standard - reset`。step256 memory gate 要求 standard 明确优于 reset。

| comparison | Δ CE [95% CI] | Δ accuracy [95% CI] | Δ margin [95% CI] | Δ view consistency [95% CI] |
|---|---:|---:|---:|---:|
| init：standard - reset | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| step256：standard - reset | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |

| Gate | 预注册规则 | 结果 |
|---|---|---|
| step256 学习证据 | `ΔCE CI95 upper < 0` 且 `Δaccuracy >= 0` 且 `Δmargin > 0` | `[待回填]` |
| step256 memory 依赖 | step256 `standard-reset` 满足同一规则 | `[待回填]` |
| analyzer `continue_training_supported` | 上述两 gate 同时通过 | `[待回填]` |
| 多重比较稳健性标签 | step256 CE 的 Holm `reject=true`；此标签单独报告，不偷改 analyzer 主 gate | `[待回填]` |

## 4. 两档 one-step optimizer probe

### 4.1 固定配对设计

两臂都从原 `checkpoint-000256.pt` 初始化，使用相同 seed、相同前 8 个 train episodes、gradient accumulation 8、global clip 1.0、fresh AdamW，只改变 learning rate：`1e-4` 与 `3e-5`。启用：

- `--audit-state-gradients`
- `--audit-gradient-sha`
- `--audit-optimizer-diagnostics`

建议固定输出目录（实际路径若不同，必须在此替换并记录）：

```text
<diagnostic-root>/outputs/one-step-probe-lr1e4-from-step256-94bfc6c
<diagnostic-root>/outputs/one-step-probe-lr3e5-from-step256-94bfc6c
```

### 4.2 全局配对结果

| 指标 | lr=1e-4 | lr=3e-5 | 配对关系 / 比值 |
|---|---:|---:|---:|
| train loss / `loss_hex` | `[待回填]` | `[待回填]` | `[应完全一致；待验证]` |
| raw gradient SHA256 | `[待回填]` | `[待回填]` | `[应完全一致；待验证]` |
| clipped gradient SHA256 | `[待回填]` | `[待回填]` | `[应完全一致；待验证]` |
| raw global grad norm | `[待回填]` | `[待回填]` | `[应一致；待验证]` |
| post-clip global grad norm | `[待回填]` | `[待回填]` | `[待回填]` |
| clipping factor | `[待回填]` | `[待回填]` | `[待回填]` |
| parameter norm before step | `[待回填]` | `[待回填]` | `[应一致；待验证]` |
| actual update norm | `[待回填]` | `[待回填]` | `U(3e-5)/U(1e-4)=[待回填]` |
| global update/weight ratio | `[待回填]` | `[待回填]` | `R(3e-5)/R(1e-4)=[待回填]` |
| state-gradient audit | `[待回填]` | `[待回填]` | `[待回填]` |
| elapsed seconds / peak VRAM | `[待回填]` | `[待回填]` | 描述性 |

### 4.3 分组梯度与更新

分别从 `gradient_norms_before_clip`、`gradient_norms_after_clip`、`updates_after_step` 提取以下表。每个 axis 内的 `squared_norm_share` 应求和约等于 1；`missing_tensor_count` 必须为 0。

| arm | axis | group | pre-clip norm | pre-clip squared share | post-clip norm | update norm | update/weight ratio |
|---|---|---|---:|---:|---:|---:|---:|
| 1e-4 | stage | down / mid / up / other | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| 3e-5 | stage | down / mid / up / other | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| 1e-4 | projection | q / k / v / out / other | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| 3e-5 | projection | q / k / v / out / other | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| 1e-4 | factor | LoRA A / LoRA B / initial state / other | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| 3e-5 | factor | LoRA A / LoRA B / initial state / other | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |

另列 `cross = stage|projection|factor` 的 pre-clip squared share 前 10 项，避免只看大模块掩盖单一投影异常。

### 4.4 State-gradient audit

QA objective 的必须类别是 `final_state`、`query_image`；若 episode 有至少两次 update，还必须有 `first_intermediate_state`。每类回填 expected、observed、positive_finite、min/max norm 与 `norm_payload_sha256`。

| arm | category | expected | observed | positive finite | min norm | max norm | payload SHA256 |
|---|---|---:|---:|---:|---:|---:|---|
| 1e-4 | final_state | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| 1e-4 | query_image | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| 1e-4 | first_intermediate_state（若适用） | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| 3e-5 | 同上 | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |

### 4.5 Probe gates

| Gate | 固定规则 | 结果 |
|---|---|---|
| lineage / execution | 两臂 commit、parent checkpoint SHA、data SHA、8 episodes、1 optimizer step 均正确；terminal 成功 | `[待回填]` |
| paired-gradient identity | `loss_hex`、raw gradient SHA、clipped gradient SHA 两臂完全相同 | `[待回填]` |
| diagnostic integrity | schema v1；pre-clip diagnostic 与 PyTorch 返回值匹配；所有 global/group 值有限；missing=0 | `[待回填]` |
| post-clip bound | global post-clip norm `<= 1.005`（兼容 BF16 5e-3 检查容差） | `[待回填]` |
| state-gradient connectivity | `state_gradient_audit.passed=true`，所有 required 类别 expected=observed=positive_finite>0 | `[待回填]` |
| lower-LR update contraction | `update_norm(3e-5) < update_norm(1e-4)` 且 global update/weight ratio 同向降低 | `[待回填]` |

解释边界：若梯度 SHA 完全一致且两臂都被 clip，降低 LR 只改变 AdamW 参数步幅，不能被表述为“修复了梯度爆炸”；它最多证明实际更新幅度受控。若梯度 SHA 不一致，则两臂不再是合法的单变量配对，不能解释 LR 效应。

## 5. `lr=3e-5` 从头 64 步短臂

### 5.1 运行契约

| 项目 | 固定值 / 待回填 |
|---|---|
| 输出目录 | `<diagnostic-root>/outputs/lr3e5-fromscratch-step64-seed0-94bfc6c` |
| commit | `94bfc6c532392d7c1304b42dea2ab2b867ce7345` |
| 初始化 | from scratch；`parent_checkpoint_sha256=null` |
| train / dev | formal-v1 train 5000 / stratified dev-128 |
| seed / adapter seed | 0 / 0 |
| optimizer | AdamW，LR `3e-5`，WD `0.01` |
| max steps / grad accumulation | 64 / 8（512 episode presentations） |
| checkpoint | 16、32、48、64；另有 best/last |
| internal dev | step 64，128 episodes；仅作训练器完整性与辅助观测 |
| optimizer diagnostics | 64/64 train records 必须存在 |
| terminal / elapsed | `[待回填]` |

### 5.2 训练与优化统计

| 指标 | 原 lr=1e-4 steps 1–64 | 新 lr=3e-5 steps 1–64 | 解释 |
|---|---:|---:|---|
| loss mean / median / SD | `[待提取]` | `[待回填]` | 同 seed/order 可逐 step 配对 |
| 首 16 步 loss mean / median | `[待提取]` | `[待回填]` | 早期 |
| 末 16 步 loss mean / median | `[待提取]` | `[待回填]` | 末端 |
| OLS slope / Pearson r | `[待提取]` | `[待回填]` | 描述性 |
| raw grad median / p95 / max | `[待提取]` | `[待回填]` | 重尾程度 |
| clip rate | `[待提取]` | `[待回填]` | 是否仍由 clip 主导 |
| clipping factor median / p05 | `旧 run 无该字段` | `[待回填]` | 新诊断字段 |
| update norm median / p95 | `旧 run 无该字段` | `[待回填]` | 新诊断字段 |
| update/weight median / p95 | `旧 run 无该字段` | `[待回填]` | 新诊断字段 |
| internal dev loss at 64 | `不可直接对应旧 run` | `[待回填]` | 不用于 head-to-head 科学 gate |
| elapsed / peak VRAM | `[待提取]` | `[待回填]` | 工程指标 |

必须另报告 64 步中 `stage / projection / factor / cross` 的 squared-share 中位数及最大值，定位异常是偶发单步还是长期集中。

### 5.3 必须补齐的相同 evaluator 外部评估

训练器的 internal dev loss 与 checkpoint sweep 的四视图 MCQ 汇总不是同一工件，不能直接比较。要回答 `3e-5` 是否优于 `1e-4`，必须对新 `checkpoint-000064.pt` 再运行同一个 `dreamlite_mcq.py`：

```text
episodes=同一 dev-stratified-128.jsonl
choice-view-family=reverse-cyclic4
conditions=standard,reset
method=lr3e5_step_064
seed/adapter-seed/training-seed=0
strict-determinism=true
```

固定输出与 stage（两者都必须在 launch 前不存在）：

```text
<diagnostic-root>/outputs/lr3e5-step64-matched-eval-94bfc6c-v2
<diagnostic-root>/stages/lr3e5-step64-matched-eval-94bfc6c-v2
```

v1 错把原始 JSONL 中显式 `type=query` turns 的数量预期写成 332，实际为 256，因而在 GPU 模型加载前 fail closed；v1 stage、日志和 runner 均保留且不覆盖。v2 与 v1 仅有两处语义差异：raw turn 断言 `332→256`，以及 output 后缀增加 `-v2`。evaluator 构造后的 332 base queries、standard/reset 各 1,328 行、总计 2,656 行的输出契约保持不变。

v2 runner 在模型加载前校验：诊断 worktree commit/clean、dev SHA/128 行/32 groups/256 个显式 query turns、上游短臂完整 `artifacts.sha256`、checkpoint SHA 与 checkpoint 内嵌 manifest（64 steps、`qa_only`、from-scratch、LR `3e-5`）。完成后生成 `audit.json`、`artifacts.sha256` 与旁置 manifest SHA。预计用时 **30–45 分钟**。

在 runner 与其 `.sha256` sidecar 已同步到 `<diagnostic-root>/inputs/` 后，精确 launch 命令为：

```powershell
wsl.exe -d Ubuntu -- /home/zhangchenxi/.local/bin/inspire notebook exec vlm-dreamlite-full-h200x2-20260720 --timeout 120 "cd /inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r3/dreamlite-diagnostics-dev128-20260720-10bde56/inputs && sha256sum --check --strict run_lr3e5_step64_matched_eval_94bfc6c-v2.sh.sha256 && /inspire/ssd/project/exploration-topic/czxs26210936/envs/vlm-r3-ngc2502/bin/python /inspire/ssd/project/exploration-topic/czxs26210936/Vision-Language-Memory-diagnostic-94bfc6c/scripts/inspire/launch_background.py --repo /inspire/ssd/project/exploration-topic/czxs26210936/Vision-Language-Memory-diagnostic-94bfc6c --run-root /inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r3/dreamlite-diagnostics-dev128-20260720-10bde56 --run-dir /inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r3/dreamlite-diagnostics-dev128-20260720-10bde56/stages/lr3e5-step64-matched-eval-94bfc6c-v2 --stage lr3e5-step64-matched-eval-94bfc6c-v2 --expected-commit 94bfc6c532392d7c1304b42dea2ab2b867ce7345 --preflight /inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r3/preflight/dreamlite-diagnostics-h200x2-20260720-94bfc6c-v3.json -- /bin/bash /inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r3/dreamlite-diagnostics-dev128-20260720-10bde56/inputs/run_lr3e5_step64_matched_eval_94bfc6c-v2.sh"
```

该命令只允许执行一次：stage 或 output 任一已存在时均 fail closed，不覆盖失败或部分结果。

若本轮时间内没有该工件，最终报告必须明确写“低 LR 只有优化诊断，没有同口径性能结论”，不得用 internal dev loss 代替。

### 5.4 同口径性能比较与 gate

所有差值仍定义为 `right - left`。

| comparison | Δ CE [95% CI] | Δ accuracy [95% CI] | Δ margin [95% CI] | Δ consistency [95% CI] | 判定 |
|---|---:|---:|---:|---:|---|
| init → lr3e5 step64 | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| lr1e4 step64 → lr3e5 step64 | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |
| lr3e5 step64 reset → standard | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` | `[待回填]` |

| Gate / 分类 | 固定规则 |
|---|---|
| 低 LR 学习 gate | init→lr3e5 step64：`ΔCE CI upper < 0`、`Δaccuracy >= 0`、`Δmargin > 0` |
| 低 LR memory gate | lr3e5 step64 `standard-reset` 满足同一规则 |
| 低 LR 显著优于 1e-4 | lr1e4→lr3e5：`ΔCE CI upper < 0`、`Δaccuracy >= 0`、`Δmargin > 0` |
| 低 LR 明确更差 | lr1e4→lr3e5：`ΔCE CI lower > 0`，或 accuracy 的整个 CI 小于 0 |
| 不确定 | 不满足“显著优于”也不满足“明确更差”；只能进入一次更长的确认臂，不能宣称胜出 |
| 可替换默认 LR | execution + diagnostic integrity + lower-LR update contraction + 低 LR 学习 gate + memory gate + 显著优于 1e-4 全部通过 |

## 6. 联合解释与下一步决策（待结果后填写）

| 观测组合 | 下一步 |
|---|---|
| checkpoint 学习门和 memory 门都通过；3e-5 显著优于 1e-4 | 采用 `3e-5`，下一轮优先做完整 5000 presentations 或第二 seed；仍保留固定 dev gate |
| checkpoint 门通过；3e-5 与 1e-4 不确定，但实际 update 明显收缩 | 只做一个 128–256 步的配对确认臂，不恢复全矩阵 |
| 学习门通过但 memory 门失败 | 当前提升可能来自 reader/fixture 偏置，先修 memory causal control，不扩量 |
| 学习门失败且长期 100% clip | 先查 loss 尺度、特定 LoRA 投影与 recurrence 路径；不延长训练 |
| one-step 梯度集中在单一 stage/projection/factor | 针对该组做 hook / normalization / loss decomposition 微诊断 |
| one-step 配对 identity 失败 | 判为实验无效，修复 seed/data/config 配对后重跑 probe |

最终建议必须分别回答：

1. 是否已有“学到东西”的统计证据；
2. 是否是 memory-dependent 的证据；
3. 梯度异常属于全局尺度还是局部模块；
4. `3e-5` 是仅缩小更新，还是同时改善同口径 dev 表现；
5. 下一轮是扩 presentations、增 seed，还是先修优化。

## 7. 远端与本地证据路径

### 7.1 远端 source of truth

```text
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r3/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r3/dreamlite-diagnostics-dev128-20260720-10bde56
```

关键子目录：

```text
<diagnostic-root>/inputs/
<diagnostic-root>/stages/checkpoint-sweep/
<diagnostic-root>/outputs/checkpoint-sweep/
<diagnostic-root>/outputs/one-step-probe-lr1e4-from-step256-94bfc6c/     # 建议名
<diagnostic-root>/outputs/one-step-probe-lr3e5-from-step256-94bfc6c/     # 建议名
<diagnostic-root>/outputs/lr3e5-fromscratch-step64-seed0-94bfc6c/
<diagnostic-root>/outputs/lr3e5-step64-matched-eval-94bfc6c-v2/         # 必需的同口径补评
```

### 7.2 本地归档目标

```text
D:\2026WorkExperience\VisonLearnableMemory\runs\formal_reports\dreamlite-diagnostics-dev128-20260720-10bde56\
D:\2026WorkExperience\VisonLearnableMemory\reports\dreamlite-diagnostics-dev128-lr3e5-20260720.md
```

当前旧报告保持不变：

```text
D:\2026WorkExperience\VisonLearnableMemory\reports\dreamlite-fullscale-formal-v1-seed0-20260720-10bde56.md
```

## 8. 最终归档证据清单

### 每个 background stage

- [ ] `launcher-config.json` 或等价 configuration 工件及 SHA256
- [ ] `runner.sh` / 输入脚本及 SHA256
- [ ] `stdout.log`、`stderr.log`
- [ ] `terminal.json`：`succeeded`、`passed=true`、`exit_code=0`
- [ ] 完成 sentinel 与其 SHA256（若启动器产生）
- [ ] preflight JSON 与旁置 `.sha256`；只归档通过的 v3 为有效前置证据

### Checkpoint sweep

- [ ] `init.jsonl`、`step-064.jsonl`、`step-128.jsonl`、`step-192.jsonl`、`step-256.jsonl`
- [ ] 每个 evaluator `.report.json`
- [ ] `analysis.json` 与 SHA256
- [ ] `analysis.json` 中的 `inputs[]` SHA 与实际文件一致
- [ ] `artifacts.sha256` 全量通过
- [ ] 7 条 summaries、4 条 checkpoint contrasts、2 条 reset contrasts、Holm 表、primary gate 均存在

### 两档 one-step probe

- [ ] 两臂 `manifest.json`、`metrics.jsonl`、`summary.json`、`state_gradient_audit.json`
- [ ] parent checkpoint SHA 固定为 step256 SHA
- [ ] 每臂恰 1 条 train record、group episode count=8、optimizer step=1
- [ ] raw/clipped gradient SHA 与 optimizer diagnostics 完整
- [ ] checkpoint/last 与 `artifacts.sha256`
- [ ] 两臂配对 identity 校验结果（机器可读 JSON，建议新增）

### lr3e5 64-step arm

- [ ] `manifest.json` 显示 clean commit、正确数据 SHA、parent checkpoint 为 null
- [ ] `metrics.jsonl` 恰 64 条 train + 1 条 internal dev；每条 train 有 optimizer diagnostics
- [ ] checkpoints 16/32/48/64、best、last
- [ ] `summary.json`、terminal、`artifacts.sha256`
- [ ] 新 step64 的外部 reverse-cyclic4 standard/reset predictions、reports、SHA 清单
- [ ] 与 init、原 lr1e4 step64 的 paired cluster bootstrap 工件

### 本地证据包

- [ ] 保留远端相对目录结构，不覆盖原正式 run 证据包
- [ ] 生成顶层 `artifacts.sha256`，随后在本地重新校验
- [ ] 报告中的每个数值可追溯到具体 JSON path 或 CSV 列
- [ ] 最终报告移除 `DRAFT` 标识前，全文搜索并清零所有 `[待回填]`、`[待提取]`、`[预期 ...]`

## 9. 回填来源速查

| 报告字段 | 唯一允许来源 |
|---|---|
| checkpoint 绝对指标 / contrasts / gate | `outputs/checkpoint-sweep/analysis.json` |
| sweep 输入 hashes | `analysis.json.inputs[]` + `artifacts.sha256` |
| probe raw/post grad、update ratio、分组 | 各 probe `metrics.jsonl[train].optimizer_diagnostics` |
| probe gradient identity | 各 probe train record 的 `loss_hex`、`raw_gradient_sha256`、`clipped_gradient_sha256` |
| state connectivity | `state_gradient_audit.json` 和 train record 的 group audit |
| 64-step loss/gradient/throughput | 新旧 `metrics.jsonl`；旧证据包 SHA 已锁定 |
| 64-step summary/VRAM/time | `summary.json` 与 stage `terminal.json` |
| lr head-to-head 性能 | 新 step64 外部 evaluator predictions 的配对分析；不得使用 internal dev loss 替代 |
| 路径、commit、data/model lineage | `manifest.json`、preflight、terminal 与 SHA 清单 |
