# R11_new Phase 0 实现与部署就绪审计

> 审计时间：2026-09-04T18:09:14Z
> 分支：`codex/r11-new-frozen-dreamlite-oracle-20260904`
> Git 基线：`47cebc97e59ffed00571d4bb67fbb08933b3f8d6`
> 目标实例：`vlm-r3-h200x2-live-20260717`
> 状态：训练提交形成前的结果无关审计；尚未启动任何 R11_new 模型运行

## 1. 三层结论

| 层级 | 当前结论 | 可解释范围 |
| --- | --- | --- |
| 工程就绪 | 通过 | Phase 0/1A 的 trainer、控制器、独立聚合器、机器配置、单元测试与部署前检查已形成闭环。 |
| 机制/诊断通过 | 未评估 | 尚未执行真实 Frozen-DreamLite backward，也没有任何 Phase 1A endpoint 结果。 |
| 科学成功 | `false` | 尚未训练共享 writer；未验证 state-level、held-out、多 seed、递归或完整因果门。 |

本报告不能被解读为 Phase 1A 成功，更不能用 canonical R11 的 8/8 结果替代 R11_new。

## 2. 唯一实验定义与结果前修订

唯一实验规范为：

- `reports/r11-new-frozen-dreamlite-oracle-training-plan-20260904.md`
- 机器镜像：`configs/experiments/r11_new_frozen_dreamlite_oracle_phase1a.json`

本轮在任何 R11_new 模型结果产生前锁定以下路线：

1. 当前链路为 Phase 0 -> Phase 1A -> Phase 2 -> Phase 3A -> Phase 3B；Phase 1B 延后，不阻塞 MVP。
2. Phase 1A 仍是 8 个固定 train-derived F1 target、完整冻结 DreamLite、仅优化 FP32 `x_T`、Adam 0.05、256 steps、固定 raw endpoint 与原 8/8 query-level gate；门槛未因 MVP 收缩而改变。
3. Phase 2 不做全量 train split。固定 `bank64`，仅当结果无关时间公式预计总体仍在约 30 小时规划窗口内时扩展为 `bank128`。
4. Phase 1A 的八个 target 是 R10 固定哈希顺序的前八项，直接作为 Phase 2 `calibration8`，不重复求解。
5. 最终 64/128 条必须全部通过 technical gate 与固定 query-level oracle gate；失败项不得替补，也不得只挑成功项进入 Phase 3。
6. Phase 3 只允许报告锁定训练子集上的 shared-writer learnability，不得声称完整训练集、泛化、state-level 或 long-term memory 成功。

固定 train F1 候选数为 7,504，且 `segment_id` 唯一。结果前复算绑定如下：

| 集合 | ID SHA-256 | Full payload SHA-256 |
| --- | --- | --- |
| calibration8 | `08c25bbb753e7ffb3a0fd760d0bbf079b113f1db12be9eba4af1505ad57e86ff` | `6198beb3a3758fd7df912c6956bc05eac0ace8603708f37147826c65a4d61845` |
| bank64 | `5cbd99fdc537f67cba311ed39144516735d0da149e87095565118b162a872fcc` | `d7d3a3d12182fd3169c5b9b5127617f9c1c5b81462a94c2d8afccb256973d98a` |
| bank128 | `c762b52c2b71ba8b977b6bec339a9586ef000440021c8bdf38bef28006d99f37` | `6b817ffdc488df1925294aa6169e8d33cb738877432fb9da46b2844aec6a3665` |

## 3. 实现审计

新增实现分工如下：

- `src/vision_memory/training/r11_new_oracle.py`：不可变配置镜像、Phase 1A schedule 与可复算 gates。
- `scripts/train/r11_new_frozen_dreamlite_oracle.py`：完整 frozen DreamLite -> VAE -> Reader 链路，仅 `x_T_fp32` 可训练。
- `scripts/inspire/run_r11_new_phase1a_target.py`：实例、Git、环境、数据、模型、存储、fresh root、suite lock 与 prerequisite 的 fail-closed 控制器。
- `scripts/experiments/compare_r11_new_phase1a.py`：从 raw receipts、evaluation rows、checkpoint tensors、hashes 与 prerequisite 链独立复算 8-target 结果。

关键边界已经代码化：

- canonical R11 直接优化最终 VAE latent；R11_new 优化 initial latent 并执行完整四步 frozen DreamLite。两者入口和结论不可互换。
- technical-preflight 固定为 1 次完整 forward/backward、0 optimizer step，不计算 reachability gate。
- preflight 独立核验有限且非零 `x_T` 梯度、五点 trajectory、四步 sigma、冻结模块、condition、checkpoint、snapshot、terminal 与 inventory。
- formal target 0 必须绑定通过的 preflight；target 1--7 还必须绑定本轮 formal target 0。
- formal 技术失败强制 `diagnostic=None` 且非零退出；诊断 gate 失败但技术有效仍保留为有效科学诊断。
- 聚合器重新验证共同 preflight 与本轮 target 0，不能只信任 trainer/controller summary。

## 4. 本地验证证据

安全定向回归：

```text
66 passed in 43.54s
```

覆盖 core、trainer、controller、aggregator、Differentiable DreamLite sampler 与 canonical R11 oracle 相邻契约。

静态验证：

```text
ruff: All checks passed
py_compile: passed
trainer/controller/aggregator --help imports: passed
git diff --cached --check: passed
Markdown UTF-8: passed; display-math markers balanced
```

额外旧回归组结果为 `67 passed, 2 skipped, 1 failed`。唯一失败发生在旧 `dreamlite_episode.make_manifest` 测试读取本机未安装的 `diffusers` package metadata；它不是 R11_new 断言失败，也不修改远端已锁定训练环境。两个 skip 分别是本机缺少双 CUDA 与 Windows symlink 权限。正式运行仍由远端环境、模型 manifest 与 preflight 再次 fail-closed 验证。

未运行 `tests/test_r3_poststart_sequence.py`：该旧测试会在 Windows Codex 桌面执行器中触发进程令牌故障，与 R11_new 代码无关；不将其记作通过。

## 5. 远端只读部署审计

最近一次完整只读核验显示：

- 实例 `vlm-r3-h200x2-live-20260717` 为 RUNNING，2 x H200；两卡空闲，无 compute application。
- 无 R11_new 训练进程；`/tmp/vision-memory-r11-new-phase1a.lock` 不存在。
- `/inspire/ssd` 可用约 1.51 TiB，高于 50 GiB fail-closed 门槛。
- 尚无 R11_new source checkout、run root、summary 或 terminal，不存在覆盖旧结果的问题。
- 固定 train/dev SHA、DreamLite/Reader manifest SHA 与 canonical R11 comparison SHA 已与机器配置匹配。

固定父件：

- train SHA-256：`24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184`
- dev SHA-256：`8b167df38022a631d4e631d3c0d66e9fca74171f4224fec436030d6650047303`
- DreamLite manifest SHA-256：`1bcf41b170c4b4a806bac6701cbdf4fabd5c3c53fa67415d065ab95ce2703159`
- Reader manifest SHA-256：`159a504daaae6dc412535978f087150a0eb8e50164afd70a8a17f83906f1127c`
- canonical R11 comparison SHA-256：`f8b048f9cbe9fd4df9460043297904b5c9d476f386d6844d12fd4a5f8f636bb5`

## 6. Go / No-Go

本地实现与结果无关协议满足提交条件。下一步顺序固定为：

1. 提交并推送本报告、规范、配置、实现和测试，记录完整 40 位 SHA。
2. 在共享 SSD 创建该 SHA 的独立 clean checkout。
3. 再次只读核验实例、GPU、锁、存储与父件。
4. 仅对 target 0 启动 `technical-preflight`。
5. 只有 preflight 的所有工程检查通过，才在 fresh root 启动 formal target 0。

当前 Go 只授权技术预检，不授权跳过预检直接启动 256-step formal，也不授权提前进入 Phase 2/3。
