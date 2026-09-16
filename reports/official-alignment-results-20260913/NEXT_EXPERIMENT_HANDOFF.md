# 新实验交接入口（2026-09-16）

## 先读真实状态

- 仓库：`https://github.com/zhangchenxi1224/Vision-Language-Memory`
- 分支：`codex/dreamlite-official-alignment-20260913`
- 完整最新结果快照：`52ddbadd4114e0a571426db2237a03c9c422e4f1`
- 本地工作树：`C:/Users/Expedition/dreamlite-official-alignment-20260913`
- 首读：同目录 `2c5189a-latest-results.md`。
- 完整复核：`2c5189a-complete-results-local-verification.json`；文件清单：
  `2c5189a-complete-results-file-manifest.json`。
- 历史方案与执行记录：`generated-source-validation-registration.md`。
- 原始证据：`2c5189a-complete-results-sync.parts/`，95个分块；重建与复核入口：
  `python scripts/reporting/verify_2c5189a_complete_results.py`。

不要以旧聊天记忆或旧R4结果替代这个快照。可直接clone指定分支；搜索引擎未收录
不是权限问题。如新会话不能访问GitHub，使用本地工作树或文件附件。

## 已完成结果与当前判断

| 指标 | 4f父版本 | b62最新版本 |
|---|---:|---:|
| 开发集 | 1510/1510 | 1510/1510 |
| 原始功能矩阵 | 1790/1800 | 1790/1800 |
| 已观察表达变化矩阵 | 1750/1800 | 1740/1800 |
| PNG读取 | 3540/3600 | 3530/3600 |

两套CLI各6次写入、30次读取，工程回放一致性通过；功能仍有失败。
本轮新增生成源图变化并追加4832次更新，不能归因为纯源图单因素。
4f在这组已验证指标上优于最新b62；不能默认“最新权重就是最佳权重”。
目标尚未达成。主要未解决问题是连续RGB写入链，表达变化下尤其明显。
下一轮先逐例配对4f与b62原始记录，区分修复、退步及共同失败，再登记实验假设。
这些矩阵已经观察过，应作为回归集；不要重新宣称为新holdout。

## 代码入口（相对仓库根目录）

| 工作 | 入口 |
|---|---|
| 核心训练、FM与source条件 | `src/vision_memory/training/latent_bank_unet.py` |
| 推理初态与时间表 | `src/vision_memory/dreamlite/differentiable_mobile.py` |
| 实际训练程序 | `scripts/train/train_latent_bank_unet.py` |
| 训练编排 | `scripts/inspire/run_native_condition_comparison.py` |
| 最新训练登记方案 | `scripts/experiments/generated_source_training_protocol.py` |
| source变化与启动前检查 | `scripts/train/generated_source_augmentation.py`、`scripts/probes/generated_source_training_preflight.py` |
| 完整功能矩阵、包导出与CLI链 | `scripts/inspire/run_broader_completion_suite.py` |
| 完整PNG读取 | `scripts/inspire/run_png_readback_suite.py` |
| 实际用户推理CLI | `scripts/inference/rgb_memory.py` |
| 原始结果逐例配对 | `scripts/reporting/compare_broader_validation_raw.py` |

上述训练/评估编排绑定已登记的提交、父模型、计划和输出目录，不是任意参数的
通用启动器。新实验需要独立分支/提交、独立输出目录、完整登记方案及相应验证绑定；
不要仅替换commit字符串后直接套用旧实验身份，也不要覆盖历史结果。
旧 `.cache` 和远端shell启动器是该次执行的定位资料，不应当作新实验的一键入口。

保留已修复协议：source只作为条件；目标侧FM为
`x_sigma=(1-sigma)*target+sigma*noise`、`v=noise-target`；训练覆盖完整时间范围，
推理从纯高斯开始并使用对应原生时间表。不得为提高当前分数退回source混入噪声的协议。

## 启智资产定位

以下是历史执行时验证过的路径；新会话必须重新检查存在性、哈希、实例状态与资源。
此交接未声称旧实例当前仍运行，也没有启动新实验。

```text
P=/inspire/ssd/project/exploration-topic/czxs26210936
R=$P/runs/dreamlite-official-alignment
Python=$P/envs/vlm-r3-ngc2502/bin/python
模型根=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory
```

- 4f父运行：`$R/4fbc857-clear-retention-full4832`
- 4f导出包：`$R/e372f3c-logical-package`
- 最新b62运行：`$R/b62ec02-generated-source-full4832`
- 最新最终权重：上述运行下 `train/checkpoint-final.pt`
- 权重SHA256：`50c3135d4d2afe091115d5541c7949fcd92ad4a408572ba0be3ab0b24fea9951`
- b62训练提交：`b62ec027ad725aeb6ecc772aa85e7a3ff6e49b36`
- 评估提交：`2c5189a0847acd6653b687031ab13e6cd4cfc53f`
- 固定bank：`$R/84cdfdb-broader151-full4832/bank/manifest.json`
- 生成source池：`$R/90b41a2-generated-source-pool/manifest.json`
- 历史GPU实例：`dl-source-aug-h200x4-20260914`
- 历史CPU传输入口：`dl-align-cpu-20260914-r3`（CPU资源空间）

完整权重、优化器状态、大型PT及模型依赖未全部放入GitHub；只有仓库足以做源码与
已归档证据分析，不足以无外部资产重跑GPU训练。按Inspire skill核实当前资源，保留
用户要求保留的旧实例。不要复用旧Unix deadline。

## 给新对话的启动说明

请接手DreamLite新实验。先读取此交接、最新结果说明及完整复核JSON，以52ddbad
结果快照为事实来源。先分析4f与b62连续RGB写入链的完整失败记录；当前最新版本
没有优于4f，不能宣称已可用。基于证据登记下一轮假设与完整评估方案，在新分支和
独立输出目录执行，重新核验启智资产及资源。保持官方FM/source条件/训练时间/
推理初态和时间表协议；保留固定终点、原始token+EOS评分、完整功能/PNG/CLI验证。
