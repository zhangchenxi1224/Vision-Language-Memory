# Vision Learnable Memory

当前主线：**官方 PrefEval 输入与任务 → 视觉教师目标 → DreamLite 官方 target–noise flow matching → 实际 RGB 写入与读取 → 内容泛化和连续保持。**

**当前四组基线**以[增强实验方案](reports/prefeval-aug4x3-init-ablation-exposure512-20261009/PLAN.md)和[机器可读协议](reports/prefeval-aug4x3-init-ablation-exposure512-20261009/PROTOCOL.json)为准：P-L/P-H 从官方 `carlofkl/DreamLite-base` 原始预训练 U-Net 开始，R-L/R-H 使用同架构标准随机初始化；采用 W0–W3 × T1–T3 增强，学习率为 `5e-5` / `5e-4`，effective batch=4，每组 93,440 步，在 32/64/128/256/512 次曝光处检查。`4f` 任务权重不属于当前初始化来源。

**2026-10-14：新增旧 B730 在最新统一协议下的补测阶段结果。** 固定93,440步终点五个split已全部核验；本次发布快照为12/17份主汇总，完整多checkpoint补测尚未全部归档。同一96条已训练内容上，B730原表达/原提问为93.62%，未见写入表达47.40%，未见提问91.15%，二者同时变化48.05%；AUG4 P-L的未见写入表达为72.27%。B730的dev90和official177匹配−错配分别为−0.69、−0.07个百分点，尚无明确未见内容收益。模型间初始化和训练体系不同，不能单独归因于增强。

[B730补测入口](reports/b730-latest-eval-20261013/README.md) · [终点五模型对比](reports/b730-latest-eval-20261013/evidence/partial-summaries-20261014-021530/ENDPOINT-COMPARISON.md) · [已核验checkpoint明细](reports/b730-latest-eval-20261013/CHECKPOINTS-PARTIAL-20261014.md) · [进度及范围](reports/b730-latest-eval-20261013/PROGRESS-20261014.md) · [证据与离线复算](reports/b730-latest-eval-20261013/EVIDENCE-20261014.md)

**2026-10-13：AUG4四组训练、全部规定评测与归档已完成。** 四组各93,440步，1,472/1,472任务，84份原始学生汇总及556,800行回读；official180和去重177均已完成，该实验的监控与资源已收尾。上述B730补测是后续独立评测任务。

| 终点组别 | Train 匹配 / 错配 | Dev 匹配 / 错配 | Official180 匹配 / 错配 | Official177 匹配 / 错配 |
|---|---:|---:|---:|---:|
| P-L | 90.26 / 41.39 | 44.31 / 40.97 | 35.49 / 34.72 | 35.03 / 34.89 |
| P-H | 40.34 / 40.38 | 40.97 / 40.69 | 34.72 / 35.21 | 34.89 / 35.38 |
| R-L | 81.80 / 41.28 | 42.36 / 41.39 | 35.21 / 35.14 | 34.46 / 35.31 |
| R-H | 40.58 / 40.58 | 40.97 / 40.56 | 35.07 / 35.07 | 35.24 / 35.24 |

单位为%。低学习率组在已训练内容上有明确匹配记忆优势；P-L在本轮有限预算内学习更快。四组dev、official180/177均没有正向且95%区间排除零的匹配−错配收益，尚未证明可靠的新内容泛化。单个配对训练种子与偏好级区间不代表跨训练种子稳定性。

[最终报告](reports/prefeval-aug4x3-init-ablation-exposure512-20261009/FINAL-REPORT-20261013.md) · [全checkpoint与控制分母](reports/prefeval-aug4x3-init-ablation-exposure512-20261009/ALL-CHECKPOINTS-20261013.md) · [旧B730全量比较](reports/prefeval-aug4x3-init-ablation-exposure512-20261009/B730-COMPARISON-20261013.md) · [第一性原理分析](reports/prefeval-aug4x3-init-ablation-exposure512-20261009/ANALYSIS-20261013.md) · [完成进度](reports/prefeval-aug4x3-init-ablation-exposure512-20261009/PROGRESS-20261013.md) · [证据及CPU复算](reports/prefeval-aug4x3-init-ablation-exposure512-20261009/EVIDENCE-20261013.md)

历史路线调整：2026-10-08 用户决定恢复**选择题硬监督（B臂）**，当时将执行代码恢复到引入下述实验之前的 `6d9fea6` 基线。10月5日至8日的Prompt Matching、Direct/PM→FM、上下文覆盖及后续曝光/排版实验全部撤回，不作为当前训练或结果依据；既有9月硬监督数据、权重及结果继续保留。当前四组的增强问法、初始化和预算采用上述新协议。

2026-10-02 已完成远端收敛：开发分支仅保留 `main`，31 个旧分支均先保留归档标签再删除引用。历史恢复、PR 收尾与 Release 核验见[部署记录](docs/REMOTE_CONSOLIDATION_20261002.md)；Git 体积与后续迁移方案见[历史瘦身评估](docs/HISTORY_SIZE_ASSESSMENT_20261002.md)。

教师可读性、共享 Writer 单次写入、未见偏好泛化和连续保持分别报告。训练完成不代表评测完成；教师分数不能替代学生分数。

## 官方对齐边界

| 修复 | 提交 | 作用 |
| --- | --- | --- |
| DreamLite FM 与采样链路 | `c7e752b`，2026-09-13 | 目标与独立噪声插值、完整 sigma 域、纯噪声初态、真实 scheduler |
| DreamLite 原生条件编码 | `03f8467`，2026-09-14 | 对齐训练条件和 Base 原生推理条件 |
| PrefEval 数据与任务 | `b615c27`，2026-09-23 | 官方主题划分、完整对话、发布回答与答案监督边界 |
| 两阶段路线与评分 | `894928f` / `9e2a374`，2026-09-24 | 教师 latent → 官方 FM；官方评分接口，替代 Judge 单独标记 |

旧自定义桥、旧 PrefEval 任务及相应结果已经退出当前结果目录。不能用其失败推断当前方法的性能或容量；不能仅按日期判断实验有效性。当前仍依赖的通用代码保留，文件名中的 R11、Mobile 或 latent-bank 不代表执行旧协议。

详见[边界与保留规则](docs/ALIGNMENT_BOUNDARY.md)、[清理记录](docs/CLEANUP_20261002.md)和[逐文件清单](docs/alignment-cleanup-manifest.json)。

## 当前实验入口

| 实验族 | 证据入口 | 范围 |
| --- | --- | --- |
| PrefEval 官方输入 | [对齐说明](reports/prefeval-official-alignment-20260923/ALIGNMENT.md)、[FM 路线](reports/prefeval-official-alignment-20260923/FM_ROUTE_CORRECTION.md) | 协议与数据；不是官方论文模型性能复现 |
| K1 / A-B 教师 / FM / 保持 / 曝光量 | [K1 目录](reports/prefeval-k1-l0-l2-20260924/) | 按 split、teacher/student、预算、链长和 Judge 分开报告 |
| B730 最新统一协议补测 | [阶段结果、协议及证据](reports/b730-latest-eval-20261013/README.md) | 原B730固定权重；未见写入/提问与未见内容分开；93440终点已核验，多checkpoint阶段快照12/17 |
| 多目标与 C8 | [多目标目录](reports/prefeval-multitarget-20260927/) | 各轮、配对 baseline、样本数和链长分开报告 |
| U-Net 初始化、学习率与曝光量 | [完成结果与证据](reports/prefeval-aug4x3-init-ablation-exposure512-20261009/README.md) | P-L/P-H/R-L/R-H四组W0–W3 × T1–T3；各93,440步及全部规定评测完成；新旧协议差异单独列明 |

机器可读入口为 [experiments/registry.json](experiments/registry.json)。默认分析只使用 `current_performance` 目录，再核对每次 run 的协议、分母和完成回执。禁止递归汇总全部 `reports/`；目录准入不代表每个计划都已完成或所有分数均可比较。

当前四组注册项 ID 为 `prefeval-aug4x3-init-ablation-exposure512`，其中 `plan`、`protocol`、`results`、`progress` 和 `evidence` 提供方案、配置、完成结果与验证入口。源代码中的 [run_prefeval_aug4_balanced.py](scripts/inspire/run_prefeval_aug4_balanced.py) 是初始8卡控制器；最终执行使用正式交接的r4混合16卡与共享评测池，冻结补充源码位于[科学证据Release](https://github.com/zhangchenxi1224/Vision-Language-Memory/releases/tag/prefeval-aug4x3-final-20261013)。本轮已经完成，不应运行历史恢复入口或另启控制器。

```bash
python scripts/reporting/list_current_experiments.py
python scripts/reporting/list_current_experiments.py --role protocol_and_data
python scripts/reporting/build_prefeval_aug4_publication.py --check
python scripts/reporting/build_b730_supplement_publication.py --check
```

其他实验族的历史状态（2026-10-02）：B730 exposure512已完成训练，dev尚无稳定改善；多目标多轮及C8全量730评测已有结果，配对B0和long10尚未全部闭环；C/R训练完成，完整评测仍有缺口。上述历史范围与本轮aug4x3的完成状态分开报告。历史清理回归工具 `verify_aligned_cleanup.py` 针对当时快照，不是本轮结果验收入口。

## 训练与生成契约

```text
x_sigma = (1 - sigma) * target + sigma * noise
v_target = noise - target
loss = MSE(v_theta([x_sigma, previous_RGB_latent], timestep, native_condition), v_target)
```

source 只提供条件，损失计算目标半边。当前路线使用 Base 原生28步生成，每次写入从新高斯噪声开始，跨写入持久状态为实际 RGB。full-U-Net、学得的教师 endpoint、指定 CFG 与记忆任务属于项目扩展，不称为官方 LoRA 配方原样复现。

## 复现与验证

模型和数据来源见 [models.lock.json](models.lock.json)、[data.lock.json](data.lock.json)及各 run manifest；当前数据见 [data/README.md](data/README.md)。安装 CUDA 匹配的 PyTorch 后，按 `requirements/` 与 `pyproject.toml` 配置环境。

```bash
python -m pytest tests/test_official_fm_parity.py tests/test_native_base_training_alignment.py tests/test_prefeval_official_pipeline.py tests/test_prefeval_k1_data.py tests/test_prefeval_multitarget_bank.py -q
```

官方算术 parity 需要 `third_party/DreamLite` 位于锁定提交 `a6e20c8cc94027f37dd7c5a81b0b3b472aa18409`。旧自定义数据集的合同测试已逐项标记退休，不计作通过；通用梯度、PNG、输入隔离和当前协议测试继续执行。

## 工程证据与历史

[DreamLite 对齐审计](reports/official-alignment-audit-20260913.md)及 `reports/official-alignment-results-20260913/` 保留转折期配对控制、bank、初始化和测试夹具，统一属于 `engineering_provenance`，不进入当前 PrefEval 成绩汇总。

被撤下文件可按清单记录的 Git 提交和 blob 身份恢复。新的结果入口以注册表为准，后续实验不得重新用未对齐基线填充当前成绩表。

历史报告中的 `codex/...` 分支已归档为 `archive/2026-10-02/codex/...` 标签。新实验从 `main` 开始；复现历史实验时先获取对应归档标签，再检出报告锁定的精确提交。既有冻结恢复目录继续使用原提交。浅克隆可能缺少 `git show` 所需的历史对象，不能直接替代历史复现环境。
