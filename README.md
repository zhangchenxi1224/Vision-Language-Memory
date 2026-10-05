# Vision Learnable Memory

当前主线：**官方 PrefEval 输入与任务 → 视觉教师目标 → DreamLite 官方 target–noise flow matching → 实际 RGB 写入与读取 → 内容泛化和连续保持。**

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
| 多目标与 C8 | [多目标目录](reports/prefeval-multitarget-20260927/) | 各轮、配对 baseline、样本数和链长分开报告 |

机器可读入口为 [experiments/registry.json](experiments/registry.json)。默认分析只使用 `current_performance` 目录，再核对每次 run 的协议、分母和完成回执。禁止递归汇总全部 `reports/`；目录准入不代表每个计划都已完成或所有分数均可比较。

```bash
python scripts/reporting/list_current_experiments.py
python scripts/reporting/verify_aligned_cleanup.py
```

截至 2026-10-02 的归档状态：B730 exposure512 已完成训练，dev 尚无稳定改善；多目标多轮及 C8 全量730评测已有结果，配对 B0 和 long10 尚未全部闭环；C/R 训练完成，完整评测仍有缺口。本次没有重新查询 GPU 实时状态。

## 并行监督实验（默认仍为 hard CE）

新增 [历史条件 Prompt Matching](docs/PROMPT_MATCHING.md)：`hard_ce` 保留原始答案监督；
`history_hard` 与 `prompt_matching` 共用同一冻结 Reader 在已观察历史下生成的精确回答序列，
分别使用 one-hot 与完整词表软分布，隔离教师来源与监督方式的影响。教师只见当前初始 exchange，
Writer 和学生 Reader 输入边界、VAE、PNG 回读及官方 FM 保持不变。

入口：`scripts/inspire/run_prompt_matching_parallel.py`；前瞻协议：
`configs/experiments/prompt_matching_parallel.json`。先运行真实梯度 smoke，再做同预算 paired pilot。
训练 loss 或教师图提升不触发替换；完整学生、未参与训练问题、偏好与保持评测通过后才审阅晋级。
本节声明实现能力，不声明实验已完成或软监督优于原方案。

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
