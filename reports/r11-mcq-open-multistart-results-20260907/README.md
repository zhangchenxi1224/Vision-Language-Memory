# 旧 R11：同题多起点实验的真实结果

实验完成于 **2026-09-07**，结果同步于 **2026-09-08**。本目录是已执行实验的结果包；实验方案保留在原预注册文件中。

**范围：直接优化 VAE latent，冻结 VAE 和 Reader。** 本次不是 R11_new 的完整 DreamLite 输入优化实验。

## 先看什么

- [完整中文结果报告](training_results_report.md)：设计、实际结果、解释与科学边界。
- [八条 MCQ 轨迹图](trajectories/eight-trajectories.png) · [PDF](trajectories/eight-trajectories.pdf) · [交互版 HTML](trajectories/eight-trajectories.html)。GitHub 可直接预览静态图；交互版请下载后用浏览器打开，加载 D3 需要联网。
- [训练输出 summary.json](multistart-5d06b76-20260907-round02/summary.json)：实际运行次数、成功率及技术状态。
- [全部 108 条开放生成原文](multistart-5d06b76-20260907-round02/generations.jsonl) · [便于阅读的回答 CSV](multistart-analysis/answers.csv)。
- [72 个终点 MCQ 视角记录](multistart-5d06b76-20260907-round02/mcq_endpoint.jsonl)。
- [独立原始产物审计结果](multistart-full-audit.json) · [执行记录及失败尝试说明](execution_notes.md)。

## 已完成的结果

固定一道音乐偏好题，答案为 `ambient`。8 个随机起点加 1 个灰图基准起点，每个分别使用 MCQ 与开放回答目标训练，共 **18 runs / 4,608 次更新**。下面的分母均为 8 个随机起点，基准单独报告。

| 终点表现 | MCQ 训练 | 开放回答训练 |
|---|---:|---:|
| 四种选择题排列全部正确 | 8/8 | 0/8 |
| 无选项原问法完整正确 | 0/8 | 3/8 |
| 原问法、改写问法都完整正确 | 0/8 | 2/8 |
| 终点平均两两 RMSE | 0.446664 | 0.444967 |

两组的初始平均两两 RMSE 都为 0.080224；本次观察中，轨迹先分离、后趋缓，终点没有合并。只研究了一道题、一个扰动半径与 256 步训练，不能推断全局唯一性或不同分布下的普遍规律。

**`technical_passed=true`、独立审计 `passed=true`；`formal_success=false`。** 技术检查通过不等于科学目标全部实现。开放训练没有监督 EOS；先生成答案词也不等于完整回答正确。

## 文件与证据范围

| 目录或文件 | 内容 |
|---|---|
| `multistart-5d06b76-20260907-round02/` | 原始文本输出：汇总、配置快照、原文回答、选择题评分、逐步几何统计，以及 18 次运行的训练记录、探针和 tensor/checkpoint 哈希索引 |
| `multistart-analysis/` | 汇总统计、逐起点/逐对距离 CSV、固定探针和科学绘图 |
| `multistart-figures/` | 各起点解码终点图、坐标直方图与通道统计 |
| `trajectories/` | 8 条 MCQ 轨迹的 2,056 个真实采样点投影、静态图与交互图 |
| `replay-9f86bc9-20260907-round01/` | 前置回放阶段：48 条开放生成、32 个旧 MCQ 锚点及核验记录 |
| `multistart-b76969a-20260907-round01/` | 因预注册配置换行符哈希差异而终止的失败尝试；未加载模型或训练 |
| `publication_sources.json` | 本地来源、原文件 SHA-256 及报告链接转换说明 |
| `SHA256SUMS` | 本结果目录发布文件的 SHA-256（不包含校验清单本身） |

原始 JSON/JSONL/CSV 按字节复制，未重写评分或训练记录。完整中文报告只修正了本地文件链接。源文件中的机器路径、历史状态和哈希是执行时的来源记录；当前完成状态应读取运行根目录的 `summary.json` 与 `terminal.json`。

**大体积逐步 latent、Adam/RNG checkpoint 和原始图片张量尚未上传 GitHub。** 约 1.2 GB 的完整归档及原始结果仍位于共享服务器：

```text
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-open/multistart-5d06b76-20260907-round02
```

完整归档 SHA-256：`800d13679f8a9ace4675cd844babc5d7dc5b845a985cbcda69257a5dc78b881f`。各运行的 `latent_index.jsonl` 和 `checkpoint_index.jsonl` 列出服务器原文件路径与哈希；仅有索引不能替代原张量。完整审计是在服务器上读取 4,626 个 latent 与 90 个 checkpoint 后生成的已有记录。本次同步没有重新运行模型或重新完成该全量张量审计。不完整的本地 `.partial` 下载未纳入本目录。

## 如何检查

从仓库根目录运行（仅使用 Python 标准库）：

```bash
python reports/r11-mcq-open-multistart-results-20260907/verify_public_results.py
```

该检查验证发布文件哈希、18 次运行与逐步记录数，使用生成原文重算严格匹配结果，并核对投影数据与保存的轨迹记录。它不替代对服务器原始张量或模型 forward 的重新核验。

轨迹采用一套共同 PCA 坐标系：在 8 条轨迹、步数 0/16/…/256 的 136 个快照上拟合，再投影全部 2,056 个点。二维图保留约 **29.76%** 的整体变化，二维靠近或交叉不能说明原空间重合；距离曲线和终点矩阵使用完整 65,536 维 RMSE。`95% 路程` 指累计移动距离，不是答题成功率。两份投影/绘图源脚本保留执行时路径，需要按运行环境调整。

## 对应代码

- [预注册方案](../r11-mcq-open-multistart-preregistration-20260907.md)
- [训练代码](../../scripts/experiments/run_r11_mcq_open_multistart.py)，真实训练提交 `5d06b7622a4f7081d02e42c656a97cffafe66207`
- [分析代码](../../scripts/experiments/analyze_r11_mcq_open_multistart.py)
- [全量原始张量审计代码](../../scripts/experiments/audit_r11_mcq_open_multistart.py)，审计/绘图提交 `0fc0f21`
