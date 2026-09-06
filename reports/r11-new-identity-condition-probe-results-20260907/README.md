# R11_new：固定 identity 条件工程探针真实结果

**工程通过；未进行训练，未评判 bridge 诊断或科学成功。** Phase 1A 仍为 6/8，Phase 2/3 未启动。

执行提交：`55c355f769c21ceb9b86e5090cdafa4090751d21`。实例：`vlm-r3-h200x2-live-20260717`。实际运行时间为 2026-09-06 17:53:10–17:53:28 UTC（北京时间 9 月 7 日 01:53），探针内部耗时 `17.12318854406476` 秒。仅首次运行、首次两次编码，没有重启择优。

## 做了什么、验证了什么

上一轮 source-only 初始化的 teacher-MSE bridge 距离和 Reader 门均失败。本项为下一轮单因素 conditioning 诊断锁定工程输入：保留原样本、固定 source 和官方编辑模板，只把实际条件文本改为 `no changes`。

加载冻结 DreamLite 基座，生成固定 source latent，然后两次独立调用原生条件编码。实际执行包括 VAE 和内部 conditioner 推理，不能称为“零模型前向”；完整四步 DreamLite、U-Net、外部 Reader、反向和优化器更新均为零。未加载 canonical teacher 或外部 Reader。

| 项目 | 实际结果 |
| --- | --- |
| 条件编码次数 | 2，两份原始 tensor 逐位相同 |
| 新 prompt embeddings | BF16，shape `[1,293,2048]`，有限且非零 |
| 新 attention mask | int64，shape `[1,293]`，合法非空二值 mask |
| 原 event 条件长度 | 322；新固定文本长度为 293 |
| 条件是否真实改变 | embeddings 与 mask 均区别于原 event 条件 |
| source RGB / FP32 latent | 与固定父证据 SHA256 精确一致 |
| 模型冻结 | 1,834 个参数 tensor，2,519,945,483 个参数；无可训练参数、无参数梯度 |
| 模型快照 | 27 个文件，前后完整验证一致 |
| U-Net / 完整 DreamLite / 外部 Reader 前向 | 0 / 0 / 0 |
| backward / optimizer updates | 0 / 0 |
| 本地回归 | 131 passed；其中新探针 42 项 |

序列长度和 mask 改变是文本干预的下游效应，不是另加独立干预；但本实验不能分开解释文本语义、长度、格式与模型先验的贡献。`no changes` 只是自然语言条件，不保证数学恒等或输出等于 source。

## 原始绑定

| 文件 / tensor | SHA256 |
| --- | --- |
| 完整原始压缩包（1,926,543 bytes） | `fb77f5bac2d3c96816b5451254ea5fdb9288c7792f3b69f37f9af1b44f68206f` |
| manifest | `6dc0652d6d1750c51341a55849757bcc3fa86b201a451eb020803e0912e0aaa2` |
| result | `e414b9ed035f102a9ef7838da6948071e9e735ee6bdb39b2965c164c3ec02ecf` |
| terminal | `6bd19ec3398edf5f869b018575a9a3da33b238abd0dc58b321af706d3e942b55` |
| inventory | `85ad0e79b98626ccb63ef760b09b9ee02d20720500ece5df79cd8ad54b37e087` |
| 新 prompt embeddings tensor | `dbb23b166711849739432ba815002e889253da78e099e59c7adbd351a8fb9f4d` |
| 新 attention mask tensor | `c3325ac723d7cc08cba37d46e1710b5067019e47236ef55c41c66cba27c519f1` |

两份 condition 文件的文件哈希不同，但其中的 embeddings/mask tensor 哈希相同，不能把容器文件字节不同误判为数值不一致。每份文件的字节数和 SHA256 见 `raw-manifest.json`。

原始数据：[probe-raw.tar.gz](probe-raw.tar.gz)。包含 `source.pt`、两份 condition tensor、配置、manifest、环境/runtime、快照核验、日志、终态和文件清单。没有训练，故 optimizer receipts/checkpoints 为**不适用**，不伪造记录。

机器可读结果：[raw-result.json](raw-result.json)；原始终态：[raw-terminal.json](raw-terminal.json)；本地补充复核：[local-audit.json](local-audit.json)。

## 核验范围与工程问题

本地逐一验证 inventory 的 14 个文件，核对 terminal→result/manifest 的哈希，重新加载两份原始 condition 与 source tensor，复算 canonical tensor SHA256、逐位相等、finite/nonzero、固定文本和原 target 绑定。模型文件的完整重哈希发生在实际 H200 探针前后；本地只核对其已绑定记录，不冒充在 CPU 重跑 CUDA 模型或独立观察全部调用。

本地第一次复核被一个从换行终端输出手工抄录的 65 字符 result hash 阻断；改为原始 terminal 与实际 result 文件一致的 64 字符 SHA256 后复核通过。远端结果、配置、阈值均未修改，后续 18:16:33 UTC 远端重算再次一致。该时刻 GPU 无计算进程、probe PID 已退出、suite lock 已释放。

复核脚本 `audit_local_raw.py` 为只读；从仓库根运行，输入分别为本轮已解包原始目录（含 `probe-raw.tar.gz` 和 `probe/`）及已闭合 source-only 父原始目录。输出 JSON 到 stdout；禁止 Python `-O`，不能关闭检查。

```text
python -B reports/r11-new-identity-condition-probe-results-20260907/audit_local_raw.py --raw-root <本轮原始目录> --source-parent-raw <source-only父原始目录>
```

## 决策与下一步

仅允许把实测条件 tensor 哈希、probe 提交、原始产物和父证据绑定写入**新的完整 bridge 配置**。实现、测试、提交后，须在全新运行根重新编码核对锁定哈希，再执行 1 次完整四步前向、1 次反向、0 次更新的 technical-preflight；预检有效后才运行固定 256 步。

下一完整实验仅替换实际 conditioning 文本，以已闭合 source-only + 原 event 臂为对照。初始化、teacher-MSE 目标、预算、raw256、Reader/reset 和距离门均不变。不能以 Gaussian 为主要对照而混入两个因素。

本次探针不能说明可达性、优化难度、Reader 改善或共享 writer 已学会；没有训练 loss 可供判优。后续即使 teacher-MSE 诊断通过，也不能代替原 QA 目标下统一固定八成员的 Phase 1A 8/8，更不能直接启动 Phase 2。

原约 30 小时规划起点仍为 `2026-09-05T07:18:59.891307+00:00`，已超时；不重置时钟、不降低成功门槛、不缩减 Phase 2 最低 64 条。`formal_success=false`、`phase2_allowed=false`。
