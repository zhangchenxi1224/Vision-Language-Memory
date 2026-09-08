# R11 填空题 EOS 实验：真实完成结果

> 2026-09-08 追加修正：下表的“全新改写”同时更改了问题句式与末尾读图/回答指令，故 6/8 不能单独归因于提问句式变化。原数据保留不变；固定指令、只改提问句式的补测方案见 [追加协议](../r11-open-eos-question-only-protocol-20260908.md)。

**补充“答案后立即结束”的监督，解决了这道题在原问法和旧改写下的多余续写；对全新改写改善明显，但仍有失败。**

固定同一道音乐偏好题，gold 为 `ambient`。8 个相同随机起点，A/B/C 各 256 步，共 **24 次训练、6,144 次更新**；另完成旧 Open 的 8×7 个 checkpoint 回放。总耗时 **1,988.911 秒（33分09秒）**。生成主指标始终为原始 greedy 32 tokens 的完整输出，未用截词或部署解码替换。

| 训练方法 | 原题完整正确 | 旧改写完整正确 | 全新改写完整正确 | 原题多余续写 |
|---|---:|---:|---:|---:|
| A：仅答案 CE，无 EOS | 3/8 | 3/8 | 0/8 | 5/8 |
| B：答案 CE + EOS CE | **8/8** | **8/8** | **6/8** | **0/8** |
| C：同 B，原题/旧改写各训练一半 | **8/8** | **8/8** | **6/8** | **0/8** |

B 没有训练旧改写，C 训练过旧改写；因此 C 的旧改写成绩不能称为未见问法测试。全新改写对三个方法都从未用于训练。B/C 在全新改写均为 6/8，当前证据不支持 C 优于 B。

原题与旧改写的正确答案前缀三组都是 8/8：A 的问题主要是 `ambient` 后继续生成额外内容。全新改写中，A/B/C 的正确前缀分别为 5/8、7/8、6/8，多余续写分别为 5/8、1/8、0/8。C 的两次失败已经是答案本身不对，单独修正停止规则无法概括全部错误。

所有方法、所有问法，换成 blank 或固定 donor 图像后完整正确数均为 **0/8**。这支持成功依赖本题优化出的图像；对照图在 seed 间重复使用，不代表 8 个独立对照任务，更不能据此宣称一般记忆泛化。

旧 Open 回放显示：第 16 步起，原题 8 个 seed 都能先输出 `ambient`，但第 16→32→64→128→192→256 步完整正确数为 **3→3→4→3→4→3/8**。答案 CE 持续下降，并没有带来完整生成持续改善；本次不支持仅把原目标训练延长到 512 步。

## 结果可信度与范围

- A 的 8×7 个 checkpoint 全部与旧 Open 逐位一致，最大误差为 0；A 与旧回放的 216 条生成 token 序列也完全一致。
- 实际 assistant 终止标签为 `<|im_end|>`（151645），生成停止集合为 `[151645,151643]`。B/C 的损失为 answer token 平均 CE + 1×EOS CE，没有按总 token 数稀释 EOS。
- `raw/` 保留 86 个轻量原始 JSON/JSONL，包含 manifest、完整生成、每步 metrics、终态与 parity 记录。逐文件 SHA256 在 `raw/download_manifest.json`，本地独立重算见 `audit.json`。
- `endpoint_metrics.csv` 是 27 个末步比较单元；`trajectory_metrics.csv` 是完整过程指标；`endpoint_raw_answers.csv` 保留逐 seed 末步原文，方便核查失败。
- 288 条部署解码单独存放，未计入上述 raw 主指标。本次多词/字段检查是 tokenizer 契约审计，尚未进行跨字段训练。
- 本实验只证明这一个 query-level VAE latent oracle 中补充 EOS 监督有效，**不是共享 Writer、未见任务、DreamLite 完整链或正式 Picture Memory 成功**。

## 溯源

训练 commit：`e6e6c8071a6c1311a966bda1aa636748fbb016f6`。

共享盘完整原始结果：
`/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-open-eos/all-e6e6c80-20260908`

轻量 ZIP：`light-science-e6e6c80-20260908.zip`，272,028 bytes，SHA256：
`f049d5dd8ea50e3dcc28f640aacbeffbe9d3271c6a934c2690ff05e6bd0f133b`。

大 tensor 留在共享盘，未包含在此轻量报告目录。可在仓库根运行 `python reports/r11-open-eos-results-20260908/audit_results.py` 重算上述 CSV 和独立 JSON 审计。
