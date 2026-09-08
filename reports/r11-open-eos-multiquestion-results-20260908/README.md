# 16 道真实题的 Open/EOS 完整配对结果

**全部完成：原题完整答案正确率从 A 的 60/64 提高到 B 的 64/64；B 在本批三种提示词下均未出现“正确答案后继续写”。但 B 在旧全新改写上仍错 3/64，不能认为 EOS 解决了所有表达鲁棒性问题。**

本轮验证“单题 ambient 上的停止监督收益，能否在其他真实问题和多 token 答案上复现”。16 道题来自锁定 train 文件的 16 个不同 semantic groups，排除旧 8 题，每题 4 个 seed，A/B 严格配对，共 **128 次独立 latent 优化、32,768 次更新**。两路 H200 各跑 64 条，于北京时间 2026-09-08 16:36:50 启动，约 18:01:53 完成，耗时 **5,102.246 秒（85 分钟）**。没有训练共享 Writer。

A 为原答案 token 平均 CE；B 为答案 token 平均 CE＋单个 assistant EOS 的 CE，权重 1。两者均只训练 original 问法，使用同一旧 R11 初始化（每题同 seed 的 A/B 起点 SHA 相同）、冻结 BF16 VAE/Reader、Adam lr=0.05、固定 256 步。A 是新题的 fresh paired baseline，没有套用旧 ambient 轨迹的一致性结论。

## 最终完整输出与停止行为

每格分母是 **16 题 × 4 seed = 64 组配对**。使用固定 step 256、greedy 32 tokens、原模型 EOS；没有选最佳 checkpoint、按答案长度截断或只保留第一个词。

| 提示词 | A 完整正确 | B 完整正确 | A 正确答案 token 前缀 | B 正确答案 token 前缀 | A 多余续写 | B 多余续写 |
|---|---:|---:|---:|---:|---:|---:|
| 原题，训练过 | 60/64 | **64/64** | 64/64 | 64/64 | 4/64 | **0/64** |
| 固定改写，未训练 | 62/64 | **64/64** | 64/64 | 64/64 | 2/64 | **0/64** |
| 旧全新改写，未训练 | 56/64 | **61/64** | 62/64 | 61/64 | 6/64 | **0/64** |

逐对比较：原题 B 改善 4 对、退步 0 对；固定改写改善 2 对、退步 0 对；旧全新改写改善 **8 对、退步 3 对**，净改善 5 对。三种提示词全部答对的终点：A 54/64，B 61/64。不能把汇总提升理解成每一对都改善。

**旧全新改写同时改了问题句和指令。** original/paraphrase 尾部是 `Use the memory image to answer. Answer with a short phrase only.`，而 new_rewrite 尾部是 `Return only a short phrase.`。因此最后一行衡量问句＋指令共同变化，不能单独归因于问句改写。旧评测保持原样；固定两行指令的 question-only 补测应单独报告。

## 哪些题、多少 token

下表每行 2 道题 × 4 seed，均为原题终点。长度来自实际 Reader tokenization，不按英语词数猜测。

| 真实字段 / 答案 | token 数 | A 完整正确 | B 完整正确 |
|---|---:|---:|---:|
| color / green | 1 | 8/8 | 8/8 |
| drink / juice | 2 | 8/8 | 8/8 |
| music / jazz | 2 | 8/8 | 8/8 |
| material / linen | 2 | **4/8** | **8/8** |
| meal / pasta | 2 | 8/8 | 8/8 |
| color / no active preference | 3 | 8/8 | 8/8 |
| drink / no active preference | 3 | 8/8 | 8/8 |
| material / no active preference | 3 | 8/8 | 8/8 |

原题增益集中在 linen 两题（target 6：3/4→4/4；target 7：1/4→4/4）。其他 14 题的 A 原本全对，因此没有“每题都获益”的证据。3 token 答案原题 A/B 均 24/24；旧全新改写 A 21/24、B 22/24。长度和题目内容一起变化，本实验不能证明长度造成差异，也没有合成 light blue/orange juice 训练题。

## 图像依赖对照

| 固定对照图 | 原题 EM，A/B 相同 | 固定改写 EM，A/B 相同 | 旧全新改写 EM，A/B 相同 |
|---|---:|---:|---:|
| Blank | 0/64 | 0/64 | 0/64 |
| 已验证的 ambient donor | **4/64** | **8/64** | 0/64 |

Matched 明显高于 blank 和 donor，支持答案依赖优化得到的图片；donor 并非全错，不能宣称输出只取决于正确记忆。该 donor 来自旧 EOS-B seed 0 的 ambient 终点，gold 与本批每题不同，具有旧原题及改写均正确的证据。**同一 donor/blank 在 seed 和 A/B 间重复，控制成功数不是 64 个独立 donor 实验。**

## 审计与复核

- [独立审计脚本](audit_results.py)只使用 Python 标准库，不导入训练器或加载 GPU 模型。
- [audit.json](audit.json)：784 份远端原始文件 SHA/字节数一致，640 份非 tensor 文件亦符合训练时 inventory；10 个训练源码哈希与固定历史 Git blob 一致。
- 128 条训练均完整记录 1–256 步，896 份 checkpoint 索引覆盖 0/16/32/64/128/192/256，全部 3,456 条 raw 与预期网格完全吻合、无重复。每条为 6 个中间步×3问法＋最终步×3问法×3图像条件＝27 条。
- 64 组 A/B 初始化 SHA 配对，192 条 step-0 matched 生成逐 token 相同；两路结束前的代码/targets/模型绑定回执均通过。
- 原始文本与 token 可独立重算 EM、答案前缀、位置 token 准确率和多余续写。teacher-forced 指标只核验记录与有限性；没有下载 logits、模型、完整训练数据或大 tensor，不能称作本地重新推理或重新验证 tensor 数值。
- [endpoint_metrics.csv](endpoint_metrics.csv)、[per_question_metrics.csv](per_question_metrics.csv)、[answer_length_metrics.csv](answer_length_metrics.csv)、[trajectory_metrics.csv](trajectory_metrics.csv)保存各层汇总；[paired_endpoints.csv](paired_endpoints.csv)逐对保留完整答案；[endpoint_raw_answers.csv](endpoint_raw_answers.csv)含全部最终换图对照；[rewrite_consistency.csv](rewrite_consistency.csv)记录每个终点的问法一致性。

```bash
python reports/r11-open-eos-multiquestion-results-20260908/audit_results.py
```

[原始 JSON/JSONL](raw/download_manifest.json)保留远端相对路径和逐文件 SHA；目录内 `.gitattributes` 禁止 Git 换行转换。[下载归档](raw-download.tar.gz)为 1,517,658 bytes，SHA256 `90813d81a438feeac3cd88ea5599358db67fe478f1cd4faaaee617b2edc6244f`。

训练固定提交：`2c0e41c899910bf0641f16ee724f85bbe3491a7e`。
题目配置 SHA256：`d356238fd5c267812dcf28d214ab062fd43388bb6b53b78602f0c1e8f5b36672`。
远端原结果：`/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-open-eos-multiquestion/paired-2c0e41c-20260908-r01`。

本结果支持的是：**在这批固定真实题上，答案＋EOS 监督比仅答案监督更稳定地给出完整短答案。** 16 题都是分别使用正确答案优化 latent 的 oracle，改写是同题重复测量；不构成共享 Writer、未见问题泛化、整段历史压缩或 DreamLite 完整生成链成功的证据。
