# R11 Open/EOS：固定指令、只改提问句式的补测结果

**已完成：保留原读图和回答指令后，四种新增句式下 B/C 全部答对；恢复原混合改写的固定指令后，B 为 6/8、C 为 7/8，剩余错误均为 `ambient` 后多余续写。**

这次执行保留了用户的控制变量修正：只更换问题句式，末尾逐字符固定为：

```text
Use the memory image to answer. Answer with a short phrase only.
```

复用 A/B/C 各 8 个 raw step 256 终点，共 24 个，不重新训练。2026-09-08 18:02:47（Asia/Shanghai）在空闲的 `vlm-r11-open-h200x2-20260907` 两卡实例启动，实际耗时 **239.454 秒，约 4 分钟**。共完成 **576 条原始生成**，其中新问法 matched 120 条、原提示词一致性重放 216 条；其余为固定图像对照。没有新增训练 run。

## 完整输出正确数

下表每格分母 8 表示同一道题的八个初始化终点。5 个问法是重复测量，不是 5 个独立任务。

| 评测问法 | A：无 EOS | B：加 EOS | C：加 EOS＋旧改写训练 |
|---|---:|---:|---:|
| 原混合改写，恢复固定指令 | 0/8 | 6/8 | 7/8 |
| 新句式 02 | 4/8 | 8/8 | 8/8 |
| 新句式 03 | 1/8 | 8/8 | 8/8 |
| 新句式 04 | 2/8 | 8/8 | 8/8 |
| 新句式 05 | 1/8 | 8/8 | 8/8 |
| 同一终点五种问法全部答对 | 0/8 | 6/8 | 7/8 |

三组在这五种问法下的 **120 条 matched 输出全部具有正确答案 token 前缀 `ambient`**，完整输出失败均为额外续写。A 的多余续写仍很突出；B/C 显著减少，但在保留的原混合改写问题句下仍有停止不稳。

所有 blank/fixed donor 条件下完整正确均为 0/8；这些固定对照图跨 seed 重复，不可放大为独立任务证据。

## 恢复固定指令前后，具体变化

这个比较保留了原 `new_rewrite_open` 的整个问题首行，只把末尾恢复为原指令。其余未列出的 B/C 终点前后都正确。

| 方法 / seed | 原混合提示词输出 | 恢复固定指令后 |
|---|---|---|
| B / 1 | `ambient synthwave` | `ambient synthwave` |
| B / 7 | `None available` | `ambient synthwave` |
| C / 4 | `None available` | `ambient` |
| C / 6 | `None available` | `ambient synthwave` |

B 的完整正确仍为 6/8，但正确答案前缀从 7/8 增为 8/8；C 的完整正确从 6/8 增为 7/8，正确答案前缀从 6/8 增为 8/8。因此不能只看完整正确总数而忽略错误类型的变化。

## 科学结论与范围

1. 原先的 6/8 同时包含问题句式和回答指令变化，不能单独归因于句式变化。该限制已追加到原报告；原结果与配置完整保留。
2. 此次固定指令的五问诊断中，正确答案前缀全部读出，B/C 在四种新句式下完整正确均为 8/8。当前主要残余问题是特定问题表达下的停止稳定性，不能概括为“换问法就读不出记忆”。
3. 恢复整段指令有助于这几个失败终点读出答案，但同时恢复了读图要求和格式措辞，不能进一步认定是“Use the memory image”这一句单独造成全部改善。
4. C 比 B 多一个终点达到五问全对，是本轮描述性差异；一题、八个配对初始化不足以证明 C 普遍更优。
5. 这是看到原结果后的控制变量诊断，不是未见任务测试。不能据此宣称共享 Writer、跨知识泛化、DreamLite 完整链或正式 Picture Memory 成功。

## 可核验产物

- [冻结的补测方案](../r11-open-eos-question-only-protocol-20260908.md)；[全部问法原文](../../configs/experiments/r11_open_eos_question_only.json)。
- [原始完整生成及 token](raw/raw_generations.jsonl)、[汇总](raw/summary.json)、[逐终点回答](endpoint_answers.json)、[独立审计](audit.json)。
- 216 条原提示词重放的输入 token、输出 token、chat prompt、完整文本与评分全部一致；24 个终点文件在评测前后未改变。
- 本地独立审计重新验证 9 个原始文件 SHA、576 条覆盖与评分、固定指令、216 条锚点一致性和逐 seed 汇总。latent/图像 tensor SHA 来自远端记录，本地审计未加载远端大 tensor。
- 本地 14 项相关测试通过；原始科研生成维持 greedy 32 tokens 与原 EOS，评分未裁成第一个词，未使用部署解码替换指标。

执行代码固定提交：`d430ff621fd2182b8560299958fa2886f9be3142`。原训练固定提交：`e6e6c8071a6c1311a966bda1aa636748fbb016f6`。

远端补测输出：
`/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-open-eos/question-only-d430ff6-20260908-r01`

归档 [question-only-d430ff6-20260908-r01.zip](question-only-d430ff6-20260908-r01.zip)：51,347 bytes，SHA256 `4f119251ed465563b046787b0d8aa896681b3fd4597257ae7bd41abe6aed3333`。不包含已有的训练 tensor。

独立重算：

```bash
python reports/r11-open-eos-question-only-results-20260908/audit_results.py
```
