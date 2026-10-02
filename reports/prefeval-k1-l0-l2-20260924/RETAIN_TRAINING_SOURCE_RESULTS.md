# B 在见过的源图上也未实现固定首例的一步保持

2026-09-24。按[已登记定位](RETAIN_TRAINING_SOURCE_PROBE.md)，固定首例、错配对象、
本轮保持后权重、官方第 1 轮干扰、两种原噪声、NativeBaseEditSampler 28 步和 T1 MCQ。
全部 8 次生成、16 条读回已完成，均无解析失败或截断。

| 输入来源与控制 | 噪声链 | 更新前 | 更新后 |
|---|---:|---|---|
| 训练原初写 PNG，匹配 | 0 | D，对 | C，错 |
| 训练原初写 PNG，匹配 | 1 | D，对 | C，错 |
| 保持后新初写 PNG，匹配 | 0 | D，对 | C，错 |
| 保持后新初写 PNG，匹配 | 1 | C，错 | C，错 |
| 训练原初写 PNG，错配 | 0 / 1 | C，错 | C，错 |
| 保持后新初写 PNG，错配 | 0 / 1 | C，错 | C，错 |

两条“训练原初写 PNG”分支使用同一张实际训练输入，只改变生成噪声。
该 PNG 与训练源图清单的哈希一致。四次新初写图重放与此前 prefix1 PNG 逐字节相同，
对应的八次更新前后读取，包括输入/输出 token，都与此前记录完全一致。

## 结论边界与下一步

这例不能仅归因于新旧初写图片的分布变化：已见输入仍可读，但保持后的 Writer
在两种固定噪声下一次更新就失去正确回答。结合[先前 VAE 往返诊断](CODEC_PROBE_RESULTS.md)，
排查重点进一步落到共享 Writer 的保持映射及其当前拟合程度。
这不证明 VAE 编码绝无影响，也不能从一个问题的错误推断输出图完全不存在任何偏好信息。

较低 FM MSE 并未兑现为该例的正确保持；本实验不能凭此区分训练预算不足、
目标图细节对近似误差敏感或模型未有效利用图片条件。
完整 pilot/dev 仍继续，固定64 FM8192 单写预算对照和730覆盖实验仍运行。
后续扩大保持预算或改变目标构建，应作为独立对照；不靠这一单例中途选择主实验检查点，
也不修改官方 FM 公式或用额外文本绕过 RGB 状态。

## 条件文本路径补查

另检查了未改动的官方 `a6e20c8` 条件编码代码及本实验事件序列化。
`encode_prompt` 的 `max_sequence_length=500` 参数在函数体内没有使用，
实际 processor/tokenizer 调用没有传入 `truncation` 或 `max_length`；编码后去除固定前缀，
没有在该函数后段裁剪到 500 token。pilot64、train730、dev90 的初写事件均逐字符保留
原 user 和 assistant 两条消息，最长分别为 1,503、1,503、1,512 字符。
因此此次源码检查没有支持“本实验主动截断偏好文本”的解释，没有据此改动官方编码路径。
这是源码与输入序列化核对，不等同于验证每个运行时 token，也不证明模型有效利用了偏好。
[检查记录](evidence/writer-condition-text-inspection.json)。

- [完整核验和逐例结果](evidence/training-source-B/training-source-B-verification.json)
- [16 条原始读取](evidence/training-source-B/B/readback.jsonl)
- [8 次实际输入/输出生成](evidence/training-source-B/B/writes.jsonl)
- [冻结权重与源图清单](evidence/training-source-B/B/manifest.json)
