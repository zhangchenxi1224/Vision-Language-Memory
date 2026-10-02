# 保持后固定首例：第一次干扰仍然失败

2026-09-24。沿用此前预先固定的 `entertain_games:0001`，要求避免像素风游戏；
错配对象仍为 `entertain_games:0007`。使用保持阶段固定 2,048 步最终权重，
从灰图重新生成两条噪声链，对每个 0—10 轮端点读取 T1 官方 MCQ。
本次是单例定位，不是完整 64 条指标，不用于选择检查点或 OOD 参数。

官方正确选项在此次固定排列中为 D。结果如下：

| 模型与输入 | 初写（0） | 第 1 轮 | 第 2—10 轮 |
|---|---|---|---|
| A 匹配图，chain 0 | A，错 | C，错 | 全为 C，错 |
| A 匹配图，chain 1 | C，错 | C，错 | 全为 C，错 |
| B 匹配图，chain 0 | D，对 | C，错 | 全为 C，错 |
| B 匹配图，chain 1 | D，对 | C，错 | 全为 C，错 |

两组在第 1—10 轮的错配图也均输出 C。
与 [保持前相同首例的 seed 0 逐轮结果](FIRST_PRERETAIN_DIAGNOSTIC.md)对照，
B 初写仍正确，但第一次干扰仍失去正确读取，固定保持预算没有修复这一例。
A 起点已经错误，不能把它之后的全部错误单独归因于遗忘。
保持前该首例的完整逐轮参照只有 seed 0，不能声称对 seed 1 完成了完整的训练前后逐轮配对。

两组各 44 条 MCQ 均无解析失败或生成截断，已独立用官方 parser 再验。
实际核对了每组 4 条链（匹配/错配 × 两噪声）、44 张 PNG、40 次递归转换，
输入/输出图像哈希与官方顺序交换一致；最终生成绑定的是各组保持后权重。
这里没有沿用训练前缀冒充新链，也没有把旧 latent 直接传给下一步。

## 下一步定位

完整 pilot/dev 两噪声链和既定 0/5/10 读回继续，不从单例外推所有偏好。
只读定位将先区分“冻结图片经过实际 VAE 编码/解码后是否仍能读出”与
“加入 Writer 更新后是否丢失”，使用相同固定首例和原问题，不改 FM、数据或检查点。
即使 VAE 往返后的 Reader 失败，也只能说明这一读取链对往返敏感，
不能直接证明编码 latent 完全没有任何信息。

- [44×2 条读回与 PNG/转换核验](evidence/first-postretain-dense/first-postretain-dense-verification.json)
- [A 原始 MCQ](evidence/first-postretain-dense/student-first-postretain-dense/A/readback-0.jsonl)
- [B 原始 MCQ](evidence/first-postretain-dense/student-first-postretain-dense/B/readback-0.jsonl)
- [启动参数](evidence/first-postretain-dense/student-first-postretain-dense-launch.json)
