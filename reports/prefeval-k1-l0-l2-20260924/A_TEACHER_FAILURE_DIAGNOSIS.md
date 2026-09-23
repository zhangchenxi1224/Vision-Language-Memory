# A 教师错误定位：监督内容与自由生成偏移需要分开

2026-09-24。基于已完成的固定 64 条 T1 教师读回与替代 judge，事后描述分析，
不使用 OOD，不改变现有训练、检查点或分母。

按 A 自身最后 12 次前向平均 CE 从低到高排序，每组固定 16 条：

| CE 四分位 | 匹配图自然回答正确 /16 | 错配图正确 /16 | 匹配图输出截断 /16 |
|---|---:|---:|---:|
| 最低 | 13 | 2 | 1 |
| 第二 | 10 | 1 | 4 |
| 第三 | 8 | 5 | 10 |
| 最高 | 9 | 2，另 1 条未评分 | 13 |

低 CE 组整体更容易读出符合偏好的自然回答，但并非单调关系，也不保证个体成功。
这些关联不能证明增加教师更新次数必然解决错误；目标内容、答案长度、生成截断也同时影响结果。
两组监督长度不同，此处不将 A、B CE 数值直接比较。
[分组成员与输入哈希](evidence/teacher-A-loss-vs-free.json)可复算。

## 两个低 CE 错误有不同来源

`lifestyle_dietary:0042` 要求食物仅来自当地野外采集。发布的自然回答却允许在没有野生蘑菇时
购买蘑菇；A 图读取也保留了这个建议。该句与“仅野外采集”的条件直接冲突。
本实验适配记录与固定官方提交的偏好、问题、`response_to_q` 逐字符一致。
因此至少这一项错误已存在于监督内容，不能仅凭未通过遵循评判就推断偏好没有写进图像。
原始记录见 [PrefEval 发布文件第 42 条](https://github.com/amazon-science/PrefEval/blob/50795054b5ff5f418d2b768a331d71e480f93331/SFT/single_pref_remind/lifestyle_dietary/mistral8x7b_lifestyle_dietary_2turn.json)。

`entertain_music_book:0016` 则不同。发布回答的非虚构推荐为 Sapiens、Bad Blood 和
The Immortal Life of Henrietta Lacks；A 图的自由生成把后两项改为 The 7 Habits 和 Atomic Habits。
其开头仍明确说避免自助励志书，替代 judge 判为违反偏好。
这里存在实际的自由生成偏移，不能把错误全部推给发布的目标回答。
原目标见 [PrefEval 发布文件第 16 条](https://github.com/amazon-science/PrefEval/blob/50795054b5ff5f418d2b768a331d71e480f93331/SFT/single_pref_remind/entertain_music_book/mistral8x7b_entertain_music_book_2turn.json)。

两份在线固定提交文件均核对过内容。既有源清单记录的是 Windows CRLF checkout 哈希，
与在线 LF 文件的原始字节哈希不同；仅转换行结束符后完全匹配，JSON 字符串内容未变。
[核验记录](evidence/teacher-A-target-source-verification.json)同时保存两种哈希。

## 下一项固定诊断：直接评判发布的 64 条目标回答

为判断这类监督问题覆盖多少条，预先封存全部固定 64 条 pilot 的官方 `response_to_q`，
不只挑上述失败案例。接续在 A/B 学生单写评分完成之后，用同一固定
`qwen3-max-2025-09-23`、PrefEval 四项提示、100-token/temperature=0、XML parser 与聚合评判。
保持传输重试规则；解析失败不重采样，留为未评分。输入是原始发布回答，不调用 Reader 重新生成，
也不做 300-token 二次裁剪、文字补全或清洗。

该结果仅衡量发布监督内容在同一替代 judge 下的遵循情况，**不是视觉记忆准确率或性能上限**。
它仍可能包含 judge 的事实误判。现有官方目标保持原样，所有样本仍进入主实验分母；
不根据该结果偷偷替换标签、过滤训练集或调整当前预算。
后续修正要先区分目标本身的矛盾、教师自由生成偏移及教师到共享 Writer 的额外损失。

输入：[64 条发布回答](evidence/published-target-consistency-input/published-answers.jsonl)、
[固定清单](evidence/published-target-consistency-input/manifest.json)。本次登记时尚无该诊断评分。
