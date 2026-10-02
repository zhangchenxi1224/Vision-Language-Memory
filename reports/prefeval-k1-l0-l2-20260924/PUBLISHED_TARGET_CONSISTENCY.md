# 发布目标诊断：52/64 通过替代评判，不能作为记忆准确率

2026-09-24。按[预先登记](A_TEACHER_FAILURE_DIAGNOSIS.md)直接评判固定 64 条 pilot 的
完整官方 `response_to_q`，没有 Reader 生成、图片输入或二次裁剪。
固定 `qwen3-max-2025-09-23`，保持官方四项提示、parser 和聚合；结果为 **substitute_judge**。

64 条全部完成：**52 条通过、12 条未通过（81.25%）**，无未评分项、解析失败或 judge 长度截断。
共 256 次分项调用、141,709 token，实际响应均为固定模型快照。
这项分数描述发布的监督回答在该评判器下的表现，**不是视觉记忆准确率、人工数据质量标签或性能上限**。

## 内容矛盾与 judge 解释要分开

全部 64 条偏好、问题和目标回答已对照 16 份固定官方提交文件逐字符验证一致，
也与本次 judge 输入一致。不是本实验适配或 Reader 截断造成了以下文本差异：

- `education_learning_styles:0001` 明确不喜欢闪卡，发布回答却在第 6 项建议制作闪卡。
- `travel_restaurant:0030` 要避免高楼与屋顶餐厅，发布回答仍推荐自称位于 95 层的餐厅，
  并把 96 层的另一场所作为感觉太高时的替代。这是原回答内部即可看出的条件冲突。
- `lifestyle_dietary:0042` 的仅野外采集条件与购买蘑菇的建议相冲突，详见此前源核对。

但不能将 12 条自动未通过全部当作已人工确认的数据错误。
例如 `entertain_shows:0037` 的偏好限定“不看由书或小说改编的节目”，judge 却用
“根据真实事件和杂志文章改编”作为违规理由之一；这一理由扩大了原偏好的范围，
不能仅凭该解释确认违规。本轮不改动它的原评分，公开原始判词并保留局限。
`entertain_sports:0009` 的发布文字在 `Zipl` 处结束，judge 将其推断为 ziplining；
这种对残缺片段的补全也属于不确定解释。其他涉及产品配方或节目背景的理由仍需外部事实复核。

这些例子用于说明监督内容和评判器都有局限，没有筛除样本、修改官方目标或手工重标成绩。
发布回答本身的尾部残缺也原样保留；“完整目标”在此指完整保存的发布字符串，
不等于每条发布回答都在语言上完整。

## 监督问题不能解释 A 学生的大幅退化

以下仅作事后诊断，主指标分母仍为 64，没有按这项评分重新定义训练集或成功标准：

| 发布目标的替代判定 | A 教师匹配图 | A 学生 chain 0 | A 学生 chain 1 |
|---|---:|---:|---:|
| 通过的 52 条 | 38/52 | 3/52 | 3/52 |
| 未通过的 12 条 | 2/12 | 1/12 | 1/12 |

即使在同一 judge 判定发布目标通过的记录中，共享 Writer 成功计数仍明显少于教师图。
而目标未通过时也有图像回答通过，进一步说明目标诊断不是图像性能上限。
这支持继续检查固定目标的 FM 拟合，而不是把所有失败归因于数据质量。
已登记的 [8,192 步 FM 对照](FIXED64_FM8192_PROTOCOL.md)和 RGB 保持对照继续，官方目标不变。

另从原始训练日志核对：A 的目标含结束符为 126—267 token，中位数 218，全部低于 300-token 预算；
B MCQ 目标均为 7 token。128 份轨迹哈希与原教师训练汇总相同。
因此大量 A 学生回答截断不能简单归因为监督目标长度本来超过上限。
这不保证自由生成采用同样长度，也不排除生成偏移或预算限制对部分自由回答的影响。

## 可复核记录

- [完整统计、未通过 ID 和 judge 哈希](evidence/published-target-consistency-summary.json)
- [64 条封存原始评分](evidence/judge-qwen3max-published-target-consistency-finalized)
- [目标判定与图片回答逐条配对](evidence/published-target-vs-image-answer-paired.json)
- [全部 64 条官方源核对](evidence/published-target-all64-upstream-verification.json)
- [原训练实际目标 token 长度](evidence/pilot-answer-token-lengths.json)
- [固定 64 条原始输入](evidence/published-target-consistency-input/published-answers.jsonl)
