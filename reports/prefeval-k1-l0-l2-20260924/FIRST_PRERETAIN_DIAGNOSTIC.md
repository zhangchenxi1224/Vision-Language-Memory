# 固定首例：B 的可读性在第一次干扰更新后丢失

2026-09-24。样本仍为此前固定的登记首条 `entertain_games:0001`：
`I don't enjoy games with pixel art graphics.`。使用固定 2,048 步单写 Writer，
seed-0 的真实落盘 RGB 链；保持 FM 尚未开始。正确 MCQ 为 D（Stray），
错误项 C 为 Celeste。问题和选项顺序在所有端点、匹配/错配条件间保持相同。

| 图片 | 初写 0 | 干扰 1 | 干扰 2—4 | 干扰 5 | 干扰 6—9 | 干扰 10 |
|---|---|---|---|---|---|---|
| A 匹配链 | C 错 | C 错 | 均 C 错 | C 错 | 均 C 错 | C 错 |
| A 错配链 | C 错 | C 错 | 均 C 错 | C 错 | 均 C 错 | C 错 |
| B 匹配链 | D 对 | C 错 | 均 C 错 | C 错 | 均 C 错 | C 错 |
| B 错配链 | C 错 | C 错 | 均 C 错 | C 错 | 均 C 错 | C 错 |

先完成固定 0/5/10 读取，观察到 B 在第 5 步已错，再对同一冻结链补读全部 0—10 步定位。
后者是事后单例诊断，不用于挑选模型、训练样本或改变预算。
两次读取中重合的 12 条记录完全一致，所有输出均有合法 XML、无截断。
这里只有一条偏好、一条生成链，不能用 11 个端点充当 11 个独立样本，
也不能代替已经部署的 64 条全量 0/5/10 保持参照。

## 首次失败的输入与含义

第一次干扰来自原官方 SFT context pool，是关于宠物主人应购买无毒植物以免猫生病的
用户故事选择题及其回答，与像素风格游戏偏好无关。Writer 接收当前完整交换和前一张 PNG，
没有收到未来游戏问题、答案或完整历史，也没有额外的 oracle RETAIN 标签。

这定位了本例 **从正确初写到不可正确读取的第一次状态更新**。它说明当前单写模型
并未自动保留这条已学偏好，不能仅解释为很多步累积后才出错。
但 MCQ 失败不证明像素里已经没有任何可恢复信息，也不能凭这一个例子判定所有偏好都在一步后失败。
A 在起点已经错误，因此本例不能量化 A 的遗忘幅度。

单写 FM 的训练输入全部为灰图，目标是写入当前偏好；遇到非灰图历史状态和无关交换时，
条件分布已经改变。当前结果与缺乏新表征的保持训练相符，但尚不是具体机制的因果证明。
已登记的第二段 FM 将使用学生真实前缀继续训练保持，并从灰图重新生成完整两噪声链评测，
不能用缓存的训练前缀或教师图替代保持后生成结果。

这与 [90 条未训练偏好单写失败](WRITE_DEV_T1_RESULTS.md) 分属不同位置：
一个是新内容首次写入不成功，另一个是已写入内容在更新后不可正确读取。
730 条覆盖实验针对前者；当前保持训练及再生成评测针对后者，二者均须完成。

## 实际 RGB 链核验

本次观察时每组前 11 条完整链的 110 次转换已核实：每个输出 PNG 的实际哈希与记录一致，
下一步输入哈希就是前一步输出哈希，每步事件与官方当前交换一致；
每条初始 PNG 还与此前单写 seed-0 图片逐字节相同。
这是局部结构核验，不能写成全部 64 条完成，更不能写成记忆准确率。

- [部分链结构核验](evidence/rgb-prefix-chain-verification-partial.json)
- [固定 0/5/10 原始记录](evidence/first-preretain/first-preretain-verification.json)
- [逐步原始记录 A](evidence/first-preretain-dense/student-first-preretain-dense/A/readback-0.jsonl)
- [逐步原始记录 B](evidence/first-preretain-dense/student-first-preretain-dense/B/readback-0.jsonl)
- [实际写入交换与哈希链](evidence/first-preretain-dense/writer/B/training-prefixes/entertain_games_0001/seed-0/writes.jsonl)

读取复用冻结提交 `6561833`；未修改正在运行的 Writer、官方 DreamLite 或 MCQ 评分。
