# 固定首例：B 的 MCQ 读回通过 Writer RGB 保留，A 尚未通过

2026-09-24。样本仍为登记序列首条 `entertain_games:0001`，偏好为
`I don't enjoy games with pixel art graphics.`。本诊断使用固定 2,048 步 Writer 终点，
无挑图、无改预算；每组读两条预定噪声链的实际落盘 PNG。
此处只有一条已训练偏好，不能作为总体成绩；随后完成的全量 pilot 另见
[64 条结果](WRITE_PILOT_T1_RESULTS.md)。

| 输入 | MCQ 输出 | 正确数 |
|---|---|---:|
| A 教师图（此前结果） | C：Celeste | 0/1 |
| A Writer 匹配图，chain 0/1 | C / C | 0/2 |
| A Writer 错配图，chain 0/1 | C / C | 0/2 |
| B 教师图（此前结果） | D：Stray | 1/1 |
| B Writer 匹配图，chain 0/1 | D / D | 2/2 |
| B Writer 错配图，chain 0/1 | C / C | 0/2 |

题目与官方选项排列在匹配/错配条件间相同。错配图来自同主题的
`entertain_games:0007`，更换的是图片，不是题目。MCQ 均正常输出完整 XML，
没有解析失败或截断。两条链是同一偏好的重复生成，不是两个独立偏好样本。

这一例中，B 的正确 MCQ 读回已从直接优化的教师图转移到共享 Writer 生成的 RGB，
且换图会改变结果，支持图像携带了对该题有用的状态。它仍可能是针对训练题/选项的表征，
尚不能证明新偏好、新语义用途、自然回答或递归保持。

后续独立四位置探针已在同样的两张冻结学生图上完成：B 每条链均依次正确输出
A/B/C/D，共 8/8；A 每条链均输出 B/A/A/A，共 0/8。无截断，图片哈希与原读回完全一致。
该结果排除了 B 在这一例只输出原位置 D 的解释，但不能排除对已有问题/选项内容的适配，
也不能代替全部 64 条的位置测试。执行代码固定为 `2bcd0f1`，没有修改主实验训练或评测。

A 的教师原题 MCQ 本来就错，因此本例 A 的 MCQ 错误不能全部归因于 FM。
此前 A 教师自由回答开头明确提到避免 pixel art；本次学生自由回答则变为通用游戏推荐。
B 教师此前也未在自由回答中明确体现该偏好，本次学生仍如此。两组本次全部 8 条自由回答
均达到 300-token 上限，保留原始截断记录；没有官方 judge，不能据此填报自由回答准确率。
这提示需要继续区分教师的输出格式迁移与 Writer 的蒸馏落差，而不能用 MCQ 成功覆盖二者。

四张匹配 PNG 的实际哈希均与自由回答/MCQ 的读回记录一致，生成 manifest 绑定到
已核验的 A/B 固定 2,048 步检查点。生成仍为原生 28 步、CFG=1，初写来源为灰图，
本诊断没有 latent 旁路。原图外观为抽象纹理，外观本身不构成语义证据。

- [A 原始输出](evidence/first-student/student-first-T1/A/readback-0.jsonl)
- [B 原始输出](evidence/first-student/student-first-T1/B/readback-0.jsonl)
- [单例汇总与固定分母](evidence/first-student/summary.json)
- [四位置原始输出 A](evidence/first-student/student-first-four-positions-A.jsonl)
- [四位置原始输出 B](evidence/first-student/student-first-four-positions-B.jsonl)
- [A chain 0 原图](evidence/first-student/writer/A/write-pilot/entertain_games_0001/seed-0/prefix-00.png)
- [B chain 0 原图](evidence/first-student/writer/B/write-pilot/entertain_games_0001/seed-0/prefix-00.png)
- [此前教师诊断](FIRST_TARGET_DIAGNOSTIC.md)

主流程继续完成全部 64 条已训练偏好、90 条未训练偏好的 T1 与 OOD 读回，以及
学生 RGB 前缀训练和重生成的 0/5/10 保持链。该单例不用于停止训练或选择更好看的检查点。
