# 部署与首轮状态

2026-09-24，实验仍在运行，不是最终性能报告。

- 专用实例：`prefeval-k1-l012-h200x4-20260924`，4 H200，RUNNING。
  禁止使用用户已另作分配的 `dl-clear-retain-h200x4-20260914`；仅在用户澄清前对它做过只读查询，未启动任务。
- 分支：`codex/prefeval-k1-l0-l2-20260924`。
- 教师执行提交：`5153444`，四分片 PID 为 40100/40101/40102/40103；A 使用 GPU 0/1，B 使用 2/3。
- 后续执行提交：`1c4aa2a`，独立工作树 `repos/prefeval-k1-stages-1c4aa2a`。
  接续进程 PID 276625 等待全部 128 个固定教师端点，再启动配对 FM 与冻结图读回。
  后续 GPU 分配：0 A 读回，1 A Writer，2 B 读回，3 B Writer。
- 所有产物根：`/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-k1-l0-l2-20260924`。
- 最近核实完成教师：A 22/64、B 22/64，其余正在更新；该计数是时间截面，不能作为准确率。

技术验证已完成：A/B 各 12 次 latent 更新、PNG 保存/回读；各两次实际 FM 更新；
各自原生 28 步生成、PNG 落盘后再输入第二次写入。技术输出放在 smoke / writer-smoke，
正式训练不沿用其权重。首个失败仅因启动环境变量不匹配，保留在 smoke-env-failed-61ce035。

## 首个固定教师端点的 T1 读回（单例诊断）

样本 `entertain_games:0001`：官方偏好是 `I don't enjoy games with pixel art graphics.`

| 输入 | 官方 MCQ 正确项 D（Stray） | 原始自由回答观察 |
|---|---|---|
| A 教师 PNG | 错，C（Celeste） | 开头明确提到不要 pixel art |
| B 教师 PNG | 对，D（Stray） | 输出通用游戏推荐，未在开头明确此偏好 |
| 灰图 | 错，C | 通用推荐 |
| 完整文本 | 对，D | 提到 pixel art 限制 |
| 错配 A 教师图 | 错，B | 换成错误图所属的避免 horror 偏好 |
| 错配 B 教师图 | 错，C | 通用推荐 |

原始记录见 evidence/first-frozen-teacher-{A,B}-T1.jsonl，实际 PNG hash、输出 token、
选项排列和解析结果均保留。此表仅用于验证读回与干预流程；没有官方自由回答 judge 分数，
不是总体准确率、不是未训练偏好泛化，也不证明 L0–L2 已完成。

第一性原理解释：可读出训练回答相关信息与支持另一种输出任务是两回事。
A/B 原题格式对比需要完整分母；B 本例正确位置 D 还恰好等于最后训练更新的正确位置，
因此补充预先固定的 T1 四位置读回诊断，检查是内容读取还是字母位置依赖。
这个诊断不改训练预算、不使用 OOD、不挑教师，也不替代原始官方 MCQ 指标。

四位置诊断已完成：同一冻结 A PNG 为 0/4，同一冻结 B PNG 为 4/4；
B 的输出随正确位置分别为 A/B/C/D，排除了该例“只输出最后训练字母 D”的解释。
它仍不能排除对这组特定选项内容的适配，也不能据此宣称获得可迁移的偏好表征。
证据见 evidence/first-teacher-{A,B}-four-positions.jsonl。

自然回答训练输入采用官方 benchmark 的 `Please respond within 300 words` 后缀，
目标仍是官方发布的完整 SFT 回答；这是本实验的 Reader 任务适配，不声称逐 token 复现
原 Mistral SFT 输入。MCQ 采用官方原提示，Qwen 生成完整 XML，预算 32 token。

## 尚待完成

完整教师与 Writer 的统一读回；两段正式 FM；训练侧与开发侧两噪声链 0/5/10 轮保持；
L1 单有效偏好的覆盖探针；根据内部开发 T1 定位瓶颈后的修正与扩容；冻结后的 180 官方评估。
自由回答评分服务尚待配置（已有 gateway 返回 401），训练与 MCQ 不受此阻塞。
不把没有执行的阶段或缺少 judge 的指标写成成功。
