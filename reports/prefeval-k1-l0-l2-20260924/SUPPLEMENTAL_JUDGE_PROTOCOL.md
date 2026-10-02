# 自然回答补充评分：固定 Qwen3-Max，保留官方规则

> 2026-09-24 用户后续要求切换至 Qwen3.8-Max；当前设置与重评范围见
> [切换记录](QWEN38_JUDGE_SWITCH.md)。下文保留为此前 Qwen3-Max 评分的历史协议，
> 旧成绩不与新模型混合。

2026-09-24，在读取任何此模型的正式评分前登记。
本机 DashScope 凭据经通用短回复探针验证可用；没有在日志、代码或 GPU 实例中保存密钥。
官方 Claude 3 Sonnet / Bedrock 凭据仍未配置，所以官方 judge 指标仍未完成。

补充评分固定 `qwen3-max-2025-09-23`，通过阿里云官方 OpenAI 兼容入口调用。
该 ID 是固定快照，官方文档列为非思考模式；不用随时间变化的模型别名。
模型身份与入口依据 [Qwen3-Max 文档](https://www.alibabacloud.com/help/en/model-studio/model-qwen3-max)
和 [官方调用说明](https://help.aliyun.com/zh/model-studio/model-calling-in-sub-workspace)。

保持 PrefEval 的四份原始提示（acknowledge / violation / hallucination / helpful）、
官方 XML parser 和 `analyze_errors` 聚合不变；temperature=0、每项 max_tokens=100，
不补写或延长被 Reader 截断的自然回答，不改其原有 300-token 生成设置。
输出明确保存 `model_label=substitute_judge`，原始请求提示、响应、模型名和 token 使用一并保留。
不与未来 Claude 的分数混合；报告标题必须注明替代 judge，不能称为官方模型复现分数。

先处理此前固定首例的 8 条学生自然回答，验证四项评分链路；
它们仍是一条偏好的重复图像读取，不能作为总体成绩。
后续对固定完整分母的 T1 自然回答评分；O1/O2 的结果不用于模型或参数选择。
解析失败保留原始响应并标为未完成，不能当作成功遵循、静默删除或改用 MCQ 替代。

这解决“可执行的补充自然回答评分”问题，尚不解决 judge 与 Reader 同属 Qwen 系列可能带来的偏差，
也不替代官方 Claude 对照。所有两组比较使用同一固定 judge 与同一规则。

执行补充：批量评分曾遇 HTTP 429。传输层从提交 `9c58631` 起每次请求前等待 2 秒，
仅对 429/500/502/503/504 作至多 5 次尝试的退避重试，并保留已有完整结果。
提示、固定模型、temperature、100-token 上限、XML 解析和聚合均不变；
认证失败和解析失败不作为限流重试处理。

首次 A 教师完整读回评分中，`education_learning_styles:0030` 的错配输入在 acknowledge
判定时触及 100-token 上限，未生成完整 XML。保留原始响应为 `judge_parse_failure`，
不改上限、不重试该记录、不手工补标签。执行器随后继续其他回答，重复运行也跳过已登记
解析失败；汇总单列 judge 解析失败及未评分数量，固定分母不变。
相关原始响应见 [首次解析失败](evidence/teacher-A-first-judge-parse-failure.json)。
