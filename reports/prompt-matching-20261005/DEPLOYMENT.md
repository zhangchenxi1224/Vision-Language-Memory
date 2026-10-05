# 历史条件软监督部署记录

2026-10-05。实现经 PR #2 合入 main（4cddd5c6679d5ca537177d04ce5bc8c3c9b527f9）；远端实验固定使用同一实现提交 a48422b250cb62a3b4ce31e236a0ac35949dcefd，后续文档更新不修改运行中的 checkout。

## 实现与兼容边界

- `hard_ce`：原有标准答案监督，继续作为默认。
- `history_hard`：同一冻结 Reader 读取当前已观察的初始 exchange，生成精确回答 token，使用硬标签监督。
- `prompt_matching`：在同一回答 token 前缀上匹配完整词表分布，温度 1，教师分布 detach。
- 新模式共用不可覆盖的教师缓存，绑定 Reader 实际权重、tokenizer、历史、问题、真实 EOS 与源码身份。
- 学生 Reader 只看记忆图片和问题；Writer 的历史输入边界、官方 FM、28 步生成、实际 PNG 持久状态保持原约定。
- 新增 Writer 目标监督身份校验；原 A/B 问答格式与旧教师 manifest 保持兼容。

原理、完整前瞻判定标准与运行方法见 ../../docs/PROMPT_MATCHING.md。

## 已完成验证

- 本地初轮定向回归：155 passed，1 skipped（本地未放置固定官方 DreamLite 源码）。缓存与恢复补强后的定向回归：94 passed。
- 实际 GPU Python 3.12.3 / Torch 2.7.0a0+ecf3bae40a.nv25.02 / CUDA 12.8 / Transformers 4.57.3：116 项定向测试通过。
- 2×H200 环境中，三种监督各完成同一训练样本的 12 步技术短测，均有非零有限 latent 梯度；三个 PNG 均成功保存和重新读取，哈希正确，生成达到真实 EOS、没有截断。
- 三组训练与独立 PNG probe 总计约 102 秒。该短测不构成准确率或软监督优越性的证据。
- 正式 evaluator 仍拒绝把不足 288 步的教师当作已完成正式目标；技术短测使用独立 probe。

## 已启动的固定预算对照

实例：`vlm-dreamlite-full-h200x2-20260720`，2×H200。项目根：`/inspire/ssd/project/exploration-topic/czxs26210936`。

- checkout：`repos/prompt-matching-20261005`，固定实现 SHA，干净 detached checkout。
- 技术短测：`runs/prompt-matching-20261005/smoke-B`，已 completed。
- 当前对照：`runs/prompt-matching-20261005/pilot-B`；启动 controller PID 79985 仅为历史线索，后续须核验实际命令、主机和锁。
- 日志：同级 `pilot-B-launcher.log`；各阶段回执在 `pilot-B/receipts`，逐作业日志在 `pilot-B/logs`。
- 三组均为 B 格式，固定 pilot64，每样本 288 教师步；Writer 同一 4f 父模型、2048 FM 步、effective batch 4。
- 随后生成 pilot64 和 dev90 的两噪声实际 PNG，使用相同五种问法、memory/blank/mismatch/text 对照读取 MCQ。
- controller 墙钟上限 8 小时；不自动扩大预算、样本或 GPU 数量。
- 本记录写入时，hard_ce 两分片各已完成 2/32 条完整教师训练，GPU 活跃，未见错误。后续进度以现场 status/receipts 为准。

## 结果与替换状态

当前没有新的完整质量结论，默认仍为 hard_ce。MCQ 初写 pilot 不能满足完整晋级条件；须补齐自由回答、共享 Writer 泛化、长期保持和图像因果对照，并与两个参考监督做完整配对比较。小样本 loss 或教师图提升不触发默认替换。

新的周期跟进未创建：自动审批要求补充明确的持续调度授权，已向用户发出请求。已有暂停的旧实验自动任务保持不变。本次已启动的固定预算 controller 不依赖周期跟进，继续按原计划运行。

原始技术与启动证据：`gpu-deployment-evidence.json`，SHA256 `f4c6c18e489cb9bd762a699f7551be7193c7ce2035baaf7b0baf0a6a2691edbd`。
平台快照未报告自动停止倒计时（auto_stop_in_seconds=0）；controller 的 8 小时上限约束本次子进程，不等于实例自动释放。
