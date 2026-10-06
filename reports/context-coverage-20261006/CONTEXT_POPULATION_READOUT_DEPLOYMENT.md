# 16→32 历史覆盖：分层读回执行登记

2026-10-06 10:52 CST，目标/训练阶段已完成并独立复核；本登记在新增读回和分数产生前冻结。科学协议完全沿用 CONTEXT_POPULATION_PLAN.md，不改变数据、问题、预算或比较方式。

新消费者 scripts/inspire/run_context_population_readout.py 只生成缺失的96张PNG和2688条读回，与1152旧行合并3840行。原/新增各16历史独立统计；各自主题内互换donor，保留旧原16 donor。Writer16 SHAb2527682…、Writer32 SHAbffc31bc…均固定128最终权重。新增192教师分布使用相同Reader和逐token目标管线，原192只读复用。文本是自一致性检查，不是准确率。

源目标/训练已经通过完整哈希和张量检查：32教师（其中新增16共4608有效更新），128 Writer更新、512draw、每ID16次；无活动owner，旧4PID退出。原始归档population-complete-0244-evidence.tar.gz，SHA24bfc8d70b905dd9138567a1a349d0b77fe545ad62b2165f287531bdc98da4c0。目标/训练结算0.537744240231 GPUh，全任务结算5.874805550178。

新独立checkout予定为 context-population-readout-20261006，output为本轮run/context-population-readout-v1。控制器须验证旧全量fit报告及来源、原/新增目标bank、固定权重、新PNG的历史/噪声与SHA，再启动两分片单例读回；最后核验所有384教师张量与精确分母并生成comparison.json。不得修改旧执行树或源输出。

整个population迭代仍为3GPUh，扣目标/训练0.537744240231后本阶段最多2.462255759769，失败/重启也扣除，计入原16池。最多2GPU，每次≤6h，启动前重新检查平台、其他owner/预留、实际PID/GPU与内存。Windows相关28检查通过；远端Linux验证/冻结执行SHA/实际启动记录待后续追加。当前未启动读回。
