# 16→32 历史覆盖：分层读回执行登记

2026-10-06 10:52 CST，目标/训练阶段已完成并独立复核；本登记在新增读回和分数产生前冻结。科学协议完全沿用 CONTEXT_POPULATION_PLAN.md，不改变数据、问题、预算或比较方式。

新消费者 scripts/inspire/run_context_population_readout.py 只生成缺失的96张PNG和2688条读回，与1152旧行合并3840行。原/新增各16历史独立统计；各自主题内互换donor，保留旧原16 donor。Writer16 SHAb2527682…、Writer32 SHAbffc31bc…均固定128最终权重。新增192教师分布使用相同Reader和逐token目标管线，原192只读复用。文本是自一致性检查，不是准确率。

源目标/训练已经通过完整哈希和张量检查：32教师（其中新增16共4608有效更新），128 Writer更新、512draw、每ID16次；无活动owner，旧4PID退出。原始归档population-complete-0244-evidence.tar.gz，SHA24bfc8d70b905dd9138567a1a349d0b77fe545ad62b2165f287531bdc98da4c0。目标/训练结算0.537744240231 GPUh，全任务结算5.874805550178。

新独立checkout预定为 context-population-readout-20261006，output为本轮run/context-population-readout-v1。控制器须验证旧全量fit报告及来源、原/新增目标bank、固定权重、新PNG的历史/噪声与SHA，再启动两分片单例读回；最后核验所有384教师张量与精确分母并生成comparison.json。不得修改旧执行树或源输出。

整个population迭代仍为3GPUh，扣目标/训练0.537744240231后本阶段最多2.462255759769，失败/重启也扣除，计入原16池。最多2GPU，每次≤6h，启动前重新检查平台、其他owner/预留、实际PID/GPU与内存。Windows相关28检查通过；远端Linux验证/冻结执行SHA/实际启动记录待后续追加。当前未启动读回。

## 实际部署

2026-10-06 11:04:25 CST：context-population-readout-v1已在原H200x2真实运行。独立checkout context-population-readout-20261006，执行cb045675baccff3b0b00002f2865ee77b89da4a4、开发00bc27f8ce07e3b2bc528b1218657ccaa1444037，同treea5c248903ca397f1e63684d008b7a0f524381d60，源码clean。Windows/Linux各28检查通过。controller2261869，Writer16 worker2265344/CUDA0、Writer32 worker2265345/CUDA1，每卡约14GiB。首批4/32与4/64新PNG的RGB1024、历史、噪声、收据及SHA通过。新增读回0/2688，效果pending。

源目标/训练全部完成：32教师、新增4608教师更新、Writer128更新/512draw/每ID16次，Writer32最终bffc31bc526c8955675ba4d76c0a85ede2b47522db8d410bc15ee8f7c70ae051。新增64组中间PNG/latent快照哈希通过，旧4PID退出，无旧owner，不重跑源训练。

新控制器自动96新PNG→2688新行，复用1152旧行得3840总行；192旧+192新教师分布。原/新增各16分层、分层内同主题donor、偏好级配对区间；不能池化成泛化。所有权重固定final128，无新dev/official评分或选择，默认不晋级。完成后用冻结cb04567中的run_context_population_readout.report全量独立重算，验收128PNG/384张量、共同prefix和完整分母。旧源/cache/图片只读，运行中不改执行树。

结算全任务5.874805550178 GPUh，其中目标/训练0.537744240231；新读回暂计0.041299844649，总暂计5.916105394827/16。population整个迭代3GPUh包含目标/训练及所有读回attempt，失败/恢复也扣除。最多2GPU、每次≤6h。失联先证明旧worker停止，再归档owner、结算、审计partial身份后恢复，禁止空转保活。

资源预检5个owner均属于其他实例/测试fixture，本host无其他计算/预留，内存足够。早先CPU预检长目录扫描未启动GPU；最终v2完成元数据清查后只启动一次。随后v3被非空输出guard拒绝，prelaunch-error-0244.json是保护记录，不是当前controller失败，不能据此重启。现场与全部诊断population-readout-0244-evidence/，归档SHA34318464ae0bb6a17a6220d9f0a230d1b853b7d58bcfc390950d943dd4b6dca9。
