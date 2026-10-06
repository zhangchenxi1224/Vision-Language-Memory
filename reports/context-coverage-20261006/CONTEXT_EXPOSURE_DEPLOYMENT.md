# 固定32历史的曝光对照：部署记录

2026-10-06 11:57:11 CST，在原vlm-dreamlite-full-h200x2-20260720启动controller2556692。Linux与Windows相关36检查通过，包含真实Adam和RNG续接相对于连续训练的逐张量一致性。执行cc3dffc864ccedc1ef354012e638190055299726，开发01b39a8c0622a585a7293bd7d54bf53ff1aac0a1，同tree3c8dc0d7562f4d8e28b329fdf51ae9010ab51f55。独立checkout context-exposure-20261006，output本轮run/context-exposure-v1。

此前population全部worker/owner已退出，两GPU现场无计算、内存充足；5个外部owner属于其他实例/fixture，完整资源元数据检查通过。ARIS tracker最新03:07UTC只有CPU续跑入口准备，其额外预算仍待批，本任务未改变其任务或资源。

控制器已完成原3840行/128PNG/384教师分布的完整来源检查并将128步resume在新output中登记为总256预算，参数、Adam、RNG与游标完整保留，原文件只读；11:58:35进入训练并创建GPU0 worker2563906。先训练129–256，再生成64PNG，再两GPU各768行读回，合并旧3840行，总5376；原/新增两个n16分层，固定目标和问法，详情CONTEXT_EXPOSURE_PLAN.md。

预算上限1.5GPUh计入原16池，之前结算6.315022416578；按attempt累计，不按阶段或恢复重置。单次≤6h、最多2GPU，不重复目标制作或前128步，不空转。当前训练进度与真实GPU运行仍待下一条现场验证，科研结果pending。

## 实际训练核验

当前以此条为准（2026-10-06 12:00 CST）：population覆盖16→32的3840行实验已完整核验为阴性，见CONTEXT_POPULATION_RESULT.md；默认不晋级。后续context-exposure-v1已真实训练到171/256，新增43更新梯度有限非零，原128日志字节前缀、1075参数/Adam/RNG完整续接核验通过，科学读回pending。

独立checkout context-exposure-20261006，执行cc3dffc864ccedc1ef354012e638190055299726、开发01b39a8c0622a585a7293bd7d54bf53ff1aac0a1，同tree3c8dc0d7562f4d8e28b329fdf51ae9010ab51f55。Windows/Linux各36检查通过，执行源码clean。controller2556692、train worker2563906，CUDA0实际约39GiB；GPU1暂无任务但后续两分片读回需双卡，不额外插任务。旧population controller及4worker均退出，预检无本host其他计算/预留/owner，未动ARIS其他资源。

按CONTEXT_EXPOSURE_PLAN.md，唯一因素为同32目标累计更新128→256；不重制目标、不重算前128、不重置优化器。控制器完成256后核验总1024draw、32ID各32次、全256连续有限梯度与resume/final一致，再生成64新PNG、两分片各768读回，复用旧3840行合并5376；所有384教师缓存只读，不得缺失时生成。原/新增各n16分层、匹配/分层内同主题错配/灰图/文本全保留，固定final256，不新评分dev/official。

下次先读DEPLOYMENT.json及ARIS tracker，再查context-exposure-v1/status、active-owner、attempts/receipts、真实PID/cmdline/GPU、优化日志/检查点与PNG。运行中不重复启动或修改执行树。完整后用冻结cc3dffc的run_context_exposure.report全量重算（同时验证旧population报告），核验192总PNG、384教师张量、5376精确分母与分层偏好级区间。旧population结果和所有父权重只读。

此前全结算6.315022416578GPUh，新曝光阶段暂计0.025515611635，总暂计6.340538028214（采集时点12:00:07）。本新迭代上限1.5GPUh计入原16池，训练/生成/读回/失败共用；最多2GPU、单次≤6h，禁止重启重置/空转。若部分训练中断，先证明旧worker停止、结算与归档owner，核验resume128/256及未提交日志尾部；当前控制器拒绝未经审计的partial直接重跑。

现场与恢复证据exposure-0345-evidence/，归档SHAe0c8de976564361541f266fd80a55d2cf7791e640893ec80bb6fa23f009b34a4，部署说明CONTEXT_EXPOSURE_DEPLOYMENT.md。第十次触发2026-10-06T03:45:35.985Z已执行，每小时继续，仅实质变化/完成/故障/需用户处理时通知。不要改其他自动任务或复活取消的ARIS A64/B730。
