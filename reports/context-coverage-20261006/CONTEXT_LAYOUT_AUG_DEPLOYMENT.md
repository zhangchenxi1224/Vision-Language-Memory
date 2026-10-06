2026-10-06 15:42:57 CST 实际续训确认：两臂均277/384，CUDA0/1 worker189888/189889的PID/cmdline/环境与GPU进程一致，各约38GiB。连续日志与有限非零梯度通过；全任务暂计7.822156585190GPUh，内含已结算7.773129298223。最新证据layout-aug-live-training-0650.json。下方source_audit为恢复早期快照，不能据此重复启动。

# 布局增强：存储故障恢复与后续衔接

真实GPU原生257–258校准已通过：两臂参数、Adam、RNG、梯度/MSE/抽样轨迹与原生精确一致。随后两臂均在step288原子保存因共享SSD满而失败，controller55309与所有5个worker实际退出、owner消失。完整resume仍为258且SHA和固定parity快照相同；失败不是数值或科学效果结论。

仅将本轮native/canonical/augmented目录全部复制到QB的runs/context-coverage-20261006/checkpoint-store/layout-aug-v1，逐文件大小与SHA相等后才替换自身SSD目录为链接，旧pilot/其他任务没有修改。失败resume.pt.tmp与259–288日志完整归档，两臂工作日志恢复258，完整Adam/RNG保持。完整恢复证据layout-aug-recovery-0650-evidence/，压缩包SHA602d217daf818198c32ee555ca81d90bc9c1c519ce686058005f00ecd771b017。不要把这30行失败尾当成有效训练步数或再次覆盖档案。

15:36:54 controller164144已按同844dda1冻结命令重新启动；15:39:51真实cmdline存活，处于source_audit，两GPU暂为空，校验后将跳过已完成native/两臂258收据，只恢复258→384。当前已结算7.773129298223/16GPUh（本轮0.117278591394，内含失败0.059778247409）。GPUh按GPU worker进程墙钟计，平台闲置点券/CPU复制审计未包含；后续attempt继续累计，不能重置0.75训练/3整轮/16全任务预算。

同实例dl-context-dev-n2-1006，约17:05 CST自动停止。一个无GPU的有界衔接进程173891等待完整training-audit/ready_for_frozen_readout、旧worker退出、owner消失，重新扫描资源预留并核验GPU后，才调用已冻结消费者；最晚16:00未就绪则结束，不盲启。状态见本轮readout-handoff-0650.json，先核对它，禁止另起重复读回。新读回图片也落独立QB目录，三格式384PNG/9216新行/16128总行，代码和协议不变，读回尚未启动/尚无增强结果。消费者仍按失败在内的原预算扣费。若后续训练或衔接失败，先证明停止并保存证据，再恢复；不得改正在运行的冻结代码。

## 先前状态（仅供溯源）

# 布局增强同预算对照部署

2026-10-06 15:17:23 CST，controller55309启动，source_audit进行中。实例dl-context-dev-n2-1006，qb-prod-gpu800，官方ngc25.02/CUDA12.8，2H200/40CPU/400GiB，当前约17:05自动停止。

执行844dda1819168b551882577a2946548dc8602f2f，开发14c097248b7e416e7079b7b227f10f4a60a1cbd9，同treea8aba6633edbe3c99d133f262f839b6947c300c3。独立repos/context-layout-aug-20261006-v2，output context-coverage-20261006/context-layout-aug-v1。Windows/Linux38项检查通过。协议CONTEXT_LAYOUT_AUG_PLAN.md已在任何训练之前冻结，未因部署修复改变科学因素。

先真实native两步精确参数/Adam/RNG/梯度轨迹校准；两臂同256状态各新增128步到384，唯一因素canonical-only vs均衡canonical/markdown。原生校准单独计费，新两臂校准步复用于正式128步，不重跑。训练阶段0.75GPUh内，完成到ready_for_frozen_readout后退出。三格式消费者同冻结树已实现并通过测试，训练完整后才启动；读回2.25GPUh且整轮train+readout≤3，全部在原16GPUh池。启动前累计7.655850706829GPUh。

首次f0716ee controller2438仅CPU源审计因旧报告绝对路径身份失败，无GPU attempt/训练。确认PID/owner退出、attempts为空后归档失败output，保留旧代码；v2调用父报告原9cf6ea7执行目录复算，本轮身份使用固定输出plan，不改写旧报告。失败GPU成本0，平台CPU等待/空闲点券不在进程GPUh口径内。

上一轮全部退出后释放n2；优先原full-h200x2恢复因资源不足取消，确认STOPPED后恢复n2到新node，原120分钟计时重启。仅此专用2GPU，无其他任务变更。完整owner库存5条及目标侧哈希复读无本host冲突，实际GPU空闲、各卡>120GiB余量。未经用户追加预算，不超过原16GPUh。

新效果尚未产生；完整读回须检验训练未见XML上的配对改善及特异性，同时报告canonical代价。不能用训练loss、所见markdown改善或部分行替换主线默认。
