# 布局增强同预算对照部署

2026-10-06 15:17:23 CST，controller55309启动，source_audit进行中。实例dl-context-dev-n2-1006，qb-prod-gpu800，官方ngc25.02/CUDA12.8，2H200/40CPU/400GiB，当前约17:05自动停止。

执行844dda1819168b551882577a2946548dc8602f2f，开发14c097248b7e416e7079b7b227f10f4a60a1cbd9，同treea8aba6633edbe3c99d133f262f839b6947c300c3。独立repos/context-layout-aug-20261006-v2，output context-coverage-20261006/context-layout-aug-v1。Windows/Linux38项检查通过。协议CONTEXT_LAYOUT_AUG_PLAN.md已在任何训练之前冻结，未因部署修复改变科学因素。

先真实native两步精确参数/Adam/RNG/梯度轨迹校准；两臂同256状态各新增128步到384，唯一因素canonical-only vs均衡canonical/markdown。原生校准单独计费，新两臂校准步复用于正式128步，不重跑。训练阶段0.75GPUh内，完成到ready_for_frozen_readout后退出。三格式消费者同冻结树已实现并通过测试，训练完整后才启动；读回2.25GPUh且整轮train+readout≤3，全部在原16GPUh池。启动前累计7.655850706829GPUh。

首次f0716ee controller2438仅CPU源审计因旧报告绝对路径身份失败，无GPU attempt/训练。确认PID/owner退出、attempts为空后归档失败output，保留旧代码；v2调用父报告原9cf6ea7执行目录复算，本轮身份使用固定输出plan，不改写旧报告。失败GPU成本0，平台CPU等待/空闲点券不在进程GPUh口径内。

上一轮全部退出后释放n2；优先原full-h200x2恢复因资源不足取消，确认STOPPED后恢复n2到新node，原120分钟计时重启。仅此专用2GPU，无其他任务变更。完整owner库存5条及目标侧哈希复读无本host冲突，实际GPU空闲、各卡>120GiB余量。未经用户追加预算，不超过原16GPUh。

新效果尚未产生；完整读回须检验训练未见XML上的配对改善及特异性，同时报告canonical代价。不能用训练loss、所见markdown改善或部分行替换主线默认。
