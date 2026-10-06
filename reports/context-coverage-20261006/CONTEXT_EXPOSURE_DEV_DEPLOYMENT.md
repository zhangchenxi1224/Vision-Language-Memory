# 固定32/256 Writer 的dev90部署

执行4bd2a8331af5142247e6ebb9794c1a2cb3df2fe0；开发27261ca8d22ff37218b4ff1b5ba0876f82a8d7c6；同tree945aceb6065a4353e2c2c960df24cfa62e5b8540，源码clean。Windows/Linux各29检查通过，包括完整分母、共同prefix、独立偏好配对、失败attempt计费及dev/训练无交集。

2026-10-06 13:11:20 CST启动controller5841，实例dl-context-dev-n2-1006，host dl-context-dev-n2-1006--04a62f93b2ff-l335qh6pof，节点qb-prod-gpu2294。独立checkout context-exposure-dev-20261006/output context-exposure-dev-v1。原2H200模型、NGC25.02/CUDA12.8、Python3.12.3均保持一致。初始现场13:13:23仍在CPU来源全量复核，尚无新GPU attempt/PNG；后续状态以SUPERVISION和DEPLOYMENT为准，不能称dev科学评测完成。

平台恢复：原实例停止后，其原资源组无卡；取消原排队后新建低优先级H200x2，预检间被抢占。重调度节点的全盘元数据遍历阻塞，未启动任何训练/评分。确认无controller/owner/attempt后停止该实例，归档prelaunch目录，再新建上述normal实例。没有两个同时活动的本任务GPU实例，没有复活ARIS取消队列或占用其他任务预留。CPU完整控制目录扫描得到5条其他实例/测试owner，与启动前定点重读一致；新实例实际进程/GPU/内存通过。

按CONTEXT_EXPOSURE_DEV_PLAN.md只新增一个固定最终权重端点：180PNG/4320行，新旧共1080PNG/28080行。冻结12问题、1080只读teacher分布、两个noise、同主题donor、灰图与文本。n90偏好配对。预算上限2GPUh计入原16；已结算6.632105855743，失败也计入，最多2GPU、单次≤6h，平台120分钟停止计时。只读评测完成前无泛化结论，不改默认。

现场证据exposure-dev-0447-evidence/；归档SHA ed5c413b755cc52a34ee2c3c6f50038fa5b15cc38e72a323057d8f429d5a4357。旧完整曝光证据exposure-result-0447-evidence/。未动已完成旧源与其他自动化。

13:17:03 CST运行补充：来源重算结束，rollout29841真实CUDA0约14GiB/100%，首张1/180 PNG收据/格式/SHA验证通过；评分pending。新dev暂计0.012781345712GPUh，全任务暂计6.644887201455。详见exposure-dev-rollout-0447.json。
