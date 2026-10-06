# 角色排版诊断部署

2026-10-06 14:12:38 CST 已真实启动controller378758，14:14状态source_audit；新PNG/评分仍pending。实例dl-context-dev-n2-1006，host dl-context-dev-n2-1006--04a62f93b2ff-l335qh6pof，2H200/官方ngc25.02-CUDA12.8。沿用120分钟平台停止计时（约15:10 CST），没有空转延寿或修改其他实例。

执行9cf6ea72da0e4b17b3e0bf27232d55ae74db6c17，开发87d65b0，同tree5a0f5331c0da3959a7a174609687fbcd94f9f898；独立repos/context-layout-20261006和runs/context-coverage-20261006/context-layout-v1。Windows/Linux各26测试通过。原native Writer及所有旧消费者未改。

只改变训练32历史角色排版，固定5aaffb99 final256权重；完整科学协议CONTEXT_LAYOUT_PLAN.md。1张标准校准+64新布局PNG，1536新/6912合并行，384教师只读；原/新增各n16分层、精确prefix及同topic donor。阶段1GPUh含全部attempt，全任务16GPUh，启动前已结算7.373056459228。canonical PNG校准失败即停止。

原dev控制器、三个worker与独立审计全部退出；新source审计前fresh CPU全量owner清查与目标侧哈希复读均为5条其他实例/fixture，没有当前host预留。nvidia现场无计算进程，两GPU各>120GiB可用，系统内存足够；新controller唯一owner。其他ARIS实例/预算/源码未动。

本轮只证明一个输入布局扰动下的训练侧稳健性，不能直接证明新历史泛化，不能替换默认；完整读回结果pending。后续状态与费用见DEPLOYMENT.json和SUPERVISION.md。

2026-10-06 14:19:53 CST：源审计已完成，rollout398060在CUDA0实际约14GiB。原格式校准PNG严格哈希一致，新14/64PNG逐份核验通过，科学读回pending。全任务暂计7.426958981487GPUh（已结算7.373056459228+新活跃0.053902522259）；详见layout-0548-evidence/，归档SHAa57281deed02d5f06d74aaf2e55ae5e7e75f9b1ae24b970263a321b18a1288b7。
