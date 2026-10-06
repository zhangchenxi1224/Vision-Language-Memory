# 同主题历史覆盖扩展：目标和训练阶段部署

2026-10-06 09:56:20 CST启动，09:57:21确认真实双卡进程，09:58:59检查恢复checkpoint与快照。原实例vlm-dreamlite-full-h200x2-20260720，controller1901347，teacher0/1=1902132/1902133，CUDA0/1；每卡约17GiB。NVML host PID与容器PID不一致，使用真实命令、CUDA绑定、启动前空闲和产物联合核验。

执行fd3ac81dec0d94eaf92248709ab671b2be5f04af，开发20f9305100518b557eb89006a2595fe980b90e66，同treeea364409f90365255337ce592995e9d09abfd441。独立checkout /inspire/ssd/project/exploration-topic/czxs26210936/repos/context-population-20261006；output为本轮run/context-population32-v1。sparse checkout内从精确HEAD blob恢复协议，执行clean。Windows/Linux各24相关检查通过。

启动前旧dev controller及4 worker全部退出，owner不存在、双卡无计算，ARIS tracker与远端owner未对本host预留。父B730、既有Writer16、旧16目标/PNG/完整288梯度与每步问题schedule已复核。只补相同8主题的另16条，不重制旧目标；不新评dev/official，不改其他任务。

控制器第一阶段上限1.5GPUh：2worker制作各8条diverse-v1 soft目标288步，完整验证新增梯度、教师cache张量/EOS/哈希与旧科学设置一致，在本output建立只读用途bank链接，然后1GPU按同B730父/seed/128更新训练Writer32。最终检查512draw（每ID16次）、有限梯度/权重后退出，标记ready_for_frozen_readout，科研评分仍pending。

下一阶段评测范围、原/新增16分层、问题和分母已在CONTEXT_POPULATION_PLAN.md冻结，读回消费者尚待在训练源独立验收后实施。总3GPUh预算包含本阶段，剩余预算必须扣本轮全部attempt。不能把阶段结束当成整个实验完成，也不能以loss判断泛化或换main默认。

09:58:59实测两个教育历史各288步，两个游戏历史35/33步；4份resume latent有限、SHA记录完整，8组72/144/216/288快照PNG/latent通过。目标bank未全齐、Writer未启动，暂无新效果结论。

此前结算5.337061309947GPUh，新增暂计0.084781107969，总暂计5.421842417916/16；原16总池不增。完整现场population-0143-evidence/；归档SHAde83e7a8ecbb84b08a655365722ab7df12c524fd97d252caddcd8cc845fc8b41。增量checkpoint核验population-checkpoints-0143.json SHA90afaab90b7450a880c178e0be76bed80a379d615c126bfaf4f5067dc3d58c67。
