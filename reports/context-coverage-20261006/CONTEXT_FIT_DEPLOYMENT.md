# 16历史上下文传递检查：真实部署

2026-10-06 07:59 CST追加核验：两组均完整128更新、各512draw；全256梯度更新有限非零，两组draw/pid/sigma配对一致。两最终checkpoint SHA重新核对通过，训练worker均exit0并退出，控制器已自动启动两个PNG生成worker。当前各8/32张真实PNG哈希通过，共16/64，尚无训练侧读回分数。见fit-progress-after-train.json。训练部分结算0.130255297157 GPUh，本阶段含活动生成暂计0.190926038557；全任务已结算3.787518723806、暂计3.848189465205 GPUh。1 GPUh阶段上限不变。

2026-10-06 07:53:23 CST在vlm-dreamlite-full-h200x2-20260720启动，controller1155936，训练worker1156921/1156922分别绑定GPU0/1。07:55现场两组38/128与40/128步；已完成日志连续，所有已读梯度有限非零，共同前38步的draw/pid/position/sigma完全相同。两组第4步检查点各1075组参数全部与B730父权重不同且有限，SHA与manifest已归档。科学结果仍pending，不能用MSE/梯度当作记忆收益。

原先PM→FM读回完整结束、旧3PID退出、owner关闭、GPU无计算进程后才启动；现场无其他训练进程或本任务其他owner。另一路ARIS当前free评测资源未变。只管理本任务独立输出context-fit-v1，不写旧bank或旧读回。

代码冻结9174325fe5c2e69acdd488150837fe977350a73e，tree fa3133f686ee830a30c81bfa0777373b6b208b41，与开发633ce47完全一致且clean；独立checkout为repos/context-fit-20261006。Windows/Linux各17项相关检查通过，覆盖继承的只读消费、严格训练分母/梯度、相同draw、偏好级统计、完整PNG与预算失败计数。未修改此前执行树。

输入检查通过：两组各16份288步最终soft目标，32latent/PNG哈希、身份和共同科学字段匹配；latent均float32 [1,4,128,128]。旧B730父checkpoint SHA独立重验。初始teacher latent为同灰图VAE编码，context suite是主要对照因素。完整固定方案见CONTEXT_FIT_PLAN.md。

训练复用原prefeval_k1_writer.py native FM：每组128更新、有效batch4、相同16历史/seed20260924/AdamW参数。第4步snapshot只作实际训练诊断，不用于选择。控制器自动依次训练、验收128连续步及512draw/组、生成64张真实PNG、复用192份pilot教师缓存读回1536行，再与旧2688行合并，完整4224行后生成comparison。没有新的dev/official评测。

新增阶段1 GPUh（失败计入原16总池）。旧阶段已结算3.657263426648 GPUh；07:55现场新增暂计0.052675468193 GPUh，全任务暂计3.709938894841，尚非最终费用或平台账单。stage执行器以单一总deadline限制全部训练/生成/读回，不按阶段重置预算；attempts为成本唯一账本，receipts不可重复计入。

下次检查status、日志、owner与实际进程后推进，不重复启动。若部分训练中断，当前native trainer可能尚无可恢复resume，监督器会明确拒绝有partial日志但无complete的盲目重启；需先审计checkpoint、恢复点和日志尾部，保持原随机状态及费用，不把从头再训称为续跑。预算不足保留partial，不自动扩大。

完成后全量检查两组256更新/1024draw、两最终权重、64新PNG/96旧PNG、192教师分布、1536新行/4224总行和偏好级报告。训练侧单seed可拟合性不能证明新历史泛化；默认不晋级。完整证据fit-2342-evidence/及同名tar.gz，归档SHA256 4f49c8e6c97659d4af95cd24b5f09da1a31c6fc9a41ca116b027079f8a16d4cc。
