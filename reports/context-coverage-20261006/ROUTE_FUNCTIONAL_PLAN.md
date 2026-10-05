# PM→FM 路线功能诊断：冻结科学协议

2026-10-06 05:52 CST 登记于PM→FM条件KL读回之前。科学协议保持原登记内容。2026-10-06 06:50 CST执行补充：只读消费者与有界监督器已实现，本地13项相关检查通过；须完成Linux检查、两seed全量资产验收、输入SHA清单及资源复核，再按独立冻结checkout部署。

假设：在相同父权重、训练draw和128次Writer更新下，先用概率匹配构建目标再做Flow Matching的路线，可能比Direct路线更好地保留历史条件行为。主要比较因素是既有训练路线配方；目标latent优化等计算属于该配方，不能声称两者GPU时长相同，也不能把差别单独归因于软监督。

## 冻结评测

- 数据只有完整internal dev90，使用相同原始V0历史、每条2个既有eval噪声。Writer不见评测问题。保护official180；不新增问题、不做dev检查点选择。
- 两seed均为20261005/20261006，最终128步端点。PM→FM对同seed Direct及共同B730逐项比较；两个seed分别报告。
- PM→FM权重SHA分别为d1956bd2009ebe9e39601e992a5ee53765a194d1fb20a0d8e757749a1efd41e8、647243c2865b440fabdc8c9e69fb4b01fe89a05cb061ff33682c9c078eda2091，来自ARIS route-comparison-v1。读取完成收据和最终权重，不读取活动resume。
- 复用context_readout_audit.json的12问题（恢复/开放应用/无关各4）、writer-readout-v1内1,080份dev教师缓存以及既有B730/Direct/灰图/文本读回。不得重新生成教师答案；缺失或不匹配时停止并报告。
- 新增90 × 12 × 2训练seed × 2噪声 × 2图片控制 = 8,640行；合并旧dev的15,120行后为23,760行。既有pilot16不混入分母。新增消费360张PM→FM PNG，合并全部5端点dev共900张来源PNG。
- 完全相同的教师continuation/logits/真实EOS；逐token KL，再同类别问题与噪声在偏好内平均，最后90偏好平均。按偏好配对bootstrap，沿用现有重采样与随机种子，不以图片/问题扩充独立n。
- 报告每类匹配、同主题错配、灰图、文本；主差值为同seed Direct KL减PM→FM KL，辅助B730 KL减PM→FM KL与错配减匹配。错配沿用相同主题、相同dev顺序的循环donor。
- KL仅表示教师前缀下的行为对齐；text是自一致性控制。dev已有研究暴露，结果探索性，无MCQ/自由生成/保持/黑盒泛化替代含义。默认不晋级。

## 只读输入与启动门槛

上游根：/inspire/qb-ilm/project/exploration-topic/czxs26210936/runs/dreamlite-aris-20261005/route-readback-v1/pm-fm/seed-{seed}/V0。

必须等两个seed各180张PNG及images-validated.json均完成，并独立核验完整manifest、checkpoint/PNG SHA、RGB1024、V0原始历史、28步/CFG1、2噪声、initial_variant=0、variants SHA 12ac93a5db22d528b10f2165d701e9c7fa2aec33920699cc2f64218d0a300dd7。噪声为stable_seed(20260924, f'rollout:{pair_id}:{chain}', 0)，历史为最初两条exchange。逐张PNG的complete/writes必须一致；不要求其MCQ任务结束，但不得消费未完成或仍会改写的图片。输入清单/代码/环境/模型/协议SHA在本任务独立输出冻结。

截至05:45:37 CST：seed20261005有126/180张、未验收；seed20261006目录尚不存在。见route-functional-dependency-2142.json。由另一主线按其既定预算生成，本任务不改其owner/协议、不接管或重复生成。

新增执行仅可位于本任务独立checkout及runs/context-coverage-20261006/route-functional-v1；不得修改已经完成的teacher/Writer输出和冻结执行树。旧readout/cache只读复用。

## 预算与恢复

本阶段新增累计上限1 GPUh，失败也计入，属于原16 GPUh总池；此前已结算3.424944881399 GPUh。最多同时2 GPU，单次不超过6小时。到上限保留partial，不减分母、不自动扩额度。

优先本任务H200x2实例；启动前重新核验实际PID/cmdline、owner/任务预留、GPU与内存。不以低利用率判空闲，不占另一主线已保留的卡。实例停机可按既有授权恢复或另找无冲突资源，但等待输入期间不空转、不为避免回收运行进程。恢复前先证明旧worker停止并结算失败时长，排他claim唯一，冻结随机状态和输入身份。

只有两seed全量8,640行、缓存/PNG全量复核、旧结果引用SHA无变化及偏好级报告重算通过，才能发布路线比较结论；单seed/partial只报进度。
