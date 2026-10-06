# DreamLite 上下文覆盖与验证选择：2026-10-06

目标是让图片保留历史对不同未来问题的影响。全词表概率监督约束“如何比较回答”，问题分布约束“必须保留哪些信息”。本轮只验证第二项及检查点选择，不预设泛化改善。

本轮为当前已完成的 prompt-matching pilot 的补充机制实验，不接管 ARIS 的共享 Writer M1/M2 和 PM→FM 对照，不重复它们的训练。原问题集的 history_hard、prompt_matching 两组 288 步教师与读回直接复用。新问题集两组仍匹配相同教师 continuation / logits、288 更新、Adam 0.05、VAE latent、冻结 Reader 与 VAE、1024 RGB、Q8 STE。原 hard_ce 是额外历史参照，不把它混作 history_hard。

## 固定协议

- 冻结 16 条 pilot 历史：8 个按名字排序的主题各 2 条，详见 `configs/experiments/context_coverage_ids.json`。选择发生在新结果产生之前，但旧 pilot 已有研究暴露。
- 每 12 更新：4 次事实恢复、4 次 MCQ 应用、2 次开放应用、2 次无关任务。MCQ 连续索引保留 T1/T2/T3 × 四个正确位置的平衡。
- 这些权重是本项目待验证假设，不是论文原始配方；论文的通用问题与重复提示机制在此适配为良性用户记忆任务。
- 老师只看最初已发生 exchange；Writer 未来若接入仍不能看到训练问题。teacher-generated 答案不当评测真值。
- 保存 72/144/216/288 步真实 PNG、latent 与哈希。继续完成 288 步，原最终端点始终保留。
- 用 6 个独立模板（恢复/应用/无关各 2）计算真实 PNG 的逐 token KL，先在任务内均值，再在三类间均值。取最小值，同分取更早端点。最终与所选端点分别评测。
- 选择只在训练历史上进行；这是未见问法验证，不是未见历史验证。没有以 dev90、official180 或 O1/O2 选择端点。
- 正式读回按 16 偏好 × 5 问法 × 4 控制 = 320 行/端点。MCQ 主诊断为同图 O1 AND O2；匹配、灰图、同主题错配、文本都报告。独立 n=16。
- 复用旧基线的 memory/blank/text；另用冻结16条的相同 donor 规则补读旧教师的 mismatch，使四格比较使用同一错配来源。
- 记录实际 token 曝光和耗时；同更新数不等于同 token/计算量。自由回答、保持深度和共享 Writer 新历史泛化仍须另外完成，不能凭此教师实验替换主线。

## 决策

第一阶段完成后报告原/新上下文 × 硬/软监督四格；先比较最终端点，另列选择收益。出现正向信号后，先把同一问题采样接口接入一个新的、冻结预算的共享 Writer 对照，复用已完成的共同父权重；不得修改正在运行的 M1/M2 checkout 或插入新目标改变其中途协议。

小样本未显著时报告不确定性，不把“没有显著收益”视为等价。新上下文造成明显退化时检查教师历史敏感度、训练/验证 KL、问法与 token 分布，每轮只更改一个因素。三条必要证据是：新历史上优于旧方案，匹配图优于错配/灰图，真实 PNG 及跨轮保持通过。替换默认方案需要完整评测，不能自动用最低训练 loss 晋级。

## 执行与恢复

入口 `scripts/inspire/run_context_coverage.py`。先双卡 24 步 smoke，再 16 条/组的正式 pilot。最大同时2 GPU；smoke累计1 GPUh，首个pilot累计16 GPUh；每次执行最多6小时。累计成本读取成功/失败 receipts 和 attempts，不按恢复次数重置。

优先 `vlm-dreamlite-full-h200x2-20260720`；其他实例仅在同时核实平台状态、实际计算进程、GPU、已有 owner/任务预留后使用。低利用率不等于空闲。实例停止可按用户授权恢复、使用无冲突空闲实例或新建同规格实例；不得停止其他任务。保持相同冻结 commit、模型哈希、输出身份和随机状态；先证明旧 worker 不再运行，再迁移唯一 owner，归档旧 claim。跨主机不依赖 flock，共享盘目录的排他创建与静态 owner 双重记录。

不要重新启用已取消的 ARIS A64/B730 队列，不改动其他自动任务。新的每小时跟进只管理本轮作用域及已完成旧 pilot 的归档，并在启动任何后续 Writer 工作前读取 `D:/2026WorkExperience/dreamlite-aris-20261005/refine-logs/EXPERIMENT_TRACKER.md` 防止重复。

文献依据：[Image Hijacks §4.3 及附录](https://arxiv.org/abs/2309.00236)。JPEG、噪声、ensemble 不在本轮训练新增项中；论文的鲁棒性测试不冒充训练增强。

## 2026-10-06 首次实际跟进后的决定

16历史pilot与12个新问题的只读审计已完成，详见 `../reports/context-coverage-20261006/READOUT_RESULT.md`。更丰富的问题覆盖改善教师图片的条件行为KL，硬/软监督方向一致，MCQ未改善。暂不启动另一个重复的Writer训练：先复用正在运行的ARIS共同warmup128完成端点及B730父权重，冻结新的只读评测协议，检查共享Writer上的收益传递。此比较有目标及优化路径等混杂，只能作为功能诊断，不能替代上下文覆盖的单因素因果对照。完成收据出现前保持等待，不读取活动resume，不占其他主线资源。

## 2026-10-06 共享Writer诊断完成后的决定

既有Direct128两个seed的17,808行功能读回完成，全636PNG与1,272教师分布独立核验，见 `../reports/context-coverage-20261006/WRITER_RESULT.md`。dev90恢复/开放应用未优于B730，匹配未优于同主题错配，训练侧小幅恢复变化不足以证明内容记忆或新历史泛化。保留默认。下一项有界诊断为复用已训练PM→FM的完整dev90图片，在相同12问题和教师缓存下比较路线；协议 `../reports/context-coverage-20261006/ROUTE_FUNCTIONAL_PLAN.md` 已冻结，等待另一主线完整图片，不重复其生成或评测。此时新增执行器尚未部署。

2026-10-06 06:50 CST执行进展：两seed上游图片完整后，只读PM→FM功能对照已部署，详见 `../reports/context-coverage-20261006/ROUTE_DEPLOYMENT.md`。新增评测运行中，尚无完整路线评分；科学问题、缓存与预算按已登记协议不变。

2026-10-06 07:55 CST：PM→FM读回全量完成，见 `../reports/context-coverage-20261006/ROUTE_RESULT.md`，微小KL变化未建立内容特异性。已按 `../reports/context-coverage-20261006/CONTEXT_FIT_PLAN.md` 部署仅训练侧16历史的窄/宽教师传递对照；同父权重、128更新，单seed，不新增dev评分。当前梯度/检查点验证只证明实际训练进行，完整读回结果pending。


2026-10-06 08:52 CST：同父/同128步的窄宽上下文共享Writer训练侧gate已完成，恢复与应用存在正配对改进和匹配特异性，见CONTEXT_FIT_RESULT.md。冻结下一步context-dev-v1评估已完成最终权重；保护official180，单seed内部dev仅探索性，不晋级默认。全任务结算4.041693349613 GPUh，新阶段上限2。

2026-10-06 08:55 CST：context-dev-v1固定权重诊断已部署原双H200，执行06e05bb；首批12/360PNG完整性通过，dev结果pending。见CONTEXT_DEV_DEPLOYMENT.md。

2026-10-06 09:52 CST：固定权重dev90完整阴性，训练侧收益未延伸，详见CONTEXT_DEV_RESULT.md；不晋级、不继续dev搜索。新冻结同8主题16→32历史覆盖训练侧对照CONTEXT_POPULATION_PLAN.md，先复用旧16并补16教师，再同父FM128。此时未部署，原阶段全部退出，累计5.337061309947GPUh。

2026-10-06 09:59 CST：同主题覆盖扩展已部署独立fd3ac81，先新增16教师目标和Writer32训练，至ready_for_frozen_readout退出；原/新增16分层评测随后按冻结协议实施。首批梯度/恢复checkpoint/8对快照通过，无新科学评分。详见CONTEXT_POPULATION_DEPLOYMENT.md。

2026-10-06 第九次跟进：新增16教师及Writer32完整结束，来源/梯度/最终权重全量复核；结算全任务5.874805550178GPUh。按原CONTEXT_POPULATION_PLAN.md实现分层3840行消费者，等待Linux预检及独立部署，结果pending。见CONTEXT_POPULATION_READOUT_DEPLOYMENT.md。

2026-10-06 11:04 CST：原/新增16分层读回已部署独立cb04567，双GPU实际worker及首批8/96新PNG核验通过，完整3840行结果pending。目标/训练结算0.537744240231，总暂计5.916105394827/16GPUh。见CONTEXT_POPULATION_READOUT_DEPLOYMENT.md；不新评dev/official、不改默认。

2026-10-06 第十次跟进：population完整3840行阴性，原16收益退化、新增16无匹配特异性；报告CONTEXT_POPULATION_RESULT.md。全任务结算6.315022416578GPUh。冻结CONTEXT_EXPOSURE_PLAN.md，仅32历史更新128→256以检验曝光不足；完整Adam/RNG续接原128，目标/数据/评测固定，不触碰dev/official或默认。当前准备待部署。

2026-10-06 12:00 CST：固定32历史更新128→256对照已部署独立cc3dffc，实际续训171/256，43新增梯度有效，Adam/RNG/权重和原128日志完整复用。完整5376行读回pending；新1.5GPUh上限，全任务暂计6.340538/16。见CONTEXT_EXPOSURE_DEPLOYMENT.md，训练侧结果不晋级默认。

2026-10-06 12:03 CST：固定32历史256更新全部完成且验收，resume/final与1024draw连续性通过；开始64PNG生成及后续固定读回，结果pending。见exposure-post-training-0345.json，任务暂计6.396397GPUh。

2026-10-06 第十一次跟进：固定32历史256步的5376行全量核验完成，两分层训练侧恢复/应用均恢复匹配特异性；报告CONTEXT_EXPOSURE_RESULT.md。累计6.632105855743GPUh。冻结CONTEXT_EXPOSURE_DEV_PLAN.md，仅评估final256的内部dev90，复用旧权重/对照，不重新训练或选择，official180和默认不变。新同镜像H200x2资源已申请；实际部署状态以DEPLOYMENT与SUPERVISION最新条目为准。

2026-10-06 13:17 CST：冻结final32/256的内部dev90已在新dl-context-dev-n2-1006真实生成，执行4bd2a83，控制器5841/worker29841，来源全量重算及首张PNG通过；完整读回pending。原实例及低优先级失败预检实例均停止，其他任务未改。见CONTEXT_EXPOSURE_DEV_DEPLOYMENT.md；全任务暂计6.644887/16GPUh。

2026-10-06 第十二次跟进：固定final32/256内部dev90完成并全量复核，28,080行显示恢复/应用没有新历史迁移或匹配特异性，见CONTEXT_EXPOSURE_DEV_RESULT.md。累计7.373056459228GPUh。下一轮冻结CONTEXT_LAYOUT_PLAN.md，只在训练32上改变历史角色排版，固定权重、内容、噪声/读者/目标，先标准PNG严格校准后64新图与1536行；已部署独立9cf6ea7到现有专用双H200，源审计运行中。无新dev/official评分或默认晋级。
