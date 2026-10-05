# 每小时执行入口

当前阶段（2026-10-06 06:51:54 CST）：PM→FM功能对照已在原H200x2实例运行，独立checkout context-route-functional-20261006，冻结c45ab75678b46812a5ad7a59bfcbd66eee97bcb9，output为原run根route-functional-v1。controller846942、worker849327/849328实际cmdline及两张GPU工作核验通过；首批625+615=1,240行绑定检查通过。源图片360/360、旧dev图片540、教师文件1,080均已SHA冻结，12教师张量抽查通过。结果pending，不缩分母、不重复启动。

本轮8,640新行、合并23,760行；同12问题、旧只读教师缓存、dev90、2训练seed各2噪声。源代码/资源/输入/测试与首批实测见ROUTE_DEPLOYMENT.md和heartbeat-2242-evidence。Windows/Linux各13相关检查通过。旧17,808行Writer结果不重跑，不改其output或执行checkout，不改ARIS主线。

下一次先读DEPLOYMENT.json、ARIS tracker和route-functional-v1/status.json/active-owner/attempts，再核对实际PID/cmdline/GPU。正在运行只核验推进；完成则按ROUTE_DEPLOYMENT.md全量复核分母、PNG、教师张量、共同前缀与偏好级报告。两finished收据完整前不报路线胜负。nvidia-smi host PID与容器PID不同，使用原始进程命令、CUDA绑定、启动前空闲现场和输出证据，不把host PID当本容器kill目标。

新增阶段1 GPUh上限计入原16池，attempts计费不得和receipts重复相加。已结算旧阶段3.424944881399 GPUh，新阶段截至此快照暂计0.047580385076 GPUh，全任务暂计3.472525266475 GPUh；终态前不当作最终费用。失败/中断先确认原worker停止并结算，再按冻结协议恢复；到上限不自动重置额度。

原smoke、16历史pilot、教师审计、Direct Writer诊断均完成。默认不晋级；训练loss/教师KL/partial比较均不能替代新历史内容特异性及完整MCQ/free/保持证据。无其他资源或任务变更。第五次实际触发2026-10-05T22:42:31.287Z已执行，后续每小时继续，只有实质进展/完成/故障时通知。

## 前次状态记录（以下仅供溯源，以当前状态为准）

当前阶段（2026-10-06 05:52 CST）：原16历史pilot、2,496行教师审计及17,808行共享Writer功能诊断均完成，不重跑。最新报告WRITER_RESULT.md；完整comparison及05:42现场/05:44全量审计见DEPLOYMENT.json。两训练seed在dev90恢复/开放应用均未显示收益，匹配图未优于同主题错配；默认不晋级。pilot16小幅恢复改善不当作新历史泛化。text=0是自一致性检查，不是准确率。

Writer两个rollout各212/212、两个readout各8,904行，全部exit0；636PNG、1,272教师张量、精确分母/共同前缀/donor全量核验，报告独立重算完全一致。controller4012735及四worker均已退出，active-owner消失，现场GPU无计算进程；执行checkout481335eb8df7efa05e41ff9cc337d2ef2670516e保持clean。不能恢复已完成控制器。Writer已结算2.075331035654 GPUh，全任务3.424944881399 GPUh（非点券账单），没有未结算活跃attempt。

下一步仅按ROUTE_FUNCTIONAL_PLAN.md：复用ARIS已训练PM→FM两seed完整dev90 V0图片，与相同问题/教师缓存下的Direct和B730比较。新增8,640行、合并23,760行、新增读取360PNG、1 GPUh上限计入原16池。暂无消费者代码和部署，先实现兼容只读缓存/资产验证并冻结执行SHA；不得把规划写成正在运行。

本次05:45:37检查上游seed20261005为126/180张且未验收，seed20261006尚无目录，见route-functional-dependency-2142.json。下次先读ARIS tracker并核对上游images-validated.json及全量资产；两seed就绪前不占GPU，不重复生成、不催改其owner/任务、不重做其MCQ/free/保持评测。任一源身份不符停止消费，不能修写源文件。保护official180。

本次未启动新GPU工作、未恢复实例、未修改其他任务或自动化。原H200x2为首选，未来启动仍须重新检查进程、资源、owner和预留；现在的空闲快照不是未来空闲承诺。用户已有资源恢复授权，不重复请求；不能复活取消的ARIS A64/B730。每次≤6小时、最多2GPU，失败时长计账、旧worker先核实停止后再恢复，禁止空转防回收。

实际第四次触发2026-10-05T21:42:00.463Z已完成审计。后续仍每小时，只在实质进展/完成/故障/需处理时通知。

## 先前阶段记录（仅供溯源，以上当前状态优先）

当前阶段（2026-10-06 04:42:23 CST）：原16历史pilot和2,496行教师只读审计均已完成，不重跑。共享Writer功能诊断仍在独立checkout `context-writer-readout-20261006-v2`、冻结481335eb8df7efa05e41ff9cc337d2ef2670516e运行；output为原run根`writer-readout-v1`。controller4012735保持存活，首seed worker4013220已exit0，第二seed worker64143实际cmdline/GPU均匹配。首seed212/212张PNG完成，第二seed57/212；尚无条件KL读回行，不能发布评分。现场同一hostname RUNNING，GPU0无计算进程，GPU1为本任务工作；不另启控制器占用GPU0，随后两分片读回仍需两卡。

两份上游warmup128均已完整完成。B730及Direct seed20261005共424张来源PNG的完整收据、原始历史/噪声、SHA与RGB尺寸再次核验通过。新增Direct seed20261006 checkpoint85bc7285…与绑定收据相等，全部128优化步骤、512训练PNG、14,336个非零有限反传梯度复核通过；两seed模型/数据/父权重/任务/优化器等冻结科学字段一致。未读取活动resume、未修改ARIS训练。

原控制器已自动接续第二seed，无需恢复或重新启动。全636来源PNG就绪后自动启动两分片17,808行读回。下次先检查实际status/active-owner/attempts/PID；正在运行则只核验推进，禁止重复启动。若受限退出或故障，先核对旧worker结束与attempt结算，在原4 GPUh阶段上限内按同一冻结命令恢复，成功收据直接复用，不重生成已验收PNG。

完整协议为3端点×106历史×2噪声共636来源PNG，12问题和四类控制共17,808行；新历史dev90与训练pilot16分开，两个训练seed各列，噪声先在偏好内平均。读回阶段在两seed图片齐全后自动启动，不以单seed缩分母。ARIS已登记的MCQ/V1/保持评测不由这里重复；只增加条件KL诊断，不能称为自由回答事实准确率或单因素因果比较。

新增阶段上限4 GPUh，包含失败重试，计入原16 GPUh。控制器attempts是本阶段唯一成本账本，receipts是其成功索引，不能把二者重复相加；没有finished的attempt先证明旧worker停止并按可证时长保守结算，禁止跳过未结算费用恢复。旧smoke+pilot+readout已结算1.349614 GPUh，本阶段运行时另外列暂计。单次最多6小时、最多2GPU。代码和计划均冻结，运行期间不得修改该远端checkout。

完成后核验两个finished收据、identity/readout文件哈希、精确17,808键分母、全部636来源PNG/complete、1,272 teacher张量/IDs/真实EOS/绑定及所有条件的共同前缀，重算comparison。灰图及同主题错配检查内容特异性；text是老师自身历史query的自一致性KL，不当成准确率。任何结论保留dev已有研究暴露及目标/路径混杂边界，默认不晋级。

本任务当前作用域：已完成上下文覆盖 × history_hard/prompt_matching 的16历史教师机制对照，并已扩展到复用已有Writer端点的只读功能诊断。科学协议见 ../../docs/CONTEXT_COVERAGE.md 和 WRITER_READOUT_PLAN.md。ARIS M1/M2、PM→FM训练及其MCQ/保持评测不由本任务修改或重复部署。

自动任务 `dreamlite-2`，名称“DreamLite 上下文鲁棒性实验跟进”，本地任务 `01a0fc2b-d3a4-75b0-a785-0e65fb49aa1d`，每1小时。已实际创建并读回 ACTIVE、1小时规则及目标任务一致；2026-10-05T18:39:27.617Z 已收到第一次实际触发，并完成本轮核查，端到端触发已验证。第二次实际触发为2026-10-05T19:40:58.514Z，已沿此协议部署Writer只读阶段。

每次读 DEPLOYMENT.json 的最新状态。GPU代码冻结后不得对该远端checkout执行pull/checkout。报告更新不改变执行SHA。不要读取过期ARIS自动消息来替换其MAINLINE_PLAN。

## 可复现启动

训练python：`/inspire/ssd/project/exploration-topic/czxs26210936/envs/vlm-r3-ngc2502/bin/python`。
checkout：`/inspire/ssd/project/exploration-topic/czxs26210936/repos/context-coverage-20261006`。
run根：`/inspire/ssd/project/exploration-topic/czxs26210936/runs/context-coverage-20261006`。

调用 `scripts/inspire/run_context_coverage.py`：

- `--phase smoke --output <run>/smoke --max-hours 0.5`，完成后核验两组24步日志、实际PNG、非零有限梯度及共享目标哈希。
- `--phase pilot --output <run>/pilot --max-hours 6`，仅在smoke通过后启动。
- 公共参数 `--base /inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory/DreamLite-base-a9a0f15-20260907`，`--reader` 同目录 `Qwen3-VL-4B-Instruct`，`--prefeval <checkout>/third_party/prefeval_reference`，`--ids-file <checkout>/configs/experiments/context_coverage_ids.json`，`--reference /inspire/ssd/project/exploration-topic/czxs26210936/runs/prompt-matching-20261005/pilot-B`。

用独立后台进程及日志启动；查看`active-owner/owner.json`、`status.json`、`receipts`、`attempts`及训练`resume.pt`。成功收据不重跑；失败收据先归档，累计GPU时间不能重置。launcher本身保存双GPU墙钟约束，结束后不停止平台实例。

GitHub出口曾在CPU准备机超时，改用已push的增量Git bundle，经CPU机器SCP传到共享盘，离线git fetch/checkout。源码仍以Git SHA验证。GPU无须联网安装，复用已经验证的环境。

## 资源恢复

用户明确授权恢复、新建或使用不干扰其他任务的空闲实例。当前首选 `vlm-dreamlite-full-h200x2-20260720`，2×H200，workspace分布式训练空间、project前沿课题探索，镜像ngc-pytorch:25.02-cuda12.8.0-py3。创建前实时查询完整group名与quota，不能照抄失效容量。

实例RUNNING且GPU无进程还不足以判为空闲，必须检查其他进程、已有计划预留、owner、训练/评测日志。`prefeval-k1-730-h2-20260924` 已由另一主线预留后续评测；默认不占。其他ARIS实例也不占。

原worker停止后才迁移，保留共享输出和冻结SHA。新实例核对模型/数据路径与哈希、CUDA真实计算、PNG读回，再从已保存状态继续。同一输出只能一个控制器；过期active-owner只在核实原主机/PID死亡后归档，不能根据心跳迟到抢占。若实例停止造成无终态收据，按最后可证起止时间保守计费并记录不确定范围。

## 如何推进

完成后核验comparison.json的所有320行端点与偏好级区间。原问题集旧教师全部复用，只重读同一donor规则的错配控制；不重训旧Writer。当前只检验教师机制，仍不能说明新历史Writer泛化。

机制证据已支持更丰富的问题覆盖；先检查另一主线的新近结果与完整父/子checkpoint，按上方当前阶段复用已有Writer作读回诊断，禁止重复部署同类训练或修改运行中的协议。每轮只改变一个主因素、先冻结完整评测和有限预算，报告token/时间代价与不确定性。原默认方案需要完整新历史、真实PNG、保持与自由回答证据才替换。若不支持，检查历史敏感度和各任务KL后决定单因素修正，不以反复挑dev成绩推进。

只在结果变化、阶段完成、故障或需用户处理时通知。没有变化保持安静。记录每次实际触发与动作，使定时配置和真正执行可以区分。

执行提交 `15ff2a93b5ec6222c37ec359b20cfadcabbba471` 与开发提交 `3995d2c9c111fdb7fac3002e658c4cc77e282e7a` 的完整 Git tree 相同；为离线部署把同一树锚定在已存在的a484执行基线。增量bundle关闭外部delta依赖后验证与fetch成功，远端clean，Linux100项检查通过。不得用开发分支后续报告提交覆盖执行checkout。

2026-10-06 03:59:44 CST更新：首seed已完成28/212张PNG，worker仍存活；最新4PNG和冻结checkout核验通过。本阶段暂计0.091366 GPUh，连同已结算旧阶段暂计1.440980 GPUh，尚非终态成本。Reader权重、配置、教师目标与损失代码SHA均与训练manifest相符。

第三次实际触发2026-10-05T20:41:29.455Z已执行；现场截至2026-10-06 04:42:23 CST，Writer阶段暂计0.801421 GPUh（其中已结算0.627029），全任务暂计2.151035 GPUh。原4/16 GPUh预算未变，没有新增训练、默认晋级或资源操作。证据为heartbeat-2041-writer-live.json与heartbeat-2041-integrity.json。
