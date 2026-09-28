# 多目标主线部署记录

## 2026-09-28 14:40 CST：C8/B0全730直接评测已准备，尚未启动

按用户最新指定的C8/730优先级，冻结独立控制器checkout `repos/prefeval-c8-eval730-20260928`，提交 `efe0f7606ed7582a0d4171525b9bc285f1c1d140`，入口 `scripts/inspire/launch_prefeval_c8_eval730.sh`。其Writer/evaluator仍调用原224cc77；不重做教师、不训练权重、不更改运行中的round3。输出仅 `runs/prefeval-multitarget-20260927/eval730/fixed-c8-b0-20260928`。

CPU预检已实际通过：两份固定checkpoint哈希、730条唯一ID、全部T1/T2/T3、V0/V1原偏好保持、完整同主题donor映射、冻结工作代码及原第二轮完成回执。分层汇总用构造数据核验已知正确率/同图全对/偏好bootstrap，并验证重复行、缺行、错误donor均被拒绝；这些是验证数据，非730实验成绩。脚本SHA256 `ab4d702b41115bcfd888d5ef7cb561577eaf8af88b035830395ebbe45be2a8cf`，协议SHA256 `c670e3db2d7d54bf449999cf693d7544fb54bbc7040420b641c31484e6f641dc`。

证据见[C8_EVAL730_PREFLIGHT.json](C8_EVAL730_PREFLIGHT.json)，原始预检归档 `c8-eval730-preflight.tar.gz`（9244字节，SHA256 `be1e7dd015d55a41492b38a52aa0f0f7f06a54d35ee629ab5cf786723987e419`），本地已独立核验归档和协议。源码已推送同分支；公网clone曾超时，准备的Git bundle及备用ready checkout不是运行入口，后续只用上述正式checkout。

当前GPU仍在round3评测，两H200100%，controller及子进程、根与round3锁正常，未启动730评测。等round3完整结束、归档后，先核验实时GPU/全部进程/各锁/完成回执，再后台运行新入口，stdout/stderr写 `eval730/pipeline.log`，记录launch.json并确认实际生成增长。不得抢占当前实验，不能将prepared/preflight当成launched/complete。新入口同样非阻塞获取本任务根pipeline锁，子进程继承根和自己的controller锁。

最新核验（2026-09-28 14:05 CST）：第三轮C8R/U16各2048步训练完成，实际两份最终权重与complete的SHA256一致；13:42:19自动进入C8R/V0生成，14:04为246/512 PNG。14:05同一NORMAL两H200均100%计算，controller1734131与生成子进程2501979/2501980正常，完整命令与分片回执一致，根及pilot64锁HELD且传给子进程，smoke及旧锁FREE；无failure/最终results/complete。快照为round3/runtime-eval-20260928-1405.json，PID仅此次记录，不重启。

训练曝光及新旧目标几何已完整CPU审计，见[ROUND3_TRAINING_GEOMETRY_20260928.json](ROUND3_TRAINING_GEOMETRY_20260928.json)和[RESULTS.md](RESULTS.md)。C8R全部511目标被抽到；U16池1022目标中1021实际被抽到，旧/新呈现4095/4097，保留零曝光记录而不改变预算。证据归档round3-training-geometry-evidence.tar.gz SHA256 `525e6241df0638c3d8a79d4bd84b4f7faa68d9dd4796567e804aa534d60e2144`，本地已核验并重算曝光。继续45分钟跟进，完整配对Writer成绩后再决定730扩展方案，正常运行静默。

最新核验（2026-09-28 13:23 CST）：第三轮512个新教师全部完成，511合格，64条偏好全覆盖；12:47:42自动进入C8R/U16两组FM。13:23实际优化日志为1320/1330步（各2048预算），两名训练子进程2003468/2003469的完整命令与fm-shard回执逐项一致，均继承根及pilot64控制器锁；主机未变，两H200约38.9GB显存且96%/97%利用率，无failure，尚无最终评测结果。独立运行核验见远端round3/runtime-fm.json，当前无需重启或修改代码。

完整教师资格、目标池组成、父权重及训练manifest校验见[ROUND3_TEACHER_RESULTS.json](ROUND3_TEACHER_RESULTS.json)。教师原始回执、全部优化记录、冻结bank及manifest归档round3-teacher-evidence.tar.gz，共1042文件、SHA256 9c5cf8bfcb5ad59f6a3956d55b81f0eca4d303cd7b7e4a247ece71cce680caec，本地保留校验副本；PNG/latent继续保留于原共享输出，1024个实际文件哈希已核验。两组按原流程自动继续完整V0/V1四逻辑分片评测，45分钟跟进保持静默规则。

最新核验（2026-09-28 05:23 CST）：正式第三轮V0/V1各256份实际学生latent已齐备，05:06:22自动进入teachers-new。05:23已完成16/512个新候选且16个均合格（两偏好各8个）；这是按偏好顺序生成的中间子集，不据此推断最终覆盖率或Writer收益。两名教师进程2139284/2139285持续更新288步优化日志，实测调用冻结224cc77，参数为8候选、CE+.1偏移、RMS<=.1。两H200各约16GB显存，controller1734131正常，根及pilot64锁HELD，smoke/旧阶段锁FREE，无failure或最终results/complete。不启动副本、不更改冻结预算，保持45分钟静默跟进。

最新核验（2026-09-28 04:30:57 CST）：第三轮烟测于04:21:02完成，四个新目标均完成288步并通过48/48项保存重载PNG检查；04:21:03自动进入正式pilot64。正式controller PID1734131、两名Writer子进程1734141/1734142为此次快照，实际完整命令与回执一致，根及pilot64锁HELD，smoke和旧阶段锁FREE，无failure。两张H200均100%利用率，V0实际学生latent已105/256，V1尚未开始；没有重复启动或修改运行代码。

烟测原始输出（含PNG、latent、优化日志及资格回执）归档round3-smoke-evidence.tar.gz，共63个文件，SHA256 bfc88b8567cd0eed994fb5a0d061988f2e129143d0d3afdcc54870fda1cdbf35，本地保留副本。逐项目标哈希及检查见[ROUND3_SMOKE_RESULTS.json](ROUND3_SMOKE_RESULTS.json)。自动跟进恢复45分钟；正式目标覆盖、C8R/U16训练与完整评测继续按冻结协议推进。以下04:15—04:18记录为启动历史。

最新状态（2026-09-28 04:15 CST）：第二轮于03:59:20完整完成，结果见[ROUND2_RESULTS.json](ROUND2_RESULTS.json)。确认旧controller与子进程均退出、两卡0MiB且根/旧/恢复控制器三层锁FREE后，04:15:53在同一NORMAL两H200资源启动第三轮，先smoke再pilot64，未重新申请算力。第三轮入口为scripts/inspire/launch_prefeval_multitarget_round3.sh，启动器PID1689886（仅启动快照），日志为任务根round3-pipeline.log，输出只在round3/smoke和round3/pilot64。

第三轮控制器独立checkout repos/prefeval-multitarget-round3，冻结5134d407a2710cef025654b8ab4207a5c269e6a0；控制器文件SHA256 e8572bb7d7d3fb36145252b3e9f8c0f3ef03eb98b9a3a3a7592052ace6ed50e7。真实Writer/教师/evaluator继续调用repos/prefeval-multitarget-round2的224cc77冻结代码。新学生快照为C8最终权重；C8R对照完整旧目标回放，U16合并完整旧池与八个新学生起点的局部修正候选。两组均额外2048步FM。方法、门槛与成本见[PLAN.md](PLAN.md)，根pipeline.lock及本阶段controller.lock均传给子进程。

04:16:45核验smoke controller1689907的两名实际子进程完整命令与回执一致，根和smoke控制器锁HELD。04:18:44核验V0/V1各两份实际学生latent均齐备，已进入teachers-new；两GPU各约13.9GB显存，首对教师优化到196/199步，有真实梯度更新记录，尚无资格完成回执。此为执行核验，不是烟测通过或方法改善结论。运行快照保留于round3/runtime-initial.json与runtime-teacher.json。自动跟进暂保留10分钟以确认烟测，正式pilot64健康推进后恢复45分钟。

第二轮原始输出及身份归档round2-final-evidence.tar.gz已下载并校验；原用户四卡实例继续停止并保留，B730/C/R任务未干预。下方第二轮正在运行或排队描述均为历史状态。

最新状态（2026-09-28 00:45 CST）：利用刚释放的2卡配额，评测已在新NORMAL资源prefeval-mt-eval-h200x2-20260928的qb-prod-gpu2226启动。原4卡实例停止排队但保留。当前唯一入口为round2/recovery-20260928-0045/resume_eval.py（ca908b6），控制器PID8515；旧四卡恢复目录从未启动。四个逻辑分片分两批在两卡执行，模型/参数/种子/评分不变。详情见[ROUND2_RECOVERY_20260928.md](ROUND2_RECOVERY_20260928.md)，下方旧PENDING说明仅为历史记录。

当前状态更新（2026-09-27 21:39 CST）：第二轮两组训练完成，但21:22:09实例被平台利用率规则自动停止；原实例重新start后PENDING。后续评测将只在round2/recovery-20260927-2135独立目录恢复，已完成权重及S1/V0图像只读复用。恢复入口resume_eval.py（本分支bb2d298），实际模型/采样/评测代码仍固定224cc77，不再执行旧全流程启动器。详见[ROUND2_RECOVERY_20260927.md](ROUND2_RECOVERY_20260927.md)。在新主机/进程/锁核验并真正启动前，不将此状态报告为恢复成功。

首轮已于2026-09-27北京时间15:51:31完整结束，下一轮采用独立代码及输出目录。首轮结果见 [PILOT64_RESULTS.json](PILOT64_RESULTS.json) 和 [RESULTS.md](RESULTS.md)；第二轮S1/C8冻结设计见 [PLAN.md](PLAN.md)。以下首轮时间线保留作为审计记录，不能将其“正在运行”描述作为当前状态。

## 当前第二轮：S1/C8

- 运行代码冻结在 `224cc77d790cf3967b5a56ce2e77c364959435a2`，独立checkout `/inspire/ssd/project/exploration-topic/czxs26210936/repos/prefeval-multitarget-round2`。本地后续文档提交不改变此冻结代码，也不修改首轮runtime。
- 仍使用本任务原四H200实例、相同主机及环境。部署前确认首轮全部退出、四卡0 MiB，无持锁子进程。没有新申请、停止或删除任何用户实例。
- 本地与实际H200环境均通过14项相关测试，Python编译、shell语法和控制器导入检查通过；没有安装新依赖。
- 启动入口 `scripts/inspire/launch_prefeval_multitarget_round2.sh`，由nohup启动，日志为任务根的 `round2-pipeline.log`。复用根pipeline.lock，阶段控制器锁传给子进程。
- 16:11:32开始 `round2/smoke`；16:13:48结束，两偏好各回放锚点与一个新修正目标，共四图全部通过48项PNG读回。两锚点PNG/latent哈希与原F1逐一一致；新目标各288步，相对锚点RMS分别.037756/.097871，均≤.1。摘要及原始逐项输出见 [ROUND2_SMOKE_RESULTS.json](ROUND2_SMOKE_RESULTS.json)；原始回执/日志归档为任务根 `round2-smoke-evidence.tar.gz`，本地有副本。烟测不代表Writer性能。
- 16:13:50自动进入 `round2/pilot64` 的C8正式教师阶段；初始控制器PID1867019，启动器1843279，四个教师PID1867216/1867219/1867222/1867225。16:14四张GPU均有约13.9GB显存及计算活动；逐一将实际/proc命令与进程回执的完整命令匹配。两层锁实测均被持有，不能重复启动。PID仅为此次快照。
- 后续自动执行C8资格冻结→S1/C8各2048步FM→V0/V1八种子四分片生成及MCQ读回。当前只有教师阶段进展，第二轮Writer结果尚未产生。评测时必须读取四个readback分片。
- `prefeval-writer`已更新为当前冻结代码、目录及决策约束，保留45分钟间隔与无变化静默设置。下一次从最新报告和实时进程判定阶段；首轮完整结果不得被初步V0记录覆盖。
- 19:31实时核验：C8完成467/512、合格466，GPU0分片正常结束、其余三分片继续。唯一失败偏好仍有7个合格目标；按冻结规则继续，不需恢复或改参数。两层锁正常，代码仍为224cc77。详细数据见RESULTS.md最新条目。

## 首轮部署历史

2026-09-27 已部署至用户原排队资源。首次控制器进入执行时间为北京时间03:13:18。

- 实例：`prefeval-k1-h200x4-high-20260924`；分布式训练空间；开发区-H200-3号机房-2-cuda13.2版本；正常优先级4×H200。
- 主机：`prefeval-k1-h200x4-high-20260924--506d5fcf84c1-heqpkyzp2a`，节点 `qb-prod-gpu963`。接手前实测四卡0 MiB/0%，无实验进程。PID只作为首次部署快照，恢复前必须重新核对主机与完整命令。
- 新建的重复低优先级实例 `prefeval-mt8-h200x4-20260927` 已停止并删除。
- 运行代码冻结于 `13d286d464a613164840850f5d364ff6b41ad7eb`，远端 `/inspire/ssd/project/exploration-topic/czxs26210936/repos/prefeval-multitarget-runtime`。后续本分支的文档提交不改变运行代码，不能直接pull正在执行的checkout。
- 输出 `/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-multitarget-20260927`。
- 共用只读环境 `/inspire/ssd/project/exploration-topic/czxs26210936/envs/vlm-r3-ngc2502`，沿用既有NGC镜像及模型，没有安装或更新共享环境依赖。
- 启动入口：`bash scripts/inspire/launch_prefeval_multitarget.sh`，由nohup脱离交互终端；`pipeline.log`记录阶段，`pipeline.lock`防重启。首个烟测控制器PID43953，子进程生成实际PNG后自动进入教师修正。
- 本地及实际H200环境的12项目标筛选、数据边界、抽样与既有变体/源银行检查通过。
- 烟测于北京时间03:18:59完成：两条训练偏好，F8/S8各两个候选，共8张教师PNG，全部通过T1/T2/T3乘四选项位置的96项读取检查。F1实际每偏好固定选1个；controller的qualified_counts记录的是筛选前共享候选池数量。原始回执摘要、哈希与逐项结果见 [SMOKE_RESULTS.json](SMOKE_RESULTS.json)。这仅验证教师构造链路，不是Writer准确率或泛化结论。
- pilot64于03:19:02自动启动学生起点生成，控制器首次PID113976；正式每类8目标对照继续执行。尚无正式Writer准确率提升结论。
- 04:06跟进确认学生起点512个全部完成，03:42:11已进入正式教师构造，四个工作进程和锁正常。阶段证据及后续判断见 [RESULTS.md](RESULTS.md)。
- 11:04:33教师阶段完成并自动进入F1/F8/S8三组FM；11:35实测各约1160/2048步。完整教师合格1018/1024，64条偏好全部保留，原预算与冻结代码未变。完整教师汇总见 [TEACHER_RESULTS.json](TEACHER_RESULTS.json)。
- 三组2048步训练已完成，11:59:09进入评测；15:11已完成V0评测，正在四卡生成V1。V0初步显示F1>S8>B0>F8，须等完整V1后再冻结下一轮。checkpoint身份、逐偏好分析及匹配对照见 [RESULTS.md](RESULTS.md)。

## 固定首轮与自动推进

启动器依次执行两条偏好、每类两个目标、288次教师更新的烟测，以及固定pilot64的F1/F8/S8实验。正式三组从同一已完成B73023360断点启动，各追加2048次官方FM；父模型B0加入同一新种子对照。pilot64覆盖16个主题，每主题4条。实际K_i和不合格目标全部保留，任何偏好零合格目标就停在教师修复阶段。

详细方法与冻结参数见 [PLAN.md](PLAN.md)。本轮保留之前直接优化model latent作为FM端点的入口，明确不冒充目标PNG再编码实验。教师正确性必须通过实际PNG的12项训练侧检查。

本任务heartbeat `prefeval-writer` 已启用；部署后再次核验当前调度为每45分钟（以应用中的最新设置为准）。它根据实际阶段结果处理技术故障、教师修复、学生快照刷新、主题对照、730扩展和K1保持，不依据dev/O1/O2选模型。无可行动变化时静默。

## 不干预的既有实验

- `prefeval-k1-l012-h200x4-20260924`：B730续训至93440步及固定端点评测。
- `prefeval-b-cr-h200x4-20260926`：C/R730保持实验。
- 它们的代码、配置、输出、进程和自动跟进均不由本任务改变。本轮只读原已完成B730权重。

## 检查入口

本地CLI：`wsl -- bash -lc 'inspire ...'`。GPU `notebook exec` 需要本地TTY；共享文件经CPU入口 `dl-align-cpu-20260914-r3` 读取。重点检查 `pipeline.log`、`smoke/controller.json`、`pilot64/controller.json`、`teachers/*/*/target-*/complete.json`、各 `*-bank.json`、各训练 `optimization.jsonl` 与最终 `results.json`。

恢复前核验controller与GPU子进程、主机、完整命令以及两层锁；子进程活着时不启动副本。方法修正另开版本与输出，不能修改已执行分母或冻结参数。正式结果未完成前不得把损失下降或程序启动报告为视觉记忆能力提升。
