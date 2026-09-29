# 多目标主线部署记录

## 2026-09-29 17:28：C8单写完整阶段归档，B0自动接续

17:22实际核验两实例status/events均无新增回收、auto_stop为0。单写仍在prefeval-mt-eval-h200x2-20260928、qb-prod-gpu2187、g2xx4gn63g主机，当前输出eval730/recovery-20260929-1325。C8全部V0/V1各5840PNG及各四片39420读回已完成，16:27:34自动进入B0/V0生成，17:22有615张完整PNG；controller21763与Writer1642034/1642035完整argv匹配、两锁继承正确，两H200均100%/14129MiB。无failure或全量最终results/complete。

长程仍在用户指定prefeval-b-read-h200x2-20260925、qb-prod-gpu2294、iwj3wpwcbs主机，当前输出long10/recovery-20260929-1425-2gpu。C8/V1从16:22的1508PNG/136完整链增至17:22的2185PNG/198完整链；controller20310和Writer69871/69872完整argv及三层锁正常，两H200均100%/14128MiB，无failure/最终results/complete。现场分别保存eval730/runtime-monitor-20260929-1722.json和long10/runtime-monitor-20260929-1722.json。PID均仅本次快照，后续继续查实际主机与增长。

CPU独立脚本scripts/inspire/audit_prefeval_c8_full_stage_20260929.py（源0663f8a）只读核验C8完成阶段；未改运行checkout、输出或协议。独立输出eval730/analysis-c8-full-20260929-1725，11680实际PNG、78840唯一读回完整，官方答案解析与donor校验通过，恢复的V0旧路径来源可追溯。原始归档c8-full-stage-evidence-20260929.tar.gz共23394文件、SHA256 8bc6ab37e195f49c99f18b2e3c6bfb1dc9ed646f3fde8edd3e6e8685cc4d7f81，已下载并逐项核验，独立重算全部原始读回一致。完整阶段数值及边界见RESULTS最新段。B0尚未完成，不能当作最终比较；按原预算继续，保持60分钟跟进。

## 2026-09-29 14:19：长程实例遭平台回收，保留V0完整读回恢复

14:34:34已确认真实生成恢复：14:32:34进入C8/V1，PNG从复制时286张增至290张、再至302张；26条完整轨迹仍保留，正在重建两条部分链。两H200均100%、各14128MiB；controller20310与Writer69871/69872完整argv匹配，两个子进程均继承long10 pipeline、当前controller与实际主机allocation三锁。无failure/最终results/complete，平台无新回收事件、auto_stop为0。现场long10/runtime-recovery-20260929-1435.json，已下载为C8_LONG10_RECOVERY_RUNTIME_20260929_1435.json；新拓扑见C8_LONG10_RECOVERY_TOPOLOGY_20260929_1425.json。V0四片35040条原始读回SHA256与旧已审计来源完全相同，四份finished回执的new_mcq_counts均为空；恢复未重复推理完成键，见C8_LONG10_RECOVERY_READBACK_VERIFICATION_20260929_1433.json。以上PID仅本次快照。

单写14:34:34仍健康读回C8/V1：前两片已从3045/2957增至4924/4788行，后两片尚未开始；全部11680张C8 PNG已生成，B0尚未开始。两个Reader完整命令及根/当前controller两锁继承正常，GPU利用率29%/24%、9439/9419MiB，无failure/最终results/complete。现场C8_EVAL730_RUNTIME_20260929_1435.json。保持现有60分钟跟进，下一次实时检查两实例status/events、实际命令、锁与增长；不重复启动活任务。

14:26:24再次核验实际新主机、无CUDA或实验进程、long10/旧新controller/本机allocation锁后，后台运行新目录launch-two-gpu.sh，SHA256 5b46a1175dcaa84450c10aeffaae45f690f475177b82737fa3b0336b9af60200；日志recovery.log，launch.json记录完整身份。原调度器按完成键先核验跳过V0，再继续V1；不重新运行烟测、不改变权重或预算。14:27 controller20310和两个Writer20722/20723完整argv匹配，各继承三锁，无failure/最终results/complete，处于已完成V0的恢复检查阶段，尚不能仅凭启动称新增PNG已恢复。新execution-topology由实际主机与controller重新生成；迁移准备文件是未改调度器需要的入口标记，内容明确是同实例回收恢复。

恢复核验并逐字节复制1486条完整轨迹、16346张PNG；两条未完整链entertain_games:0028/seed1和:0029/seed1保留在旧目录、在新目录重建。全部35040条已审计C8/V0 readback及finished回执逐字节保留，具体复制哈希见C8_LONG10_RECOVERY_20260929_1425.json。中断原始元数据、日志、读回、平台事件和脚本归档long10-interruption-20260929-1346.tar.gz，共3022文件，SHA256 322a5b58f25f523d44a04b786b5c23d2e35e8dfb52eeeee108be13e2c92d89a1；本地下载并逐项核验全部清单哈希。原始PNG留在共享盘，已在恢复复制时全部重算哈希。启动记录C8_LONG10_RECOVERY_LAUNCH_20260929_1425.json。

14:19实际status为STOPPED，events记录prefeval-b-read-h200x2-20260925在13:46:41因CPU/GPU/MEM利用率规则stop-and-save，13:47:24完成停止。具体阈值未知；这不是原定时停止，也没有模型failure回执。仍遵循用户指定使用该实例，14:20只执行一次start，随后平台分配NORMAL两H200到qb-prod-gpu2294，新主机prefeval-b-read-h200x2-20260925--dd8912853710-iwj3wpwcbs。14:23实际两GPU各1MiB、无计算或实验进程，long10及旧controller锁释放；旧ivytbz3gqk已终止，旧controller/PID不再代表实时状态。平台证据LONG10_PLATFORM_STOPPED_20260929_1419.json和LONG10_PLATFORM_EVENTS_20260929_1346.json。

停止时C8/V0保留1460条完整11PNG轨迹及全部四片35040读回；C8/V1保留26条完整轨迹、两条部分轨迹。原目录long10/recovery-20260929-0005-2gpu只读。新独立恢复准备脚本scripts/inspire/recover_prefeval_long10_with_reads_20260929.py，源367fe3b、SHA256 b48b94728c2accda563e22dcfccc1481dafcee70bf30f496c76e7be93f29c1cb；使用原冻结链核验函数、6563d163 prepare及未修改的39263899两卡调度器，实际Writer/evaluator仍224cc77。两份权重、所有数据、每步原种子/交换/前图PNG及完整链全部重新校验。新输出仅long10/recovery-20260929-1425-2gpu，保留协议b43fd72a和四逻辑片两物理GPU执行拓扑；旧主机/PID绑定的topology及进程、sentinel文件不复制。V0读回须与已审计长程阶段原始哈希一致，并保留原png_path；新目录PNG按哈希对应，最终汇总包含全部复用轨迹，不改写原始读回。

单写730不受影响：14:19在g2xx4gn63g主机C8/V0和V1各5840PNG，C8/V1于13:55进入Reader，前两片3045/2957行；controller21763及Reader268715/268716命令和两锁继承正常，无failure/最终results/complete。现场eval730/runtime-monitor-20260929-1420.json。此前单写和长程V0阶段结果均已报告，此次回收不新增模型结论。

## 2026-09-29 13:27：单写730再次遭平台回收，独立目录接续

13:36:18再次核验已进入C8/V1实际生成，完整PNG从5598增至5616，两H200均100%、各14129MiB。controller21763及Writer82209/82210完整命令匹配且两锁继承正常；旧C8/V0的39420条原始readback哈希未变，已完成键未新增推理。现场eval730/runtime-recovery-20260929-1337.json。无failure或最终results/complete，后续仍按完整四分片及复用分母汇总。长程13:32 C8/V1为142PNG、12完整轨迹，两个Writer4163633/4163634与三锁均正确。

13:19实际status为STOPPED，远程执行失败；events记录13:13:53因CPU/GPU/MEM利用率规则stop-and-save，13:14:32停止。具体阈值未知，不由此推断Writer故障，也不伪造负载。13:20执行一次start，随后平台重新分配NORMAL两H200到qb-prod-gpu2187，实际主机prefeval-mt-eval-h200x2-20260928--9be487d35048-g2xx4gn63g；旧alucvbi43a已终止。平台证据见EVAL730_PLATFORM_STOPPED_20260929.json和EVAL730_PLATFORM_EVENTS_20260929.json。

CPU独立恢复脚本scripts/inspire/recover_prefeval_c8_eval730_20260929.py，源8a77887，实际SHA256 a15fd146ee60c41889a8ad6fa8afe72991a60d1d1b0a2ca6a393d485d4a9d8dd。重新核验两份权重、冻结代码/协议/数据、原种子及当前交换，逐图重算并逐字节复制11438条完整单写：C8/V0为5840、C8/V1为5598。两个未完成目录professional_work_location_style:0005/seed7和:0006/seed7不复制，在新目录按原种子重建。C8/V0四份原始readback和finished回执与12:25已审计版本哈希完全一致，共39420条；全部逐字节复制，保留原始PNG路径，且对应新目录PNG的实际哈希相同。后续完整审计必须允许这一有证据的原路径来源，不得擅改原始读回或只统计新增。

当前唯一单写GPU输出为eval730/recovery-20260929-1325，旧recovery-20260928-1920和fixed目录均只读。入口新目录launch-recovery.sh，SHA256 76ca3fa1f83954e9e0bbbe0bcc9d1d2886d8e34145414c79c5e85f151ebf1085，仍调用未修改的efe0f760控制器及224cc77 Writer/evaluator。13:26:56再次检查新主机GPU无计算进程、全部实验命令为空、根和旧/新锁空闲后后台启动，日志recovery.log，launch.json记录完整身份。原冻结Worker按完成键跳过已完成PNG及读回，因此先校验C8/V0再补V1；这不是重新推理或新增训练。恢复报告C8_EVAL730_RECOVERY_20260929.json、C8_EVAL730_RECOVERY_LAUNCH_20260929.json。13:32实测controller21763及Reader56424/56425命令匹配，各继承根及新controller两锁，无failure/最终results/complete；PID仅此次快照，仍需确认后续新增PNG。

中断原始回执/日志/读回/全部完整和部分链元数据归档eval730-interruption-20260929-1313.tar.gz，共22927文件，SHA256 b2231aacfc0a964de91e6b09ca8a0d24df944a8c13ca7be3536a37e65562b0eb。本地已核验归档哈希及清单内每个文件。长程在用户指定prefeval-b-read实例原样继续，13:19完整结束C8/V0全部四片35040条读回并自动进入C8/V1，不受本次回收影响。

## 2026-09-29 12:25：阶段结果只读审计，两GPU任务继续原协议

12:20实际现场：单写仍在prefeval-mt-eval-h200x2-20260928/alucvbi43a，C8/V0全部5840PNG及四片39420读回已完成，C8/V1生成4966PNG；原controller85531及当前Writer3577810/3577811的完整argv、两锁继承正常。长程在用户指定prefeval-b-read-h200x2-20260925/ivytbz3gqk，C8/V0共16060PNG/1460完整轨迹，前两片各8784条读回完成，后两片1029/1031条；controller1465983及Reader3575952/3575953完整argv和三锁继承正常。两平台无新停止事件、auto_stop为0，均无failure/最终results/complete。PID仅此次快照，现场各自保存runtime-monitor-20260929-1220.json。

用户询问阶段结果后，CPU运行独立脚本scripts/inspire/audit_prefeval_c8_v0_stage.py，读取完整预定C8/V0全部四片及实际5840PNG，未触碰运行checkout、预算或输入。输出仅eval730/analysis-c8-v0-20260929-1225，并归档到任务根c8-v0-stage-evidence-20260929.tar.gz。结果仅是V0切片，无B0或V1，详细分母、真实性核验、界限见RESULTS.md最新段及C8_V0_STAGE_RESULTS_20260929.json；不能作为完整730比较结果或改变模型选择。全量流水线与长程原样继续。

## 2026-09-29 00:11 CST：按用户指定迁至prefeval-b-read两H200继续长程

用户明确指定长程任务使用prefeval-b-read-h200x2-20260925。00:04现场核验该实例RUNNING、NORMAL两H200、节点qb-prod-gpu2260，实际主机 `prefeval-b-read-h200x2-20260925--dd8912853710-ivytbz3gqk`；两GPU各1MiB、无CUDA计算进程、无实验任务，long10及旧控制器/本机allocation锁空闲。原dl-clear-retain四卡仍排队，迁移后已取消排队并确认STOPPED；原用户实例保留不删除、不再start。

新输出仅 `long10/recovery-20260929-0005-2gpu`，从未启动GPU的恢复准备目录recovery-20260928-2310再次逐项hash校验并复制800条完整轨迹，即8800PNG、10400个轨迹文件。原固定目录、2310准备目录和readiness目录都保持只读，四条未完成轨迹在新目录按原种子重建。准备证据见[C8_LONG10_TWO_GPU_PREPARATION_20260929.json](C8_LONG10_TWO_GPU_PREPARATION_20260929.json)。

新调度脚本 `scripts/inspire/resume_prefeval_c8_long10_two_gpu.py` 冻结源befc4477361d266790bc99f5e2789b20e0d13158，实际SHA256 `392638990e1327d67283a8b07da53a849777d176ae1446fbccdceeb62567c6d1`，共享入口long10/resume-two-gpu-20260929.py。仅把四逻辑分片0/1/2/3按两批映射到物理GPU0/1；32条Writer/Reader命令与原控制器逐项比对完全一致，CPU调度检查最大并发2、映射0/1/0/1及三把锁继承通过。该检查是技术验证，不是实验成绩，见[C8_LONG10_TWO_GPU_VALIDATION_20260929.json](C8_LONG10_TWO_GPU_VALIDATION_20260929.json)。Writer/evaluator仍为冻结224cc77，准备、PNG链检查和最终汇总直接复用冻结6563d163函数，种子、输入、模型、预算、全部四逻辑片和评分不变。

原protocol.json逐字节保留，SHA256仍b43fd72a86e8a50547b359ac1ad7b0bf97bcdcbc90cb542f884206980caaae33。其中physical_gpus=4是原部署元数据，本次实际两卡及映射以execution-topology.json为准，不能在报告中声称本次四卡执行。每组仍须全部四片生成完成后才读回，最终汇总不能只用当前两片或只计新增。

00:11:24再次核验目标主机、GPU、实际进程、旧/新long10锁及恢复回执后，后台运行新目录launch-two-gpu.sh，SHA256 `9afaf5e0333af53f88c1f4f2cbe94f0279c40e6891730a12a4107bd0a7f60cca`，stdout/stderr为新目录recovery.log，launch.json记录完整身份。00:12:15实测controller1465983、Writer1466390/1466391命令与回执一致，两个子进程各继承long10/pipeline、当前controller及实际主机allocation三把锁；两GPU均100%、各14124MiB、无failure。启动后PNG已从8800增长至8808，说明实际继续生成。PID仅现场快照，后续仍须实时查主机及进程。平台auto_stop_in_seconds=0。启动记录见[C8_LONG10_TWO_GPU_LAUNCH_20260929.json](C8_LONG10_TWO_GPU_LAUNCH_20260929.json)，实际拓扑见[C8_LONG10_TWO_GPU_TOPOLOGY_20260929.json](C8_LONG10_TWO_GPU_TOPOLOGY_20260929.json)。

00:14:12再次核验两卡均100%、各14128MiB，PNG已到8822、完整轨迹到802，无failure；两实际子进程及三层锁继续正确。迁移代码、准备清单、协议、启动记录、实际拓扑、CPU命令/调度验证及现场共14文件归档long10-two-gpu-migration-20260929.tar.gz，SHA256 `c3fd6f526bb0adabe023a2f0066ba31e5edfe26acf25cb7b0b2e1ee9c6320d75`，本地已逐项核验。

另一实例prefeval-mt-eval-h200x2-20260928的单写730继续原任务，不受迁移影响。未来长程只监控并使用上述新实例、新恢复目录和新入口；旧四卡不再排队，原四卡launcher、烟测和2310四卡恢复入口都不得启动。

## 2026-09-28 23:13 CST：长程按定时停止，完整链恢复已备妥、四卡重新排队

平台实际记录22:27:16触发user timedShutdown，22:28:16完成停止及镜像保存。23:09新鲜status确认为STOPPED；这次是原定时停止，不是利用率回收。23:10已对原用户实例dl-clear-retain-h200x4-20260914执行一次start，原实例保留；23:13仍为NORMAL四H200的PENDING，不重复start。事件显示项目总GPU配额108、已用106、请求4，配额不足；尚未分配新的实际运行主机。原cwprkl2mqx和旧GPU732不再作为当前运行证据。平台现场见[C8_LONG10_REQUEUE_20260928.json](C8_LONG10_REQUEUE_20260928.json)。

CPU已完成停止后恢复准备：独立目录 `long10/recovery-20260928-2310`，重新hash两权重和冻结协议/代码，逐项核验并逐字节复制800条完整轨迹，即8800PNG与1600份complete/writes，共10400文件。四条未完成轨迹shop_home:0002至0005的seed0未复制，将在新目录按原参数重建；原目录共8835PNG，保留不动。新目录正式协议SHA256仍为b43fd72a86e8a50547b359ac1ad7b0bf97bcdcbc90cb542f884206980caaae33。报告[C8_LONG10_RECOVERY_20260928.json](C8_LONG10_RECOVERY_20260928.json)，不是GPU启动回执。

下一次资源RUNNING后，必须先核验实际新主机、全部四GPU、完整任务命令和long10/旧新controller/allocation锁，确认无活任务或持锁子进程，才后台运行新目录 `launch-recovery.sh`，SHA256 `548c532edcbd78ef744a511a8fa594dee57e2dc24ce9b934694a6774ad0502d1`。仍调用原6563d163控制器和224cc77 Writer/evaluator，只运行正式阶段，不重跑烟测。stdout/stderr写新目录recovery.log，保存launch.json，然后验证子进程继承三把锁、完整argv和新PNG增长。不要重跑已经成功的准备脚本，也不要启动旧fixed目录或readiness目录；PENDING时不进行GPU启动。最终统计从恢复目录读取，分母包含复用的800条轨迹。

完整中断元数据/控制器日志/部分链writes和停机事件另存long10-interruption-20260928-2227.tar.gz，1624文件，SHA256 `ceb7d826106019181787e8636f9ab6e4293980d9d000d3373807418c6c107dd7`。本地已下载并核验归档hash及1600份完整链元数据；大PNG的实际hash绑定在恢复报告。已完成链元数据子归档1606文件，SHA256 `f44b7dc7ae923242c40a1a6ddb9801fa682eba9b3dfa668c900ba6bcf62686b3`。

单写730不受影响：23:10:25实际alucvbi43a主机两GPU均100%、各14129MiB，C8/V0为3332张完整PNG，controller85531及当前Writer1842880/1842881命令和两锁正确；这是原控制器进入后两个逻辑分片，不是重启训练。仍无正式读回或最终成绩。自动跟进配置实际保存为60分钟，之前文字中的10分钟不能当作实际调度频率；排队和恢复状态以每次实时核验为准。

## 2026-09-28 21:38 CST：长程定时停止前的独立恢复准备

21:33:46实际核验：单写730在alucvbi43a主机持续生成C8/V0，2240张完整PNG，两GPU100%、各14129MiB，原恢复controller85531及两Writer完整命令匹配，根和恢复锁持有；长程在cwprkl2mqx主机已有7634PNG、692条完整11PNG轨迹，四Writer命令与三层锁正常。两分支没有failure/results/complete，也没有正式读回。平台没有新回收事件；长程auto_stop_in_seconds=3196，仍预计22:26停止。瞬时一张卡利用率为0但显存及实际进程存在，不能据单个采样断言任务停止。

恢复准备21:38完成，晚于原定21:30，仍早于定时停止；原GPU任务持续运行，未抢停或伪造负载。新脚本 `scripts/inspire/prepare_prefeval_c8_long10_recovery.py` 冻结提交 `c41ec03a2eebe76ba0d2c279f2837baec32f6075`，实际SHA256 `76bb631064b256343b48c8c837227514e8a105d5ce802937ef370fb846e8bc9b`，共享盘副本 `long10/prepare-recovery-20260928-2135.py`。在CPU入口执行只读审计，重新核验原6563d163控制器、224cc77工作代码、两权重及全部协议数据；生成协议哈希仍为 `b43fd72a86e8a50547b359ac1ad7b0bf97bcdcbc90cb542f884206980caaae33`。

快照中的700条完整轨迹、7700张实际PNG及所有complete/writes逐项校验通过：每条恰好11PNG/11次写入、每步原噪声种子、当前交换、前一张PNG链接及输出哈希均一致。错误seed、错误前图链接、错误交换三个负例均被拒绝。四条当时未完成的轨迹没有复制；本次全部copied_files=0、GPU launch=false，属于恢复准备而非恢复已启动。报告 [C8_LONG10_RECOVERY_READINESS_20260928.json](C8_LONG10_RECOVERY_READINESS_20260928.json)；原始完整链元数据归档1406文件，SHA256 `3d1e69010e877f19f7cdd7f2ca36cc6593ce5f284519cb764712b8a494f767c9`，已下载并核验其中1400个complete/writes文件哈希。原始PNG保持原目录不动。

后续在平台确认该原实例STOPPED后，将新鲜status JSON存到共享盘，调用上述脚本（不带--audit-only，带--stopped-status，指定全新 `long10/recovery-20260928-2230` 或按实际恢复时间命名）。它拒绝非STOPPED/超过5分钟的证据或已出现Reader输出的情况，重新扫描并核验停止时的全部完整链后逐字节复制；部分链完全不复制，在新目录按冻结种子重建。脚本生成只运行正式阶段的launch-recovery.sh，不重跑烟测。运行中的原目录和readiness目录都不能作为新GPU入口。

资源默认在原实例实际STOPPED后start一次，PENDING时不重复start，原用户实例保留不删除；若原资源无法获得，可另申请NORMAL四H200。RUNNING后核验新的实际主机、四GPU、全部任务进程和分支/controller/allocation锁，确认无活进程或持锁子进程，再后台运行新恢复入口，记录launch.json和recovery.log并核验实际PNG增长。跨主机FREE锁不构成旧任务结束证据。停止前不重复启动，也不修改原运行checkout。本次自动跟进临时改为10分钟，覆盖约22:26停止及后续恢复，确认健康后恢复45分钟。

## 2026-09-28 19:23 CST：平台自动回收后，在新容器独立恢复730

用户指出两卡节点没有资源占用。19:18实时复查确认属实：两卡0MiB/0%、无任务进程、根及旧controller锁FREE。平台事件明确记录19:00:54因CPU/GPU/MEM利用率自动回收规则执行stop-and-save，19:02:02停止；19:03:28重建容器，19:03:55ready，但原评测进程未恢复。18:55:54旧容器仍实测两GPU100%，因此旧现场不能作为19:18的运行状态。具体策略阈值及重启发起者未从事件中确定，不能推断。节点仍qb-prod-gpu2226，主机已从tphujecw4c变为 `prefeval-mt-eval-h200x2-20260928--9be487d35048-alucvbi43a`。事件见[EVAL730_PLATFORM_EVENTS_20260928_1900.json](EVAL730_PLATFORM_EVENTS_20260928_1900.json)。

原 `eval730/fixed-c8-b0-20260928` 保持只读，保留765份完成的C8/V0单写轨迹和两个未完成目录；无读回。恢复准备脚本提交 `6f2fa151f70f91661380c76ea247e2887740616f`，实际SHA256 `679483acbd2c999768b37d6ca6f4917fe95828d82ebcd22e221d59195716663b`。重新计算两权重、全部协议数据与冻结代码身份，并逐项核对765PNG、complete、writes中的种子、当前交换、源图和输出哈希；三文件逐字节复制到独立 `eval730/recovery-20260928-1920`，未完成目录不复制，按原种子重建。准备证据见[C8_EVAL730_RECOVERY_20260928.json](C8_EVAL730_RECOVERY_20260928.json)。原始中断证据1546文件归档SHA256 `59f5d47757304dd4ea4201ca4b0e5d8b544659c17ba2423d3397191d2ac372f7`。

19:23:05再次核验新主机两GPU零占用、全部任务命令为空、根/旧锁释放后，后台运行恢复目录launch-recovery.sh，仍调用冻结efe0f760控制器及224cc77真实Writer/evaluator，共享vlm-r3-ngc2502环境、权重/种子/4逻辑片两GPU批次/评分完全不变。协议SHA256仍为c670e3db2d7d54bf449999cf693d7544fb54bbc7040420b641c31484e6f641dc。日志仅新目录recovery.log，launch.json记录完整命令与主机；旧全流程入口不再使用。19:23:53实测两卡100%、各14129MiB，765张完整PNG已增至767；controller85531、Writer85949/85950命令匹配，子进程继承根及新controller两锁，旧锁FREE，无failure/results/complete。PID仅现场快照，后续必须核验新主机和实际进程。

恢复后临时10分钟跟进，确认持续增长及无新回收后恢复45分钟；报告状态必须附实际检查时间，平台RUNNING不能代替实验进程证据。后续结果从新恢复目录完整results/complete和全部四分片汇总，分母包含复用的765图。长程分支不受本次恢复操作影响，未修改其他实例或伪造资源占用。

## 2026-09-28 17:52 CST：第三轮归档后接续固定C8/B0全730单写评测

17:54:37启动后实测：两卡均100%、各14129MiB，C8/V0已由0增长至10张实际PNG及10份完成回执。两Writer实际argv与回执逐项一致，均继承两层锁，根及eval730锁HELD，旧round3锁FREE，无failure/最终results/complete；协议实际哈希与预检冻结值一致。现场完整证据见[C8_EVAL730_LAUNCH.json](C8_EVAL730_LAUNCH.json)。确认健康启动后，将prefeval-writer从收尾临时10分钟恢复45分钟；正常运行保持静默。启动并不表示全730成绩已完成。

第三轮17:41:45完整结束，全部16读回分片各864行。独立最终审计及本地归档核验通过，结论见RESULTS.md与ROUND3_RESULTS.json；已完成的round3/pilot64保持只读。17:50:58及17:52:59两次现场核验：原NORMAL两H200主机 `prefeval-mt-eval-h200x2-20260928--9be487d35048-tphujecw4c` 的两GPU均0MiB、无实验任务进程，根pipeline及所有本机旧controller锁FREE。

17:52:59在同一实例后台启动正式入口 `repos/prefeval-c8-eval730-20260928/scripts/inspire/launch_prefeval_c8_eval730.sh`，冻结commit `efe0f7606ed7582a0d4171525b9bc285f1c1d140`，控制器SHA256 `ab4d702b41115bcfd888d5ef7cb561577eaf8af88b035830395ebbe45be2a8cf`，launcher SHA256 `748e2d29d0e1610f628f8311986f9dcbe3a07f955868abaa0c98dad9c7c34f99`。实际launch记录 `eval730/launch.json`、日志 `eval730/pipeline.log`，输出仅 `eval730/fixed-c8-b0-20260928`。未使用ready备用checkout，也未重启第三轮或教师/训练。

17:53:02进入generate-C8-V0；17:53:30核验controller486352与Writer486747/486748完整命令匹配回执，两个子进程均继承根与eval730控制器锁，尚处模型加载。launcher486332及上述PID仅为启动现场快照，后续必须重新核对实际主机/命令/锁，不据PID判断运行状态。冻结C8/B0与协议身份在C8_EVAL730_PREFLIGHT.json；没有新增教师或FM更新。

长程10轮在另一四卡主机继续独立运行；17:41:41实测C8/V0已有2427PNG、220完整轨迹，四Writer与三层锁正常。两卡本机显示长程锁FREE不代表远端停止，必须在长程实际主机核验。该用户原四卡仍预计22:26自动停止，须21:30前安排续跑或独立恢复准备，不删除原实例、不触碰C/R或B730资源。

## 2026-09-28 15:49 CST：长程10轮在空闲四H200独立启动

最新核验15:54：烟测15:52:58完整结束，22PNG/60唯一读回，两条完整10轮PNG传递及当前交换哈希全部通过。15:53:02自动进入正式730的C8/V0；正式controller389148、四Writer389545–389548，15:54:07四卡100%、各14129MiB，完整argv与回执一致、均继承三把锁，smoke锁FREE，正式及allocation/pipeline锁HELD；正式PNG已从3增长至12，无failure。正式输出尚无results/complete。详见[C8_LONG10_SMOKE_RESULTS.json](C8_LONG10_SMOKE_RESULTS.json)及[C8_LONG10_PREFLIGHT.json](C8_LONG10_PREFLIGHT.json)。

烟测含原始PNG、全部读回、writes、manifest、完成回执、协议和启动现场的归档 `c8-long10-smoke-evidence.tar.gz`，54文件、20699684字节，SHA256 `e4fbff6f8e08ed89501771e06c6994b84a86b569d8b9a9259a2f714a39fabcab`。该阶段按完整性放行；两条样本T1为0轮2/2、5轮0/2、10轮0/2，不能外推全730结果，也不根据它调模型或缩短预算。

用户追加真实PNG长程10turns并行评测。原high四卡在启动前复查时已被C/R接手，未在其上启动长程进程。改用已经RUNNING的 `dl-clear-retain-h200x4-20260914`，NORMAL四H200、qb-prod-gpu732，实际主机 `dl-clear-retain-h200x4-20260914--a823c55e800a-cwprkl2mqx`。15:48:53确认全部GPU零占用、无实验进程及本分支持锁者，15:49:29后台启动；没有start、停止或删除用户原实例。

独立冻结checkout `repos/prefeval-c8-long10-20260928`，提交 `6563d16322c792d6ad0007c17cd6be9c6ca279e3`；入口 `scripts/inspire/launch_prefeval_c8_long10.sh`，控制器SHA256 `c5a9051c1d8426f45e017b69c960188c0eacf2c2dc507a782fc264a63783e4ce`。实际Writer/evaluator仍为224cc77，不修改旧checkout。日志 `long10/pipeline.log`、身份回执 `long10/launch.json`；先 `long10/smoke`，完整性通过后自动进入 `long10/fixed-c8-b0-730-20260928`。

CPU预检实际核对730条22消息结构、固定C8/B0权重和原始数据哈希、每轮当前交换哈希；合成数据验证0/5/10计数、初始正确条件保持率、按偏好配对bootstrap及重复/缺行/错donor拒绝。预检不是模型成绩。正式协议SHA256 `b43fd72a86e8a50547b359ac1ad7b0bf97bcdcbc90cb542f884206980caaae33`，烟测协议 `57ecc7aaec91b4bf32209cc2c0d4cc7f2e38bcbff8069b1d9fe5c6f4e2799c63`。正式每模型2920条轨迹、32120张PNG、70080条读取；同730、V0/V1各两预设种子，冻结C8对B0，零教师/FM更新，见PLAN。

15:50:06核验启动器358284、烟测controller358305、实际Writer358331/358332，完整命令与进程回执一致，继承 `long10/pipeline.lock`、smoke/controller.lock及以实际主机命名的allocation锁，三把锁HELD。这些PID仅当次快照。两卡烟测完成后正式阶段使用四卡；另一两卡实例上的round3正常推进，单写730仍接在round3完整归档之后。

资源限制：该用户实例原有定时停止，15:48平台剩余23890秒，截止约2026-09-28 22:26 CST；当前CLI的notebook/start命令无运行时长修改入口，未声称取消成功。现有跟进须在21:30前检查/安排后续资源或独立恢复，并持续记录实际平台截止；不伪造负载、不删除用户实例。若停止，完整链按complete及11张PNG/写入记录哈希只读复用；部分链在独立恢复目录按同权重/种子重建，不能向旧writes.jsonl重复追加后冒充完整链。

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
