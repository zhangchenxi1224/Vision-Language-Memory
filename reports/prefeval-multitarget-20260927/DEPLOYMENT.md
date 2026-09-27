# 多目标主线部署记录

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
