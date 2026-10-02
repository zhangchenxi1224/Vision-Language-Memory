# 两张H200接续第二轮评测

2026-09-28 00:40平台事件显示项目GPU从108/108降至106/108，原四卡申请仍PENDING。余下任务仅为独立推理评测，因此用两张H200分两批运行原四个逻辑分片，所有模型、种子、偏好、donor、生成参数、输出解析和评分分母均保持冻结。

## 资源与入口

- 新资源prefeval-mt-eval-h200x2-20260928，NORMAL/priority4，2 H200、40 CPU、400GiB、128GiB共享内存；原同组开发区-H200-3号机房-2-cuda13.2版本，同镜像ngc-pytorch:25.02-cuda12.8.0-py3。
- 00:42:41申请后即分配qb-prod-gpu2226；00:44实测RUNNING，主机prefeval-mt-eval-h200x2-20260928--9be487d35048-tphujecw4c。
- 原用户实例prefeval-k1-h200x4-high-20260924在确认新资源已分配后停止排队，实例保留未删除，避免两个申请同时占用资源。B730、C/R未干预。
- 输出仅runs/prefeval-multitarget-20260927/round2/recovery-20260928-0045。旧recovery-20260927-2135保存中断归档但从未启动，其四卡脚本不再使用。
- 实际resume_eval.py来自ca908b6，SHA256 c1dec268b9e3b546b377e7fbe704aa6865f43c04d5b3d6ad208d1dfe6517b3c5。仅修改执行并发与恢复输出目录；调用的Writer/evaluator仍为repos/prefeval-multitarget-round2冻结224cc77d790cf3967b5a56ce2e77c364959435a2。

## 启动前核验与实际执行

00:44:50检查两卡均0MiB且无计算进程；无PrefEval旧进程。根pipeline.lock、首轮/第二轮smoke及原pilot64控制器锁均FREE，新恢复目录无launch/failure/complete记录。脚本SHA一致。未修改冻结代码或共享环境。

00:45:15通过独立后台会话启动一次，controller PID8515，launch.json记录完整命令与主机。stdout/stderr进入recovery.log。新控制器持有根pipeline.lock、旧round2/pilot64/controller.lock及新controller.lock，传给子进程。

00:45:21进入eval-read-S1-V0。在进入该阶段之前，脚本实际重新计算两份2048步权重及512张原PNG的SHA256，并逐一核对完成收据；检查通过，protocol.json及reused-inputs.json已写出。2720条旧读回逐字节复制到新目录，由冻结评测器按原键跳过。S1/V0使用旧PNG只读路径，其他三版本将写新目录。

四个逻辑分片仍为0、1、2、3，先0/1后2/3；映射物理GPU为shard%2。聚合依然读取四份readback文件，不能把并发数2误写成逻辑分片数2。不会重做教师或FM，不新增评分样本或挑选图片。

本报告记录资源恢复与执行身份，尚无S1/C8完整性能结论。原中断的2036文件归档及哈希见ROUND2_RECOVERY_20260927.md。

00:47:02实际复核：原始2720行增至2970行（四分片808/806/679/677），运行中的shard0/1 PID9126/9127完整/proc命令与进程收据一致，两GPU约9.5GB显存并有计算活动。根、旧控制器及新控制器三把锁全部HELD，failure为空。现场runtime-0048.json使用内含UTC为准。自动跟进恢复45分钟，以新目录的完整结果为准；此增量是恢复执行证据，不是准确率结果。
