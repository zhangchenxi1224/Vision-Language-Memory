# B730 固定曝光扩展：四卡主实例与两卡评测并行

**23:12 CST 阶段更新：训练完成固定 93,440 步，完整断点和日志核验通过，禁止重复训练。17/19 项评测完成；pilot64/dev90 最终两版已归档并下载复算，只剩主 GPU1/2 的 train730 V0/V1。最终 loss 和两类阶段结果见[训练终点报告](B730_EXPOSURE512_TRAINING_FINAL_20260928.md)、[pilot64](B730_EXPOSURE512_STEP93440_PILOT_20260928.md)、[dev90](B730_EXPOSURE512_STEP93440_DEV_20260928.md)。reader 已发布完成 sentinel 并正常退出；根 complete/results 尚未生成。以下部署过程是历史记录。**

2026-09-28 20:14:43 CST，依用户最新要求“优先在4卡实例上部署，并行运行”，已启动四卡主实例；20:15 启动两卡评测调度。科学代码与绝对 93,440 步终点均未改变。

**21:00 前最新核验：20:27:49 已验证完整 89,856 步断点，69 个重放更新的 draw、loss、梯度范数全部精确一致。20:55 在不停止科学训练进程的条件下修正调度器；20:56:25 日志已连续推进到 90,966。主 GPU0 约 40,746 MiB、利用率 99%，13 份完整汇总保持不变，尚无 93,440 步最终成绩。**

| 实例 | 节点 / 优先级 | 分工 |
|---|---|---|
| `prefeval-b-cr-h200x4-20260926` | `qb-prod-gpu997` / NORMAL | GPU0 完成训练后评测 pilot64/V0；GPU1 train730/V0；GPU2 train730/V1；GPU3 dev90/V0 |
| `prefeval-b-read-h200x2-20260925` | `qb-prod-gpu2260` / NORMAL | GPU0 pilot64/V1；GPU1 dev90/V1 |

主实例实际主机 `prefeval-b-cr-h200x4-20260926--c1325f0297e5-ahequpaewo`，当前主启动器 195370、控制器 195391，原训练进程 **65159 保持不变**。辅助主机 `prefeval-b-read-h200x2-20260925--dd8912853710-ivytbz3gqk`，当前启动器 376001、控制器 376035。原调度器 65077/179952 已退出。PID 仅为部署证据，后续操作必须核对当前 host、命令、cwd 与锁，不能在另一台机器上按同 PID 判断进程。

两实例均为 NGC 25.02、H200，驱动 570.124.06；先前低优四卡使用 595.58.03。PyTorch/CUDA runtime 沿用已验证的 `2.7.0a0+ecf3bae40a.nv25.02` / `12.8`。重放核验以实际数据为准，不预设跨驱动逐位相等。

迁移恢复源为完整 **89,728** 步断点，cursor=358,912，1,075 份 AdamW 状态步数一致，Python/NumPy/torch CPU/CUDA RNG 齐全。实际文件 SHA256 为 `37c3d87457242d75a4f4989ab175c4cec8c8fa0cbccd13c73e194ed541eb2ea0`。迁移前日志到 89,797，未提交的 69 步已归档并精确重放，不额外计入预算。新完整 **89,856** 步断点 SHA256 为 `34ff9f3cab0b608d96506961c8bf1c75b4c3b58e8654f5f5ff6ec9894be6675a8`，优化器全部同一步，完整日志连续唯一。当前训练已继续超过此保存点。

20:16 现场确认：主训练 GPU0 正在重建条件缓存，两台控制器均存活，主/辅助 launcher 与 controller 锁正常，未见 failure。此时最终权重尚未生成，评测进程在等待它；不能把已部署的六卡资源写成六卡已经同时计算。终点产生后六项矩阵能够并行执行。

## 冻结代码与调度边界

共享输出 RUN 为 `/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-b730-exposure512-20260927`。

- 科学代码仍为 `RUN/code`，提交 `656fdf029c4a7c05e53bccb75483a03cf62d5f54`，工作区洁净。原 23,360 步目录只读。
- 当前独立调度代码位于 `RUN/operations/distributed-20260928/code`，冻结提交 `44d251dc629212b61f7373de4875b74dbc459b76`。旧 `operations/two-gpu-20260928` 的 2bd7230 调度已退出，保留为历史。
- 唯一当前入口为调度目录下 `scripts/inspire/launch_prefeval_b730_exposure512_distributed.sh primary` 或 `reader`。旧 `two_gpu`、`auxiliary` 及科学代码中的四卡入口均不再用于恢复。
- 调度器导入冻结 core 的 train/evaluate/summarize；没有修改 Writer、FM、教师、读回、有效 batch、样本顺序或步数。GPU0 保留单卡顺序 micro-batch 累积。
- 实测共享盘的 `flock` 只在本机生效，因此以**固定唯一主机分工**保证跨机不重复：主实例绝不执行 reader 的 V1 pilot/dev，只等待其完整 sentinel；reader 故障必须在 reader 主机恢复。文件锁仅用于本机重复启动/同项评测互斥。主调度器只在 19 个完整汇总、训练终点及 reader 完成证据齐全后写根 results/complete。
- 6 项 Linux 调度测试通过，涵盖完整登记范围、唯一训练调用、本机同项互斥、失败取消、主实例只等待 reader，以及接管已有进程而不虚构退出码。主实例原 9 项预算边界测试及冻结预检通过。

原 13 个完整评测 summary 的实际 hash 全部保留；70,080 步六项矩阵的 3,536 份完成 PNG 元数据、18,000 条原始读回前缀保持一致。评测只覆盖原登记 train730 T1、pilot64/dev90 同图 3+2、V0/V1 两噪声及四控制。dev 只报告，不用于加预算或选模型。

## 资源切换记录

旧 `prefeval-k1-l012-h200x4-20260924` 已停止排队并保留对象。20:06 曾按前一条指令在两卡实例启动；收到“优先四卡”后，在条件缓存阶段停止，未产生新增优化步。初始四卡辅助等待器也已停止，确认退出及锁释放后才启用新分工。`priority-switch-20260928` 保存两侧停止证据；旧 `recovery-20260928-2000` 与 `auxiliary-deployment-20260928` 是历史，不重复执行。

当前四卡迁移证据在 `RUN/recovery-20260928-2015`；辅助部署证据在 `RUN/reader-deployment-20260928`。一次性启动守卫有已存在收据，不能重复运行。C/R 实验当前在另一个 NORMAL 实例 `prefeval-k1-h200x4-high-20260924`，本次没有停止或改动它。

20:55 调度修正证据在 `RUN/coordinator-switch-20260928`。只向两个旧控制器发送 TERM，训练 PID 65159 及其 `/proc` start ticks `901642051` 前后相同。新主控制器使用 `B730_ADOPT_TRAIN_PID=65159` 接管等待，未重建训练缓存、未重放已提交更新、未新增训练进程；切换前 90,913、后续实测 90,966。`RUN/train-adoption.json` 保存完整身份。

如果今后主控制器失效但训练仍活，先验证实际 host/PID/完整命令/cwd/start ticks 和本机控制器锁，再通过同一环境变量接管等待，不能启动第二份训练。若训练确实已退出，则按实际完整 resume 恢复。由于接管者不是原训练进程父进程，最终训练 receipt 的 OS exit_code 将为 null；必须依冻结 Writer 发布的 train/complete.json steps93440、完整权重/优化器验证和实际进程退出证据认定完成，不声称拿到了该进程退出码 0。

四卡实例 Jupyter 终端 POST 曾超时，但本次调用实际创建了 terminal `1`（12:02:33 UTC）。已通过 GET 列表和 WebSocket核实，临时包装器 `b730_reuse_cr_terminal.py` 仅复用这个终端，未修改 CLI/平台权限/实验代码。普通 CLI 成功时直接使用；需要包装器时，同一个 terminal 串行调用。

## 后续执行与收尾

heartbeat 已更新为当前主/辅助实例与入口。主机状态使用 `b730_observe_distributed_20260928.py` 分别在两台运行，分别写 `runtime-primary.json` 和 `runtime-reader.json`，只对本机 receipt.host 的 PID 做实际检查。

本地证据位于 [migration evidence](evidence/b730-exposure512-migration-20260928)。恢复原始包 SHA256 `d01575a55fc35320fd3bccf4c8ff76b85412b421241199f354dda7c36e9d734b`，本地复算全部 69 个重放更新相同、日志连续，13 汇总/3,536 PNG 元数据/18,000 读回前缀保全。最终双机部署包 366,257 字节，SHA256 `aae2adb8642f59c6699e1bb6d537016e2d3eeffa25c00fa80721f72e21dc0e25`，含本机实际身份、角色切换、科学训练接管、运行时和调度源文件。初次跨机锁审计失败的未加 `-final` 目录是诊断中间产物，不是完成证据。

训练达到 93,440 后归档完整 AdamW/RNG、全 draw 日程及 loss；六项最终矩阵按完整分母归档原始读回、实际 PNG hash、匹配收益、两噪声全对和相对 23,360 的修复/退步。待 19 项汇总、根 complete/results 和两台正常退出证据齐全后，使用 `b730_audit_final_completion_distributed.py` 分别核验两台无其他 GPU 任务，停止两实例但保留对象和共享盘，随后删除本 heartbeat。

截至本次部署，最新完整性能阶段仍为 70,080：train730 V0/V1 匹配 1307/1460、1314/1460；pilot64 T1 均 119/128；dev90 匹配和错配均 80/180。迁移成功本身不是新的性能成绩。
