# C/R 中断归档：2026-09-28 13:35 CST

13:32 现场查询确认 `vlm-r11-trust-h200x4-20260907-r4` 已 STOPPED，两个普通候选 `b-cr4-normal-0927` 和 `prefeval-b-cr-h200x4-20260926` 仍为 PENDING / NORMAL(priority 4)。未重启实例、未新建请求、未占用其他实验资源。

平台事件记录 13:23:11 因 CPU/GPU/MEM 使用情况不满足管理员自动回收规则而停止并保存，13:29:33 完成停止及镜像推送。仅据该事件不能确定具体触发阈值或原因，不将其称为已证实的低优抢占。完整事件见 `evidence/interruption-20260928-1335/platform-events.json`。

## 共享盘现场

通过 CPU notebook 对共享盘作只读核验及证据归档；未在已停止的旧主机直接检查 /proc。平台 STOPPED，C refresh、C launcher、R refresh 三个共享锁均可非阻塞取得并释放，状态 FREE。

| 项目 | 中断时完整链 | 实际 PNG（含未完成链） | 阶段 |
|---|---:|---:|---|
| C final-train-V1 | 391/1460 | 4307 | 独立终点评测生成，尚无全量成绩 |
| R round-3 源图 | 278/730，两分区各139 | 2783 | 第四段源银行未冻结 |

三条未完成链的 9 张实际 PNG 已另存：C `travel_restaurant_0042/seed-1` 的 prefix-00 至 05；R shard-0 `lifestyle_dietary_0024/seed-0` 的 prefix-00；R shard-1 `lifestyle_dietary_0027/seed-0` 的 prefix-00 至 01。恢复前保留这些证据，按原种子恢复未完成链，核验并跳过已完成链。

R 完整主优化日志仍为 1–17520，连续唯一。round-3/bank.json 不存在，尚无 17521 梯度。冻结协调代码提交仍为 `55596810cee2062022b894ef93feb1ab4cd6b07b`。

## 实际断点身份

- resume.pt：4680992656 字节，SHA256 `86f4536f99b6f008b6cc2c81e3b3e349314c30f8aa9d6d4d10d97e8a6e7ce9bb`。
- checkpoint-step-017520.pt：1560329525 字节，SHA256 `3da185795c65c8a3425086040f83cab18d5c334ef68055b2e4e8603025e52998`。

两个实际文件哈希与 09:20 完整审计一致，绑定已核验的 step17520/cursor70080、1075 个优化器状态全部17520，以及 Python/NumPy/torch CPU/CUDA RNG。此次在 CPU 上重新读取文件计算哈希，未重新反序列化 resume；完整状态审计见 `B_MCQ_PROGRESS_20260928_0920.md`。

## 原始证据

本地与远端同名目录 `evidence/interruption-20260928-1335` / `runs/prefeval-b-mcq-20260925/interruption-20260928-1335`。

`raw.tar.gz` 为 14638334 字节，SHA256 `c513f49f02d00aaeadc960cf8593a1a56795c1c4d2a9b4b0bfe20578e01235c9`，本地下载哈希与远端一致。

归档包含 C/R 驱动及训练原始日志/状态、完整优化日志及历史未提交尾部、当前 C train 与 R round-3 的 manifest/complete/writes 收据、全部当前 PNG 的哈希与 mtime 清单、9 张未完成 PNG 实体、运行现场快照和归档文件清单。完整链 PNG 实体、模型及 resume 留在共享盘；不声称压缩包包含全部图像实体。`verify-archive.py` 用于复核 1483 个清单文件的哈希、连续步数、锁状态、计数、未冻结银行及 9 张未完成 PNG。

## 后续边界

继续保留并每10分钟查询两个 NORMAL 候选。普通实例 RUNNING 且实际 GPU 空闲后，依据本次实际17520/round-3状态准备恢复守卫；不能直接执行旧11680迁移守卫或 launch-trust.py。恢复前再次确认旧实例停止、父子任务不再存活、共享锁释放；不并行启动副本。维持冻结代码、原预算、原种子与完成键，不重做已完成 dev/pilot/源银行。

本次是资源中断及可恢复性证据，不是新的记忆性能结果。C 全量 train 评测与 R 最后源银行、最后5840步及终点评测仍待完成。
