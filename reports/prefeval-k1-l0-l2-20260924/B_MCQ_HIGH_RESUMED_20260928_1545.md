# C/R730 在空闲 NORMAL 四卡实例恢复（2026-09-28 15:45 CST）

用户明确授权：“随便一个空闲实例都可以，优选4卡，不要与其他任务抢实例”。2026-09-28 15:41:24 CST 已实际恢复 C/R，未增加方法、训练步数、评测预算，也未停止其他实验。

## 资源选择与启动

- 实例：`prefeval-k1-h200x4-high-20260924`，平台现场为 RUNNING / NORMAL priority 4，4×H200。
- 实际主机：`prefeval-k1-h200x4-high-20260924--506d5fcf84c1-rnoaxyojmb`。
- 启动前四卡无计算进程、0 MiB；逐项读取 /proc，未发现其他实验进程，C refresh/launcher 与 R refresh 三锁 FREE。此前该实例的 multitarget 历史状态已不适用。本次没有停止、移动其他任务。
- 最初只读检查 l012 时空闲；完成共享盘审计后再次检查，发现 B730 exposure512 已启动且其协调器会用四卡，因此未在 l012 启动 C/R。两卡 mt-eval 仍由 multitarget 使用，也未动它。禁止使用的 dl-clear 未使用。
- 新守卫 `high-recovery-20260928/launch-high.py` 已执行，不能重跑。旧 trust / 11680 守卫未执行。
- C 父 PID 320836，shell 321051，GPU1；R 父 320847，round3 源生成 321839 / 321842，GPU2 / GPU3。PID 仅为本次现场记录，后续必须重新核实。
- C/R 协调器固定提交 `55596810cee2062022b894ef93feb1ab4cd6b07b`；C 内部仍使用 `52d98d85f84e1e4910e922e34f59f5f14bbabf5c`。完整命令、cwd、CUDA 环境和继承锁均有现场记录。

## 断点、预算及已有结果校验

在共享盘对实际 R resume 使用同一文件描述符先计算 SHA256，再反序列化：

- optimizer_step=17520，episode_cursor=70080；
- 1075 个优化器状态全部为 step17520；
- Python / NumPy / torch CPU / CUDA RNG 齐全；
- trainer_state 绑定 refresh_round=2 和 bank `4ca3bc2fc0d11394a1f6e22119a539e7656039a6b438f7ba2a71058b2b44c234`；
- resume SHA256：`86f4536f99b6f008b6cc2c81e3b3e349314c30f8aa9d6d4d10d97e8a6e7ce9bb`。
- 新主机启动守卫再次计算实际 resume hash，与上述值一致。
- B/C/R 权重、C 固定银行和 R round2 银行实际 hash 与归档一致。
- 完整 R 优化日志复核为 1–17520 连续唯一；历史未提交尾部仍单独保存。
- 恢复前 C train 4307 PNG、R round3 2783 PNG 全部 hash 和 mtime 与 13:35 中断清单一致。
- 恢复后对旧完整链实际 PNG 再查：C dev180/1980 PNG、pilot128/1408 PNG、train391/4301 PNG、R 两分区各139/1390 PNG 均与原 complete 收据一致，PNG mtime 早于启动。C 已完成主读取和四个位置读取文件逐字节未变。

恢复前原始日志、收据、状态等保存于共享盘 `l012-recovery-20260928/before`，该目录只是只读审计阶段的归档命名，绝不表示在 l012 启动了 C/R。本次 raw 包已包含这份冻结快照。

## 实际续接

R 从 round3 的每分区139条完整链继续生成，15:43已各140条；随后现场已各141条并持续生成新 PNG。当前源权重仍为 R17520，V1 / refresh-3；完整730链银行冻结后，原驱动自动继承优化器/RNG继续17521–23360。当前未宣称第17521步已开始。

C 原冻结 shell 按完成键核验并跳过已完成训练、dev、pilot，已进入 final-train-V1 rollout（PID374722，GPU1），从原391条完整链续接。没有重新训练 C，也没有重复完成评测的推理。GPU0保持空闲。

运行快照中的三个锁均 HELD，父子进程继承锁正常。后续以实际进程和输出为准，不重复启动副本。

## 归档

本地目录：`evidence/high-recovery-20260928`；远端同名目录在 `runs/prefeval-b-mcq-20260925` 下。

- raw.tar.gz：12,170,045 字节。
- SHA256：`53144ad89c530f29670f8dd9d4672d5ca3b8d3ce5707905f8e51349b4eca5fa0`。
- 包含恢复前完整日志/收据快照、启动守卫和收据、实际 resume 审计、进程/GPU/锁、已有完整链保全复核及中断时 PNG hash 清单。
- 下载后本地复核全部6473文件的大小与hash通过，R优化日志连续性通过。
- 模型、resume 和完整 PNG 实体仍留共享盘；中断时9张未完成PNG实体保存在13:35中断归档。
- 自动跟进已更新为当前 NORMAL 实例和17520/round3阶段；正常推进保持安静，完成/失败/必要干预/实质结论再通知。

这是部署恢复与训练流程证据，不是新增记忆性能成绩。
