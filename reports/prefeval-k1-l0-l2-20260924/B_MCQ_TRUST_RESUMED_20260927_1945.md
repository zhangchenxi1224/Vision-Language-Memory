# B/MCQ C/R：trust 实际恢复与终端超时诊断

2026-09-27 19:37:10 CST，已在用户指定的 `vlm-r11-trust-h200x4-20260907-r4` 恢复 C/R。19:25 报告是当时连接受阻的历史记录，本记录取代其“尚未恢复”状态。

## 连接问题与纠正

资源实例可用。JupyterLab 入口和终端列表均返回 HTTP 200，页面 baseUrl 与 CLI 使用路径一致。新建终端 POST 响应超时，但再次列举时出现本次请求新建的终端 5；接入该终端后 hostname、GPU/进程/锁核查和守卫启动均成功。因此，先前不能执行命令的直接原因是新建终端响应超时，不能据此认定资源不可用。后台为何未正常返回响应尚未确定。

本地 `evidence/trust-recovery-20260927/reuse-diagnostic-terminal.py` 仅在当前进程复用本次诊断创建的终端 5，使用 CLI 原执行路径；未修改安装的 CLI，也未改冻结实验代码。同一终端须串行执行。没有停止、重启或重建 trust 实例。

## 实际恢复

- 主机：`vlm-r11-trust-h200x4-20260907-r4--c94e44dc00e8-erlc4tfr57`。平台 LOW priority 1，4 H200；本次使用符合用户最新明确授权。
- 启动前：4 卡无计算进程，无 PrefEval 进程，C refresh/launcher 与 R refresh 锁全部 FREE，trust/normal 两个 launch.json 均不存在。
- `launch-trust.py` 仅执行一次，19:37:10 写入启动收据。C 父 103680、shell 103738；R 父 103681、round2 生成 104218/104221。完整命令、cwd、CUDA、继承锁已记录；C 使用 GPU1，R 使用 GPU2/3。
- 原协调器/R 为冻结 55596810cee2062022b894ef93feb1ab4cd6b07b，C 内部仍为 52d98d8。
- 守卫核验 B/C/R 权重、C 银行、R round1 银行。R 实际 resume step=11680、cursor=46720、1075 个优化器参数状态均 11680，Python/NumPy/torch CPU/CUDA RNG 齐全，refresh_round=1。
- R resume SHA256：`1c0886430cb5daaa225e2784a8eeefe30cef8d019ca94078acf3f48c27d6e272`。R11680 权重 SHA256：`8300e2eb389445a7819e8540f35b6b5d0d768e3db52b19f9d93a31592c4da1d3`。

19:41 现场：R 两分区分别从 32 条完整链推进到 33 条，各 336 PNG（含在途）；GPU2/3 实际利用率 100%。round2 尚未冻结，第11681步尚未开始。这是执行恢复证据，不是记忆成绩。

19:43补充现场（另存 `runtime-final.json`）：C已转入pilot生成，PID130460，GPU1利用率78%；R各34完整链、PNG347/348，GPU2/3利用率100%。GPU0空闲。C/R恢复已实际推进。

## 跳过已完成内容

19:39 逐文件核验原有完整链的实际 PNG hash 与修改时间：C dev 180 链/1980 PNG、C pilot 12 链/132 PNG、R 两分区各32链/320 PNG全部通过，均早于此次启动。三条中断的不完整链已经在16:55证据归档，原代码从相同种子完成这些链。

C 不重训23360步；原 shell 重新经过 dev 命令入口，由原完成键及 hash 校验跳过已完成生成与读取。dev 原始读取仍9450行，位置0/10各900行，文件mtime仍早于恢复，未新增或改写原始读取。19:41已经过dev主读取并在核查位置10，随后自动继续pilot及train。进程名包含dev不等于重复推理。

训练预算仍C/R各23360，总46720。无新增探针、方法或资源请求；A、自然回答与付费Judge继续暂停。

## 资源监控与归档

19:39 两个普通候选 `b-cr4-normal-0927`、`prefeval-b-cr-h200x4-20260926` 均 PENDING/NORMAL(priority4)。继续每10分钟检查；普通候选RUNNING且实际空闲后迁移，或当前LOW中断后恢复。迁移前归档现场，确认旧父子进程退出且锁释放，禁止两份C/R并行。不干预其他实验，不使用禁止实例。

`evidence/trust-recovery-20260927/raw.tar.gz` 包含完整启动收据、实际resume摘要、启动前后现场、跳过审计、原始阶段日志、完整优化日志及未提交历史尾部、dev原始读取和当前链收据/manifest。模型和resume留共享盘，GitHub只存摘要和日志。

- 快照时间：19:41:45 CST；572文件。
- 字节数：5601035。
- SHA256：`4ffecd421e8435281dcf8b0f14364093ae54169cd978256134e36bab60ed902c`。
- 已下载并核验，本地与远端字节数及SHA256一致。

下一轮关注C pilot实际接续、R round2完整银行及普通候选。正常运行、排队或计数增长保持安静；仅重要阶段、失败、必要干预或完整结果通知。
