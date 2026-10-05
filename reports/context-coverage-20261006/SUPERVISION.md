# 每小时执行入口

本任务当前作用域：上下文覆盖 × history_hard/prompt_matching 的16历史教师机制对照，最终端点与验证选择分开报告。科学协议见 ../../docs/CONTEXT_COVERAGE.md。正在运行的 ARIS M1/M2 和 PM→FM 不由本任务修改。

自动任务 `dreamlite-2`，名称“DreamLite 上下文鲁棒性实验跟进”，本地任务 `01a0fc2b-d3a4-75b0-a785-0e65fb49aa1d`，每1小时。已实际创建并读回 ACTIVE、1小时规则及目标任务一致；尚未到第一次实际触发，不将配置成功当作端到端触发证据。

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

若机制证据支持，先检查另一主线新近结果与父checkpoint，接入同一问题采样接口做新配置的共享Writer对照，禁止修改运行中的协议。每轮只改变一个主因素、先冻结完整评测和有限预算，报告token/时间代价与不确定性。原默认方案需要完整新历史、真实PNG、保持与自由回答证据才替换。若不支持，检查历史敏感度和各任务KL后决定单因素修正，不以反复挑dev成绩推进。

只在结果变化、阶段完成、故障或需用户处理时通知。没有变化保持安静。记录每次实际触发与动作，使定时配置和真正执行可以区分。

执行提交 `15ff2a93b5ec6222c37ec359b20cfadcabbba471` 与开发提交 `3995d2c9c111fdb7fac3002e658c4cc77e282e7a` 的完整 Git tree 相同；为离线部署把同一树锚定在已存在的a484执行基线。增量bundle关闭外部delta依赖后验证与fetch成功，远端clean，Linux100项检查通过。不得用开发分支后续报告提交覆盖执行checkout。
