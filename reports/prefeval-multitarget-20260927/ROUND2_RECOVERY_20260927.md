# 第二轮评测中断与恢复准备

2026-09-27 21:34–21:39 CST核验。平台21:22:09自动停止原实例，事件明确写为CPU/GPU/MEM利用率未满足管理员自动回收规则；不是根据本次证据推测的抢占，也未发现应用报错。停止前阶段为S1/V0四分片MCQ读取。

原实例prefeval-k1-h200x4-high-20260924已通过start重新提交，21:39仍为PENDING/NORMAL、4H200。没有删除用户实例，没有重复申请GPU，也未干预B730或C/R。资源重新启动不等于实验已经恢复。

## 已完成且保留的工作

- S1/C8各2048步FM已完成，两份complete.json齐备。S1最终权重SHA256为d9b796e99a73649026f8d9786e246a6af1a06e56182eab6ee4b22654b5f12f9c；C8为efd7b4a22a40508e9ae5a9d7ca72305f5a1c8b3130b3e22c5e2b72aa7bb36978。恢复脚本启动前将重新计算实际权重SHA256。
- S1/V0的512张图全部已生成。四个读回文件分别683、681、679、677行，共2720行；全部行可解析且各分片键唯一。完整V0应有3456行，因此还不是该版本完成成绩，不发布偏好顺序子集的性能排名。
- S1/V1及C8/V0/V1尚未开始；两组完整results/complete不存在。C8教师511/512合格及64偏好全覆盖结果保持不变。
- 冻结实验代码仍为224cc77d790cf3967b5a56ce2e77c364959435a2，工作区干净。恢复过程中不改教师、训练或评测实现。

## 独立恢复目录

共享输出为runs/prefeval-multitarget-20260927/round2/recovery-20260927-2135。恢复脚本resume_eval.py来自本分支bb2d298，SHA256为7386fc6dfe8e3390ad972298b57990342a2d3aac20bc0e051e3c2a1ce35b9a8e；本地源为scripts/inspire/resume_prefeval_multitarget_round2_eval.py。

脚本仅运行原冻结Writer rollout和原冻结MCQ evaluator，不执行教师或FM训练。检查两份权重身份、原协议/目标池及全部512张原PNG哈希；将2720条原读回逐字节复制到新目录，后续按原键集合跳过已完成项并追加缺失项。S1/V0图片只读复用原目录；其他三组版本的图片写入恢复目录。保持原64条、paired mt8-eval种子、四分片、全64同主题donor、官方解析和分母。

启动后持有根pipeline.lock、原round2/pilot64/controller.lock及新controller.lock，锁传给GPU子进程；核验GPU占用后才启动。控制器、阶段日志、进程收据、failure-PID.json、results.json及complete.json都写新目录。启动前需现场确认新主机、GPU、全部旧任务进程已退出且锁空闲；原实例若仍排队，不重复start。

资源就绪后用共享环境envs/vlm-r3-ngc2502/bin/python运行该resume_eval.py，必须以后台独立会话启动，将stdout/stderr写recovery.log并记录launch.json。启动后实际核验controller、GPU子进程及读回增长。当前脚本已部署并通过本地AST检查，但GPU恢复尚未执行。

## 原始证据

interruption-evidence.tar.gz归档中断前2036个JSON/JSONL/阶段日志，保留所有教师回执、目标池、训练日志、S1/V0生成收据与部分原始读回；权重和PNG实体留原共享目录。压缩包6651280字节，SHA256为67be64e98b902e8a3ec88797006650e8df7e68da4661349316823a69529c5123。interruption.json记录停止原因、旧控制器、完成权重身份、逐分片记录数及归档摘要。

恢复完成后从新目录的全部readback-0..3汇总S1/C8，联合原首轮结果分析；不得只读本次新增计数或shard0。排队与技术恢复不能当作记忆收益。自动跟进在恢复等待期间缩短至10分钟；实际恢复后返回45分钟，状态不变保持安静。
