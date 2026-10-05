# 共享 Writer 只读评测部署

2026-10-06 03:56 CST。第二次小时触发后，已在vlm-dreamlite-full-h200x2-20260720部署新的功能诊断，不重复ARIS的训练或MCQ/保持评测。协议先冻结于WRITER_READOUT_PLAN.md。

执行树481335eb8df7efa05e41ff9cc337d2ef2670516e与开发535ae7a一致，独立checkout context-writer-readout-20261006-v2；Windows和Linux各102项检查通过，运行源码clean。首个准备核对发现ARIS manifest_sha256是规范化JSON对象摘要而非文件摘要，修正并增加兼容回归检查，尚未产生GPU worker。第一次控制器又因继承的稀疏检出未物化已跟踪协议文档而在启动worker前退出；随后从相同HEAD恢复该文档，代码树不变，确认旧PID退出再启动4012735。两次准备故障均无GPU计算，日志与初始checkout保留。

资源检查：同一已使用的hostname，两张H200各约143GB空闲，无其他计算进程或活跃owner；本任务资源不属于ARIS预留的其他四实例。父模型已有212张V0 PNG的哈希、RGB尺寸、完整收据、原始历史及相同噪声逐项检查通过；27个基础模型文件SHA与上游manifest相等。父模型图片只读复用，无重算。

Direct seed20261005的128步完成收据、最终checkpoint a78c6f808d7fb0ecb87bcc591185cf0168f352a21d580fca9f8d175ed1e27f8c、manifest与128行连续优化日志核对通过；又独立核对全部512张训练输出PNG及14,336个有限非零反传梯度。另seed仍未完成，因此当前先做已完成端点的图片生成；不能据此给出两seed整体结果。

03:54:57实测controller4012735、worker4013220均存活，cmdline与冻结命令一致，GPU0实际占用约14GB；首张1024×1024 RGB PNG及complete哈希核实通过。第一GPU attempt从03:54:15起记费，03:54:57暂计0.011621GPUh，不能当最终阶段费用；此前已结算1.349614GPUh。新增阶段4GPUh上限计入原16GPUh，总分母仍636张来源PNG、17,808读回行。

后续由同一控制器继续：首seed rollout完成后重查第二seed完整端点；缺失则退出到waiting_upstream，下一次小时跟进再接续已完成收据。只有三端点的PNG全部通过验证，才开始两分片读回与偏好级配对报告。当前没有新的科学效果分数，主线默认不变。

03:59:44 CST复查：首seed已有28/212张完成PNG，最新4张哈希/尺寸通过，实际worker仍存活、checkout clean；本阶段暂计0.091366 GPUh，总暂计1.440980 GPUh。Reader权重、配置、目标构造和目标函数的SHA与上游训练完全相符。原03:54:57的首图记录作为历史验证保留。

2026-10-06 04:42:23 CST小时跟进：首seed的212张评测PNG全部完成，原worker exit0；控制器已自动接续第二个128步完成端点，当前57/212张，GPU1与实际cmdline正常。父模型+首seed全424张来源PNG复核通过；第二seed完整checkpoint/manifest/128步优化、512训练PNG与14,336个梯度复核通过。两seed冻结模型、数据、任务和优化器字段配对一致。全部新图齐全后才计算完整17,808行，当前没有读回评分。阶段暂计0.801421 GPUh，全任务暂计2.151035 GPUh；不含平台空闲与点券。
