# 生成源图条件增强：固定预算继续训练

16:38更新：训练源码ef163b26e33f62c496ed0da8744ebb7bf1163873已在独立干净目录部署；完整源图证据已下载重建并复核。固定完整登记SHA256 d60e941f2dfbf0e6cdee8695defaa2b96ffb0ed5ac3a7b2b4ec6a771e73f51ce。新四卡仍PENDING，尚未优化。见[源图完整结果](official-alignment-results-20260913/generated-source-pool-review.md)。以下先前进度按原记录保留。

本轮从已完成的4f参数继续训练。4f保留原注册1790/1800、已观察表达1750/1800及PNG3540/3600的真实失败，不改评分或已报告结果。来源变化是待检验假设：固定源图的训练可能不足以覆盖实际生成图作为下一次输入的情形。

训练材料已在固定源码90b41a2上实际生成：24张新种子PNG、120条原始回答，远端完整检查120/120通过，manifest SHA256 `9431b9a11eac1c4b956d8178447223f92c59058ef8a851d75b575abb41b60858`。原生成计划SHA256 `7603bd62e10241e0542e6b5b30168a5c2ed6d22159e7113d7754c8860b95c2f3`。整个源图池被保留，未丢弃失败、换种子或选择最佳图片。完整210809795字节证据正在分块传回本地；首次整文件SCP在300秒后超时，保留其部分字节，只有与远端分块SHA匹配的5块被复用。

固定训练合同：

- 初始化为4f checkpoint `7294684170578dfc617b4fafcea97e6480c08642ca4f1e5967ff8966aa103182`，package SHA `267152e0c1c059f1f1c3a3e041839a68ba0b8f96a6ca4d782a7ef90ba5ccf05f`。先重新测完整302张canonical-source图、全部29步轨迹和3020 raw，与4f训练终点逐位比较，通过后才能优化。
- fresh AdamW，lr1e-5、weight decay1e-4、global batch4、clip1，额外4832更新、19328 draw、seed20260915、全部31逻辑条件。历史16条件的9表达增强完整保留，所有teacher/noise/sigma抽样保持原流。
- 只在108个已有音乐源条件上增加source变化：每个条件有原图及该源状态的8张生成PNG，共9个选择，按每条件出现次数均衡循环。原43个gray/历史条件保持原source。
- 所选PNG同时决定官方VAE source latent及native Base事件条件编码。所有24张图先完整重算raw、张量、种子与轨迹；训练进程再次实际VAE编码并要求与生成端latent逐位相同。四个训练rank必须同意全部source/embedding/mask哈希；逐draw记录真实变体选择，恢复时拒绝绑定变化。
- 目标加噪仍为`(1-sigma)*target + sigma*noise`，速度目标仍为`noise-target`，完整0–1训练时间范围，整数训练timestep；source仅进入条件侧。开发及完整功能评估仍使用原生28步、CFG1、纯Gaussian初态与真实PNG写回。
- 固定4832步终点，全1510开发、两套完整1800功能回归、真实CLI和全部PNG读取均须验收。原有两套表达是已观察回归集；本轮训练材料检查不冒充新holdout。额外训练和source增强同时变化，不能作单因素消融归因。

实现为显式`--generated-source-continuation`，需同时开启原历史表达增强。未选择时保持已有路径。生成池以`--generated-source-pool`和完整SHA传入训练，默认不会启用。固定计划生成器为`scripts/experiments/generated_source_training_protocol.py`，实际启动时绑定干净完整源码commit。

本地检查：新增实际19328抽样覆盖、FM与source/condition连接、完整训练计划等4项通过；原FM、Base条件及初始化复现回归30项通过。新增完整计划测试最初误用不存在的`steps`字段，按既有schema改为`optimizer_steps`后通过（29.11秒）；没有更改训练预算来适应测试。源图生成/收集器另有6项通过。

资源：新建`dl-source-aug-h200x4-20260914`，4H200/80CPU/900GiB、shm128、NGC25.02 CUDA12.8镜像、8小时自动停止，16:12:11创建；目前PENDING。旧`dl-clear-retain-h200x4-20260914`保留，原用户r3实例也仍PENDING。当前尚未在新实例上开始优化，GPU实际条件编码与四卡梯度/完整baseline验收仍须执行。
