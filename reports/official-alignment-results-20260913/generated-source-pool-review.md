# 24张独立生成训练源图：完整证据

固定源码`90b41a2f1e7e4e0a8d41e709cedb69c4b32e3231`于1789373269.0241914启动，1789373395.6137743结束；四个实际worker78348–78351在原四H200上执行，未优化任何参数。全部24张PNG与120条原始读取保留，**120/120通过**；ambient、jazz、clear各8张，三组分别有8个不同PNG哈希。没有换种子或筛选成功子集。

完整归档210809795字节，SHA256 `b8b36757b36781020f1e309f0d5a402a94e47001204d15e37801ccebe6c49403`。归档超过100MiB，因此保存全部26个8MiB分块（末块为余数）和manifest；已在本地实际重建完整原归档。首次整文件SCP于300秒超时，留下47784960字节，原文件留在.cache；仅复用了与远端分块SHA完全相同的5个完整块，剩余全部逐块下载验证，未把部分下载当作完整证据。

本地已逐项核验24张PNG、24份实际PT、每份29个采样状态、全部120 raw及原五问法/原始token+立即EOS。实际source latent绑定、noise等张量的完整字节均保留。训练仍须在其实际GPU运行环境中将这些PNG重新经过官方VAE，并与保存的source latent逐位比较。

第一次Windows本地收集器在重建高斯噪声时拒绝通过。独立排查显示：本地为Windows/PyTorch2.11.0+cpu/AVX2；原生成环境为Linux/PyTorch2.7.0a0+ecf3bae40a.nv25.02/AVX512。24张噪声各有少量浮点元素不同，每张最大绝对差2.384185791015625e-7。版本、平台与CPU执行能力共同不同，未据此作单因素原因归属。

随后用原生成环境的独立脚本重新生成全部24个登记种子的完整高斯张量，与全部原PT逐位相等，封存独立重放JSON，SHA256 `7d919108dec63151117c28d6e1d57dbb223506ade07a47b2eca8d94d5ae34bf3`。本地收集器显式校验这个独立SHA及其模型/计划/全部case/种子/PT身份，再将每份实际noise与重放SHA精确比较。没有用数值容差替代原生噪声证明；Windows差异完整保留。生成和训练默认收集路径继续执行原本的精确本机RNG重放。

这只是下一轮的训练材料合格证明，不是新模型或功能验收分数。当前最优4f仍原注册1790/1800、已观察表达1750/1800、PNG3540/3600。训练实现已在独立源码`ef163b26e33f62c496ed0da8744ebb7bf1163873`准备，固定4832额外更新、完整baseline/功能回归，不将这24张训练图冒充holdout。

证据与重算入口：

- [完整原始分块清单](90b41a2-generated-source-pool-evidence.tgz.parts/manifest.json)，SHA256 `6c80c1003fa65c18ec0d5ead5dd9d4e87d137dca246f61c41d3f9bd29ef466a0`，全部分块均在同目录。
- [独立远端归档哈希](90b41a2-generated-source-pool-observed-artifacts.json)、[原始生成manifest](90b41a2-generated-source-pool-manifest.json)、[原生噪声独立重放](90b41a2-generated-source-pool-native-noise-replay.json)。
- [完整本地复核与Windows差异](90b41a2-generated-source-pool-local-verification.json)。
- `scripts/reporting/verify_generated_source_pool_local.py`可直接从本仓库全部分块重建、提取并复核；不需要失败的SCP缓存。独立原生重放脚本为`scripts/reporting/replay_generated_source_native_noise.py`，按固定远端原运行目录执行，输出已存在时拒绝覆盖。
- [下一轮完整训练登记](generated-source-training-preregistered.json)，186845字节、SHA256 `d60e941f2dfbf0e6cdee8695defaa2b96ffb0ed5ac3a7b2b4ec6a771e73f51ce`。此文件绑定ef163b2源码；不是新训练已经启动的证明。

报告端收集器7项测试通过，包括跨运行环境的独立参考路径仍拒绝篡改种子。下一轮训练新增4项及原回归30项通过。新`dl-source-aug-h200x4-20260914`已于16:12:11创建，16:35平台仍PENDING，调度事件报告节点资源不足。原有实例对象全部保留，未向另一个汇报任务追加工作。
