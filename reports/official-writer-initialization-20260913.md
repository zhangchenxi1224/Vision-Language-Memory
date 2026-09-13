# 从封存Writer参数开始新的官方FM实验

固定2880步端点在完整开发集为880/900，4张jazz清除图仍生成旧状态。为了支持后续可审计的微调，训练入口新增`--initial-writer-package PATH --initial-writer-package-sha256 MANIFEST_SHA`，平台pilot同步传递并记录这两个参数。这里只增加明确的初始化途径，尚未证明加训可修复失败，也尚未启动新优化任务。

仅允许official Base、full-U-Net。包必须是此前完成端点的封存参数导出，显式manifest SHA、权重SHA、全部参数形状和FP32数值均需通过检查；Base内容、官方源码commit、Reader快照、原生步数及guidance须与新实验一致。目标bank可以是新的合法训练数据，但初始化不能读取父优化器或父RNG。

加载发生在新baseline、参数变化参照和AdamW初始化之前。所有U-Net参数恢复可训练，VAE、条件编码器和Reader仍冻结；FM公式、source仅作为条件、完整sigma范围及原生采样均不改变。新实验使用配置的种子和全新AdamW，身份文件记录父commit/result/checkpoint、父优化步数、包与权重SHA，以及“重新实测起点”的基线口径。新的2880步如果之后被配置，表示本次额外更新数，不能冒称从预训练模型只训练了2880步。

`--resume`只恢复这次新实验自己的checkpoint和RNG，继续绑定相同初始化包。不能把未训练baseline引用与已训练参数初始化混用；包在训练结束时再次核验。默认未提供初始化参数时仍使用原来的预训练/零LoRA起点。

五项新CPU测试覆盖全参数加载、RNG不被父状态替换、真正可反向更新且AdamW第一次step为1、Base/Reader/guidance不匹配时零参数改动、显式SHA及篡改拒绝。与原包和官方FM测试本次10 passed。这是入口正确性验证，不是新模型功能验收。下一轮具体预算、种子和数据应在当前独立诊断结果审阅后固定，并以实际加载参数的baseline及完整端点评测验收。

另已运行full-U-Net控制、latent-bank训练与checkpoint恢复相关回归，39 passed；本轮相关验证合计49 passed。正在运行的0f40767诊断checkout和b9eec7c4套件未修改。
