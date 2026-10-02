# 从封存Writer参数开始新的官方FM实验

固定2880步端点在完整开发集为880/900，4张jazz清除图仍生成旧状态。为了支持后续可审计的微调，训练入口新增`--initial-writer-package PATH --initial-writer-package-sha256 MANIFEST_SHA`，平台pilot同步传递并记录这两个参数。这里只增加明确的初始化途径，尚未证明加训可修复失败，也尚未启动新优化任务。

仅允许official Base、full-U-Net。包必须是此前完成端点的封存参数导出，显式manifest SHA、权重SHA、全部参数形状和FP32数值均需通过检查；Base内容、官方源码commit、Reader快照、原生步数及guidance须与新实验一致。目标bank可以是新的合法训练数据，但初始化不能读取父优化器或父RNG。

加载发生在新baseline、参数变化参照和AdamW初始化之前。所有U-Net参数恢复可训练，VAE、条件编码器和Reader仍冻结；FM公式、source仅作为条件、完整sigma范围及原生采样均不改变。新实验使用配置的种子和全新AdamW，身份文件记录父commit/result/checkpoint、父优化步数、包与权重SHA，以及“重新实测起点”的基线口径。新的2880步如果之后被配置，表示本次额外更新数，不能冒称从预训练模型只训练了2880步。

`--resume`只恢复这次新实验自己的checkpoint和RNG，继续绑定相同初始化包。不能把未训练baseline引用与已训练参数初始化混用；包在训练结束时再次核验。默认未提供初始化参数时仍使用原来的预训练/零LoRA起点。

五项新CPU测试覆盖全参数加载、RNG不被父状态替换、真正可反向更新且AdamW第一次step为1、Base/Reader/guidance不匹配时零参数改动、显式SHA及篡改拒绝。与原包和官方FM测试本次10 passed。这是入口正确性验证，不是新模型功能验收。下一轮具体预算、种子和数据应在当前独立诊断结果审阅后固定，并以实际加载参数的baseline及完整端点评测验收。

另已运行full-U-Net控制、latent-bank训练与checkpoint恢复相关回归，39 passed；本轮相关验证合计49 passed。正在运行的0f40767诊断checkout和b9eec7c4套件未修改。

后续实验已在`official-transition-warm-start-plan-20260913.json`固定：训练源码d9a1a117cd497ad72d5bcc1630d657a43cebd611，初始父checkpoint b4251975...、result3d747a7c...，同一45条件bank，seed20260914，额外2880次更新/11520draw/每组256次，其余FM和优化参数不变。优化器重新初始化与噪声变化已明确记录，因此这不是单因素因果对照。四个新开发噪声的baseline与final均实测900matched，加controls各1350raw。不得依据中途分数改预算或挑checkpoint。

同时预注册72单图和16组六步链：8个新事件表达与当前训练bank及前轮确认表达不相同，确认噪声与本轮/父训练和开发噪声分离。此计划由`transition_warm_start_plan.py`实际生成并验证数量及相异性；新的确认尚未运行。

`scripts/inspire/run_transition_warm_start.py --deadline-unix ...`是固定实验派发入口，`--dry-run`在资产就绪后输出实际命令。它校验包来自指定失败端点、真实独立六写30读parity已通过、bank和训练checkout不变，然后将原始计划及解析后的参数包SHA复制到新run。必须使用实际lease余量至少200min的新空闲H200，覆盖baseline、追加训练和final。新run固定为`d9a1a11-transition-warm2880-seed20260914`，重复输出拒绝。**尚未派发；当前RGB链/包导出/独立重放仍在运行。**
