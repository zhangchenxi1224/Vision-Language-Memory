# 训练历史上的上下文覆盖传递：冻结协议

2026-10-06 07:49 CST，于任何新训练/生成/评测之前登记。Direct与PM→FM的dev诊断均未建立匹配特异性；下一步缩回训练侧，检验已证有效的宽上下文教师能否被同一个共享Writer训练过程学到。已检查另一主线tracker；它的512draw、8步PM→FM和free/保持评测与本实验不同，不重复。

唯一主要因素：软监督教师的上下文覆盖（原MCQ问题 vs diverse-v1），两组均复用现成的16历史、288步最终latent/PNG。不用验证所选端点，不重新优化教师、不新增目标答案。原bank来自prompt-matching-20261005/pilot-B/prompt_matching/teachers，宽bank来自本轮pilot/prompt_matching/teachers；两者只读。

共同起点为已完成B730 checkpoint-final，SHA088c003d24cb2982571d763dd218c57f72ce80853615a09a97d8b760ae880c5b。两组用同16历史顺序、同seed20260924、同4微批/步、同128次AdamW更新、lr5e-5/betas(.9,.999)/eps1e-8/weight_decay1e-4/grad clip1。复用prefeval_k1_writer.py官方FM bridge与原随机sigma/noise逻辑，Writer输入仅灰图初始状态和最初历史exchange；不看评测问题。两组共1024训练draw，不混入dev/official历史。共同Reader/VAE、数据/模型和代码身份在执行前冻结。

这是一个训练侧传递/可拟合性诊断，仅1个训练seed。固定训练128步，保留最终端点，不按loss早停或用读回选权重。有限预算下失败不能证明模型表达能力不足。更好的训练侧KL也不能证明新历史泛化，必须有后续独立协议才开展dev或推广。

两最终Writer各生成16历史×2噪声=32张真实V0 PNG，合计64；原28步/CFG1/uint8 RGB1024、eval噪声和相同历史协议。全量验证PNG与checkpoint/历史/noise绑定。仍用12个已有冻结问题，每组匹配/同主题循环错配控制；teacher cache直接复用writer-readout-v1内192个pilot目标，禁止缺失时生成。

新增16×12×2Writer×2噪声×2图片控制=1536行。原Writer诊断的全部pilot2688行只读复用（B730、两Direct、灰图、文本），合并4224行。主要对比diverse Writer减narrow Writer的KL方向与匹配减错配；各问题族单列，先问题与噪声在偏好内平均，bootstrap独立n=16。正收益定义为narrow KL−diverse KL、错配KL−匹配KL。分别列两组与B730的差值。教师层已完成的宽/窄结果作为历史机制背景，不跨不同prefix混算比值。

整个新增阶段上限1 GPUh，含训练、生成、读回和失败，计入原16 GPUh总池；截至此前累计3.657263426648。最多2GPU，单次最多6小时。先验证32教师latent/PNG/完整绑定和共同父权重；代码/环境检查通过、现场无其他worker或owner/预留后启动。训练日志要求128连续步、有限非零梯度及完整512draw/组，核对两组draw/pid/sigma一致。最终checkpoint及manifest哈希固定，推理只读。

全部代码/输出只在context-coverage-20261006独立checkout及context-fit-v1。源bank、旧读回、其他任务只读。不能根据现成训练loss提前宣布成功；完成全部1536行、192目标、64新PNG及96旧pilotPNG核验和报告重算后，才能报告训练侧传递结论。若到预算边界保留partial、结算退出，不自动扩容、加步或换seed。恢复前先证明原worker停止并保留冻结命令/随机状态，禁止重复控制器与空转防回收。
