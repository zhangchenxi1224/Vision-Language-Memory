# 同主题历史覆盖16→32：冻结训练侧协议

2026-10-06 09:52 CST，在新增目标生成、训练和评分之前冻结。已有context-fit-v1在16历史上有明显功能收益；context-dev-v1的完整dev90结果为阴性，故回到训练侧检验学习覆盖范围，不继续用dev挑选模型。

唯一主要因素是训练历史数量：相同8主题，每主题2→4条，共16→32条。原16沿用context_coverage_ids.json，新增16是这8主题在既有pilot64中的其余全部历史，按ID排序；配置context_population32_ids.json、context_population_added16_ids.json固定，不能按结果更换。全部在官方训练侧且与dev/official分离，未增加主题。B730父本来见过train730，所以新增16只称未用于本轮16历史微调，不称完全未训练样本。

只补新增16条的diverse-v1全词表软监督教师，仍冻结Reader/VAE、B格式、初始灰图latent、Q8 STE、288次Adam更新/.05学习率、同12步4恢复/4MCQ/2开放/2中性循环、温度1和512token生成上限。沿用72/144/216/288快照但不进行选择，最终288目标进入Writer。旧16 final目标只读复用，不重跑、不以新身份改写；新16分两GPU各8条，单条教师没有跨历史共享参数。

Writer16直接复用context-fit-v1/context-diverse/train/checkpoint-final.pt，SHAb2527682b9560ea29c4c9bbaa5f67b361a3fd8057de3026d15c81bf27ad6ecea。Writer32重新从共同B730 SHA088c003d24cb2982571d763dd218c57f72ce80853615a09a97d8b760ae880c5b开始，相同nativeFM、seed20260924、AdamW5e-5、4微批/更新、128更新/512draw、sigma/noise种子和灰图初写条件。因总体32条，每条曝光从32draw降到16draw，这是固定更新预算下扩大样本覆盖的预期权衡；不宣称教师制作总计算相等。固定final，不按loss选端点，不继续16权重优化。

先进行一个上限1.5GPUh的目标制作+Writer训练阶段，包含失败；完成32目标及完整128更新核验后控制器退出，标记ready_for_frozen_readout，而非实验完成。这个阶段不得自动启动尚未核验的评分。随后只在本轮总上限3GPUh的剩余额度内实施以下已冻结读回；总预算不是新授权池，全计入原16GPUh（此前已结算5.337061309947，整个新轮最坏8.337061309947）。最多同时2GPU、单次≤6h，按attempt计一次，终止宽限计入；预算/时限跨子阶段共用，不按重启重置。

读回固定原16与新增16两个分层，每层8主题×2条，均使用已有12问题。每个Writer每历史2个原eval噪声、28步CFG1、RGB1024真实uint8 PNG。原16的Writer16图片32张只读复用；新增16的Writer16图片32张和全部32的Writer32图片64张新生成，共新增96/总128PNG。Writer输入仍只有最初历史exchange和灰图，不含评测问题。

每个分层内按同主题循环错配，不能跨原/新增分层换donor。每偏好12问题×(两Writer×两noise×匹配/错配+灰图+文本)=120行；32偏好共3840行。原16的Writer16 memory/mismatch768行从已完成fit精确映射endpoint后复用，灰图/文本384行及192教师分布从writer-readout-v1只读复用，计1152旧行。新增2688行；新增16的192个问题教师分布只在本轮独立cache制作，保持相同Reader/源码/生成契约并在两Writer间共享，禁止使用教师回答作正确率真值。

主要分析分别为原16保留能力与新增16覆盖收益，不混成新历史泛化。每层按偏好内平均问题/noise，报告n16配对bootstrap：Writer16 KL−Writer32 KL，以及各自错配KL−匹配KL；恢复/应用/中性、灰图/文本分别呈现。对新增16训练侧收益不能证明dev泛化。若新增16无法被覆盖或原16收益明显下降，下一轮才冻结优化预算等单因素诊断；不根据局部结果扩容/加步/重新挑seed，不触碰official180，不修改默认。

启动前检查ARIS tracker和实际owner/PID/GPU；其目前free预算申请待回复，不能借用/改变其资源或费用。本任务使用原专属H200x2，dev旧worker已退出。独立checkout context-population-20261006、output run/context-population32-v1。新bank只在本output建立指向已核验旧目标/新目标的只读用途目录链接，任何旧源文件不写入。新旧目标须逐一验证receipt/latent/PNG和问题生成绑定；训练日志必须128连续有限非零梯度、512draw覆盖32ID各16次，并核对固定sigma序列。

恢复前先证明旧worker退出、归档owner和结算；未完成native训练日志在未审计resume点与尾部前拒绝自动重新开始。只做必要计算，不空转。全部科学结果仍须完整读回后才发布。
