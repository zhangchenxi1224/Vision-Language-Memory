# 角色排版增强的同预算对照

2026-10-06 冻结于任何本轮训练/图片/评分之前。前置gate为context-layout-v1完整6912行独立核验：改变角色标签格式使两分层恢复/应用的匹配KL及特异性同时退化。只读复用旧数据、目标、权重与评分；不重跑旧实验。

唯一训练主因素：Writer输入历史的角色排版分布。两臂均从同一个已完成32历史/256步resume续接Adam、权重、RNG与cursor，不重训256步。各增加128步到固定final384，batch4，各新增512draw，即每历史16次。控制臂全用canonical；增强臂按全局cycle奇偶均衡canonical/markdown，每历史各8次。canonical=`user: CONTENT\nassistant: CONTENT`，markdown=`### user\nCONTENT\n\n### assistant\nCONTENT`。内容、角色、消息顺序保持逐字节不变，不替换assistant回答，不把问题传给Writer。

32历史、原32张288步概率教师latent、draw顺序、sigma/noise seed、父Adam状态、native FM loss/clip/lr、Reader/VAE/DreamLite权重、1024图、28步/CFG1及两noise全部固定。为对齐准备开销，两臂都缓存两种排版条件。源final SHA5aaffb9983f2c16b176790237109502fc02516dda8a3e5c0ee9ecedba5f036a8；源resume SHA8f5bd84c06c91611a3a38aee7967b05ceaba136e8f331dfca7b74c5f1636e5f3。各臂只修改自己的新manifest，旧native代码/输出不变。

先进行真实GPU实现校准：用旧native trainer从同resume执行257–258两步，和新控制臂相同两步逐参数/Adam/RNG/日志比较，必须精确相同。增强臂的这两步仍处于canonical cycle32，亦必须相同。两臂校准步属于正式128新增步并从258继续，native参考两步单独计费。校准失败即停止，不改成近似接受。完整384步后检查每历史曝光、布局频次、源256日志前缀、有限非零梯度、checkpoint/manifest/optimizer身份；只评固定final384，不保存用于选优的候选端点。

评测在本轮训练前冻结，但可由单独消费者在两权重均完整后部署。只用原/新增各16训练历史，分别n16偏好级配对；不新评dev90/official180。每个臂评三格式：canonical、训练所见markdown，以及训练未见XML角色块`<user>\nCONTENT\n</user>\n<assistant>\nCONTENT\n</assistant>`。XML条件不得进入训练缓存。两臂×三格式×32历史×两noise=384新PNG；12旧问题×两图控制得到9216新行。复用6912旧行，总16128；384教师分布只读，同主题同分层donor、灰图/文本复用，缺缓存即失败。

每偏好先平均四问题×两noise，再bootstrap95%区间；恢复/应用/中性和原/新增两个分层全部列出。主要新证据为XML上控制−增强匹配KL与增强−控制特异性，同时报告增强自身错配−匹配；canonical与markdown分别报告收益/代价，不以挑格式或合并分层制造阳性。训练所见markdown恢复只证明学会该格式；XML正向才支持本次未训练格式的迁移，仍不证明新历史泛化。若canonical明显退化则保留代价，不晋级默认；不根据训练loss或局部显著性改变评测。

独立context-layout-aug-20261006 checkout、context-layout-aug-v1输出；训练阶段完成到ready_for_frozen_readout后退出。整个迭代上限3GPUh，训练/真实校准/失败阶段上限0.75，预留读回2.25；全部归入原16池（当前7.655850706829），不因拆消费者刷新预算。最多2GPU，每次≤6小时，共用保守deadline、唯一owner和未结算attempt守卫。先证旧worker退出及无资源/预留冲突。恢复只读核验checkpoint和日志尾后继续，不重复启动。原方案、未训练数据、其他任务与自动化均不改。
