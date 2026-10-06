# 固定32历史/256步 Writer：内部 dev90 功能诊断

2026-10-06 第十一次跟进，新增图片或评分前登记。context-exposure-v1 已完成5376行，32历史在每历史32次draw后，原/新增各16均出现恢复和应用的匹配特异性。该结果只回答训练侧曝光问题。本阶段独立复核其全部来源后，固定最终256权重，检验是否延伸到已有内部dev90。

唯一新增评测端点 writer32-256：checkpoint SHA5aaffb9983f2c16b176790237109502fc02516dda8a3e5c0ee9ecedba5f036a8。不训练、不选检查点、不增加教师目标，不评分32历史/128步的dev端点。与已完成context-dev-v1的16历史/128步宽上下文Writer比较：两者每历史32draw，但32组总计算更多，不能将差异单独归因于覆盖范围。窄组、B730和Direct两个seed均作历史参照。

预先固定内部dev90，与本轮32训练ID无交集；此前被研究观察，属于探索性验证，不能称盲测。official180继续封存。沿用12问题、原Reader/温度1、同1080份旧teacher-prefix/logits、同主题循环donor、V0初写、28步/CFG1/两个原noise、1024 uint8 RGB PNG。Writer只接收已发生的最初exchange，不接收评测问题。缺旧cache时失败，不生成替代缓存。

新增180PNG、4320行（90×12×2noise×匹配/错配）；旧context-dev的23760行与900PNG只读复用，合并28080行/1080PNG。最终端点单列匹配、错配、灰图、文本；文本0为教师自一致性，不是准确率。以每个偏好平均问题/noise后的n=90配对bootstrap区间为单位。主要对比 context-diverse(16/128) KL−writer32-256 KL，以及256的错配KL−匹配KL；恢复和应用分别报告，单列中性偏移。两类上述区间均下界>0才支持内部集方向性迁移，仍不升级默认或宣称黑盒迁移。阴性则回到训练侧机制诊断，不加步或用dev搜索最优端点。

独立checkout context-exposure-dev-20261006，output仅context-coverage-20261006/context-exposure-dev-v1。执行前独立重算旧exposure与dev报告，核验训练连续性、冻结权重、源图片、原始行、共同prefix与完整分母；完成后同样全量复核。源源码和数据只读；专用新消费者不得改变旧消费者源码，保证原source SHA绑定。

阶段上限2 GPUh，所有生成/读回/失败计入原16总池；此前全结算6.632105855743 GPUh。最多2GPU，每次≤6h、共用deadline与未结算attempt阻止重复启动。优先恢复已停止的原H200x2，确认旧执行全部exit0、owner删除，实际GPU/内存/进程及其他预留后再启动。已读ARIS tracker，其冻结128共享Writer不包含本权重；不动其资源、追加预算和其他自动化。
