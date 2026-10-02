# 原生Base条件一致性四卡训练对照

固定bb34092模型的全部302个首步对照已完成，在新H200驱动下逐位复现全部旧原生首步。独立CPU复核全部速度、原始高斯噪声、目标与Euler更新通过，本地便携证据也已逐项对应原开发记录。10张开发失败图的平均首步速度MSE：原生包装0.164105，raw训练条件0.000681887，sigma0.999整数999训练样本0.000684905。全部302图的对应均值为0.00898460、0.000962292、0.000969427。该结果不能证明raw条件的完整28步/Reader验证一定成功，但支持优先检查并修复训练与推理的条件分布差异，而非先改FM时间范围或只加训练预算。

官方固定a6e20c8的LoRA例子使用原始编辑指令（默认transfer the image into Snoopy style）。原生Base编辑推理在指令外加diptych包装，并用三个prompt一起编码。当前实验原样保留了这两种接口，但当前模型对它们的差异表现敏感。下一轮显式把训练条件改成原生Base编码器三个分支中的条件行：`[empty, empty, diptych(event)]`一起编码后取第三行及其完整mask。保留共同padding与多模态position语义。Writer依然仅接收源图和事件，不接收目标、问题、答案或教师id。

固定初始化d1536d原warm包、151组bank c27cd65、seed20260915、逻辑条件均衡抽样、全部19328draw、4832更新、global4、fresh AdamW5e-5和全U-Net。每个条件/教师/噪声/sigma与bb34092相同。官方FM公式、完整uniform sigma、整数训练timestep、纯高斯起点、原生Base28步CFG1推理、数据与评分均保持。**仅训练条件编码改变**；明确不同于官方LoRA例子的raw-event训练文字，不把它包装为未修改的官方复现。

先重测全部302张初始native图和3020条原始记录，再与bb34092初始基线逐位比较全部latent/RGB/29状态与完整raw。runtime只允许训练条件hash及train_prompt元数据不同，模型、源图、native推理、scheduler、termination等其他字段必须相同。新条件下仍要通过全模型首4draw串行/四卡梯度一致性及四rank初态校验。任一失败则保留证据、停止优化，不放宽门控。

四卡实例使用dl-official-exp-h200x4-20260914；旧单卡当前验证保留。固定终点后评估全部1510开发、360单次、480连续链、960历史前缀严格答案/EOS，并执行独立导出CLI。全部表达/noise已被观察，属于配对诊断，不声称是fresh holdout。只完成代码、FM误差改善或旧集合通过都不能证明未见实体、多事实记忆可用。

首步完整证据SHA **24cdf7bb88dc23298167a165d3c011cde066a1c866bfdfe0944ee2d9149ec611**，CPU summary SHA **4c9fc17f97b25788f8fa3573be4752d5c2075e71be8982ce427911515e2119d5**，probe complete **a469e8a20b22bc5913ea92f8477d9ac1081390b0cd25ac7db744a083db7c621c**。结果和本地核验文件已保留在official-alignment-results-20260913。新训练源码commit及实际计划SHA在启动时锁定。
