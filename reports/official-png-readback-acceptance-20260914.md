# 完整 PNG 部署读取验收

在观察 b9 历史表达训练终点前固定此检查。原单次写入及历史前缀验证直接读取 FP32 解码图；实际 CLI 读取 uint8 RGB 图。保存 PNG 的舍入可能改变 Reader 输出，故两者必须分开记录。

对象为 b9f90e956eea7bda15f638c8877919941ce4fec5 的同一最终 checkpoint。等待 7b82309 完整旧案例验证和 56b56a8 完整新表达验证结束、实际四卡空闲后，以新的独立源码 checkout 执行。原训练和两个验证 checkout 均不修改。

两套各 1990 条，包括单写 390、真实 RGB 链 480、历史前缀两路各 560；总计 3980 条、3600 条 matched、796 张输入图。全量读取，与原输出对错无关。matched 直接打开原先封存的 PNG；全部 76 张 blank/donor 图从原封存 PT 按相同舍入规则保存 PNG。实际加载 PT 检查其像素与原行一致，PNG 与 PT 舍入结果一致，然后从磁盘打开 PNG，以与 CLI 相同的 uint8→float32/255 路径执行冻结 Reader。

生成仅接收图像和原问题，greedy 32 tokens；不先执行 target-conditioned CE。生成完成后使用原固定 GOLD_IDS 与立即 EOS 151645 评分，不放宽同义词、大小写或多余输出。对原 FP32 与 PNG 输出完整配对，保留正确、修复、退步与仍错分别计数。两套真实 RGB 链全部 960 条应保持像素和 token 序列一致，否则单独标记路径复现失败。

脚本：`scripts/probes/complete_png_readback.py`、`scripts/reporting/collect_png_readback.py`、`scripts/inspire/run_png_readback_suite.py`。固定矩阵定义在 `scripts/experiments/png_readback_protocol.py`。全量原始输出、源 complete/raw、像素绑定和 control PNG 归档；matched PNG 复用对应原验证归档。本地重算可传入原验证归档解压目录，实际复核每张 PNG 及 Reader 像素哈希。

本文件是预先固定的验收要求，尚不是成功结果。即使 PNG 全通过，可支持的范围仍为这些已见实体与语义问题，以及预先固定的新事件表达；不宣称未见实体或任意多事实记忆泛化。
