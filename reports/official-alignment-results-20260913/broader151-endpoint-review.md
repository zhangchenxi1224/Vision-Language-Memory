# 151条件固定4832更新端点：状态转换改善，多题部分失败

训练源码`84cdfdb58ace96954243de5caf427948717c9abf`已完成4832更新、19328次实际抽样，优化阶段3857.9325秒。result SHA`c230451494fd8809621757b747bf3635d80fda5bc998e208a757128167147f82`，checkpoint SHA`12a579fcde24289fc0d8dd0d9c12c07be7b5270509bd238b5bfbfd05e89edbbc`。同一seed20260915的完整两噪声开发基线与最终结果配对；所有回答仍按原始gold token加立即EOS计分。

| 分区 | 训练前 | 训练后 | 变化 |
|---|---:|---:|---:|
| 135个音乐状态转换/表达条件 | 1230/1350 | 1340/1350 | +110 |
| 16个历史完整前缀问题 | 43/160 | 37/160 | −6 |
| 全部matched | 1273/1510 | 1377/1510 | +104 |
| 五个问法全部严格通过的图像 | 250/302 | 270/302 | +20 |

全部配对记录：1223条持续正确，154条由错转对，50条由对转错，83条持续错误。音乐部分120条修正、10条退步；历史部分34条修正、40条退步。**整体分数的增长不证明多题Writer成功；本端点功能门槛失败。**

音乐部分原始三种训练表达与扩展表达3–7均通过，唯一失败是`jazz → clear / wording-8`，两噪声五问法共10次仍输出`jazz`。该表达为已训练的“Unset the music preference ...; it should have no stored value.”。此结果是开发失败，不能归因于没见过这个字符串。

历史问题每题有两噪声×五问法：

| target index | 正确答案 | 最终严格正确 |
|---|---|---:|
| 0、1 | green | 各8/10 |
| 2、3 | juice | 1/10、8/10 |
| 4、5 | jazz | 0/10、3/10 |
| 6、7 | linen | 各0/10 |
| 8、9 | pasta | 0/10、9/10 |
| 10–15 | no active preference | 全部0/10 |

所有16题都至少有一次失败。green两题的paraphrase_2共四条输出`Green`，pasta target9有一条输出`Pasta`，大小写不同的原始token仍判错；不改评分。其余失败还包括错误值、无法确定、同义但不符合固定答案的表述，以及带实体说明的长回答。材料题例如输出`Silver`、`glass`、`wood`；某些clear题仍输出`green`、`blue`或`linen`，不能简单解释为全部只是EOS格式问题。

此次真实采样每条件128次。135个音乐条件合计17280/19328次（89.4%），16个历史问题合计2048/19328次（10.6%），每历史目标只有128次。音乐三个独立目标张量各有45个条件记录，各获5760次目标曝光；每历史目标则只有128次。扩展等价表达也增加了同一个任务的采样权重。**这是已验证的数据分配事实，尚不足以单独证明失败根因。** 下一步需要结合全部独立验证和目标拟合误差，区分任务采样不平衡、优化曝光不足及目标附近读取鲁棒性。一个有依据的对照是按15种source/operation转换加16个历史前缀构成31个逻辑条件分层抽样，再在等价表达中均衡选择，保持官方FM与推理协议不变；该对照尚未实现、注册或执行。

完整端点由`c2ec407fe9f9600a23dc8009b657b7391b9aff6d`真实检查：604份评估PT、完整checkpoint、6040条raw、19328次确定性draw及初始/首步/末步四rank一致性。实际sigma范围`[6.788969039916992e-05, 0.9999235272407532]`，9583次大于0.5。远端summary SHA`a5cdb990673c999560e95a1552e04b90f46608145b8c92bcc56734529bda673c`；归档3,594,927字节，SHA`d3d2cb39939e43fd604e2b06ae440bfd4111c61d6e428e0784bbb6810cdf0d0d`。二者下载后实际SHA一致，`verify_broader_outputs_local.py`已真实完成全部6040raw和19328draw重算。大型PT仍远端，本地遗漏明确。

证据：[远端完整核验摘要](c2ec407-broader-endpoint-summary.json)、[原始文本归档](c2ec407-broader-endpoint-evidence.tgz)、[本地完整重算](c2ec407-broader-endpoint-local-verification.json)。原pilot terminal中的`single-question regression evaluation`是遗留范围文本，与bank及training identity中真实17题不符；本报告按固定bank和实际全矩阵统计，不改写原始完成记录。未来pilot范围元数据需要据实际语义题数修正。

四路独立验证已在同一用户四H200实例启动：single、RGB chain、两组historical prefixes。父开发失败被保留为`diagnostic_after_development_failure`，独立完整结果及新包重放仍待完成。goal保持active。
