# 继续训练后的完整已观察表达回归

4fbc857终点在原56登记的完整表达与噪声矩阵上，已完成1990条raw、360张生成PNG及1800 matched格的全部本地复核：**1750/1800**。原独立CLI六写三十读也已核验一致，但仍有功能失败。

| 分区 | b9原56结果 | 4f本轮e372结果 |
|---|---:|---:|
| 单次写入 | 360/360 | 360/360 |
| 六步RGB连续链 | 420/480 | 430/480 |
| 历史前缀0 | 480/480 | 480/480 |
| 历史前缀1 | 480/480 | 480/480 |
| 全部matched | 1740/1800 | 1750/1800 |

[完整原始逐格配对](clear-retention-observed-wording-paired-comparison.json)覆盖全部1990 raw，包括190个固定负对照；逐项核对case、condition、noise、query、事件文字、原始token和立即EOS。全部负对照图像SHA和token保持不变。结果是1740条保持正确、10条修复、50条仍错、0条新增退步。

修复发生在 `sequence-0-rep-2-step-4` 清除和 `step-5` 保持，每步5问。剩余5条失败链为sequence0的rep0/3，以及sequence1的rep0/2/3；它们的step4/5仍保留先前值，前两条链读出jazz共20问，后三条链读出ambient共30问，正确答案均为no active preference。完整16条链中11条全对；96个写后状态中86个五问全对。50条读取不是50次独立清除，而是5条链的10个失败状态各问5种表达。

四路归档都已完整下载和本地复核，原summary从下载归档按原字节提取，再与远端独立观察SHA核对。历史两路超过100MiB，保存全部13块及manifest，并实际重建为与远端相同的完整归档，未省略成员：

- [单次原摘要](e372f3c-fresh-wording-confirmation-summary.json)、[本地复核](e372f3c-fresh-wording-confirmation-local-verification.json)。归档SHA `b970f53b7b178aa099a8773b43177dafa27431da8e872edb61e2cfa986a25ef3`。
- [连续链原摘要](e372f3c-fresh-wording-chains-summary.json)、[本地复核](e372f3c-fresh-wording-chains-local-verification.json)。归档SHA `3136bc3b1ae0829a0d2d82b828f0f0de81060cb28bab7c84ab9c9a1551e3039a`。
- [历史0完整分块](e372f3c-fresh-wording-prefix0-evidence.tgz.parts/README.md)、[本地复核](e372f3c-fresh-wording-prefix0-local-verification.json)。归档SHA `9d1ef82d61ab55c4c453d587ef60c6f8f523b93388365d3825c182cc1f87afe6`。
- [历史1完整分块](e372f3c-fresh-wording-prefix1-evidence.tgz.parts/README.md)、[本地复核](e372f3c-fresh-wording-prefix1-local-verification.json)。归档SHA `80979508cda588df059e1c6f99e0bed999ace9a3ad9250f738c3d01c5342d03a`。

每路本地验证还重算父训练6040条raw和19328次实际抽样；大PT/checkpoint仍由远端审计，本地归档不包含这些大张量。生成源码始终是e372f3c，训练源码4fbc857；收集元数据故障由0c4d893恢复，原失败保留，未重新生成、改变评分或选择检查点，详见[恢复记录](clear-retention-collection-recovery.md)。

这套表达在选择4f修复前已被观察，不能重新称作未见holdout。03在原注册链上的480/480不能代表它在这里的能力，故补测[固定03检查点完整对照](../official-03-observed-wording-baseline-plan-20260914.md)，再判断哪些错误是已有弱点、哪些是训练后的退步。当前Goal仍未完成。
