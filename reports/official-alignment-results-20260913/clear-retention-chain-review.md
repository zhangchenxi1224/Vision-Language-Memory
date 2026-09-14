# 继续训练后的连续清除回归

4fbc857终点在e372f3c原注册功能集的远端完整汇总为1790/1800；其中单次写入360/360、连续链470/480、两个历史分路各480/480。当前本地已完整复核单次写入、连续链和真实CLI；历史分路的大归档仍在分块传输，不能称全1800格都已完成本地复核。另一套已观察表达回归仍在运行，随后才进行本轮全PNG读取。

本地直接重新核算390条单次原始读取及72张PNG、480条链读取及96张PNG，均与远端完整摘要一致。每路验证同时重放父终点6040条raw及19328次训练抽样，并检查固定4f训练/e372验证版本和原生推理登记。没有改变原始token与立即EOS判定。

| 完整480条链读取的对照 | 对照正确 | 本轮正确 | 保留正确 | 修复 | 退步 | 仍错 |
|---|---:|---:|---:|---:|---:|---:|
| 03原生条件终点 | 480 | 470 | 470 | 0 | 10 | 0 |
| b9历史表达增强终点 | 460 | 470 | 460 | 10 | 0 | 10 |

剩余全部10条错误来自`sequence-0-rep-2-step-4`（清除）与`sequence-0-rep-2-step-5`（noop），每张图的5种问题都应回答`no active preference`，实际均回答`jazz`。它们是b9既有错误的子集；本轮修复了b9另一重复中的10条错误，但仍未恢复03的完整链能力。此处比较覆盖完整480条链原始数据，不是挑选成功案例；其它功能分路另行完整复核。

实际CLI重放的6次写入和30次读取本地逐项一致，`parity_pass=true`，而`reference_functional_pass=false`。CLI一致性说明导出部署重放一致，不能消除原整体链功能失败。

已完成的原始证据与本地复核：

- [单次写入完整归档](e372f3c-logical-confirmation-evidence.tgz)，66562791字节，SHA`8a58df5b774d95cf66da20baa307c0290b3d1e1b628becbbe5e3db64aec04ee8`；[本地完整复核](e372f3c-logical-confirmation-local-verification.json)，[原摘要](e372f3c-logical-confirmation-summary.json)SHA`5e294de7e483e399d9fe658f4f3f59edeab459552032751bc43db8af950af3f5`。
- [连续链完整归档](e372f3c-logical-chains-evidence.tgz)，88523183字节，SHA`b52e34a52fc32dcb9ca8ce6d544659388c84bdd33725ef89b6b8006257f36b92`；[本地完整复核](e372f3c-logical-chains-local-verification.json)，[原摘要](e372f3c-logical-chains-summary.json)SHA`8997b26c180a584cd50bb819bc92ea67edeb31ab91173293b850bafecb8e233e`。
- [完整逐案例配对](clear-retention-chain-paired-comparison.json)，包含与03及b9的全部480格比较。
- [CLI完整归档](e372f3c-logical-cli-evidence.tgz)，6501995字节，SHA`d66af435810ab15adc908bf2283af91deecc5384d8dbcd9437c6b90cd1f3bf82`；[本地CLI复核](e372f3c-logical-cli-local-verification.json)。

传输曾遇到CPU中转SSH隧道不稳和整文件180秒超时；训练与评估没有重跑。已改为8MiB分块，逐块校验后重建完整归档并校验原始SHA。上述两份本地归档是完整文件，失败传输的残缺文件已由完整验证后的文件替换。历史两份原归档分别105641188、105553263字节，完整无损XZ重压缩仍超过GitHub100MiB；将保存完整分块和可重建manifest，不删减PNG或原始读数。

这些证据支持本轮改善了部分旧链错误，但仍未通过完整功能验收。开发1510/1510不能替代该结论；不能宣称未见实体或任意多事实记忆泛化。Goal保持active。
