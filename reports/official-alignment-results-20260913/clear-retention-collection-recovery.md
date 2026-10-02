# 已观察表达收集故障与恢复

e372f3c的四路已观察表达生成已全部完成，但首次collector在1789366050.5606384拒绝收集。实际日志为`Parent development outcome or validation interpretation changed`：probe已正确记录`observed_wording_regression_seen_semantic_questions`，collector仍要求旧`fresh_event_wording_seen_semantic_questions`。这是一处来源标记漏同步，不是模型评分失败；原注册功能中的清除失败另有完整数据证明。

修复源码`0c4d8935429cd71669ccccc8e86942ac9baf4507`按封存计划是否复用既有案例严格区分两种标签，错误标签仍拒绝；相关完整收集/评分回归测试17项通过（353.24秒）。修复独立部署至`repos/dreamlite-clear-retention-recovery-20260914`，原e372生成源码保持干净不变。新的`--generation-source-root`对原生成目录检查精确git commit、clean状态及scripts/src每个Python文件的实际SHA，随后仍核对全部原产物、张量、原始token/EOS和固定覆盖范围。

恢复driver3500970从收集阶段开始，没有重新生成图像、重新训练或选择其它checkpoint。原失败状态原字节保留为[initial-failure.json](e372f3c-fresh-wording-initial-failure.json)，本地独立下载校验SHA`d1e74b673177f973f3732d12bc03d3b0978a455cff3ef367ef089005930407e6`。恢复后的suite状态记录原失败路径、SHA、原生成commit和新collector commit。原e372 PNG失败状态也保留；新PNG使用不同0c4d893前缀。

四路收集及原e372源码的实际CLI重放于1789367380.1655116恢复完成。远端已观察表达汇总1750/1800：single360/360、chain430/480、prefix0与prefix1各480/480；50条链错误仍在。全套大归档本地复核进行中，不能提前称全部完成。实际[CLI完整归档](e372f3c-fresh-wording-cli-evidence.tgz)6276456字节，SHA`fe2b3b323a57696de8383b3c9e52f3d84e162a52d880ef4fda75783d3504bf3b`；[本地CLI复核](e372f3c-fresh-wording-cli-local-verification.json)确认6写30读parity true、整体reference functional false。

新的0c4d893全PNG已经实际启动，仍使用原e372的两套完整生成数据、相同Reader和全部固定案例。恢复了运行流程并不代表模型已可用。

传输故障独立处理：旧CPU中转r2的事件多次显示重新等待Notebook就绪，SCP反复掉线。已保留r2，新建`dl-align-cpu-20260914-r3`（CPU资源-2、4CPU/16GiB、cpu-nat-206、0点券/小时），恢复传输时复用所有已校验分块。GPU训练与已完成生成没有因此重跑。
