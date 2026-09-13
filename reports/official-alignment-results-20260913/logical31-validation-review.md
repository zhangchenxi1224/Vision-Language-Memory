# 31逻辑条件抽样后的配对独立验证

训练版本bb34092完成4832更新；验证源码固定16bc3d077fa7604a0d84c3b008a80baf327b265b。沿用此前c2ec407已观察的全部case、表达和噪声，属于配对诊断，不是新的未见测试集。当前版本尚未达到可用标准。

| 完整矩阵 | 原151条件均匀抽样84cdfdb | 31逻辑条件均匀抽样bb34092 |
|---|---:|---:|
| 单次写入，360条严格答案/EOS | 360/360 | 280/360 |
| 单次写入，72张图五问法全对 | 72/72 | 56/72 |
| 连续写入，480条严格答案/EOS | 470/480 | 390/480 |
| 连续写入，16条完整链全对 | 15/16 | 7/16 |
| 历史前缀，960条严格答案/EOS | 170/960 | 895/960 |
| 历史前缀，192张图五问法全对 | 1/192 | 166/192 |

单次写入全部390条原始记录（含30条对照）和72张PNG已下载、校验并在本地完整重算。80条失败全部属于从灰图执行原始清除表达的16个噪声种子，每张图五问法均未达到约定的`no active preference`加立即EOS；其余280条通过。这不是仅一两个噪声的偶发失败。

| 原始失败输出 | 条数 |
|---|---:|
| No music preference recorded. | 32 |
| No music preference set | 1 |
| No music preference specified. | 16 |
| No music preference stored. | 16 |
| indigo desk train 001123 music preference: none | 13 |
| no music preference | 2 |

这些输出与开发集灰图清除的非约定回答一致。连续写入的完整480条raw、96PNG也已本地重算通过，90条失败分布在18张图、9条链：全部成对发生于清除及紧接的no-op；其余步骤通过。sequence0的rep0/1/2在step4/5仍输出jazz（30条）；sequence1的rep2在step4/5仍输出ambient（10条）；sequence1的rep0与sequence2/3的rep0/1在清除及其后保持时输出非约定的“未记录偏好”等回答（50条）。这包含明确的旧值残留，不只是答案格式偏差。保持步骤直接读取并接续失败的实际RGB图，没有重置到正确教师图。

完整单次证据archive为`16bc3d0-logical-confirmation-evidence.tgz`，64,114,706字节，远端和本地SHA均为 **be548e5bc63560ccb3578c58a8824f91cf2f3a34b8c3ccd0b02f0f26593e973d**。600秒SCP超时后保留61,102,080字节已验证前缀，仅补3,012,626字节尾部，完整哈希随后一致；恢复元数据一并归档。远端collector实际检查78份PT张量；本地校验覆盖全部raw、输入绑定、PNG和既定19328次训练抽样重放，大型PT仍留在共享盘。

完整连续写入archive为`16bc3d0-logical-chains-evidence.tgz`，85,861,869字节，远端和本地SHA **61d56b95032377341a080333ac4887888c5559e8aa0daf2e857d5948ca66669b**，下载正常完成，全部480raw/96PNG重算与远端summary一致；96份PT在远端实际核验。complete SHA **9ffa78a310a2860b2fdab8def0978776dc7238d587f10e934639a04da8900554**。prefix0/prefix1和真实CLI重放仍在继续。

09-14 补齐：prefix0 为 428/480、76/96 图五问法全对；prefix1 为 467/480、90/96 图五问法全对。两份完整 archive 已下载并验证远端 SHA，再由本地完整重算全部 raw、19328 次训练抽样及192张 PNG；PT 在远端实际核验。本地读取包含 Unicode 的原始回答时使用 UTF-8，不修改回答或评分。prefix0 SHA 为 `19a0b435b1682d756adb338adb53866266b43d4b044c828e8c3c030226fb922d`，prefix1 为 `7aabfe0290289cac192f0c06b422e9d87c3862bf7512664859f292bb8151dba4`。全四路共1990条原始记录、360张PNG均完成本地重算。

真实导出 CLI 已在旧单卡完成6次写入、30次读取，parity_pass=true、reference_functional_pass=false：工程一致性通过，但模型仍有功能失败。完整 archive `16bc3d0-logical-cli-evidence.tgz` 为6,208,749字节，远端/本地SHA **e0e8f9d4720b9717ddfa3d763c5a0e28a7c94c22021fe82b1d47b22fa357483f**。独立本地复核全部36条command/result、30次原始读取、6次写入PNG及最终持久PNG与原始parity记录一致；不把仅相同的错误回答算成功。

抽样改动显著改善历史前缀，却损害清除和连续更新，不能交付为可用版本。下一轮固定相同预算与抽样，只改变训练条件编码，依据全302格真实首步速度对照进行；保持官方原生推理不变。

两轮全1990条验证raw现已按case、condition、noise和query逐条配对，并检查query/gold/event完全相同、固定负对照输出与图像不变，见[逐格配对结果](logical31-paired-raw-comparison.json)。历史前缀原先170条正确全部保留，新增725条正确，剩65条仍错；单次写入新增80条错误，连续更新也新增80条错误，原先10条链内错误均未修复。这是同一组格子的转换统计，不是只比较两个总分。

历史剩余65条中，33条应答juice却返回无偏好或coffee，13条应答linen却返回未指定或silver等错误，19条应答约定的no active preference却返回其他原始文本。prefix1的13条失败均属于后一类，包括11条no preference、1条no drink preferred和1条混合语言异常输出；不将它们重新记为正确。既有严格评分保持，当前对照仍必须同时检查历史改善是否保留以及清除/链失败是否修复。
