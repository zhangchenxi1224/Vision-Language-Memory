# 仅初写源图的保持训练：B 已完成，准确率待读回

2026-09-24。[预先登记](INITIAL_SOURCE_RETAIN_PROTOCOL.md)的对照仅将保持输入改为学生真实初写 PNG，
仍使用原教师目标、官方 FM、原干扰交换和固定 2,048 次更新。此报告先记录已完成的 B；A 仍在训练。

## B 完整训练核对

- 2,048 次更新，8,192 次抽样；初写和保持各 4,096 次，每个偏好在两类抽样中各出现 64 次。
- 全部抽样的偏好、干扰位置、sigma 与原保持训练一致，前 32 步的损失和梯度逐值相等。
- 64 张实际初写源 PNG 哈希核对通过；目标 latent 清单与原 write2048 完全一致。
- 全部损失和梯度有限，零梯度步骤为 0；完整最终权重实际哈希与完成记录一致。

| B 的 FM 均方误差 | 前 128 步均值 | 后 128 步均值 |
|---|---:|---:|
| 初写抽样 | 0.161249 | 0.135727 |
| 保持抽样 | 0.516086 | 0.249323 |

这些是训练损失，不是记忆准确率。新旧保持实验的输入图片分布不同，
不能仅凭均方误差低于原保持组就认定递归能力更强。

最终权重 SHA256：`914baf66f122e05307a62e11e96a7973d9c9abce39e53531d497f5536e78761b`。
父权重仍为 B 的 write2048，SHA256：`d446c67bedf2b23a5f5e8294e9738c5101fb431d06eef9536eb3525d0c38e4c3`。
训练实际记录约 3,228 秒，不含条件缓存与后续生成时间。

## 接续阶段

B 原训练进程正常结束，新的 pilot RGB 链生成进程为 4137593，
参数实际指向本轮最终权重，仍从灰图开始、每步保存/重读 PNG、使用 28 步原生生成。
固定首例 0/1/5/10 的读取任务等待完整匹配/错配两噪声链；完整 pilot/dev 评测同时保留。
尚没有本轮最终链的准确率结论。

独立的固定64 FM8192 B 对照也已按等待条件接续，训练进程为 4141893，
仍从原 4f 父权重出发；它不使用本轮保持权重。
这是两个不同的问题：本轮检查一步保持输入，本预算对照检查教师到共享 Writer 的单写拟合。

- [完整训练核验](evidence/initial-source-retain-B-training/initial-source-retain-B-training-verification.json)
- [B 的 2,048 步原始轨迹](evidence/initial-source-retain-B-training/B/train/optimization.jsonl)
- [目标、父权重及输入模式](evidence/initial-source-retain-B-training/B/train/manifest.json)
- [完成记录](evidence/initial-source-retain-B-training/B/train/complete.json)
