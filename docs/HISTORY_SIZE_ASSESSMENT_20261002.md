# 2026-10-02 Git 历史瘦身评估

本轮仅完成评估，没有执行历史重写、过滤、GC 或强制推送 main。测量对象为整理提交 `e6fa9c8669823747f32a62ad325b9ae6d4a948c6`；随后增加的整理文档不计入以下冻结快照数值。

## 当前体积

| 对象 | 数量 / 字节 |
| --- | ---: |
| main 当前 tree | 5,302 文件 / 6,017,878,736 bytes（5.605 GiB） |
| 当前 tree 去重 blob | 5,226 个 / 6,002,233,165 bytes |
| 当前 reports | 6,009,310,822 bytes |
| 9/13 engineering provenance | 5,228,297,695 bytes，约占当前 tree 86.9% |
| K1 报告目录 | 711,402,704 bytes |
| multitarget 报告目录 | 40,710,640 bytes |
| PrefEval 官方协议目录 | 28,428,946 bytes |
| 原本地仓库 4 个 pack 文件 | 9,609,118,607 bytes（8.949 GiB） |
| 原本地 loose objects | Git 报告 553,020 KiB（约 540 MiB） |

当前 tree 的大文件主要是证据压缩包：106 个 `.tgz` 合计 3,600,950,333 bytes，173 个 `.part` 合计 1,427,890,647 bytes，84 个 `.gz` 合计 581,505,688 bytes。最大单文件是 `reports/official-alignment-results-20260913/56b56a8-fresh-wording-prefix0-evidence.tgz`，102,725,299 bytes。

上述三种尺度不同：tree 是当前文件逻辑体积；blob 去重后是对象原始内容大小；pack 是本地实际压缩文件大小。它们都不能直接作为 GitHub 下一次 clone 的精确网络传输量。

## 为什么删文件、删分支不会直接缩小历史

621 个移出主线的路径合计 1,915,598,165 bytes，对应 603 个不同 blob，去重后为 1,913,954,173 bytes。603/603 仍可从 main 的祖先提交访问，只是当前 tree 不再包含。因此，不能把“当前目录撤下约 1.78 GiB”描述为“完整克隆节省约 1.78 GiB”。

整理提交的完整祖先可达 7,005 个 blob，逻辑内容合计 7,939,214,994 bytes。它与原 32 个远端分支的并集可达 10,033 个 blob，共 10,556,784,547 bytes。由归档历史额外保留 3,028 个 blob、2,617,569,553 bytes 和 83 个不同提交。归档标签使这些对象继续可达，删除原分支引用不会删除它们。

保留的 Release 标签 `prefeval-official-ab-teachers-20260924` 指向 `9e2a374`，其历史本身可达 2,919 个 blob、7,170,657,144 逻辑 bytes。因此，即使未来仅重写 main，只要归档标签或 Release 标签继续指向旧历史，大量旧对象仍会保留。同仓库保存完整历史与彻底移除对应历史对象不能同时成立。

27 个 Release assets 合计 7,953,287,944 bytes，是独立存储的科学资产，未纳入上述 Git pack 数字。现有 Release 资产全部保留，不因其体积大而删除。

## 后续方案

1. **先控制新增体积。** 新的大型训练权重和完整证据压缩包放入 Release 或持久对象存储，Git 保存来源、下载入口、文件大小、SHA-256、run manifest 与复现脚本。迁移先验证下载和校验，再调整读取入口。
2. **优先评估 9/13 工程证据的当前 tree 迁移。** 该目录占当前文件体积约 86.9%，收益最大，但包含 bank、初始化和测试夹具；需逐项确认依赖及外部存储可靠性。移出当前 tree 可以减小检出体积，仍不会自动减小旧 Git 历史。
3. **将真正的历史瘦身作为独立迁移。** 先制作不依赖 alternates 的完整 mirror 或 bundle，并做独立恢复验证。当前 shared bare 审计库及增量 bundle 不满足这个备份条件。随后在隔离副本过滤目标路径或 blob，核对 branches、tags、Release、文档 SHA、下载地址和恢复脚本。
4. **选择历史保留位置后再改引用。** 可保留当前仓库为历史与 Release 宿主，另建精简开发仓库；若必须沿用当前开发地址，则先迁出完整历史，再处理原仓库所有保留旧对象的引用。后者涉及历史改写、提交 SHA 变化、Release 标签关系和协作者重新同步，应单独授权后执行。本轮不实施这一步。

短期按需下载可以缓解传输压力，但不能替代历史恢复：浅克隆不适用于直接运行依赖旧 `git show` 的 preflight；历史复现应获取对应标签及所需完整历史。

## 测量方式

使用只读 Git tree 与对象头查询，没有解压读取全部大文件内容：

```bash
git ls-tree -rlz e6fa9c8669823747f32a62ad325b9ae6d4a948c6
git count-objects -v
git rev-list --objects e6fa9c8669823747f32a62ad325b9ae6d4a948c6
git cat-file --batch-check="%(objectname) %(objecttype) %(objectsize)"
```

分支并集按部署前 manifest 中的固定 SHA 计算；pack bytes 按原仓库 `.git/objects/pack/*.pack` 文件长度合计。共享审计对象库不重复计为另一份物理备份。
