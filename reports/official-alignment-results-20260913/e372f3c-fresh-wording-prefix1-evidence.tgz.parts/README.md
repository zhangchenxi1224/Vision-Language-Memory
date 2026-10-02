# 完整已观察表达回归历史分路 1

保存原始 105494599 字节归档的全部 13 块，不省略或更改任何成员。分块用于满足 GitHub 单文件 100 MiB 限制。

原归档 SHA256：`80979508cda588df059e1c6f99e0bed999ace9a3ad9250f738c3d01c5342d03a`；独立远端观察的 manifest SHA256：`54d331e1b40175ff1aab92b85b0e2ec144614886d0f216f65535391c6d5bf27a`。

在仓库根目录重建：

```bash
python scripts/reporting/assemble_portable_archive.py --manifest reports/official-alignment-results-20260913/e372f3c-fresh-wording-prefix1-evidence.tgz.parts/manifest.json --manifest-sha256 54d331e1b40175ff1aab92b85b0e2ec144614886d0f216f65535391c6d5bf27a --output .cache/e372f3c-fresh-wording-prefix1-reconstructed.tgz
```

全部分块已在本地实际重建为相同 SHA256 的原归档。完整本地复核通过 560 条 raw、480 个 matched、96 张 PNG；该分路 480/480。
见[本地复核](../e372f3c-fresh-wording-prefix1-local-verification.json)及[原摘要](../e372f3c-fresh-wording-prefix1-summary.json)。

训练源码 4fbc857；生成源码 e372f3c；收集修复源码 0c4d893。这是此前已观察的表达回归，不是新 holdout；不能用单一历史分路通过代替连续链验证。
