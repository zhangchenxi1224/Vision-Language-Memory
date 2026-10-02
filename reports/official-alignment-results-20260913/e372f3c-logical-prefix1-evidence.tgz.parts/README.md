# 完整历史分路1归档

本目录含原105553263字节gzip归档的全部13个分块，未删改tar成员。原完整归档SHA256为`f665810884ac21bff74184decbe1870bb362155b341b393af9020a50d6aeba1b`，独立远端manifest SHA256为`8a1bd715506988631351b5f796c70bd93b951642aae53df169e6deb498d2d3d3`。

在仓库根目录重建：

```bash
python scripts/reporting/assemble_portable_archive.py \
  --manifest reports/official-alignment-results-20260913/e372f3c-logical-prefix1-evidence.tgz.parts/manifest.json \
  --manifest-sha256 8a1bd715506988631351b5f796c70bd93b951642aae53df169e6deb498d2d3d3 \
  --output .cache/e372f3c-reconstructed-prefix1.tgz
```

已实际重建全部分块并校验原完整SHA；随后本地重算560条raw、480个matched格及96张PNG，480/480通过。[完整本地复核](../e372f3c-logical-prefix1-local-verification.json)，[原摘要](../e372f3c-logical-prefix1-summary.json)SHA`b75e584590ce245046b60899ca8fc1f0816f1279531d4d8408818e2966e09066`。

分块仅为传输和GitHub单文件限制所需，归档内容完整。父训练4fbc857，生成e372f3c，原注册验证；不能代替整体功能或其它表达结果。
