# 完整历史分路0归档

这里保存原始105641188字节gzip归档的全部13个分块，没有移除或改变任何tar成员。使用分块是因为原文件超过GitHub的100MiB单文件限制；完整无损XZ重压缩仍超过限制。

原完整归档SHA256：`20c7078016c20ea73bed6367cc83da4833d54199ac318e0288ded1be1f07457b`。
独立远端观察的manifest SHA256：`31b42c0aaa4ae68dee354965110524c6a4f83a2d17d87cf91ac04269f92f1e2f`。

在仓库根目录重建：

```bash
python scripts/reporting/assemble_portable_archive.py \
  --manifest reports/official-alignment-results-20260913/e372f3c-logical-prefix0-evidence.tgz.parts/manifest.json \
  --manifest-sha256 31b42c0aaa4ae68dee354965110524c6a4f83a2d17d87cf91ac04269f92f1e2f \
  --output .cache/e372f3c-reconstructed-prefix0.tgz
```

该操作已在本地对全部13块实际执行，得到与远端原归档完全相同的完整SHA。随后用`verify_broader_outputs_local.py`重算全部560条raw、480个matched格及96张PNG，结果480/480，见[完整本地复核](../e372f3c-logical-prefix0-local-verification.json)和[原摘要](../e372f3c-logical-prefix0-summary.json)。原摘要SHA为`e6ae773da0cc3dba148455deaff15667d0b28bd648f9721957108548d16e480d`。

父训练4fbc857，生成源码e372f3c，原注册验证。不能将这一分路的通过替代完整功能链或其它表达的结果。
