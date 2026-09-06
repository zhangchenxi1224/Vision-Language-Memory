# 本轮执行、结果解释与下一步

## 一句话结论

固定 target1 的 source-only 初始化诊断真实完成 256 次 Adam 调用。工程与 teacher 正对照通过；距离门、Reader 门及相对 Gaussian 父臂的二级改善均失败。它不能推进 Phase 1A 通过数，也不能启动 Phase 2。

## 改了什么，依据什么比较

唯一变化是 `xT_init = source_latents.float().clone()`。初始化只接收输入 source 的内容；完整冻结四步 DreamLite、固定 event conditioning、canonical teacher MSE、256 次 Adam/post128-cosine、raw256 主终点及评测口径全部保留。Teacher 仍向优化提供标签，因此本轮不是答案无关的完整求解器，更不是共享 writer 训练。

| 固定 target1、raw256 | endpoint MSE | Reader mean CE | Reader | 两个主门 |
| --- | ---: | ---: | ---: | --- |
| Gaussian/post128-cosine（唯一二级对照） | 0.09553645551204681 | 25.536474171257463 | 0/4 | 均失败 |
| Teacher-matched 初始化（仅描述性参考） | 0.06570044904947281 | 10.97591445709591 | 0/4 | 均失败 |
| 本轮 source-only 初始化 | 0.11177215725183487 | 25.765637596946004 | 0/4 | 均失败 |

四视图是同一 target 的固定排列，不是四个独立样本。跨臂只比较同一 teacher 下的绝对指标；各臂的完整四步 M0 不同，不能直接混用比值分母。

## 第一性原理解释

1. **链路没有断。** 每次都有完整四步、有限非零 xT 梯度，模型保持冻结，且没有梯度裁剪；本轮最小梯度范数为 0.00015465387672581563。这只排除了“完全没有梯度”，不证明目标容易到达。
2. **目标图片确实可被 Reader 读取。** 同轮 teacher replay 是 4/4，mean CE 为 0.00017690610513911291。不能把新输出失败解释为 teacher 本身不可读。
3. **优化降低了误差，但没有达到恢复门。** 本轮完整四步 M0 MSE 为 0.22256910800933838，raw256 降到 0.11177215725183487；仍保留 M0 误差的 0.5021907948121229，而门槛是至多 0.01。L2 比值 0.7086542138533595、teacher NRMSE 0.5106778086988586，也分别高于 0.1。
4. **此次 source-only 起点没有优于固定 Gaussian 起点。** 绝对 MSE 和 Reader CE 均更高，预注册二级决策为 `distance_fail_reader_fail_secondary_init_not_improve`。这仅说明此 target、此 solver、此预算下的这一次初始化干预不足；不证明所有答案无关初始化无效或完整 DreamLite 输出空间不存在可读解。
5. **MSE 下降不是答案恢复。** 本轮 normal/reset Reader 均为 0/4。真实图片见图表；人眼观感不作为门槛，也不据此追加新的评分标准。

## 执行身份与时间

- 训练提交：`e0a43c43f893ac19c6976ecd4c84ea3dc2b23006`；先本地测试和推送，再部署固定提交。
- 实例：`vlm-r3-h200x2-live-20260717`；DreamLite cuda:0，Reader cuda:1；既有锁定运行环境不改动。
- 预检：2026-09-06 16:39:24.626291 至 16:40:22.530702 UTC，controller wall time 57.90054371487349 秒；一次完整前向、一次反向、零更新。
- 正式诊断：2026-09-06 16:41:24.402720 至 16:46:25.548594 UTC，controller wall time 301.14318819437176 秒。
- 原 H200 环境的独立 aggregator 核验了全部 256 receipts、5 个 checkpoint/full Adam、20 条 Reader rows，以及实际 CUDA 初始化/每 checkpoint 的原生起点重建。工程通过，主次诊断均失败。

有效代码目录：

```text
/inspire/ssd/project/exploration-topic/czxs26210936/Vision-Language-Memory-source-init-e0a43c4-20260907-retry01
```

唯一科学运行根：

```text
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-new/r11-new-source-init-e0a43c4-20260907-round01
```

最初带 reference 的 Git clone 报 `pack has 3 unresolved deltas`，未进入模型运行；Git 自行清理了未完成的 `.../Vision-Language-Memory-source-init-e0a43c4-20260907` 检出。随后在新的 `-retry01` 目录用 Git shallow clone 成功取得相同完整 SHA，核验 clean detached 状态。该部署失败为 **0 模型前向、0 更新的工程事件**，不是第二个科学实验；未改配置、未覆盖旧实验目录。

## 校验与完整交付

- 本地最终联合回归 564 项全部通过（含新渲染器 15 项）；原始 JUnit、源文件哈希及启动前两项 fail-closed 修复见 [就绪审计](../r11-new-source-only-init-readiness-20260907/README.md)。两个修复前测试快照单独保留，不冒充最终代码证据。
- 下载的三个原始 tar 和两个 launch log 共五个文件，其字节数/SHA256 与远端完全一致；解包前检查了目录越界与链接类型。
- 本地渲染器重新核验完整 inventory、256 行 receipt 和 Reader logits、checkpoint 距离与决策。CUDA 逐位算术和 Linux cosine 精确校验以原运行环境聚合器为准；本地 CPU 归约只作已有容差内的补充核验，单独标记，不替换原始展示数值。
- 四张图已目视检查：距离、优化器/梯度、Reader、真实 RGB。机器可读完整展示数据见 [training_diagnostics.json](training_diagnostics.json)。
- 原始归档：[预检](delivery-v1/technical-preflight.tar.gz)、[256 步诊断](delivery-v1/formal-target01.tar.gz)、[独立聚合](delivery-v1/aggregation-v1.tar.gz)；launch logs 同在 `delivery-v1/`。
- 原始 [comparison.json](aggregation-v1/comparison.json)、[RAW_ARTIFACTS.json](aggregation-v1/RAW_ARTIFACTS.json)、[逐行 CSV](aggregation-v1/distance_trajectory.csv) 同步交付。完整交付哈希见 `ARCHIVE_DELIVERY_MANIFEST.json`。
- comparison SHA256：`f7ec5913ac6dcd40860428e3c3ab929f8559752173599ffeae8cc0b01d1db09c`；RAW SHA256：`2889f183c04df70c2763010b863914e378ffb1a0c85a1d8ca0f6a22ac150d4d7`。

训练提交与本结果交付提交分离；后者以 Git 历史及任务回复中的真实 SHA 为准，不把报告提交当训练提交。

## 下一项最小候选：单独隔离 conditioning 内容

按本轮预注册失败分支，下一候选是仅把实际 conditioner 输入的 event prompt 换成固定 identity 文本 `no changes`，保留官方 edit 模板与 source 图像。对照必须是本轮 source-only + 原 event conditioning，不能换回 Gaussian 对照而同时改变两个因素。

该候选需先另行预注册，锁定实际文本、prompt、embeddings/mask 的哈希与同一预算/主门；不修改原始数据或原始 event，不试多个 prompt 后择优。它只检验“这一固定 conditioning 内容替换是否影响 dense-supervised 恢复”，不能据此直接判断 event 语义、prompt 格式或生成先验哪一个是根因。

**本轮交付时该候选仅完成只读分析，尚未预注册、实现或启动。** 即使其日后通过，也不代表 event→答案学习成功；回主线仍需恢复 event conditioning、不依赖逐样本 canonical teacher 的统一 QA solver，并完整重验固定八成员。

## 主线与时间边界

Phase 1A 仍为 6/8，失败成员为 1、7；Phase 2/3 未启动。不能用不同 solver 拼接 8/8。Phase 1B 延后和 Phase 2 仅做固定 64/128 的 MVP 修订继续有效。

原 30 小时规划从 2026-09-05 07:18:59.891307 UTC 起算；本轮结束时已约 33.46 小时，不重置起点，不再承诺 30 小时闭环，也不降低门槛或把 Phase 2 缩至 64 条以下。只有统一 Phase 1A 真正通过后，才能按既定公式重估 Phase 2/3 时长。
