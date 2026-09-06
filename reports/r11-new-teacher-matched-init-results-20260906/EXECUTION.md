# 本轮执行、归档与下一步

## 一句话结论

本轮 teacher-matched initial xT 诊断真实完成 256 次 Adam 调用，工程和 teacher replay 通过；距离门、Reader 门均失败。相对上一轮 Gaussian/post128-cosine 的绝对 MSE 和 Reader CE 均改善，但正确率仍为 0/4。它不是主线 Phase 1A 通过，更不是共享 writer 训练成功。

## 执行身份与时间

- 目标实例：`vlm-r3-h200x2-live-20260717`；H200 两卡，DreamLite 在 cuda:0、Reader 在 cuda:1。
- 训练提交：`5bffcb9a1605c7cdef77df1126daca6f9578bdda`。先本地测试和推送，再经 CPU 联网实例准备 clean detached checkout；GPU 离线读取。
- 真实预检：2026-09-06 15:17:40.838453 至 15:18:40.180295 UTC；controller wall time 59.3400651300326 秒。
- 正式诊断：2026-09-06 15:20:32.313721 至 15:25:34.394929 UTC；controller wall time 302.07757396250963 秒。
- 训练结束后，使用同一提交在 H200 上完成独立聚合，逐行复算 256 receipts、全部 5 个 checkpoint、20 行 Reader logits，以及 CUDA 初始化与每 checkpoint 的起点重建。
- 完成后只读查询未发现 GPU 计算进程，suite lock 已释放，远端 checkout 仍干净且 HEAD 未变；未停止实例。
- 本任务为唯一执行方；原任务保持只读，避免双重启动和 Git 写入。

真实运行根：

```text
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-new/r11-new-teacher-init-5bffcb9-20260906-round01
```

独立代码检出：

```text
/inspire/ssd/project/exploration-topic/czxs26210936/Vision-Language-Memory-teacher-init-5bffcb9-20260906
```

## 第一性原理拆解

链路是：initial xT → source-anchor 混合起点 → 完整四步冻结 DreamLite → 最终 latent → 冻结 VAE/Reader。

1. **起点确实放对了。** 初始混合状态相对 teacher 的 NRMSE 为 0.0021728272911401753，低于锁定技术阈值 0.01。
2. **起点接近 teacher，不等于最终结果接近 teacher。** 零更新但完整跑完四步后的新 M0 MSE 为 0.1855299025774002；不能把起点当作终点，或绕过完整链路。
3. **优化确有改善，但改善不足。** raw256 MSE 为 0.06570044904947281，MSE/新 M0 为 0.3541232337038694，未达到 0.01；NRMSE 为 0.3915293636832485，未达到 0.1。
4. **Reader 正对照可用，但新输出仍不可正确回答。** teacher 为 4/4；raw256 为 0/4，CE 为 10.97591445709591。这不是“teacher 本身无法读取”，也不能只凭非零梯度声称求解有效。
5. **初始化是一个有影响的因素，但不是已证明的唯一根因。** 相对固定父臂 MSE 0.09553645551204681、CE 25.536474171257463，两项均严格改善，满足预注册二级审计；主门依然失败。本轮只检验一个 target，不能证明全部初始化、全部任务或模型容量的结论。

## 工程测试与数值语义

- 训练前联合回归 260 项全部通过；证据位于上级目录的 `r11-new-teacher-matched-init-readiness-20260906/`。
- 真实预检：一次完整四步前向、一次反向、零 optimizer updates，唯一 FP32 xT 可训练，模型冻结且无梯度，teacher 四视图回放通过。
- [backend-probe.json](backend-probe.json) 是提交后在 H200 上运行的纯张量探针；0 模型前向、0 训练。它验证了 CPU/CUDA 算术差异与同 CUDA 重放一致性，不是科学成功证据。
- CUDA 初始化/起点逐位重建和 Linux cosine 学习率逐位核验由原运行环境独立 aggregator 完成，不使用 Windows 替代其精确算术语义。
- 本地报告额外从原始 checkpoint 和 logits 复算；浮点归约结果用于既有容差核验，图表和主结果始终显示绑定的远端原始值。本地复算值单独保存，不更改原始数据。
- 渲染脚本测试见 [validation/pytest-junit.xml](validation/pytest-junit.xml)；报告正文及可视化另见 [README.md](README.md)。

## 完整交付

| 产物 | 内容 |
| --- | --- |
| [technical-preflight.tar.gz](delivery-v1/technical-preflight.tar.gz) | 配置绑定、manifest、环境、初始化、step0、teacher rows、日志、终态与哈希清单 |
| [formal-target01.tar.gz](delivery-v1/formal-target01.tar.gz) | 256 receipts、5 个完整 checkpoint/Adam 状态、真实 RGB、20 Reader rows、全部日志和终态 |
| [aggregation-v1.tar.gz](delivery-v1/aggregation-v1.tar.gz) | 独立 CUDA 聚合原始封装 |
| [comparison.json](aggregation-v1/comparison.json) | 机器可读主门、二级审计与决策 |
| [RAW_ARTIFACTS.json](aggregation-v1/RAW_ARTIFACTS.json) | 每个源 artifact 的可追溯绑定 |
| [distance_trajectory.csv](aggregation-v1/distance_trajectory.csv) | 远端原样 receipt 数据；loss 是更新前测量 |
| [ARCHIVE_DELIVERY_MANIFEST.json](ARCHIVE_DELIVERY_MANIFEST.json) | 下载归档、额外交付与校验值 |
| [DELIVERY_MANIFEST.json](DELIVERY_MANIFEST.json) | 渲染器生成图表/正文的哈希与验证范围 |

下载时三个完整 tar 的 SHA256 均与远端一致；本地另逐项核验源 inventory 的字节、文件集合和哈希。未覆盖已有远端目录。训练 commit 与交付 commit 分离；本轮实际交付 SHA 以 Git 历史和任务最终回复为准，不将报告提交冒充训练提交。

## 下一项候选：只移除初始化中的 teacher 信息

当前预注册分支为 `distance_fail_reader_fail_secondary_init_improves`。最小候选是 **source-only initial xT**：

```text
xT_init = fixed_source_latents.float().clone()
```

只改变初始化；固定 target01、teacher MSE 标签、模型/条件、完整四步、256 Adam 调用、cosine 学习率和原主要门。对照为已闭合的 Gaussian 父臂，teacher-matched 臂作为补充参考。各臂使用自身完整四步 M0，跨臂只比较绝对 MSE/CE。

此候选检验“不在初始化注入逐样本 teacher，输入本身提供的起点是否也有帮助”。即使通过，teacher 仍用于 dense loss，因此仍只是 bridge 诊断。不得称其为已不依赖 teacher 的完整 Phase 1A solver。

只读搜索当前 configs/scripts/reports 未发现相同完整链路 source-only xT 实验；canonical R11 的 blank VAE latent 初始化、R6 source/noise 混合及“source-only import”均不是此实验。该搜索未覆盖其他 Git 分支或未归档远端内容。

**本轮交付时，候选尚未预注册、未启动。** 必须先冻结新单因素配置、控制与判门，再实现、测试、预检和运行。不能在旧配置内追加试验，也不能根据结果改门槛。

## 主线与时间边界

Phase 1A 仍为 6/8，失败成员为 1 和 7；Phase 2/3 未启动。回到主线必须使用不依赖逐样本 canonical teacher 的统一 solver，在同一固定八成员上完整重验，不能拼接不同 solver 的成功项。

Phase 1B 延后、Phase 2 限定 64/128 的用户修订仍有效。约 30 小时规划从 2026-09-05 07:18:59.891307 UTC 起算，不重置；本轮交付时已经超过，不能继续承诺 30 小时闭环，也不以降低门槛或缩至少于 64 条补救。统一 Phase 1A 真正通过后，才按锁定公式选择 64/128 并报告修订 ETA。
