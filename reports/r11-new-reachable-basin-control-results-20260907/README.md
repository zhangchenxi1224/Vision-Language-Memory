# R11_new 固定可达目标吸引域判别结果

## 核心结论

**两个 warm start 均未通过；尤其 `alpha=.99` 被第一步 Adam 明显推离精确解。** 工程 gate 与独立审计通过，但本轮仍是诊断失败，Picture Memory `formal_success=false`、Phase 2 `false`。

同一固定目标按构造有已知精确 teacher `xT`。把学生初始化到 source→teacher 线段的 50% 和 99% 位置后，仍沿用 parent 的 Adam、LR=`0.05`、256 步、loss 与三重门槛。`alpha=.50` 只改善 42.7%；`alpha=.99` 不但没有改善，最终误差反而是初始值的 4.36 倍。

这把问题进一步收窄为：**Adam 的参数尺度与局部 DreamLite Jacobian 不匹配，首步发生确定性过冲。** 它不是“训练链路断了”，也不能再仅解释为“source 初始化离解太远”。

## 固定设计与工程完整性

- 精确复用 parent target artifact，SHA-256 `5352fac1cf614c8458d4f61b6ffcbf7b053f8a947dedac1286f80261ba8a772a`；禁止重新采样或重建 target。
- 初始化仅两项：`source + 0.50(teacher-source)`、`source + 0.99(teacher-source)`；teacher `xT` 只用于 oracle warm start，不进入 loss，因此本轮不可部署、只做几何诊断。
- 两条件各自新建 Adam；LR/调度、256 步、无裁剪、checkpoint 0/64/128/192/256、raw256 主 endpoint 均与 parent 相同。
- 总计 523 次完整 DreamLite forward、512 次 backward、512 次 optimizer step、0 次 Reader forward；模型快照前后一致。
- 运行提交 `7d4a84569ea422572fe2e474939f9648b099210a`；formal 用时 532.97 秒。
- Preflight 24 项、formal 40 项 inventory 的远端独立 audit 与下载后本地独立 audit 均通过。

## 精确结果

| 条件 | M0 MSE | raw256 MSE | MSE/M0 | L2/initial L2 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `alpha=.50` | 0.011216725 | 0.006426430 | 0.572933 | 0.756923 | false |
| `alpha=.99` | 0.000135312 | 0.000590490 | 4.363910 | 2.088997 | false |

门槛固定为 raw256 MSE `≤0.001`、MSE/M0 `≤0.01`、L2/initial L2 `≤0.1`，三项必须同时满足。`alpha=.99` 的绝对 MSE 虽低于 0.001，但相对误差恶化，不能算成功。

两条件连续 256 步均有有限非零梯度；最小非零元素比例分别为 99.9161% 和 99.9435%，无裁剪。因此 gate 失败不是缺失更新、NaN 或稀疏断梯度造成的。

## 决定性现象：第一步过冲

`alpha=.99` 距 teacher `xT` 的初始 L2 距离约为 `2.56`。Adam 第一步的 `xT` 更新范数为 `12.1878`，是剩余距离的约 `4.76×`；endpoint loss 随即从 `1.35312e-4` 跳到 `3.93658e-3`，放大约 `29.09×`。后续 255 步虽把过冲后的误差降到 `5.90490e-4`，仍从未优于 step 0。

其原因来自 Adam 首步近似按每个坐标的梯度符号更新：65536 维 `xT` 在 LR=`0.05` 时，理论首步 L2 量级约为 `sqrt(65536)×0.05=12.8`，与实测 `12.19` 一致；而 1% warm start 到精确解只剩 L2≈`2.56`。所以这里“梯度小”并不会自动令 Adam 首步小，当前 LR 与局部距离尺度相差约一个数量级。

`alpha=.50` 到精确解距离约 128，首步更新约 12 的量级，不会立刻越过 teacher，但仍只进入局部平台；它没有否定首步尺度诊断，反而说明固定 LR 无法同时适配远、近两种距离。

## 下一项最小实验

继续暂缓 LoRA、全量数据与 ID/OOD。固定复用同一 target，仅保留 `alpha=.99`，把唯一自变量改为 Adam base LR：

- `0.005`：理论首步 L2≈1.28，约为剩余距离的 50%；
- `0.001`：理论首步 L2≈0.256，约为剩余距离的 10%。

两项仍运行 256 步 cosine 调度、同一 loss/checkpoint/gate、无裁剪且独立重置 optimizer。若低 LR 通过，即确认主故障是优化尺度；若两者仍失败，再检查 Adam 符号归一化、局部有限差分/Jacobian 与残差参数化，而不是盲目增加训练数据。

## 交付内容

- `preflight/`、`formal/`：结果、终态、两条件完整 metrics、日志、环境、manifest、模型快照校验及 inventory。
- `images/`、`figures/`：固定 target、两个条件的 0/64/128/192/256 图像、loss/首步过冲/潜空间距离图。
- `derived/summary.json`：从 raw checkpoint 与 metrics 重新计算的结构化结果。
- `raw/r11-new-basin-7d4a845-20260907-round01.tar.gz`：全部 target、checkpoint 与运行产物；SHA-256 `5134e63f4cc2574a8d553341cc536052038727dd1b117cec64506b7165a37b4e`。
- `render_results.py`：安全校验 tar、临时物化后调用项目审计器、重建选定证据/图表/摘要与交付清单。
