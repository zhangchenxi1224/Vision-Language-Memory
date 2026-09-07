# R11_new 固定可达目标低学习率对照结果

## 核心结论

**降低学习率修复了上一轮的第一步灾难性过冲，但 `lr=0.005` 与 `0.001` 都停在几乎相同的误差地板，未通过预注册门槛。** 工程 gate、远端独立 audit 与下载后本地独立 audit 均通过；本轮分类为 `learning_rate_reduction_insufficient`，Picture Memory `formal_success=false`、Phase 2 `false`。

这说明此前 `lr=0.05` 的确过大，但它只是一个根因，不是全部根因。完整链路可以回传且精确解真实存在；当前剩余问题已收窄到冻结 DreamLite 映射的局部几何/数值精度与优化器动态。

## 固定设计与工程完整性

- 精确复用 parent basin 的 target artifact，SHA-256 `5352fac1cf614c8458d4f61b6ffcbf7b053f8a947dedac1286f80261ba8a772a`；teacher endpoint 可由 teacher `xT` 逐位重放。
- 两组都从完全相同的 `source + 0.99 × (teacher-source)` 开始；preflight 证实初始 `xT`、endpoint、M0、完整 gradient tensor SHA-256 均一致。
- 唯一自变量是 Adam base LR：`0.005` 与 `0.001`。两组各自新建空 optimizer state；每组 256 步，后 128 步使用同形 cosine 衰减到 0；无裁剪。
- loss、checkpoint 0/64/128/192/256、raw256 主 endpoint 与三重 gate 均未改变；禁止选择 best checkpoint。
- 总计 523 次完整 DreamLite forward、512 次 backward、512 次 optimizer step、0 次 Reader forward；模型快照前后一致。
- 正式运行 commit `af00d0418f4ead501a27bb1cab03b5cd9e3bc51a`，用时 533.84 秒；preflight 24 项、formal 40 项 inventory 全部通过独立审计。
- round01 曾因误用系统 Python、缺少 `diffusers` 在模型 forward 前技术失败；失败目录与日志完整保留。随后只固定到上一轮已验证的 Python 环境，科学变量未变。

## 精确结果

| 条件 | M0 MSE | raw256 MSE | MSE/M0 | L2/initial L2 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `lr=0.005` | 1.3531206e-4 | 1.6681601e-5 | 0.123282 | 0.351116 | false |
| `lr=0.001` | 1.3531206e-4 | 1.6364127e-5 | 0.120936 | 0.347759 | false |

门槛固定为 raw256 MSE `<=0.001`、MSE/M0 `<=0.01`、L2/initial L2 `<=0.1`，三项必须同时满足。两组绝对 MSE 都低于 `0.001`，但只消除了约 88% 的初始 MSE，而门槛要求至少消除 99%，因此不能算通过。

## 第一性原理分析

1. **首步尺度诊断得到验证。** `lr=.005` 首步 `xT` L2 更新实测 `1.21878`，是剩余精确距离的 `0.47609`；`lr=.001` 实测 `0.243756`，比值 `0.09522`，均与上一轮线性预测一致。两者第一步 loss 分别下降到初始的 `0.5455` 与 `0.5962`，不再发生过冲。
2. **但最终误差几乎不依赖 LR。** 两个相差 5 倍的 LR 最终 MSE 只相差约 1.9%，都停在 `1.6e-5`。这排除了“只需继续把 LR 调小”作为充分解释。
3. **链路没有断、梯度也没有被裁掉。** 两组 256 步均有有限非零梯度，最低非零元素比例约 99.94%，最低梯度范数仍约 `1.26e-4/1.34e-4`；不存在 NaN、漏 step 或 clipping。
4. **`xT` 距离与 endpoint loss 明显解耦。** `lr=.001` 把到 teacher `xT` 的距离从 2.56 降到 2.05；`lr=.005` 却增到 4.90，但两者 endpoint MSE 几乎一样。这表明冻结映射存在大量低敏感/多对一方向，直接用 `xT` 欧氏距离无法代表 endpoint 可辨识误差。
5. **当前最可能是局部非光滑/量化与 Adam 动态共同造成的地板。** DreamLite 主干以 BF16 运行，而学生 `xT` 是 FP32；接近目标后极小参数变化可能跨过 BF16 激活/舍入边界，使真实有限差分与反向梯度、Adam 动量方向不再稳定一致。精确 teacher `xT` 仍能得到零误差，所以这是“优化能否找到解”的问题，不是“解不存在”。

## 下一项最小判别实验

暂不增加数据、LoRA 或全量训练。固定同一 target，在 alpha=.99 起点与本轮 plateau checkpoint 上做局部方向/Jacobian fidelity scan：比较归一化负梯度方向、Adam sign/momentum 方向、已知 teacher residual 方向及确定性正交对照，在预先锁定的多个 L2 半径上测真实 forward loss 与一阶预测。

- 若负梯度存在稳定下降半径而 Adam 方向失败：改用尺度可控的 SGD/L-BFGS 或 trust-region；
- 若 autograd 方向与真实有限差分不符、但 teacher residual 可下降：优先修复 BF16/Jacobian 精度或重参数化；
- 若只有极窄 teacher residual 邻域可下降：说明吸引域/可辨识子空间极窄，应采用 continuation，而不是盲目延长训练。

## 交付内容

- `preflight/`、`formal/`：结果、终态、完整 metrics、日志、环境、manifest、模型快照校验及 inventory。
- `images/`、`figures/`：固定 target、两组 0/64/128/192/256 图像、loss/相对门槛/首步尺度/潜空间距离图。
- `derived/summary.json`：从 raw checkpoint 与 metrics 独立重算的结构化结果。
- `technical-failure-round01/`：缺失依赖技术失败的终态、日志与证据。
- `raw/r11-new-lr-af00d04-20260907-round02.tar.gz`：成功 preflight 与 formal 的全部 checkpoint/产物，SHA-256 `9bfe553c8aa7a8eef8a69dcc19ef9d37fa182c7766548dc81b5df39737acfcd1`。
- `raw/r11-new-lr-88a0b48-20260907-round01-technical-failure.tar.gz`：round01 技术失败全部产物，SHA-256 `89d893798d96fe9d338f83cc5fdc75cc2fb1aff9ec11b18b1ec3529e889f770f`。
- `render_results.py`：安全验证归档、重新调用项目审计器，并可重建证据、图表、摘要和交付清单。
