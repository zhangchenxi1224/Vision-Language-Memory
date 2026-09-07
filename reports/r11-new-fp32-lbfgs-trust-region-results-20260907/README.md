# R11_new FP32 L-BFGS trust-region：正式结果交付

## 核心结论

L-BFGS 在**完全固定的单 target oracle 口径**下真实跨过了预注册阈值：第 `115` 次接受更新后，loss ratio 降至 `0.0099389207`，分类为 `fixed_target_trust_region_success`。相对 PRP+ 最终 ratio 降低 `65.24%`，相对 Horizon-128 最终值降低 `81.18%`。

这证明固定目标位于当前 DreamLite 参数化的可达域内，之前长尾主要是搜索方向/局部曲率问题，而不是断梯度或绝对不可达。但这**仍不是 Picture Memory 正式成功**：本轮没有训练 event→state 共享 writer，没有 Reader、multi-target、multi-seed、ID/OOD、SET/overwrite 或因果替图评测。因此 `formal_success=false`、`phase2_allowed=false` 保持不变。

## 实验口径与结果

| 项目 | Horizon-128 | PRP+ | L-BFGS 本轮 |
| --- | ---: | ---: | ---: |
| 搜索方向 | 单位负梯度 | PRP+ 共轭方向 | L-BFGS two-loop，m=10 |
| 最大预算 | 128 更新 / 771 前向 | 128 更新 / 771 前向 | 128 更新 / 771 前向 |
| 实际执行 | 128 更新 / 771 前向 | 64 更新 / 393 前向 | `115` 更新 / `693` 前向 |
| 最终 loss | `9.4914816e-7` | `5.1399650e-7` | `1.7867640167e-07` |
| 最终 ratio | `0.05279661` | `0.02859119` | `0.0099389207` |
| 停止原因 | 最大更新数 | 无可接受候选 | `success_threshold` |
| ratio ≤ 0.01 | 否 | 否 | **是** |

唯一科学变量是首步后的搜索方向。target、原始 plateau、DreamLite 快照、FP32 map、endpoint MSE、五个半径 `(0.1, 0.03, 0.01, 0.003, 0.001)`、候选选择、`1e-6` 接受规则、128-update/771-forward 上限及 0.01 门槛均未改变。没有 optimizer、gradient clipping 或 Reader forward。

## 关键观察

- `115/115` 次更新被接受，loss 全程严格单调；`114` 个后续曲率对全部通过门控，历史最大长度 `10`，除首步初始化外无重启。
- 首次进入 ratio≤0.1 在 update `12`；首次达到 ratio≤0.01 在 update `115`。
- 最终 ratio 是门槛的 `0.9939` 倍，属于刚跨线而非大裕量成功；下一阶段需要多 target 与多 seed 验证，不能把单点结果外推。
- 解码图像依旧近似人类不可读的灰色纹理。图像只能展示同一 latent 优化轨迹，不能替代 Reader 正确率、因果替图或泛化证据。
- 正式运行期间 Inspire 控制 WebSocket 曾断开，但后台迭代继续增长，最终终态、运行锁释放和 255 个产物独立审计均正常；不存在重复启动或残缺结果。

## 测试、审计与完整性

- 远端锁定环境全量测试：`1413 passed / 5 skipped / 0 failed`，另有 `88` subtests passed；跳过项仅为节点无指定 TrueType 字体。
- 技术预检审计：27 个产物，1/1 更新接受；正式审计：255 个产物，分类、曲率历史、two-loop、候选、checkpoint、inventory、模型快照与最终 bitwise replay 全部通过。
- Windows 交付脚本可移植地重算 inventory、JSONL 候选选择、接受规则、状态链和分类，并校验远端权威审计 SHA；逐张量浮点重放必须使用锁定 Linux/PyTorch 环境，避免把跨平台归约差异伪装成 bitwise 一致。
- 完整原始归档：`702895016` bytes，SHA-256 `64acb0c4bc16b8aa5ad0fc4eafcb064c11b84f12c031e3129cb377856f5ef005`；15 个 GitHub 分片各自有 size/SHA 绑定。
- 运行 commit：`f995248e5d730a478d2398a49c4e533fd05aa2ae`。其中 `9196f9a…` 冻结 L-BFGS 方案，`f995248…` 仅修复旧 R3 DAG 对 Linux venv 符号链接的等价路径校验，发生在首次 DreamLite forward 之前，不改变科学口径。

## 下一步主线

停止继续微调单 target 局部优化器。下一最小判别实验应把本轮证明可达的 oracle latent 当作教师信号，预注册并训练**一个跨多个事件/目标共享、推理时非 oracle 的 writer**；先检验 held-in multi-target 写入能力，再接冻结 Reader 与因果替图，最后扩展到 multi-seed、ID/OOD、SET/overwrite。只有这条链路通过，才可讨论 Picture Memory 成功。

## 交付内容

- `raw/`：15 个完整原始归档分片及重组清单。
- `preflight/`、`formal/`：配置、环境、日志、指标、审计 inventory 与代表性图片。
- `audits/`、`readiness/`：两份独立审计和远端全量测试 JUnit。
- `derived/`：重算摘要、逐轮 CSV 与同 update 对照表。
- `figures/`：loss 对照、L-BFGS 动力学、matched-update、最终值与图像轨迹。
- `render_results.py`：从分片重组并重新验证上述全部交付。

复核命令：

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python reports/r11-new-fp32-lbfgs-trust-region-results-20260907/render_results.py
```
