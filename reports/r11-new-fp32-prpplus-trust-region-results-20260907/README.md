# R11_new FP32 PRP+ trust-region：正式结果交付

## 核心结论

PRP+ **显著缓解但没有解决**固定目标上的一阶优化长尾。它在第 16 次接受更新进入 strong capture，第 35 次已优于父方法 128 步终值；最终 64 次更新后的 loss ratio 为 `0.02859119`，比父方法 128 步的 `0.05279661` 低 **45.85%**。但它仍是固定成功阈值 `0.01` 的 **2.86 倍**，第 65 次尝试的五个固定半径候选全部变差，因此按预注册规则停止。

正式分类是 `prpplus_material_improvement_only`，不是 `fixed_target_trust_region_success`，更不是 Picture Memory 成功。下一项预注册判别实验只能是在同一 target、精度、loss、半径、接受规则与门槛下测试一次 L-BFGS limited-memory inverse-curvature 方向。

## 实验口径与结果

| 项目 | Horizon-128 父方法 | PRP+ 本轮 |
| --- | ---: | ---: |
| 搜索方向 | 单位负梯度 | 第 1 步相同；之后为 PRP+ 共轭方向 |
| 最大预算 | 128 更新 / 771 前向 | 128 更新 / 771 前向 |
| 实际执行 | 128 更新 / 771 前向 | 64 更新、65 次尝试 / 393 前向 |
| 初始 loss | `1.7977445e-5` | `1.7977445e-5` |
| 最终 loss | `9.4914816e-7` | `5.1399650e-7` |
| 最终 ratio | `0.05279661` | `0.02859119` |
| 停止原因 | 达到 128 步上限 | 第 65 次无可接受候选 |
| `ratio <= 0.01` | 否 | 否 |

本轮唯一科学变量是搜索方向。target、plateau、DreamLite 快照、FP32 lift、endpoint MSE、五个半径 `(0.1, 0.03, 0.01, 0.003, 0.001)`、候选选择与接受规则、Reader 禁用和所有解释边界均固定。首步梯度、方向、五个候选及选择与父实验逐字段精确一致。

## 观察与解释

- 64 次被接受的 loss 全部严格单调下降；正式链路为 393 次完整前向、65 次反向、0 optimizer step、0 gradient clipping、0 Reader forward。
- 相同第 64 更新处，PRP+ loss 比父方法低约 **55.59%**，证明历史曲率/方向信息确实缓解了最速下降的之字形长尾。
- 65 次尝试中有 56 次正 beta、8 次负 beta 按预注册规则截断重启；方向始终为下降方向。最佳半径计数为 `0.03×13、0.01×11、0.003×25、0.001×16`。
- 最终失败不是断梯度：梯度仍为全稠密；而是当前 PRP+ 方向上连最小半径 `0.001` 都使 loss 上升 `0.603%`。这说明单一共轭递推仍不能稳定表达局部逆曲率，支持下一步测试有限记忆 L-BFGS，而不是继续增加相同 PRP+ 预算。
- 解码图像仍近似人类不可读的灰色纹理；它只能展示 latent 轨迹发生变化，不能替代 loss、Reader、因果替图或 ID/OOD 证据。

## 审计与边界

- 技术预检：27 个产物、1/1 更新接受；正式：153 个产物、65 次尝试/64 次接受。两者 inventory、配置、快照、张量递推、候选、checkpoint、状态链与最终 bitwise replay 均通过独立审计。
- 外部父实验审计 SHA-256 为 `e43335d1536510674dcb402f11b330407e29cbd429128f7f27a7b4ac39a63695`；完整原始归档为 384,333,897 bytes，SHA-256 `e6abd466e26302a052ce43df1f7a3c326c8620aa76151df4c38540bc1eb68015`。
- 一次未带 `OMP_NUM_THREADS=1` 的额外审计在第 2 步 beta 的最后约 `4e-17` 出现并行归约差异，但保存的 raw direction 与 unit direction最大差严格为 `0`。使用预注册完整环境后，正式与外部审计均通过；该诊断不改变数据或分类。
- 本轮仍是单个 oracle target：没有 event→state 共享 writer、Reader、SET/overwrite、多 target、多 seed、ID/OOD 或因果替图评测。因此 `formal_success=false`、`phase2_allowed=false`。

## 交付内容

- `raw/`：9 个逐片 SHA 锁定的完整原始归档分片及重组清单；包含全部日志、配置、环境、指标、张量、checkpoint、图片和审计文件。
- `formal/`、`preflight/`：便于直接阅读的关键原始文件和代表性图片。
- `derived/`：独立重算摘要、逐轮 CSV、matched-update 对照表与本地首步审计。
- `figures/`：loss 对照、递推动力学、matched-update 比较和图片轨迹。
- `render_results.py`：从 9 个原始分片重组、核验完整包、审计 preflight/formal/父实验、重算表格和图表，并生成 `DELIVERY_MANIFEST.json`。

复核命令（需先运行父 Horizon-128 交付脚本，使其完整父归档在本地重组）：

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python reports/r11-new-fp32-prpplus-trust-region-results-20260907/render_results.py
```
