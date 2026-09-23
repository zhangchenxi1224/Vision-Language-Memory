# FM 日志与输入条件：读出完成前的定位记录

2026-09-24。以下只读分析使用全部 64 条训练教师和已经完成的 FM 日志。
未增加训练、未使用 OOD 选择参数，也未宣称学生图已成功或最终失败。

## 1. FM 有数值学习，但不能把误差下降等同于视觉记忆

对训练桥 `x=(1-sigma)z+sigma*epsilon`，有
`z_hat=x-sigma*v_hat`，因此相对于该目标的重建 MSE 等于 `sigma² × velocity MSE`。
这是带真实目标加噪的训练输入上的代数换算，不是从纯噪声运行 28 步后的生成误差。

| 指标 | A | B |
|---|---:|---:|
| 全部教师相对银行均值的 latent MSE | 0.423418 | 0.139687 |
| 前256更新、sigma∈[.75,1) 的重建 MSE | 0.450402 | 0.150463 |
| 后256更新、同一区间的重建 MSE | 0.029207 | 0.011221 |

两组误差都明显下降。但后256更新的输入仍含有真实目标成分，不能据此证明纯噪声起点下
已学会写入；均值误差也只是几何参照，没有实际评估一个无条件均值 Writer。
窗口内抽样不同，所有统计和分母保存在 `fm-training-geometry.json`。
不能仅凭这些数字断言“模型只输出平均图”，也不能断言剩余误差足够小、不会影响 Reader。

## 2. 当前训练内容读出并不是相同训练条件的精确回放

逐例比较全部64条输入，结果如下：

| Writer 初始交换的组成 | SFT训练与benchmark相同数 |
|---|---:|
| 用户披露偏好 | 64/64 |
| 助手 acknowledgment | 0/64 |
| 完整初始交换 `writer_event(record,0)` | 0/64 |

这是既定官方流程的区别：训练使用发布的 SFT acknowledgment，benchmark 使用被测
冻结 Reader 生成的 acknowledgment。保留这种区别；不能把 SFT 确认回复放进正式
benchmark 来提高成绩。逐例比较见 `writer-input-condition-comparison.json`。

因此，当前64条“train-content student”测的是**已见偏好在不同确认回复下的生成**，
并非完全相同输入的训练拟合。若这些学生图弱于教师图，可能同时包含 FM 生成误差、
教师表示对扰动敏感、以及 Writer 对完整交换的泛化误差。现有证据尚不能单独归因。

## 3. 完成固定矩阵后的最小定位顺序

先保存当前两组完整结果，再在全部64条训练内容上，用原始 SFT 初始交换、相同
seed0 和原生28步生成诊断 PNG。与现有 benchmark seed0 图配对，继续使用相同 Reader、
官方题目及评分，明确标记为训练条件诊断，不能混入正式benchmark成绩。

若原始训练条件可读而 benchmark 条件失效，优先定位输入泛化；若两者都不可读，
再定位纯噪声生成的 FM 蒸馏和 Teacher→RGB 的读出裕量。此时应先修复单写，不能
靠增加保持轮次掩盖单写问题。若单写可用，再按既定2048步保持阶段及后续 K2/K4 推进。
不同时改变损失、Reader、官方采样过程和数据；不根据本轮 OOD 分数决定纠正路线。

## 复查入口

- `scripts/reporting/diagnose_prefeval_fm_geometry.py`：在指定 GPU 实例中以 CPU 张量运行，
  不加载 Writer/Reader 或分配 CUDA；CPU辅助实例缺少 torch，因此未在其中安装依赖。
- `scripts/reporting/compare_prefeval_writer_conditions.py`：从已归档历史复算输入比较，不读评分。
- 两组 Writer 权重已上传实验 Release，GitHub SHA256 与训练端相同。
  共享 Writer 的完整评测、自由回答 judge 和真实递归阶段仍未完成。
