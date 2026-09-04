# R12 共享事件到视觉 latent 写入器：双臂结果

**判定：** `diagnose_shared_writer_fit_boundary`

## 固定因果门槛

| 实验臂 | train-audit | dev-select | sealed dev-final | 总门槛 |
| --- | ---: | ---: | ---: | --- |
| conditioned | 0/36 | 0/24 | 0/24 | 未通过 |
| constant-control | 0/36 | 0/24 | 0/24 | 未通过 |

## 核心事实

conditioned 的平均 CE（M0→normal→donor）分别为：train-audit 23.949→1.141→1.141；dev-select 25.008→1.359→1.349；dev-final 24.358→1.368→1.373。
constant-control 的 normal CE 为：train-audit 1.376、dev-select 1.431、dev-final 1.422。

## 第一性原理归因

冻结事件表征仍能预测目标值：线性探针在 train-audit/dev-select/dev-final 上为 100.0%/83.3%/95.8%。
经过写入器系数头后降为 72.2%/20.8%/20.8%。事件特异信息主要在表征到视觉码的映射阶段丢失，而不是冻结事件编码器中不存在。

R12 仅是单步 SET 诊断，不能作为完整 Picture Memory 成功。下一轮必须保持现有 normal/reset/donor 门槛，并结构性消除事件无关通用视觉码捷径。
