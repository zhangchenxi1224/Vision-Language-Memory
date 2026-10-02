# 官方对齐边界与证据准入

2026-10-02。用户确认以 DreamLite / PrefEval 对齐作为当前主线边界。本文件取代“全部旧实验分支作为同等有效结果汇入main”的整理方案。

## DreamLite：两个独立修复

`c7e752b` 恢复官方 target–noise flow：`x=(1-sigma)y+sigma*epsilon`、`v=epsilon-y`，source只作条件，并恢复完整sigma域、纯噪声初态与实际scheduler。旧桥 `x=(1-2sigma)y+sigma*s+sigma*epsilon`、`v=s+epsilon-2y` 是另一种条件流，其结果不能充当官方微调结果。

`03f8467` 修复训练raw条件与Base native推理条件的差异。同预算配对开发1460/1510→1510/1510是条件修复的工程证据，不是PrefEval泛化成绩。302格条件输入一致，严格数值阈值仅296/302通过，不能称全面逐位等价。

依据：[项目审计](../reports/official-alignment-audit-20260913.md)、[原生条件控制](../reports/official-alignment-results-20260913/native-condition-development-review.md)、[数值边界](../reports/official-native-training-branch-parity-20260914.md)、[固定官方训练源码](https://github.com/ByteVisionLab/DreamLite/blob/a6e20c8cc94027f37dd7c5a81b0b3b472aa18409/lora/train_edit_lora.py)。

## PrefEval：任务与输入协议修复

`b615c27` 对齐官方dialogue、SFT输入与发布目标：seed42、820/180、完整对话和acknowledgment、`response_to_q`与最终答案mask。此前seed2026、自定义子集、复述与人工排序、独立latent优化不能作为当前任务基线。旧 `reports/prefeval-rgb-20260917/` 已撤出当前树。

`894928f` 保留“教师latent→官方FM共享Writer”两阶段路线，取代未经运行比较的全28步答案反传建议。`9e2a374` 增加官方Judge适配；替代Judge仍需单列，不等于原Claude成绩。

依据：[对齐差异](../reports/prefeval-official-alignment-20260923/ALIGNMENT.md)、[FM路线修正](../reports/prefeval-official-alignment-20260923/FM_ROUTE_CORRECTION.md)、[固定官方SFT源码](https://github.com/amazon-science/PrefEval/blob/50795054b5ff5f418d2b768a331d71e480f93331/SFT/train_sft.py)。

## 撤下与保留

| 材料 | 当前处理 |
| --- | --- |
| 旧非官方桥、旧时间/起点协议成绩及图表 | 从当前结果树撤下，不参与当前结论 |
| R系列、Direct、frozen、U-Net早期诊断 | 退出当前baseline；独立探针不一概判错，历史方法身份由Git保留 |
| 旧PrefEval自定义任务数据和结果 | 整族撤下，不与新协议合并分母 |
| 9/13–14对齐配对控制与诊断 | `engineering_provenance`，不进入当前PrefEval成绩表 |
| 当前K1、exposure512、多目标结果 | 完整保留，按run身份、完成度和对照条件分别分析 |
| 原始官方数据、当前manifest、冻结Reader/VAE | 保留 |
| 当前实验仍引用的旧bank与初始化 | 保留谱系和哈希；依赖保留不恢复旧baseline资格 |
| 独立Qwen文本baseline | 不受DreamLite训练错误直接否定；须另核对任务、划分与评分 |
| 旧名称通用代码 | 保留；当前Writer仍使用`latent_r11_vae_oracle._save_image`、`latent_bank_unet.official_flow_bridge`等 |

对齐工程目录含raw/native、旧初始化和修复后结果的混合谱系，整体退出性能数据源，仅供审计、复现依赖和工程回归。不能把其中全部记录标为双对齐性能结果。

## 对齐后的独立限制

- full-U-Net、学得的FP32 endpoint、3+2问法、K1/多目标和RGB递归属于项目适配。
- `eba0166`只修复一个官方答案的token前缀边界，既有20个目标哈希不变，不作整批失效处理。
- 官方发布目标52/64通过替代Judge属于目标一致性诊断，不是记忆准确率，不据此筛掉失败标签。
- qwen3-max与Qwen3.8评分分开；新Judge对固定输入独立重评，不改名旧成绩。
- 历史开发暴露、180重复及177去重视图须披露，不能重新包装为密封OOD。
- 对齐后的失败和负结果保留；撤下依据是协议适用性，不是分数高低。

## 测试和恢复

撤下数据对应的成员数量、问法、谱系与预算合同测试已逐项标记退休，清单保存测试名，跳过不计作通过。纯函数、梯度、PNG边界、输入隔离和当前官方协议测试继续执行。

清理仅影响隔离整理分支的当前目录。Git历史、原实验分支、原工作树和外部资产保留。按清单中的`pre_cleanup_commit:path`可恢复；本次没有重写历史或缩小Git对象库。
