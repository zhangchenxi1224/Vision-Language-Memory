# R11 Open Answer：停止监督、真实生成与多词答案协议

本协议于新 EOS 训练结果出现前锁定。它属于旧 R11 VAE latent oracle，**不执行 DreamLite U-Net**，不得混入 Frozen DreamLite geometry 的结果。

研究问题是：旧 Open 优化是否已经让模型预测正确答案前缀，但因为没有学习 assistant 回合终止而继续生成；补充终止监督能否改善完整自由生成，而不仅是 teacher-forced loss？

## P0：正确终止标签

从当前 Reader 的真实 chat template 渲染 user→assistant 完整回合，取答案后第一个特殊 token，并验证它属于 model generation_config 的停止集合。不能默认 tokenizer.eos_token 是唯一停止规则。

2026-09-08 只读检查原锁定 checkpoint：generation_config 停止 ID 为 `[151645,151643]`；tokenizer EOS 为 `<|im_end|>`。正式执行还必须用实际模板验证上述关系并保存原始尾部、token IDs 和选择原因，不能只相信此文字。

保留 A 的原始 `L_answer = mean(answer token CE)`。B/C 使用 `L_answer + 1.0 × L_end_of_turn`，答案部分仍取 mean，EOS 单独权重；不能把所有 token 一起平均，以免长答案稀释 EOS 权重。

## P1：先回放旧轨迹

默认读取旧 Open 的 8 个 seed 在 `0,16,32,64,128,192,256` 的已有原始 latent。验证 index/file/tensor SHA，保持 VAE、Reader、BF16、resize、chat prompt、greedy 32 tokens 和原 EOS 不变。支持额外回放旧 MCQ 轨迹，但它不是新增训练。

每点记录 original、原 paraphrase、预注册 new rewrite 的原始生成文本、完整 tokens、答案前缀正确率、teacher-forced answer token accuracy、EOS accuracy、CE、normalized exact match、额外生成、停止原因。末步加 blank 与固定 donor；不覆盖旧结果，不凭前缀正确宣称完整生成正确。

## P2：严格配对 A/B/C

同一固定旧 target 1，gold `ambient`，8 个原 seed。初始化从原灰图 VAE reference、原 CPU FP32 噪声和 FP64 RMS 公式重新生成，必须与旧 step0 逐位一致。

| Arm | 训练目标 | 训练问题 |
|---|---|---|
| A | 原 answer token mean CE，无 EOS | 仅 original |
| B | answer mean CE + EOS CE，lambda=1 | 仅 original |
| C | 同 B | original/paraphrase 交替，128次各一半 |

全部 Adam 256 步，lr=0.05，betas=(0.9,0.999)，eps=1e-8，无权重衰减、无裁剪、无 early stop。每个 arm 重置同一随机状态和全新优化器；VAE/Reader 冻结，只优化 latent_fp32。A 在所有指定 checkpoint 与旧 Open tensor 强制逐位比对，失败就停止比较，不能事后把不一致算作复现。

末步 256 是唯一主终点；保存中间 checkpoint 是观察过程，不允许挑 best。C 训练过的原 paraphrase 明确标作 exposed；新 rewrite 从未训练，不用于模型选择，但它仍是同一道题的改写，不等价于未见任务泛化。

## P3：部署输出与科研指标分开

raw 科研输出固定 greedy 32 tokens、原 EOS，完整字符串评分。额外的部署输出单独保存：task schema 声明 short_word 时最大 4 tokens、short_phrase 时 8 tokens；当前问题既可出现多词偏好又可撤销为 no active preference，因此 schema 统一 short_phrase，不能根据当前 gold 长度改变预算。

部署同时接受原模型 EOS/end-of-turn 或 newline 停止；不在第一个空格处截断。规范化采用 casefold、空白规整和末尾 Unicode 标点处理；`ambient trance` 仍不能被裁成 `ambient`，`light blue`/`orange juice` 完整保留。部署 EM 不得替代 raw 科研 EM。

## P4：多词与字段契约审计

使用真实 tokenizer 在实际 assistant 左上下文记录 `green`、`juice`、`ambient`、`light blue`、`orange juice`、`no active preference`，另含 food 示例；覆盖 color/drink/music/material/food。不能凭英语词数假定 tokenizer token 数。

此轮 P4 产物是 tokenization 和指标契约审计；只有固定 anchor 真正训练。跨字段训练泛化必须另锁真实数据样本、重新执行，不能把这些文本示例标作训练成功。

## P5：指标与边界

同时报告 generated answer-token accuracy、teacher-forced answer-token accuracy、完整 raw EM、overgeneration、paraphrase consistency，以及 matched-minus-blank / matched-minus-fixed-donor。多 checkpoint、prompt、image condition 是同 seed 重复测量，不能放大独立样本数。固定 donor 是同一个旧 target02 图像，只作为图像损坏对照，不是假定具有反事实 gold 的任务。

仅当固定 original 的 64→128→256 raw EM 严格持续提升且固定 answer CE 持续下降时，才记录“可以讨论 512”；本次实现不自动增加步数。单题 oracle 结果不得宣称共享 Writer、held-out task 或正式 Picture Memory 成功。

## 运行

在分配给本任务的两张 H200 上运行，建议先 `audit`，再 `replay`，之后 `train`。`CUDA_VISIBLE_DEVICES=2,3` 可将整台 4GPU 作业的后两张卡映射为本进程 cuda:0/1。

```bash
bash scripts/inspire/launch_r11_open_eos.sh replay /inspire/ssd/.../eos-replay-01
bash scripts/inspire/launch_r11_open_eos.sh train /inspire/ssd/.../eos-train-01
```

支持 `--seeds 0` 或 `--arms A B` 用于 smoke/preflight；manifest 保存实际子集，子集不得声称完整 8seed 设计。所有输出目录必须新建。原模型路径、环境、配置、parent manifest SHA 和运行 commit 均写入 manifest。
