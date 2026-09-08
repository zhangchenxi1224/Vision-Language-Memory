# 新 EOS Oracle → 成功 latent 集合 → DreamLite U-Net

本文件定义两条 Oracle 路线完成后自动执行的同一个 U-Net 学习入口。**代码已经实现不等于 GPU 实验已经完成**；实际结果必须由运行目录的 `terminal.json`、`result.json`、原始生成记录与模型 checkpoint 验证。

## 目标与可训练对象

Direct 路线只直接优化 final latent；Frozen 路线只优化固定 DreamLite 的输入 xT。两条路线都必须是本轮重新训练的填空答案序列 + EOS 协议。银行只接收完整 256 步且 original_open 在原始 greedy / 32 tokens 下通过 Exact Match 的终点。

两条 bank 分开训练全新的 U-Net LoRA，使用相同固定预算：512 optimizer steps、AdamW、LR=1e-4、rank=4、alpha=4、dropout=0、weight_decay=0、梯度裁剪 1。LoRA 注入仓库既有的 `to_q/to_k/to_v/to_out.0` 模块，实际更新 U-Net A/B 参数。VAE、Reader、condition encoder、U-Net 基础权重冻结。DreamLite FP32，Reader BF16。

相同总 512 步不等于相同每题曝光：Direct 当前只有 ambient，而 Frozen 最多 8 题。最终明确输出逐题、逐 teacher 的实际更新次数；两条路线 EM 的差异不能直接归因于 bank 的可达性或质量。后续可另立每题预算对齐对照，本轮不根据成功数动态改预算。

U-Net 输入只有真实事件文本、固定 source latent、正常随机噪声和时间；不输入问题、答案、题目 ID、teacher ID 或目标 latent。事件本身的语义内容必须保留，例如“音乐偏好变为 ambient”；这与把评测答案额外拼到模型输入不同。

更精确地说，训练时成功 target 参与构造插值状态与速度监督；不把 target 另作条件送入 U-Net。推理时没有 target，只使用正常 source、事件和新噪声。

## 集合监督与真实部署起点一致

源码核验依据是 `src/vision_memory/dreamlite/differentiable_mobile.py` 的真实 sampler，而不是假定 Diffusers API：U-Net 沿宽度接收 `[current_latent, source_latent]`，读取左半幅 velocity；Euler 随 sigma 递减，实际推理四步为 0.5、0.375、0.25、0.125 → 0。内部文本使用 `encode_latent_path_condition()` 的官方 edit 模板。

设 source 为 s，正常噪声 ε ~ N(0,I)，bank 中随机抽取的正确目标为 z⁺。部署的起点为 y=0.5s+0.5ε。我们训练连接 y 与 z⁺ 的条件流：

```
x_sigma = z⁺ + (sigma / 0.5) * (y - z⁺)
v_target = (y - z⁺) / 0.5
L_FM = mean((U-Net(x_sigma, s, event, sigma) - v_target)^2)
```

sigma 从 (0,0.5] 均匀采样。sigma=0 恰为成功目标；sigma=0.5 恰为实际部署起点。Euler 更新 `x_next=x+(sigma_next-sigma)*v` 的方向已通过 CPU 测试，运行时还核对真实 scheduler 时间单位。

不能把标准纯噪声 flow 的 sigma=0.5 插值直接照搬：它的中间状态可能是 `0.5*z⁺+0.5*ε`，不等于当前真实部署的 `0.5*source+0.5*ε`。本桥接消除了这个起点不一致。

每个完整采样周期均衡访问题目；题内均匀抽取一个成功成员作为 target，**不先求多个 latent 的均值**。每题 ≥5 个不同成功成员时，用稳定 hash 留约 20% 成员作未训练目标距离诊断，其余成员参与训练。少于 5 个时全用，但明确没有目标留出。每题被采到同一 sigma 附近时可能仍有不同速度监督，这是条件 flow matching 的分布回归；不能仅凭此保证有限模型、512 步训练和四步离散积分恢复所有模式。

**L_FM 是向量场损失，不是问答损失。** 本版不会暗中添加 QA loss、改变训练步数，或以低 MSE 宣称 Reader 可读。U-Net 四步生成的新图是否回答正确，必须独立测量。

## 配对的真实 Writer 评测

在训练前的原始预训练 U-Net 与训练后的 U-Net 上，使用相同的 8 个新噪声 seed；这些噪声与训练噪声采用不同命名空间，且没有优化过。每张图跑 original_open + 四种只修改提问句的改写，固定后两行逐字为：

```
Use the memory image to answer.
Answer with a short phrase only.
```

全部是 Reader 原始 greedy、max_new_tokens=32、原模型 EOS；不裁掉多余单词。保存原始 token、完整输出、答案 token prefix、EM、over-generation、answer CE、EOS CE。blank 和 different-answer donor 每题每问法各评一次，不把完全相同的对照复制八次扩大分母。单题 Direct 的 donor 使用绑定 hash 的历史 orange RGB，仅作为对照，不作为监督。

几何报告新输出到所有 bank 成员的最小 RMS 距离、最近成员计数/熵、生成样本两两距离，以及覆盖半径内的 bank 比例。半径固定为该 bank 两两距离中位数的 0.25 倍；报告其定义、数值及有限样本限制。最近成员被访问并不证明该成员所在模式已被复原，因此同时报告实际距离与半径覆盖。另报留出目标的覆盖率。不自动把坐标分组叫作语义 cluster，不以有限样本 PCA 宣称真实低维流形。

Direct 当前 ambient 单题最多支持“同题 Writer 集合学习机制”；Frozen bank 即使包含多题，也只报告训练涉及题目的表现，不能宣称未见题泛化。某题 Oracle 零成功时，该题明确列为 bank 覆盖缺失并从 U-Net 训练中排除，不能计成 U-Net 学会。

## 可恢复与审计

入口：

```sh
python scripts/train/train_latent_bank_unet.py \
  --bank-manifest /absolute/route-bank/manifest.json \
  --output-dir /absolute/route-unet \
  --dreamlite /absolute/DreamLite-mobile \
  --reader-model /absolute/Qwen3-VL-4B-Instruct \
  --dreamlite-device cuda:0 --reader-device cuda:1 \
  --expected-commit EXACT_COMMIT --steps 512 --seed 20260908
```

`--validate-only` 只验证 bank/张量/模板，无 CUDA。恢复使用 `--resume`；可加 `--deadline-unix` 在实例回收前保存并退出。每次更新原子保存 `checkpoint-latest.pt`，含 LoRA、optimizer、所有 RNG 状态、当前 cursor、完整 source/data-bank/model/config 绑定与 metrics row。最终另外保存 `checkpoint-final.pt`。baseline 和 final generation 保存独立完成标记及原始文件 SHA；中断评测可确定性重放，不把半次评测当完整结果。

源码必须是指定 clean commit，bank manifest/每个 FP32 target 都验证 SHA，实际模型完整 snapshot 在运行前后核验，冻结参数还检查内存版本与梯度。源 blank latent 与当前 FP32 VAE 编码逐位一致才允许训练。失败写 `failure.json`；资源截止/信号写 `terminal.json: paused`，退出码 75；完成写 `terminal.json: completed`。完成只意味着约定训练与评测执行完毕，不能替代 QA 质量判断。

本地验证：13 项 CPU 测试通过，覆盖桥接边界/方向、真实 sampler 的 U-Net 拼接与 timestep 参数/梯度、均衡成员采样、模式坍缩诊断、MCQ/no-EOS/模板/哈希污染拒绝、optimizer/RNG 精确恢复、评测分母和前后配对的完整性。另 5 项 Direct 几何回归测试通过。未安装的本地 Diffusers/PEFT 与实际 GPU 数值不伪造为已测试；需要部署环境进行真实模型运行验证。
