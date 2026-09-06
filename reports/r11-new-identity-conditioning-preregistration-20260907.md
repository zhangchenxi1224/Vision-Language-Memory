# R11_new：固定 identity 文本的单因素 conditioning 诊断预注册

日期：2026-09-07（北京时间）。当前仅实施 condition-only 工程探针；本文在该探针编码及任何新完整 DreamLite 前向之前锁定。主计划仍为 `reports/r11-new-frozen-dreamlite-oracle-training-plan-20260904.md`，本项来自 source-only 初始化失败分支，不改变 Phase 1A/2/3 的成功门。

## 1. 从第一性原理提出的问题

冻结模型构成函数 `z = F(xT, source, condition)`。上一轮在固定 source、原 event conditioning 下，只调整 xT，仍未恢复可读 teacher endpoint。梯度存在不证明该函数的局部几何适合当前有限步优化；失败也不能证明其完整输出空间无解。

本轮只把实际送入官方条件编码器的文本，从原 event 替换成固定 `no changes`。其他变量保留 source-only 父臂。它测试的是这个特定条件干预是否改变有限预算下的 teacher endpoint 恢复，不能将该提示词当作数学恒等映射，也不能称为完全取消 conditioning。

新 prompt 不含 query、答案、teacher 或 target ID。原始 event、数据行、答案及评测仍原样保存；报告分别记录“原样本 event”和“实际 conditioning 文本”，不得覆盖前者伪造训练输入。

## 2. 固定对照与不可变证据

唯一正式比较臂是已闭合的 **source-only 初始化 + 原 event conditioning**；不得以 Gaussian 臂替代，否则同时改变了初始化与条件两因素。

| 对照证据 | 固定值 |
| --- | --- |
| 训练 commit | `e0a43c43f893ac19c6976ecd4c84ea3dc2b23006` |
| 交付 commit | `a9f9e1226f2c593e0a2c89797ba966f54871d0b9` |
| comparison SHA256 | `f7ec5913ac6dcd40860428e3c3ab929f8559752173599ffeae8cc0b01d1db09c` |
| RAW_ARTIFACTS SHA256 | `2889f183c04df70c2763010b863914e378ffb1a0c85a1d8ca0f6a22ac150d4d7` |
| formal manifest SHA256 | `02905cfa89f7cca0eebaa4d763c99264c128278deef0f12885823982fabbec35` |
| raw256 absolute MSE | `0.11177215725183487` |
| raw256 Reader mean CE | `25.765637596946004` |
| raw256 Reader | `0/4`，同一 target 的四个固定视图 |
| 工程 / teacher replay | 均通过 |
| 距离 / Reader transfer | 均失败 |

固定 target1：`r5-f1-392d41fd097d069c42218e0a`。原 Phase 1A 父 manifest SHA256：`cd5b740f1f60b32bfb3b8ccf8ba2cfe84bb4ea9c650649810c07d9d9b3972184`。不更换 target、teacher、source、模型快照、数据划分、四视图排列、优化预算或评测口径。

## 3. 唯一干预的字节定义

原 event：

```text
R3 Train Standard Templates 03: For the indigo desk train 001123, remember that the preferred music is ambient.
```

其 UTF-8 SHA256：`f170a7e2dfe0070fbd160c09d29dbcf897ddbf5f75929a3ee4af84cf627965bb`。

实际 conditioning 文本固定为 `no changes`，UTF-8 SHA256：`44f9161c3a252925f55022d60f68918abf8fdc1a9a3fadbf5ab766549f6b3461`。不试其他同义句、不按 loss 选择提示词。

完整官方模板及 UTF-8 SHA256：

```text
[Edit]: A diptych with two side-by-side images of the same scene. Compared to the right side, the left one has no changes
```

`4620b977a7aff51c9ee9a6e2901555b39ee4a3e4bd705d1402818acf330d22e6`。

调用原生 `encode_latent_path_condition`，保持官方 `mode=edit`、source VAE 解码与图像预处理、模板及 text encoder。不能清零 embeddings、删 attention mask、跳过 text encoder 或更改 source。

## 4. 先做 condition-only 工程探针

配置：`configs/experiments/r11_new_identity_condition_probe.json`。

- 文件 SHA256：`5f1db61ac1e7db179dd3cb39c463c891b3b7aba056fafbe9d6618066f4eab73e`。
- canonical JSON SHA256：`d8933ae014c1199302eec0f5cd71ee16446e0df93be22e9a501d6a8c084c0ad4`。

只加载冻结 DreamLite 基座，生成原固定 blank source、编码 source latent，然后两次执行同一原生条件编码。禁止 Reader/teacher 加载，禁止 U-Net 前向、完整 DreamLite 前向、反向传播、optimizer 创建或更新。VAE 的 source 编码与 source 解码属于条件预处理，不应误报为完整 DreamLite 前向。

技术门必须从实际 tensor 和文件复核：

1. 固定 parent manifest、原 event、source RGB 和 FP32 latent 哈希精确一致；source latent shape 为 `[1,4,128,128]`。
2. 真实输入是上述固定文本和完整 prompt；编码两次，保存两份原始 embeddings/mask，逐位相等且数值合法。
3. 新 embeddings 必须区别于旧 event embeddings SHA256 `473bd457d6fff070a71b119a19d950b8d094cfaf6f126ceb817330eb01263a60`。mask 可以与旧 SHA256 `4f941a468150ea22f64ac4f7304e9a94a3dd1c721d07dd7f8ebd10185fbe2ea9` 相同；mask 相同不表示文本干预未发生。
4. CUDA/BF16 与固定环境严格一致；不回退 CPU。DreamLite 全参数冻结且无梯度，完整模型快照前后核验不变。
5. 计数为 condition encoder 2 次，其余 U-Net/Reader/full-chain/backward/optimizer 均 0；若禁止路径被调用，立即 fail closed。

保存配置、commit、manifest、环境/runtime、原始 `source.pt`、`condition.pt`、`condition-repeat.pt`、结果、终态、日志与逐文件哈希。没有训练所以 receipts/checkpoints 标记为不适用，不能伪造零步训练记录冒充 Phase 1A。

探针只能报告工程通过/失败。成功后将实测 embeddings/mask SHA256、probe commit/root/manifest/原始文件哈希锁入另一个不可变完整 bridge 配置，完成代码、单测、独立核验并提交，**此后才能发生新完整链路前向**。这些哈希来自预定唯一 prompt 的编码，不使用 M0、teacher 距离、QA 或优化结果选择。探针失败则保存失败、定位工程原因；更换文本必须另立预注册。

只使用该次运行的第一对编码；不一致即技术失败，不反复重启挑选哈希。进程内两次一致不等于跨进程可复现，后续完整链路加载时仍须重新编码并核对锁定哈希。文本长度、mask、语义和模型先验是本输入干预的下游效应，本实验不能进一步分离它们各自的贡献。

## 5. 后续完整 bridge 的预定不变项

本段先固定科学定义，不声称完整训练实现已经就绪；正式配置需补齐探针的实测绑定，并与本段一致后才能执行。

```text
initial xT = fixed source_latents.float().clone()
loss = mean((full_four_step_endpoint.float() - fixed_teacher.float())^2)
trainable = xT FP32 only
DreamLite / VAE / conditioner / Reader = frozen
full denoising steps = 4; actual scheduler sigmas unchanged
Adam calls = 256; weight decay = 0; clipping = none
updates 1..128: lr = 0.05
updates 129..256: lr = 0.025 * (1 + cos(pi * (u - 128) / 128))
checkpoint updates = [0,64,128,192,256]
primary endpoint = raw256, not best checkpoint
seed = 0; DreamLite cuda:0 BF16; Reader cuda:1
```

保持原生 sigma / source-xT mul-add 运算次序和 CUDA 逐位初始化审计；actual sigma、256 个真实更新计数、全部 Adam state 和固定 checkpoint 等工程门沿用父臂。最后一步学习率为零仍需实际 forward/backward/Adam 调用及非零有限梯度。

必须重新测量 identity conditioning 下零更新但已完整四步前向的 M0，不能挪用 source/h0/teacher 或原 event 臂 M0。M0 MSE 必须有限且大于零，不加 epsilon。

完整链路先 technical-preflight：1 次四步前向、1 次反向、0 次更新，保存证据并过工程与 teacher replay 门。正式 bridge 再运行固定 256 次。Teacher replay 仍需本轮四视图 4/4 且 mean CE <= 0.001；Reader 不参与梯度优化。保存原始 ordered logits 并逐行复算；normal/reset 与固定四视图不变。

仅在技术门及 teacher replay 通过时，raw256 距离门同时要求：

| 指标 | 不变门槛 |
| --- | ---: |
| endpoint MSE / 本轮新 M0 MSE | <= 0.01 |
| endpoint L2 / 本轮新 M0 L2 | <= 0.1 |
| endpoint RMSE / teacher population std | <= 0.1 |
| Reader transfer（独立主门） | 四视图 4/4 |

Reader mean CE 原样报告，不新增 endpoint CE 主门。仅当技术有效、teacher replay 通过且两主门均失败，才计算二级审计：absolute endpoint MSE **严格小于** `0.11177215725183487` 且 Reader mean CE **严格小于** `25.765637596946004`。两项都要改善，不接受等值、NaN/Inf/非法值；二级不能挽救主门失败，也不是统计显著性检验。

## 6. 决策与不能声称的结论

| 实际结果 | 允许结论与下一步 |
| --- | --- |
| 探针或完整技术门失败 | 仅工程失败，先修工程，不讨论科学可达性。 |
| 距离、Reader 均通过 | 此固定 identity conditioning 下 teacher-MSE bridge 诊断通过；另行预注册恢复原 event、无逐样本 teacher 的统一 QA solver。 |
| 距离通过、Reader 失败 | 只表明所定 latent 距离不足以保证 Reader；另立 teacher 邻域鲁棒性诊断。 |
| 距离失败、Reader 通过 | 找到可读 endpoint，但未满足距离合同；不能称 bridge 两主门通过。 |
| 两主门失败、二级通过 | 该条件干预有帮助但不足；不证明 conditioning 是唯一/主导根因。 |
| 两主门失败、二级失败 | 本次干预未同时改善两指标；不证明所有条件都无效或完整输出空间无解。 |

后两分支需要根据实际轨迹另行预注册一个能区分候选原因的最小诊断；本方案不自动授权追加提示词、步数、target 或门槛。原 event 被本干预遮蔽，因此任何本轮结果都不证明 writer 已学会 event→latent，也不支持 state-level memory、held-out 泛化或长期 recurrence。

始终设置 `formal_success=false`、`phase2_allowed=false`。即使本轮诊断通过，也不能把旧六条成功样本与本 target 拼成 Phase 1A 8/8。回主线仍须同一统一、无逐样本 canonical teacher 的 QA solver 完整重验固定八成员，之后才能进入固定 64/128 条 Phase 2，再验证 Phase 3A/3B。

## 7. 执行与时间边界

按 Inspire 规范，本地测试并推送 Git 后，由 CPU 出口准备独立 clean detached checkout；只在指定 H200 实例的新 `/inspire/ssd/` 根执行，检查 GPU 占用与 suite lock，不覆盖旧数据。探针与后续完整 bridge 分别交付配置/提交、哈希、原始产物、机器可读结果、中文报告与决策。

原 30 小时规划时钟固定为 `2026-09-05T07:18:59.891307+00:00`，现已超出；不得重置时钟、降低科学门槛或把 Phase 2 缩到 64 以下来声称按期完成。现阶段不承诺 Phase 2/3 完成时间；只报告真实阶段与下一个有界步骤。
