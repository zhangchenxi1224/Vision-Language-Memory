# R11_new：Teacher 审计 → LoRA 单样本可学习性诊断

## 授权与定位

2026-09-07 用户授权执行：先检查 canonical teacher 4 ↔ 7 的互换读取，再进行“已知可实现正对照 → canonical teacher 单样本 LoRA 拟合”。这是新的诊断分支，不修改旧 R11_new 的阶段门槛。

canonical R11 直接优化最终 VAE latent；R11_new 原 Phase 1A 优化 initial latent 并运行冻结 DreamLite。两者不能混淆。原 Phase 1A 的 6/8 结果不升级，Phase 2 仍不开放。

直接前序 identity-conditioning bridge 已由独立聚合器确认：raw256 MSE `0.1113223583`、Reader `0/4`、mean CE `25.64063823`，相对原事件父臂只改善约 0.40%/0.49%。这把“condition prompt 是主瓶颈”降级，并使 canonical teacher 是否具有样本特异读取信息成为进入生成器适配前必须排除的假阳性来源。

## D0 实现与运行前审计

- 配置状态：`preregistered_before_any_d0_donor_or_reset_reader_outcome`；截至本节哈希写入时，尚未执行新的 donor/reset Reader forward。
- 部署实例：`vlm-r11-identity-h200x4-20260907`；物理 4×H200，但固定 `CUDA_VISIBLE_DEVICES=0,1`，运行时只允许看见两张 H200。
- config bytes SHA-256：`3d11e602d8ac70cc3716ad375db92fe7c83fc8ba2c13d692d3e063d3f477ef1f`
- config canonical JSON SHA-256：`3f65185444e7c711571cc7cf19c56ee0e52937912e83bbb1403c319ee8a7f536`
- core SHA-256：`f9be0de1eeb5a1979a01561663d9017e498dd66ef889ba406cf25c1e6690e717`
- runner SHA-256：`83f1c82971f36290cc01923f24082735762d69801d784a97fc45106dea3b3e1d`
- tests SHA-256：`77db103ef16655e9d69682ab492c163706ba37c00ce6098bd7a2e274df295849`
- 定向测试：`10 passed`；Ruff、`compileall` 与 JSON 解析均通过。
- 与 identity-conditioning 前序实现的联合回归：`86 passed, 0 failed, 0 errors, 0 skipped`；JUnit SHA-256：`18f392b3a7ac57d58eee80240b383bf02ff0040617c727f7e5261840084ee5fb`。
- 测试明确覆盖“own 与 donor 都触发正确答案”的通用触发图情形，并要求该情形的 specificity gate 必须失败；不能用 own replay 4/4 单独宣称 teacher 有记忆区分力。

## D0：先检查标签是否具有样本区分力

唯一执行配置：`configs/experiments/r11_new_teacher47_swap_audit.json`。配置在读取任何新的 donor 结果前提交 Git。

- 固定 canonical target 4（blue）与 target 7（green），保留原 query、候选答案、目标索引和四个 reverse-cyclic4 排列；不生成新数据。
- 两条 query 分别读取自己的 teacher 图、另一条 teacher 图、固定 127/255 空白图：2 × 3 × 4 = 24 条原始结果。
- 从哈希绑定的原始 FP32 endpoint latent 以 BF16、原 VAE scaling/shift 和 clamp 解码，要求与原存档 BF16 RGB 逐位一致；不经过 U-Net，不训练任何参数。
- 使用原冻结 Qwen3-VL-4B、256 像素 tensor-native resize、teacher-forced 四候选平均 token NLL 排名。不是自由生成准确率。
- 保存每行四个原始 choice logits、实际图像张量哈希、query/排列/目标、CE 和预测；独立从 logits 复算，不信任报告里的布尔值。

### D0 预注册门槛

工程门槛：输入文件/模型快照/代码提交绑定，24 行完整无重复且有限，图像与 query 绑定正确，原 teacher RGB 重放逐位一致，模型冻结，0 U-Net 调用、0 optimizer step，96 次 Reader 候选前向，产物清单完整。

Teacher 重放门槛：两条 own 条件各 4/4 正确，平均 listwise CE ≤ 0.001。

区分力门槛：对每条 target，own 分别相对 donor 和 reset 均须满足：4/4 排列 CE 严格更低；平均 CE 至少降低 20%；准确率差至少 0.25。两个方向都必须通过。门槛边界不四舍五入。

这是静态配对审计，不报告训练 DiD。如果技术无效，修工程后使用新目录重跑同一审计；如果 own 重放或区分力不通过，不进入 LoRA 拟合，不改门槛或挑选另一对来冒充本轮通过。

### Round 01 技术终止与唯一修复

预注册提交 `08687bb57f59eb9246fbe3f448485a84fdcca67d` 的 Round 01 在 24/24 receipts 产生后被 forward-count 守门截停：评分实现调用 `reader.model(...)`，但计数钩子挂在未被调用的外层 `reader(...)`，因此计数误报并形成 `technical_failed` 终态。两条 canonical RGB 均逐位重放成功，但本轮没有生成 `result.json`，不构成有效 D0；donor/reset logits 和效果值未被查看、未用于修改设计。

唯一允许的 Round 02 修复是把钩子移到实际执行的 `reader.model`，并在守门前保存 observed/expected execution counts。配置、两条样本、三种图像条件、四个排列、96 次候选前向、所有数值门槛和停止规则完全不变；Round 01 原目录只读保留并作为失败产物交付。

Round 02 修复后、重新读取任何远端结果前的实现哈希：

- config bytes SHA-256（不变）：`3d11e602d8ac70cc3716ad375db92fe7c83fc8ba2c13d692d3e063d3f477ef1f`
- core SHA-256：`5bbfb32af5b215024035ebd728a85d4750b1b662242165ed3b95ed591d1a361c`
- runner SHA-256：`acdce1782292dc727489f1a4469cbabf678d7d3f315214749e597d232cac823f`
- tests SHA-256：`faf3ae727074ba971c5ce64dd25b71936e64ad979d0272ae55f637ea2be00c0f`
- 联合回归：`86 passed, 0 failed, 0 errors, 0 skipped`；JUnit SHA-256：`e98da606f597554ce6eb3ce9e1147d521545d859fa393c7fa949f55b4576d1e1`。

即使 D0 通过，也只排除这两个现成不同实体、同槽同候选样本之间的一种通用读取捷径，不证明完整 state 语义、事件因果性或长期记忆。

## D1：条件激活的单样本 LoRA 诊断（当前尚未部署）

D0 通过并经独立复算后，先锁定 D1 的完整数值配置、实现和 CPU 测试，再进行固定样本 GPU preflight；不得直接启动未经测试的训练。

1. 已知可实现正对照：使用相同冻结基座、相同 rank-4 q/k/v/out LoRA 架构、固定原 target-01 事件/空白 source/固定 initial latent。由固定非零 teacher LoRA 生成可实现的 endpoint。学生从标准 LoRA 初始化学习该 endpoint；teacher 权重不得复制给学生，初始误差必须明显高于数值重放误差，避免“本来已经相同”的伪正对照。
2. 正对照通过后：重新从同一学生初始化，以 canonical target-01（ambient）FP32 endpoint 作为唯一目标，使用完整四步 DreamLite 路径，只训练 LoRA。initial latent、基座、conditioner、VAE、Reader 全部不训练。
3. 两臂使用相同固定事件、初始化、优化器/学习率/步数/裁剪和原 sigma 序列。主损失是原坐标系 FP32 endpoint mean MSE；不增加 QA 梯度、不归一化目标坐标、不修改提示词。QA 只用于独立读取评测。
4. 数值拟合与可读性分别判定：MSE 改善不等于 Reader 可读；单样本成功不等于共享模型或 event 因果依赖。只有正对照通过而 canonical 失败，才收窄为“当前优化和 LoRA 约束下不能拟合该 oracle”，不能宣称数学上不可达。

D1 的确切 teacher LoRA 幅度、学习率、固定步数和拟合门槛尚需代码审计后单独预注册；本提交仅提供 D0 可执行入口，不将 D1 设计草案称为训练已部署。

## 范围、保存与失败处理

- 首轮不生成 64/128 条标签，不增加数据集、recurrence，不解冻整个 U-Net，不恢复定时任务。
- 每轮使用新的远端目录，保存配置、commit、输入/模型哈希、环境、原始 receipts、张量/图像、结果 JSON、Markdown 报告、日志、终态及 SHA256 inventory。D0 无 optimizer，所以不制造训练 checkpoint。
- 仅在哈希核验、单元测试、GPU 技术检查通过后运行对应昂贵阶段；发现失败保留证据、按决策树停止下游。
- 工程通过、机制/诊断通过、科学成功分栏报告。本轮所有结果的完整记忆科学成功标志固定为 false。
