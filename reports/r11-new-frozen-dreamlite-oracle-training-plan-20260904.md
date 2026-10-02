# R11_new：Frozen-DreamLite Oracle 与共享 Writer 最小闭环训练方案

> 文档性质：实验设计与预注册草案
> 日期：2026-09-04
> 工作名称：R11_new / Frozen-DreamLite Oracle-to-Writer Program
> 当前状态：MVP 路线已修订为 Phase 1A -> Phase 2 -> Phase 3；Phase 1A v1 等待训练代码提交与技术预检

> 2026-09-04 结果前修订：当前 MVP 不新增数据集，并将 Phase 1B 从推进门中移除。Phase 1A 通过后，直接在现有锁定 train split 的确定性子集上构建 query-level oracle bank，再用 Phase 3 判断共享 writer 是否具有训练子集可学习性。Phase 1B 保留为未来 state-level / long-term-memory 确认实验，不是当前 MVP 的前置条件。

> 2026-09-05 结果前规模修订：Phase 2 不做全量 train split。固定最低规模为 64 条；仅当首 8 条的实测吞吐按第 24.4 节外推后，128 条方案仍预计落在约 30 小时总体规划窗口内，才扩展为 128 条。样本顺序、规模决策公式和结论边界均在任何 R11_new 模型结果产生前锁定。

## 快速阅读路径

- 若只想理解核心逻辑：阅读第 0、1、2、4、11 和 22 节。
- 若要实现实验：阅读第 3、5 至 10、13 和 16 节。
- 若要判断是否成功：阅读第 12、14、15 和 19 节。
- 若要长期自动迭代与交付：阅读第 17、18 和 20 节。

## 0. 重要命名说明

本方案中的 **R11_new** 不是现有 canonical R11 的重复实验，也不能用现有 R11 的结果替代。

现有 canonical R11 采用以下链路：

    可训练的最终 VAE model-space latent
        -> 冻结 VAE Decoder
        -> 冻结 Qwen-3VL Reader
        -> QA loss

它明确绕过了 DreamLite U-Net、文本条件编码器、Scheduler、event noise、LoRA 和 recurrence。现有结果为 8/8 通过，只证明：

> 冻结 VAE 的可解码 latent 空间中，存在 Qwen-3VL Reader 可读取的 target-specific code。

R11_new Phase 1 则采用以下链路：

    可训练的 initial/noise latent x_T
        -> 完整且冻结的 DreamLite 推理链
        -> next-state endpoint z_t
        -> 冻结 VAE Decoder
        -> 冻结 Qwen-3VL Reader
        -> QA loss

因此，R11_new 测试的是一个更严格的新命题：

> 当完整 DreamLite 映射保持冻结时，能否仅通过优化其 initial/noise latent，在完整 DreamLite 输出空间内找到 Reader 可读的 endpoint？

现有 canonical R11 的 8/8 不能表述为 R11_new Phase 1 已通过。R11_new 必须使用新的代码提交、预注册、运行根和报告，不得覆盖现有 R11 结果。

---

## 1. 对“我的方案和 Phase 1 有什么区别”的直接回答

你的整体方案与三阶段路线没有本质冲突。准确关系是：

- 你提出的 Frozen-DL Oracle A，就是 Phase 1 的核心。
- 把预先锁定的训练样本逐条转换成 oracle latent 标签库，就是 Phase 2；本轮 MVP 固定为 64 条并在时间门通过时扩展到 128 条。
- 用 oracle latent 显式监督 DreamLite U-Net/LoRA，就是 Phase 3。

真正需要升级的不是这条主思路，而是以下三个定义。

### 1.1 从 per-query 升级为 per-state / per-transition

单独为每个 query 优化一个 latent，可以作为最小可行性诊断，但容易只得到答案特定编码：

$$
q \rightarrow \mathrm{answer\mbox{-}specific\ code}
$$

长期目标希望共享 writer 学习的是状态更新：

$$
(z_{t-1}, e_t) \rightarrow z_t
$$

因此，最终 state-level 版本的 oracle 标签应以 **memory state 或 state transition** 为基本单位。同一状态对应的一组独立查询应共享同一个 oracle endpoint，而不是每道题拥有互不相关的 latent。

当前 MVP 有意推迟这项升级：先允许“一条现有训练 query 对应一个 oracle endpoint”，用它回答 oracle supervision 能否把 shared writer 从不可学变为可学。由此得到的标签必须称为 **query-level oracle label**，不能称为完整 memory state。

### 1.2 区分搜索变量与监督标签

Phase 1 优化的是进入完整 DreamLite 的 initial/noise latent：

$$
x_T
$$

完整冻结 DreamLite 执行结束后产生的 next-state endpoint 才是：

$$
z_t^\star
$$

Phase 2 应同时保存二者，但 Phase 3 默认监督 writer 生成的是 $z_t^\star$，不是 $x_T^\star$。

### 1.3 将拟合查询与审计查询分离（未来 state-level 确认要求）

如果优化和评测使用完全相同的 query，实验只能证明对该 query 的过拟合。状态级命题要求：

$$
\mathcal{Q}_t^{\mathrm{fit}}
\cap
\mathcal{Q}_t^{\mathrm{audit}}
=
\varnothing
$$

同一个 oracle state 必须在未参与优化的同状态 probe、paraphrase 和 choice permutation 上仍然可读。

所以最准确的结论是：

> 当前 MVP 等于 Phase 1A + Phase 2 + Phase 3：先寻找 per-query 可读 endpoint，再把它们作为 query-level teacher labels 训练共享 writer。它可以验证 supervision 是否可学，但只有未来补做 Phase 1B 与 held-out 因果评测后，才能升级为 state/transition-level memory 结论。

---

## 2. 第一性原理：当前问题究竟由哪些子问题组成

最终需要学习一个共享状态更新器：

$$
z_t = W_\theta(z_{t-1}, e_t)
$$

状态通过冻结的 Decoder 转为图像：

$$
I_t = D(z_t)
$$

冻结 Reader 根据图像和 query 输出答案：

$$
\widehat{a} = R(I_t, q)
$$

直接以最终 QA loss 训练 $W_\theta$ 时，模型被要求同时完成两件困难的事：

1. 搜索“什么样的 visual latent 能被 Reader 读取”；
2. 学习“怎样根据 previous state 和 event 写出这种 latent”。

这会把五个不同问题混在一起：

| 层级 | 第一性问题 | 需要的实验 |
| --- | --- | --- |
| Q1 | Reader 的图像接口中是否存在答案相关方向？ | direct-pixel oracle |
| Q2 | 冻结 VAE 的可解码区域中是否存在可读 code？ | canonical R11 direct VAE-latent oracle |
| Q3 | 完整冻结 DreamLite 的输出映射能否到达可读 endpoint？ | R11_new Phase 1 |
| Q4 | 一个共享 writer 能否摊销这些逐样本 oracle 解？ | R11_new Phase 3A/3B |
| Q5 | 共享 writer 是否真正泛化、具有因果性并支持递归？ | held-out、controls、multi-seed、rollout |

R11_new 的基本策略是先解决目标存在与可达性，再解决共享学习问题：

$$
\boxed{\text{Find readable target states before learning the writer}}
$$

这比端到端依靠稀疏 QA loss 盲目搜索更容易诊断，也能在失败时明确定位问题属于表示、可达性、监督标签、writer 容量、优化还是泛化。

---

## 3. 统一符号

| 符号 | 含义 | Phase 1 是否优化 |
| --- | --- | --- |
| $s_t$ | 时刻 $t$ 的真实语义状态 | 否 |
| $z_{t-1}$ | previous memory latent | 否 |
| $e_t$ | 当前写入事件 | 否 |
| $x_T$ | 完整 DreamLite 推理轨迹的 initial/noise latent | 是 |
| $\Phi_{\mathrm{DL}}$ | 完整冻结 DreamLite 映射 | 否 |
| $z_t(x_T)$ | 完整冻结 DreamLite 对 $x_T$ 的 endpoint | 否 |
| $x_T^\star$ | oracle 优化得到的 initial/noise latent | 优化结果 |
| $z_t^\star$ | 由 $x_T^\star$ 产生的 oracle next-state endpoint | Phase 3 教师标签 |
| $D$ | 冻结 VAE Decoder | 否 |
| $R$ | 冻结 Qwen-3VL Reader | 否 |
| $W_\theta$ | 待训练的共享 DreamLite writer | Phase 3 训练 |

完整冻结链路定义为：

$$
z_t(x_T)
=
\Phi_{\mathrm{DL}}
\left(
x_T;
z_{t-1},
e_t
\right)
$$

$$
I_t(x_T)
=
D
\left(
z_t(x_T)
\right)
$$

Phase 1 的 oracle 搜索为：

$$
x_T^\star
=
\operatorname*{arg\,min}_{x_T}
\frac{1}{\lvert \mathcal{Q}_t^{\mathrm{fit}} \rvert}
\sum_{(q,a)\in\mathcal{Q}_t^{\mathrm{fit}}}
\mathrm{CE}
\left(
R(I_t(x_T),q),
a
\right)
$$

随后定义真正的 next-state 教师：

$$
z_t^\star
=
\Phi_{\mathrm{DL}}
\left(
x_T^\star;
z_{t-1},
e_t
\right)
$$

必须始终保持：

> $x_T^\star$ 是搜索到的 steering input；$z_t^\star$ 才是 oracle next-memory state 的候选标签。

---

## 4. 总体路线

当前 MVP 采用以下顺序：

1. **Phase 0：冻结协议与技术预检**
2. **Phase 1A：per-query Frozen-DL bridge oracle**
3. **Phase 2：在现有 train split 的固定 64/128 条子集上构建 query-level Oracle Latent Bank**
4. **Phase 3A：Endpoint Distillation 锁定训练子集可学习性**
5. **Phase 3B：Native Flow-Matching Distillation**
6. **未来确认：Phase 1B、held-out、多 seed、递归与因果评测**

Phase 1A 是低成本桥接实验；通过后不再等待 Phase 1B，而是直接把同一求解过程扩展到现有训练数据的固定 MVP 子集。Phase 2 不新增样本，只为预注册的 64 条核心样本生成派生的 $x_T^\star$、$z_t^\star$ 与 $I_t^\star$ 标签；仅在时间门通过时扩展到 128 条。Phase 3A 先验证这些标签能否被共享 writer 在同一训练子集上拟合；Phase 3B 仅在 3A 通过后恢复 DreamLite 原生训练范式。

这条 MVP 路线缩短了闭环，但同步降低结论强度：即使 Phase 3 训练子集通过，也不能声称完整训练集、state-level、long-term memory 或 held-out 泛化成功。

---

## 5. Phase 0：冻结实验协议

正式观察任何 R11_new 模型结果前，必须锁定以下内容。

### 5.1 数据与状态定义

- 固定 train、dev-select、sealed dev-final 的样本 ID 与 SHA-256。
- 明确一个 state 对应哪些 query。
- 明确 transition ID、previous state、event 和 next state 的构造规则。
- 明确同一状态的 fit query 集与 audit query 集。
- 明确 held-out entity、value、transition 和 query template。
- 在构建 oracle bank 前完成 split；不得把 dev/test oracle 混入 writer 训练。

### 5.2 模型与推理契约

- DreamLite、VAE、Reader 的快照路径与清单哈希。
- 完整 DreamLite 实际执行的组件。
- 固定推理步数、Scheduler、scale/shift、prompt 模板和分辨率。
- $x_T$ 的形状、dtype、初始化分布和 seed。
- previous state 的来源：teacher-forced、oracle parent 或 model rollout。
- 唯一 primary endpoint；禁止用中间最佳 checkpoint 挽救正式结果。

### 5.3 优化契约

以下数值必须写入机器可读配置后再启动正式运行：

- optimizer；
- learning rate；
- optimizer steps；
- restart/seed 数量；
- gradient clipping 是否启用；
- loss 权重；
- checkpoint steps；
- accumulation；
- precision；
- determinism 配置。

可以先做只验证数值正确性的技术 preflight，但不得根据正式 dev/test 结果反复修改这些值而继续沿用同一预注册结论。

### 5.4 预注册输出

Phase 0 至少交付：

- 预注册 Markdown；
- immutable JSON config；
- 数据与模型哈希；
- 实现测试；
- technical preflight 报告；
- 训练锁定 Git commit；
- 唯一 run ID 和空运行根检查。

---

## 6. Phase 1A：Frozen-DL Bridge Oracle

### 6.1 目的

使用与 canonical R11 尽可能相同的固定 F1 targets、Reader、choice views 和评测门槛，仅改变一个变量：

| 实验 | 唯一可训练对象 | 是否执行完整 DreamLite |
| --- | --- | --- |
| canonical R11 | 最终 VAE latent | 否 |
| R11_new Phase 1A | initial/noise latent $x_T$ | 是，全部冻结 |

这样可以最小化归因歧义，直接回答：

> canonical R11 已知的可读区域，是否能够通过完整冻结 DreamLite 的输入空间被找到？

### 6.2 固定链路

对一个固定 transition：

$$
(x_T,z_{t-1},e_t)
\xrightarrow{\mathrm{Frozen\ DreamLite}}
z_t
\xrightarrow{D}
I_t
\xrightarrow{R}
\mathcal{L}_{\mathrm{QA}}
$$

只有 $x_T$ 可以更新。以下组件全部执行但全部冻结：

- DreamLite condition encoder；
- DreamLite U-Net；
- Scheduler；
- VAE Decoder；
- Qwen-3VL Reader。

### 6.3 最小目标

Phase 1A 可以先对一个 query 做可行性验证，但应保留 fit/audit choice views：

$$
\mathcal{L}_{1\mathrm{A}}
=
\frac{1}{\lvert \mathcal{Q}_t^{\mathrm{fit}} \rvert}
\sum_{(q,a)\in\mathcal{Q}_t^{\mathrm{fit}}}
\mathrm{CE}
\left(
R(D(z_t(x_T)),q),
a
\right)
$$

主实验不建议先加入 latent 正则，以免把“是否存在”与人为先验混淆。可以记录 $x_T$ 范数、endpoint 位移、图像饱和度和频谱等诊断，但任何额外正则必须提前预注册。

### 6.4 技术门

Phase 1A 只有在以下条件全部满足时才可解释：

- 只有一个 $x_T$ tensor 为 trainable；
- 所有 DreamLite、VAE、Reader 参数保持冻结且哈希不变；
- 完整 DreamLite 固定步数确实执行；
- 每个 optimizer step 都有 finite、nonzero gradient receipt；
- 步数、query views、checkpoint 数量完全匹配配置；
- stdout、stderr、环境、版本、GPU 信息完整；
- raw endpoint 固定，无 best-checkpoint selection；
- 每个 target 均产生独立 terminal 和 artifact inventory。

### 6.5 建议沿用的可复算诊断门

令条件 $c$ 的 CE 变化为：

$$
\Delta_c
=
\mathrm{CE}_{c,\mathrm{end}}
-
\mathrm{CE}_{c,\mathrm{M0}}
$$

normal 与控制条件的 difference-in-differences 为：

$$
\operatorname{DiD}(\mathrm{normal},c)
=
\Delta_{\mathrm{normal}}
-
\Delta_c
$$

建议沿用现有主线口径，并在正式运行前锁定：

- normal endpoint CE 相对 M0 至少下降 20%；
- 四个固定 reverse-cyclic views 全部改善；
- accuracy 至少提高 0.25；
- $\operatorname{DiD}(\mathrm{normal},\mathrm{reset}) < 0$；
- exact technical gate 通过。

Phase 1A 成功只能表述为：

> 在锁定链路、优化预算和初始化集合下，找到了至少一个可读的 Frozen-DreamLite endpoint。

Phase 1A 失败只能表述为：

> 在锁定搜索协议下尚未证明可达。

非凸优化失败不等于数学上证明输出空间不存在可读 code。

---

## 7. Phase 1B：State/Transition-Level Oracle（当前 MVP 延后）

### 7.1 为什么不能停在 per-query oracle

如果同一个 transition 为不同 query 分别生成不同标签：

    相同的 (z_previous, event)
        -> query 1 对应 latent A
        -> query 2 对应 latent B
        -> query 3 对应 latent C

共享 writer 在写入时看不到 query，因此无法判断应输出 A、B 还是 C。直接使用 MSE 可能只学到三个互不兼容标签的平均值，而平均 latent 未必可读。

### 7.2 状态级定义

对同一个真实状态 $s_t$ 构造查询集合：

$$
\mathcal{Q}_t
=
\left\{
(q_1,a_1),
(q_2,a_2),
\ldots,
(q_m,a_m)
\right\}
$$

一个共同的 $x_T^\star$ 和 $z_t^\star$ 必须支持整组查询：

$$
x_T^\star
=
\operatorname*{arg\,min}_{x_T}
\sum_{(q,a)\in\mathcal{Q}_t^{\mathrm{fit}}}
\mathrm{CE}
\left(
R(D(z_t(x_T)),q),
a
\right)
$$

正式审计使用未参与优化的：

- 同状态不同 probe；
- held-out paraphrase；
- held-out choice permutation；
- 同一事实的不同问法；
- 必要时针对未被直接询问字段的 query。

仅改变四个答案的位置不能证明 state-level memory；它只能排除答案位置捷径。若当前数据对每个状态只有一道语义 query，则必须诚实地把结果称为 **query-level feasibility**，不能升级为 state-level claim。

### 7.3 必需控制

- reset：恢复初始或空状态；
- state swap：换入另一个 transition 的 previous state；
- wrong event：写入错误事件；
- shuffled event：打乱 event 与样本的配对；
- choice permutation：改变答案位置；
- paraphrase：保持语义、改变模板；
- oracle swap：交换不同状态的 $x_T^\star$ 或 $z_t^\star$；
- multiple seeds：排除单次优化偶然性。

### 7.4 Phase 1B 通过条件

- exact technical gate 通过；
- 一个共同 endpoint 在全部 fit queries 上达到预注册门；
- 同一 endpoint 在全部 audit probes 上达到预注册门；
- normal 显著优于 reset、wrong-event 和 swapped-state；
- choice permutation 与 paraphrase 不破坏正确性；
- 多 seed 至少达到预注册的复现比例；
- 不允许 query、choices、target index 或答案进入 DreamLite 写入输入。

本节不再是当前 MVP 构建 query-level oracle bank 的前置门。当前允许在 Phase 1A 通过后直接进入 Phase 2，但 bank manifest 必须逐条标记 `supervision_scope=query_level`，且不得据此声称 state-level memory。若未来要主张状态语义、长期记忆或跨 query 一致性，本节全部条件仍必须补做并通过。

---

## 8. Phase 2：Oracle Latent Bank Construction

### 8.1 目标

对现有锁定 train split 中预先选定的 64/128 个 query/sample 离线求解；不新增、改写或扩展原始数据集：

$$
(z_{t-1}, e_t)
\longrightarrow
(x_T^\star, z_t^\star)
$$

当前 MVP 形成显式的 query-level 监督数据：

$$
\boxed{
(z_{t-1}, e_t, z_t^\star)
}
$$

这一步将原本稀疏的下游 QA supervision 转换为密集的 latent target supervision。它是派生标签库，不是新数据集，也不是 state-level oracle 的证据。

Phase 2 先把 Phase 1A 的固定 8 条结果封装为工程与吞吐 smoke；不重复优化，且这 8 条计入最终 bank。随后至少完成固定的 64 条核心 bank，并仅在第 24.4 节的时间门通过时扩展到 128 条。当前 MVP 明确不追求 train split 的 100% 覆盖；这只是避免批量制造损坏 artifact 并控制总耗时，不构成额外 Phase 1B 科学验证。

### 8.2 每条 bank 记录必须保存

| 字段 | 内容 |
| --- | --- |
| transition_id | 固定 transition 标识 |
| split | train、dev-diagnostic 或 sealed-test |
| previous_state | tensor 或不可变 artifact 引用 |
| previous_state_sha256 | previous state 哈希 |
| event_payload | writer 可见的 event |
| event_sha256 | event 哈希 |
| fit_query_set_sha256 | oracle 优化 query 集 |
| audit_query_set_sha256 | 未参与优化的状态 probes |
| $x_T^\star$ | oracle 搜索输入 |
| $z_t^\star$ | oracle endpoint 教师 |
| decoded_image | $D(z_t^\star)$ |
| solver_seed | oracle seed |
| solver_config_sha256 | 优化协议哈希 |
| checkpoints | 固定轨迹 |
| raw_predictions | 每个 query/control 的原始输出 |
| technical_gate | 技术有效性 |
| oracle_gate | query-level 或 state-level 门 |
| source_commit | 训练锁定提交 |
| artifact_inventory | 文件大小与 SHA-256 |

### 8.3 解决 oracle 多解问题

Reader 可读 latent 通常不唯一。两个都能答对的 latent 可能在欧氏空间相距很远。如果每条样本随机落入不同坐标系，共享 writer 会被迫拟合任意标签。

因此必须在看正式结果前选择一种固定策略：

1. 固定初始化、seed、optimizer 和 endpoint；
2. 对每个 transition 运行固定数量的 oracle restarts；
3. 按仅依赖 train-fit objective 的规则选择 canonical teacher；
4. 测量不同 seed 的 latent 距离与交叉 query 可读性；
5. 禁止看 dev/test 后挑选最漂亮或最易拟合的 teacher。

首轮建议使用一个确定的 canonical teacher，以保持最小实验。若证据证明多解导致标签不一致，再预注册以下最小修复之一：

- fixed anchor；
- minimum-norm teacher；
- prototype teacher；
- multi-teacher；
- min-over-K distillation；
- Reader-consistent objective。

不得在同一正式轮次中事后切换规则。

### 8.4 数据泄漏边界

- 只为 train split 构建可用于 writer 梯度的 oracle bank。
- dev oracle 只能用于诊断 ceiling，不得参与训练或 checkpoint 选择。
- sealed test 在最终 endpoint 前不得打开。
- writer 永远不能读取 query、choices、answer、target index、segment ID 或 per-item trainable latent。
- Phase 3B 必须围绕 clean teacher $z_t^\star$ 重新采样独立噪声，不能把 $x_T^\star$ 当作样本 ID 式噪声输入。

### 8.5 Phase 2 的通过含义

Phase 2 对最终锁定规模（64 或 128）的 100% coverage、manifest 和 hash 通过属于 **工程与数据通过**。它只证明该 MVP 子集的监督库构建完整，不代表整个 train split 完整，不证明共享 writer 已经学会，也不证明泛化。

---

## 9. Phase 3A：Endpoint Distillation

### 9.1 目的

回答最小学习问题：

> 一个共享 DreamLite U-Net/LoRA writer，能否仅根据 previous state 与 event，拟合已经明确给出的 oracle endpoint？

writer 输入严格为：

$$
(z_{t-1}, e_t)
$$

writer 禁止输入：

- query；
- answer choices；
- answer text；
- target index；
- item/segment ID；
- entity-specific trainable embedding；
- per-item latent。

预测 endpoint 为：

$$
\widehat{z}_t
=
W_\theta(z_{t-1},e_t)
$$

### 9.2 Latent 监督

为避免分辨率或 latent 维数改变损失尺度，使用按维数归一化的 MSE：

$$
\mathcal{L}_{\mathrm{latent}}
=
\frac{1}{d}
\left\lVert
\widehat{z}_t-z_t^\star
\right\rVert_2^2
$$

可选 cosine 项为：

$$
\mathcal{L}_{\mathrm{cos}}
=
1-
\frac{
\left\langle
\widehat{z}_t,
z_t^\star
\right\rangle
}{
\left\lVert \widehat{z}_t \right\rVert_2
\left\lVert z_t^\star \right\rVert_2
+
\varepsilon
}
$$

Reader consistency 为：

$$
\mathcal{L}_{\mathrm{QA}}
=
\frac{1}{\lvert\mathcal{Q}_t^{\mathrm{fit}}\rvert}
\sum_{(q,a)\in\mathcal{Q}_t^{\mathrm{fit}}}
\mathrm{CE}
\left(
R(D(\widehat{z}_t),q),
a
\right)
$$

组合目标为：

$$
\mathcal{L}_{3\mathrm{A}}
=
\lambda_{\mathrm{lat}}
\mathcal{L}_{\mathrm{latent}}
+
\lambda_{\mathrm{cos}}
\mathcal{L}_{\mathrm{cos}}
+
\lambda_{\mathrm{QA}}
\mathcal{L}_{\mathrm{QA}}
$$

所有 $\lambda$ 必须在正式结果前锁定。建议同时保留 pure-latent ablation，以区分“教师能否被模仿”和“Reader QA 梯度是否帮助或干扰”。

### 9.3 训练顺序

1. 单 transition 严格过拟合；
2. 8 个 transitions 的共享 writer 过拟合；
3. 固定 query-level 小集合；
4. 最终锁定的 64/128 条 MVP bank；
5. 训练内 reset/donor/constant-event controls。

以上五项构成当前 MVP，其中“完整”只指最终锁定的 64/128 条子集，而不是完整 train split。小规模 state-level 集合、held-out entity/value/transition、多 seed 与多步 rollout 延后到未来确认阶段，不作为当前进入 Phase 3B 的前置门。

如果验证对象是 DreamLite U-Net/LoRA，3A 必须通过完整、可微的 DreamLite endpoint 产生 $\widehat{z}_t$。如果另加一个 direct regression head，它只能称为 writer capacity oracle，不能据此宣称 DreamLite U-Net 已学会。

### 9.4 结果解释

- train latent loss 下降：教师模仿开始生效；
- train QA 通过：锁定训练子集拟合通过；
- train QA 通过且 own-event 优于训练内 donor/reset/constant-event：当前 MVP 子集的共享 writer 可学习性诊断通过；
- held-out QA、state-level probes、rollout、multi-seed、sealed ID/OOD 全部通过：未来才接近正式科学成功。

由于 oracle latent 可能非唯一，MSE 未达到很低但 train QA 与训练内因果控制通过，不应自动判当前 MVP 的 writer 拟合失败。反之，train MSE/QA 通过也不等于泛化成功。未来最终科学门仍应以 Reader 正确性、held-out 泛化和完整因果控制为主，latent 距离只是诊断指标。

---

## 10. Phase 3B：Native Flow-Matching Distillation

### 10.1 目的

Phase 3A 直接约束最终 endpoint；Phase 3B 则按 DreamLite 原生 diffusion/flow 训练方式，监督每个噪声位置的运动方向。

将 oracle endpoint $z_t^\star$ 视为 clean target，独立采样：

$$
\epsilon
\sim
\mathcal{N}(0,I)
$$

$$
\sigma
\in
[0,1]
$$

采用以下约定时：

$$
z_\sigma
=
(1-\sigma)z_t^\star
+
\sigma\epsilon
$$

其中 $\sigma=0$ 为 clean，$\sigma=1$ 为 noise。对 $\sigma$ 增大方向的解析速度为：

$$
u^\star
=
\frac{\partial z_\sigma}{\partial \sigma}
=
\epsilon-z_t^\star
$$

U-Net 接收：

$$
U_\theta
\left(
z_\sigma,
z_{t-1},
e_t,
\sigma
\right)
$$

flow-matching loss 为：

$$
\mathcal{L}_{\mathrm{FM}}
=
\mathbb{E}_{\sigma,\epsilon}
\left[
\frac{w(\sigma)}{d}
\left\lVert
U_\theta
\left(
z_\sigma,
z_{t-1},
e_t,
\sigma
\right)
-
u^\star
\right\rVert_2^2
\right]
$$

从 noise 端向 clean 端积分时：

$$
z_{\sigma_{k+1}}
=
z_{\sigma_k}
+
(\sigma_{k+1}-\sigma_k)
U_\theta(\cdot)
$$

其中：

$$
\sigma_{k+1}
<
\sigma_k
$$

因此即使解析 target 写作 $\epsilon-z_t^\star$，负的积分步长仍会使轨迹朝 clean endpoint 移动。

### 10.2 必须遵守官方实现契约

上面的公式只是一个明确的参数化例子。正式实现必须逐行继承当前 DreamLite 代码的：

- time/sigma 方向；
- scheduler shift 后的 effective sigma；
- model output 类型；
- velocity、noise 或 clean-target 转换；
- latent scale/shift；
- loss weighting；
- inference update rule。

如果官方实现把时间定义为 noise-to-clean，或模型预测的是 $x_0$、$\epsilon$ 或经过 scheduler 变换的输出，target 符号和转换会不同。不得凭通用公式直接实现。

训练前必须通过三个数值测试：

1. 令网络输出解析 $u^\star$，从 $\sigma=1$ 积分到 $\sigma=0$ 能恢复 $z_t^\star$；
2. 输入解析 target 时，flow loss 接近零；
3. 训练构造的 sigma 与 scheduler 实际 effective sigma 逐点一致。

### 10.3 不能复用 $x_T^\star$ 作为训练噪声

Phase 3B 应针对固定 clean teacher $z_t^\star$ 重新采样与 query、答案和 sample ID 无关的 $\epsilon$。否则模型可能把 oracle $x_T^\star$ 当作 per-sample identifier，形成新的信息泄漏。

---

## 11. Phase 3A 与 Phase 3B 的区别

| 维度 | Phase 3A：Endpoint Distillation | Phase 3B：Flow-Matching Distillation |
| --- | --- | --- |
| 监督对象 | 最终 endpoint | 噪声轨迹上的局部方向 |
| 核心问题 | writer 至少能否拟合 oracle target？ | 原生 DreamLite 机制能否学会到达 target？ |
| 实现复杂度 | 较低 | 较高 |
| 调试难度 | 较低，归因直接 | 较高，涉及时间、符号、scheduler |
| 与 DreamLite 预训练目标的一致性 | 较弱 | 较强 |
| 最适合的角色 | proof-of-concept 与容量诊断 | 正式主方案 |
| 失败含义 | 容量、conditioning、标签或优化问题 | 若 3A 已通过，则优先检查 flow 参数化与训练/推理不一致 |

推荐顺序是：

$$
\boxed{
\mathrm{Phase\ 3A}
\rightarrow
\mathrm{Phase\ 3B}
}
$$

不是二选一。3A 先回答监督是否可学；3B 再回答能否保留 DreamLite 原生生成机制。

---

## 12. Baselines 与最小对照

| 编号 | 实验 | 回答的问题 |
| --- | --- | --- |
| B0 | 当前 QA-only DreamLite/LoRA baseline | 稀疏端到端监督的实际水平 |
| B1 | canonical R11 direct VAE-latent oracle | VAE 可解码区域是否存在可读 code |
| B2 | R11_new Phase 1A per-query Frozen-DL oracle | 完整冻结 DreamLite 是否存在 query-readable endpoint |
| B3 | R11_new Phase 1B state-level oracle | 一个 endpoint 是否支持同状态多 query |
| B4 | Phase 3A shared endpoint writer | 显式 endpoint supervision 是否可被共享 writer 学习 |
| B5 | Phase 3B native flow writer | 原生 DreamLite 训练是否可摊销 oracle states |
| C0 | constant/zero-event arm | 通用图片或 Reader prior 能否伪造提升 |
| C1 | wrong-event/donor arm | 输出是否真正依赖正确 event |
| C2 | reset/state-swap arm | 正确状态是否具有因果作用 |

每次最小判别实验只改变一个核心因素，其他数据、模型、endpoint 和因果门保持不变。

---

## 13. 因果评测与信息边界

### 13.1 Writer 写入时允许看到

- previous state $z_{t-1}$；
- 当前 event $e_t$；
- 预注册的独立噪声；
- 固定 timestep/sigma。

### 13.2 Writer 写入时禁止看到

- query text；
- answer choices；
- answer text；
- target index；
- segment/item ID；
- per-item latent；
- entity lookup table；
- dev/test label；
- 由答案派生的 noise seed。

### 13.3 必需评测条件

| 条件 | 目的 |
| --- | --- |
| normal | 自己的 previous state 与正确 event |
| reset | 去除学习到的状态 |
| donor | 使用另一个 target value 的 event/state |
| wrong-event | 保留样本但写入错误事件 |
| shuffled-event | 打乱 batch 中 event 配对 |
| state-swap | 交换 previous state |
| constant-event | 所有条件表示置零或常量 |
| choice permutation | 排除答案位置捷径 |
| paraphrase | 排除 query 模板捷径 |
| rollout | 检查误差累积与递归状态更新 |
| multi-seed | 检查可复现性 |

共享 writer 除 normal/reset 门外，还应沿用 own-event 对 donor 的同强度门：

- own-event CE 至少优于 donor 20%；
- 四个固定 views 全部优于 donor；
- accuracy 至少高 0.25；
- $\operatorname{DiD}(\mathrm{normal},\mathrm{donor}) < 0$；
- constant-event arm 必须失败。

---

## 14. 严格区分三类“通过”

| 层级 | 含义 | 不能声称什么 |
| --- | --- | --- |
| 工程通过 | commit、数据、模型、步数、梯度、checkpoint、冻结参数、日志和哈希均正确 | 不能声称机制有效 |
| 诊断通过 | oracle 可达，或 writer 能在训练子集拟合教师 | 不能声称泛化和 Picture Memory 成功 |
| 科学成功 | 共享 query-free writer 在 held-out、多 seed、因果控制和递归下通过预注册门 | 仍需限定到本实验覆盖的任务范围 |

R11_new Phase 1 成功只是 **full-chain reachability 诊断通过**。

Phase 2 bank 完整只是 **数据工程通过**。

Phase 3A 在锁定子集上的 train loss 或 train QA 通过只是 **拟合诊断通过**。

R11_new 范围内的科学成功至少要求：

- 一个共享且 query-free 的 writer；
- held-out entities、values、transitions 和 query templates；
- normal 明确优于 reset、donor、wrong-event 和 state-swap；
- constant-event arm 失败；
- 多个预注册 seed；
- 固定 raw endpoint，不用 best checkpoint；
- 完整 rollout，而非只看 teacher-forced one-step；
- 全新 sealed split 上的 confirmatory run；
- 全部结果可由原始 rows 独立复算。

完整 Picture Memory 成功还需要覆盖 SET、OVERWRITE、CLEAR、干扰、长期递归以及完整 ID/OOD 任务；不得由单步 F1 结果越级表述。

---

## 15. 失败后的第一性原理归因与最小判别实验

### 15.1 Phase 1A 失败

已知 canonical R11 的 endpoint latent 可读，但完整 Frozen-DL oracle 未找到可读 endpoint。

下一项最小判别实验不是直接扩大 LoRA 或改多个超参，而是固定 canonical R11 的已知可读 latent $z_{\mathrm{R11}}^\star$，只优化：

$$
\mathcal{L}_{\mathrm{bridge}}
=
\frac{1}{d}
\left\lVert
\Phi_{\mathrm{DL}}
\left(
x_T;
z_{t-1},
e_t
\right)
-
z_{\mathrm{R11}}^\star
\right\rVert_2^2
$$

解释：

- 若能逼近 $z_{\mathrm{R11}}^\star$：Frozen-DL 路径具备可达性，问题更可能在 Reader-loss landscape 或 oracle 优化。
- 若不能逼近：只能说明该已知 code 在锁定预算下未被该 Frozen-DL 路径到达，不能宣称所有可读 code 都不存在。

随后每轮只改变初始化、预算、目标或 scheduler 中的一个因素。

### 15.2 Phase 1B 失败

| 现象 | 最可能解释 | 下一最小实验 |
| --- | --- | --- |
| 单 query 成功，多 query 失败 | answer code，而非 state code | 交叉 query matrix 与 state probe |
| fit query 成功，paraphrase 失败 | query-template shortcut | 固定语义的 held-out paraphrases |
| 多 seed 都答对但 latent 相距很远 | oracle 多解、坐标不一致 | K-teacher 距离与交叉可读性 |
| normal 与 state-swap 相同 | state 没有因果作用 | 强化 state-swap/wrong-event 对照 |

### 15.3 Phase 2 失败

- hash、receipt 或 artifact 缺失：只修工程，原口径 fresh root 重跑，不做科学解释。
- oracle 覆盖不足：完整报告失败样本分布，禁止静默删除困难样本。
- 同 transition 标签不一致：先测试 canonicalization 或 multi-teacher，再训练 writer。
- teacher 大量依赖 query choice：标记为 query-level shortcut 风险；当前 MVP 只能继续做锁定训练子集可学习性诊断，不得升级为 state-level 结论。若要解决该风险，再激活 Phase 1B。

### 15.4 Phase 3A 失败

| 现象 | 定位 |
| --- | --- |
| 单 transition 无法过拟合 | writer 参数化、梯度、固定噪声或 endpoint unroll |
| 单 transition 成功，小集合失败 | 共享容量不足或标签坐标不一致 |
| latent MSE 下降但 Reader 失败 | 欧氏距离不是充分语义指标 |
| train 成功、held-out 失败 | 记忆训练样本或 event 表示不泛化 |
| normal/reset 改善但 donor 不败 | 通用图片或 Reader prior 假阳性 |

### 15.5 Phase 3B 失败

- 3A 成功而 3B 失败：优先检查 flow target、时间方向、scheduler shift、scale/shift、loss weighting 和 train/inference mismatch。
- analytic target 的 loss 不接近零：实现错误，属于技术失败。
- train metric 成功但因果 controls 失败：检查 query/answer 泄漏与 shortcut。
- one-step 成功但 rollout 失败：属于 exposure bias 或误差累积，需要独立预注册 rollout 修复。

任何技术失败都不得计入 scientific pass/fail；修复必须使用新 commit、新 run ID 和新报告。

---

## 16. 每阶段数据保存与报告要求

### 16.1 通用运行根

建议每一轮使用不可复用的新根：

    runs/r11_new/<training-commit>/<round-id>/
        phase0_preflight/
        phase1a_bridge_oracle/
        phase1b_state_oracle/
        phase2_oracle_bank/
        phase3a_endpoint_distillation/
        phase3b_flow_matching/
        evaluation/
        delivery/

每个阶段至少包含：

- launch manifest；
- RUNNING、SUCCEEDED 或 FAILED terminal；
- immutable config；
- environment 与 dependency versions；
- source commit；
- data/model hashes；
- stdout 与 stderr；
- raw metrics/receipts；
- checkpoints；
- raw prediction rows；
- causal control rows；
- technical gate；
- artifact inventory 与 SHA-256；
- 阶段报告。

### 16.2 Phase 1 交付

- 每个 state/transition 的 $x_T$ checkpoint trajectory；
- 每个 checkpoint 的 $z_t$ 与 decoded image；
- fit/audit query 的原始 logits、CE、accuracy；
- reset、swap、wrong-event、permutation、paraphrase rows；
- gradient norm、finite audit、trainable/frozen parameter audit；
- target-level terminal；
- 汇总 coverage 和失败类型报告。

### 16.3 Phase 2 交付

- oracle bank manifest；
- train-only bank index；
- tensor files 与 hashes；
- solver seeds 与 config hashes；
- per-transition technical/scientific gate；
- 多解稳定性分析；
- missing/failed transition 清单；
- bank coverage 报告；
- 可复算的数据字典。

### 16.4 Phase 3 交付

- optimizer receipts；
- fixed checkpoint trajectory；
- latent、QA 和 flow losses；
- raw/EMA endpoint，但 primary endpoint 必须预先固定；
- train、dev-select、sealed-final 的逐项 rows；
- reset、donor、wrong-event、constant-event、swap 对照；
- per-seed 与跨 seed 汇总；
- rollout 轨迹；
- gradient 路由与冲突诊断；
- 最终 comparison、analysis、report 与图表。

---

## 17. 每轮 GitHub 与完整交付闭环

每轮必须执行以下闭环。

### 17.1 运行前提交

1. 写入新的 preregistration；
2. 写入 immutable config；
3. 完成实现与测试；
4. 固定数据、模型和父产物哈希；
5. 提交并推送训练代码；
6. 记录完整 40 位 training commit；
7. 在远端使用 clean checkout；
8. 预创建检查必须确认输出根为空且无重复进程。

### 17.2 运行中保存

- 每一步或固定间隔的真实 receipt；
- checkpoint；
- latent/image trajectory；
- 每个 control 的原始结果；
- stdout/stderr；
- GPU、环境和进程信息；
- fail-closed terminal；
- 不得只保留最后一行汇总。

### 17.3 运行后分析

1. 冻结 raw artifacts；
2. 生成 RAW_ARTIFACTS；
3. 由独立脚本复算 ANALYSIS；
4. 生成 REPORT；
5. 生成 technical/deployment audit；
6. 若失败，生成 failure attribution；
7. 明确下一项最小判别实验；
8. 所有报告写入独立 aggregation commit。

### 17.4 结果推送与交付

- 每轮同步 GitHub；
- training commit 与 aggregation commit 分开记录；
- 报告、配置、分析脚本、日志索引、manifest 和 checksums 必须进入版本控制；
- 大型 checkpoint/latent 可保留在不可覆盖的远端根或压缩归档，但 GitHub 中必须保存 URI、大小和 SHA-256；
- 完整 stdout/stderr 必须交付，公开前去除凭据；
- 失败轮次不得删除或改写；
- bug fix 不得覆盖旧根，必须新建版本和 run ID；
- 不得 amend 已经用于训练的 commit。

---

## 18. 结果驱动的迭代规则

每一轮严格执行：

1. **观察事实**：只使用原始 receipts、固定 endpoint 和因果 rows。
2. **判定层级**：区分工程失败、诊断失败和科学失败。
3. **冻结证据**：生成 hash、inventory、报告和 Git commit。
4. **提出互斥假设**：列出能够解释结果的最小根因集合。
5. **选择最小判别实验**：只改变一个主要机制。
6. **重新预注册**：固定新假设、改动、门槛与决策树。
7. **fresh-root 运行**：旧结果只读保留。
8. **独立复算**：报告由 raw rows 重建，而非人工摘录。
9. **通过则推进，失败则继续归因**。

整个循环必须保持：

- 数据与因果评测口径不漂移；
- 成功门槛不因结果降低；
- 不能用 train loss 下降替代 scientific success；
- 不能用中间最佳 checkpoint 替代 fixed endpoint；
- 不能在反复查看同一 test 后仍称其 sealed；
- 如果 dev-final 被打开并参与决策，下一轮必须建立新的 sealed final；
- 每次修改都要说明它针对哪个已观测根因。

“持续迭代直到成功”指持续推进有证据支持的最小实验，而不是保证某一机制必然成功。若证据否定当前机制，应预注册新的机制分支；无论结果是否成功，都不得降低门槛或制造假阳性。

---

## 19. 预注册决策树

| 阶段结果 | 结论 | 下一步 |
| --- | --- | --- |
| Phase 1A 技术失败 | 无科学结论 | 修复技术问题，原协议 fresh-root 重跑 |
| Phase 1A 未找到 endpoint | 锁定预算下未证明 full-DL 可达 | 做已知 R11 latent 的 bridge-distance 最小实验 |
| Phase 1A 8/8 成功 | query-level full-DL endpoint 可达；非 state-level | 直接进入 Phase 2，不启动 Phase 1B |
| Phase 2 工程 smoke 失败 | 标签生成管线无有效科学结论 | 修复工程问题，用 fresh shard 重跑 |
| Phase 2 前 8 条 smoke 技术失败 | 标签生成管线无有效科学结论 | 修复工程问题，用 fresh shard 重跑 |
| Phase 2 的 64 条核心 bank 覆盖不足 | 最低 MVP teacher bank 不完整 | 分析失败 sample，不静默过滤或降到少于 64 条 |
| Phase 2 任一固定成员 oracle gate 失败 | 该 64/128 条 bank 未形成完整可读 teacher 集 | 停止在 Phase 2；不得替补或只挑成功项进入 3A |
| Phase 2 时间门通过 | 预计 128 条加 Phase 3/报告仍在约 30 小时窗口内 | 使用同一固定顺序扩展到 128 条 |
| Phase 2 时间门未通过 | 规模固定为 64 条；不是科学失败 | 停止扩 bank，进入 64 条 Phase 3 MVP |
| Phase 2 锁定 bank 完整、3A 单样本失败 | writer/unroll/gradient 技术或参数化问题 | 单 transition 梯度与 endpoint 单元测试 |
| 3A 单样本成功、小集合失败 | 容量或 teacher 坐标不一致 | 最小 K-teacher/canonicalization 实验 |
| 3A train 成功 | query-level oracle supervision 在锁定的 64/128 条训练子集可学 | 进入 3B；不得声称完整训练集、泛化或长期记忆 |
| 3A train 失败 | 当前 shared writer 未能摊销 oracle labels | 依次做 1 条、8 条、小集合最小归因 |
| 3B train 失败 | flow 参数化或训练/推理不一致 | 检查解析目标与 scheduler 合约 |
| 3B normal 通过、donor/constant 失败 | 假阳性 | 不推进，定位通用 code 或泄漏 |
| 3B train 与训练内 controls 通过 | MVP 可学习性诊断通过 | 未来再激活 Phase 1B、held-out、多 seed 与 rollout |
| 未来多 seed、held-out、controls、rollout 全通过 | R11_new state-level 范围内科学通过 | 启动全新 sealed confirmatory run |
| confirmatory run 通过 | 可复现且非假阳性的成功 | 推进更完整的 Picture Memory 任务 |

---

## 20. 推荐的实际执行顺序

### 里程碑 M1：Phase 1A 最小桥接

- 固定少量 F1 targets；
- 完整冻结 DreamLite；
- 只优化 $x_T$；
- 与 canonical R11 采用同一 Reader gate；
- 确认 full-DL endpoint 是否可读。

### 里程碑 M2：Phase 2 Query-Level Oracle Bank（64/128 条）

- 不新增数据，使用当前锁定 train split；
- 直接复用 Phase 1A 固定前 8 条作为工程与吞吐 smoke，并计入最终 bank；
- 每条保存 $x_T^\star$、$z_t^\star$、$I_t^\star$、receipts 与哈希；
- 最低完成固定 64 条；只有第 24.4 节时间门通过才扩展到固定 128 条；
- 最终要求所选 64/128 条 100% 有明确终态，不静默过滤失败样本；不声称覆盖完整 train split。

### 里程碑 M3：Phase 3A 过拟合阶梯

- 1 个 transition；
- 8 个 transitions；
- 64 条核心 bank；
- 若时间门通过，再验证 128 条扩展 bank；
- 训练内 reset/donor/constant controls；
- 只判断锁定训练子集可学习性。

### 里程碑 M4：Phase 3B 原生训练

- 先通过 flow 数值单元测试；
- 使用独立 $\epsilon$；
- 锁定官方 scheduler contract；
- 固定 endpoint 和因果门；
- 与 3A、QA-only baseline 配对比较。

### 里程碑 M5：未来 state-level 科学确认

- 激活 Phase 1B multi-query state probes；
- 新 sealed split；
- 多个预注册 seeds；
- 无 checkpoint selection；
- 完整 causal controls；
- 独立报告复算；
- GitHub 与远端完整交付。

---

## 21. 最终润色后的实验表述

> 直接以冻结 Reader 的下游 QA loss 训练 DreamLite writer 时，模型必须同时搜索“什么 latent 能被 Reader 读取”与“怎样由 event 生成这种 latent”。当前 R11_new MVP 将两者拆开：Phase 1A 在完整冻结的 DreamLite–VAE–Reader 链路下仅优化 initial/noise latent $x_T$；Phase 2 不新增数据，而是对现有 train split 中预先锁定的 64 条核心样本重复求解，并仅在时间门通过时扩展到 128 条，保存 query-level $x_T^\star$、$z_t^\star$ 与 $I_t^\star$；Phase 3A/3B 再训练只接收 source state、event 与独立噪声的共享 U-Net/LoRA writer。该路线首先回答 oracle supervision 是否让 writer 在锁定训练子集上变得可学。即使 MVP 通过，也只有未来补做完整数据覆盖、Phase 1B、held-out、多 seed、递归 rollout 与完整因果控制后，才可称为 state-level、可复现且非假阳性的 Picture Memory 科学成功。

---

## 22. 一句话结论

当前 R11_new MVP 的核心是先完成最小闭环：

$$
\boxed{
\mathrm{Query\mbox{-}level\ Oracle\ Bank}
\rightarrow
\mathrm{Shared\ Event\mbox{-}conditioned\ Writer}
}
$$

Phase 1A 证明 query-level full-DreamLite reachability，Phase 2 在现有 train split 的固定 64/128 条子集上构建显式 oracle supervision，Phase 3A 验证该训练子集可学，Phase 3B 恢复原生生成机制。Phase 1B 被延后而非被证明不需要；因此 MVP 结果不得越级解释为完整训练集、state-level 或 long-term memory。

---

## 23. Phase 0 锁定实例：Phase 1A v1

本节是在观察任何 R11_new 训练结果前形成的首轮可执行实例。机器可读镜像为：

    configs/experiments/r11_new_frozen_dreamlite_oracle_phase1a.json

若本节、机器可读配置、训练入口或运行 manifest 不一致，控制器必须 fail closed，不得解释训练结果。

### 23.1 本轮只回答的问题

本轮只检验以下诊断命题：

> 在 R6 至 R8 已采用的 source-anchored DreamLite 四步更新律下，仅优化 initial noise latent，是否能为固定 F1 query 找到 Reader 可读的 full-chain endpoint？

本轮不检验 state-level multi-query memory，不构建训练集 oracle bank，也不训练共享 writer。只有 Phase 1A 的技术门与 8-target query-level 诊断门均通过后，才可直接进入 Phase 2；当前不插入 Phase 1B。

### 23.2 固定父件与数据

- Git 基线父提交：`47cebc97e59ffed00571d4bb67fbb08933b3f8d6`。
- 激活父件：canonical R11 `comparison.json`。
- 激活父件 SHA-256：`f8b048f9cbe9fd4df9460043297904b5c9d476f386d6844d12fd4a5f8f636bb5`。
- 激活条件：canonical R11 的 8 个固定 F1 target 全部通过 direct-VAE-latent 诊断，但 formal success 仍为 false。
- train SHA-256：`24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184`。
- dev SHA-256：`8b167df38022a631d4e631d3c0d66e9fca74171f4224fec436030d6650047303`。
- 目标集合：与 canonical R11 完全相同的 8 个 F1 segments。
- 目标 payload SHA-256：`6198beb3a3758fd7df912c6956bc05eac0ace8603708f37147826c65a4d61845`。

### 23.3 固定模型链路

- previous state：RGB `127/255` 均匀画布的冻结 VAE posterior-mean model latent。
- event：目标 F1 segment 中唯一可见的 SET event text。
- condition：DreamLite-mobile 原生 edit prompt 与冻结内部 condition encoder；允许按固定 source/event 缓存，但必须保存 tensor hash。
- U-Net：直接加载未经新增 PEFT/LoRA 包装的 DreamLite-mobile base U-Net，全部冻结。
- Scheduler：沿用仓库已经数值校验的 DreamLite `FlowMatchEulerDiscreteScheduler` 合约。
- 推理步数：4。
- effective sigma schedule：`[0.5, 0.375, 0.25, 0.125]`。
- 初始混合：source-anchored，第一步状态为 `0.5 * source + 0.5 * x_T`。
- Decoder：冻结 DreamLite VAE，使用既有 model-latent scale/shift 与 unit-RGB clamp。
- Reader：冻结 Qwen3-VL-4B-Instruct。
- 唯一可训练对象：一个 unconstrained FP32 `x_T` tensor。
- `x_T` 初始化：使用既有 event-noise generator，由全局 seed 0、source episode ID 和 source turn ID 确定的标准高斯噪声；不得由 query、choices、答案或 target index 派生。

### 23.4 固定优化与 endpoint

- 每个 target 独立运行，不共享可训练 latent。
- optimizer：Adam。
- learning rate：`0.05`。
- weight decay：`0.0`。
- schedule：constant。
- optimizer steps：`256`。
- gradient clipping：不启用。
- restart/seed：首轮只使用 seed 0；多 seed 属于未来 state-level / confirmatory 要求，不能由首轮结果事后挑选。
- training views：四个 fixed forward-cyclic choice views，各暴露 64 次。
- checkpoints：`0, 64, 128, 192, 256`。
- primary endpoint：raw step 256；禁止 best-checkpoint selection。
- endpoint audit：四个 fixed reverse-cyclic choice views。
- reset：不执行 event update，直接解码同一 blank previous-state latent；在 M0 与 endpoint 均保持完全相同。

### 23.5 固定门槛与解释边界

每个 target 必须先通过 exact technical gate：

- 256 个连续 optimizer receipts；
- 每步恰好执行四个冻结 DreamLite denoising steps；
- 每步 `x_T` 梯度 finite、nonzero；
- observed effective sigma 与锁定 schedule 一致；
- 只有 `x_T` 为 FP32 trainable；
- U-Net、condition encoder、VAE 和 Reader 参数全部冻结且无 parameter gradient；
- 固定 checkpoint、trajectory、endpoint、图片、日志、环境、snapshot binding 和 artifact hash 齐全。

技术门通过后，query-level reachability gate 仍沿用既有可复算口径：

- normal endpoint mean CE 相对 M0 至少下降 20%；
- 四个 reverse-cyclic views 全部改善；
- accuracy 至少提高 0.25；
- `DiD(normal, reset) < 0`。

八个 target 均通过时，只能报告 **Phase 1A query-level full-chain reachability 诊断通过**。它不等于 state-level oracle、共享 writer 或 Picture Memory 科学成功。

### 23.6 技术预检与运行顺序

部署锁定到 Inspire 实例 `vlm-r3-h200x2-live-20260717`。结果根必须位于 `/inspire/ssd/`，启动时至少保有 50 GiB 可用空间，并由 suite 级互斥锁阻止重复进程；HDD 不得承载本轮结果。

1. 本地完成纯 CPU 单元测试与 source-only fail-closed 检查。
2. 提交并推送训练代码，记录完整 40 位 commit。
3. 在目标实例创建该 commit 的独立 clean checkout。
4. 使用 target 0 运行一次 `technical-preflight`：执行完整 forward/backward，但不执行 optimizer step、不计算科学通过。
5. 预检通过后，在全新目录运行 target 0 的正式 256 步实验。
6. 若 target 0 技术有效，则继续其余 7 个 target；诊断失败也必须保留，不能静默删样本。
7. 八个目标完成后冻结 raw artifacts，由独立聚合脚本复算结果并生成报告。

---

## 24. 结果前路线修订：跳过 Phase 1B 的 MVP

本节由用户在任何 R11_new 模型训练结果产生前明确提出，因此属于预注册路线修订，不是观察结果后的门槛调整。它覆盖本文中“Phase 1B 是 Phase 2 强制前置门”的旧表述；Phase 1A v1 的样本、优化器、步数、endpoint 和诊断门槛全部保持不变。

### 24.1 当前唯一推进链

$$
\boxed{
\mathrm{Phase\ 1A}
\rightarrow
\mathrm{Phase\ 2}
\rightarrow
\mathrm{Phase\ 3A}
\rightarrow
\mathrm{Phase\ 3B}
}
$$

- Phase 1B 当前不启动，也不阻塞 Phase 2。
- 不新增或改写原始数据集。
- Phase 2 只对现有锁定 train split 的确定性 64/128 条子集生成派生 oracle artifacts，不做全量转换。
- Phase 3 首先只回答 shared writer 能否拟合这些 query-level oracle labels；结论限定到所选训练子集。

### 24.2 Phase 2 的直接进入条件

只有 Phase 1A 出现以下结果才进入 Phase 2：

- 8 个固定 F1 target 均技术有效；
- 8 个 target 均通过锁定的 query-level reachability gate；
- 聚合结果由原始 256-step receipts 与 16-cell endpoint rows 独立复算；
- aggregation decision 为 `proceed_to_phase2_oracle_bank_mvp`。

若不足 8/8，不降低门槛、不选择性过滤失败 target，也不直接批量生成 bank；按第 19 节执行 bridge-distance 最小诊断。

### 24.3 Phase 2 的数据与样本边界

Phase 2 的候选池只来自哈希已锁定的现有 train split：以 `pairing_seed=0` 调用 `build_r5_family_pools(...)`，取 `F1` 中 `len(events)=1` 且 `R5Segment.segment_id` 唯一的 segment；本节中的 `sample_id` 精确定义为该 `segment_id`。随后复用 Phase 1A/R10 已锁定的结果无关顺序，按以下键升序排列：

    sha256("R10-VisualAlignment-LowerBound\x1f20260831\x1fF1\x1f" + segment_id), segment_id

由此在任何 oracle 优化前冻结并提交：

- `bank128`：排序后的前 128 个唯一 sample；
- `bank64`：`bank128` 的前 64 个，必须是严格前缀；
- `calibration8`：`bank64` 的前 8 个，必须与 Phase 1A 的 8 个固定 target 完全相同，并直接复用其技术有效 oracle artifacts 与真实 wall-clock；
- 候选 F1 segment 数量：`7504`，segment ID 全部唯一；
- `calibration8` ID SHA-256：`08c25bbb753e7ffb3a0fd760d0bbf079b113f1db12be9eba4af1505ad57e86ff`，full payload SHA-256：`6198beb3a3758fd7df912c6956bc05eac0ace8603708f37147826c65a4d61845`；
- `bank64` ID SHA-256：`5cbd99fdc537f67cba311ed39144516735d0da149e87095565118b162a872fcc`，full payload SHA-256：`d7d3a3d12182fd3169c5b9b5127617f9c1c5b81462a94c2d8afccb256973d98a`；
- `bank128` ID SHA-256：`c762b52c2b71ba8b977b6bec339a9586ef000440021c8bdf38bef28006d99f37`，full payload SHA-256：`6b817ffdc488df1925294aa6169e8d33cb738877432fb9da46b2844aec6a3665`。

禁止依据 Phase 1A/2 的 CE、梯度、耗时、oracle gate、答案类别、图片观感或“容易成功”程度改变名单；耗时只允许通过第 24.4 节公式决定最终规模为 64 还是 128。若样本技术失败，保留失败终态，只允许在修复明确工程错误后以同一 ID、新 commit/new root 重跑；不得用名单外样本替补后仍声称原 bank 完整。

对每个被选中的 train sample/query，保存至少：

- `sample_id`、原始 split 与输入 artifact 哈希；
- $x_{T,i}^\star$；
- 完整 Frozen-DreamLite endpoint $z_i^\star$；
- Reader 实际接收的 RGB $I_i^\star$；
- 固定 M0、raw endpoint、优化 receipts、checkpoint 与技术门；
- `supervision_scope=query_level`；
- `state_level_validated=false`。

Phase 2 首先把 Phase 1A 的固定 8 个技术有效、诊断通过结果封装为 `calibration8`，不重复求解；随后使用完全相同的锁定 solver 从第 9 条继续，至少完成 `bank64`。只有最终选择的 64/128 条 **全部同时通过 technical gate 与固定 query-level oracle gate**，且 raw artifacts 可哈希复算，Phase 2 才通过并允许进入 Phase 3。任何一条 oracle gate 失败都属于 Phase 2 诊断失败，必须停止在 Phase 2；不得只挑成功项训练 writer。当前 MVP 不要求、也不得报告 train split 100% coverage。

### 24.4 64/128 规模与约 30 小时时间门

30 小时是本轮从 Phase 1A technical-preflight 启动到 Phase 3 报告完成的规划窗口，不是科学门槛；不得通过降低 256-step oracle solver、修改 gate、跳过 artifact 或筛掉慢/失败样本来追赶时间。

固定规模规则如下：

1. 64 条是最低 MVP，不允许临时缩减到 8、16、32 或其他规模后宣称 Phase 2 完成。
2. Phase 1A 的 `calibration8` 串行完成、8/8 技术有效且 query-level gate 通过后，直接使用八个正式 controller roots 的每条真实 wall-clock seconds 计算 nearest-rank $t_{p90}$：对 8 个耗时升序排列并取第 $\lceil 0.9 \times 8 \rceil=8$ 个，即最大值；失败重跑和启动开销保留在已消耗时间中。
3. 从 technical-preflight 的锁定 `program_clock_start_utc` 到第 8 条完成，记真实已消耗小时为 $h_{elapsed}$。
4. 为 Phase 3A/3B 与最终复算报告固定预留 9 小时，并对剩余 oracle 生成时间乘以 1.15 安全系数。
5. 对 $N\in\{64,128\}$ 分别计算：

$$
H_N
=
h_{elapsed}
+
\frac{1.15 \times (N-8) \times t_{p90}}{3600}
+
9
$$

仅当 $H_{128}\leq30$ 时选择 128；否则选择 64。规模决策必须写入 `bank-size-decision.json`，保存八条原始耗时、$t_{p90}$、已消耗时间、$H_{64}$、$H_{128}$、`planning_window_feasible` 和最终 `selected_bank_size`，并在任何第 9 条 oracle 启动前落盘、哈希和提交到阶段报告。若 $H_{64}>30$，仍锁定最低规模 64，但必须将 `planning_window_feasible=false` 与修订 ETA 主动报告，不能继续声称约 30 小时可完成；不得静默降规模或更改 solver。若平台中断造成 wall clock 不可比较，默认选择 64 并标记时间投影不可判定。

### 24.5 Phase 3 的最小可学习性问题

Phase 3A 依次执行单样本、8 样本和最终锁定的 64/128 条 bank 过拟合阶梯。共享 writer 写入时只能接收 source/previous latent、event 与独立噪声，禁止接收 query、choices、answer、sample ID 或每样本可训练 code。主监督标签是 $z_i^\star$，$x_{T,i}^\star$ 只作为 oracle 搜索证据，不作为默认 clean target。

Phase 3A 通过时，只能报告：

> query-level oracle latent supervision 在锁定的 64/128 条训练子集上可被共享 writer 摊销学习。

只有 Phase 3A 通过后才进入 Phase 3B，以独立于 sample ID 的新噪声训练原生 flow-matching 目标。Phase 3B 的 scheduler 符号、权重、训练步数、LoRA 范围、seed 和 gate 必须在其训练代码提交前另行写入机器配置。

### 24.6 明确不能得出的结论

即使 Phase 1A、Phase 2、Phase 3A 和 Phase 3B 的 MVP 全部通过，仍不能仅凭这些结果声称：

- oracle latent 表示完整 state，而非 query/answer-specific code；
- writer 能回答同状态的未见 query 或 paraphrase；
- writer 在 held-out sample/entity/value 上泛化；
- writer 已覆盖或学会完整 train split；
- 系统具有长期 recurrence、OVERWRITE/CLEAR 或抗干扰能力；
- Picture Memory 已达到最终可复现科学成功。

这些结论需要未来重新激活 Phase 1B，并完成 held-out、多 seed、因果 controls、rollout 与 sealed confirmatory run。

## 25. Phase 0 工程纠错：运行时 sigma 数值校验一致性

首轮真实 technical-preflight 使用提交 `a6cba6d508ecd864ba07d391733a546d2937f980`，执行一次 backward、零 optimizer steps 后被技术门拒绝。实际 scheduler 返回 `[0.4999999701976776, 0.375, 0.25, 0.1249999925494194]`；checkpoint 保存了这些未经舍入的实测值，但后续校验使用了与名义十进制列表的精确等值，和同一提交中 sampler/forward 已有的浮点校验不一致。

这次修复只统一工程校验，不改变第 23、24 节的科学定义：

- 配置与 manifest 中的名义 sigma 仍精确锁定为 `[0.5, 0.375, 0.25, 0.125]`。
- 运行时实测 sigma 与名义值比较，统一继承首次运行前即存在的 `math.isclose(rel_tol=2e-6, abs_tol=2e-6)`；长度必须为 4，各项必须为有限数值，禁止布尔值。
- checkpoint、receipt、summary 中继续保存实测原值，禁止 round、替换为名义常量或改写旧产物。
- 同一运行的 payload、hash record 等对同一实测序列的声明必须精确相同；容差不能用来掩盖跨产物不一致。
- DreamLite、scheduler 计算、source interpolation、优化器、256 步预算、数据、reset/choice-view 控制与科学门均保持原定义。
- 首轮终态继续保留为技术失败，reachability 未评估。修复须单元测试、提交推送，并在新目录重新进行 target 0 technical-preflight；不能直接将旧失败目录改判通过。
- 30 小时规划时钟继续以首次预检 controller 的 `started_at_utc=2026-09-05T07:18:59.891307+00:00` 为起点，工程诊断与重跑耗时计入预算。
