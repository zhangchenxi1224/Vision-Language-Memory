# Vision Learnable Memory 项目汇报稿

> 版本：2026-07-20  
> 当前代码基线：`94bfc6c532392d7c1304b42dea2ab2b867ce7345`  
> 全量训练证据基线：`10bde565d30d119a68e8460757d979b1c35e1b8f`  
> 报告口径：项目全链路实现、实验方案迭代、已落盘实验数据与当前决策建议  
> 数学渲染：Markdown + KaTeX；行内公式使用 `\(...\)`，行间公式使用 `$$...$$`

---

## 阅读提示与结论边界

本报告把证据状态严格分为三类：

- **已实现**：代码、配置、测试或运行脚本已经落盘，但不自动等价于科学结论成立。
- **技术门通过**：梯度链、冻结边界、确定性、恢复等工程合约通过。
- **科学门通过/未通过**：只有达到预注册阈值且工件完整时才作判断。

截至本报告生成时，`dreamlite-diagnostics-dev128-lr3e5-20260720-DRAFT.md` 仍含待回填字段。因此，其中预注册的 checkpoint sweep、逐 LoRA block 梯度和低学习率对照只作为**下一阶段方案**，不伪装成已经得到的结果。

## 内容导航

1. [执行摘要](#1-执行摘要)
2. [项目目标、研究假设与边界](#2-项目目标研究假设与边界)
3. [全链路输入、状态、路由与输出](#3-全链路输入状态路由与输出)
4. [数据集构造与防泄漏设计](#4-数据集构造与防泄漏设计)
5. [训练算法与代码实现](#5-训练算法与代码实现)
6. [评测、因果控制与统计](#6-评测因果控制与统计)
7. [可复现工程与 Inspire 执行链](#7-可复现工程与-inspire-执行链)
8. [实验方案迭代与全部关键结果](#8-实验方案迭代与全部关键结果)
9. [DreamLite 全量实验与失效诊断](#9-dreamlite-全量实验与失效诊断)
10. [当前结论、风险与下一步](#10-当前结论风险与下一步)
11. [实现索引与证据入口](#11-实现索引与证据入口)

---

## 1. 执行摘要

### 1.1 项目做成了什么

项目已经实现一套“**用可学习图像/潜变量承载动态记忆，由冻结视觉语言模型读取**”的完整研究系统：

1. 将多轮偏好记忆抽象为 `SET / OVERWRITE / CLEAR / NOOP` 四类状态转移，并支持 `event / query / mixed` 三种回合。
2. 实现两类可学习 updater：轻量 FiLM-ConvGRU 视觉状态机，以及 DreamLite-mobile 四步扩散式 LoRA updater。
3. 实现 whole-episode BPTT、direct-latent 递归、decode/re-encode 消融、event 间 detach 负对照、NOOP 更新/跳过对照和可学习初始状态消融。
4. 实现冻结 Qwen3-VL Reader 的 target-only 历史目标与正式 listwise-choice 目标，并保持从 Reader 图像输入到 updater 参数的梯度。
5. 实现 QA-only 与 Teacher-assisted 两条不可混淆的训练谱系；Teacher-assisted 又分为蒸馏阶段和卸载 teacher 后的 QA 阶段。
6. 实现 synthetic、R3 micro、R3 formal、PrefEval 适配、固定 choice permutation、OOD、因果控制、bootstrap 与 Holm 校正。
7. 实现双 GPU、严格确定性、不可变提交/模型/数据指纹、原子 checkpoint、精确 resume、后台 stage、terminal、自动报告和 SHA256 工件链。

### 1.2 当前最重要的实验结论

| 结论 | 证据 | 判定 |
|---|---|---|
| 文本显式状态可被冻结 Reader 正确读取 | oracle text `61/64 = 95.31%`；blank `16/64 = 25%` | 输入/Reader 基线通过 |
| 轻量 updater 可以获得梯度且位级复现，但正式可达性不足 | D2R `77/128`；R2c canonical `62/128`，门槛均为 `116/128` | 科学门未通过 |
| listwise 训练对 choice 位置泄漏更稳健，但没有解决语义可达性 | R2a/R2b 位级复现；R2c rotate view `58/128` | 技术门通过、性能门失败 |
| 事件前缀可以控制视觉状态语义 | transductive event-prefix probe held-out permutation `100%` | 只证明通道存在，不是正式方法结果 |
| 完整 DreamLite 训练链可以在 2×H200、3–4 小时内跑通 | 256 steps、2048 episodes、3:07:14、无 OOM/NaN | 工程执行成功 |
| 当前 DreamLite 训练不能证明有效学习 | loss 无清晰收敛；无 step-0 dev；256/256 步裁剪 | 科学结论不可判定 |
| 极端梯度更像优化轨迹/尺度问题，不像简单的数据难度问题 | 最大梯度 `27,106,712`；loss-gradient 相关弱；21 个组成特征 Holm 后均不显著 | 当前优先工作假设 |

### 1.3 一句话项目状态

**全链路工程闭环已经建立，核心梯度路径和审计体系已验证；当前瓶颈不再是“能不能跑”，而是“视觉状态可达性与优化尺度是否足以支持稳定、可泛化的记忆学习”。**

---

## 2. 项目目标、研究假设与边界

### 2.1 研究问题

给定按时间到达的可见事件 (e_t)、查询 (q_t) 与候选集合 \(\mathcal A_t=\{a_{t,1},\ldots,a_{t,4}\}\)，系统需要把当前动态语义状态压缩进视觉状态，并在未来查询时恢复正确答案：

$$
z_t = \mathcal U_{\Delta\theta}(z_{t-1}, e_t, \epsilon_t),
\qquad
\hat y_t = \arg\max_j \mathcal R(z_t,q_t,a_{t,j}).
$$

其中：

- \(z_t\) 是持续视觉状态；
- \(\mathcal U_{\Delta\theta}\) 是只训练 LoRA 或轻量 updater 的状态更新器；
- \(\mathcal R\) 是冻结的 Qwen3-VL Reader；
- \(\epsilon_t\) 是按 `global_seed + episode_id + turn_id` 确定的事件噪声。

### 2.2 核心可证伪假设

| 假设 | 可观测量 | 失败意味着什么 |
|---|---|---|
| H1：视觉通道能承载答案语义 | oracle/visual-control/event-prefix 上界 | Reader 或 renderer 流形不可达 |
| H2：更新后的状态确实影响预测 | reset、blank、shuffle、state-swap 的配对差值 | 模型可能只利用 query/choice 偏置 |
| H3：多事件梯度能穿过中间状态 | intermediate-state gradient、detach 负对照 | 只有末次更新在学习 |
| H4：正式目标与正式评测一致 | listwise-choice、四视图、位置分层 | target-only 训练可能学到位置捷径 |
| H5：训练改善可跨 checkpoint 与固定 dev 重复 | init/64/128/192/256 的 CE、accuracy、margin | 单一末端 loss 不构成学习证据 |

### 2.3 当前研究边界

当前正式范围固定为：oracle router、MCQ 分类、冻结 Reader、DreamLite direct-latent 状态、conditioning stop-gradient、LoRA-only 训练。自动路由、开放生成、全 conditioning BPTT 被明确延期，避免在主机制未成立前扩张问题空间。

---

## 3. 全链路输入、状态、路由与输出

![全链路输入输出](D:/2026WorkExperience/VisonLearnableMemory/reports/project_report_assets/07_end_to_end_io.png)

*图 1　从 JSONL episode 到潜变量、RGB 状态、Reader 分数、梯度与审计工件的完整输入输出链。该图由 image2 生成，公式与状态转移按当前代码实现校正。*

### 3.1 输入契约

| 输入层 | 字段/张量 | 是否进入模型 | 作用 |
|---|---|---:|---|
| Episode | `episode_id`、`turns` | 仅路由所需字段 | 决定噪声与回合顺序 |
| Event | `event_kind`、`event_text` | 是 | 调用 updater |
| Query | `text`、四个 `choices` | 是 | 调用 Reader |
| Label | `target_index` | 只在 loss 边界 | 不拼入 Reader prompt |
| Hidden ledger / sidecar | before/after semantic state | 否 | 仅生成验证或 teacher cache |
| Model/data metadata | split、topic、pair IDs | 否 | 审计、统计、配对 |

递归 schema 检查会拒绝任何 `hidden_ledger`、`preference_ledger` 或等价字段进入模型可见 JSON。

### 3.2 Oracle 路由语义

设当前状态为 \(z\)，updater 为 \(U\)，Reader 为 \(R\)：

$$
\begin{aligned}
\text{event}:&\quad z' = U(z,e), && \text{无 Reader loss},\\
\text{query}:&\quad z' = z, && \ell = R(z,q,\mathcal A),\\
\text{mixed}:&\quad z' = U(z,e), && \ell = R(z',q,\mathcal A).
\end{aligned}
$$

`NOOP` 默认仍是一次 updater 调用；只有显式 `noop_policy=skip` 才跳过。纯 query 永远是严格只读分支。

### 3.3 内部状态与外部输出

| 模块 | 输入 | 内部状态/输出 | 形状或格式 |
|---|---|---|---|
| 轻量 updater | event text + \(H_{t-1}\) | ConvGRU hidden | `[1,64,64,64]` |
| 轻量 renderer | \(H_t\) | RGB image | `[1,3,1024,1024]`，bilinear |
| DreamLite updater | \(z_{t-1},e_t,\epsilon_t\) | model-space latent \(z_t\) | batch 1，1024 对应 latent 网格 |
| Frozen VAE | \(z_t\) | Reader RGB \(I_t\) | `[1,3,1024,1024]` |
| Deterministic resize | \(I_t\) | Qwen processor input | `[3,256,256]` |
| Qwen processor | resized RGB | visual pixels/grid | `pixel_values=[256,1536]`，`grid=[1,16,16]` |
| Reader loss | image + query + choices | \(s_j,L\) | 标量 loss 与四个 option scores |
| Trainer | loss + optimizer state | metrics/checkpoint | JSONL、PT、JSON、SHA256 |

### 3.4 输出工件

一次正式训练至少产生：`manifest.json`、`environment.txt`、`metrics.jsonl`、`checkpoint-*.pt`、`last.pt`、`summary.json`、`state_gradient_audit.json`；外层 stage 再产生 `running.json`、stdout/stderr、原子 `terminal.json` 和报告 SHA256 清单。

---

## 4. 数据集构造与防泄漏设计

![数据状态机与构造顺序](D:/2026WorkExperience/VisonLearnableMemory/reports/project_report_assets/02_dataset_state_machine.png)

*图 2　语义状态机、split-before-expansion、反事实/NOOP 配对与 delayed probe。图中公式对应 `r3_synthetic.py` 的真实转移函数。*

### 4.1 语义状态机

每个 entity-slot 只有 `unset / active / cleared` 三种规范状态。设状态为 \(S=(\sigma,v)\)，事件值为 \(v'\)：

$$
T(S,k,v')=
\begin{cases}
(\mathrm{active},v'), & k\in\{\mathrm{SET},\mathrm{OVERWRITE}\},\\
(\mathrm{cleared},\varnothing), & k=\mathrm{CLEAR},\\
S, & k=\mathrm{NOOP}.
\end{cases}
$$

实现不会根据 `OVERWRITE` 的文字隐式推断历史；状态真值只由显式事件种类和 sidecar 转移得到。

### 4.2 “先分组、后展开”的防泄漏顺序

R3 先把不可分割的 semantic group 分配到 split，再展开两个语义反事实和 clean/NOOP 流：

$$
G_{\text{split}}
\longrightarrow
\{s_0,s_1\}\times\{\mathrm{clean},\mathrm{noop}\}
\longrightarrow
\text{episodes}.
$$

因此同一语义组、同一 entity、同一 template family、反事实 mate 和 clean/NOOP mate 不可能跨 split。验证器会检查 reciprocal link、候选集合一致、目标差异/相同、mixed 后立即 delayed probe，以及 sidecar 连续性。

### 4.3 数据规模与覆盖

| 数据集 | train | dev | test-ID | test-OOD | 关键用途 |
|---|---:|---:|---:|---:|---|
| synthetic-v2 main | 5,000 | 500 | 1,000 | 1,000 | mechanism-v1 主语料 |
| synthetic-v2 set-only | 5,000 | 500 | 1,000 | 1,000 | 独立生成 curriculum 消融 |
| R3 pilot | 1,000 | 500 | 1,000 | 1,000 | R3 试运行 |
| R3 formal | 5,000 | 500 | 1,000 | 1,000 | 正式 DreamLite |
| R3 Set8 micro | 8 states | — | 32 gate views | — | 最小 SET 科学门 |
| R3 Transition16 micro | 16 histories | — | 64 delayed views | — | 四类事件与 mixed/separate |

test-OOD 均分为 `heldout_entity`、`heldout_topic`、`heldout_paraphrase`、`heldout_length`；长度外推为 9–16 turns，其余 ID episode 为 4–8 turns。

### 4.4 Topic、候选与延迟探针

主语料覆盖 `color / material / drink / style / meal / music` 六类 topic；held-out topic 为 `fragrance / lighting`。四个候选在 semantic group 层构造并确定性轮转，使目标位置接近均匀分布。

mixed 回合必须在下一次 updater 前出现同 target、同 choice multiset 的纯 query：

$$
y_{\mathrm{mixed},t}=y_{\mathrm{query},t+\Delta},
\qquad
\mathcal A_{t}=\mathcal A_{t+\Delta}.
$$

这把“同回合即时回答”和“写入后持续保持”分开。

### 4.5 Analysis sidecar 与 teacher 边界

模型可见 JSONL 只含事件、问题、候选和标签。train-only sidecar 另存每个 updater turn 的 `before_state / after_state`，用于：

- 验证状态机连续性；
- 建 teacher cache；
- 生成 query-independent state ID；
- 事后按事件/状态归因。

状态 ID 使用规范 JSON 与域分离哈希：

$$
\mathrm{state\_id}
=\operatorname{SHA256}\!\left(
\texttt{vlm.semantic\_state.v1}\,\Vert\,\operatorname{CanonicalJSON}(S)
\right).
$$

### 4.6 PrefEval 适配

PrefEval 适配固定 20 topics，把 explicit、implicit-choice、implicit-persona 三种 form 绑定到同一 base pair；支持 `oracle-sparse` 与 `forced-write` 的 \(k\in\{0,2,5,10\}\)。固定 topic split 为 16 train topics / 4 OOD topics；锁定快照为 730/82/188 base pairs，对应三 form 的 2190/246/564 records。forced-write 正式子集为 200 base pairs。

适配器拒绝把 implicit preference 等特权字段输入模型；不同 protocol 和 \(k\) 共享同一 base-pair/form 噪声键，避免对比时扩散噪声漂移。

---

## 5. 训练算法与代码实现

### 5.1 轻量视觉 updater

![轻量视觉 updater](D:/2026WorkExperience/VisonLearnableMemory/reports/project_report_assets/03_lightweight_updater.png)

*图 3　Hash + BiGRU + 16 模式 DCT-II spatial writer + FiLM-ConvGRU + RGB head。*

#### 5.1.1 确定性 token hash 与 BiGRU

文本先用固定正则切词，再用 BLAKE2b 映射到词表：

$$
h(w)=2+\operatorname{BLAKE2b}(w)\bmod(V-2).
$$

`0` 保留给 padding，空文本回退到 token `1`。embedding 经一层双向 GRU，连接双向末状态得到 \(f_t\in\mathbb R^{256}\)。这一设计不依赖外部 tokenizer，可在 CPU/API smoke 中稳定复现。

#### 5.1.2 事件条件的空间写入

模型固定 16 个低频、零均值、单位 RMS 的 DCT-II 基：

$$
\Phi_k(x,y)=
\cos\!\left(\frac{\pi u_k(x+1/2)}{N}\right)
\cos\!\left(\frac{\pi v_k(y+1/2)}{N}\right).
$$

事件图为：

$$
E_t=\tanh\!\left(
W_0f_t+\frac{1}{\sqrt{16}}
\sum_{k=1}^{16}a_{t,k}\Phi_k
\right).
$$

这样比把一个全局向量广播到整张图更有表达力，同时固定基不会增加随空间尺寸增长的自由参数。

#### 5.1.3 有界 FiLM 与 ConvGRU

$$
X_t=\tanh\!\left[
\left(1+0.1\tanh\gamma_t\right)E_t
+0.1\tanh\beta_t
\right].
$$

ConvGRU 使用：

$$
\begin{aligned}
r_t &= \sigma(W_r*[X_t,H_{t-1}]+b_r),\\
u_t &= \sigma(W_u*[X_t,H_{t-1}]+b_u),\\
\widetilde H_t &= \tanh(W_h*[X_t,r_t\odot H_{t-1}]+b_h),\\
H_t &= (1-u_t)\odot H_{t-1}+u_t\odot\widetilde H_t.
\end{aligned}
$$

update gate bias 初始化为 \(b_u=-1\)，所以初始写入比例约为：

$$
\sigma(-1)\approx 0.269.
$$

该修订是轻量模型迭代中最关键的稳定性改动之一：既保留旧状态，又给 overwrite/mixed 足够的初始写入容量。

#### 5.1.4 RGB 输出与静态对照

`Conv3×3 → GELU → Conv1×1 → Sigmoid` 把 \(H_t\) 转为 64×64 RGB，再 bilinear 到 1024×1024。另实现 `StaticLearnedInitialImage`：所有事件为 identity map，只学习一张全局图，用于排除“单张静态提示图即可完成任务”的可能。

### 5.2 DreamLite 四步可微 updater

![DreamLite 递归与 BPTT](D:/2026WorkExperience/VisonLearnableMemory/reports/project_report_assets/01_dreamlite_recurrent_bptt.png)

*图 4　DreamLite direct-latent 递归、stop-gradient conditioning、冻结 Reader 与 LoRA 梯度路径。*

#### 5.2.1 官方 pipeline 与训练 wrapper 的分工

官方 `DreamLiteMobilePipeline` 保留为数值/推理参考；训练使用窄化的 `DifferentiableDreamLiteMobileSampler`，只保留 edit 模式、显式噪声、四个 denoising steps 与 latent 输出。训练 wrapper 不使用 PIL 输出、不使用隐式噪声、不做 CFG 或模型 offload。

每个 event 从显式噪声 \(x_{\tau_0}=\epsilon_t\) 开始，执行四步：

$$
\begin{aligned}
m_t^{(i)} &= [x_{\tau_i},z_{t-1}]_{\text{spatial}},\\
\widehat\epsilon_i &=
\mathcal U_{\theta,\Delta\theta}(m_t^{(i)},\tau_i,c_t),\\
x_{\tau_{i+1}} &=
\operatorname{SchedulerStep}(x_{\tau_i},\widehat\epsilon_i,\tau_i),
\quad i=0,1,2,3,\\
z_t &= x_{\tau_4}.
\end{aligned}
$$

U-Net 输入在宽度维拼接噪声 latent 与 source latent；输出只取当前 latent 对应部分。每次 event 都重置 scheduler 的可变 step index。

#### 5.2.2 Conditioning stop-gradient 与 direct-latent BPTT

内部 DreamLite Qwen condition 仍看见由当前状态解码出的图像，但该辅助分支显式停止梯度：

$$
c_t=operatorname{sg}\!\left[
Q_{\mathrm{int}}
\left(
\operatorname{VAE}^{-1}(\operatorname{sg}(z_{t-1})),
p(e_t)
\right)
\right].
$$

递归梯度从下一状态经 U-Net 的 source-latent 空间拼接路径回到 \(z_{t-1}\)，因此不是“整个 event 间 detach”。这是当前 milestone 的准确边界：**状态 BPTT 成立，conditioning 分支的 BPTT 被有意停止。**

#### 5.2.3 冻结模块与 LoRA 白名单

VAE、DreamLite 内部 text encoder、U-Net base weights、外部 Qwen3-VL Reader 全部冻结。只有 attention 的：

```text
to_q, to_k, to_v, to_out.0
```

注入 LoRA；默认 rank 4，\(\alpha=r\)，dropout 0。训练启动前检查所有 trainable name 必须属于 `lora_A / lora_B`，或在显式消融时属于 `initial_state`。任何意外 base weight 可训练都会 fail closed。

#### 5.2.4 初始状态、递归与负对照

- 正式默认：把均匀中性灰 RGB 127 图像编码一次为 \(z_0\)。
- `learn_initial_state`：把 blank latent 变为参数，属于预注册消融。
- `fixture`：彩色确定性图，只允许技术 probe。
- `direct_latent`：直接令 \(z_t\) 成为下一步状态。
- `decode_reencode`：\(z_t\to I_t\to\operatorname{VAEEncode}(I_t)\)，用于 RGB bottleneck 消融。
- `detach_between_events`：第二次及以后 update 使用 \(\operatorname{sg}(z_{t-1})\)，作为 BPTT 负对照。

### 5.3 冻结 Qwen Reader 的目标函数

#### 5.3.1 Joint continuation tokenization

目标文本不是独立 tokenize，而是附在 chat generation prompt 后共同 tokenize。实现要求 joint token 序列的前缀与原 prompt token 完全相同，否则立即报错，避免 BPE 上下文造成 train/eval 不一致。

#### 5.3.2 历史 target-only CE

对正确答案 token 序列 \(Y\) 计算：

$$
L_{\mathrm{target}}
=-\frac{1}{|Y|}\sum_{n=1}^{|Y|}
\log p(y_n\mid I,q,y_{<n}).
$$

它只对正确选项做一次 forward，计算便宜，但与四选一评测不完全对齐，因此 R3 只把它保留为历史 R1 诊断。

#### 5.3.3 正式 listwise-choice CE

对每个候选分别做 teacher-forced continuation：

$$
\ell_j=-\frac{1}{|Y_j|}\sum_{n=1}^{|Y_j|}
\log p(y_{j,n}\mid I,q,y_{j,<n}),
\qquad
s_j=-\ell_j.
$$

再做四类交叉熵：

$$
L_{\mathrm{QA}}
=-\log\frac{\exp(s_y)}{\sum_{j=1}^{4}\exp(s_j)}.
$$

四个 option forward 都保留到同一图像的 autograd。`target_index` 只在外层 loss 边界使用，从不写进 Reader prompt。

#### 5.3.4 Choice permutation

训练使用 cyclic4 family，评测使用 disjoint reverse-cyclic4 family。若轮转量为 \(r\)：

$$
\mathcal A'=(a_r,a_{r+1},a_{r+2},a_{r+3}),
\qquad
y'=(y-r)\bmod 4.
$$

这要求语义目标在 candidate 位置变化时保持稳定，并暴露位置偏置。

### 5.4 QA-only 与 Teacher-assisted 训练谱系

![两类训练谱系](D:/2026WorkExperience/VisonLearnableMemory/reports/project_report_assets/04_training_regimes.png)

*图 5　QA-only 单阶段与 Teacher-assisted “distill → checkpoint → 卸载 teacher → fresh AdamW → QA” 两阶段。*

#### 5.4.1 QA-only

只加载 event/query JSONL、DreamLite、Reader 与 \(L_{\mathrm{QA}}\)。任何 teacher manifest、sidecar、calibration 或 teacher-derived loss 都被禁止。

#### 5.4.2 Teacher-assisted distillation

Teacher cache 为每个 query-independent semantic state 保存三类目标：latent \(z_t^*\)、Reader RGB \(I_t^*\)、Qwen query-free visual feature \(F_t^*\)。学生 latent 先按 batch/channel 在空间维标准化：

$$
\operatorname{Norm}(z)_{b,c}
=\frac{z_{b,c}-\mu_{b,c}}
{\sqrt{\operatorname{Var}_{b,c}(z)+10^{-6}}}.
$$

三个原始分量为：

$$
\begin{aligned}
d_z &= \operatorname{SmoothL1}(\operatorname{Norm}(z_t),\operatorname{Norm}(z_t^*)),\\
d_I &= \operatorname{SmoothL1}(I_t,I_t^*),\\
d_F &= \operatorname{mean}\!\left[1-\cos(F_t,F_t^*)\right].
\end{aligned}
$$

用训练前冻结的正标量 \(a,b,c\) 校准：

$$
L_{\mathrm D}
=\frac{1}{3}\left(
\frac{d_z}{a+10^{-6}}+
\frac{d_I}{b+10^{-6}}+
\frac{d_F}{c+10^{-6}}
\right).
$$

蒸馏阶段不调用 query loss；QA 阶段必须从 distill checkpoint 只载入 trainable weights、完全卸载 teacher，并新建 AdamW/RNG 训练状态。teacher lineage 不能改名成 QA-only。

#### 5.4.3 Teacher attribution controls

- `correct`：state ID identity mapping。
- `shuffled`：按排序 state ID rotate-one，形成无固定点置换。
- `random-moment-matched`：在每个 channel 内确定性独立置换，精确保留每通道矩。

这些对照用来判断提升来自真实 state teacher，还是仅来自额外正则/尺度。

### 5.5 Episode loss、BPTT 与梯度累积

一个 episode 内对所有 query 的已归一化 query loss 做等权平均：

$$
L_{\mathrm{episode}}
=\frac{1}{Q}\sum_{q=1}^{Q}L_q.
$$

它避免长答案 token 或更多 query 的 episode 获得额外权重。默认 8 个 episode 做梯度累积；若 epoch 末是不满 8 的尾组，先按 nominal 8 反传，再用 \(8/n_{\mathrm{actual}}\) 重标梯度，保证 optimizer step 仍代表实际 episode mean。

### 5.6 裁剪、优化器与新增模块级诊断

全局原始梯度范数为：

$$
g_{\mathrm{raw}}
=\sqrt{\sum_{p\in\Theta_{\mathrm{train}}}\|\nabla_p L\|_2^2}.
$$

裁剪因子及裁剪后梯度：

$$
\alpha=\min\!\left(1,\frac{c}{g_{\mathrm{raw}}+\varepsilon}\right),
\qquad
g'_p=\alpha g_p,
\qquad c=1.
$$

当前 `94bfc6c` 新增 opt-in 诊断，按以下互相完备的轴统计 pre/post-clip norm、平方范数占比、参数范数、实际 AdamW update norm 与 update/weight ratio：

- stage：`down_blocks / mid_block / up_blocks / other`；
- projection：`to_q / to_k / to_v / to_out / other`；
- factor：`lora_A / lora_B / initial_state / other`；
- cross：`stage|projection|factor`。

实际更新比例定义为：

$$
\rho_{\mathrm{update}}
=\frac{\|\theta_{k+1}-\theta_k\|_2}{\|\theta_k\|_2}.
$$

### 5.7 Checkpoint 与精确恢复

checkpoint 原子保存 trainable weights、AdamW state、epoch、episode cursor、optimizer step、Python/NumPy/Torch/CUDA RNG 和 manifest。resume 时验证 manifest compatibility，截断超过恢复点的 metrics，恢复 cursor 与 RNG；DL-S 比较 continuous 与 resume 的 checkpoint/optimizer/trace 哈希。

---

## 6. 评测、因果控制与统计

![固定 checkpoint 与因果控制](D:/2026WorkExperience/VisonLearnableMemory/reports/project_report_assets/05_evaluation_causal_controls.png)

*图 6　固定 dev checkpoint sweep、blank/reset/shuffle/state-swap、反事实/NOOP 配对与 semantic-group bootstrap。*

### 6.1 主指标

对于四个 option scores \(s_1,\ldots,s_4\)：

$$
\operatorname{Accuracy}=\mathbb 1[\arg\max_j s_j=y],
$$

$$
\operatorname{CE}=-\log\frac{\exp(s_y)}{\sum_j\exp(s_j)},
$$

$$
\operatorname{Margin}=s_y-\max_{j\ne y}s_j.
$$

同时报告 choice-view consistency、topic/form macro accuracy、event subtype、probe role、target position、clean/NOOP agreement 和 mixed delayed persistence。

### 6.2 因果控制

同一 query/choice 下比较：

- `standard`：真实更新后的 \(z_t\)；
- `blank`：固定空白图；
- `reset`：每次读取前回到 \(z_0\)；
- `shuffle`：打乱状态/像素或状态对应关系；
- `state-swap`：换成配对 episode 的状态。

若指标方向定义为越大越好，则：

$$
\Delta M=M(\mathrm{standard})-M(\mathrm{control}).
$$

对 CE 则改用 \(\Delta\mathrm{CE}=\mathrm{CE}_{\mathrm{standard}}-\mathrm{CE}_{\mathrm{control}}\)，负值代表 standard 更好。

### 6.3 数据配对

- 反事实 pair：同候选集、不同最终 target，要求 \(y_A\ne y_B\)。
- clean/NOOP pair：相同语义 target 与对应 choice，要求 \(y_{\mathrm{clean}}=y_{\mathrm{noop}}\)。
- choice views：同 base query 的四个 permutation 不是四个独立样本，必须先聚合。

### 6.4 统计协议

正式统计按 `semantic_group_id` 做 paired cluster bootstrap，固定 \(B=10{,}000\)、seed 2026、95% CI。多个主题/事件/检查点比较使用 Holm 校正。小样本 stratum 只用于定位，不独立升级为科学结论。

### 6.5 预注册门槛示例

R2c/D2L exact64 要求每个 fresh replica：

- canonical 与 left-rotate-one 均至少 `116/128`；
- 每个 target position 均至少 `28/32`；
- canonical mixed 至少 `20/24`；
- matched clean/NOOP prediction-text agreement 至少 `60/64`；
- 2000/2000 步均有有限正 updater gradient；
- 两个 fresh process 的关键 payload 位级一致。

---

## 7. 可复现工程与 Inspire 执行链

![可复现执行 DAG](D:/2026WorkExperience/VisonLearnableMemory/reports/project_report_assets/06_reproducibility_dag.png)

*图 7　提交/数据/模型/配置指纹、六级技术门、原子 stage 状态、双 GPU 与报告工件链。*

### 7.1 不可变输入

正式运行绑定：git commit、dirty status、DreamLite revision、Qwen revision、模型 snapshot manifest SHA、train/dev SHA、配置 SHA、preflight SHA 和 launcher configuration SHA。严格模式缺任一模型 snapshot manifest 会拒绝启动。

### 7.2 六级技术门

R3 规定的顺序为：

```text
R3-R0 → R3-S0 → G4-L → G5-L → G6-L → DL-S
```

- `R3-R0`：确定性 resize 前向等价、严格 backward 重复性与 native CUDA reference 容差。
- `R3-S0`：环境/数据/模型/路由静态合约。
- `G4-L`：一个 event 的 listwise 梯度链。
- `G5-L`：两个 event 的非 detach BPTT。
- `G6-L`：detach 负对照；前向 loss 应相同，中间状态梯度语义应不同。
- `DL-S`：连续训练与 checkpoint-resume 等价。

任何一级失败都会阻断后续科学 stage。

### 7.3 严格 CUDA 确定性

进程启动前固定 `PYTHONHASHSEED`、`CUBLAS_WORKSPACE_CONFIG`、线程数与 tokenizer parallelism；运行时启用 deterministic algorithms、关闭 TF32、使用 math-only SDPA。诊断路径的 CE 使用与交叉熵等价的 FP32：

$$
L=\operatorname{logsumexp}(\ell)-\ell_y,
$$

因为 CUDA `NLLLoss` backward 在 strict determinism 下不可用。生产默认仍可使用 `F.cross_entropy`。

### 7.4 Qwen resize 的精确修复

历史 H200 DL-S 在第一次 backward、零 optimizer step 时失败：`upsample_bicubic2d_aa_backward_out_cuda` 没有 deterministic 实现。不能简单 `do_resize=False`，因为那会把 Qwen grid 从 `[1,16,16]` 放大为 `[1,64,64]`，visual tokens 增加 16 倍。

修复保持 CUDA torchvision bicubic-antialias 前向完全不变；因 resize 对输入是线性算子，其 Jacobian 与像素值无关，所以 backward 在单线程 CPU 以 FP32 重放同一线性算子的 adjoint，再传回原 device/dtype。这不是 STE 或 surrogate gradient。

### 7.5 双 GPU 与性能边界

正式 trainer 要求两张可见 GPU 且设备不同：DreamLite/LoRA 在 `cuda:0`，Qwen Reader 在 `cuda:1`。当前实现不是 DDP；第二张 GPU 解决模型放置和峰值显存，不会并行多个 episode。增加 GPU 数量不会自动提升单臂吞吐。

### 7.6 Stage 与证据包

后台 launcher 为每个 stage 保存不可变 worker input、`running.json`、完整 stdout/stderr，并用原子 `terminal.json` 宣告终态。自动报告将 source artifacts、terminal、图表与机器摘要复制到独立目录，再生成 `artifacts.sha256`。报告生成器不会因为进程退出成功就伪造 scientific gate pass。

---

## 8. 实验方案迭代与全部关键结果

### 8.1 从机制框架到 R3 的迭代时间线

| 日期 | commit | 关键变化 | 产生的判断 |
|---|---|---|---|
| 07-15 | `8d45504` | 初始化可微视觉记忆框架 | 建立主假设与 probe 骨架 |
| 07-16 | `641a40c`–`3faaf59` | mechanism-v1、数据/PrefEval、消除 query-only choice 泄漏 | 数据与 Reader 基线可审计 |
| 07-16 | `d49efe4`–`950faa0` | DCT spatial writer、动态稳定、update bias、强制 final-step gate | 早停“通过”被识别为无效，正式末步仍失败 |
| 07-16 | `88b3afe`–`712f451` | 严格确定性、Qwen grid 对齐、D2R exact64 | 技术复现成立，target-only 性能不足 |
| 07-17 | `b39d164` | 正式 listwise exact64 | R2a/R2b 通过，R2c 性能门失败 |
| 07-17 | `fea241c`–`1ac4a90` | R3 协议、H200 runtime、teacher cache 与 provenance | 进入 DreamLite 技术门 |
| 07-17 | resize amendment | deterministic bicubic backward 修复 | 归类为 pre-optimizer 实现故障，不是科学失败 |
| 07-17–19 | `b35f495`–`e5e1b76` | 固定 DAG、报告绑定、Set8 evaluator 审计 | 旧技术证据不得授权新科学 stage |
| 07-19 | `10bde56` | 可审计 Inspire GPU job | 直接开展 3–4 小时 DreamLite 全量机制观察 |
| 07-20 | `94bfc6c` | per-module optimizer diagnostics | 为千万级梯度根因定位补齐代码能力 |

### 8.2 Reader 与数据可达性基线

固定 64 个 comparison queries 的结果：

| 条件 | 正确数 | Accuracy | 阈值/用途 | 判定 |
|---|---:|---:|---|---|
| oracle text state | 61/64 | 95.31% | ≥90% | 通过 |
| query-only blank | 16/64 | 25.00% | ≤30% | 通过 |

oracle text 证明冻结 Reader 能理解显式状态；blank 接近四选一随机，说明 query/choice 本身没有足以完成任务的泄漏。

### 8.3 轻量 updater 的结构与超参数迭代

#### 8.3.1 早期与 sweep

| 运行/设置 | steps | final train | best train | dev | 门 |
|---|---:|---:|---:|---:|---|
| `formal-3faaf59` | 2000 | 40.63% | 42.97% | 28.0% | 未通过 |
| `formal-d49efe4` | 2000 | 25.78% | 32.03% | 24.3% | 未通过 |
| `lr3e-4, clip1` | 2000 | 42.19% | 42.19% | 25.3% | 未通过 |
| `lr3e-4, clip5` | 2000 | 64.06% | 77.34% | 32.0% | 未通过 |
| `lr3e-4, clip10` | 2000 | 63.28% | 64.06% | 28.6% | 未通过 |
| `lr1e-3, clip5` | 2000 | 25.78% | 30.47% | 26.1% | 未通过 |

DCT spatial writer 提升了表达力，但超参数 sweep 仍显示明显不稳定与过拟合；最佳 dev 只有 32%。

#### 8.3.2 Update gate bias 与 final-step 规则

`formal-89eda56` 在 step 1900 达到 `117/128 = 91.41%` 并提前终止，旧逻辑曾把它标为通过；但预注册门要求**恰在最终第 2000 步**。修复为不早停后，`formal-950faa0` 的最终准确率只有 `57/128 = 44.53%`，因此科学门失败。

这次迭代的价值不是“模型从通过变失败”，而是纠正了选择偏差：中途峰值只作 trajectory diagnostic，不能替代固定终点。

### 8.4 D2R：target-only 的位级复现与可达性失败

两个 fresh process 完成 2000 steps，关键梯度、optimizer、RNG、trace 和 prediction payload 位级一致；`reproducibility_valid=true`。但每个 replica 只有：

- final canonical `77/128 = 60.16%`；
- 门槛 `116/128`；
- 2000 步均有正梯度；
- 1967/2000 步触发裁剪。

因此 D2R 是“**可复现的性能失败**”，而不是基础设施或随机性失败。

### 8.5 R2：listwise-choice 对齐

| 阶段 | 预算 | 结果 | 说明 |
|---|---:|---|---|
| R2a | 1 step × 2 replicas | 通过 | updater/image gradients 正且有限；payload 位级一致 |
| R2b | 100 steps × 2 | 通过 | 100/100 正梯度；93/100 裁剪；位级一致 |
| R2c/D2L | 2000 steps × 2 | 科学门失败 | 完整训练和复现成功，accuracy 未达阈值 |

R2c 每个 replica 的关键结果：

| 指标 | 结果 | 门槛 |
|---|---:|---:|
| canonical | 62/128 = 48.44% | 116/128 |
| left-rotate-one | 58/128 = 45.31% | 116/128 |
| canonical mixed | 3/24 | 20/24 |
| clean/NOOP agreement | 60/64 | 60/64 |
| positive updater-gradient steps | 2000/2000 | 2000/2000 |
| clipped steps | 1939/2000 | 记录项 |

训练 loss 的 first-100 mean 从 `6.9394` 到 last-100 mean `2.9263`，但 final step 又为 `15.1276`；这说明 loss 有下降信息，却没有转化为稳定、位置鲁棒的 MCQ 可达性。

### 8.6 视觉通道与 renderer 上界诊断

| Probe | Accuracy | 阈值 | 判定 | 能回答的问题 |
|---|---:|---:|---|---|
| target-selected learned images | 50.00% | 90% | 未通过 | 任意可学习图像能否直接控制 Reader |
| target-selected production RGB head codes | 70.31% | 90% | 未通过 | renderer manifold 是否足够可达 |
| visible event-prefix semantic codes，held-out permutations | 100% | 90% | 通过 | 事件文本前缀是否能编码答案语义 |

这些都是**目标监督或 transductive diagnostic**，不是 memory method、baseline 或 ablation。最有信息量的对比是：简单 target code/renderer 不足，但用可见 event prefix 构造的语义 code 可以跨候选位置达到 100%。因此瓶颈更像“如何从事件序列学到可读视觉码”，而不是 Reader 绝对无法读取视觉状态。

### 8.7 R3 DreamLite 技术门

历史提交 `1ac4a90` 的验证结果：

| Gate | 事件 | loss | 验证 |
|---|---|---:|---|
| G4-L | SET | \(6.7949\times10^{-6}\) | valid |
| G5-L | SET → OVERWRITE | 3.77427 | valid |
| G6-L | 同前向、event 间 detach | 3.77427 | valid |
| G5/G6 forward pair | — | absolute diff 0 | valid |

G4/G5/G6 证明一事件、两事件 BPTT 与 detach 负对照的技术语义成立。随后 DL-S 在第一次 deterministic bicubic backward 失败，发生在 optimizer step、metric 和 checkpoint 之前，因此按 amendment 归类为实现故障。`r3_pre_set8_launch_amendment.json` 记录后续 `176fc1e` 已得到“C0 passed through DL-S”，但在该 amendment 冻结时 Set8 optimizer steps 与科学 predictions 仍为 0；新修订又要求 final implementation commit 重跑六门后才能启动 Set8。

### 8.8 方案转向：从层级门改为限时全量机制观察

在轻量模型已经完成大量分层实验、DreamLite 技术链已经可执行的背景下，项目选择直接运行完整 DreamLite + Qwen Reader，保留 1024 分辨率、四步 denoising、whole-episode BPTT 与正式 R3 数据，只通过 `max_optimizer_steps=256` 将总时长约束在 3–4 小时。

该实验回答的是：

1. 完整机制是否能稳定执行；
2. 3–4 小时能消费多少正式数据；
3. loss、gradient、显存和吞吐表现如何；
4. 是否值得立即扩量。

它不替代三 seed、完整 epoch、ID/OOD 与 causal scientific gate。

---

## 9. DreamLite 全量实验与失效诊断

### 9.1 运行契约

| 项目 | 值 |
|---|---|
| run ID | `dreamlite-fullscale-formal-v1-seed0-20260720-10bde56` |
| Inspire | 2× NVIDIA H200 SXM 141 GiB；40 CPU；400 GiB RAM |
| image/runtime | `ngc-pytorch:25.02-cuda12.8.0-py3`；Python 3.12.3；CUDA 12.8 |
| commit | `10bde565d30d119a68e8460757d979b1c35e1b8f`，clean |
| data | formal train 5000；dev 500 |
| regime | `qa_only / qa / listwise-choice` |
| recurrence | direct-latent；不 detach；NOOP update |
| LoRA | rank 4；1,644,544 trainable parameters |
| optimizer | AdamW；LR \(10^{-4}\)；WD 0.01 |
| accumulation/clip | 8 episodes；global norm 1.0 |
| budget | 256 steps；2048 episodes；seed 0 |
| checkpoint | 每 8 steps，共 32 个周期 checkpoint |
| dev | 仅 step 256；8 episodes × 4 views |

### 9.2 工程结果

| 指标 | 结果 |
|---|---:|
| terminal | succeeded；`exit_code=0` |
| trainer wall time | 11,233.80 s = 3:07:14 |
| optimizer steps | 256 |
| consumed episodes | 2048/5000 = 40.96% |
| mean step time | 43.67 s |
| throughput | 0.1832 episode/s |
| peak VRAM | cuda:0 21.16 GiB；cuda:1 21.39 GiB |
| OOM / NaN / crash | 0 |

完整机制在时限内稳定跑通，显存远未达到 H200 上限；当前吞吐主要由四步 DreamLite、逐 episode 的四选项 Reader forward 和 strict math kernels 决定，而不是显存容量。

![训练 loss](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/figures/loss_total.png)

*图 8　256 个 optimizer step 的 train loss 与 EMA。*

### 9.3 Loss：没有清晰收敛

| 指标 | 结果 |
|---|---:|
| final train loss | 14.4677 |
| 全程 mean / median / SD | 14.3718 / 14.2466 / 3.0215 |
| first 32 mean → last 32 mean | 14.2365 → 13.9335（-2.13%） |
| first 32 median → last 32 median | 13.7783 → 14.0529（+1.99%） |
| OLS slope | -0.002179 / step |
| Pearson \(r\) | -0.0534 |
| only dev loss | 10.0845 |

均值略降但中位数反升，step 与 loss 几乎无稳定线性关系。唯一 dev loss 发生在 step 256，且只覆盖 8/500 dev episodes；没有相同 slice 的 step-0，因此不能声明泛化改善。

### 9.4 梯度：全程被 clip 主导

![梯度范数与裁剪](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/figures/gradient_norm_and_clip.png)

*图 9　裁剪前全局梯度范数、clip threshold 与 clip rate。*

| 指标 | 结果 |
|---|---:|
| clip trigger | 256/256 = 100% |
| median | 17,250.46 |
| P90 / P95 / P99 | 174,768 / 311,314 / 1,000,842 |
| max | 27,106,712（step 256） |
| groups over \(10^6\) | 3 |
| max / second max | 9.321 |
| step256 derived clip factor | \(3.689\times10^{-8}\) |

最后一步 gradient 比全程中位数约高 1571 倍，但 loss 14.4677 并非异常高，说明 clip 抑制了即时 loss 爆炸，却也意味着实际方向长期被单位范数投影主导。

### 9.5 数据构成与离线归因

分析按 seed 0 精确重建 shuffle，把 2048 个 episode 对齐到 256 个 8-episode accumulation groups。数据与 manifest SHA 完全一致，counterfactual、NOOP 与 mixed-delayed 结构检查全部通过。

| 维度 | 实际消费分布 |
|---|---|
| updates/episode | 2：1019；3：1029 |
| queries/episode | 2：1015；3：1033 |
| read form | mixed 1033；separate 1015 |
| topic | color 360；drink 377；material 361；meal 301；music 316；style 333 |
| delayed probe | 2048/2048 |
| event sequences | set→set 497；set→overwrite 395；set→noop→overwrite 377；set→set→noop 276；set→noop→set 251；set→clear 127；set→noop→clear 125 |

![事件组成与梯度](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/figures/diagnostic_event_update_gradient.png)

*图 10　事件类型、update 数和 group-level raw gradient 的交叉分析。*

### 9.6 哪些组成与梯度相关

以 \(\log_{10}g_{\mathrm{raw}}\) 为因变量，256 个 accumulation groups 为独立单位：

| 特征 | Spearman \(\rho\) | raw \(p\) | Holm \(p\) |
|---|---:|---:|---:|
| mean updater calls | +0.134 | 0.0316 | 0.6004 |
| SET count | +0.138 | 0.0272 | 0.5429 |
| NOOP count | +0.134 | 0.0316 | 0.6004 |
| CLEAR count | -0.108 | 0.0834 | 1.0000 |
| mixed episode count | -0.026 | 0.6813 | 1.0000 |
| group loss | +0.102 | 0.1032 | 1.0000 |
| optimizer step | +0.150 | 0.0165 | 0.3458 |

21 个探索特征经 Holm 校正后均不显著。SET、NOOP、3-update、turn 数又被生成规则结构性绑定，所以不能把未校正的小相关解释为独立因果来源。

### 9.7 训练阶段：梯度尾部在后期加重

![阶段梯度分布](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/figures/diagnostic_gradient_by_phase.png)

*图 11　四个 64-step 窗口的 raw gradient 分布。*

| steps | loss mean | grad median | grad P90 | grad max |
|---|---:|---:|---:|---:|
| 1–64 | 14.5385 | 18,556 | 99,787 | 1,039,446 |
| 65–128 | 14.6914 | 10,311 | 117,323 | 983,802 |
| 129–192 | 14.1135 | 16,821 | 114,662 | 2,908,102 |
| 193–256 | 14.1436 | 24,709 | 268,440 | 27,106,712 |

梯度四分位 Kruskal-Wallis \(p=0.03285\)，loss 对应 \(p=0.79132\)。Theil-Sen 对 \(\log_{10}g\) 的斜率为：

$$
\widehat\beta=0.001387\ \text{per step},
\qquad
\mathrm{CI}_{95\%}=[0.000266,0.002560].
$$

loss 未同步恶化而梯度尾部抬升，更支持“参数轨迹/反向尺度”工作假设。

### 9.8 哪些任务在学习

![类别学习趋势](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/figures/diagnostic_category_learning_trends.png)

*图 12　控制事件序列、mixed 比例和 choice rotation 后的 topic/sequence loss 趋势。*

| topic | step 1→256 modeled loss change | 95% bootstrap CI | 可判定学习 |
|---|---:|---:|---|
| color | +2.872 | [-4.517, +9.674] | 否 |
| drink | +2.587 | [-3.741, +9.054] | 否 |
| material | -0.322 | [-6.162, +5.420] | 否 |
| meal | -1.990 | [-10.760, +6.307] | 否 |
| music | -1.374 | [-7.501, +5.138] | 否 |
| style | -5.362 | [-12.470, +2.571] | 否 |

style 和 `set→set` 点估计方向最好，但 CI 均跨 0；没有固定 dev checkpoint accuracy/margin，当前不能宣称任一类别已经学会。

### 9.9 Step 256 异常组

step 256 的 8 个候选 episode：

| # | episode | topic | sequence | updates | queries | mixed |
|---:|---|---|---|---:|---:|---:|
| 1 | `r3-train-semantic-000103-s0-noop` | style | set→noop→overwrite | 3 | 2 | 0 |
| 2 | `r3-train-semantic-000262-s0-clean` | drink | set→overwrite | 2 | 2 | 0 |
| 3 | `r3-train-semantic-000139-s1-noop` | meal | set→noop→overwrite | 3 | 2 | 0 |
| 4 | `r3-train-semantic-000199-s1-clean` | color | set→overwrite | 2 | 2 | 0 |
| 5 | `r3-train-semantic-000831-s0-clean` | material | set→set | 2 | 3 | 1 |
| 6 | `r3-train-semantic-001046-s0-clean` | drink | set→overwrite | 2 | 3 | 1 |
| 7 | `r3-train-semantic-000463-s0-clean` | drink | set→set | 2 | 2 | 0 |
| 8 | `r3-train-semantic-001006-s0-noop` | material | set→set→noop | 3 | 2 | 0 |

该组平均 2.375 次 update，只有 2/8 mixed，没有 CLEAR；它并不是长 episode 或 mixed 特别集中。因此旧日志只能把根因缩小到这 8 条，不能把 group gradient 虚假分配给单个 episode。

### 9.10 三个诊断问题的当前答案

1. **哪类样本导致千万级梯度？** 尚不能归到一种事件或 topic；step 256 的 8 条在组成上普通。
2. **哪些任务已经在学习？** 没有可靠类别；所有 topic 的趋势 CI 都跨 0。
3. **问题来自梯度尺度、数据难度还是多轮状态更新？** 当前最支持优化轨迹/梯度传播尺度；多轮更新只有弱相关，数据难度与梯度关联也弱。

---

## 10. 当前结论、风险与下一步

### 10.1 已经可以确认

- 数据路由、状态机、split 防泄漏、反事实/NOOP 配对和 delayed probe 都有可执行校验。
- 冻结 Reader 的图像梯度可以传回 updater；一事件和两事件 direct-latent BPTT 技术上成立。
- 轻量和 DreamLite 两套 updater、两种 Reader loss、两种训练谱系与主要消融都已实现。
- 完整 DreamLite 训练可在 2×H200 上稳定执行，并能在 3–4 小时产生 256 steps 的完整审计工件。
- 当前 256-step run 处于持续裁剪、重尾梯度、无明确 loss 收敛的状态。

### 10.2 尚不能确认

- step 256 是否优于未训练初始化；
- 模型是否真正依赖写入后的视觉 state，而不是 query/choice 偏置；
- 千万级梯度由哪个 episode、哪个 U-Net stage、哪个 projection、LoRA A 还是 B 主导；
- 降低学习率是否只缩小 AdamW update，还是也改善固定 dev 表现；
- 任一 topic/event subtype 是否具有可重复泛化改善；
- ID/OOD、PrefEval 或多 seed 的正式科学结论。

### 10.3 第一优先级：复用 checkpoint，不先重跑长训练

对相同 stratified dev-128，用相同 reverse-cyclic4 四视图评测：

```text
init_seed0, step64, step128, step192, step256
×
standard, reset（至少 init 与 step256）
```

每个 base query 先聚合四 views，再按 semantic group 做 10,000 次 paired cluster bootstrap。主要判据：

$$
\Delta\mathrm{CE}_{256-0}<0,
\qquad
\Delta\mathrm{Accuracy}_{256-0}\ge 0,
\qquad
\Delta\mathrm{Margin}_{256-0}>0,
$$

且 CE 的 95% CI 上界小于 0。memory gate 对 `step256 standard - reset` 使用同样方向规则。

**停止规则**：若学习 gate 或 memory gate 未通过，不启动更长全量训练。

### 10.4 第二优先级：Step 256 的 8 条逐 episode 反向传播

固定 parent checkpoint、AdamW fresh state、seed 与 event noise，对 8 条候选逐条执行：

```text
zero_grad → run one episode → backward → record raw grads → discard grads
```

每条必须保存：

- episode loss、query-level loss、update/query/mixed 信息；
- `down/mid/up × q/k/v/out × LoRA A/B` pre-clip L2 norm 与 squared share；
- state/image/intermediate-state gradient；
- raw gradient tensor SHA；
- 不做 optimizer step，防止样本之间相互污染。

这样才能把 group-level \(27{,}106{,}712\) 分解为：

$$
g_{\mathrm{group}}
=\left\|
\frac{1}{8}\sum_{i=1}^{8}g_i
\right\|_2,
$$

并识别是单条 \(g_i\) 极端、多个样本同向叠加，还是跨 episode 抵消后的异常方向。

### 10.5 第三优先级：配对的一步 optimizer probe

从同一个 step256 checkpoint、同一批 8 episodes、fresh AdamW 启动两个臂，只改变 LR：\(10^{-4}\) 与 \(3\times10^{-5}\)。两臂的 pre-clip loss/gradient SHA 必须完全相同；比较：

$$
\frac{\|\Delta\theta_{3\times10^{-5}}\|_2}
{\|\Delta\theta_{10^{-4}}\|_2},
\qquad
\frac{\rho_{3\times10^{-5}}}{\rho_{10^{-4}}}.
$$

若两臂梯度相同且都被 clip，降低 LR 只能证明**实际更新变小**，不能声称“修复了梯度爆炸”。

### 10.6 第四优先级：64-step 低 LR 短臂

只有一步配对完整后，再从初始化跑 `lr=3e-5`、64 steps，并与原 `lr=1e-4` 的前 64 steps 对齐。报告 loss、raw/post-clip gradient、update/weight、固定 dev-128 standard/reset。只有固定 dev 指标更好，才把低 LR 作为正式候选。

### 10.7 因果评测与扩量决策

若 checkpoint learning gate 通过，再依次补：blank、reset、shuffle、matched state-swap。只有 standard 在 CE/margin 上稳定优于控制，才说明模型开始使用视觉状态。

扩量顺序建议：

1. 先完成 5000 presentations 的单 seed；
2. 再增加第二 seed；
3. 最后恢复 ID/OOD、三 seed 和 ablation matrix；
4. PrefEval 必须等 synthetic formal learning + memory gates 后解锁。

### 10.8 风险矩阵

| 风险 | 当前信号 | 严重度 | 缓解 |
|---|---|---:|---|
| 全程裁剪掩盖方向尺度 | 256/256 clipped | 高 | per-block + update/weight + 低 LR 配对 |
| 末步偶发极端梯度 | step256 为 second max 的 9.32× | 高 | 8 条逐 episode backward |
| train loss 与科学性能脱钩 | R2c loss 降但 accuracy 门失败 | 高 | 固定 checkpoint MCQ sweep |
| 状态未被 Reader 使用 | 尚无 formal reset/state-swap | 高 | 因果控制作为扩量前置门 |
| choice position 捷径 | 多次 position breakdown 不均 | 中 | disjoint permutation families + per-position gate |
| 确定性降低吞吐 | 0.183 episode/s | 中 | 先保持科学模式；另做非科学 profiler |
| teacher 泄漏/谱系混淆 | teacher sidecar 含真值 | 高 | query-independent state、train-only、QA stage unload |
| 工程 pass 被误写为科学 pass | 全量 run 无 scientific gate | 高 | terminal/report/scientific gate 分离 |

---

## 11. 实现索引与证据入口

### 11.1 代码规模快照

当前工作树包含约 50 个 `src` 文件、78 个 `scripts` 文件、49 个测试文件和 88 个显式 test 定义；主分支为 `codex/dreamlite-optimizer-diagnostics-20260720`。`reports/` 为当前未跟踪的本地报告目录，未改写用户已有源码变更。

### 11.2 模块—实现映射

| 主题 | 主要实现 |
|---|---|
| schema / oracle routing | `src/vision_memory/data/schema.py`；`src/vision_memory/training/episode.py` |
| synthetic-v2 | `src/vision_memory/data/generator.py`；`validation.py` |
| R3 formal data | `src/vision_memory/data/r3_synthetic.py` |
| Set8 / Transition16 | `src/vision_memory/data/r3_micro.py` |
| lightweight updater | `src/vision_memory/lightweight/model.py` |
| static image baseline | `src/vision_memory/training/lightweight.py` |
| DreamLite sampler | `src/vision_memory/dreamlite/differentiable_mobile.py` |
| direct-latent recurrence | `src/vision_memory/dreamlite/recurrent.py` |
| stop-gradient condition | `src/vision_memory/dreamlite/conditioning.py` |
| VAE latent codec | `src/vision_memory/dreamlite/latent_codec.py` |
| episode model | `src/vision_memory/training/dreamlite_model.py` |
| formal trainer | `scripts/train/dreamlite_episode.py` |
| Reader target/listwise loss | `src/vision_memory/reader/qwen3vl.py` |
| deterministic resize | `src/vision_memory/reader/deterministic_resize.py` |
| teacher state/cache | `src/vision_memory/teacher/state.py`；`cache.py`；`io.py` |
| teacher renderer/loss | `src/vision_memory/teacher/renderer.py`；`loss.py` |
| evaluation/statistics | `src/vision_memory/eval/` |
| exact determinism | `src/vision_memory/repro/determinism.py`；`probes.py` |
| checkpoint | `src/vision_memory/training/checkpoint.py` |
| PrefEval | `src/vision_memory/prefeval/`；`scripts/eval/prepare_prefeval.py` |
| Inspire orchestration | `scripts/inspire/`；`INSPIRE.md` |
| report generation | `scripts/reporting/render_training_report.py` |
| failure diagnosis | `runs/formal_reports/.../tools/analyze_training_failure.py` |

### 11.3 关键配置与指纹

| 工件 | 指纹 |
|---|---|
| current commit | `94bfc6c532392d7c1304b42dea2ab2b867ce7345` |
| full-scale commit | `10bde565d30d119a68e8460757d979b1c35e1b8f` |
| DreamLite source | `a6e20c8cc94027f37dd7c5a81b0b3b472aa18409` |
| DreamLite-mobile revision | `6695c3f4be230f0493fa5dbf78be3bc4d3bb2ab4` |
| Qwen3-VL-4B revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| formal train | `24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184` |
| formal dev | `8b167df38022a631d4e631d3c0d66e9fca74171f4224fec436030d6650047303` |
| formal manifest | `089beefa00b9b78149b3d7f4bd40cf802dc2c92a3757c04f30d9534bbdc51215` |
| full-scale metrics | `6531cbe0c42232cbb8e97b1deec8f418129a39e60e2f55b808f204fb04feb40e` |
| Reader resize contract | `r3-qwen-reader-1024-to-256-bicubic-antialias-cpu-adjoint.v1` |

### 11.4 报告与原始证据

- [DreamLite 全量训练中文正式报告](D:/2026WorkExperience/VisonLearnableMemory/reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56.md)
- [DreamLite 全量训练 HTML](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/report.html)
- [训练失效诊断 Markdown](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/diagnostic_report.md)
- [训练失效诊断 HTML](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/diagnostic_report.html)
- [机器可读训练摘要](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/metrics/report_summary.json)
- [机器可读诊断摘要](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/diagnostics/diagnostic_summary.json)
- [基线 SHA256 清单](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/artifacts.sha256)
- [诊断 SHA256 清单](D:/2026WorkExperience/VisonLearnableMemory/runs/formal_reports/dreamlite-fullscale-formal-v1-seed0-20260720-10bde56/diagnostic_artifacts.sha256)

### 11.5 本报告的 image2 图件

| 图 | 文件 | 内容 |
|---:|---|---|
| 1 | `07_end_to_end_io.png` | JSONL→route→latent/RGB→Reader→audit |
| 2 | `02_dataset_state_machine.png` | SET/OVERWRITE/CLEAR/NOOP 与数据展开 |
| 3 | `03_lightweight_updater.png` | 轻量 updater 公式链 |
| 4 | `01_dreamlite_recurrent_bptt.png` | DreamLite direct-latent BPTT |
| 5 | `04_training_regimes.png` | QA-only / Teacher-assisted |
| 6 | `05_evaluation_causal_controls.png` | checkpoint/因果控制/统计 |
| 7 | `06_reproducibility_dag.png` | 技术门与工件链 |

所有图件保存在：

```text
D:\2026WorkExperience\VisonLearnableMemory\reports\project_report_assets\
```

### 11.6 Markdown / KaTeX 自动验收

本稿使用 KaTeX `0.16.22` 严格模式逐式编译，并检查本地媒体与证据链接：

| 检查项 | 结果 |
|---|---:|
| 行间公式 | 34 条，编译失败 0 |
| 行内公式 | 61 条，编译失败 0 |
| 未配对公式定界符 | 0 |
| 单美元公式写法 | 0 |
| 图片 | 12 张，缺失 0 |
| 绝对本地链接 | 8 条，缺失 0 |

验证脚本保存在 `reports/project_report_work/validate_katex.cjs`，会忽略 fenced/inline code 后再提取 `$$...$$` 与 `\(...\)`，对每个表达式调用 `katex.renderToString(..., {throwOnError: true, strict: "error"})`。

---

## 结语

项目已经从“构想一个视觉记忆机制”推进到“拥有严格输入边界、两套 updater、三类训练目标、因果评测、确定性 GPU 运行和可审计报告”的完整研究工程。下一阶段不应再用更长训练替代诊断：先用现有 checkpoint 建立学习与 memory 因果证据，再把 step 256 的 8 条候选分解到逐 episode、逐 LoRA block 和实际 optimizer update。只有这三层证据同时支持，扩展到完整 5000 presentations、多 seed 与 OOD 才有科学价值。
