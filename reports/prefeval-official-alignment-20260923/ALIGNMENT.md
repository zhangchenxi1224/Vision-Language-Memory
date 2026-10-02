# PrefEval 官方 pipeline 对齐：核对结论与已落地数据

日期：2026-09-23。用户要求直接核对官方材料并对齐，不再调用 codex-with-chatgpt。

**结论：旧实验使用官方 PrefEval 样本，但不是官方 PrefEval SFT 的复现。**
数据来源相同，不等于对话构造、训练目标、可训练参数或评估协议相同。
本次完成官方代码核对、数据导出和 CPU 验证；没有进行新的 GPU 训练，
没有新的模型性能结果。既有 Plan12 失败证据保持不变。

## 核对依据

- [PrefEval 项目主页](https://prefeval.github.io/)。
- [论文 §3.7 与 Appendix A.13/A.14](https://arxiv.org/html/2502.09597v1)。
- [官方 SFT 实现](https://github.com/amazon-science/PrefEval/blob/50795054b5ff5f418d2b768a331d71e480f93331/SFT/train_sft.py)。
- [官方 SFT 说明](https://github.com/amazon-science/PrefEval/blob/50795054b5ff5f418d2b768a331d71e480f93331/SFT/readme.md)。
- [官方上下文构造](https://github.com/amazon-science/PrefEval/blob/50795054b5ff5f418d2b768a331d71e480f93331/utils/common_utils.py)。
- [官方隐式对话处理](https://github.com/amazon-science/PrefEval/blob/50795054b5ff5f418d2b768a331d71e480f93331/utils/implicit_utils.py)。

已读取本地官方源码，并通过 `git ls-remote origin HEAD` 确认当前上游 HEAD
仍为 `50795054b5ff5f418d2b768a331d71e480f93331`。导出清单固定该版本。

## 逐项比较

| 项目 | 官方材料/代码 | 既有 DreamLite PrefEval 实验 |
|---|---|---|
| 原始数据 | 20 个主题、1,000 个基础偏好—问题对、三种表达形式 | 原始来源对齐，但主要使用显式样本的自定义子集 |
| 对话输入 | 显式偏好与助手回复，或完整隐式对话，再插入无关对话并提出原问题 | 显式披露变为写入事件；choice 省略 acknowledgment；persona 旧 adapter 用隐藏偏好定位单个 user span |
| 主题划分 | 原代码主题顺序，seed=42，16 个训练主题、4 个评估主题 | seed=2026 的另一组 16/4；另加去重组、概念隔离、dev/ID-test、pilot |
| 训练样本量 | 发布文件共 1,000 条；820 train / 180 eval | 登记 541 个 full train 语义组、64 个 pilot train；实际多轮诊断反复优化 40 个状态 |
| SFT 监督 | 已发布 `response_to_q` 自然语言回答；只监督最终答案 token | 完整偏好字符串复述、人工应用场景候选排序及其组合 |
| 被优化对象 | Mistral-7B 的共享 LoRA 参数 | 每个状态独立的 `latent_fp32`；Reader/VAE 冻结；本阶段共享 Writer 更新为零 |
| 干扰上下文 | SFT 用指定对话流的前 0/5/10 个 user-assistant 交换 | 旧 adapter 按样本抽取无关交换，并解释为 0/2/5/10 次额外写入 |
| 主评估 | 生成回答的偏好遵循；分类任务另报 MCQ accuracy | 原始 MCQ 子集 + 自定义 exact recovery、排序、CLEAR、整状态及离线链指标 |
| 多偏好与冲突 | 论文有单独动态偏好实验 | 现有 K1/K2/K4、SET/CLEAR/RETAIN 协议是本项目扩展，不能当作官方 SFT 协议 |

使用官方已经筛选并发布的数据，不需要重新生成一遍原始偏好—问题。
我们没有复现作者的人工筛选全过程，也不能把后续自己编写的复述题/应用题
称为经过官方构造流程认证的数据。

## 官方 SFT 具体做法，以及发布物的边界

论文 §3.7 的方法是：先用无干扰上下文的 Reminder 回答作为监督，再在训练输入中
插入 0、5、10 轮无关对话。它们分别构成总长 2、7、12 轮的设置。
这应理解为不同上下文长度的训练条件，不应擅自合成一个三倍样本的混合实验。
官方启动脚本默认运行 10 轮干扰条件。后续可先选这个主条件，避免同时扩张分支。

发布代码明确读取 `*_2turn.json` 的 `response_to_pref` 和 `response_to_q`；
训练 query 本身不附加 Reminder。输入依次为：

```text
user: 偏好
assistant: response_to_pref
[N 个 user-assistant 无关交换]
user: 原始问题
assistant: response_to_q  ← 只在这一段计算答案监督
```

代码实际加载 `mistralai/Mistral-7B-Instruct-v0.2`，8-bit 权重加 LoRA，
rank=16、alpha=32、dropout=.05、学习率 2e-4、默认 1 epoch。
**但 README/答案文件名写的是 Mistral 8x7B，而论文描述的是 Mistral-7B 教师。**
本次复用的是“官方发布的回答”，不声称已独立确认这些回答的实际生成模型。
这些是模型生成的训练目标，也不意味着每一条都经过人工判定为完美答案。

官方 SFT collator 使用 `filtered_inter_turns.json` 最后四段 conversation 拼接后的前缀；
benchmark 公共工具使用全部 conversation 拼接后的前缀。两者不同，已分别保留。
一轮是一个 user-assistant 交换，而不是一条 message。

对齐过程中还发现：发布的 SFT 与显式 benchmark 有 30 条偏好字符串相差
`absolutely `。问题与行顺序一致。新导出保留各自原文，不静默替换；
详见 `data/wording-differences.json`。

训练侧最终答案 mask 使用模型自己的 chat template，完整 token 序列一次编码，
排除历史和问题。精确 Mistral 输入文本复现另由 `render_released_mistral` 提供。
新的模型模板 collator 是可移植实现，不宣称与官方原 collator 的字符串偏移实现逐 token 等价。
真实 tokenizer 的模板边界、长度与设备训练仍须在接入相应模型时验证；不静默截掉答案。

## 对目前 latent 问题的解释

旧实验优化的是每个样本独有的状态：

`z_i* = argmin_z L(R_frozen(decode(z), q_i), target_i)`。

官方 SFT 更新跨样本共享的语言模型参数，让模型学习“从对话推断偏好并用于回答”。
我们的目标则应是学习跨样本共享的 Writer，使其在新对话上形成可被 Reader 使用的 RGB 记忆。
这三件事不能互相替代。

因此，当前任务偏移会使 latent 向有限的复述措辞和候选排序适配，而无法证明它学到了
跨问题通用的“主题—偏好—应用”关系。这是需要检验的机制解释，不是已证明的单一病因。
Plan12 M 在训练复述形式上为 941/968，在排除的两个形式上为 142/176；
后者 34 个失败中有 14 个恰好回答了另一个活动槽的偏好。
K4 全部恢复成功仅 2/12，但这些结果不能证明 latent 的信息容量装不下四个槽。
严格字符串评分、槽位绑定、Reader 可读性、训练目标以及共享更新规则都可能影响结果。

同时，官方 SFT 成功并不能保证 DreamLite RGB 递归会成功。
文本完整历史和有损视觉记忆是不同的信息通道；论文没有验证我们所要求的视觉递归更新。

## 已实现并验证的对齐入口

- `src/vision_memory/prefeval/official_pipeline.py`：官方 topic split、上下文前缀、完整隐式对话、
  SFT 样本、最终答案 mask、精确 Mistral 文本模板、benchmark 生成输入和 Writer 输入视图。
- `scripts/data/prepare_prefeval_official.py`：从固定官方源码导出数据。
- `tests/test_prefeval_official_pipeline.py`：7 项针对性测试通过。
- `data/manifest.json`：来源、划分、生成物 SHA256、统计和教师身份限制。
- `data/sft-{train,eval_topic}-{0,5,10}interturn.jsonl.gz`：每个长度条件 820/180 条。
- `data/benchmark-disclosures.jsonl.gz`：1,000 个基础对的三种完整输入，共 3,000 条。
- `data/context-pools.json`：分别保存 SFT 与 benchmark 上下文流。
- `validation.json`：将上游实际 `process_input` 函数从 AST 载入，逐条比较 3,000 个
  SFT 输入，全部相同；主题顺序与 seed=42 的实际排列也一致。

复现导出：

```powershell
python scripts/data/prepare_prefeval_official.py --prefeval-root .cache/prefeval-upstream --output reports/prefeval-official-alignment-20260923/data
python -m pytest tests/test_prefeval_official_pipeline.py -q
```

Benchmark 的显式偏好 acknowledgment 要由被测模型生成。
新入口不会用 SFT 教师回答冒充被测模型自己的历史回复。
隐式样本中 preference、explanation、aligned_op 不进入模型输入，完整原对话保留。
benchmark 导出把评分信息放在 `evaluation_only`；消费者须通过输入函数使用这些记录。
MCQ 保留官方四个选项和原正确项下标，实际评估仍需使用既有官方选项打乱/解析规则。

## 接下来应怎样保持 DreamLite 主线

1. 使用同一批官方对话、原问题和回答监督，先采用 10 轮干扰主条件。
   完整文本历史可作为对应任务的参照输入；精确复现论文模型结果需另跑官方 Mistral SFT，
   本次没有启动该额外训练，也不把数据复现当作性能复现。
2. RGB 路径逐个读取完整对话交换：`M_t = W_theta(M_(t-1), exchange_t)`。
   每个交换都实际经过 Writer，包括无关对话；跨步只携带真实 RGB。
   Writer 不能看到最终问题、答案或历史文本缓存。Reader 只读固定最终 RGB 与原问题。
3. 主任务监督改为对官方自然语言回答的答案 token CE；共享 Writer 必须实际获得更新。
   Reader 可保持冻结以检验现有接口；冻结 Reader 不等于禁止梯度通过它传到 Writer。
   若继续使用现有官方 flow-matching 蒸馏路线，须明确答案监督如何产生可用视觉目标，
   并实际训练共享 Writer。当前新增代码只提供这些输入与标签，尚未接入 GPU 训练循环。
   不能把 token CE 直接改名为 flow-matching，也不能仅换答案继续无限优化 40 个独立 latent。
4. 先检验单偏好内容泛化和真实干扰写入，再扩至多偏好/覆盖/清除。
   官方生成任务按偏好遵循评分，原始 MCQ 另外报告；exact recovery 与 CLEAR 是额外诊断。
   新评分不回填旧表、也不抹掉旧失败。
5. 新的官方评估主题为 `travel_transportation`、`shop_technology`、`education_resources`、
   `shop_motors`，其中前两项与最后一项已出现在历史 sentinel 优化中。
   **不得拿历史 latent 续训结果直接宣称全新的官方 OOD。** 新 Writer 的 PrefEval 训练
   必须遵循此次 820/180 边界；从 PrefEval 阶段之前的已登记起点初始化，并披露历史开发暴露。
   对研究过程来说，这些题已不是从未看过的密封测试集。

用户的新指令改变下一阶段优先级：Plan13 C13/S13 继续保留为“已准备、未执行”，
不因旧心跳文本而自动开跑；先完成上述官方任务对齐。已有各轮数据不改动。
所有未来 GPU 工作仍限 `dl-clear-retain-h200x4-20260914`。
本次没有查询它的实时状态，2026-09-21 的 PENDING 不能作为 2026-09-23 的实时结论。

**完成边界：官方数据/对话构造已准备并核对；新 SFT、共享 Writer 训练、官方生成评分及
真实 RGB 递归结果尚未完成。项目可用版本的 Goal 仍未达成。**
