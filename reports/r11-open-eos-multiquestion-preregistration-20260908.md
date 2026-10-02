# 多题 Open/EOS 配对验证：训练前固定方案

原单题 `ambient` 的 B/C 已达到原题8/8，但这不能说明答案长度、字段变化后仍有效。
本实验检验 EOS 监督的机制是否在多道独立问题上成立。每题仍独立优化 VAE latent，
不训练共享 Writer，不宣称未见问题的模型泛化，也不替代 Frozen DreamLite 几何主线。

## 数据与固定选择

使用原 train.jsonl，SHA256 `24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184`。
只选真实 clean episode 中原有的 query，排除旧8题的semantic groups。
固定8个字段/答案层，每层按独立于模型输出的SHA256排序选2个不同semantic groups，总16题：

| 字段 | 原始真实答案 | 题数 |
|---|---|---:|
| color | green | 2 |
| drink | juice | 2 |
| music | jazz | 2 |
| material | linen | 2 |
| meal | pasta | 2 |
| color | no active preference | 2 |
| drink | no active preference | 2 |
| material | no active preference | 2 |

16题之间不共享semantic group，counterfactual或distractor变体不能增加独立样本数。
选择规则固定后不替换失败题。题目和答案从真实query的choices/target_index复算并绑定来源。
原文仅移除选择题指令；改写保留原有时间条件。gold只用于teacher/scorer，不注入Reader问题。
source_prefix仅供审计，不作为Reader额外上下文；本实验不检验事件序列写入能力。

实际长度由相同Qwen tokenizer在真实问答上下文中计算，不能使用数据集选择题的target_token_count字段。
原数据没有light blue/orange juice，不合成这些标签冒充真实任务。
不同长度层内容也不同，因此长度分组结果是描述性结果，不是纯长度因果实验。

## 实验与评测

- 每题seed0/1/2/3，原make_initial_latent函数、rho=.1；同一题同seed A/B起点完全相同。
- A：原答案token平均CE；B：答案token平均CE + 1×EOS CE。
- 每条Adam lr=.05×256steps，其余冻结BF16 Reader/VAE与原协议一致。
- 只有original_open参与训练；paraphrase_open/new_rewrite_open始终仅用于评测。
- 固定记录step0/16/32/64/128/192/256；主终点step256，禁止挑best或自动延长到512。
- raw greedy max_new_tokens=32与原EOS保持一致，主指标不靠裁词或部署解码改分。
- 换图对照为blank及已完成单题B-seed0的ambient图；新题gold均非ambient。
  donor绑定源checkpoint及原始正确回答的SHA256，不随新题输出重新挑选。
- 报告每题/每seed原题与两种未训练问法的完整EM、答案前缀、token accuracy、多余续写和换图差值。
- 样本单位为16个问题；每题4个seed为重复实验，不把64条轨迹当64个独立问题。

共128次独立训练、32768次optimizer更新。新4×H200实例按GPU0/1和2/3分两路，
每路选每一层的一道题（even/odd target_index），共8题×4seed×A/B=64次训练。
每路独立输出目录；可验证完成run复用，partial不得覆盖。

## 资源与停止条件

用户指定 `dl-base-h200x4-20260907`，实际4×H200，原NGC25.02镜像。
启动前核实无其他GPU计算进程，核对源码/模型/数据/起点哈希；新输出与原几何作业完全独立。
任何非有限loss/梯度、冻结参数梯度、SHA失配均停止受影响任务，保留失败证据。
先验证两路首个A/B均实际运行，再以吞吐更新耗时预算；不以平台RUNNING代替科学进程状态。
