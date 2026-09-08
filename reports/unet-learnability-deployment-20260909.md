# U-Net 逐级可学性实验：部署协议

使用 `dl-base-h200x4-20260907` 完成的新一轮 96 个 Direct 终点；不使用第一轮单问法 bank 或旧失败 U-Net 的权重。每个不同阶段/rank 从同一预训练 DreamLite 开始注入新 LoRA；仅预算延长对照恢复原 optimizer、RNG 和权重。

## 数据绑定

共享根目录 `P=/inspire/ssd/project/exploration-topic/czxs26210936`。

- Oracle：`P/runs/direct-multiprompt-eos/46cd36b-20260909-r01`。
- Oracle 源码：`46cd36b1eb421471dafa7865fbab4dfe02336336`；运行终态为 96/96 完成、96 个原问正确终点。
- 已封存 bank：`P/runs/oracle-to-unet/f5c9c9e-direct-multiprompt-dl-base-seed20260908/bank/manifest.json`。
- bank SHA256：`20ef4a9fc53b254fd99b12cbc01cf1a6d41dee8d04dd3120c70ecaa141f30722`。
- 同一道 `ambient` 题，96 个不同 latent，均为 256 步的答案 + EOS 终点；不是 96 道题。
- Oracle 训练问法 original/p1/p2；p3/p4 仅评测。U-Net 本轮只训练终点 MSE，QA/EOS 仅作测量，所有问题继续使用固定两行指令。
- 起始 teacher 由预先指定的 `direct-gaussian-s00-a1` 选择，不依据最终测试问法分数挑选。

启动前复核 manifest、seal、oracle audit、每个 latent 文件/张量 SHA、来源 commit 与完整 96 条终态。随后在独立 GPU 上重新解码全部 96 个终点，检查图像 SHA 与原问 raw answer + EOS。任何一条不通过就记录失败，不进入训练或暗中删除该样本。

## 逐级控制变量

固定 Reader、VAE、condition encoder 与 U-Net 基础权重；仅训练 attention LoRA。真实四步 DreamLite、source=.5 gray+.5 ordinary Gaussian、FP32 Writer/VAE、BF16 Reader、AdamW LR=1e-4、clip=1。每次都直接拟合四步最终 latent：`mean((generated_z-target_z)^2)`，不换成中间速度的单步拟合。

| 阶段 | 训练输入 | 目标 | 新增变化 |
|---|---|---|---|
| single | 同一个普通 Gaussian 噪声 | 一个真实正确 teacher | 检查最基本的可学性 |
| noise | 96 个固定普通 Gaussian 噪声 | 仍是同一个 teacher | 只增加噪声种类 |
| set | 与上一阶段完全相同的 96 个噪声 | 固定一对一匹配 96 个真实 teacher | 只增加目标种类 |

`set` 中每个噪声与目标的对应关系一次固定，训练重复访问时不重新随机换标签。teacher ID 只用于调度监督，不输入 U-Net；模型输入仍为 source、事件、普通噪声与 timestep。这个简单配对是可学性对照，不宣称已经找到语义 cluster 或最优目标分布。

## 预算与自动门槛

先运行 single rank4 ×512 步。通过才进入 noise；noise 通过才进入 set。不同阶段从相同预训练模型和同 seed 初始化，避免把额外累计预训练步数混进阶段比较。

single 若失败，自动运行两个独立因素对照：rank16 ×512 步（改变容量），以及 rank4 从512继续到2048步（只增加预算，保留 optimizer/RNG）。两个对照均保留512步终点与指标。后续优先采用通过的 rank4；只有 rank4 的2048步仍未通过而 rank16的512步通过，才采用 rank16。后续阶段沿用该 rank 与总预算。两种 single 对照仍失败时保存结果并停止升级，不让更复杂训练继续耗卡。

门槛事先定义为：原始正确答案后立即 EOS，并且每个受检输入的终点 RMS 误差不超过该输入训练前误差的 10%。这个90%误差下降门槛是本轮操作定义，不是正确区域大小的科学定理；始终同时报告绝对误差、相对误差与原始输出，不能把门槛失败直接说成模型绝对不可学。

- single 门槛只看固定训练噪声；8个新验证噪声和8个测试噪声用于观察，不要求此时泛化。
- noise 门槛检查全部96个训练噪声与8个验证噪声。
- set 的训练噪声检查固定目标误差；新验证噪声允许进入任意正确区域，只检查原问答案+EOS，不人为指定陌生噪声必须输出哪个teacher。
- 最终测试的8个噪声与所有训练/验证噪声分开；五种问法全部保留原始输出。测试问法、测试噪声不参与门槛、rank或预算选择。
- 每个阶段另测 blank/donor，共10格；不重复复制对照扩大分母。
- 下一步多题实验需要独立多题 bank。当前数据不足，不伪造多题结果，也不擅自改动正在运行的 Frozen 数据。

## 资源与隔离

现场核查时 dl-base 正在运行两组新 bank U-Net，trust 四卡正在运行 Frozen，多任务均保留原代码、PID及输出路径。原 open 两卡 GPU 虽空闲，但平台运行时限只剩约7分钟，无法承载此次完整实验。

因此新建独立 `vlm-unet-fit-h200x2-20260909`，2×H200/40CPU/400GiB，NGC25.02，开发区-H200-3号机房-2-cuda12.8版本，平台运行时限180分钟。内部训练截止预留退出时间；持有GPU UUID锁，真实 GPU 已占用时拒绝启动，绝不终止占用者。

申请实例不等于已开训。排队时由本地 `watch_unet_learnability.ps1` 等待 RUNNING 后执行独立 bootstrap；bootstrap 再检查 hostname、空闲 GPU、源代码 commit 与重复 dispatch。该接续进程依赖本机保持运行，最多等待120分钟；未排到会留下明确状态，不能报告成正在训练。

训练每16步保存精确 optimizer/RNG checkpoint，关键步检查原问生成；预算终点独立保存 checkpoint 与全部评测。资源截止或停止信号按 checkpoint 暂停，错误不自动重试。输出中 `completed_gate_not_met` 表示实验已跑完但门槛未通过；不能当作功能成功。

## 实现与验证

- 配置：`configs/experiments/unet_learnability.json`。
- 训练与阶段控制：`scripts/train/run_unet_learnability.py`。
- 确定性配对与门槛：`src/vision_memory/training/learnability.py`。
- 独立实例启动：`scripts/inspire/launch_unet_learnability.py`。
- 已通过21项CPU测试，涵盖配对控制变量、训练/评测噪声隔离、QA+EOS与坐标联合门槛、现有 sampler 计算和 checkpoint 恢复；实际 GPU 状态另以 dispatch、日志与训练 metrics 为准。
