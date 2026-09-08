# 新 96/96 多问法 Direct → 参考轨迹蒸馏

用户授权：使用 `dl-base-h200x4-20260907` 上已完成的 96/96 优化终点部署蒸馏；不干扰任何现有训练，有空闲 GPU 则直接使用。若确需排队，建立每 20 分钟检查一次的本任务心跳，资源到位后移除心跳并启动实验。

## 固定输入

- Direct 源码 `46cd36b1eb421471dafa7865fbab4dfe02336336`。
- Oracle：`P/runs/direct-multiprompt-eos/46cd36b-20260909-r01`。
- 已审计 bank：`P/runs/oracle-to-unet/f5c9c9e-direct-multiprompt-dl-base-seed20260908/bank/manifest.json`。加载时核对 manifest/seal SHA、所有 96 个 FP32 latent SHA、源 campaign 96/96 终态、EOS 与原问评测契约、模型快照和三训练/两留出问法。
- 仍是同一 ambient 题的 96 个终点，不是 96 道题。源 Oracle 的原问、paraphrase_1、paraphrase_2 各训练 86/85/85 次；paraphrase_3、paraphrase_4 留出评测。
- 本次参考场使用全部 96 个成员，不另留 19 个 teacher；评测明确没有 teacher 留出，不沿用旧 77/19 的说法。记录参考算法实际选中各成员的次数，全部参与参考场不等于每个成员都成为了某个噪声的最终目标。

## 第一版优化规则

从原始 DreamLite 初始化新的 rank-4 attention LoRA（alpha=4，dropout=0），不复用旧失败 adapter。VAE、Reader、condition encoder 与 U-Net 基础权重冻结。AdamW LR=1e-4、weight decay=0、clip=1，512 次更新，seed=20260908；只有 LoRA 权重更新。

每次取一个新的普通 Gaussian 噪声。参考算法与学生使用相同起点 `.5*source+.5*noise`、相同事件和四步 sigma：`.5,.375,.25,.125→0`。

参考算法读取整个成功 bank，以闭式条件均值速度计算四步参考轨迹。学生完整运行四步 U-Net，梯度穿过所有四步。损失固定为：

```
L_end  = mean((student_final - reference_final)^2)
L_path = mean of MSE at the first three generated states
L      = L_end + L_path
```

初始状态不计入损失。参考轨迹停止梯度；学生输入只有 source、事件条件、噪声、时间，不读取 bank 或 teacher ID。相同噪声的参考路径是确定的，不在不同更新里随机改配另一个 endpoint。

本版不同时加入新的 QA 损失、改变 LoRA rank 或全量微调，以先验证参考轨迹蒸馏。EOS 已用于 teacher 搜索并通过独立 QA/EOS 评测检验学生；这里的蒸馏损失本身不是 token CE。

每个新参考终点若与已验证 bank 成员逐位相同，直接绑定该成员。否则真实解码并执行原问 greedy/32 token generation，必须正确且立即 EOS。参考失败时保存原始轨迹并在参数更新前报错，不能偷偷裁答案、替换目标或跳过噪声掩盖覆盖率。

## 验证与恢复

启动前先在相同 8 个评测噪声上验证参考算法，原问必须 8/8；四种改写照常报告，不拿留出问法挑 teacher。新 U-Net 的 baseline、step64、step256、step512 均评测相同 8 个新噪声 × 5 种问法，并各做一次 blank/donor 控制，不复制控制扩充分母。

每次更新保存 LoRA、optimizer、RNG、步数和参考目标 receipt。中断时可通过 `--resume` 从完整 checkpoint 继续。每步检查梯度有限且非零、冻结参数未变；运行前后核验源码/输入银行。终态 `completed` 表示约定工作执行完毕，原问/改写 EM 才表示功能表现。

## 资源隔离

本轮检查发现 `dl-base` 上旧可学性实验的父子进程仅使用物理 GPU0、1；GPU2、3 无 CUDA 进程，协作锁空闲。蒸馏计划显式 `CUDA_VISIBLE_DEVICES=2,3`，内部 DreamLite 为 cuda:0、Reader 为 cuda:1；只申请这两个 GPU 的锁，加载前再次检查占用。单 CPU 计算线程，独立 immutable checkout、独立运行目录；不停止、修改、恢复或迁移任何别的实验。

代码入口 `scripts/train/train_reference_distillation.py`。启动记录补充实际 commit、bank SHA、PID、输出路径和运行证据。若临近启动时 GPU 已被使用，则保持既有任务并重新申请资源，不强占。

`P=/inspire/ssd/project/exploration-topic/czxs26210936`。
