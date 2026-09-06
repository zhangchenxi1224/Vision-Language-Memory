# R11_new identity-conditioning bridge：预注册与部署就绪审计

## 状态

- 协议：`R11-New-Identity-Conditioning-Bridge-Source-Init-Target01`
- 状态：identity condition-only probe 已完成；尚未执行本协议的任何完整 DreamLite forward、backward 或 optimizer step。
- 本轮是单目标、teacher-MSE 监督的机制诊断；`formal_success=false`、`phase2_allowed=false` 固定不变。

## 已有证据与本轮问题

- R11_new Phase1A 为 6/8，target-01 与 target-07 未通过；不能拼接成 8/8。
- teacher-matched 初始化在 target-01 得到 endpoint MSE `0.06570044904947281`、Reader mean CE `10.97591445709591`，但初始化直接使用 teacher，只能说明初始化敏感性，不能成为可用 writer。
- source-only 初始化＋原事件 conditioning 的直接父臂得到 endpoint MSE `0.11177215725183487`、Reader mean CE `25.765637596946004`、Reader accuracy `0/4`，主门失败。
- condition-only probe 已在 0 个完整 DreamLite forward、0 个 backward、0 个 optimizer step 下锁定官方 `no changes` condition：prompt embeds SHA-256 `dbb23b166711849739432ba815002e889253da78e099e59c7adbd351a8fb9f4d`，attention mask SHA-256 `c3325ac723d7cc08cba37d46e1710b5067019e47236ef55c41c66cba27c519f1`。

第一性原理问题：原事件是自然语义编辑指令，而 canonical R11 teacher 是 Reader 可读的非自然视觉码。冻结 DreamLite 可能持续执行语义编辑，导致优化后的 `x_T` 仍无法到达该 teacher。为隔离这一点，本轮只把实际 conditioning 改成官方 identity 文本 `no changes`。

## 唯一变量与固定合同

- 唯一变量：实际传给冻结 DreamLite conditioner 的文本由原事件改为 `no changes`。
- 原事件、query、choices、answer、target index 只保留为不可变数据与评测来源，不进入 conditioner 或初始化器。
- `x_T` 初始化仍为固定 source latent；初始化不使用 teacher、query、answer、choices、sample ID。
- teacher、source、数据、模型快照、四步 sigma 路径、Adam、256 step、cosine LR、无梯度裁剪、checkpoint、Reader/reset、raw step-256 主终点和全部门槛保持不变。
- canonical teacher 仍用于 dense MSE 优化，因此本轮不是 answer-independent shared writer，也不评估 event-to-state 学习。

主门不变：raw step-256 的 MSE/M0 `<=0.01`、L2/M0 `<=0.1`、teacher-normalized RMSE `<=0.1`，且固定 reverse-cyclic Reader `4/4`。次级 conditioning 审计仅在两个主门都失败时评估，并要求 endpoint 绝对 MSE 与 Reader CE 同时严格优于直接父臂；次级结果不能挽救主门。

## 部署合同

- Notebook：`vlm-r11-identity-h200x4-20260907`
- Workspace / group：`分布式训练空间` / `开发区-H200-3号机房-2-cuda12.8版本`
- 物理资源：4×H200、80 CPU、900 GiB；镜像 `ngc-pytorch:25.02-cuda12.8.0-py3`；共享内存 128 GiB；priority 4；自动停止 360 分钟。
- 运行时固定 `CUDA_VISIBLE_DEVICES=0,1`，使现有 DreamLite=`cuda:0`、Reader=`cuda:1` 双卡合同保持不变；GPU 2、3 不进入本单目标运行。
- 提交必须来自 clean detached checkout；实现 commit 先推送；technical preflight 与 formal 使用不同 fresh roots；suite lock 防止重复启动。

## 实现哈希（提交前）

- config bytes：`e4d375f10796d70f64ed91e201aebdbaa6a3939d5c7e37fd90f7034525fa60c8`
- config canonical JSON：`9ecd610e7c001d001af38cd04d4950f8470d3d8cfef7a6315cbd8ef40fd76f17`
- core：`d57d134534f4b83a049d9b978298a82d4f4fb0d457556a52899afec171620b9a`
- condition binding：`98246719a77bb36f409d9dfa8242c7dc5b8d10fa23862539e6164d67fd414374`
- trainer：`7a73a6fe04e65d71189b299c08c99f8a86b61ebee5016cff4aeec9a8be7e3fbc`
- controller：`a59cd23e98663055212b9978e81ea84311fa94fd71849c264e912e548b7840d0`
- independent aggregator：`0777e4ad0facb3c9833b162ba30695a1662e339787470858f19eb3fdc5fadb3a`

## 本地验证

- 定向 identity bridge 测试（最终部署配置）：`76 passed`；原始 JUnit 为 `pytest-junit.xml`（SHA-256 `25ed8ddf762592a63811be642ace525a3b28bbf7a419db95961c2997a55f2e20`）。
- R11_new 扩展回归（最终部署配置）：`682 passed, 561 deselected`；原始 JUnit 为 `pytest-r11-new-junit.xml`（SHA-256 `c6d8dde3f6ec5f5ec41d327214d26db3c4dfe31e22c953bbf30e4d30a465f579`）。
- Ruff：通过。
- `compileall`：通过。
- 配置 JSON 独立解析：通过。

全仓收集运行保留在 `pytest-collection-junit.xml`（SHA-256 `f0eec07da6cd234075c32801c300d7969900cb83da98be1959822b5f8cee2bc0`）。运行至 75% 后收到外部 `KeyboardInterrupt`；中断前为 `949 passed, 4 skipped, 9 failed`。九项失败均不触及本协议实现：一项由本机解释器缺少 `diffusers` 引起，七项由当前 worktree 未检出旧 `data/r3_micro_v1/*.jsonl` 引起，一项是既存 R13 source-preregistration 文件与其配置记录的哈希不一致。它们不被删除、不被改成通过，也不作为本协议放行证据。本协议远端 forward 的放行证据限定为上述 76 项定向测试、682 项 R11_new 回归、静态检查、JSON 解析，以及远端 technical preflight；任何本协议相关失败仍禁止 formal run。

## 解释边界

即使本轮主门通过，也只说明固定 identity conditioning 下、单个 target、teacher-supervised latent solver 的经验可达性。它不能证明 identity mapping、原事件写入、共享 writer、ID/OOD、递归记忆或 Picture Memory 科学成功。下一步必须另行预注册恢复原事件且不使用 per-sample teacher 的共享训练方案。
