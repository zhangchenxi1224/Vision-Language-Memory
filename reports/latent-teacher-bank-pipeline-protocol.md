# 新 EOS Oracle → 成功 latent 集合 → 独立 U-Net 训练

## 2026-09-09 多问法 Direct 接续

新增支持 oracle 提交 `46cd36b1eb421471dafa7865fbab4dfe02336336`：同一道 ambient 题、96 个新起点，训练问法为 original_open / paraphrase_1 / paraphrase_2，按 optimizer step 从零开始轮换，256 次更新的曝光数为 86/85/85；paraphrase_3 / paraphrase_4 仅评测。封库同时验证配置、每条 manifest、256 条实际 prompt receipts、全部评测行的 question_trained 标记，并保留训练与留出问法分组。旧单问法协议继续受单独校验。

新一轮只从该 oracle 的完整 step-256 终点建库，继续以原问 raw exact match 决定入库，不用留出问法筛选目标。U-Net 架构、flow matching 目标、两个 seed 各512步及训练前后新噪声评测保持原设置；输出独立保存。多问法 oracle 在 dl-base 实例运行，U-Net 接续进程可以在共享盘可见的 trust 实例等待完成，因此不会占用正在优化 latent 的 GPU。

这仍然是单题实验。第一轮 U-Net 的两组新噪声原问均为0/8，训练执行完成不能解释为已经学会生成正确记忆；新一轮结果同样必须查看原始答题评测。

本文件登记实现规则，不是 GPU 训练成功报告。当前两条新实验继续使用其冻结的原 checkout；下述工具部署到单独 checkout，由 CPU sidecar 等待完成，再接续同一实例的空闲 GPU。

## 两条 arm 保持独立

- Direct：新 96 个起点，只优化 `z`，冻结 VAE 和 Reader，训练使用 `mean_answer_ce + eos_ce`。本轮原源码 `0f704178137ee4beac952224f3175a3923d5e438`。
- Frozen：新 158 条计划，优化 `x_T`，完整穿过冻结 DreamLite、VAE 和 Reader，使用相同 EOS 目标。本轮原源码 `c97a75435b05d995698739b9118c548539d03dbb`。A1 的 6 条重复实验进入完整性审计，完全相同的成功终点不重复加权。
- 每路分别封库、分别初始化和训练 U-Net adapter。来自旧 MCQ、历史 Open/EOS 或 best-step 的终点均不能作为本轮 teacher；历史固定 donor 仅是评测对照，不进入 teacher 集合。

## 固定入库标准

每条必须完成 256 个真实 optimizer update，保存 0..256 共 257 个轨迹状态和 11 个 checkpoint，原始 step-256 endpoint 的原问句 raw greedy 32-token 完整输出通过现有 exact-match scorer，才能入库。首答案词正确、低 loss 或中途某一步答对都不能替代终点 exact match。

完整 15 格端点评测必须存在：5 个问法 × matched / blank / 不同答案 donor。重新从 raw 文本和 token ID 计算分数，核对所有已保存 scorer 字段。改写只改变第一行提问句，后两行逐字固定：

```text
Use the memory image to answer.
Answer with a short phrase only.
```

原问句 EM 决定入库；全部五问法正确是单独的鲁棒性指标。blank 和 donor 的正确数按真实结果保存，不预设零，也不通过它们是否为零倒推成功。

所有 planned runs 和 campaign end audit 完成后才封库。训练中、文件缺失、failed 和 `paused_at_run_boundary` 均不能触发 U-Net。全库没有正确终点时输出 `blocked_no_success` 和原因，实际不启动训练。部分问题没有成功终点时显式列出 `excluded_question_ids`；`question_coverage` 的分母仍是原计划问题数，训练只针对有合法 teacher 的问题，不能报告全题已学会。

## 导出与几何

`bank/manifest.json` 使用 `latent-teacher-bank/v1`：逐 teacher 保存一个原始 FP32 `[1,4,128,128]` tensor、文件 SHA256、canonical tensor SHA256、真实来源、EOS/decoding 协议与问法正确性。每个 question group 保存真实单事件 `event_text`、原始 source latent、不同答案 donor 和全部 teacher IDs。问题和答案只是监督/评测 metadata，不能进入 U-Net conditioner。

同一道题、完全相同 tensor SHA 的终点去重并保留全部 source runs；不同成功坐标全部保留，不把均值当作 target。初始化分布、seed 和 scale 保留于每个来源的 `run_spec`。

CPU 分析产出 `geometry.json`、每题 NPZ 和 PNG，包含原始/中心化两两距离、范数、PCA 谱、成功终点与优化位移、同样本数同维度的 isotropic 参考。共同平移不改变两两距离；两份距离是审计对照。二维 PCA 图只是投影，`rank ≤ N−1` 只是有限样本事实。簇数保持未确定，不能从少量成功点强行宣称低维流形或固定类别数。

## 自动接续的资源边界

入口：`scripts/inspire/run_oracle_to_unet_pipeline.py --config <pipeline.json>`。

1. sidecar 不占 GPU，等待其对应 oracle 完整结束。
2. CPU 全量校验、封库和作图。
3. 核验 same host、指定 GPU UUID、GPU 无 compute process，并取得协作文件锁。
4. 从封库的模型 manifest SHA 明确设置子进程模型绑定环境变量，启动 U-Net trainer。
5. checkpoint、返回码、terminal 与最终 result 完整验证后才将状态写成 completed。训练执行结束不等于科学成功；仍须查看实际 QA 和 coverage。

各训练 seed 使用不同 `output_root`；各 pair 绑定不同 GPU UUID，单独 bank 输出目录避免并行写入冲突。两路模型/数据分开审计，未跑完一路不阻塞另一实例。

资源到期会暂停，绝不自动创建新启智实例。`resource-lease.json` 是显式资源期限，可在用户实际延长同一实例后原子更新，不需要修改正在冻结校验的源码。只更新 lease 不等于重启已暂停 oracle；Direct 的原实验须明确续跑：

```sh
python scripts/inspire/run_oracle_to_unet_pipeline.py \
  --config /absolute/path/pipeline.json \
  --resume-direct-oracle --deadline NEW_EPOCH
```

该入口只接受原 campaign 与 lane 均在完整 run 边界暂停、同原 source commit、同 output root、同四 GPU、空闲且更晚的 deadline，并调用原冻结 `run_direct_geometry_h200x4.py`。已有完整 run 由原 runner 逐条 hash 验证后跳过；缺损轨迹不会静默覆盖。

U-Net 每步原子保存 `checkpoint-latest.pt`。收到停止信号或靠近显式期限会写 `paused`、返回 75；sidecar 后续使用 `--resume` 保留 optimizer/RNG。baseline/final evaluation 的确定性重放不伪造 optimizer updates。

## 配置示例

示例中的 hostname / UUID 必须从同一 allocation 的实际记录读取；不要按实例名猜 GPU 身份。

```json
{
  "schema": "vision_memory.oracle-to-unet-pipeline.v1",
  "route": "direct",
  "expected_commit": "EXACT_NEW_PIPELINE_TRAINER_COMMIT",
  "output_root": "/absolute/runs/direct-writer-seed20260908",
  "resource_lease_path": "/absolute/runs/direct-writer-seed20260908/resource-lease.json",
  "runtime": {
    "expected_hostname": "READ_FROM_ALLOCATION",
    "gpu_indices": [0, 1],
    "gpu_uuids": ["READ_GPU_0_UUID", "READ_GPU_1_UUID"]
  },
  "oracle": {
    "root": "/inspire/ssd/project/exploration-topic/czxs26210936/runs/direct-latent-geometry/0f70417-20260908-r01",
    "repo": "/inspire/ssd/project/exploration-topic/czxs26210936/repos/direct-latent-geometry-20260908",
    "config": "/inspire/ssd/project/exploration-topic/czxs26210936/repos/direct-latent-geometry-20260908/configs/experiments/direct_latent_geometry.json",
    "commit": "0f704178137ee4beac952224f3175a3923d5e438",
    "resume_gpu_indices": [0, 1, 2, 3],
    "resume_gpu_uuids": ["READ_GPU_0_UUID", "READ_GPU_1_UUID", "READ_GPU_2_UUID", "READ_GPU_3_UUID"]
  },
  "trainer": {
    "dreamlite": "/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory/DreamLite-mobile",
    "reader_model": "/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory/Qwen3-VL-4B-Instruct",
    "steps": 512,
    "seed": 20260908
  },
  "minimum_training_seconds": 900
}
```

同目录 lease：

```json
{
  "expected_hostname": "READ_FROM_ALLOCATION",
  "gpu_uuids": ["READ_GPU_0_UUID", "READ_GPU_1_UUID"],
  "deadline_epoch": 1788867600
}
```

以上 epoch 仅示例，必须替换为实际、已确认的启智资源到期时间。Frozen 配置使用 `route: frozen`，原 repo `.../repos/frozen-oracle-eos-20260908`，run root `.../runs/frozen-oracle-eos/c97a754-20260908-r01`，config `configs/experiments/frozen_oracle_eos.json`，另加 `oracle.planned_manifest` 指向 `configs/experiments/frozen_oracle_eos_manifest.json`，并绑定该 Job 实际主机和 GPU。

## 本地验证

Windows CPU Python `D:/st_python/python.exe`，`PYTHONPATH=src;.`：

```text
pytest tests/test_latent_teacher_bank.py tests/test_oracle_to_unet_pipeline.py tests/test_latent_bank_unet.py -q
26 passed
```

测试覆盖：EOS/固定模板拒错、15 格完整性、raw EM 与首 token 分离、分数重算、paused 不封库、失败停止、同坐标去重、缺成功问题分母、空库拒训、距离/PCA 有限样本限制、路径越界、bank 导出→真实 trainer loader、GPU/host/lease 身份、checkpoint resume 与模型 SHA 环境变量。测试使用合成 CPU fixture，不是实验结果。
