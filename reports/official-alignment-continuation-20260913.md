# Goal 接续位置

用户目标：新分支修复DreamLite与官方训练偏差，通过启智新实例实训、评测、迭代到有证据支持的可用版本。Goal已创建，仍active，未达成；不能以源码测试通过或训练结束标记complete。

## 本地与代码

- 工作树：`C:/Users/Expedition/dreamlite-official-alignment-20260913`。
- 分支：`codex/dreamlite-official-alignment-20260913`，源自latent-bank分支`f68bf06`，已推送origin。
- 官方源码在本地 `third_party/DreamLite`，HEAD=`a6e20c8cc94027f37dd7c5a81b0b3b472aa18409`。
- 已通过45项针对性测试（`PYTHONPATH=src`，本地默认`python`有torch/pytest；home/.venv python主要用于HF API，不是测试环境）。
- 重要修改：默认official FM、全0–1训练、整数t、原始source图条件、原始raw vs effective调度、纯噪声起点、rank16/accum4/AdamW官方默认；base选项直接官方pipeline训练/原生28步CFG评测；hash/optimizer/RNG/teacher EOS/real QA审计保留。

## 当前远端

Windows原生无inspire命令；使用 `C:/Windows/System32/wsl.exe -d Ubuntu -- bash -lc 'inspire ...'`。InspireSkill7.1.6，参考CLI help。

- 新CPU：`dl-align-cpu-20260913`，CPU资源空间、CPU资源-2、前沿课题探索，已refresh SSH connection。
- 新GPU：`dl-align-h200x2-20260913-r2`，分布式训练空间、开发区-H200-3号机房-2-cuda12.8版本，2×H200 143771MiB，40CPU/400GiB，NGC25.02，driver570.124.06。当前任务专用。
- 初次申请 `dl-align-h200x2-20260913` 因原组无GPU已停止，可以在结束时删除这个本次空实例；不要动历史其他实例。
- GPU restricted exec必须使用 `exec_command(..., tty:true)`，否则CLI误判本地管道stdin并拒绝；所有普通只读日志/文件工作通过CPU访问共享盘更方便。遵守用户禁用IAB/Browser Use规定。
- CPU命令：`inspire notebook exec dl-align-cpu-20260913 "<shell command>" --timeout 60`。
- GPU命令同上改GPU名并加workspace，但宿主exec分配tty。
- 准备依赖不使用Browser，CPU具备公网Git/HF访问。

共享根 `P=/inspire/ssd/project/exploration-topic/czxs26210936`。
模型根 `M=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory`。
GPU Python `P/envs/vlm-r3-ngc2502/bin/python`；CPU纯stdlib脚本用python3（3.10.12）。不要更改系统Torch。

## 已完成与正在跑的实验

1. `P/runs/dreamlite-official-alignment/c7e752b-single-20260913`：官方Mobile逐位parity通过，但首次训练因launcher缺少确定性env在0步停止。后续已修复，保留失败记录。
2. `P/runs/dreamlite-official-alignment/106c8eb-single-20260913`：完整512步官方FM Mobile单目标，2048独立noise；teacher原问输出ambient+EOS；最终原问与四个改写均0/8，loss首末64步均值0.752814/0.439556，adapter deltaL2=12.6377。结果和checkpointSHA已复核，已下载 `reports/official-alignment-results-20260913/mobile-evidence.tgz`、展开文本和mobile-preview.png。
3. `P/runs/dreamlite-official-alignment/e76114c-base-single-20260913`：旧Base缺text_encoder权重，0步失败，已补全。串行queue目录后缀 `-queue` 已终止失败，不会自动再跑。
4. **当前活跃**：`P/runs/dreamlite-official-alignment/ac34ab2-base-single-20260913`。固定源码 `ac34ab20b50fae82d56d1aef13fc995ee35c302a`，checkout `P/repos/dreamlite-official-base-20260913-r2`。Base官方类、28步native CFG、同一teacher/seed/rank16/accum4/lr5e-5，预算512。已通过完整权重、VAE匹配、teacher EOS复测、baseline50格，并观察实际优化step16。日志 `stage-0-1789234161819887929.log`；`train/training.jsonl`、`train/metrics`、`train/checkpoint-latest.pt`。接续先读 `terminal.json`、`train/result.json`；无终态则查看最新metric和进程。

worker deadline Unix `1789254021`（2026-09-13 07:00:21北京时间）；GPU平台8h上限约08:54。需要延长/追加时仍须根据新结果安排，不能让平台关闭前丢失检查点。未经结果证据不得扩大成“可用”或“多题泛化”。

## Base补全与数据绑定

- bank路径 `P/runs/oracle-to-unet/f5c9c9e-direct-multiprompt-dl-base-seed20260908/bank/manifest.json`，SHA256=`20ef4a9fc53b254fd99b12cbc01cf1a6d41dee8d04dd3120c70ecaa141f30722`。96个终点是同一个ambient问题，三训练/两留出问法；本轮按哈希选一个训练teacher做最低可学性试验。
- Base目录 `M/DreamLite-base-a9a0f15-20260907`，HFrevision=`a9a0f151ffd99d3c37f3fd0472f5e8f1b31215aa`。
- 官方API返回text_encoder权重4,255,140,312字节、sha=`7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`。缓存Mobile权重同SHA，经核验后硬链接补全Base缺文件，未替换不同模型。记录 `P/runs/dreamlite-official-alignment/base-weight-materialization.json`。
- 完整27文件seal `P/runs/dreamlite-official-alignment/base-complete-snapshot-seal.json`，payloadsha=`fc70dd11ccc652805388e0b0e687383c1441fbf9ea34c5384c8d7a241c7f7a72`。
- 旧26文件partial seal保留为base-snapshot-seal.json，禁止拿它恢复已补全Base。
- 官方源码远端 `P/Vision-Language-Memory/third_party/DreamLite`（锁定a6e20c8）。
- Base与Mobile VAE权重SHA均=`e175c3284b8514a2ecb1a70f004b0b09320decd6842e77dfebbdb5dbbdf0312b`；base运行时又比较全部非元数据VAE配置，并复验bank source。

## 收集结果

读取新结果可以用CPU纯stdlib报告器：
`python3 P/repos/dreamlite-alignment-reporter-20260913/scripts/reporting/collect_official_alignment.py --run <RUN> --output <SUMMARY.json> --archive <EVIDENCE.tgz>`。
报告器校验result/terminal与最终checkpoint SHA，汇总所有真实generation和训练draw；没有terminal仅输出进度，不能视为完成。

下载：`inspire notebook scp dl-align-cpu-20260913 --download <remote> /mnt/c/Users/Expedition/dreamlite-official-alignment-20260913/reports/official-alignment-results-20260913/<name>`。

预览脚本（仅可视化、无训练作用）在 `P/runs/dreamlite-official-alignment/render_alignment_preview.py`；GPU节点用已装torch/PIL的env执行并设置CUDA_VISIBLE_DEVICES为空可CPU渲染，输入--run/--output。已确认Mobile8个baseline图多为蓝色列车，训练后8个图主要为纹理，均无法读出ambient。不能用“多模式塌缩”解释本轮单目标的预期趋同，更不能把可视变化当功能成功。

## 接下来

先完成并分析当前Base512的成对实测。若仍失败，应区分官方目标对齐、模型变体/蒸馏、可训练容量、优化预算与oracle目标本身的可读邻域；可以做有明确控制变量的新训练，保留未见噪声与原始QA/EOS，不复用测试问法来挑teacher。不能单纯无限增加旧方案预算或声称源码修复已解决全部失败。达到可用还需多事件或反事实内容验证，单题0/8显然未达标。
