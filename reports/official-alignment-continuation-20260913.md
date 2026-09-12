# Goal 接续位置

用户目标：新分支修复DreamLite与官方训练偏差，通过启智新实例实训、评测、迭代到有证据支持的可用版本。Goal已创建，仍active，未达成；不能以源码测试通过或训练结束标记complete。

## 本地与代码

- 工作树：`C:/Users/Expedition/dreamlite-official-alignment-20260913`。
- 分支：`codex/dreamlite-official-alignment-20260913`，源自latent-bank分支`f68bf06`，已推送origin。
- 官方源码在本地 `third_party/DreamLite`，HEAD=`a6e20c8cc94027f37dd7c5a81b0b3b472aa18409`。
- 已通过46项针对性测试（`PYTHONPATH=src`，本地默认`python`有torch/pytest；home/.venv python主要用于HF API，不是测试环境）。新增摘要严格EOS统计及缺失/重复五问法拒绝；活跃3500实验使用旧固定提交，结束仍以原始generation计算EOS，不能只看旧terminal的exact字段。
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
4. **已完成Base512**：`P/runs/dreamlite-official-alignment/ac34ab2-base-single-20260913`。固定源码 `ac34ab20b50fae82d56d1aef13fc995ee35c302a`，checkout `P/repos/dreamlite-official-base-20260913-r2`。Base官方类、28步native CFG7.5、同一teacher/seed/rank16/accum4/lr5e-5，512步。五问法均0/8、raw EOS也全失败；loss首末64均值0.684343/0.436569，adapterdelta13.174183。result与checkpointSHA已复核，resultsha=`afd3d93c222654c2754060610a93c072449288752db3733867f8f4e1550b92b4`。完整文本证据已下载为base-evidence.tgz，preview生成器修正动态Base标题（旧cache脚本错误硬编码Mobile，勿再用）。
5. **已完成oracle邻域诊断**：`P/runs/dreamlite-official-alignment/1bb0d28-neighborhood`，源码checkout `P/repos/dreamlite-neighborhood-20260913`。其`-queue`目录terminal completed，不再运行。48条输出、哈希已复核；原问及paraphrase_4，RMS0.001/.003/.01/.03均3/3正确立即EOS，RMS.1为3/3与2/3，RMS.3均0/3。teacher向Base512 seed0插值.01/.03/.1/.3两个问法都通过；纯Writer失败，RMS0.444540。这否定“只能精确坐标读出”，不证明任意方向鲁棒。诊断不算Writer成功。
6. **已完成CFG1**：`P/runs/dreamlite-official-alignment/32d70c0-guidance1`，源码checkout `P/repos/dreamlite-guidance-20260913`，日志同路径加`.log`。Base512同checkpoint、8噪声、native28steps，仅CFG7.5改为1。50条结果，五问法均0/8。已下载guidance1-evidence.tgz并验证文本哈希；`.pt`留在远端。
7. **当前活跃Base3500**：`P/runs/dreamlite-official-alignment/ac34ab2-base3500-single-20260913`。与Base512完全相同的固定代码`ac34ab2`/checkout，仅预算3500、fresh initialization。同seed同目标同超参同nativeCFG7.5，不改旧512运行身份；前512draw应相同。已实际确认GPU PID2622580，启动时间Unix1789235717.7。已观察85/3500步，并下载当前training.jsonl到本地.cache/base3500-prefix.jsonl；前85步除耗时外所有字段（loss/grad/draw/cursor）与Base512完全一致，报告reports/official-alignment-results-20260913/base3500-prefix-check.json。接续检查train/training.jsonl与terminal，而非重复启动。预注册在本地reports/official-base-3500-preregistration-20260913.md，提交2e08e97先于调度。预计约100分钟到端点评测；不以启动或loss成功替代功能证据。

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

先继续当前Base3500，核验前512draw/loss与Base512；等待真实结果时保持进程核验及适度进度沟通，不重新启动。Base512/CFG1失败、oracle邻域相对鲁棒共同支持测试官方3500预算；若仍失败，应再区分容量、优化目标与oracle训练目标，不能无边界只加步数。成功后还须全新固定噪声、反事实事件/多题验证；目前8benchmark噪声未参与训练，但研究迭代中已反复观察，不是 untouched research holdout。

预览生成器现在版本化为scripts/reporting/render_alignment_preview.py，远端临时副本 `P/runs/dreamlite-official-alignment/render_alignment_preview_v2.py`，会从identity读正确model_variant与预算标题。旧render_alignment_preview.py硬编码Mobile，只适用于历史Mobile图。
