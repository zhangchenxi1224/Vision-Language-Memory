# Goal 接续位置

用户目标：新分支修复DreamLite与官方训练偏差，通过启智新实例实训、评测、迭代到有证据支持的可用版本。Goal已创建，仍active，未达成；不能以源码测试通过或训练结束标记complete。

## 本地与代码

- 工作树：`C:/Users/Expedition/dreamlite-official-alignment-20260913`。
- 分支：`codex/dreamlite-official-alignment-20260913`，源自latent-bank分支`f68bf06`，已推送origin。
- 官方源码在本地 `third_party/DreamLite`，HEAD=`a6e20c8cc94027f37dd7c5a81b0b3b472aa18409`。
- 已通过47项针对性测试（`PYTHONPATH=src`，本地默认`python`有torch/pytest；home/.venv python主要用于HF API，不是测试环境）。新增摘要严格EOS统计及缺失/重复五问法拒绝，以及全冻结inference审计；活跃3500实验使用旧固定提交，结束仍以原始generation计算EOS，不能只看旧terminal的exact字段。
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

**更新至北京时间02:29：主训练已实际观察1232/3500步，PID2622580仍占用GPU0约30GB、GPU1约9GB。前缀核验已扩展到完整512步，除耗时外字段全部一致，JSON报告已更新（读取时采样到了714步）。预计到端点还约60分钟，不将等待或源码测试当功能通过。**

8. **历史多题审计已完成**：`P/runs/dreamlite-official-alignment/historical-multiquestion-audit.json`，本地报告同名及historical-multiquestion-review.md。原16题实验已completed；CPU重跑固定commit2c0e41c的inventory/score函数，128run/32768updates/3456raw全验证。B原问64/64、第一改写64/64、第二61/64，旧BF16 VAE；不等于Writer成功，也不能未经FP32重放就移植为当前teacher。原数据路径和plan限制详见报告。
9. **提示词接口对照已完成**：第一次 `P/runs/dreamlite-official-alignment/4694fdf-event-format`（代码checkout `P/repos/dreamlite-event-format-20260913`）80条输出后因误用要求trainableLoRA的审计退出；日志同路径加.log。修复后 `P/runs/dreamlite-official-alignment/5daf0b2-event-format`（代码checkout `P/repos/dreamlite-event-format-20260913-r2`）全完成，80行逐字段与首次一致。两事件ambient/jazz×两形式raw/memory_note×4新噪声×5问法全部0/4；只有模型生成图片，没有外部答案栅格化。完成证据tar/PNG montage/replay检查已落地。此probe在GPU1曾使用约23GB额外显存，已经退出；当前只剩主训练PID2622580，不要重复启动。全冻结推理可以显式`load_base_runtime(...,inference_only=True)`同卡加载，但正常训练仍要求两卡。其审计必须使用`audit_inference_only_runtime`，不要调用要求可训练LoRA的training.frozen_audit。

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

先继续当前Base3500；其完整前512draw/loss已核验一致。等待真实结果时保持进程核验及适度进度沟通，不重新启动。Base512/CFG1失败、oracle邻域相对鲁棒共同支持测试官方3500预算；若仍失败，应再区分容量、优化目标与oracle训练目标，不能无边界只加步数。成功后还须全新固定噪声、反事实事件/多题验证；目前8benchmark噪声未参与训练，但研究迭代中已反复观察，不是 untouched research holdout。

预览生成器现在版本化为scripts/reporting/render_alignment_preview.py，远端临时副本 `P/runs/dreamlite-official-alignment/render_alignment_preview_v2.py`，会从identity读正确model_variant与预算标题。旧render_alignment_preview.py硬编码Mobile，只适用于历史Mobile图。

## 02:50容量对照更新

- 源码提交 `b0a06f5c597fce64457748b5d81f580116174547` 已推送。51项针对性测试通过，新增完整U-Net训练范围、冻结边界、周期checkpoint精确恢复和基线数值拒绝测试。
- Base3500在02:47实测1935/3500，仍使用原ac34ab2源码和原双H200实例，不修改它。
- 新两卡容量实例 `dl-align-full-h200x2-20260913` 因父项目quota剩余1GPU无法调度，0步停止并删除；本轮最早的空实例 `dl-align-h200x2-20260913` 也已核实STOPPED后删除。两者均无训练产物。**保留运行中的r2实例和CPU实例。**
- 新单卡 `dl-align-full-h200x1-20260913` RUNNING，node qb-prod-gpu2459；同project/workspace/group/NGC25.02，1H200/20CPU/200GiB/shm64，02:39:59创建，240分钟平台上限。显式将Writer和Reader放cuda:0，默认两卡训练路径不变。
- 新checkout `P/repos/dreamlite-full-unet-20260913`，新run `P/runs/dreamlite-official-alignment/b0a06f5-full512-single-20260913`，launcher日志为run路径加`.log`，阶段日志 `stage-0-1789238966133268987.log`。已调度，GPU实际模型进程PID40541；02:50正在执行未训练baseline，尚未确认首个优化步骤。
- 调度前预注册 `reports/official-full-unet-preregistration-20260913.md`。512更新、accum4、相同teacher/2048draw/seed/lr/wd/clip/native28CFG7.5；仅全量U-Net替代rank16 LoRA，其他模型冻结。worker deadline1789251300（06:15），早于平台06:40停止。
- 训练前必须生成 `train/baseline-reference-check.json`：与原Base512 runtime相同、8个latent/image和全部28步轨迹逐位相同、50条raw Reader记录逐字段相同，原resultsha afd3d93c222654c2754060610a93c072449288752db3733867f8f4e1550b92b4。不通过则0步失败，调查差异，不静默放宽容差。
- 全量checkpoint每16步、首步/末步/正常停止均保存，完整optimizer+RNG；硬中断保留physical log并重放未保存更新。默认LoRA仍每步保存。最终collector请用新checkout版本，读取通用unet_parameter_delta_l2；旧collector仅认识adapter字段。
- 接下来先确认full baseline gate及首步/16步checkpoint，随后收集两轮最终raw答案+EOS与哈希，再决定功能迭代；goal仍active。

**02:53实测更新：full baseline gate已经通过，8张初始latent/image、全部轨迹和50条raw记录均与Base512逐位/逐字段一致。核验JSON已下载为reports/official-alignment-results-20260913/full512-baseline-reference-check.json。PID40541占用45840MiB，full已31/512步，checkpoint-latest.pt为4.4GiB，已跨过16步周期保存；约1.9秒/步。主LoRA已观察2152/3500，预计端点约03:30；full端点预计03:10左右。不要重复启动。**

**03:00更新：full258/512，LoRA2447/3500，均实际确认GPU进程。主LoRA的nvidia-smi宿主PID2622580对应容器内PID411798；用pgrep完整命令核验，不能因容器ps找不到宿主PID误报退出。Full节点PID40541仍有效。**

- 条件诊断代码 `712c00d56323bc5c628d1766bf377beecb17c037` 已推送并部署到 `P/repos/dreamlite-condition-control-20260913`。53项针对性测试通过。预注册在reports/official-base-condition-control-20260913.md。
- 复用scripts/probes/official_base_guidance.py，默认native CFG1；新`--condition-style training_raw`在原生28步循环中使用训练缓存condition/mask，三分支重复同一条件、CFG和imageCFG均1，恢复hook并拒绝未消费override。加载LoRA/full权重之后全部冻结，结束检查无梯度/版本不变；修复旧probe仅能审计LoRA的问题。
- **已经排好本次full退出后的串行无训练对照，不要重复提交**：GPU上实际queue PID112379，脚本 `P/runs/dreamlite-official-alignment/run-full-condition-controls-712c00d.sh`，日志 `P/runs/dreamlite-official-alignment/712c00d-full-condition-queue.log`。检查/proc/40541/cmdline必须仍指向full run/train才等待；父进程退出后各probe各自核验completed/result/checkpoint。队列等待上限06:00（1789250400）。
- 固定两输出：`P/runs/dreamlite-official-alignment/712c00d-full-native-guidance1` 和 `P/runs/dreamlite-official-alignment/712c00d-full-training-raw-guidance1`。每个50raw generations、8图/轨迹和complete.json；第二个phase叫training_raw_guidance1，第一个guidance1。不要使用只找train/trained的主训练collector解析这两probe，应独立读phase complete/summary/generations并验hash。
- 队列原脚本本地.cache/run-full-condition-controls.sh；单卡H200仍只本任务使用，不修改运行中的full或LoRA源码。完成图预览标题已增加全U-Net/LoRA rank标识。

## 03:10 首个功能正结果

**Full512正式completed**，resultSHA `51332a99686e4de865bdfc344e68f371b132e0d9a741c1f0410c5852563f485b`；CPU新collector重新核验result和最终checkpoint通过。全部5问法8/8正确且立即EOS，raw40条均ambient、tokens[59614,151645]，baseline全部错误，blank/donor10条均错误。389,968,388参数，deltaL2=19.940394，末64FMloss0.054692 vs相同draw LoRA0.436569。完整512updates/2048draw已对齐。8生成距训练teacher RMS0.05682–0.06772。**这只证明单题可学，不标goal完成。**

- 已下载 `reports/official-alignment-results-20260913/full512-evidence.tgz`、full512-summary.json、full512-preview.png；文本展开到full512/，本地phase文本SHA和result再次通过，PNG已view。PT和4.4GiBcheckpoint留远端。训练配对报告full512-draw-comparison.json。
- Full训练PID40541已结束。Queue112379已自动进入第一臂：native CFG1 wrapper PID190916、GPU worker PID191076（显存22624MiB）。两臂既已预注册，继续收集，不重启父实验、不取消来替换成功端点。
- 主LoRA3500仍活跃，最后观察2786步。容器PID411798/宿主GPU PID2622580。预计03:30附近结束。
- 完成图生成器最新共享副本 `P/runs/dreamlite-official-alignment/render_alignment_preview_v3.py`，title正确区分fullU-Net/LoRA rank；后续用它。
- 下一步：收集两条件臂和主LoRA；事前固定全新噪声的full最终权重确认测试（继续原生CFG7.5，不基于待出对照结果挑配置），并用相同实体的事件值替换/清除检验是否仅恒定输出ambient。随后基于该证据准备多事件teacher与共享Writer训练，不能把当前单题结果充当可用版本。
- 旧16题数据有`mixed`类型，既含event_text也含query；后续提取事件必须包括它，不能只筛type==event。之前读取前缀出现gold与首个事件不一致就是漏看mixed更新，并非原始数据错误。旧BF16 teacher仍需当前FP32读取验证，不能直接导入。

## 03:22 新噪声确认已启动

- 两个712c00d条件对照和queue112379都已经结束。两臂各5问法8/8正确+EOS，40matched raw均ambient。CPU已逐个核验全部PT/JSON哈希与原始token评分，下载full-condition-evidence.tgz和full-condition-summary.json，文本展开full-conditions/并再次验SHA。不要再启动这两臂。
- 确认源码提交 `cb41fa1bc9c1cc50a6c4d178dacd6f9a89ec2765`，checkout `P/repos/dreamlite-writer-confirmation-20260913`，run `P/runs/dreamlite-official-alignment/cb41fa1-full-confirmation`，日志为run加`.log`。单卡实例dl-align-full-h200x1-20260913实际wrapper PID249356、GPU worker PID249886（03:22显存16960MiB，正在加载/生成），后来日志已见28步进度。
- scripts/probes/official_writer_confirmation.py绑定成功full512 resultSHA51332a...，检查完整parent manifest/cursor和checkpoint；加载后所有模型冻结。16个新namespace噪声对原始事件、同前4噪声对jazz/clear事件，blank/donor各一次；24张图、130条raw答题，原生CFG7.5/28步，deadline1789251300（06:15）。无训练、不改父模型、不选择CFG对照结果。已编译并验证固定/配对/无重叠seed计划，调度前note+exactplan JSON已提交。
- 主LoRA3500最后观察3320/3500，仍等待端点及最终校验，容器PID411798。继续收集，不能因剩余少而提前停。
- .gitattributes新增结果目录 -text，实测即使Windows core.autocrlf=true，Git checkout过滤后仍保持full result原始SHA。证据tar仍为原始bytes来源。
- 旧16题FP32兼容性还没有执行。若要复用：B/seed0/step256在各lane/runs/target-XXX-seed-00-B/checkpoints/step-256.pt，payload键latent_fp32，checkpoint_index.jsonl有SHA；原训练是BF16VAE、仅original_open CE+EOS。旧三问法的instruction行也不同于当前五问法，不能只换dtype后就宣称满足新bank合同。后续应固定新五问法做FP32读取，或按已验证FP32三训练问法+两留出问法重新准备同实体ambient/jazz/clear的独立目标，再训练共享Writer；取决于当前确认的条件敏感性结果。

## 03:31 全部现有训练/确认已完成；下一步多状态

**当前GPU上已无活跃计算进程**，两个实例都实际查询nvidia-smi确认空闲（2卡r2与1卡full）。保留供紧接的目标构建/共享Writer训练使用，不再等待/重启旧PID或队列。CPU仍可用。主LoRA3500正式completed，全部五问法0/8；resultSHA `dc5b38055139bf202b92ad860300e984dee29a075294a419722589a7d4516604`，末64loss0.332762，deltaL2=33.412885。CPU重验result/checkpoint通过，base3500-evidence.tgz、summary、preview均下载、文本展开base3500/复核，图已view。

Full确认cb41fa1也completed：原事件16新噪声×5问法80/80正确立即EOS；jazz和clear各20条全失败，仍raw ambient/[59614,151645]，blank/donor各5失败。全部artifact哈希远端复核，full-confirmation-evidence.tgz/summary下载，展开full-confirmation/，文本/PNG再次验SHA，paired事件图full-confirmation-preview.png已生成并view。这证明当前权重是单状态记忆，并未实现事件值变化；goal仍active，下一步必须共享Writer多状态学习。

已查明更合适的已验证oracle配方（不必先迁移旧BF16多题）：当前ambient teacher来自 `P/runs/direct-multiprompt-eos/46cd36b-20260909-r01/lane-1/runs/direct-gaussian-s07-a0.5`，源码commit46cd36b在本地git可读。其FP32 VAE+bf16 Reader、Adam lr.05、256更新、EOS权重1、三训练问法original_open/paraphrase_1/paraphrase_2按zero-based round_robin；paraphrase_3/4只用于评测。不是每步累积三个问法。`git show 46cd36b:scripts/experiments/direct_geometry_eos_training.py`已读，可据此恢复显式可选multiprompt能力，当前worktree该helper仍为仅original版本，不能误调用默认就声称复现三问法。

建议下一步具体执行：在已空闲的2-H200 r2上准备同实体jazz/clear两个FP32 latent目标，与既有ambient teacher一起构成3状态bank，再用1-H200 full实例训练共享官方FM U-Net。初始化固定复用上述hash选定teacher的原始高斯seed7/scale.5起点；原始`latents/step-000.pt`(payload latent_fp32)可从source_run读取并用latent_index.jsonl及bank记录的index SHA验证，避免按新状态结果选初始化。每个新目标256更新、三问法轮转、两问法留出；所有5问法成功+EOS后才能封存，不能失败后换seed择优。仍须事前确定新实验配置后再派发。

原ambient bank teacher记录（.cache/teacher-bank-manifest.json内）有完整来源：source manifest SHA d331859338726a6c90cbca1448f74c47840a264b27603a0d60a5af2792bb201e，latent_index SHA2ee4b8c90d254649bb2b68324b20e03cc492963acad5a4c238190aa2e5b48d8e，endpoint SHAa58c978cf78dc2955705a8d370f7d60bf9e1805343a6e563d108be9f455e6cdd。当前独立teacher tensor SHA e910e9b89ed3861ef7c36176c762bccde7d9d809e2510618b5318202313777f6。

新bank若按状态使用不同group ID，必须明确这是同一语义问题的3种条件状态，不能报成3道独立题；问题5种问法沿用原组，Writer只接事件和source，不接query/gold标签。沿用source_kind blank_gray_1024、原封存source tensor及PIL128条件差异约定。不同答案donor仍可用原orange RGB control。暂未实现或调度多状态oracle/Writer，不要把建议写成已运行。

## 03:56 三状态目标完成，共享 Writer 实际开始优化

上述03:31建议已执行。代码c3bee0e8564ca379300326ed3f2d2ebe9ec8d8d5、checkout `P/repos/dreamlite-state-oracles-20260913`、run `P/runs/dreamlite-official-alignment/c3bee0e-state-oracles` 已完成。ambient复用固定原目标；jazz和clear各从已核验的原seed7/scale.5初态训练256更新，三问法计数86/85/85，p3/p4不参加梯度。三个目标都五问法正确且立即EOS。CPU核验45条raw、两轮512更新、所有中间latent/checkpoint文件SHA；本地再验三个bank tensor canonical SHA。证据state-oracles-evidence.tgz、state-oracles-summary.json、state-oracles/已下载。三目标pairwise RMS分别0.36555954、0.43353733、0.42337517。这些是oracle目标，不是共享Writer输出。

原bank SHA4e68df0fe38a4eb38c6cbf1903229ac81598d7b582c480e1700dbcf69e15b2f9遗漏snapshots字段，首次共享Writer `63483ac-three-state-full1536-20260913` 在模型加载前KeyError、0更新退出，three-state-first-attempt.tgz保留。修复exporter并用scripts/reporting/complete_state_bank_metadata.py产生新manifest；该工具验证原始bank和parent bank SHA、目标及raw hashes，只复制原有snapshot绑定且记录provenance，原manifest不变。新bank `P/runs/dreamlite-official-alignment/c90896c-state-bank/manifest.json`，SHA **5165c059a0a4c0760cd4b0df1663c642132e52bf55a13031cd99207fd200bb86**，本地state-bank-complete-manifest.json再验SHA及groups/teachers逐字段不变。

**当前唯一活跃GPU训练**：源码c90896c55c7b6cd6f489ccb7848ade2cff587daa，checkout `P/repos/dreamlite-three-state-writer-20260913-r2`，run `P/runs/dreamlite-official-alignment/c90896c-three-state-full1536-20260913`，stage-0-1789242667553720835.log。单H200实例 `dl-align-full-h200x1-20260913` 实际GPU PID396075、45842MiB；03:56观察28/1536更新，约1.9秒/步，已出现checkpoint。不是仅提交/加载。预注册reports/official-three-state-writer-20260913.md：fresh Base full U-Net、lr5e-5/accum4/clip1/wd1e-4、三状态各2048训练draw、原生28CFG7.5/image1；没有继承ambient-only权重。1536结束后还要完成3组baseline/trained的真实Reader评价。deadline1789251300（06:15），实例约06:40到期。55项相关测试通过。不要重启这个run。

2-H200实例 `dl-align-h200x2-20260913-r2` 已实际核验无GPU计算进程，目标/旧训练证据均在共享盘且本地下载，现已stop并delete。保留上述单H200和CPU实例；不要等待已删除实例的历史PID。三状态Writer尚未功能验证，goal继续active。正在预注册新噪声及两种留出事件表达的确认测试；只有固定1536端点的每个状态/问法/噪声全部通过，才进入该确认。

## 04:10 确认代码已准备，主训练继续

本地新commit **1e4acd2bcd9cbf9f0b2878e51694c17bebe06110** 已产生，包含所有三状态目标证据、修复后manifest、首次失败记录、scripts/probes/official_three_state_confirmation.py、固定JSON计划和预注册。57项原套件/确认门槛测试加1项按组统计回归，共58 passed。确认方案：每状态原始事件16全新配对噪声、两种未训练事件表达各4噪声；72张图/360真实答题，加按三答案重复评分的blank/donor30行。确认脚本在任何模型加载前要求父commit/bank/1536步设计和完整已成功的120个development matched单元，拒绝重复、漏项、失败、额外token；保存全部轨迹、raw和冻结边界。不是自动调参或挑选最佳端点。

本地exec session72824的git commit/gc/push已全部exit0，1e4acd2已推到origin。不要重新等待或重启它。Git common dir为D:/2026WorkExperience/VisonLearnableMemory/.git。后续audit、此续接记录、render_alignment_preview.py多状态修复另行提交。新render用(question_id,noise_seed)映射，避免三组同噪声答案覆盖，按组排版。

**确认队列已实际启动，不要重复提交**。CPU已经fetch并创建 `P/repos/dreamlite-three-state-confirmation-20260913` 锁定1e4acd2，shell位于 `P/runs/dreamlite-official-alignment/run-three-state-confirmation-1e4acd2.sh`（本地.cache/run-three-state-confirmation.sh）。单H200实际bash PID540574，GPU上仍只有父训练PID396075/45842MiB。队列每20秒等父terminal到06:00，随后固定调用确认脚本，deadline06:15；失败门槛会在加载模型前拒绝。固定输出 `P/runs/dreamlite-official-alignment/1e4acd2-three-state-confirmation`，worker日志同路径加.log，queue.json记录退出码，外层nohup日志同路径加-queue.log。仅退出0还不等于功能成功，须检查complete.json、各state/style/prompt的raw/EOS和所有hash。父训练仍为c90896c，不修改其checkout。

主训练最近实际观察335/1536（04:06–04:07），1536预计04:45前后结束优化，其后还有3组原生采样/Reader评测；具体以terminal/raw为准。没有新功能结果。已上传新版CPU报告器为 `P/runs/dreamlite-official-alignment/collect_groups_1e4acd2.py`（scp session73539已exit0）；其per-conditional-group计数避免constant-state被aggregate掩盖。它可先采进度，也可在端点后核验checkpoint/result并打包。原state bank遗留descriptive planned_count/successful_run_count=96来自parent，但实际每组teacher列表只有1，训练完全不读这两个字段；future exporter已改为1，正在运行的manifest保持不变，预注册也公开此点。
