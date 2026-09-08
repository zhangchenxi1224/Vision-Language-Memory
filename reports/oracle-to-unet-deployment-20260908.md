# 两路 EOS oracle 自动接续 U-Net：实际部署记录

截至北京时间 2026-09-08 22:40。Job 正在运行 Frozen DreamLite 输入端优化，Direct 剩余任务已转移到 `vlm-r11-trust-h200x4-20260907-r3` 并确认新轨迹步数增长。两路的 U-Net 自动接续进程均已启动等待；U-Net 本身尚未开始训练。

## 已交付的自动流程

完整新起点优化 → 检查原始 EOS 答题结果和 257 点轨迹 → 分别封存 Direct / Frozen 成功 latent 集合 → 几何分析 → 分别训练 DreamLite U-Net LoRA → 新随机噪声下生成图片 → Reader 训练前后配对评测。

U-Net 训练从成功集合采样，采用符合真实 DreamLite 初始状态的条件 flow matching，不取多个 latent 的平均值作目标。每路配置两个独立 seed（20260908、20260909），每个512步；每个训练占一组两卡。VAE、Reader和condition encoder冻结，更新U-Net LoRA。集合拟合损失不代表答题成功，最终以新的生成结果评测。Direct单题、Frozen最多8题，同512总更新的每题曝光量不同，不据此直接推断哪路bank更优。

联合实现提交：`a43b2e4434d0a97428c5d80280092bbdba687d21`。
启智固定源码：`/inspire/ssd/project/exploration-topic/czxs26210936/repos/latent-bank-unet-20260908`。

## Job：已实际启用自动接续

平台名仍为 `vlm-oracle-geometry-h200x4-20260908-r02`。旧MCQ训练已停止，原allocation继续运行新Frozen DreamLite xT填空题＋EOS实验；没有重新排队。新oracle提交为 `c97a75435b05d995698739b9118c548539d03dbb`，输出为 `P/runs/frozen-oracle-eos/c97a754-20260908-r01`。

22:40:39 从共享盘逐条复核 `campaign_terminal.json` 的成功返回码及 `summary.json` SHA 后，完成 **42/158**：6 条重复性验证、36 条独立起点。当前36条独立起点都针对 ambient 同一道题，原问句 **23/36** 正确，五种问法全部正确 **0/36**；两条新的 Uniform 轨迹到105/256和101/256。22:34 的上一快照为40/158，独立部分21/34正确，不同时间的分母不能混用。

重复性组中linen原问和五问法均正确，ambient原问错误；这6条不作为独立成功样本统计。上述42条也不是全158条或全8题的最终结论。

两条CPU sidecar均已验证存活，状态 `waiting_for_oracle`、`unet_training_started=false`，不会占用oracle正在使用的GPU：

| U-Net seed | GPU组 | sidecar PID | 独立输出目录 |
|---|---|---:|---|
| 20260908 | 0,1 | 180593 | `P/runs/oracle-to-unet/a43b2e4-frozen-seed20260908` |
| 20260909 | 2,3 | 180598 | `P/runs/oracle-to-unet/a43b2e4-frozen-seed20260909` |

各目录保存 `pipeline.json`、资源期限、启动命令、状态、日志，并将分别保存bank及U-Net结果。验证证据：`P/runs/oracle-to-unet/a43b2e4-frozen-preflight/dispatch_verification.json`，SHA256 `670c43d1d1765f5c7a3396a171559ddce3858bf1cfabf9f218bb94ab13d7a17d`。

仅当158条计划完整结束、oracle总完成状态及原始产物校验通过、成功集合非空，才自动进入U-Net。没有成功目标会报告实际失败原因，不伪装为完成。旧campaign PID54只暂停保活，不能恢复MCQ；全链路结束后用平台正式停止动作释放资源。

## 四卡开发实例：Direct 已在 trust 实际接续

`dl-base-h200x4-20260907` 已于原定约19:43自动停止。Direct此前在完整轨迹边界保存36/96条，原问句36/36正确、五种问法全正确26/36；两lane的源码/模型末尾校验均通过。源提交 `0f704178137ee4beac952224f3175a3923d5e438`，输出 `P/runs/direct-latent-geometry/0f70417-20260908-r01`。

原来的两套 `a43b2e4-direct-seed20260908/20260909` 配置没有启动，保留为历史准备记录。后来确认已排到的 `vlm-r11-trust-h200x4-20260907-r3` 四卡实例处于运行状态、GPU空闲，现已使用该实例接续；`dl-base-h200x4-20260907` 没有恢复，也不是当前训练主机。

22:38:43 已实际启动原冻结代码接续，保留原 `0f70417` 输出目录，完成的36条由原runner验证并跳过，继续剩余60条。resume helper PID22052、原launcher PID22070、两条worker PID22477/22478。22:40:16 的主机检查确认：lane0 的 `direct-uniform-s07-a1` 已保存 step73，lane1 的 `direct-sphere-s07-a1` 已保存 step71，两个日志均已输出 optimizer_step65。GPU0/2分别占3564 MiB，GPU1/3分别占10320 MiB，四卡均已加载相应模型并执行两条独立优化。

按此前每条约98秒、剩余两lane各30条估算，Direct 剩余优化约50–70分钟；这不包含后续bank审计、几何分析及U-Net训练，失败或平台重启会改变时间。当前36条既有成绩仍是原问36/36、五问法全对26/36，新轨迹不能在完成前计入终点成功数。

两套新trust sidecar已启动，均为 `waiting_for_oracle`、`unet_training_started=false`：

| U-Net seed | GPU组 | sidecar PID | 新的独立输出目录 |
|---|---|---:|---|
| 20260908 | 0,1 | 22053 | `P/runs/oracle-to-unet/a43b2e4-direct-trust-seed20260908` |
| 20260909 | 2,3 | 22054 | `P/runs/oracle-to-unet/a43b2e4-direct-trust-seed20260909` |

每份配置绑定trust当前容器的实际hostname和GPU UUID。oracle计划执行上限为北京时间9月9日04:38:18，sidecar/U-Net执行上限为10:38:18；分别是设置时起6小时、12小时的任务预算，**不是平台到期时间**。平台当时显示 `auto_stop=0`，不会据此宣称任务可以无限运行。

## 迁移归档与新容器校验

接续前已归档274个文件，共42,206,946 bytes，包括原root/lane元数据、reference/control tensor、日志和36条已完成run的manifest、terminal、generation与轨迹索引receipt；全部复制件SHA复核通过。原始257步latent仍保存在原run目录，不移动、不覆盖。

- 归档：`P/runs/direct-latent-geometry/0f70417-20260908-r01/provenance/resume-to-trust-20260908`。
- `archive_manifest.json` SHA256：`f21c41b3404af5d7206bdaee84a6259d7840b559bf5b942f20e5d2673b8014cd`。
- 36条与后来60条所属主机阶段可通过归档和续跑dispatch区分，不能用后写入的lane manifest抹去迁移历史。

trust 在正式启动前发生过容器更换，hostname guard 按设计拦截了旧绑定，未启动训练。确认四张GPU身份不变后，仅修改尚未激活配置的容器绑定；旧9份配置/metadata已备份到 `continuation-trust-20260908/provenance/container-rebind-1529e224dfc5-wj6s2jn4v3`。后续任务预算修改也独立归档，未修改科学源码、数据、初始化或优化参数。

最终有效容器：`vlm-r11-trust-h200x4-20260907-r3--1529e224dfc5-wj6s2jn4v3`。在这个容器上，先按原顺序开启严格确定性，再执行零更新技术预检；两lane的reference和blank逐位一致，Python/依赖/CUDA/cuDNN一致、模型文件验证通过、EOS合同一致，完整VAE→Reader初始loss及latent梯度SHA与原首条完全一致：loss `13.129262924194336`，gradient SHA `c7418ef0cfd6ce5ee12324b0a99cd555b333ae82a5e457678e687bc53c3d395b`。

预检记录：`P/runs/direct-latent-geometry/0f70417-20260908-r01/continuation-trust-20260908/preflight-container-wj6s2jn4v3.json`，SHA256 `a6945bc3e1171f9b4efaed8ce1a8ea5ea83fd2bd659b36b5356d041774aeea33`。首次预检曾因在VAE编码后才启用严格确定性而不匹配，该未通过检查没有启动训练；最终检查修正执行顺序后才接续。

真实启动记录：同目录 `dispatch.json`，SHA256 `5342a684589d9d6d8055411b5734f7bd468092da085940dea1b27665d612a273`。`planned_resume.json`、`allocation.json`、`config_ready.json` 记录当前实例、绑定、期限和校验链。原冻结launcher会在 `launch.json.instance` 写入硬编码的旧名 `dl-base-h200x4-20260907`，这个遗留标签**不代表当前实际实例**；本轮接续以真实dispatch/hostname及GPU进程为准，源码未为改标签而热更新。

## 验证及限制

40项联合CPU检查通过，随后完成状态检查回归通过。已在真实启智环境读取Direct首条和两条完整Frozen EOS产物，验证轨迹、checkpoint、EOS及评测行、图片和latent SHA，实际schema衔接通过。

U-Net尚未实际开始训练，所以此记录没有U-Net训练成绩或CUDA训练成功声明；真实模型加载、梯度与checkpoint检查会在自动训练启动时执行，失败会停止并保存原因。几何低秩/PCA只作有限样本诊断，不直接宣布发现低维流形或真实语义簇。

`P = /inspire/ssd/project/exploration-topic/czxs26210936`。
