# 两路 EOS oracle 自动接续 U-Net：实际部署记录

截至北京时间 2026-09-08 19:56。下列训练代码已提交并同步到启智共享盘；部署状态区分实际运行与准备完成。

## 已交付的自动流程

完整新起点优化 → 检查原始 EOS 答题结果和 257 点轨迹 → 分别封存 Direct / Frozen 成功 latent 集合 → 几何分析 → 分别训练 DreamLite U-Net LoRA → 新随机噪声下生成图片 → Reader 训练前后配对评测。

U-Net 训练从成功集合采样，采用符合真实 DreamLite 初始状态的条件 flow matching，不取多个 latent 的平均值作目标。每路配置两个独立 seed（20260908、20260909），每个512步；每个训练占一组两卡。VAE、Reader和condition encoder冻结，更新U-Net LoRA。集合拟合损失不代表答题成功，最终以新的生成结果评测。Direct单题、Frozen最多8题，同512总更新的每题曝光量不同，不据此直接推断哪路bank更优。

联合实现提交：`a43b2e4434d0a97428c5d80280092bbdba687d21`。
启智固定源码：`/inspire/ssd/project/exploration-topic/czxs26210936/repos/latent-bank-unet-20260908`。

## Job：已实际启用自动接续

平台名仍为 `vlm-oracle-geometry-h200x4-20260908-r02`。旧MCQ训练已停止，原allocation继续运行新Frozen DreamLite xT填空题＋EOS实验；没有重新排队。新oracle提交为 `c97a75435b05d995698739b9118c548539d03dbb`，输出为 `P/runs/frozen-oracle-eos/c97a754-20260908-r01`。

抓取时完成6/158（都是重复性验证），两条A2独立起点到254/256。重复性组中linen原问和五问法均正确，ambient原问错误；重复轨迹不作为独立成功样本统计。

两条CPU sidecar均已验证存活，状态 `waiting_for_oracle`、`unet_training_started=false`，不会占用oracle正在使用的GPU：

| U-Net seed | GPU组 | sidecar PID | 独立输出目录 |
|---|---|---:|---|
| 20260908 | 0,1 | 180593 | `P/runs/oracle-to-unet/a43b2e4-frozen-seed20260908` |
| 20260909 | 2,3 | 180598 | `P/runs/oracle-to-unet/a43b2e4-frozen-seed20260909` |

各目录保存 `pipeline.json`、资源期限、启动命令、状态、日志，并将分别保存bank及U-Net结果。验证证据：`P/runs/oracle-to-unet/a43b2e4-frozen-preflight/dispatch_verification.json`，SHA256 `670c43d1d1765f5c7a3396a171559ddce3858bf1cfabf9f218bb94ab13d7a17d`。

仅当158条计划完整结束、oracle总完成状态及原始产物校验通过、成功集合非空，才自动进入U-Net。没有成功目标会报告实际失败原因，不伪装为完成。旧campaign PID54只暂停保活，不能恢复MCQ；全链路结束后用平台正式停止动作释放资源。

## 四卡开发实例：需恢复资源后接续

`dl-base-h200x4-20260907` 已于原定约19:43自动停止。Direct此前在完整轨迹边界保存36/96条，原问句36/36正确、五种问法全正确26/36；两lane的源码/模型末尾校验均通过。源提交 `0f704178137ee4beac952224f3175a3923d5e438`，输出 `P/runs/direct-latent-geometry/0f70417-20260908-r01`。

两套Direct接续配置已实际写入 `P/runs/oracle-to-unet/a43b2e4-direct-seed20260908` 和 `...20260909`，但没有启动sidecar，`deployment_pending.json`明确记为等待实例恢复及延长。不能把配置准备完成说成U-Net已训练。

恢复流程：先重新取得实际hostname/GPU UUID及已延长的运行期限，核验并更新尚未启用的配置与lease；用 `--resume-direct-oracle --deadline <新期限>` 继续原36条之后的剩余60条；启动两条sidecar等待完整96条结束。只更新lease不会自动重启oracle。实例重启后主机/GPU身份可能改变，不能沿用过期绑定直接启动。

## 验证及限制

40项联合CPU检查通过，随后完成状态检查回归通过。已在真实启智环境读取Direct首条和两条完整Frozen EOS产物，验证轨迹、checkpoint、EOS及评测行、图片和latent SHA，实际schema衔接通过。

U-Net尚未实际开始训练，所以此记录没有U-Net训练成绩或CUDA训练成功声明；真实模型加载、梯度与checkpoint检查会在自动训练启动时执行，失败会停止并保存原因。几何低秩/PCA只作有限样本诊断，不直接宣布发现低维流形或真实语义簇。

`P = /inspire/ssd/project/exploration-topic/czxs26210936`。
