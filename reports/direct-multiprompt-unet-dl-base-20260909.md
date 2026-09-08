# 新 Direct 与后续 U-Net：全部在 dl-base 完成

北京时间2026-09-09 00:44现场验证：`dl-base-h200x4-20260907` 上新Direct完成48/96，两个U-Net自动接续进程均存活，状态为 `waiting_for_oracle`。96条Direct完成、原始结果校验和成功latent入库后，在同一个实例分别训练两组U-Net各512步，再完成新生成图片答题评测。Direct持续运行，没有重启或改动其训练源码。

## 原数据已经完成

旧单问法Direct的两组U-Net（seed20260908、20260909）已各完成512步，`terminal.json` 为completed，`result.json` SHA与终态记录一致。读取两组 `unet/trained/generations.jsonl`，新生成图片的原问法matched严格正确数均为0/8。完成训练不等于答题成功。

旧结果分别保留在 `P/runs/oracle-to-unet/a43b2e4-direct-trust-seed20260908` 和 `...seed20260909`，输入仍绑定原Direct提交 `0f70417`。未重跑、覆盖或改接到新数据。

## 新数据的执行位置

- Direct源码：`46cd36b1eb421471dafa7865fbab4dfe02336336`。
- Direct输出：`P/runs/direct-multiprompt-eos/46cd36b-20260909-r01`。
- U-Net/接续源码：`f5c9c9ef1604f6b2d0466e73a80acab692cb763a`。
- U-Net独立源码：`P/repos/latent-bank-unet-multiprompt-20260909`。
- 实例：`dl-base-h200x4-20260907`，容器 `dl-base-h200x4-20260907--46095613ce18-kk5gt4zzxr`。

| U-Net seed | dl-base接续PID | GPU组 | 独立输出目录 |
|---|---:|---|---|
| 20260908 | 676655 | 0,1 | `P/runs/oracle-to-unet/f5c9c9e-direct-multiprompt-dl-base-seed20260908` |
| 20260909 | 676656 | 2,3 | `P/runs/oracle-to-unet/f5c9c9e-direct-multiprompt-dl-base-seed20260909` |

每份配置从此前新数据配置复制，只调整输出目录和主机/GPU资源绑定；保持oracle路径、代码提交、seed、512步训练与评测设置。使用与Direct一致的当前任务预算（截至02:15:48），未修改平台自动停止设置。接续进程在CPU上等待Direct的96条完整结束与GPU释放，之后才加载训练模型。

原先在trust为新数据创建的PID867115和867116均已收到停止信号并确认退出，原输出status为paused且 `unet_training_started=false`。各自 `cancelled_for_dl_base.json` 记录取消原因，防止把这两份历史等待配置当作当前执行位置。

## 证据及指标边界

Direct输出根目录保存 `unet_handoff.json`（当前路由和两组dispatch）与 `unet_handoff_verification.json`（00:44真实PID/状态/48条进度）。每个新U-Net输出目录保存config、resource-lease、dispatch、status及sidecar.log，主机和GPU身份绑定已明确。

接续沿用现有审计：256条问法轮换记录、257个latent、11个checkpoint、EOS与15格终点评测、模型/数据/源码绑定；不把训练提示或保留提示混淆。成功集合仍按原问法step256 raw exact match入库，q3/q4不参与筛选。新U-Net当前尚未开始参数更新，最终成绩应读取各自 `unet/result.json` 和原始generations，不能用部署通过代替科学成功。

`P = /inspire/ssd/project/exploration-topic/czxs26210936`。
