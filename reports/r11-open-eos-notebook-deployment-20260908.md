# Open/EOS 已迁移到可用两卡实例

更新时间：2026-09-08 15:15（Asia/Shanghai）。

用户指定 `vlm-r11-open-h200x2-20260907` 已分配资源，要求立即在该实例运行。
已于 15:13:01 启动原配对方案 `all`，PID `29021`。

## 实际资源与固定源码

- Workspace：分布式训练空间；Project：前沿课题探索。
- 实例：`vlm-r11-open-h200x2-20260907`；节点 `qb-prod-gpu2161`。
- 镜像：`ngc-pytorch:25.02-cuda12.8.0-py3`。
- 实际两张 NVIDIA H200，每卡 143771 MiB；启动前无 GPU 计算进程。
- GPU 0：冻结 VAE；GPU 1：冻结 Reader。
- 固定训练源码：`e6e6c8071a6c1311a966bda1aa636748fbb016f6`；远端源码树保持干净。
- 本文是后续运行记录，不改变固定训练源码、目标、seed、初始化或生成设置。
- 平台启动时显示约 3 小时后自动停止；本次未修改该计时设置。

## 命令与产物

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/inspire/launch_r11_open_eos.sh \
  all /inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-open-eos/all-e6e6c80-20260908
```

工作目录：
`/inspire/ssd/project/exploration-topic/czxs26210936/repos/r11-open-eos-20260908`

启动凭据及后台日志：
`/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-open-eos/notebook-launch-e6e6c80-20260908/launch.json`

`/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-open-eos/notebook-launch-e6e6c80-20260908/stdout.log`

## 已核实与尚待验证

实际模型、软件版本、旧 blank VAE 编码及冻结对照检查已通过；旧 Open checkpoint 的真实生成回放已经落盘。
运行顺序为：8 seeds × 7 checkpoints 回放，然后 8 seeds × A/B/C × 256 步配对训练。
A 的每个保存步都必须逐位复现旧 Open latent；失配立即停止，不能继续宣称 B/C 改善具有配对因果效力。
启动成功与回放成功尚不能证明 EOS 方案有效；结论须以完整 raw generation 与 terminal 为依据。

## 四卡队列防重复

主线 `vlm-oracle-geometry-h200x4-20260908-r02` 仍在原队列，本次未取消或重建。
其原辅助命令引用与本实例完全相同的 Open/EOS 输出路径。
当前 runner 在加载模型之前执行 `output_dir.mkdir(exist_ok=False)`，因此队列辅助命令未来若再调用，
会因输出目录已存在而退出，不会重复训练或覆盖本实例产物。
主线 campaign 对辅助命令的非零退出仅记日志并释放 GPU 对，继续几何任务。
届时该辅助退出表示重复启动被阻止，不能据此把本实例的 Open/EOS 科学结果判成失败；
其真实终态必须读取上述输出目录自己的 `terminal.json`。

四卡几何实验尚未在本两卡实例运行。
