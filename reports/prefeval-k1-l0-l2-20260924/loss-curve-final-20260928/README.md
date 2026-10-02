# B730 最终训练 loss

覆盖连续且唯一的 1–93,440 步全部已提交更新。每步值为四个 micro-batch 官方 FM MSE 的均值，平滑为固定 1,000 步后向均值。数据源与原始压缩日志 SHA256 见 source.json；图表与 CSV SHA256 见 plot-summary.json。

PNG、PDF、SVG、每步 CSV 和每 1,000 步分块 CSV 均由 plot-loss.py 生成。使用 Python、NumPy、Matplotlib 和 `C:/Windows/Fonts/msyh.ttc`，脚本可接收第一个参数作为 Python 绘图库目录。图中 loss 不是偏好读回准确率。

训练完整断点证明、原始包和本地复算位于 ../evidence/b730-exposure512-training-final-20260928。完整说明见 ../B730_EXPOSURE512_TRAINING_FINAL_20260928.md。
