# R11_new 局部方向 / Jacobian 保真度扫描：结果交付

## 结论先行

本轮技术执行有效，但科学上仍是失败诊断，不是 Picture Memory 成功：

- 115 次冻结 DreamLite 完整前向、2 次反向、0 次 optimizer step、0 次 Reader forward；
- preflight 与 formal 工程门均通过，模型快照不变，独立审计通过；
- 预注册分类为 `plateau_oracle_not_strongly_descending`；
- `formal_success=false`，`phase2_allowed=false`。

核心发现是：`lr=0.001` 的 step-256 平台并非“梯度太小”或“还没多训练几步”，而是
进入了一个 endpoint loss 较低、但其局部梯度与精确 teacher 解方向几乎正交，且 BF16
前向有限差分与 autograd 方向明显失配的区域。此时继续调 Adam 学习率没有充分依据。

## 固定实验与真实计数

本轮逐字节复用低学习率父实验的同一 target、`alpha=0.99` step-0 checkpoint、
`lr=0.001` raw step-256 checkpoint 及 Adam state。两个锚点分别扫描：负 autograd、
负 Adam proposal、oracle teacher-residual、确定性正交控制；L2 半径固定为
`[1e-4, 1e-3, 1e-2, 0.1, 0.3, 1, 2]`，每个方向测正负两侧。

总计 `2 × 4 × 7 × 2 = 112` 个 scan forward；加 1 次 teacher replay 与 2 次 anchor
gradient forward，formal 总计 115 forward。全程无训练更新、无梯度裁剪、无 Reader。

| 锚点 | 方向 | 最佳正向 loss / anchor loss | 半径 | 达到 `<=0.9` |
|---|---|---:|---:|---|
| alpha=.99 start | 负 autograd | 0.509852 | 0.3 | 是 |
| alpha=.99 start | 负 Adam proposal | 0.389120 | 1.0 | 是 |
| alpha=.99 start | oracle teacher residual | 0.236818 | 2.0 | 是，但未达 oracle `<=0.1` |
| alpha=.99 start | 正交控制 | 0.964847 | 0.001 | 否 |
| lr=.001 raw256 | 负 autograd | 0.959377 | 0.001 | 否 |
| lr=.001 raw256 | 负 Adam proposal | 0.952706 | 0.01 | 否 |
| lr=.001 raw256 | oracle teacher residual | 0.912159 | 0.0001 | 否 |
| lr=.001 raw256 | 正交控制 | 0.960788 | 0.0001 | 否 |

## 第一性原理解释

固定前向可写为 `L(x)=MSE(F(BF16(x)), F(BF16(x_teacher)))`。虽然 BF16 cast 允许
autograd 回传一个 surrogate gradient，但真实前向关于 FP32 `x` 是量化、分段且可能
高度不光滑的。实验给出四条直接证据：

1. 平台梯度非零且稠密：L2=`1.65834e-4`，非零比例 `99.9603%`；所以失败不是无梯度，
   更不是梯度裁剪造成，本轮根本没有 clipping。
2. 平台负梯度与 teacher residual 的余弦仅 `0.001492`，Adam proposal 也仅
   `0.007854`；二者几乎不朝向已知精确解。起点对应余弦仍有 `0.0960` 与 `0.1519`。
3. 平台负梯度的解析方向导数为负，但 7 个中心有限差分只有 1 个与其下降符号一致；
   正交负控制的解析导数接近零，真实有限差分仍明显波动。这是 surrogate Jacobian 与
   BF16 实际扰动不一致的直接证据。
4. 平台到 teacher `xT` 的距离为 `2.051211`。沿 oracle 方向移动 2.0 后只剩
   `0.051211`，已有 `92.0685%` BF16 坐标与 teacher 相同，但 endpoint loss 反而为平台的
   `1.63788×`；而精确 teacher `xT` 可逐位重放、loss 为 0。也就是说，尚未解析的变化
   集中在最后 `0.051211` L2 区间内，不能假设这条路径平滑。

因此当前最符合数据的机制是：低学习率 Adam 从起点进入了一个“不同 `xT`、相似
endpoint”的量化等价/近等价盆地；进入平台后，autograd 的局部方向已不再可靠地预测
真实 BF16 前向下降。增加 step 很可能只在该误差地板附近移动，不等于学会记忆。

## 下一步（尚不改变最终实验口径）

按预注册决策，下一轮先做极小的 oracle terminal-capture 扫描，而不是直接再训 Adam：

- 固定同一 plateau 和 teacher，只扫描二者直线在“距 teacher 还剩多少 L2”这一坐标；
- 在 `[0, 0.051211]` 内加密，包含 exact teacher、BF16 code 转换区和双侧重复重放；
- 回答 endpoint loss 从 `1.64×` 何处降到 0、是否由少数 BF16 临界坐标控制；
- 若存在可解析的连续 capture 区，再设计 continuation/residual 参数化；若只在精确
  BF16 cell 内骤降，则当前 FP32+BF16 surrogate 优化本身不适合作为 writer 训练接口，
  应改变可微参数化/计算精度，而非继续扫学习率。

该后续仍是固定 target 几何诊断。只有它解决优化可达性后，才回到事件→状态 writer、
Reader 因果读取、固定数据、多 seed 和 ID/OOD；这些最终科学门槛没有被本轮替代。

## 交付与复现

- 实现 commit：`8f696faa1176873045c39dcd8fa910db3e9cb558`；
- 原始归档：`raw/r11-new-direction-8f696fa-20260907-round01.tar.gz`；
- 原始归档 SHA-256：
  `02cf6befb4171168d87c4b6512820b73904a05ef20712185177666b1613ea6f3`；
- preflight inventory：23 项，SHA-256
  `ed84cf70a09c52d955a4bb541f4957068fa9ccef07a2dc4ad4ef2d9509e6635d`；
- formal inventory：136 项，SHA-256
  `307883bf485b4630d2c2242ad36da378db66853fdcf552968ecdecdcb7744db7`；
- `formal/` 与 `preflight/` 提供可直接阅读的日志、环境、配置、指标与终态；
- `derived/summary.json` 是独立重算摘要；`figures/` 与 `images/` 是可视化；
- `render_results.py` 会校验 archive、两个 inventory、159 个清单内 artifact，并在临时
  目录调用独立审计器重算结果后再生成图表；`DELIVERY_MANIFEST.json` 覆盖交付包文件。

原始归档包含全部 endpoint tensor、112 张 scan 图片、父 target/checkpoint 副本、完整
metrics、日志、运行环境、模型快照、配置、manifest 与哈希清单；精选目录不代替原始包。
