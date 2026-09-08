# Frozen DreamLite Oracle Geometry：4×H200 执行协议

本协议在新实验的任何 GPU outcome 出现前冻结。原始设计来自用户
2026-09-08 附件《Frozen DreamLite Oracle Latent Geometry 与共享 Writer 学习工作方案》。
代码基线为 `7c27edb`，独立分支 `codex/frozen-oracle-geometry-20260908`。
新协议不修改旧 R11、R11_new 或旧结果的定义。

## 要回答的问题

固定 source 和事件后，完整 Frozen DreamLite 是否把不同控制输入导向
有规律、可复用、对扰动稳定的可读状态？若是，先学习预测控制输入的
小 Controller，再根据真实结果决定是否需要修改 DreamLite。

选择题只有少量行为约束，控制输入有 65,536 个自由度，因此不预设唯一解，
也不把一个任意成功 endpoint 直接当作唯一标签。控制空间 xT 与输出空间 z
分别分析；z 几乎相同不能推出 xT 可以做单标签回归。

## 固定实验条件

- Frozen DreamLite-mobile、官方真实事件条件编码、Frozen VAE、Frozen Qwen3-VL-4B Reader。
- DreamLite 全 FP32，Reader BF16，唯一可训练变量 xT 为 FP32。
- source 为 RGB 127/255 灰图的 VAE 编码；完整四步，sigma 为 .5/.375/.25/.125。
- 实际首个 flow state 是 `.5 source + .5 xT`；xT 本身不是图片。
- Adam，lr=.05，256 次更新，无梯度裁剪，无 checkpoint 挑优。
- 训练按原四个循环选项视图；主指标为第256步四个逆序视图全部严格正确。
- 原 Phase1A CE/accuracy/reset gate 另报，不能用它替代严格正确率。
- 数据、模型 snapshot、Python源码、配置、初始化和产物全部绑定 SHA256。

这是一项新的 FP32 几何协议，与旧 BF16 Phase1A 的数值结果不能直接作因果差分。
旧 R11 8条轨迹直接优化 VAE latent；此次完整执行 DreamLite，不能混用。

## 顺序与数量

| 阶段 | 预注册实验 | 独立优化 run |
|---|---|---:|
| Probe | 相同完整前向/反向重复2次，0更新 | 0 |
| A1 | target0/1，每题相同初始化与配置重复3次 | 6 |
| A2 | 固定 anchor target1，Gaussian seed0–31 | 32 |
| A3 | 五分布各8，Gaussian复用A2前8 | 32新增 |
| A4 | Gaussian相同8seed，scale=.25/.5/1/2/4，1复用 | 32新增 |
| A9 | 原8题×相同8seed，anchor复用 | 56新增 |
| A5 | 首个成功anchor，rho=.01/.05/.1/.25，各8方向 | 32条件新增 |
| A8 | 首8个成功anchor所有pairs，11插值点，xT/z分别读 | 0更新 |

主体158个独立优化run；存在成功anchor才增加32个局部重新优化run。
A5同时对原扰动点直接评估，区分局部功能鲁棒性与重新优化可达性。
全部研究成员关系由JSON manifest明确记录；复用结果不能重复计入独立样本。

## 从第一性原理修正的归因问题

1. 分布比较采用总体零均值/单位方差。原生Gaussian不逐样本归一；Sphere才
   固定RMS=1，否则两者会成为同一个实验。Uniform=U(-sqrt3,sqrt3)，
   Rademacher=±1，Heavy-tail=t(df5)×sqrt(3/5)。每次报告实际矩和峰度。
2. scale实验只改变同一噪声样本的乘数；局部rho表示坐标RMS，L2约rho×sqrt(D)。
3. “成功数/K”是指定分布、优化器和预算下的可达率，不是几何basin体积。
   技术失败、科学失败、缺失分开；报告实际分母和Wilson区间，不重抽失败seed。
4. 插值网格成功仅说明采样直线保持功能；中间失败不证明没有曲线连接。
   z插值中间点不保证由Frozen DreamLite可达；xT插值才完整经过生成链路。
5. 64个中心化样本的秩最多63。“r95只有几十维”不能单独证明低维协议。
   同时给样本秩上限、同形随机对照，并单独审查跨任务结构。
6. 全部257步xT/z保存，因此路径长度由每一步全维距离累计。PCA只用于展示，
   必须标解释率；二维轨迹交叉不能解释为高维轨迹相交。
7. 共同灰图/初始噪声可能支配绝对cosine，因此同时分析delta_xT/delta_z，
   并控制跨任务seed，区分初始化因素与任务因素。

## 自动门槛与资源

启智：分布式训练空间 / 前沿课题探索 / 4×H200、80 CPU、900 GiB RAM。
NGC25.02镜像复用原项目只读模型和overlay环境；新实例须通过实时4卡核验。
两对设备(0,1)/(2,3)，各自DreamLite+Reader，独立训练，不做DDP梯度平均。
科学任务总时限72小时；中断保留原始产物，不覆盖、不补造成功。

A1要求相同target三次的初始化、末步xT、末步z、loss序列、gradient序列哈希
完全相同。A1失败停止多起点研究。每个target的三次重复固定到同一设备对。
任一技术失败会停止本轮主线并保留日志；普通QA失败继续计入固定分母。
填空题Open/EOS支线有独立配置、分支和原始结果，受同一4GPU总额管理。

## Stage B/C/D

Stage A没有结果前，不输出虚构的canonicalization或监督选择。
`build_canonical_oracle_bank.py` 要求已审计的geometry decision、32–64个独立
事件任务（默认至少32train及独立heldout）、多成功端点、真实扰动稳定性、
正margin和prior阈值。失败任务保留ledger，不用成功任务替换。

`train_frozen_oracle_controller.py` 实现仅source+event输入的共享基底Controller，
根据控制空间的审计结论使用single/set MSE，先做小训练集过拟合。
预测xT及完整链路评估spec都会保存；仅MSE下降不构成成功。
同episode/event不能跨train与heldout；query/choices/answer/id不能成为Controller输入。

容量顺序为Controller→LoRA r4/r8/r16/r32/r64→扩大覆盖→Partial FT→Full FT上界。
扩大容量必须有前一级充分优化且完整DreamLite+Reader验证失败的证据。
本轮先部署可运行Stage A和有门槛的Bank/Controller入口；不在几何未知时自动
训练LoRA/Full FT。FM监督必须沿真实scheduler定义和宽度拼接接口另作实施审计。

## 结果边界

选择题正确不等于开放回答正确；单题oracle不等于共享Writer泛化；小bank过拟合
不等于部署成功。最终仍须原始生成EM、改写问法、扰动、matched/blank/donor、
episode隔离的heldout以及循环写入验证。
