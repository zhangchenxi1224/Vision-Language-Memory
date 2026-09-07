# R11_new Oracle terminal-capture 扫描：预注册

状态：在 local direction formal 完成、完整交付并推送后，且任何本轮 terminal-capture
DreamLite forward 之前锁定。网格只使用父 `xT` tensor 的 BF16 code 离线计算，不使用
任何尚未运行的 endpoint loss。

## 为什么必须先做这一轮

上一轮在 `lr=.001 raw step 256` 平台点得到：负 autograd、负 Adam proposal、oracle
teacher residual 的最佳正向 loss ratio 分别为 `0.959377`、`0.952706`、`0.912159`，
均未达到 `<=0.9`。沿 oracle 方向移动 2.0 后，距精确 teacher 只剩 `0.051211` L2，
`92.0685%` BF16 code 已相同，endpoint loss 却升至平台的 `1.63788×`；精确 teacher
本身可逐位重放且 loss=0。上一轮预注册要求此分支停止训练，先审计未测量的末端区间。

本轮只回答：**沿精确 plateau→teacher 直线，距 teacher 还剩多远时，冻结 BF16
DreamLite endpoint 才进入 `loss ratio<=0.1` 的强捕获区；该区域是否仅等于完全相同的
BF16 cell？**

## 固定父证据

- direction 实现 commit：`8f696faa1176873045c39dcd8fa910db3e9cb558`；
- direction 交付 commit：`912d987d41cdb0008f265a75c3367b11cb276147`；
- direction formal result SHA-256：
  `19f5ea49a4df0f23745748246c3aeb0e27f75dcb5927317fc50a1fee8f90ca97`；
- direction raw archive SHA-256：
  `02cf6befb4171168d87c4b6512820b73904a05ef20712185177666b1613ea6f3`；
- 固定 target artifact SHA-256：
  `5352fac1cf614c8458d4f61b6ffcbf7b053f8a947dedac1286f80261ba8a772a`；
- 固定 plateau checkpoint SHA-256：
  `3d166294353511a33b160f8f97a677de1440e7225245a9057d39a182c2563c80`；
- teacher `xT`、teacher endpoint、plateau `xT`、plateau endpoint tensor SHA-256 均在
  config 中锁定；禁止重生成、改 target 或挑选 checkpoint。

运行环境、数据、模型快照、conditioning、设备与精度逐项引用并校验上一轮 locked
direction config；仍只暴露物理 4×H200 中的 GPU 0、1。

## 预先选择的 17 个点

坐标 `delta` 定义为从候选 `xT` 到 teacher 的“请求剩余 L2”：

`[0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4,
3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 2.0512112034227754]`。

- `delta=0` 逐位复制 teacher；`delta=D` 逐位复制 plateau；
- 中间点固定为
  `float(teacher.double + delta × (plateau.double-teacher.double)/D)`；
- config 锁定每个候选 FP32 tensor SHA-256、实际 FP32 L2、BF16 L2 与 BF16 code
  相同比例；任一平台/操作系统重建漂移即技术失败。

网格选择前只看 BF16 code：`delta=1e-8` 时仍 100% 相同；`3e-8` 首个 code 改变；
`1e-4` 时相同率 `99.9741%`，但 BF16 L2 已为 `0.00501`；`0.05` 时相同率
`92.2073%`。这些只是输入量化边界，不包含新 endpoint 信息。

## 执行与门槛

技术 preflight：1 次 teacher replay + 1 次 plateau replay，共 2 forward、0 backward、
0 optimizer step、0 Reader forward；两端 endpoint 必须与父 artifact 逐位相同，17 个
候选 tensor/hash 必须全部匹配，模型冻结且仅 student `xT` 保持可求导属性。

formal 对 17 点做两遍：

1. `teacher-outward`：index 0→16；
2. `plateau-inward`：index 16→0。

两遍同点 endpoint 必须逐位相同，用于排除顺序/状态污染。扫描 34 forward；加模型加载后
独立 teacher replay，formal 总计 35 forward、0 backward、0 optimizer step、0 Reader。
保存 34 个 endpoint tensor、图片、逐行 metrics 与哈希。

loss 为候选 endpoint 与固定 teacher endpoint 的 FP32 elementwise MSE；ratio 分母为精确
父 plateau MSE=`1.6364127077e-5`。强捕获区为 ratio `<=0.1`，有意义捕获为 `<=0.9`。
主统计量是从 `delta=0` 开始连续满足强门槛的最大 `delta`，不允许结果后改网格或阈值。

## 预注册归因

- 强捕获连续到 `delta>=0.01`：`macroscopic_terminal_capture`；下一轮可测试 oracle
  continuation 是否让优化始终留在相邻捕获区；
- 连续到 `1e-4 <= delta < 0.01`：`narrow_terminal_capture`；先比较 BF16 cast 与更高精度/
  residual-centered 参数化；
- 只延伸到 `<1e-4`，但包含已改变 BF16 code 的点：
  `microscopic_partial_code_capture`；当前优化精度很可能不足；
- 只覆盖与 teacher BF16 code 100% 相同的点：`bf16_exact_cell_only`；当前 cast 对
  gradient writer 构成离散终端盆地；
- 仅 `delta=0`：`exact_teacher_only`；先审计构造和确定性；
- 强点不是从 teacher 连续前缀：`nonmonotone_capture_anomaly`；先定位少数敏感 BF16
  坐标，不选择优化器。

## 解释边界

本轮使用部署时不可获得的 teacher `xT`，只做固定自生成 target 的数值几何诊断；无论
结果如何都固定 `formal_success=false`、`phase2_allowed=false`。不得声称事件到状态
学习、共享 writer、Reader 因果读取、SET/overwrite、ID/OOD 或 Picture Memory 成功。
