# R11_new 局部方向 / Jacobian 保真度扫描：预注册

状态：在固定可达目标低学习率实验完成并交付后、任何本轮方向扫描 DreamLite forward
之前锁定。本轮是几何与数值诊断，不是 Picture Memory 成功实验。

## 已知事实与研究问题

同一 `alpha=0.99` 起点上，Adam `lr=0.005` 与 `lr=0.001` 均把 endpoint MSE
从 `1.3531206059e-4` 降到约 `1.6e-5`，但都未达到预注册的相对误差门槛
`MSE/M0 <= 0.01`。`lr=0.001` 已消除首步灾难性过冲，raw step 256 的
`MSE/M0=0.1209362`，同时 student `xT` 到 teacher `xT` 的 L2 仍为
`2.0512112`。因此，继续只缩小学习率已不能判别失败根因。

本轮只回答：在精确 `alpha=0.99` 起点和精确 `lr=0.001, raw step 256` 平台点，
autograd、Adam 预条件方向和 oracle teacher-residual 方向，是否真的能在冻结的 BF16
DreamLite 完整映射上产生有限差分下降？这把“优化器方向/步长策略失败”与“BF16 cast、
局部 Jacobian 或 residual 参数化失败”分开。

## 固定父证据与锚点

- 训练实现 commit：`af00d0418f4ead501a27bb1cab03b5cd9e3bc51a`；
- 低学习率完整交付 commit：`496c136ec6cf359ba81348302942acddea41fe40`；
- parent raw archive SHA-256：
  `9bfe553c8aa7a8eef8a69dcc19ef9d37fa182c7766548dc81b5df39737acfcd1`；
- 固定 target artifact SHA-256：
  `5352fac1cf614c8458d4f61b6ffcbf7b053f8a947dedac1286f80261ba8a772a`；
- `alpha099-start` checkpoint SHA-256：
  `623f2eea39f94f3ea8666dc3cc22cb6e707f9befe1cddac4304af9a47a5dd9d2`；
- `lr001-raw256` checkpoint SHA-256：
  `3d166294353511a33b160f8f97a677de1440e7225245a9057d39a182c2563c80`。

target、两个锚点 `xT`、baseline endpoint 和 plateau Adam state 全部逐字节复制父实验；
禁止重生成、挑选 best checkpoint 或更换 target。teacher endpoint 必须逐位重放，两个
anchor endpoint 必须与父 checkpoint 逐位相等。

## 唯一诊断网格

每个锚点固定扫描四个单位 L2 方向：

1. `negative-autograd`：`-g / ||g||₂`；
2. `negative-adam-preconditioned`：用该锚点当前梯度和父 Adam 的超参数/精确 state
   重建“下一步”Adam proposal，再单位化；起点 state 为空，平台点使用 step 256 state；
3. `teacher-residual`：`(teacher_xT - anchor_xT) / ||·||₂`，只作为不可部署的 oracle
   正控制；
4. `deterministic-orthogonal-control`：seed=`20260907` 的随机向量，同时投影掉梯度与
   teacher residual 分量，作为负控制。

固定 L2 半径为 `[0.0001, 0.001, 0.01, 0.1, 0.3, 1.0, 2.0]`；每个方向同时测
`anchor + r·d` 和 `anchor - r·d`。总扫描量为
`2 anchors × 4 directions × 7 radii × 2 signs = 112` 个完整 DreamLite forward。
loss 始终为 scanned endpoint 与固定 teacher endpoint 的 FP32 elementwise MSE。

“有意义下降”预先定义为最佳正向 `loss/anchor_loss <= 0.9`；oracle 强下降定义为
`<=0.1`。同时保存解析方向导数 `g·d`、中心有限差分、BF16 code 相同比例、每个 endpoint
tensor、图片与 SHA-256。禁止 optimizer step、gradient clipping、Reader forward、
半径后验修改或结果后改阈值。

## 技术 preflight 与 formal 计数

技术 preflight 仅做 1 次 teacher replay 和 2 次 anchor gradient forward：共 3 次完整
forward、2 次 backward、0 optimizer step、0 Reader forward。必须满足：父 target/
anchor/baseline 逐位重放；梯度有限、非零比例 `>=0.99`；四方向单位范数误差
`<=1e-6`；随机控制与梯度、teacher residual 的绝对余弦均 `<=1e-5`；模型冻结且只有
student `xT` 可求导。

formal 固定为 115 次完整 forward（1 teacher + 2 anchor gradient + 112 scan）、2 次
backward、0 optimizer step、0 Reader forward。所有 112 行及 endpoint tensor 必须有限、
完整、带哈希；模型快照前后相同。formal 必须与 preflight 使用同一已推送 commit、同一
Python 环境和同一两张可见 GPU（物理 4×H200 中只暴露 `0,1`，保持前轮可比性）。

## 预注册归因与下一步

- 平台点负梯度和 Adam 方向都下降：局部梯度可用，主因是 step policy；下一轮预注册
  单位负梯度 trust-region + 确定性 line search；
- 仅负梯度下降：Adam momentum/preconditioning 是主因；移除 Adam，改用单位原始梯度
  trust-region；
- 仅 teacher residual 强下降：精确解路径存在，但计算梯度不可执行；优先判别 BF16
  cast/Jacobian 保真度或 continuation/residual 参数化，不再盲跑 Adam；
- teacher residual 也不能强下降：停止训练并审计父锚点、target 重放和扫描半径分辨率；
- 仅 Adam 或随机控制出现反常下降：作为非单调/实现异常处理，先审计，不据此训练。

## 解释边界

本轮 `formal_success=false`、`phase2_allowed=false`，无论扫描结果如何均不得宣称
Picture Memory 成功。teacher residual 使用了部署时不可获得的 teacher `xT`；固定 target
是自生成几何目标，不是事件到状态映射；Reader、共享 writer、ID/OOD、SET/overwrite
能力均未评测。扫描只决定下一轮最小判别优化方案，不改变最终科学口径。
