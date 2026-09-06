# R11_new：冻结映射可达 endpoint 正对照预注册

## 为什么做这一轮

已完成的 canonical teacher 4↔7 因果换图审计表明，两张 R11 teacher 图不是通用触发器：own 均 4/4，交换后预测随 donor 的 blue/green 切换，灰图回到 no active preference。因而 teacher 标签有效性不再是当前首要嫌疑。

前序 identity-conditioned 完整链路虽然 256/256 步梯度有限且非零，但 raw256 MSE 仍为 `0.1113223583`、MSE/M0 为 `0.503226`、Reader 为 0/4。仅凭 canonical 失败无法区分两件事：原优化器根本不会解冻结映射，或 canonical teacher 对当前冻结映射不可达/几何上过难。本轮只回答这个最小问题。

## 固定因果设计

令固定 source、固定 `no changes` conditioning 和冻结四步 DreamLite 组成确定映射 `F_identity(x_T)`。

1. 学生初值仍为原 source-only `x_T_init`。
2. CPU FP32 固定 seed `2026090701` 生成高斯噪声，以 FP64 population RMS 归一化后转回 FP32；不搜索 seed、不重试幅度。
3. `teacher_x_T = x_T_init + normalized_noise × 1.0`。
4. `teacher_endpoint = F_identity(teacher_x_T)`，立即重复前向并要求 latent 与 RGB 逐位一致；随后恢复学生初值。
5. 学生只看到 detached `teacher_endpoint`，不把 `teacher_x_T` 传给 loss；唯一可训练量仍为学生 `x_T_fp32`。

目标由同一映射自产生，所以按构造保证存在原像。它不是语义 teacher，不做 Reader 评测，也不算 Picture Memory 成功。

## 不变训练口径

- 原目标 index 1 与原数据仅作身份/来源验证；condition 固定为 `no changes`，source 固定为 RGB `127/255`。
- 完整冻结 DreamLite-mobile 四步路径，effective sigma 为 `[0.5, 0.375, 0.25, 0.125]`。
- FP32 endpoint mean MSE；Adam、weight decay 0、无梯度裁剪。
- update 1–128 的 LR 为 0.05；129–256 使用原 cosine 衰减，update 256 为 0。
- 固定 256 次 optimizer step；只判 raw step 256，不用 best checkpoint；保存 0/64/128/192/256。

## 预注册门槛与停止规则

Technical preflight 必须先通过：teacher/replay、学生 M0、gradient probe 共 4 次完整前向；1 次 backward、0 次 optimizer；teacher 重放逐位一致；M0 MSE ≥ `1e-4`；梯度有限、norm > 0、非零比例 ≥ 0.99；仅学生 `x_T` 可训练，模型全冻结。

Formal 工程门槛：与 preflight 同一 pushed commit，重新构造的四个 target tensors 逐位相同；256 条连续 receipts 和原 LR 精确一致；每步有限非零梯度、无裁剪；counters 精确为 263 次完整前向、256 backward、256 optimizer、0 Reader；checkpoints 恰为 0/64/128/192/256；模型快照前后不变。

可达正对照通过需同时满足：raw256 MSE ≤ `0.001`、MSE/M0 ≤ `0.01`、L2/M0 ≤ `0.1`。边界不四舍五入。

- 若通过：只排除“该优化器连映射内目标都不会拟合”的解释，下一轮才预注册最小 generator-adaptation/canonical-teacher 诊断；不宣称 canonical 数学不可达。
- 若失败：暂停 LoRA 和扩数据，继续在同一可达目标上检查 Jacobian 条件数、尺度、精度与参数化。
- 任一技术门槛失败：保留产物，只修违反的工程合同并用新目录重跑，不查看部分 endpoint 调门槛。

所有分支的 `formal_success=false`、`phase2_allowed=false`，不产生 ID/OOD、共享 writer、事件到状态学习或长期记忆结论。

## 运行前锁定证据

- config bytes SHA-256：`08ea3d3a5ea41605dc245a891c5c38597148c73f3b4e36c09d2df0b4a0e26ce0`
- config canonical SHA-256：`6844f978e1032c096301e5918e5facbd4331262b2521f81170deb11886be15f3`
- core SHA-256：`061347e55fc0fde2d40367ea7a5a3fbab9c025ec164be0ee20748c6755da332c`
- runner SHA-256：`c0ffe7a82fc396ae5cfcb98b94f16e0ba9344e232a1fe60adcfe293b84253509`
- tests SHA-256：`11566fe9987c96dfff6f762efb605f8e33424f9921c408b1055e1d2b1f5750e0`
- 定向测试：17 passed；联合 D0/identity 回归：103 passed、0 failed/errors/skipped。
- 联合 JUnit SHA-256：`e1fa1d9f2e4b142c27b5311d5221dfdaf3ec66ae88c6742c5840cf2bb66ad8cd`
- Ruff、compileall、JSON 解析和 synthetic preflight/formal 独立交付复算均通过。
- 部署：`vlm-r11-identity-h200x4-20260907`，物理 4×H200；为保持与前序数值口径一致，运行时固定 `CUDA_VISIBLE_DEVICES=0,1`，另外两卡不得进入本轮。

截至本预注册提交，尚未执行任何 reachable-target GPU forward，也没有观察 M0 或训练结果。
