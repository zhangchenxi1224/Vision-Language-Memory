# FP32 trust-region：运行前就绪证据

本目录在任何本轮 trust-region DreamLite forward 之前生成。

- 算法：`student_xT = fixed_plateau + residual_delta`；每轮计算单位负梯度，完整评测
  固定半径 `(0.1,.03,.01,.003,.001)`，以 loss 最低且相对改善至少 `1e-6` 的候选
  更新；最多 32 轮，ratio `<=.01` 停止。
- 不使用 torch optimizer、不裁剪梯度、不调用 Reader；teacher `x_T` 不进入更新规则。
- preflight：9 forward、1 backward、5 candidate forward、预期 1 update。
- formal 上限：195 forward、32 backward、32 update；实际计数由 metrics 与保存 tensors
  独立重算。
- 测试结果：216 passed，0 failed，0 errors，0 skipped；pytest 输出耗时 26.70 s。
- JUnit SHA-256：`c6de9a3c89863b6375f8f8dedfffad6e7b35fa5f42f24e67123d1e16bd2b300b`。
- Config byte SHA-256：`f533d376aa9d377ec1a0ed04ad6cffc6977f3e05b9cfa00b6df32dd8502d5d01`；canonical SHA-256：`8d1e7ed000e4016c7cd15c3ca0bc507c144ebfba61c8510a55b23ae2a46e6c11`。
- Core SHA-256：`652561837f8628313eaffbbfd1e1c8771c4843f28ccd21b840918be525b549a2`。
- Runner SHA-256：`fca4349f8a1a1974362dd4a398034d68db2b54e873d256c3650b1e8be8306dae`。
- Tests SHA-256：`315f3324eecd22d099e71b52efaf45f976f7336e0ea6e772b71a41bd6b02e2c8`。
- Preregistration SHA-256：`5b539053e548c770efffd5021d3c969d332b0339968286651d59d559bb70dec0`。
- Ruff、Ruff format、Python compile、Git diff check 与 locked-config load 均通过。

测试从已交付 precision-control raw archive 读取 exact target、plateau、FP32 teacher、
gradient、direction 与 overlap candidates；覆盖固定选择/接受规则、四种结论、单调链、
checkpoint、final replay、动态执行计数、preflight/formal 独立审计和损坏拒收。

这只证明实现就绪；`fixed_target_trust_region_success` 即使通过也不是 Picture Memory、
Reader、ID/OOD、多 target 或多 seed 成功。
