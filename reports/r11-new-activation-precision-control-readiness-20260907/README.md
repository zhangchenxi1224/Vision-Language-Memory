# Activation precision control：运行前就绪证据

本目录在任何本轮 precision-control DreamLite forward 之前生成。

- 协议：在同一次模型加载中先完成原始 BF16 对照，再把同一组 BF16-valued
  UNet/VAE 权重与固定 conditioning 无损提升到 FP32；模型、数据、target、plateau、
  teacher `x_T`、scheduler 与路径点均不变。
- formal 计数：每条件 14 个 path forward、1 个 plateau-gradient forward、12 个
  signed-gradient scan forward；两条件合计 54 full-chain forward、2 backward、
  0 optimizer step、0 Reader forward。
- preflight 计数：4 full-chain forward、2 backward、0 optimizer/Reader。
- 测试结果：202 passed，0 failed，0 errors，0 skipped；pytest 输出耗时 24.44 s。
- JUnit SHA-256：`ffdcd073aba59f509e82cfaba3fb727b1da7030499c541351d6a3db1bf1781da`。
- Config byte SHA-256：`6fc1caf6e9760c295b25582a804584d6fa326137d48d1b6fcbd6e4146cf89cc5`；canonical SHA-256：`7d11a0f1bcc2b96bc71ba500410b9a60a092a84204584078558f98451f31c1c3`。
- Core SHA-256：`a01d5c6074d99980949dc8d136007361e32448fabac3b8218d5b3b1f6387a743`。
- Runner SHA-256：`5c6e6cb8789230e719f38fa4ae484b2cf092658078ca44ebc68c85e707d7f816`。
- Tests SHA-256：`22b322c6f97f8a7a8568ef2b3e9485f5c5a10a537c02846d2e8710aa3694c95c`。
- Preregistration SHA-256：`d7f171041022e5099fccca7e05153ba933c91fead7bc81c5c27371c57fb2c579`。
- Ruff、Ruff format、Python compile、Git diff check 与 locked-config load 均通过。

测试从上一轮原始归档直接加载并哈希核验 parent target、plateau checkpoint 和
terminal endpoint tensors；覆盖 BF16 复现阳性控制、FP32 capture/gradient 四种预注册
分类、失败闭锁、54/4 次执行计数、两遍路径 bitwise 一致性、全 endpoint/gradient
tensor 重算、artifact inventory 完整性以及损坏指标拒收。

这只证明实验实现与审计器就绪，不预示 FP32 会改善，也不构成 Picture Memory、Reader、
ID/OOD 或 Phase 2 成功。
