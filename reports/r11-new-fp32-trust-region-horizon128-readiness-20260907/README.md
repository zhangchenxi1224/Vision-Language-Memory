# FP32 trust-region horizon-128：运行前就绪证据

本目录与预注册文档在任何 horizon-128 DreamLite forward 之前生成。

- 唯一科学变量：`maximum_formal_iterations: 32 -> 128`。
- 固定不变：target、plateau、FP32 lift、模型、loss、单位负梯度、五个候选半径、选择/接受规则、成功阈值、Reader 禁用及解释边界。
- 新 formal 必须从同一 plateau 重跑；独立审计要求前 32 条 iteration 与前 160 条 candidate 逐字节复现已交付 parent。
- 最大计数：771 full-chain forward、128 backward、128 update、0 optimizer step、0 Reader forward。
- 测试结果：221 passed，0 failed，0 errors，0 skipped；pytest 输出耗时 27.53 s。
- JUnit SHA-256：`29f8fc3a30879b0c771ec4f7bdd8bb2b1b5583f57cb2abd64adeb36f0d3786c8`。
- Config byte SHA-256：`d687c159644551ea06cb0597ac6b46efbcd1761367e249b30251a6de06931c78`；canonical SHA-256：`542ae95692c3e97c580878cddbd200669d846275ebc09abc7c24762e20835992`。
- Horizon core SHA-256：`ba549628367a9d96b0fb3388e2286cb7d15ecdb9f9c3f5990c13e5d0a0b95766`。
- Runner wrapper SHA-256：`102d7f91c0aa50e40d8d500bb4f2e0df65bc4efc14a146e512af1dafd4093d2d`。
- Parent-prefix auditor SHA-256：`7522450a2f96d73c04401c9433263e443c748a0dff85280254f1078b6c30f151`。
- Tests SHA-256：`4b1f522e8c6f3685b5115f3a704ab968fbe9e2fb3b63ff35ce97e7ea0942a564`。
- 原数值 runner/core 复用且未修改：`fca4349f…06dae` / `65256183…49a2`。
- 预注册文档 SHA-256：`5ae99e1d13f23f73a2a64dbe9b6d205390de5566b1799d31908a94d977b7bc8d`。
- Ruff、Ruff format、Python compile 与配置加载均通过。

新增 facade 只替换哈希锁定的配置加载；全部数值更新、候选评测、checkpoint、重放与基础审计继续调用已交付 parent 实现。外部 auditor 在结果分类被接受前独立绑定 parent 文件哈希并检查 exact prefix。

即使达到 `fixed_target_trust_region_success`，`formal_success` 和 `phase2_allowed` 仍固定为 false，因为这仍是单 target oracle 优化，不是 Picture Memory、Reader、ID/OOD 或 overwrite 成功。
