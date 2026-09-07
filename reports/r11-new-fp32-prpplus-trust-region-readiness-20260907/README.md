# FP32 PRP+ trust-region：运行前就绪证据

本目录与预注册文档在任何 PRP+ DreamLite forward 之前生成。

- 父 horizon-128 结果已独立审计并交付：final ratio `0.05279660988009456`，分类 `strong_capture_reached_only`，交付 commit `88dddced0656cdc57d29bb9d10907de708cad76c`。
- 唯一科学变量：第 2 步起的搜索方向从单位负梯度改为确定性 PRP+ 共轭方向；第 1 步按定义与父实验完全一致。
- 固定不变：target、plateau、FP32 lift、模型、loss、五个半径、选择/接受规则、128 步/771 前向预算、`0.01` 成功阈值、Reader 禁用及解释边界。
- 新增审计会逐轮重构 PRP+ beta/raw direction/unit direction，并核验全部候选张量、checkpoint、状态链和最终 replay。
- 测试结果：233 passed，0 failed，0 errors，0 skipped；pytest 输出耗时 28.60 s。
- JUnit SHA-256：`72d7e583e9a7d1246856f5c502cd25b8978f02508de8a94c5aeacedb8ed82a9f`。
- Config byte SHA-256：`7786f9d9175e9c087bba0d2b12bcd08c42acad43356cf1158b91c4fe5aae7312`；canonical SHA-256：`6a3443aadc96e342631559884a988f18d18568bbb8f42c0d4b43c5213f742c35`。
- Core SHA-256：`ae9e4fe848a203c7ed55706036fba1acad9740f112488d03f027166ea3982de1`。
- Runner SHA-256：`c9757318f808ff5c18f880e4c2c12fb872927b1d177b24ef1e9130445b142f20`。
- External auditor SHA-256：`a6e29ae165f864f5122a5620c38550ef627bb72ee97546ddf430a8f4dbcfecfe`。
- Tests SHA-256：`e1cb81a287ff117bb1c54e400fd5f1f104f8c4838594d9fde82383f402a9862b`。
- Ruff、Ruff format、Python compile、配置哈希加载、新算法单测、合成 delivery 独立审计及父实验第 1 步语义审计均通过。

正式运行前必须把本文件、配置、core、runner、auditor 和 tests 一起提交并推送；远端只允许从该精确 commit 的干净 checkout 启动。technical preflight 通过后，formal 必须在不同的新目录从同一 plateau 重新开始。

即使达到 `fixed_target_trust_region_success`，这也只是单 target oracle 优化通过；Picture Memory、Reader、ID/OOD、SET/overwrite、共享 writer 与 Phase 2 均保持 false。
