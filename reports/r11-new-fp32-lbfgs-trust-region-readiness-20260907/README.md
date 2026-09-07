# FP32 L-BFGS trust-region：运行前就绪证据

本目录与预注册文档在任何 L-BFGS DreamLite forward 之前生成。

- 父 PRP+ 结果已独立审计并完整交付：65 attempts / 64 accepted，final ratio `0.02859118701878591`，分类 `prpplus_material_improvement_only`，交付 commit `efc2a9c626cf5846d9d7970cc15d45ea85c8f9ae`。
- 唯一科学变量：首步之后的搜索方向由 PRP+ 改为确定性 L-BFGS；首步仍按定义与父实验完全一致。
- 固定不变：target、plateau、FP32 lift、模型、loss、五个半径、选择/接受规则、128-update / 771-forward 上限、`0.01` 成功阈值、Reader 禁用及解释边界。
- L-BFGS 采用最新 10 个通过相对曲率门槛的 `(s,y)` 对；CPU float64 两环递推；非有限或非下降时清空历史并回退负梯度。
- 独立审计会从保存张量逐轮重建曲率对、历史截断、gamma、rho/alpha/beta、raw/unit direction、全部候选、checkpoint、状态链和最终 replay。
- 直接相关因果链回归：`116 passed`，`0 failed`，`0 skipped`，耗时 `37.53 s`。它覆盖 precision、direction fidelity、terminal capture、trust-region、horizon-128、PRP+、L-BFGS，并包含两步合成 delivery 的曲率/两环独立审计。
- 本机全仓尝试透明保留：在 `210 passed, 1 skipped` 后，因当前本机 Python 缺少锁定依赖 `diffusers==0.39.0` 而停止；这是环境失败，不计作全仓通过，也不计作算法失败。正式启动前须在远端完整锁定环境补跑。
- JUnit（116 项）SHA-256：`4e63f34d92758d2617494d141ba29bb3cd6cfbd0ace088ef04fb677bb3f5e2f6`。
- 本机全仓环境失败摘要 SHA-256：`10b2767548a3566392c0b874e2746c18a40f2a8b7b43619597823644eb3463e7`。
- Config byte SHA-256：`2fb70c6474812b3fac42b740cb0d50762fa9314f79de7d7ef84b0f827769abab`；canonical SHA-256：`35bedc5c5e99abb3b37324097226712915b7681bb279721de00ac3a4ba287542`。
- Core SHA-256：`4375f504e465a2c6c946a6981a07de15e30ce75126be0ae0456ed312b4f76632`。
- Runner SHA-256：`598066882b103fe22681d423c82c72cdc3c3b600cabcd8d9519b18d7115d1246`。
- External auditor SHA-256：`75d178611bd1d027d4124a729c8096b815a91b617f25f6ebb671abc4f44e6398`。
- Tests SHA-256：`c2d8b93fc55cb4cccd90738fb54f631357aa3aabfe3081e7b95941771b678e9e`。
- Preregistration SHA-256：`f111aeaa21a63c3c32d34947580637ad671e5fb06697e5290c497c719abbcca0`。
- Ruff、Ruff format、Python compile、配置哈希加载、确定性两环递推、历史顺序/曲率门槛、分类、runner 策略替换、父实验首步语义比较和合成 delivery 独立审计均通过。

正式运行前必须把本文件、配置、core、runner、auditor 和 tests 一起提交并推送；远端只允许从该精确 commit 的干净 checkout 启动。先在完整依赖环境通过测试，再运行 technical preflight；formal 必须在不同的新目录从同一 plateau 重新开始。

即使达到 `fixed_target_trust_region_success`，这也只是单 target oracle 优化通过；Picture Memory、Reader、ID/OOD、SET/overwrite、共享 writer、多 seed 与 Phase 2 均保持 false。
