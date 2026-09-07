# Local direction fidelity：运行前就绪证据

本目录在任何本轮方向扫描 GPU forward 之前生成。

- 协议：逐字节复用低学习率父实验 target、`alpha=0.99` step-0 checkpoint 与
  `lr=0.001` raw step-256 checkpoint；扫描负 autograd、负 Adam 预条件、oracle
  teacher-residual、确定性正交控制四个单位方向。
- 固定网格：2 anchors × 4 directions × 7 L2 radii × 2 signs = 112 scan forwards；
  formal 总计 115 forward、2 backward、0 optimizer step、0 Reader forward。
- 测试命令：`python -B -m pytest -q tests/test_r11_new_direction_fidelity.py tests/test_r11_new_reachable_lr_control.py tests/test_r11_new_reachable_basin_control.py tests/test_r11_new_reachable_control.py tests/test_r11_teacher_swap.py tests/test_r11_new_identity_condition_binding.py tests/test_r11_new_identity_condition_bridge.py --junitxml=reports/r11-new-direction-fidelity-readiness-20260907/pytest-junit.xml`
- 结果：153 passed，0 failed，0 errors，0 skipped；pytest 输出耗时 18.41 s。
- JUnit SHA-256：`728c1a377e863c474e3829934a85733ab707dd786ef4879791120d9f2b304ed0`。
- Config byte SHA-256：`1dd314776247683487df7ba17c1a04d86f079c19c20ce0d01db6a9ed363e0511`；canonical SHA-256：`5e42cd51bd1a35a59b51cbff8dc696539e33023d2dda53240eb47b3b6f243bbd`。
- Core SHA-256：`69c65e339b2879cc3fdcb80e6d5719a9949461a69408773e5b2fd90c6c565929`。
- Runner SHA-256：`800f88395f4ae690b44cec922a37031123b326eb89a8bd2474324b07ceea6ac3`。
- Tests SHA-256：`e0bb5fb4b8595280be869d5e2228df50adf2c5db7a0f1921f14d3f2239a7b58b`。
- Preregistration SHA-256：`7f182d84d93794cce5ebe54d31cf6597b2c1d93109d53ed4fa2d7f88856829f7`。
- Ruff、Python compile、locked-config load 均通过；提交前继续执行 staged diff check。

测试覆盖 parent 实际交付文件与 archive 哈希、固定锚点与 Adam state、单位/正交方向、
完整 112 行网格、有限差分汇总、五种预注册归因、115/2/0/0 计数、inventory 完整性、
endpoint 覆盖及 checkpoint 损坏 fail-closed。

这只证明实现和独立审计器就绪；不证明梯度方向有效，也不构成 Picture Memory 科学成功。
