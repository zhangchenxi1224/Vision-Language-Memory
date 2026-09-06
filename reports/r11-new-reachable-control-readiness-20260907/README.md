# Reachable endpoint control：运行前就绪证据

本目录在任何 reachable-target GPU forward 之前生成。

- 命令：`python -m pytest -q tests/test_r11_new_reachable_control.py tests/test_r11_teacher_swap.py tests/test_r11_new_identity_condition_binding.py tests/test_r11_new_identity_condition_bridge.py --junitxml=reports/r11-new-reachable-control-readiness-20260907/pytest-junit.xml`
- 结果：103 passed，0 failed，0 errors，0 skipped；pytest 输出耗时 11.92 s。
- JUnit SHA-256：`e1fa1d9f2e4b142c27b5311d5221dfdaf3ec66ae88c6742c5840cf2bb66ad8cd`
- 新模块定向测试：17 passed；包含 fixed RNG、精确 LR、门槛反例、fail-closed receipts、teacher/student 信息边界、计数、路径逃逸和 synthetic preflight/formal 全交付审计。

这只证明实现就绪；不证明可达正对照会通过，更不证明 Picture Memory 已成功。
