# Reachable low-LR control：运行前就绪证据

本目录在任何 `lr=0.005/0.001` GPU forward 之前生成。

- 协议：逐字节复用 parent basin target；两个 condition 固定使用同一 alpha=0.99 初始化，只改变 Adam base LR；256 步、cosine 形状、loss、checkpoint 与 gate 不变。
- 测试命令：`python -B -m pytest -q tests/test_r11_new_reachable_lr_control.py tests/test_r11_new_reachable_basin_control.py tests/test_r11_new_reachable_control.py tests/test_r11_teacher_swap.py tests/test_r11_new_identity_condition_binding.py tests/test_r11_new_identity_condition_bridge.py --junitxml=reports/r11-new-reachable-lr-control-readiness-20260907/pytest-junit.xml`
- 结果：135 passed，0 failed，0 errors，0 skipped；pytest 输出耗时 15.87 s。
- JUnit SHA-256：`4a1b5a7a1b4cf4abd89facf72907ca07fb9ef29f6c22ae7ad2cc43971af556c2`。
- Config SHA-256：`bfbb32dfadde2cf65f8399695f3eaaea61c585578af78aab38567761eb0a7b55`。
- Core SHA-256：`6483bc367bba1d952c29f4c21b62b655933349a52059d052fadf722e71ccc1fa`。
- Runner SHA-256：`5ce2fc96a8f149556f1c3d7da9e0e13e06991f93cf34f6f6a42cf5ff37b508a2`。
- Tests SHA-256：`b71f72cffc90e605e4300165c9f7992ce9c0df70de324d7653ccade23a1d56ec`。
- Ruff、Python compile、JSON parse 均通过；提交前继续执行 staged diff check。

合成 preflight/formal 审计覆盖：parent 实际交付归档与 target 字节/四 tensor 哈希、两组完全相同的 alpha=0.99 初始 `xT`/endpoint/M0、逐位相同的 gradient probe、独立空 Adam 初始 state、condition-specific LR schedule、512 行 receipts、10 个 checkpoint、523/512/512/0 计数、三重 gate、四种分类分支、inventory 完整性及 target 损坏 fail-closed。

这只证明实现与审计器就绪；不证明任一低 LR 会通过，更不构成 Picture Memory 科学成功。
