# Reachable low-LR control：运行前就绪证据

本目录在任何 `lr=0.005/0.001` GPU forward 之前生成。

- 协议：逐字节复用 parent basin target；两个 condition 固定使用同一 alpha=0.99 初始化，只改变 Adam base LR；256 步、cosine 形状、loss、checkpoint 与 gate 不变。
- 测试命令：`python -B -m pytest -q tests/test_r11_new_reachable_lr_control.py tests/test_r11_new_reachable_basin_control.py tests/test_r11_new_reachable_control.py tests/test_r11_teacher_swap.py tests/test_r11_new_identity_condition_binding.py tests/test_r11_new_identity_condition_bridge.py --junitxml=reports/r11-new-reachable-lr-control-readiness-20260907/pytest-junit.xml`
- 结果：135 passed，0 failed，0 errors，0 skipped；pytest 输出耗时 16.10 s。
- JUnit SHA-256：`38b1d1a387a19227dc11e5aa1a9102240fee7053c1d61b940f2f5f6f36a4d0c0`。
- Config SHA-256：`2d87a5f3f669409d7e8e5b45029e5a5b75229f11353049a4a2a07c23b44c13cd`。
- Core SHA-256：`3d8a83548e925f2ac7b2c684c18cc7aceae8c2a08bdad6c357facce0b78eb32c`。
- Runner SHA-256：`03c3419d391dd673156d084ba6eb5f5ff29cf55d29c53fa8f07eb17b45cb059e`。
- Tests SHA-256：`f03cc98cefab3323bd5d5d4bb2ca05a18e3e05ff529e023f0f3de2b390ee4dbd`。
- Ruff、Python compile、JSON parse 均通过；提交前继续执行 staged diff check。

合成 preflight/formal 审计覆盖：parent 实际交付归档与 target 字节/四 tensor 哈希、两组完全相同的 alpha=0.99 初始 `xT`/endpoint/M0、逐位相同的 gradient probe、独立空 Adam 初始 state、condition-specific LR schedule、512 行 receipts、10 个 checkpoint、523/512/512/0 计数、三重 gate、四种分类分支、inventory 完整性及 target 损坏 fail-closed。

这只证明实现与审计器就绪；不证明任一低 LR 会通过，更不构成 Picture Memory 科学成功。
