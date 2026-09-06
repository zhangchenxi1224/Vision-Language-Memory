# Reachable basin control：运行前就绪证据

本目录在任何 `alpha=0.50/0.99` warm-start GPU forward 之前生成。

- 协议：固定复用 parent target artifact；只改变初始化 alpha；optimizer、256 步、loss、checkpoint 与 gate 均不变。
- 测试命令：`python -B -m pytest -q tests/test_r11_new_reachable_basin_control.py tests/test_r11_new_reachable_control.py tests/test_r11_teacher_swap.py tests/test_r11_new_identity_condition_binding.py tests/test_r11_new_identity_condition_bridge.py --junitxml=reports/r11-new-reachable-basin-control-readiness-20260907/pytest-junit.xml`
- 结果：119 passed，0 failed，0 errors，0 skipped；pytest 输出耗时 14.17 s。
- JUnit SHA-256：`ff323e5513b1b375f7c4fa1482a32944a9b304723e334cc94b7930d70e7c1947`。
- Config SHA-256：`4f5bf2d73bc2d7bd10f57f0d4797aef17e9c6f42585ecbe44436ba5368d30cf0`。
- Core SHA-256：`043c2fc01810610c17263facff3582d1b9f229adb0aecafa1a297735c864f2c5`。
- Runner SHA-256：`febabb896e49d4ef0507244f8161e0e2982316a7946f421b2dd74c9da87b239d`。
- Tests SHA-256：`196c7894f1f97f67e847dc28073926a9585dc4b2138f2d3bb8ef4b0f86cf6d0d`。
- Ruff、Python compile、JSON parse 与 `git diff --check` 均通过。

合成 preflight/formal 审计覆盖：固定 target 字节/四 tensor 哈希、parent 实际归档绑定、两个 warm start、独立 optimizer、512 行 receipts、10 个 checkpoint、523/512/512/0 计数、三重 gate、四种分类分支、inventory 完整性及 target 损坏 fail-closed。

这只证明实现与审计器就绪；不证明任一 warm start 会通过，更不构成 Picture Memory 科学成功。
