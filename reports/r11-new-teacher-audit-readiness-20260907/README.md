# R11_new teacher 4↔7 audit：运行前测试证据

本目录在读取任何新的 donor/reset Reader 结果之前生成并提交，属于 D0 预注册就绪证据，不是实验结果。

- 执行时间：2026-09-07（Asia/Shanghai）
- 命令：`python -m pytest -q tests/test_r11_teacher_swap.py tests/test_r11_new_identity_condition_binding.py tests/test_r11_new_identity_condition_bridge.py --junitxml=reports/r11-new-teacher-audit-readiness-20260907/pytest-junit.xml`
- 结果：86 passed，0 failed，0 errors，0 skipped；耗时 11.251 s。
- JUnit：`pytest-junit.xml`
- JUnit SHA-256：`18f392b3a7ac57d58eee80240b383bf02ff0040617c727f7e5261840084ee5fb`

这只证明实现与前序锁定合同通过 CPU 回归；不能证明 teacher 有样本区分力，也不能宣称 Picture Memory 训练成功。
