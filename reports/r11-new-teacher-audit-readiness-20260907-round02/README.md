# R11_new teacher 4↔7 audit：Round 02 工程修复就绪证据

Round 01 因 Reader forward 计数钩子挂错模块而技术终止。本目录在修改任何实验配置/门槛以及读取 donor/reset 数值之前生成；它只验证计数修复和前序合同，不是 D0 结果。

- 唯一数值无关修复：把 forward hook 从未执行的 `reader` 外层移到评分函数实际调用的 `reader.model`，并在守门前落盘 observed/expected counters。
- config bytes SHA-256（不变）：`3d11e602d8ac70cc3716ad375db92fe7c83fc8ba2c13d692d3e063d3f477ef1f`
- core SHA-256：`5bbfb32af5b215024035ebd728a85d4750b1b662242165ed3b95ed591d1a361c`
- runner SHA-256：`acdce1782292dc727489f1a4469cbabf678d7d3f315214749e597d232cac823f`
- tests SHA-256：`faf3ae727074ba971c5ce64dd25b71936e64ad979d0272ae55f637ea2be00c0f`
- 测试命令：`python -m pytest -q tests/test_r11_teacher_swap.py tests/test_r11_new_identity_condition_binding.py tests/test_r11_new_identity_condition_bridge.py --junitxml=reports/r11-new-teacher-audit-readiness-20260907-round02/pytest-junit.xml`
- 结果：86 passed，0 failed，0 errors，0 skipped；pytest 输出耗时 11.35 s。
- JUnit SHA-256：`e98da606f597554ce6eb3ce9e1147d521545d859fa393c7fa949f55b4576d1e1`

科学状态仍为 false；Round 02 必须使用新输出目录并再次经过独立 audit。
