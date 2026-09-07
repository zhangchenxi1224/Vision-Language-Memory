# Oracle terminal-capture：运行前就绪证据

本目录在任何本轮 terminal-capture DreamLite forward 之前生成。

- 协议：固定 parent target 与 `lr=.001 raw256` plateau；17 个预先由 BF16 input code
  边界选定并逐 tensor 哈希锁定的点，按 teacher→plateau 与 plateau→teacher 各重放一次。
- formal 计数：34 scan forward + 1 teacher replay = 35 full-chain forward；0 backward、
  0 optimizer step、0 Reader forward。
- 测试命令：`python -B -m pytest -q tests/test_r11_new_oracle_terminal_capture.py tests/test_r11_new_direction_fidelity.py tests/test_r11_new_reachable_lr_control.py tests/test_r11_new_reachable_basin_control.py tests/test_r11_new_reachable_control.py tests/test_r11_teacher_swap.py tests/test_r11_new_identity_condition_binding.py tests/test_r11_new_identity_condition_bridge.py --junitxml=reports/r11-new-oracle-terminal-capture-readiness-20260907/pytest-junit.xml`
- 结果：173 passed，0 failed，0 errors，0 skipped；pytest 输出耗时 22.07 s。
- JUnit SHA-256：`e7aebb9b72d599e66a00123eae96ea333932464bab127b3e831f1a2480f2831a`。
- Config byte SHA-256：`cbf92ece9c2b64a28a10d0d7c637a05caa208396e9717ccbb88d6671a4094a7f`；canonical SHA-256：`3e0a8b2f912d75fd302d310403f8783fac455495989ba04fadc35fc9d84f0b01`。
- Core SHA-256：`c989719fea723f509bcfb4c1aafb7db4e201c2fcaf3c3e2164e557e0aff66f0a`。
- Runner SHA-256：`c4d2654ec02d3d07b502d281cdbdb0b29000244aa6620e8eecd58a83ffdd91dc`。
- Tests SHA-256：`979847e1c4b32697b2f6a135820df76e3cc9a422d590c41a28c78c64ebe74a9d`。
- Preregistration SHA-256：`28942da6030b2e64b1bcff6d4946a6e8d7edfd17227cdfbc1e4c6c5c800cea9c`。
- Ruff、Python compile、locked-config load 均通过。

测试直接从上一轮 56 MB 原始归档加载 parent target/checkpoint，重建并核验全部 17 个
FP32 candidate tensor；另覆盖两遍 endpoint 一致性、强捕获连续前缀、六种预注册分类、
preflight/formal 独立审计、artifact inventory 完整性及 metrics 损坏 fail-closed。

这只证明协议和审计器就绪；不预示 terminal capture 的宽度，更不构成 Picture Memory
科学成功。
