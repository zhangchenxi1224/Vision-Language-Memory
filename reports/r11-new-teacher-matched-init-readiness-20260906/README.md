# Teacher-matched initialization 实现就绪审计

2026-09-06：联合 CPU 回归 **260 passed，0 failed，0 errors，0 skipped**；pytest 终态耗时 238.49 秒。静态检查、diff 检查及 trainer/controller/aggregator 的配置路径与双哈希绑定通过。

覆盖初始化的 FP32/BF16 正例与重签篡改负例、sigma/std/MSE/NRMSE/device 绑定、每 checkpoint 当前 xT→h0、完整 Adam 状态与256步日程、主/次诊断决策、父件与原 Phase 1A 门控。完整原始测试明细见 `pytest-junit.xml`；命令、环境和源文件 SHA256 见 `readiness.json`。

H200 纯张量检查表明 CPU/CUDA 算术不保证逐位一致，因此生产复算采用被产物与 manifest 共同绑定的 CUDA 后端。不放宽 tensor hash、torch.equal 或科学门槛；详细观察见 `../r11-new-initialization-backend-audit-20260906.json`。

本审计只是本地实现/单测通过。该提交尚须在目标 H200 的 clean detached checkout 中完成零 optimizer 更新、一次完整四步 forward/backward 的真实预检；之后才可启动固定256步诊断。所有结果仍为 `formal_success=false`、`phase2_allowed=false`；Phase1A仍是6/8，本地绿色测试不等于模型训练成功。
