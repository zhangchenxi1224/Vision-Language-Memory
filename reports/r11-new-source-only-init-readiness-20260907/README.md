# Source-only initialization 实现就绪审计

本轮只改变 initial xT：使用固定 source latent 的 FP32 clone，不向初始化注入 teacher。完整冻结四步 DreamLite、teacher MSE、固定 target1、256 次 Adam 调用、cosine 日程、主要距离与 Reader 门均不变。

这是失败分支中的单因素 bridge 诊断，不是共享 writer 训练，也不是主线 Phase 1A 已通过。Teacher 仍用于优化目标；初始化不依赖答案，不能推出整个求解器不依赖答案。

## 工程校验与修复

本地最终联合回归 **564 passed，0 failed，0 errors，0 skipped**，pytest 终态耗时 230.68 秒。最终命令、环境、源文件 SHA256、JUnit 和独立复核结果见 `readiness.json`；原始逐项测试见 `pytest-junit.xml`。静态检查和 diff 检查均通过。

新配置、core、trainer、controller、独立 aggregator、报告渲染器及测试采用新文件；历史实验代码、配置、日志、结果不改写。新代码的 import/config 双哈希检查通过，未调用模型。生产初始化和 checkpoint 的逐位重建仍使用原始 CUDA 后端；CPU 只用于本地测试与报告的补充归约校验。

启动前交叉审查发现并修复两项 fail-closed 缺口：

1. 控制器不能只相信 preflight 的总通过标志。必须逐项检查全部 11 个硬子门，并从四条原始 ordered logits 独立复算 teacher CE、预测、margin 和四视图一致性；formal 准入重新执行完整 child 校验和终态文件哈希核验。
2. 当前 manifest 与 rows 内部一致仍不足。完整 target 必须与固定 SHA256 的 Phase 1A 父 manifest 逐值相等，防止改动答案后对日志整体重签仍被放行。

这些是运行前的工程修复，不是训练结果，也没有改变科学阈值。前两轮本地联合测试属于修复前快照，单独保留，不能作为最终代码的通过证明。

## 尚未证明的事项

本就绪提交产生前，本轮实际完整模型前向为 0，optimizer updates 为 0。必须先推送提交，再在目标 H200 的全新 clean detached checkout 中完成一次完整四步前向、一次反向、零更新的真实预检；预检有效后才运行固定 256 次诊断。

`formal_success=false`、`phase2_allowed=false`。主线 Phase 1A 仍是 6/8（失败成员 1、7）；不得拼接不同 solver 的成功项。Phase 2/3 未启动，30 小时规划时钟不重置。

实验定义与分支见 [预注册](../r11-new-canonical-latent-bridge-source-only-init-preregistration-20260907.md)。
