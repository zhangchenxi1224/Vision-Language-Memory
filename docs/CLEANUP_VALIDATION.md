# 清理验证记录

日期：2026-10-02。环境：Windows，Python3.13.5，现有本地PyTorch/pytest；未启动GPU训练或远端作业。

## 已完成检查

- 清理边界检查通过：621个撤下文件不存在于候选工作树，恢复索引无重复；注册目录存在，工程证据不属于当前性能来源。
- 当前核心实现、配置、数据锁、K1/多目标/官方PrefEval报告与数据，和四支汇合后的清理前快照保持一致。
- 锁定官方DreamLite源码已检出为 `a6e20c8cc94027f37dd7c5a81b0b3b472aa18409`。正式针对性检查中的官方FM源算术与梯度parity实际运行，没有跳过。
- 针对性检查共119个测试项：首次82通过、35个旧数据合同跳过、2个PNG核验测试因新检出目录无`.cache`失败。补上核验器目录创建后，单独复测这2项均通过。因此本次相关检查最终为**84通过、35退休跳过**。
- 保留的检查覆盖原生条件、官方PrefEval输入、K1的pilot64/train730/dev90/official180、官方MCQ解析、Writer输入隔离、目标采样、source/refresh bank、PNG真实读取与像素篡改拒绝，以及仍适用的通用梯度和边界合同。

35个跳过项只绑定已撤下的旧自定义PrefEval数据，具体函数名见 `alignment-cleanup-manifest.json`，不计作通过。

## 全仓库检查的限制

全仓库收集1051项。探索性全量试跑未完成，不能声称全仓库测试通过。已观察到的两个PNG失败在上述修复与复测中解决；另一个旧R3合同测试 `test_real_manifest_places_report_schemas_at_top_level` 需要本机未安装的`diffusers`发行元数据。

该R3失败已在未清理的原K1工作树复现，错误同为 `importlib.metadata.PackageNotFoundError: diffusers`，并非删除旧报告导致。该原分支的两个PNG测试通过，因为原目录已有缓存。此次没有为旧R3环境安装全局依赖，也没有跳过该测试来宣称全量通过。

未验证外部checkpoint文件的当前可达性、完整GPU数值轨迹或未完成实验的最新状态。历史运行继续绑定原冻结提交；本地目录清理不构成重新训练或重新评分。

## 验证日志

审计目录 `C:/Users/Expedition/codex_work/vlm-main-audit-20261002/` 中保存：

- `aligned-focused-pytest.log`：119项针对性检查及首次两个缓存目录错误。
- `aligned-png-retest.log`：修复后的两项PNG检查通过。
- `pre-cleanup-failures-check.log`：原K1分支的环境失败复核。
- `aligned-collected-tests.log`：1051项全仓库收集。
- `aligned-cleanup-pytest.log`：未完成的探索性全量运行，不作为通过证据。
