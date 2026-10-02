# 2026-10-02 远端主线收敛回执

GitHub 仓库：[zhangchenxi1224/Vision-Language-Memory](https://github.com/zhangchenxi1224/Vision-Language-Memory)。远端状态在 2026-10-02 10:48:28 UTC 核验；以下是实际执行结果。随后提交的文档补充不改变科学实现和实验文件。

## 发布与归档

| 项目 | 结果 |
| --- | --- |
| main | 从 `4b8a3303568fd82e73514932abac8e47c27cf33c` 正常快进至整理提交 `e6fa9c8669823747f32a62ad325b9ae6d4a948c6` |
| 开发分支 | 32 个收敛为 1 个，仅 `main` |
| 旧实验分支 | 31 个均已删除远端 branch 引用；删除前逐项核验同 SHA 的远端归档标签 |
| 新归档标签 | 33 个：31 个分支快照、原 main 快照、四支汇合后的清理前快照 |
| 现有 Release 标签 | 保留 `prefeval-official-ab-teachers-20260924`，指向 `9e2a374a372fb949d563bca2782bf8293333b2cf` |
| 远端标签总数 | 34 个 |
| PR #1 | 已关闭，未合并；独立 Qwen 历史保存在对应归档标签 |
| Release assets | 27 个全部保留，ID、名称、大小和更新时间与操作前一致 |
| 历史改写 | 未执行，main 全程正常快进 |

完整原分支名称、SHA、分类、归档标签及 Release 资产清单见 [remote-archive-manifest-20261002.json](remote-archive-manifest-20261002.json)。该文件记录清理提交发布时的核验状态，文档后续提交会继续推进 main。

归档先通过原子推送发布并读回核验；分支删除采用原子推送和每个引用的精确 SHA lease。任一分支在检查后发生变化都会使整组删除失败。本次 31 个分支删除全部成功。

归档保留了未进入新 main 的 83 个不同提交：77 个旧 DreamLite 探索提交与 6 个独立 Qwen 提交。它们不重新并入当前结果目录。Qwen 路线不因归档被判为“未对齐 DreamLite 的无效实验”。[PR #1](https://github.com/zhangchenxi1224/Vision-Language-Memory/pull/1) 以分支收敛方式关闭，不能解读为其 6 个提交已合并。

## 继续工作与历史恢复

新工作以 `main` 为起点。已有干净的 main 检出可执行：

```bash
git fetch origin --prune --tags
git switch main
git merge --ff-only origin/main
```

已有未提交修改的实验工作树保留原样；不要对其执行 reset 或批量切换 main。`fetch --prune` 只整理远端跟踪引用，不删除本地实验分支。

旧分支 `codex/<name>` 对应标签 `archive/2026-10-02/codex/<name>`。例如恢复 C8 的精确冻结提交：

```bash
git fetch origin tag archive/2026-10-02/codex/prefeval-multitarget-20260927
git worktree add --detach ../vlm-c8-recovery 224cc77d790cf3967b5a56ce2e77c364959435a2
```

恢复操作建议使用新的目录；已有冻结目录不改指。旧报告中“clone 指定分支”的文字属于当时的运行记录，现在按归档标签获取历史，再按报告中的精确 SHA 检出。不得直接用标签 tip 或最新 main 替代记录的运行提交。

| 当前恢复约束 | 保留内容 |
| --- | --- |
| C8 eval730 / round2 | `repos/prefeval-multitarget-round2` 与提交 `224cc77d790cf3967b5a56ce2e77c364959435a2` |
| long10 两 GPU 控制器 | 提交 `6563d16322c792d6ad0007c17cd6be9c6ca279e3` |
| exposure512 两 GPU | `runs/prefeval-b730-exposure512-20260927/code` 与提交 `656fdf029c4a7c05e53bccb75483a03cf62d5f54` |
| K1 / C / R | `K1_CODE_ROOT` 等指定的原冻结工作树 |
| K1 bank / exposure preflight | `git show 5153444:...` 和 `git show 5d1045f:...` 所需历史对象 |

这些依赖是磁盘目录与提交约束。代码审计未发现当前脚本需要从本仓库的待删除分支名称获取代码；外部 DreamLite / PrefEval 来源继续锁定原 SHA。浅克隆可能缺少上述历史对象，历史恢复需要获取足够的完整历史。

## 验证与范围

- 清理边界核验再次通过：621 个旧结果文件退出当前 tree，核心科学实现与当前实验文件未变。
- 针对性测试结果为 84 通过、35 个退休合同跳过，详见 [CLEANUP_VALIDATION.md](CLEANUP_VALIDATION.md)；全仓库测试没有宣称全部通过。
- GitHub 默认分支仍为 main，归档标签指向逐项一致，PR 为 closed 且 merged=false。
- 现有预发布 Release、27 个资产及其标签保留。资产共 7,953,287,944 bytes，这部分不属于 Git pack。
- 未删除本地实验工作树、本地独有提交、服务器共享盘、训练权重或平台实例；没有触发训练或重新评分。

原始执行计划、分阶段回执和验证输出位于操作机 `C:/Users/Expedition/codex_work/vlm-main-audit-20261002/deployment-*.json`。归档标签是同仓库的恢复入口，不是独立灾备；历史瘦身的约束与方案见 [HISTORY_SIZE_ASSESSMENT_20261002.md](HISTORY_SIZE_ASSESSMENT_20261002.md)。
