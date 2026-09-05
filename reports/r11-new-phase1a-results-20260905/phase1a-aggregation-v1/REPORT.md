# R11_new Phase 1A 八目标独立复算报告

> 本报告由 controller roots、原始 256-step receipts、16-cell endpoint rows 与固定 checkpoint hashes 独立复算；不以 trainer/controller summary 作为结果来源。

## 三层结论

| 层级 | 结果 | 含义 |
| --- | --- | --- |
| 工程通过 | 通过 | 8 个 target 均满足 exact receipts、四步 DreamLite、冻结/梯度、固定 checkpoint、snapshot 与 artifact hash 契约。 |
| 诊断通过 | 未通过（6/8） | 仅表示 query-level Frozen-DreamLite endpoint reachability。 |
| 科学成功 | `false` | 未证明 state-level memory，未训练或验证共享 writer，也未做 held-out、多 seed、rollout 和完整因果确认。 |

- 决策：`run_canonical_r11_latent_bridge_distance_diagnostic`
- 原因：The preregistered 8/8 Phase 1A arm gate was not met despite technically valid runs. Next run the minimal bridge-distance diagnostic against the known readable canonical-R11 latent; do not lower the gate or infer mathematical non-reachability.
- 训练 commit：`2cde77ece6f020ab8c747d7c73e19dac4d8fba1b`
- 固定 endpoint：`raw_x_T_step256`
- Phase 1B：本 MVP 路线中不作为 Phase 2 的前置门，仅保留为未来 state-level confirmation。
- `formal_success = false`；即使 8/8，也不能表述为 R11_new 或 Picture Memory 训练成功。

## 逐目标原始 endpoint 复算

| Target | Gate | M0 normal CE | Endpoint normal CE | Relative change | Improved views | Accuracy delta | Normal/reset DiD |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | PASS | 15.283861 | 0.779338 | -94.90% | 4/4 | +0.750 | -14.504523 |
| 1 | FAIL | 27.921883 | 6.691034 | -76.04% | 4/4 | +0.000 | -21.230850 |
| 2 | PASS | 30.807291 | 0.933170 | -96.97% | 4/4 | +0.250 | -29.874121 |
| 3 | PASS | 35.333333 | 0.032681 | -99.91% | 4/4 | +1.000 | -35.300652 |
| 4 | PASS | 33.348958 | 0.579022 | -98.26% | 4/4 | +0.750 | -32.769935 |
| 5 | PASS | 17.406250 | 1.898205 | -89.09% | 4/4 | +0.250 | -15.508044 |
| 6 | PASS | 13.312504 | 0.407325 | -96.94% | 4/4 | +0.750 | -12.905179 |
| 7 | FAIL | 35.604166 | 7.600396 | -78.65% | 4/4 | +0.000 | -28.003770 |

## 解释边界

Phase 1A 优化的是每个 query 各自的 initial latent `x_T`。它不证明一个共同 endpoint 支持同状态多 query，也不证明 query-free shared writer 已学会。若 8/8 通过，下一步只允许构建现有 train split 的 query-level Phase 2 MVP oracle bank；后续 Phase 3 仍只是共享 writer 可学习性诊断，必须继续保留固定门槛和因果评测。
