# C730 / R730 递归训练实际部署

2026-09-26 01:54 CST核验。执行[递归刷新方案](B_MCQ_RECURSIVE_REFRESH_PLAN_20260926.md)，已完成代码部署并启动两个依赖等待驱动。**C/R尚未开始梯度更新，正在等待共同B730固定终点；没有新增记忆成绩。**

## 资源与实际进程

实例：`prefeval-k1-h200x4-high-20260924`，分布式训练空间。

| 工作 | GPU | 父/实际进程 | 状态 |
|---|---|---|---|
| 原B730 | 0 | 父2009358，训练2009378 | 18,351 / 23,360，97% GPU利用率 |
| C730固定初写源对照 | 1 | 驱动2478528 | 等待B730第23,360步与hash |
| R730当前学生多步源训练 | 2/3生成，2训练 | 驱动2478529 | 同上 |

两驱动于01:52:50 CST启动。核对了 `/proc` 完整命令、cwd、实际GPU和状态文件。GPU1/2/3当前均0 MiB模型占用，等待驱动不加载模型。三个指定空闲GPU由新流程在前置完成后使用；遇到其他GPU任务时等待，不抢占。

`prefeval-k1-read-h200x2-20260924` 原C64及Reader继续原脚本。未使用禁止实例 `dl-clear-retain-h200x4-20260914`，未恢复A或付费Judge。

## 固定代码与入口

新驱动与R训练冻结提交：`55596810cee2062022b894ef93feb1ab4cd6b07b`。

代码目录：`/inspire/ssd/project/exploration-topic/czxs26210936/repos/prefeval-b-refresh-5559681`。

```text
scripts/inspire/run_prefeval_k1_refresh730.py --arm C --gpus 1
scripts/inspire/run_prefeval_k1_refresh730.py --arm R --gpus 2 3
```

共同输出根：`/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-b-mcq-20260925`。

- C输出 `retain730-C`，内部仍调用原 `prefeval-b-deploy-52d98d8` 代码生成2920初写源图并训练/评测，保持原对照行为。
- R输出 `retain730-R`，依次生成四段当前学生源图并各训练5840步。每段7300实际PNG；四段优化器连续继承、逻辑总预算23360。
- `driver-state.json` 记录当前阶段与子进程，`driver-launch.json` 记录启动身份，`*-exit.json` 记录各阶段退出码。
- 原I730尚未启动；其预算已用于R，不要再执行旧 `retain730-I` lane。此调整有用户本轮“推进真实连续状态训练”的授权，已写入自动跟进。

## 验证与证据

本地22项相关检查通过，包括新源图深度/源输出配对、全量数据的训练/评测噪声分离、固定预算曝光覆盖、断点未提交尾部处理及优化器/RNG恢复检查。远端原torch环境入口导入/参数解析通过，并重跑6项新源图及恢复检查通过。

上述属于运行正确性验证，不是记忆能力证据。此次没有为了部署增加实验训练步数，也没有提前用B730的2048快照生成R源图。

完整启动、验证、运行状态与日志收据归档于 [raw.tar.gz](evidence/recursive-refresh-deployment-20260926/raw.tar.gz)，其中：

- `recursive-refresh-deployed-5559681.json`：两驱动实际PID与命令。
- `refresh-deployment-validation-5559681.json`：远端原环境验证输出。
- `refresh-runtime-20260926-0155.json`：B730与两驱动实际命令、cwd、状态及GPU。
- `retain730-{C,R}/driver-{launch,state}.json` 与原始驱动日志。

归档SHA256：`abb5710510e678b77ce2d2da06610200f96ab2b20a4e94f26d3bded58249a617`。

自动跟进已更新为此方案。B730完成后源图→FM→刷新→独立生成评测自动续接，不需要再申请一次启动批准。最终按730/64/90分层、匹配对错配/灰图、同图3+2、0/1/5/10链评估；不凭缓存/loss或训练源图宣布成功。
