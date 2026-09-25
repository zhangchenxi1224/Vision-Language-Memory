# C730/I730：共同四源图与配对保持实施

> 2026-09-26用户更新：未启动的I730预算改用于当前学生多步源图刷新R730，C730保留本文件固定初写银行对照。执行顺序以[B_MCQ_RECURSIVE_REFRESH_PLAN_20260926.md](B_MCQ_RECURSIVE_REFRESH_PLAN_20260926.md)为准；不要继续按旧安排启动I730。

2026-09-25。落实既有[全量方案第5节](B_MCQ_FULLSCALE_PLAN_20260925.md)，不增加训练预算。
本次是代码准备与远端暂存，尚未开始源图生成或C/I730训练；必须等B730固定终点完成。
执行代码已冻结为`52d98d85f84e1e4910e922e34f59f5f14bbabf5c`，19:37:59 CST暂存到
`/inspire/ssd/project/exploration-topic/czxs26210936/repos/prefeval-b-deploy-52d98d8`。
六个执行文件的远端字节与Git对象逐一一致，已保存暂存收据。
现有B730/I64继续使用各自旧冻结目录，本次修改不影响运行中的进程。

## 共同源图

同一个B730第23360步父模型，为全部730条偏好生成V0/V1×两噪声，共2920张实际PNG。
V0和V1可以各占一张空闲GPU并行，不能占用仍在执行其他任务的卡。
源图专用噪声命名空间`source-bank:rollout:{pid}:{chain}`；正式评测继续原
`rollout:{pid}:{chain}`。两回复版本使用同一组噪声作配对。
不按Teacher/Reader表现筛源图，不将V2或dev加入源银行。

新增启动lane为`sourcebank-0`与`sourcebank-1`，输出到共同`sourcebank730/V0`和`V1`。
每lane调用现有native Writer rollout，纯高斯、28步、CFG1，输出保存为uint8 PNG。
父模型完成收据和checkpoint hash须一致，不能误用2048快照建立训练银行。

两lane均完整结束后，在装有原torch环境的GPU实例执行以下冻结命令（仅CPU文件核验）：

```bash
"$PY" scripts/experiments/prefeval_k1_source_bank.py \
  --root "$RUN/sourcebank730" \
  --checkpoint "$RUN/robust730/train/checkpoint-final.pt" \
  --variants "$RUN/variants-train.json"
```

这里`RUN`指`/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-b-mcq-20260925`，
`PY`指该项目`envs/vlm-r3-ngc2502/bin/python`，cwd为新冻结代码根。
程序验证完整730条、每条四图、生成端点/父模型/回复版本/hash/噪声，生成`bank.json`。
缺图直接拒绝；错误答案图照常保留。该manifest是C/I共用输入的绑定依据。

## 配对训练

两个lane为`retain730-C`、`retain730-I`，各23360步、effective batch4，初写和保持各半。
相同初始化权重、源图银行、教师、条件缓存、偏好顺序、sigma、噪声与优化器参数。
唯一实验差异：C保持监督教师latent，I保持监督本次输入PNG的官方VAE编码。
两组初写都监督教师latent。

每个偏好每条lane出现64次，`cycle=0..63`：

- 初写：V0/V1交替，各32次。
- 保持源图顺序：V0/seed0、V0/seed1、V1/seed0、V1/seed1，`source=cycle%4`，各16次。
- 无关交换位置：`1+(cycle//4+[0,2,5,7][source])%10`。

此安排使每条偏好的十种交换分别出现6或7次；每张源图均接触全部十种交换，每个组合
出现1或2次。避免同时使用`cycle%4`和`cycle%10`造成源图与交换奇偶位置固定绑定。
两组的实际日志同时记录偏好、初写版本、源图索引和交换位置，可核验完整曝光次数。

730×（2初写条件+4图×10交换）=30660个原生条件。采用共享磁盘缓存并按当前microbatch
读取，完整float32 source/embeds和原mask保持不变，避免两组把全部条件重复驻留内存。
每个条件一次完整原生编码，原子保存，文件锁防止C/I同时编码同一记录。
缓存绑定源图银行、官方版本、输入历史、回复版本、Writer及condition代码；C/I读取同一份。
缓存仅减少重复编码与内存驻留，不把缓存完成当作训练完成或记忆成绩。

官方FM方程、source条件位置、native采样器、PNG重新打开路径和优化器参数没有修改。
已有B730与I64继续原代码；新代码使用单独冻结worktree。中断仍按原manifest/resume.pt恢复。

## 后续读取

每组固定终点后先执行V1的90dev和64pilot完整3+2，再执行730训练全量T1；均为两噪声
真实0/1/5/10链、匹配/错配/灰图/全文对照。64/90的0和10端点另做四位置读取。
本驱动实现这一主要保持矩阵。V0/V2输入表达比较继续由既有B730单写驱动完成，
不能将单写V0/V2的结果写成C/I递归V0/V2结果。最终180与覆盖探针等流程冻结后另行启动。
O1/O2只报告，不据此挑检查点；主全量端点不因dev好坏增减23360步预算。

尚无本阶段新成绩。已有C/I64一步证据支持进入该机制比较；B730若在dev无匹配图收益，
即使C/I训练内容保持成功，也必须保留“训练内容机制结果”的边界。

## 验证与部署顺序

8项检查通过：曝光平衡、全源图×交换覆盖、既有V0/V1隔离与target选择、银行完整性/
父模型绑定/PNG变更拒绝、训练源噪声与正式噪声隔离、共享缓存tensor逐元素/dtype相等。
Python编译、两份shell语法检查通过。19:39原训练实例环境中，两个CLI入口检查及相同8项
测试再次通过（2.39秒），没有加载额外GPU模型。父模型完成且GPU空闲后运行源银行；缓存建立及
前32次更新是实际部署验证，若失败按原因修复后续跑，不更改实验条件或预算。

资源优先级保持：续跑B730/I64；新两卡可用后先恢复C64剩余链/Reader；父模型完成并有
空闲GPU后运行两源银行lane，再冻结银行并启动C/I730。不得把“代码暂存”写成“已训练”。

暂存收据、原生环境检查输出和19:27现场记录见
[实施证据](evidence/b-mcq-ci730-code-20260925)。压缩包SHA256：
`0de9ffdc38c10e925944dfdeb0cc9e752ef07af90b5247d9e2f9817e756b64ca`。
