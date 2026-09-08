# 4×H200 部署记录

更新时间：2026-09-08 15:58（Asia/Shanghai）。

## 当前真实状态

四卡作业已由排队转为 `job_running`，节点 `qb-prod-gpu738`。
实际四张 H200 的环境检查通过，完整 Frozen DreamLite→VAE→Reader 的真实梯度 probe 通过。
正在执行 A1 重复性验证：首条 `A1-t00-gaussian-s00-a1-r0` 已完成256步，
`technical_pass=true`、`qa_pass=true`，controller 实际耗时701.628秒。
第二条重复运行已到208/256步；尚未产生完整 A1 determinism gate，不能提前宣称重复性通过。
158条主体优化中目前1条完成，尚未进入其余多起点、分布和尺度实验。

填空题已按用户指定迁到两卡实例 `vlm-r11-open-h200x2-20260907`：
旧8seed×7checkpoint回放及A/B/C共24条256步训练全部完成，terminal耗时1988.911秒，
保存864条raw generation。A的56个检查点全部逐位复现旧Open。
原题EM为A=3/8、B=8/8、C=8/8；三组均未训练的全新问法EM为0/8、6/8、6/8。
这是单题配对机制实验结果，不是跨问题共享Writer成功。

四卡job原Open/EOS辅助命令已触发输出目录存在检查，在模型加载前以FileExistsError退出，
成功防止重复训练及覆盖；主线按既定逻辑继续。该辅助退出不代表两卡实例上的科学实验失败。

## 资源与代码

- 当前有效作业：`vlm-oracle-geometry-h200x4-20260908-r02`
- Workspace：分布式训练空间；Project：前沿课题探索。
- Compute group：开发区-H200-3号机房-2-cuda13.2版本。
- 单节点4×H200（141 GB/卡），80 CPU，900 GiB RAM，128 GiB共享内存。
- 官方镜像：`ngc-pytorch:25.02-cuda12.8.0-py3`。
- 提交priority=4；平台显示Priority20、NORMAL。公平调度不接受priority10。
- 运行上限72小时；不自动重启失败科学任务，不覆盖未完成轨迹。
- 主线分支：`codex/frozen-oracle-geometry-20260908`。
- 实际部署源码：`3c8cf7d31f8641f6f95d8f58a1f8b844822acdb9`。
- 后续测试提交`79d4706`仅修改测试fixture，未更改实际部署源码。
- 填空题分支：`codex/r11-open-eos-20260908`。
- 填空题固定源码：`e6e6c8071a6c1311a966bda1aa636748fbb016f6`。

```text
主线代码：
/inspire/ssd/project/exploration-topic/czxs26210936/repos/frozen-oracle-geometry-20260908
主线输出：
/inspire/ssd/project/exploration-topic/czxs26210936/runs/frozen-oracle-geometry/3c8cf7d-20260908-r02
填空题代码：
/inspire/ssd/project/exploration-topic/czxs26210936/repos/r11-open-eos-20260908
填空题输出：
/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-open-eos/all-e6e6c80-20260908
```

## 自动执行顺序

1. 核对实际4卡H200、环境、模型/data/source哈希。
2. 两次完整Frozen DreamLite→VAE→Reader前向/反向probe，要求梯度非零且重复一致。
3. GPU0/1：A1两题各3次精确复现；GPU2/3：旧Open回放及配对A/B/C实验。
4. A1失败停止多起点；通过且填空题支线终态后，两个GPU对共同执行主线。
5. 158个独立主体优化run，保存全部257步xT/z、11个完整checkpoint及原始指标。
6. 有成功anchor才追加32个局部重新优化及直接扰动/插值评估；输出分析与科研图。
7. 几何结论经审计后才选择canonical bank与Controller监督；不自动猜测标签或直接Full FT。

## 本地/CPU已经完成的真实验证

- 主线61项相关测试通过，含生产Oracle计算图、梯度冻结、step对齐、9项子进程集成、哈希失败门槛与图形测试。
- 填空题38项相关测试通过；原8seed、初始化、lr=.05、256steps、Reader/VAE及raw32token生成保持配对。
- CPU真实tokenizer审计确认结束符为`<|im_end|>`（151645），生成停止集合为151645/151643。
- 模板末尾换行198不是训练EOS标签；`juice`实际为2token，不能把单词数当token数。
- 旧16条matched输出都先输出正确ambient并最终EOS；错误出在答案后续写。
  这支持检验“立即停止监督不足”的假设，但不构成新EOS方案有效的证据。

## 尚未完成

- 四卡真实preflight、梯度probe及首条训练已通过；A1完整重复性与全量几何结果仍待完成。
- EOS单题原题EM已改善为8/8，原题与固定改写两问均正确也从2/8改善为8/8；
  B/C在全新问法上均6/8，现有证据不支持C优于B，也未证明跨问题泛化。
- Stage B/C入口已实现，但32–64独立事件bank和共享Controller尚未实际训练。
- LoRA/Expanded/Partial/Full FT尚未启动，具体FM训练实现仍需在前置结果支持后推进。
- 填空题P4已完成真实tokenization审计，尚未做跨字段、多问题实际训练。

原首个排队作业`vlm-oracle-geometry-h200x4-20260908`已停止，未进行科学计算；
当前只保留上述r02作为有效四卡申请，未并行重复申请GPU。
