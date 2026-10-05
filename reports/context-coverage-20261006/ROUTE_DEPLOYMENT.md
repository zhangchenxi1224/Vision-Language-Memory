# PM→FM功能对照已部署

2026-10-06 06:49:58 CST，在本任务原实例 vlm-dreamlite-full-h200x2-20260720 启动独立只读消费者。06:51:54快照中两个worker均存活，实际cmdline分别绑定shard0/1及CUDA_VISIBLE_DEVICES=0/1；两张卡各有计算进程，输出625+615=1,240行。分母、finite KL、错配donor、源图与共同目标绑定检查通过，12份教师张量内容抽查通过。**科学评分仍pending；须等8,640行完整，不发布partial胜负。**

科学协议ROUTE_FUNCTIONAL_PLAN.md保持原有问题、端点和1GPUh预算。两seed各180张上游真实V0 PNG已完整验收，本消费者独立核对360PNG的checkpoint、manifest、原始历史/噪声、PNG哈希/RGB1024。旧Writer全17,808行收据与报告重算相等，提取dev的15,120行；540张旧dev图片及1,080份教师文件哈希已验证并冻结。教师缓存严格只读：不存在或身份变化即失败，没有生成新teacher的代码分支。

新增8,640行与旧dev合并后固定23,760行、90个独立偏好。两个训练seed分别对同seed Direct及B730报告；噪声和问题先在偏好内平均。仍是条件KL诊断，未新增MCQ/free/保持证据，不用这些结果自动替换默认。

执行代码位于 `/inspire/ssd/project/exploration-topic/czxs26210936/repos/context-route-functional-20261006`，冻结commit c45ab75678b46812a5ad7a59bfcbd66eee97bcb9，tree 6aa56846b119d59c28e878eb52acd35367ae6b6a，与开发b353ecf一致且clean。代码以独立Git bundle同步，旧执行树未修改。Windows/Linux各13项相关检查通过，包含完整360PNG契约、同主题donor、缺失缓存不生成、相同教师配对、完整分母和失败预算计数。按实物收据使用manifest文件字节SHA，不混用JSON对象digest。

控制器PID846942，worker849327/849328。nvidia-smi呈现的是另一PID命名空间（271084/271085），容器NSpid不提供host映射；证据保留原始数值，不假称两组PID数值相同。启动前GPU无计算进程、无本任务其他活跃owner，现场无其他训练/评测进程，内存充足；原任务已完成且旧worker退出。另一主线的实例、owner及预留保持原状。只读代码无新训练、无图片生成、无教师生成，因此本阶段不产生训练梯度或checkpoint。

本阶段累计上限1 GPUh，计入原16 GPUh，最多2GPU，每次最多6小时；失败也计费。旧阶段已结算3.424944881399 GPUh，截至06:51:54新增暂计0.047580385076 GPUh，全任务暂计3.472525266475 GPUh，非最终平台账单。attempts为本阶段成本唯一账本，receipts不可再相加。

下次直接检查此控制器与输出，不重复启动。完成后核验两份finished收据、8,640新行和23,760合并行、全360新PNG及540旧PNG、1,080缓存文件/张量/真实EOS与共同前缀，重算comparison，并核实worker退出、owner关闭、成本结算。中断只在证明旧worker停止后恢复同一冻结身份；输入源全部只读。到上限保留partial，不自动扩额度。

证据位于heartbeat-2242-evidence/；完整inputs清单也保存在同名tar.gz中。归档258,621字节，SHA256 ad951ee79093591c3d20d822e63a13242b06289432e7ad96475b6dfe089a5314，下载后独立核对通过。
