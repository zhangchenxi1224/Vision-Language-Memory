# 45条件组最终端点：大幅改善，但清除仍失败

9628d71 的固定2880步 full-U-Net 训练和完整final已完成。相同开发噪声/问法下，基线为0/900，训练后为 **880/900** 精确答案后立即EOS；**176/180** 张生成图的五种问法全部正确。开发验收失败，尚不能称为可用Writer。

本轮是一道语义问题、三个固定教师状态、45个源状态/操作/事件表达条件组，不能解释为45题或跨实体能力。推理采用已登记的官方Base原生28步、纯高斯初态和CFG1；CFG1是此前实测选择的实验配置，不是声称官方默认7.5。source仅在条件侧，FM覆盖完整单位区间。

| 源状态 | 精确答案及EOS | 说明 |
| --- | --- | --- |
| gray | 180/180 | 三种目标状态的初次写入/清除 |
| ambient | 240/240 | 写入、清除、no-op |
| jazz | 220/240 | 清除到clear失败20条 |
| clear | 240/240 | 写入、清除、no-op |

失败全部来自4张图：jazz→clear的原始事件表达3个噪声、第二种改写1个噪声，每张图五种问法均回答`jazz`。不是额外标点或EOS格式错误。相应两组为5/20与15/20；另外43个条件组全部20/20。

`transition-wording-state-geometry.json` 对全部180个生成张量核对文件SHA、同图五问绑定，并比较三种教师latent。4张失败图都最接近jazz：RMS约0.0196–0.0233，而到clear约0.4193–0.4212。原始回答与距离诊断共同支持“生成仍保留旧jazz状态”；单独的latent距离不作为语义评分或因果证明。后续应继续诊断清除表达与源图条件的交互，并检查真实连续写入是否出现同样问题。

远端严格collector已验证最终checkpoint、两个phase各180张量、2700条原始回答及完整11520次条件/教师/噪声/sigma抽样。每组256次，sigma范围0.000002682209–0.999930083752，其中5717次高于0.5；blank/donor的图像及原始输出跨phase保持一致。本地脚本`verify_transition_endpoint_local.py`已重新验证全部文本和抽样，两边结果一致。大型checkpoint和PT留在项目共享盘，局部数值距离在远端真实计算；本地没有冒称下载这些张量。

- parent result SHA：`3d747a7c5ba97430587aa11d4787e708df7bc2ebfee8430c4a358ee7ea28f8c8`
- checkpoint SHA：`b4251975684314171ae35bfbd1db2e4009b14a39eeb5e5127834824fd6d8cdf1`
- summary SHA：`ed3355a203129562aced3fcfde5eb8ea170569cf376bc67e09dd09a1f1c1e27c`
- 完整文本归档：`transition-wording-endpoint-evidence.tgz`，1639169bytes，SHA `18d4e8f71b123853dffb829ef2daeadeadf17c9915eef1d0a6517960b62b1ee5`
- 全180图距离诊断SHA：`40cbc010b5f51bfa8c587f667b72ba32a645c5a075c5686c2c7f0d98b4c129f5`

新建`dl-transval-h200x1-20260913`已启动固定72单图/96连续写/独立六写30读重放套件，使用显式`--diagnostic`，保留父开发失败。此处只记录启动和真实进程/首批采样证据，尚无完整独立测试结论。旧训练实例`dl-transitions-h200x1-20260913`已在全部工作结束、证据下载及本地复核后stop/delete。
