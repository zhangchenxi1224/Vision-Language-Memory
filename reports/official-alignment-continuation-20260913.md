# Goal 接续位置

## 10:38 更广16题训练数据已真实构建；当前四卡优化超过1900步

- 本轮开始实测训练四rank181990–181993、suite470773均存活，1144步；后续实测1684和**1906/2880、elapsed1523.86s**，baseline845/900、final仍0。仍属固定046训练，不能修改/重启；suite仍等待完成。未取得新端点，goal active。上轮归档补齐已commit/push **8f81c1f8f849225bdfca592afdaa4cb6ebf996a7**；所有旧下载/verify会话都结束。
- 新 **7ee3a927d0abc7ead642fdc3e9795498efa20171** 添加显式 `historical-r11-five-prompts/v1`。现有默认三行格式不变，历史原问/改写按各自固定suffix完整保留，拒绝额外问题行及任意suffix；没有把历史问句改写成更容易版本。新增bank构建器保留16目标×4原始step256/B端点，全部64均由FP32和RGB原问+EOS正例资格支持；五问法只有57/64通过的失败原样保留，不按改写表现过滤七个端点。28相关tests实际通过。
- Builder已部署 **P/repos/dreamlite-historical-writer-bank-20260913**，CPU隐藏CUDA在用户实例真实执行完成（78691已exit0），无GPU初始化/零optimizer。实际输出 **P/runs/.../7ee3a92-historical-writer-bank/manifest.json** SHA **d54895adb15b91c3befd52c57f6248c76cf2abadb44916af830726a0736e688c**，64新rawFP32latent副本、16groups，load_teacher_bank真实通过。此处不是新Writer训练，不是GPUruntime确认，不要声称16题已学会。
- Manifest和complete已完整下载（87500/93047已exit0），本地 `verify_historical_writer_bank_local.py` 已实际通过全16事件/问句字节、64teacher/source memberships、128original positives、720raw失败保留及固定donor选择。64新latent副本仍远端，本地遗漏明示。报告 **official-historical-writer-bank-20260913.md**、results **historical-writer-bank-{manifest,complete,local-verification}.json**；新证据/本地verifier待本轮提交。没有待传输会话。
- 这份新bank用于后续更广seen-question共享Writer：每组灰源+完整1/2事件有序prefix、4原始教师，全部问法已经用于历史资格检查，不能冒充freshholdout。当前没有启动更广training或锁定其预算/初始化。先完成当前固定warm端点和独立链验收，再据结果安排四卡后续；仍须更广Writer功能证据，单实体通过不足以宣布goal完成。

## 10:25 证据归档补齐，四卡稳定优化

- 上节独立single/chain原始归档、四卡预检/首步证据及review均已提交push **fc5148904672fc9910ab8e783d4b0e26fb3dc1be**，不是待提交。训练仍固定046、validation1201、套件ff17，不要fetch/修改正在执行的checkout。
- 历史readback归档61321已exit0、61,365,808bytes，完整SHA **b002f07d9ab7ca37f7dadebf8282a0ac0fb312f8a38d37b21bcfbe04b5ddc5c5**。新增 `scripts/reporting/verify_historical_readback_local.py` 已实际执行成功（83387已exit0），全部65PNG/720raw/原始面板和覆盖重算通过；64PT仍远端，缺失清单明确。完整归档、local-verification与工具待本轮提交；没有待下载会话。下节“61321仍在下载”已过时。
- 10:23附近最新实际 **847/2880更新、679.26秒**，baseline845/900，final0；训练和后续套件PIDs都实测存活。套件470773的stage仍waiting_for_fixed_endpoint。这不是阻塞或新结论；保持原fixed预算，完成后自动严格collector及fresh independent验证。
- 用户实例生命周期已实查：本轮2026-09-13 09:17:55启动，running；平台CLI lifecycle没有给停止期限。当前代码自己有限截止（训练12:00、后续12:30），没有擅自更改用户实例生命周期。后续需要更长实验先检查资源实际仍可用；不得stop/delete用户实例。

## 10:16 用户四H200已实际训练，等待固定终点与自动独立验证

- **优先保留用户实例 vlm-r11-trust-h200x4-20260907-r3，不得stop/delete。** 四H200/80CPU/900GiB/shm128、ngc25.02 CUDA12.8、nodeqb-prod-gpu2104，四卡实际空闲核验后迁入。旧dl-warm-h200x1和dl-transval-h200x1均已stop，未delete；不要再轮询其旧worker。CPU2仍运行，约15:46截止。
- 四卡训练固定commit **046c1f1d1c398dbd08578d7c4ba6814343fea0d5**，checkout **P/repos/dreamlite-four-gpu-20260913**，run **P/runs/dreamlite-official-alignment/046c1f1-transition-warm2880-h200x4-seed20260914**。driverPID180440，GPU ranks181990/181991/181992/181993；启动一次，不要重复launch。训练deadline **1789272000=12:00CST**。真实梯度预检通过(relativeL2 3.2334e-08)，初始化及首步四卡参数hash一致；完整baseline1350raw/matched845/900/169of180。最新实际326/2880、elapsed264.47s、final尚无。状态辅助 **P/runs/.../four_gpu_status.py** 可用CPU2的python3只读执行，减少日志输出。
- 旧warm run **d9a1a11-transition-warm2880-seed20260914** 在baseline期间SIGTERM后pid16891/18084真实退出，terminalpaused/optimizer_steps0/checkpointnull，SHA **ff4b8059e7c8c763a9792e47139444a783045ef02d0bcc60e73576e2d515ff2a**。没有新参数可迁；旧输出完整保留。新run的migration-amendment绑定原warm计划SHAe4781068...、父b425checkpoint及4bf556package。原45组、freshAdam、seed20260914、额外2880步及新validation计划不变；全局4draw分给4rank，不能声称串行FP32轨迹bitwise。
- **自动后续套件已启动一次，PID470773**，source **ff17a5a7a68f0525a80ea1827beff5073a684640** checkout **P/repos/dreamlite-four-gpu-completion-20260913**。锁/状态/日志/PID都在 **P/runs/.../four-gpu-completion-suite{.lock,-status.json,.log,.pid}**。deadline **1789273800=12:30CST**。当前等待parent terminalcompleted及GPU实际释放。验证/collector/package代码固定 **1201efe53f0a8886ff854d03a42432f84bb7fb07**，checkout **P/repos/dreamlite-four-gpu-validation-20260913**，已部署；修改本地新代码不会影响正在训练的046checkout。
- 套件顺序：collect fixed warm endpoint→GPU0 single_writes/GPU1 RGBchains并行（都原nativebatch1、完整矩阵）→各自严格collector与96PT审计→新package→固定第一链六写30读独立CLI parity。固定输出 **1201efe-four-gpu-warm-{confirmation,chains,package,parity,inference}**；摘要/归档prefix **four-gpu-warm-{endpoint,confirmation,chains}**。失败development保留diagnostic标签，不能改pass标准。需要45min剩余资源才派发完整验证。尚无这些新最终产物。
- 上一9628端点独立suite已全部完成，旧0f single **360/360**、72图，390raw含30controls；RGBchain **430/480**、86/96图、**11/16整链**，五次clear失败（4jazz、1ambient）及其followingnoop共50raw错。new281bpackage实际六写30读parity true、reference functional false。完整single65,696,825byte/018647ba...、chain87,279,711byte/c7bbda0f...已下载SHA相同且本地verifier实际通过72/96PNG和全部390/480raw，PT/checkpoint缺失明确。下载会话77591失败后，tail82161已完成拼接；不存在待传的旧chainarchive。证据与说明待本轮提交。
- 更广16题历史回读 **5bcf6d0-historical-fp32-readback-20260913** 已完成720raw；两图像形式各312/320、57/64成员十读全部正确、80blank全错。**c61c283b2f19a4754bb05b5eb43fb1485c86eb6d** collector在旧GPU隐藏CUDA真实核验全部64源latent/PT/PNG量化+720raw后完成，source **P/repos/dreamlite-historical-readback-collection-20260913**。summary **3ea31c28bcf985c914af17bc5bd11c182feb7ebf06365fd4a9d9448cf5f630eb**、complete **091eb3d68eba1553ff7f34dab7a620e044003443ac811064ccd3730dee890a0e** 已下载并SHA匹配。archive **61,365,808bytes**, SHA **b002f07d9ab7ca37f7dadebf8282a0ac0fb312f8a38d37b21bcfbe04b5ddc5c5**，**SCP会话61321仍在下载，不能提交partial**。远端prefix historical-fp32-readback。这里只是直接oracle兼容，不是共享Writer。
- 相关验证：本地四进程Gloo真实reduce+Adam+checkpoint recovery通过；相关66passed+1缺diffusers本地环境项，远端包含该项和多进程测试的19passed已实际补齐；新warm collector/probe/parity相关13passed。完成goal仍需要可用功能证据，当前active无阻塞。

## 08:46 接续：完整880/900失败已核验，新GPU独立诊断真实运行

- **原9628d71训练/评测/collector均真实完成，旧GPU dl-transitions-h200x1-20260913已stop/delete。不要再poll旧6087/463114或旧实例。** 完整1350final，matched880/900、176/180图五问法全部正确；baseline0/900；20错全是4张jazz→clear图回答jazz，wording0三噪声、wording2一噪声。父terminal为completed仅表示工作结束，development gate为false。
- final result SHA **3d747a7c5ba97430587aa11d4787e708df7bc2ebfee8430c4a358ee7ea28f8c8**；checkpoint SHA **b4251975684314171ae35bfbd1db2e4009b14a39eeb5e5127834824fd6d8cdf1**。collector已实际核验360PT、2700raw、11520draw、每组256次及不变controls；sigma min2.682209014892578e-06/max0.9999300837516785/高于半区5717次。
- 本地已完整下载summary、archive、geometry、collector-status，所有SCP已exit0无待传输。summary SHA **ed3355a203129562aced3fcfde5eb8ea170569cf376bc67e09dd09a1f1c1e27c**；archive **1639169bytes** SHA **18d4e8f71b123853dffb829ef2daeadeadf17c9915eef1d0a6517960b62b1ee5**；geometry SHA **40cbc010b5f51bfa8c587f667b72ba32a645c5a075c5686c2c7f0d98b4c129f5**。均在results中以transition-wording-endpoint-*或transition-wording-state-geometry.json命名。不要重复下载。
- 新 `scripts/reporting/verify_transition_endpoint_local.py` 已真实运行成功，会话79366已exit0：解包临时目录，独立重放全部2700raw/11520draw，生成transition-wording-endpoint-local-verification.json；明确361个远端PT/checkpoint缺失，本地没有数值重算大张量。全文分析见results/transition-wording-endpoint-review.md。新证据/本地工具/报告待本轮commit。
- 新 **cb60fc2** 的`scripts/reporting/transition_state_geometry.py`已push并部署到 **P/repos/dreamlite-transition-geometry-20260913**，fetch会话63875已exit0。实际在旧GPU上的CPU隐藏CUDA执行成功，会话40627已exit0；核验180张量/3教师/900raw，4失败图全部最近jazz(RMS0.0196–0.0233)，距clear约0.4193–0.4212。这是生成旧状态的诊断依据，不把距离当语义验收或因果证明。全部输入大张量留共享盘。旧GPU在以上完成、本地核验通过、确认nvidia-smi无进程且旧PIDs消失后才释放。
- **唯一当前owned GPU：dl-transval-h200x1-20260913，RUNNING**，新建08:32:49，开发区-H200-3号机房-2-cuda12.8版本，ngc-pytorch:25.02-cuda12.8.0-py3，1H200/20CPU/200GiB/shm64，节点qb-prod-gpu2136，240min约12:33到期；suite deadline **1789272000=12:00CST**。CLI不允许H200 SSH/rtunnel，直接exec --workspace 分布式训练空间加外层tty:true已验证；无IAB。CPU仍dl-align-cpu2-20260913。
- 已实际执行一次 **P/runs/.../launch-transition-validation.sh**（本地.cache同名，上传会话已结束），调用b9eec7c4 checkout suite和 **--diagnostic**。**suite PID32512，真正GPU worker PID33032**，08:43ps都live，日志已出现ambient_original_training_event-seed-06及真实28步采样。当前stage single_writes/state running，不是只建实例或只写计划。不能重复launch或套件重启已有输出。
- 状态 **P/runs/.../transition-validation-suite-status.json**，PID文件同前缀.pid，总日志同前缀.log；stage日志 **0f40767-transition-confirmation.log**。已完成的端点summary是launcher前置。输出single **0f40767-transition-confirmation**，chain **0f40767-transition-chains**，后续package/parity/inference均281b653-transition-*；固定b9suite串行运行collectors、96PT审计、新端点导出和独立六写30读重放。完整独立结果尚无，开发失败必须仍保留。下步检查同一live33032/32512、完成stage后下载核验，依实际失败调整下一训练，不能因局部正确或queue exit0宣称可用。
- 更广16题FP32/RGB720回读probe仍 **未GPU派发**，已部署5bcf6d0 checkout和17f00e92面板，不在当前suite中；当前独立诊断完成后用剩余idle资源继续。没有新的训练优化进程。goal仍active，无阻塞。

## 08:04 接续：2880步更新完成，完整final评测已启动

- 实际GPU观察 **2880/2880 optimizer updates**，elapsed5517.603s。`train/checkpoint-final.pt`已经存在，4681006404bytes；目前只是存在/大小核验，最终SHA及载荷绑定仍等待completed后严格collector。08:03最新 **final_generation_rows58/1350**，parent terminal仍不存在。
- `ps`实测训练PID6087状态Rl/存活，collector463114状态S/存活；collector仍waiting_for_training。不是训练停止，也不是完整实验完成；不能重启、换checkpoint或提前按部分raw公布功能结论。训练run仍9628d71，当前GPU仍dl-transitions-h200x1-20260913，worker/collector/实例截止仍09:15/09:20/约09:49。
- 新可执行串行launcher **scripts/inspire/run_transition_validation_suite.sh** 已提交push **b9eec7c4da0f3547618813312d6537e0e13713e2**。CPU2 fetch/worktree会话1969已exit0；独立checkout **P/repos/dreamlite-validation-suite-20260913** 已实测相同HEAD和远端bash -n通过。尚未执行suite，未创建新验证GPU。
- Suite须在真正completed父端点之后、**新空闲H200且实测lease至少剩90min** 执行：`bash P/repos/dreamlite-validation-suite-20260913/scripts/inspire/run_transition_validation_suite.sh DEADLINE [--diagnostic]`。development若失败必须显式diagnostic且原失败不变。脚本绑定已部署probe0f40767、collector3c335a2、package/parity281b653，拒绝dirty source/正在占GPU/已存在输出。
- 顺序为72单图+collector、96RGB写链+collector+张量验证、**新2880端点**package导出、第一预注册六步准备、独立CLI六写30读、parity检查。所有阶段单独日志，状态 **P/runs/.../transition-validation-suite-status.json**（尚不存在）；任一工程错误停止并保留阶段。固定run输出0f40767-transition-confirmation/chains、281b653-transition-writer-package/package-parity/package-inference，归档prefix transition-confirmation和transition-chains。输出重复时拒绝，不能盲目重启suite。CLI最后阶段有deadline timeout。完成这些工作仍须原始功能结果，不能仅按queue exit0宣称可用。
- CPU连接只能用 **dl-align-cpu2-20260913**，旧CPU已删除；新实例约15:46到期，具体实时status。所有查询、部署会话已结束，无待SCP/fetch。本轮此前2484→2880为真实连续观察等待，最终58条证明已进入final。更广16题FP32/RGB720回读尚未GPU运行，独立于本suite，后续仍需安排。

## 07:49 接续：CPU连接实例已更换，训练继续

- **后续CPU exec/scp/fetch统一使用 `dl-align-cpu2-20260913`**。新实例已创建并实测RUNNING，CPU资源空间 / 前沿课题探索 / CPU资源-2 / 0,2,8 / ubuntu-inspire-base:22.04 / shm32 / 480min，节点cpu-nat-417。07:47状态剩7h58m，约15:46到期；以后以实时status为准。
- `connection refresh`会话38429已exit0；新连接实际读取共享bank文件220499bytes和result-verification checkout HEAD3c335a284f430d924ff27178341dd1de547e12b3。通过新CPU完整下载historical-fp32-payload-verification.json到本地.cache/cpu2-transfer-check.json，SHA212fc526c29302b71a4dfef7657afb072d4f66983a884d349b0159d366c541db与封存结果一致。所有相关查询与传输均已结束。
- 旧 `dl-align-cpu-20260913` 已确认没有实验/传输工作进程，仅平台服务；新连接验证后已成功stop并delete。不要再向旧名称exec/scp。GPU `dl-transitions-h200x1-20260913` 及训练PID6087、collector463114完全保持原样。
- 最新07:47训练实测 **2423/2880**、elapsed4643.94s，父terminal不存在、final raw0、collector waiting_for_training。仍在进行真实优化，不能重启或选择中途checkpoint；继续等待2880完整final后收集和原定独立验证。
- 下文07:38的“待本轮提交”已完成于 **2647bb5** 并push；64历史载荷检查不是待提交/待重跑项目。当前没有新GPU验证结果，goal仍active。

## 07:38 接续：多问题回读probe已部署，64载荷真实CPU验证通过

- 最新功能commit **5bcf6d0990e6e428c536ff21d58c0b4af69ce66b** 已push。新增scripts/probes/historical_fp32_readback.py，固定panel SHA17f00e92...，全64端点×FP32/RGB两形式×五问法640matched+80blank；零optimizer/零Writer，FP32 VAE/bf16 Reader，真实raw32-token/EOS，完整PT/PNG/hash和冻结审计。要求idle GPU且deadline至少剩40min，不能抢当前训练GPU。两新tests通过，与panel两项本次4 passed，未改累计85项；不是功能回读结果。
- CPU fetch/worktree session23826已经exit0，**P/repos/dreamlite-historical-fp32-20260913** 固定5bcf6d0，已部署。**尚未执行GPU回读，尚未派发输出/deadline/实例。** CLI需要--panel/--base-model/--base-seal/--mobile-model/--reader-model/--output/--expected-commit/--deadline-unix；模型路径仍P/M公共固定路径。独立当前训练final/probe及实际package CUDA parity仍须先完成，面板不是新teacher bank。
- public probe中的load_historical_latent已通过真实CPU全量载荷检查。执行本地.cache/check-historical-payloads.py的上传副本P/runs/.../check-historical-payloads.py，import远端新probe副本P/runs/.../historical_fp32_readback.py（SHA b8a4d80593f89bf8953e72a33f7c55659b9f0193260ca6bacaea6153fbc2a953），PYTHONPATH使用3c335a2 result-verification checkout及src，CUDA空、OMP/MKL1。64个step256/B端点全部weights_only加载成功，4194304值、每个FP32[1,4,128,128]、文件/张量SHA及target/seed/arm绑定均通过；CUDA未初始化、VAE/Reader调用0。
- 输出 **P/runs/dreamlite-official-alignment/historical-fp32-payload-verification.json** 已下载，SHA **212fc526c29302b71a4dfef7657afb072d4f66983a884d349b0159d366c541db** 本地远端一致。真实CPU校验工具exec已exit0；所有上传（92257等）、下载、fetch和查询会话均已结束。新结果和说明待本轮提交。
- 最新07:36实测训练 **2084/2880更新、elapsed3995.5s**，PID6087与collector463114存活，parent无terminal、final rows0、collector等待。当前任务仍有效进展，无阻塞；不要重启或选择中途checkpoint。下一步等完整fixed final，自动collector会核验并打包，然后按原预注册新72图/96写链及真实package parity继续；更广范围teacher兼容性也不能替代共享Writer验证。

## 07:28 接续：64个历史端点的FP32/RGB回读面板已真实封存

- 上轮55e8b2f（真实CPUpackage加载证据）已提交push。当前最新功能 **acc0b68** 已push，新增scripts/experiments/prepare_historical_fp32_readback.py和两项测试（2 passed，累计未改覆盖83）；不是新Writer训练代码。**历史16题审计本来已经完成**，见historical-multiquestion-review.md和audit.json，不要重复重跑它。之前摘要只说检查root并不完整，以当前仓库审计为准：128run/32768updates/3456raw，B原问64/64、第一改写64/64、第二61/64，旧BF16 VAE，非共享Writer。
- 已核验旧lane-0/runtime_attempts/attempt-000/manifest.json，其Qwen Reader revision和manifest/payload SHA与当前一致，Mobile VAE也同已封存快照；软件仍torch2.7.0a0/diffusers0.39.0/transformers4.57.3。不是擅自更换Reader的计划。旧panel原始SHA d356238fd5c267812dcf28d214ab062fd43388bb6b53b78602f0c1e8f5b36672，audit SHA c1a9706a7eb3fa2ee9d3e25486f07ac738e45017d032cbffd96ee616ec21a460。
- 新prepare脚本副本P/runs/dreamlite-official-alignment/prepare_historical_fp32_readback.py已在CPU真实执行成功，读取3c335a2 checkout的旧panel、已封存audit和原paired-2c0e41c campaign。64个B端点的inventory/manifest/checkpoint-index/step256文件SHA全部核验，无按成功替换。输出 **P/runs/dreamlite-official-alignment/historical-fp32-readback-panel.json**，SHA **17f00e924ea8a2e66999fd63fb45db8848be8f1c91083155b4a21f9d011b45b5**。已下载同名results文件并本地验证SHA、16×4完整矩阵及event-only字段。所有相关上传/下载/查询已exit0（96661、24631、17499等已结束），没有待传输会话。
- 面板覆盖color4/drink4/music2/material4/meal2，共16实体/64旧oracle。计划FP32-decoded和RGB uint8两形式×每端点5问法=640matched，加统一gray128 RGB空白80control。三个历史问法原样保留，两新问法已固定。event_stream同时保留event和mixed的更新/clear文本，剥离query/choices/scorer，避免漏掉mixed中的更新。**这里只prepare了面板；实际FP32/RGB Reader探针尚需实现并在空闲资源上执行，未创建新teacher bank，也不是16题Writer成功。** 详见reports/official-historical-fp32-readback-plan-20260913.md。
- 最新07:25实测训练 **1743/2880、elapsed3340.1s**，PID6087及collector463114存活，final rows0，parent无terminal，collector waiting_for_training。当前训练仍9628d71固定运行，继续等待final及后续原定独立验证；新多题兼容性工作不改变当前预算或验收条件。只读短观察器P/runs/.../observe-transition.py可继续用。

## 07:16 接续：真实独立CPU加载通过，训练1425步

- 本轮前段是verified wait：实际PID6087/463114持续存活，1025→1118→1168→1183→1231→1269。复查官方源码与实际Base config，dropout=0且官方模型源码未找到BatchNorm/self.training分支；未发现要中断训练的新偏差，也未更改训练源码。新增**只读临时观察器**本地.cache/observe-transition.py、远端P/runs/dreamlite-official-alignment/observe-transition.py，直接在GPU实例用python3运行，可输出简洁live/step/final-row/collector状态；非训练进程，不能把观察超时当任务结束。
- 最新功能commit **4c2f0fd** 已push，新增scripts/probes/load_rgb_package_cpu.py。副本已上传P/runs/.../load_rgb_package_cpu.py，PYTHONPATH指向已部署 **P/repos/dreamlite-result-verification-20260913/src**（3c335a2 loader与a333cf8相同）。实际CPU探针exec session99081已exit0，CUDA_VISIBLE_DEVICES空、OMP/MKL1，未占训练GPU。输入仅旧a333cf8-export-engineering-full1536 package、Base快照和官方source，不传bank/parent checkpoint。
- 实际官方pipeline六组件加载成功，U-Net1075 tensors/389968388值与导出逐位一致；VAE2445063参数、text_encoder2127532032参数，三模块全部CPU/FP32/frozen eval。OfficialRGBMemory接受并构造RGB1024初态，CUDA未初始化，耗时31.6337s。**Writer调用0、Reader调用0；不是native采样或新模型功能成功，旧c90896c/CFG7.5失败结论保持。**
- 原始结果P/runs/.../**rgb-writer-real-cpu-loading.json**已下载，SHA **2a3055f5a4f0787382ea3cf0be38c19beab5a801fbd6bfbadc658b2445cd4ec1** 本地远端一致；下载session16315已exit0。新结果和推理文档待本轮提交。全部上传/观察/CPU探针均结束，无待传输会话。
- 最新07:15只读实测 **1425/2880更新、elapsed2734.65s**，训练PID6087和collector等待器463114存活；parent_terminal不存在、final_generation_rows0、collector waiting_for_training。不要把训练接近一半误作完成。GPU实例/worker/collector截止仍09:49/09:15/09:20，CPU约08:47。后续按固定final→完整独立验证→实际package CUDA replay推进，goal active。

## 07:02 接续：新验证collector已部署，训练962步

- 最新功能commit **3c335a284f430d924ff27178341dd1de547e12b3** 已push；CPU fetch/worktree session16827已exit0，独立checkout **P/repos/dreamlite-result-verification-20260913** 已固定此commit，包含新endpoint/validation collectors与此前独立package工具。不要修改正在运行的训练9628d71、probe0f40767、已等待的collector副本。
- 新scripts/reporting/collect_transition_validation.py绑定 **0f4076788bf4125bbca8851b6b270d5f8418d534** probe、新bank/parent/result/checkpoint/事前计划字节SHA，复算父900个开发单元；开发失败必须仍标diagnostic_after_development_failure。核验single72图/390raw/152artifact、chain96图/480raw/194artifact，所有事件/问法/seed/source前图链接、固定gold token、五问法同图；统计16条完整链，任一no-op失败都使所在整链失败。旧09b324d collector保持不变。
- 新tests/test_transition_validation_collection.py两项通过，连同endpoint两项本次4 passed；未改覆盖累计81项。详见reports/official-transition-validation-collection-20260913.md。**实际新probe尚未运行、没有新独立确认结果**。此工具只能在endpoint/probe真完成后收集；本地--text-only使用archive原始preregistered-plan.json，保留所有PNG，明确缺少远端PT。已有verify_rgb_chain_tensors.py可直接核验新96写输出，但也尚未对新结果执行。
- 07:00最新实际训练 **962/2880更新、elapsed1847.8s**，PID6087活跃；自动收集等待器PID463114也存活，仍waiting_for_training。无final结果，不能据loss单点变化选择检查点。07:00平台status实测RUNNING，Auto-stop In2h49m（约09:49）。CLI notebook没有原地延长子命令，start也无auto-stop参数；后续如时间不足，完整持久化并停妥当前任务后再新建验证实例，不缩减72图/96写/独立CLI重放。
- 无未完成的上传/fetch/查询会话。待完成的真实工作是正在运行的训练及随后final、独立验证和实际package replay；不要为了等待重复生成同类报告或再次运行已完成旧模型实验。goal active。

## 06:55 接续：严格端点collector已接到真实训练之后

- 最新功能commit **c2cf878** 已push，包含0c0066a严格端点collector、79caf79完成后收集worker及退出竞态修复。训练仍锁定9628d71，不修改训练checkout。最新训练只读查询session63196已exit0：**687/2880更新，elapsed1317.4s，PID6087存活/46068MiB，checkpoint-latest4.4GiB于06:51更新**。尚无final结果。
- scripts/reporting/collect_transition_endpoint.py要求真正completed父terminal、result/checkpoint SHA、45组/four-seed/five-prompt完整覆盖、两个phase各180PT+2文本文件、固定gold token IDs、问法共享图片、blank/donor跨phase不变，以及全部11520 draw的condition/teacher/noise/sigma精确重放、每组256次。仅完成的2880端点才输出summary/archive；显式--text-only可本地复核并列出远端PT缺失，不宣称本地权重核验。详见reports/official-transition-endpoint-collection-20260913.md。实际baseline1350raw已通过phase验证，两项新测试拒绝漏项/重复/伪造gold token/问法换图，2 passed，未改覆盖累计79项；真正final collector尚未执行。
- collector副本已上传 **P/runs/dreamlite-official-alignment/collect_transition_endpoint.py**，实际SHA **8949c0b3acbda7cc7cabbf356c67f3d3d446fd8742c978253985949abe6bc261**。等待器P/runs/.../collect_after_training.py已上传c2cf878版本。启动脚本P/runs/.../launch-transition-collector.sh（本地.cache同名）已经执行**一次**，不要重复。
- 实际后台收集等待器 **PID463114** 已pgrep核验存活，父训练PID6087。状态P/runs/.../**transition-wording-endpoint-collector-status.json**为waiting_for_training，日志同前缀collector.log，目前空且无错误。截止 **1789262400 =09:20CST**。父completed后仅CPU隐藏CUDA/OMP1/MKL1运行collector，PYTHONPATH固定训练9628d71/src；输出P/runs/.../**transition-wording-endpoint-summary.json**和**transition-wording-endpoint-evidence.tgz**。父paused/failed或进程消失三次观察后终止并保留failed状态，不重启训练。不把watcher waiting状态冒称收集已完成。
- 所有本轮上传/启动查询已exit0（40163、84090等均关闭），无未完成传输。训练GPU实例约09:49到期，CPU实例约08:47到期；完整final预估仍约08:40，按实测为准。0f40767新72图/96写validation探针尚未运行；281b653独立package parity checkout已部署。后续需新schema验证结果collector（旧collect_cfg1_validation固定09b324d，不直接适用新probe）；真正endpoint完成后再安排完整GPU验证，不能仅凭开发结果宣称可用。

**06:45部署补记：281b65360bd42e4822f2d5750b541d85661a97dc 已提交并push成功，包含以下预检证据、本地核验、parity工具和两项新测试。CPU fetch/worktree session90632已exit0，独立checkout P/repos/dreamlite-package-parity-20260913 已固定281b653，可用于新端点导出、CLI推理和parity。无待下载或待fetch会话；训练仍在原9628d71 checkout运行。**

## 06:44 接续：基线封存完成，已观察414步优化

- 下文06:29“尚未optimizer”和“10826a2待提交”的记录已过时：10826a2已成功推送。当前唯一GPU dl-transitions-h200x1-20260913 的 PID6087 仍活跃，46068MiB。最新只读查询session27633已经exit0，实际 **414/2880更新、elapsed794.8s**。此前06:31为42步、06:38为279步。不要重启训练；尚无final端点或新功能结果。训练run/checkout/deadline均保持9628d71固定配置。
- baseline已完成1350raw。新scripts/reporting/collect_transition_preflight.py实际远端运行成功，45/45教师原问正确且立即EOS，36个非灰源绑定RMS0，900 matched为0/900。远端collector验证全部baseline PT/JSON。已下载transition-wording-preflight-summary.json和-evidence.tgz；summary SHAdf05f300b7587c681fce0b71dd7366e04973d200c2b8d4a4031b06172fb42132，archive SHA f8e78e0db220f45c03872a27249dfa2f60eb4cbe9dd37584f8e76f09f15662ba。scripts/reporting/verify_transition_preflight_local.py已实际核验二者SHA、archive文本和1350raw，生成local-verification；PT留远端。全部相关下载已exit0，无待传输。46896曾因读错根目录training.jsonl返回1，只是只读路径错误；正确路径train/training.jsonl随后成功，不是训练失败。
- 新scripts/probes/rgb_package_parity.py准备/核验独立CLI重放，固定新链计划第一组六步/30读，不挑成功样本。prepare只在已完成新链+匹配导出端点时允许，CLI只收到event/seed/query commands，不收到reference/gold/bank/checkpoint。verify比对PNG字节、全部raw/input/output token/EOS和最终状态。parity不改变reference功能失败结论。两个新测试通过，连同包3项本次5 passed；此前未改覆盖累计77项。用法和未GPU执行限制已写推理入口文档。
- **真正的独立package native/Reader replay仍未运行**，需要等新训练final、完整新链和新CFG1导出完成；旧a333cf8工程导出仍是已知功能失败的CFG7.5端点。不能用它替代新候选。0f40767验证checkout已部署但未dispatch；本轮新parity工具待独立commit部署，不改训练和验证checkout。
- 接下来继续固定2880端点评测；预计训练约08:02后还需完整final评测，实际完成时间为准。CPU实例08:47左右到期，GPU训练worker09:15截止、实例约09:49到期；为72新图+96连续写及独立重放预留完整资源时长，不缩减已登记测试。

## 06:29 接续：独立推理入口与真实参数导出

- 当前功能commit **a333cf8c957f9d537ff1d319ae64213d0f390ed5** 已推送，独立checkout **P/repos/dreamlite-inference-package-20260913** 已fetch/worktree成功（session25956已exit0）。不修改正在训练的9628d71或待验证的0f40767 checkout。
- 新src/vision_memory/dreamlite/writer_package.py负责只含FP32参数的导出/完整性检查/加载，绑定官方Base源和快照，支持同内容快照移动目录；不读取teacherbank或oracle。scripts/inference/export_rgb_writer.py导出完成的Base/full/official端点，导出状态始终experimental_endpoint_requires_independent_functional_validation。scripts/inference/rgb_memory.py消费严格write(event,seed)/read(query)JSONL，拒绝gold等额外字段，Reader惰性加载，无答案输入；状态仅实际RGB PNG，Reader不能改变图或调用Writer，确定性设置在新worker生效。用法与限制见reports/official-rgb-inference-interface-20260913.md。
- 新tests/test_writer_package.py三项通过，覆盖剥离optimizer/teacher/query、完整参数加载/篡改拒绝、最后参数形状错误时零修改、无gold接口及连续只读。相关回归本次12 passed，未改旧覆盖累计75项；不是实际功能通过。
- 已用**旧c90896c full1536端点**做真实CPU工程导出，CUDA_VISIBLE_DEVICES为空、OMP/MKL各1线程，不占训练GPU。输出 **P/runs/dreamlite-official-alignment/a333cf8-export-engineering-full1536**，保留旧端点原nativeCFG7.5（不是新的CFG1发布候选）。导出session67165已exit0，逐值验证session17604也已exit0。
- 真实结果：1075参数张量、389968388值逐位等于原checkpoint。导出1560240719bytes，原训练checkpoint4680971012bytes；weights SHA **6a202dce122dde37877fbaa5c30989754b9ee9295fe99a4945c5cf9fc7ab17b8**，package manifest SHA **7b8fc78458b155c9ec0e17c8c311a4280bb82e1b50ab021818291ac020529184**。完整weights仅留远端，三个小证据文件rgb-writer-export-{manifest,complete,verification}.json已下载，manifest与seal哈希重验通过。旧端点功能失败结论不变；实际独立加载/native图与Reader parity尚未GPU运行，须在空闲资源上做，不能凭参数相同直接宣布全流程通过。
- 新report helper scripts/reporting/verify_writer_export.py已在远端真实验证通过，副本P/runs/dreamlite-official-alignment/verify_writer_export.py；运行它需PYTHONPATH指向a333cf8 checkout的src（副本在checkout外）。本地helper和工程证据/此记录待本轮提交。
- 当前训练GPU6087仍活跃，06:26实测baseline1230/1350；尚未观察optimizer更新。最近只读查询session34400等待输出（不是训练句柄，不能据其超时重启实验）。接下来先核验baseline完成与实际首次优化，再按固定2880端点收集，goal active。

**06:14接续：99f816a已提交并push成功（session95368已exit0），下文所述完整链归档/原始文本/本地核验/预览图与README均已入库。没有待传输文件或待push会话。GPU查询session99384也已exit0：模型PID6087、22660MiB、baseline750/1350，尚未优化。新验证探针0f40767的独立checkout已部署但没有启动；不要重复运行旧完成探针，也不要修改9628d71训练checkout。**

## 06:12 补记：旧链完整归档已恢复并复核

- 下载session34331已900秒timeout退出，得到59535360字节前段，SHAa73f979116e69ab11a059cbf35e91e0d4aead6a111eb83761ea256a56cd299ec。使用已提交推送4bdea12的scripts/inspire/prepare_transfer_tail.py在远端分出尾段；远端前段SHA与本地一致。尾段22482260字节，SHA4de62ec96bcf8b1b52cdbe045f208682bf30170dd490dca935b36e17c3753855，下载session76135已经exit0。
- 本地.cache/assemble_chain_archive.py核验两个片段并原子组装原文件，最终82017620字节SHA66f978d0d8f4d1f59504e02ab96fe5c2126fba61f8b0d6659a7c33ce02ba737e与远端封存archive完全一致。已展开cfg1-chains/，原始JSON/全部96PNG/480 raw及链链接复核通过，报告cfg1-chains-local-verification-summary.json；大型PT未下载，远端原96张量额外审计仍有效。
- cfg1-chains-preview.png按四顺序的首个重复固定渲染，并实际view检查。完整archive/展开文本/本地复核/图和结果README待本轮提交；忽略重复展开PNG，原图均在archive中。**现已没有待完成的下载会话，不要再等34331、76135或重传。**
- 最新实测GPU6087 baseline630/1350（06:09），训练进程仍活跃，尚未优化；下一步继续等待实际基线完成后观察首个参数更新，并为新端点验证准备足够lease。Probe checkout0f40767已部署，尚未运行新权重；训练checkout9628d71保持不变。goal active。

## 06:06 补记：新端点验证探针已实现并部署

- 最新功能源码 **0f40767** 已提交推送。新增scripts/probes/official_transition_confirmation.py，--mode single_writes/rgb_chains，绑定parent源码9628d71、bank962f0284、2880步/四开发噪声/nativeCFG1，校验900个开发单元、完整artifact/checkpoint/runtime/condition SHA。默认开发失败在加载前拒绝；--diagnostic才允许明确标成diagnostic_after_development_failure，不能覆盖原失败。记录固定/解析后计划、源文件SHA、真实raw/EOS、RGB源链接，冻结审计和结束复核，Reader查询不修改记忆。
- 新检查tests/test_transition_confirmation.py两项通过，包含漏项/重复/伪造token评分拒绝和原始事件/新表达绑定；相关回归本次10 passed，加之前未改的覆盖后总计72项。新probe还没有实际运行最终模型（父训练尚未完成），不能把--help或CPU检查当功能确认。
- CPU fetch/worktree exec session58377已exit0。独立远端checkout **P/repos/dreamlite-transition-validation-20260913** 锁定0f40767，实际GPU环境--help通过。训练仍使用自己的9628d71 checkout，未修改。验证输出/实例/deadline尚未派发，后续先根据训练完成时间和剩余lease决定；完整72图确认+96写链约需40分钟，不能在剩余不足时缩减已登记测试。
- 06:06真实模型进程6087仍活跃，baseline **491/1350 raw**，未开始optimizer更新。新实例dl-transitions-h200x1-20260913保持运行，worker截止09:15。按当前速率基线可能约06:30结束，再优化约90分钟并完成配对final评测；以实际终态为准。
- 链archive下载session34331仍活跃，900秒timeout。远端准确长度 **82017620**、SHA **66f978d0d8f4d1f59504e02ab96fe5c2126fba61f8b0d6659a7c33ce02ba737e**；本地06:02仅42040320字节，需等传输结束后完整比对再解包。不得将文件存在当作完整。如果timeout后文件不全，保留并只修复传输；旧链与模型都已经完成，不重跑它们。

**05:57接续补记：5457ee1已经提交并push成功（exec session35980已exit0），包含下文待提交的确认完整archive/原始文本/PNG预览、本地核验、链摘要/96张量核验、45组bank和报告；当前唯一未跟踪项是仍在下载的cfg1-chains-evidence.tgz（session34331）。GPU6087实测baseline已150/1350回答，未到优化。不要再等已结束的push或重传已验证的confirmation包。**

## 最新状态：09月13日05:54

- **旧09b324d链已全部完成**：480条raw/96写，99/480严格正确+EOS，16/96图通过全部五问法，0/16完整链通过；首次写入80/80、no-op0/240、后续写入19/160。complete SHA88339078bfdc2e0ebe43945f5385a6e97b32da862e318b46652d382e95e6fa2b。CPU collector核验全部PT/PNG/JSON、事前plan、tokens/EOS、源图文件链接，summary已下载。额外实际读取全部96PT，Reader tensor逐位等于PNG像素，初态逐位等于固定独立Gaussian、完整29状态轨迹起终点正确；cfg1-chain-tensor-verification.json已下载。旧queue/所有旧GPU进程均结束。
- **旧实例dl-align-full-h200x1-20260913已stop并delete**，不要再查询旧PID1055496。释放quota后新 **dl-transitions-h200x1-20260913** 已RUNNING，nodeqb-prod-gpu800、1H20020CPU200GiB/shm64、NGC25.02，05:49左右启动240分钟lease。
- 新训练 **9628d71-transition-wording-full2880-20260913已启动一次**，用下文9628d71固定checkout和962f0284新bank。实际GPU训练脚本PID **6087**，05:52约22652MiB、正在原生28步未训练baseline；尚未观察首个optimizer更新。日志run+.log及run/stage-0-*.log。deadline1789262100（09:15）保持；预计最终paired评测也要完成后才能判断。勿因baseline阶段没有training.jsonl而重启。
- 45组bank已下载为transition-wording-bank-manifest.json并复核SHA；每目标15组，源gray9组/其余各12组，只有三个unique targets。新独立确认计划已提交 **0c152ec**，与训练checkout9628d71分离，不修改正在运行的源码。新增计划检查通过，相关测试合计70；scripts/probes/transition_validation_plan.py和reports/official-transition-validation-plan-20260913.json已固定，但对应实际新权重probe尚需实现/部署，未排队。
- confirmation archive的exec session66540最终900秒timeout，**已结束**；实际上65102668字节已经到齐。文件锁释放后本地SHA **03ab9224538a367bcab8e4fe3106d4b7bc6ecf8948561a4cd981b2bee3d3b334** 与远端完全一致，已解包cfg1-confirmation/、重验所有本地JSON/PNG和390 raw，cfg1-confirmation-local-verification-summary.json使用archive原始preregistered-plan.json（保持LF字节SHA59912642...；仓库原plan在Windows为CRLF但JSON内容相同）。图cfg1-confirmation-preview.png已渲染并view通过。大型PT留远端，明确列出未本地重验文件。
- **当前仍在传输链archive**：exec session **34331**，下载 `P/runs/dreamlite-official-alignment/cfg1-chains-evidence.tgz` 到reports/official-alignment-results-20260913/cfg1-chains-evidence.tgz，timeout900。源archive已完成核验生成；先等客户端结束。若再出现收齐字节但超时，先从GPU读取远端stat/SHA，等文件锁释放后本地核对完整SHA，再解包/验证，不重复训练或盲目重传。未完成file不得commit。
- scripts/reporting/verify_rgb_chain_tensors.py已实际运行通过但待本地提交；其remote副本在P/runs/dreamlite-official-alignment/。确认本地证据、结果README与此记录待同次提交。PNG原图在完整archive中，.gitignore仅排除重复展开的PNG，montage仍跟踪。下一步实现新45组权重的已预注册确认/链探针，跟踪当前baseline→优化，完成链本地归档，goal active。

## 最新状态：09月13日05:40（优先于下列历史段落）

- 当前源码 **9628d7142db5a81a9d11a35b89d0515ef32d2e4f** 已推送，69项相关测试通过（原67+新2，新的采样计数测试首次调用参数错误已修正后通过）。新增45条件组的固定事件增广和pilot显式--eval-seeds（默认8保持不变）。源FM/运行中的09b324d代码不变。
- Stage A `09b324d-cfg1-confirmation` 已完成，complete SHA **9febcb5037d425e9b4fb067f3d9b39a1a691da66de377e84d5d67868518d0446**。远端collector逐个核验所有PT/PNG/JSON哈希、独立事前plan与390条raw：360 matched中320正确且立即EOS，64/72图通过全部五问法。三个原始事件各80/80，ambient/jazz两改写各20/20，**clear两改写各0/20**；第二种改写全部仍ambient。不要宣称确认通过。summary已下载，63MiB evidence.tgz正重传，exec session66540（900秒timeout）尚未完成；此前60秒/300秒下载超时，目标文件仍可能不完整，必须等待exit0再解包/提交。CPU实例RUNNING、剩余约3小时。
- Stage B 已自动启动，单H200 `dl-align-full-h200x1-20260913` 实际wrapper PID1055266、GPU worker **1055496**，约22642MiB；输出 `P/runs/dreamlite-official-alignment/09b324d-cfg1-chains`，log同路径+.log。固定96写/480回答继续跑，首两六事件链已观察no-op立即失败和后续覆盖不稳，尚非最终汇总。不要重启。新collector `P/runs/dreamlite-official-alignment/collect_cfg1_validation.py` 支持--kind chains/confirmation、--plan固定JSON、--output-prefix，可核验全部文件+tokens+链链接；--text-only仅显式允许本地缺失PT。PNG仍需全部下载。固定首样本渲染脚本scripts/reporting/render_cfg1_validation.py已提交。
- 根据已封存清除改写失败和真实链保留失败，新训练事前协议 `reports/official-transition-wording-training-20260913.md` 已提交。45条件=15已验证source/operation组合×3表达，仍一题/三个不同目标。fresh Base全U-Net2880更新、accum4、每条件256draw、每目标3840draw，lr5e-5/wd1e-4/clip1，native28CFG1；四开发噪声×五问法×45组=900 matched、450重复controls/phase。新两训练改写不包含旧确认改写，但旧确认已成为开发观察，未来需新留出验证。
- checkout `P/repos/dreamlite-transition-wordings-20260913` 已成功fetch并固定9628d71。CPU已验证原15组bank完整artifact/tensors并导出新 **P/runs/dreamlite-official-alignment/9628d71-transition-wording-bank/manifest.json**，SHA **962f02846ed1a1933e6c219604bc22ee520e28f2dfe2721e26f111dc36ea122e**。旧source/target原样复用、未新优化。新bank尚未通过45上下文GPU加载，训练入口将逐项验证。
- 新实例 **dl-transitions-h200x1-20260913** 已创建，当前PENDING；同project/group/NGC25.02、1H20020CPU200GiB/shm64、240分钟。05:38:54 events有parent project quota检查失败；应在旧链完成并释放旧单卡后让它调度，不干扰他人实例。之前过长名称dl-align-transition-h200x1-20260913被API拒绝，未创建任何资源。
- 新训练启动脚本已上传 `P/runs/dreamlite-official-alignment/run-transition-wording-9628d71.sh`（本地.cache/run-transition-wording.sh），**尚未执行**。输出将是 `P/runs/dreamlite-official-alignment/9628d71-transition-wording-full2880-20260913`，日志同路径+.log；deadline1789262100（09:15），早于新实例预计到期。等RUNNING、核验GPU空闲/共享环境后nohup仅启动一次；不要把排队当训练开始。若排队过久需在首次dispatch前核对截止预算，不能更改已运行任务身份。
- 下一步收全旧链final/raw/哈希，持久化证据后停止删除旧单H200释放quota，再启动新实例的新训练。完成本地confirmation归档与图预览；补充新bank/dispatch证据。goal active，仍无可用版本结论。

## 最新状态：09月13日05:20（以下历史段落不代表当前进程）

- 当前源码09b324dc7574ed33c0236ee09a4f5c3526fb85bb已推送；67项相关测试通过。唯一GPU实例仍为dl-align-full-h200x1-20260913，另有CPU实例；其他本任务GPU实例已删除。
- c90896c三状态全U-Net已完成1536更新/6144draw（各状态2048）。result SHA4dfb56f942d7dea62ac1d52ec91a0a814df30d4f6accc1238f9310dd822dff14；最终checkpoint SHA3c4b0679f16dd7714a662cfaddcd7716f7d3d43a49920898ab19a522d77af38d。原生CFG7.5仅20/120正确且立即EOS、0/24图通过全部五问法。全部draw本地精确重放，证据已提交three-state-full1536-*。旧训练PID396075和旧确认队列540574均结束，旧确认因开发门槛拒绝，没有生成新确认结果。
- 固定1e4acd2探针的native CFG1与training_raw CFG1两臂均完成，各120/120、24/24；同一个父checkpoint、相同八噪声、零更新、冻结审计通过。完整远端PT/JSON哈希及240条matched raw核验完成，three-state-controls-*证据已下载提交。这是开发结果，不能宣称可用版本。
- 新候选保留native条件和28步，只显式使用CFG1；固定新namespace计划已在09b324d提交。远端checkout `P/repos/dreamlite-cfg1-confirmation-20260913`；已启动一次队列 `P/runs/dreamlite-official-alignment/run-cfg1-validation-09b324d.sh`，外层日志/终态JSON为同目录09b324d-cfg1-validation-queue.log/.json。**禁止重复提交。**
- 05:20实际确认Stage A wrapper PID956812、GPU worker PID957334（22650MiB）正在运行。输出 `P/runs/dreamlite-official-alignment/09b324d-cfg1-confirmation`，日志同路径+.log；72图、360真实回答+30控制行，独立新噪声和两种未训练事件表达。门槛绑定已成功native CFG1对照及所有哈希。
- 队列在A技术完成后自动运行Stage B `P/runs/dreamlite-official-alignment/09b324d-cfg1-chains`，日志同路径+.log；16个六事件链、96次写入、480回答。实际生成RGB连续传递，覆盖/清除后插入相同no-op；错误后也不重置oracle，Reader查询不改图。A的功能失败仍保留，不能用B覆盖。两个worker截止06:15（1789251300），平台实例约06:40到期。
- 15组source-dependent bank已完成输入验证但未训练；后续依据新确认/连续链失败类型决定。尚无完整可用版本结论，goal active。新报告图脚本render_three_state_controls.py待生成预览/提交；不要修改09b324d远端运行checkout。

用户目标：新分支修复DreamLite与官方训练偏差，通过启智新实例实训、评测、迭代到有证据支持的可用版本。Goal已创建，仍active，未达成；不能以源码测试通过或训练结束标记complete。

## 本地与代码

- 工作树：`C:/Users/Expedition/dreamlite-official-alignment-20260913`。
- 分支：`codex/dreamlite-official-alignment-20260913`，源自latent-bank分支`f68bf06`，已推送origin。
- 官方源码在本地 `third_party/DreamLite`，HEAD=`a6e20c8cc94027f37dd7c5a81b0b3b472aa18409`。
- 已通过47项针对性测试（`PYTHONPATH=src`，本地默认`python`有torch/pytest；home/.venv python主要用于HF API，不是测试环境）。新增摘要严格EOS统计及缺失/重复五问法拒绝，以及全冻结inference审计；活跃3500实验使用旧固定提交，结束仍以原始generation计算EOS，不能只看旧terminal的exact字段。
- 重要修改：默认official FM、全0–1训练、整数t、原始source图条件、原始raw vs effective调度、纯噪声起点、rank16/accum4/AdamW官方默认；base选项直接官方pipeline训练/原生28步CFG评测；hash/optimizer/RNG/teacher EOS/real QA审计保留。

## 当前远端

Windows原生无inspire命令；使用 `C:/Windows/System32/wsl.exe -d Ubuntu -- bash -lc 'inspire ...'`。InspireSkill7.1.6，参考CLI help。

- 新CPU：`dl-align-cpu-20260913`，CPU资源空间、CPU资源-2、前沿课题探索，已refresh SSH connection。
- 新GPU：`dl-align-h200x2-20260913-r2`，分布式训练空间、开发区-H200-3号机房-2-cuda12.8版本，2×H200 143771MiB，40CPU/400GiB，NGC25.02，driver570.124.06。当前任务专用。
- 初次申请 `dl-align-h200x2-20260913` 因原组无GPU已停止，可以在结束时删除这个本次空实例；不要动历史其他实例。
- GPU restricted exec必须使用 `exec_command(..., tty:true)`，否则CLI误判本地管道stdin并拒绝；所有普通只读日志/文件工作通过CPU访问共享盘更方便。遵守用户禁用IAB/Browser Use规定。
- CPU命令：`inspire notebook exec dl-align-cpu-20260913 "<shell command>" --timeout 60`。
- GPU命令同上改GPU名并加workspace，但宿主exec分配tty。
- 准备依赖不使用Browser，CPU具备公网Git/HF访问。

共享根 `P=/inspire/ssd/project/exploration-topic/czxs26210936`。
模型根 `M=/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory`。
GPU Python `P/envs/vlm-r3-ngc2502/bin/python`；CPU纯stdlib脚本用python3（3.10.12）。不要更改系统Torch。

## 已完成与正在跑的实验

1. `P/runs/dreamlite-official-alignment/c7e752b-single-20260913`：官方Mobile逐位parity通过，但首次训练因launcher缺少确定性env在0步停止。后续已修复，保留失败记录。
2. `P/runs/dreamlite-official-alignment/106c8eb-single-20260913`：完整512步官方FM Mobile单目标，2048独立noise；teacher原问输出ambient+EOS；最终原问与四个改写均0/8，loss首末64步均值0.752814/0.439556，adapter deltaL2=12.6377。结果和checkpointSHA已复核，已下载 `reports/official-alignment-results-20260913/mobile-evidence.tgz`、展开文本和mobile-preview.png。
3. `P/runs/dreamlite-official-alignment/e76114c-base-single-20260913`：旧Base缺text_encoder权重，0步失败，已补全。串行queue目录后缀 `-queue` 已终止失败，不会自动再跑。
4. **已完成Base512**：`P/runs/dreamlite-official-alignment/ac34ab2-base-single-20260913`。固定源码 `ac34ab20b50fae82d56d1aef13fc995ee35c302a`，checkout `P/repos/dreamlite-official-base-20260913-r2`。Base官方类、28步native CFG7.5、同一teacher/seed/rank16/accum4/lr5e-5，512步。五问法均0/8、raw EOS也全失败；loss首末64均值0.684343/0.436569，adapterdelta13.174183。result与checkpointSHA已复核，resultsha=`afd3d93c222654c2754060610a93c072449288752db3733867f8f4e1550b92b4`。完整文本证据已下载为base-evidence.tgz，preview生成器修正动态Base标题（旧cache脚本错误硬编码Mobile，勿再用）。
5. **已完成oracle邻域诊断**：`P/runs/dreamlite-official-alignment/1bb0d28-neighborhood`，源码checkout `P/repos/dreamlite-neighborhood-20260913`。其`-queue`目录terminal completed，不再运行。48条输出、哈希已复核；原问及paraphrase_4，RMS0.001/.003/.01/.03均3/3正确立即EOS，RMS.1为3/3与2/3，RMS.3均0/3。teacher向Base512 seed0插值.01/.03/.1/.3两个问法都通过；纯Writer失败，RMS0.444540。这否定“只能精确坐标读出”，不证明任意方向鲁棒。诊断不算Writer成功。
6. **已完成CFG1**：`P/runs/dreamlite-official-alignment/32d70c0-guidance1`，源码checkout `P/repos/dreamlite-guidance-20260913`，日志同路径加`.log`。Base512同checkpoint、8噪声、native28steps，仅CFG7.5改为1。50条结果，五问法均0/8。已下载guidance1-evidence.tgz并验证文本哈希；`.pt`留在远端。
7. **当前活跃Base3500**：`P/runs/dreamlite-official-alignment/ac34ab2-base3500-single-20260913`。与Base512完全相同的固定代码`ac34ab2`/checkout，仅预算3500、fresh initialization。同seed同目标同超参同nativeCFG7.5，不改旧512运行身份；前512draw应相同。已实际确认GPU PID2622580，启动时间Unix1789235717.7。已观察85/3500步，并下载当前training.jsonl到本地.cache/base3500-prefix.jsonl；前85步除耗时外所有字段（loss/grad/draw/cursor）与Base512完全一致，报告reports/official-alignment-results-20260913/base3500-prefix-check.json。接续检查train/training.jsonl与terminal，而非重复启动。预注册在本地reports/official-base-3500-preregistration-20260913.md，提交2e08e97先于调度。预计约100分钟到端点评测；不以启动或loss成功替代功能证据。

**更新至北京时间02:29：主训练已实际观察1232/3500步，PID2622580仍占用GPU0约30GB、GPU1约9GB。前缀核验已扩展到完整512步，除耗时外字段全部一致，JSON报告已更新（读取时采样到了714步）。预计到端点还约60分钟，不将等待或源码测试当功能通过。**

8. **历史多题审计已完成**：`P/runs/dreamlite-official-alignment/historical-multiquestion-audit.json`，本地报告同名及historical-multiquestion-review.md。原16题实验已completed；CPU重跑固定commit2c0e41c的inventory/score函数，128run/32768updates/3456raw全验证。B原问64/64、第一改写64/64、第二61/64，旧BF16 VAE；不等于Writer成功，也不能未经FP32重放就移植为当前teacher。原数据路径和plan限制详见报告。
9. **提示词接口对照已完成**：第一次 `P/runs/dreamlite-official-alignment/4694fdf-event-format`（代码checkout `P/repos/dreamlite-event-format-20260913`）80条输出后因误用要求trainableLoRA的审计退出；日志同路径加.log。修复后 `P/runs/dreamlite-official-alignment/5daf0b2-event-format`（代码checkout `P/repos/dreamlite-event-format-20260913-r2`）全完成，80行逐字段与首次一致。两事件ambient/jazz×两形式raw/memory_note×4新噪声×5问法全部0/4；只有模型生成图片，没有外部答案栅格化。完成证据tar/PNG montage/replay检查已落地。此probe在GPU1曾使用约23GB额外显存，已经退出；当前只剩主训练PID2622580，不要重复启动。全冻结推理可以显式`load_base_runtime(...,inference_only=True)`同卡加载，但正常训练仍要求两卡。其审计必须使用`audit_inference_only_runtime`，不要调用要求可训练LoRA的training.frozen_audit。

worker deadline Unix `1789254021`（2026-09-13 07:00:21北京时间）；GPU平台8h上限约08:54。需要延长/追加时仍须根据新结果安排，不能让平台关闭前丢失检查点。未经结果证据不得扩大成“可用”或“多题泛化”。

## Base补全与数据绑定

- bank路径 `P/runs/oracle-to-unet/f5c9c9e-direct-multiprompt-dl-base-seed20260908/bank/manifest.json`，SHA256=`20ef4a9fc53b254fd99b12cbc01cf1a6d41dee8d04dd3120c70ecaa141f30722`。96个终点是同一个ambient问题，三训练/两留出问法；本轮按哈希选一个训练teacher做最低可学性试验。
- Base目录 `M/DreamLite-base-a9a0f15-20260907`，HFrevision=`a9a0f151ffd99d3c37f3fd0472f5e8f1b31215aa`。
- 官方API返回text_encoder权重4,255,140,312字节、sha=`7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`。缓存Mobile权重同SHA，经核验后硬链接补全Base缺文件，未替换不同模型。记录 `P/runs/dreamlite-official-alignment/base-weight-materialization.json`。
- 完整27文件seal `P/runs/dreamlite-official-alignment/base-complete-snapshot-seal.json`，payloadsha=`fc70dd11ccc652805388e0b0e687383c1441fbf9ea34c5384c8d7a241c7f7a72`。
- 旧26文件partial seal保留为base-snapshot-seal.json，禁止拿它恢复已补全Base。
- 官方源码远端 `P/Vision-Language-Memory/third_party/DreamLite`（锁定a6e20c8）。
- Base与Mobile VAE权重SHA均=`e175c3284b8514a2ecb1a70f004b0b09320decd6842e77dfebbdb5dbbdf0312b`；base运行时又比较全部非元数据VAE配置，并复验bank source。

## 收集结果

读取新结果可以用CPU纯stdlib报告器：
`python3 P/repos/dreamlite-alignment-reporter-20260913/scripts/reporting/collect_official_alignment.py --run <RUN> --output <SUMMARY.json> --archive <EVIDENCE.tgz>`。
报告器校验result/terminal与最终checkpoint SHA，汇总所有真实generation和训练draw；没有terminal仅输出进度，不能视为完成。

下载：`inspire notebook scp dl-align-cpu-20260913 --download <remote> /mnt/c/Users/Expedition/dreamlite-official-alignment-20260913/reports/official-alignment-results-20260913/<name>`。

预览脚本（仅可视化、无训练作用）在 `P/runs/dreamlite-official-alignment/render_alignment_preview.py`；GPU节点用已装torch/PIL的env执行并设置CUDA_VISIBLE_DEVICES为空可CPU渲染，输入--run/--output。已确认Mobile8个baseline图多为蓝色列车，训练后8个图主要为纹理，均无法读出ambient。不能用“多模式塌缩”解释本轮单目标的预期趋同，更不能把可视变化当功能成功。

## 接下来

先继续当前Base3500；其完整前512draw/loss已核验一致。等待真实结果时保持进程核验及适度进度沟通，不重新启动。Base512/CFG1失败、oracle邻域相对鲁棒共同支持测试官方3500预算；若仍失败，应再区分容量、优化目标与oracle训练目标，不能无边界只加步数。成功后还须全新固定噪声、反事实事件/多题验证；目前8benchmark噪声未参与训练，但研究迭代中已反复观察，不是 untouched research holdout。

预览生成器现在版本化为scripts/reporting/render_alignment_preview.py，远端临时副本 `P/runs/dreamlite-official-alignment/render_alignment_preview_v2.py`，会从identity读正确model_variant与预算标题。旧render_alignment_preview.py硬编码Mobile，只适用于历史Mobile图。

## 02:50容量对照更新

- 源码提交 `b0a06f5c597fce64457748b5d81f580116174547` 已推送。51项针对性测试通过，新增完整U-Net训练范围、冻结边界、周期checkpoint精确恢复和基线数值拒绝测试。
- Base3500在02:47实测1935/3500，仍使用原ac34ab2源码和原双H200实例，不修改它。
- 新两卡容量实例 `dl-align-full-h200x2-20260913` 因父项目quota剩余1GPU无法调度，0步停止并删除；本轮最早的空实例 `dl-align-h200x2-20260913` 也已核实STOPPED后删除。两者均无训练产物。**保留运行中的r2实例和CPU实例。**
- 新单卡 `dl-align-full-h200x1-20260913` RUNNING，node qb-prod-gpu2459；同project/workspace/group/NGC25.02，1H200/20CPU/200GiB/shm64，02:39:59创建，240分钟平台上限。显式将Writer和Reader放cuda:0，默认两卡训练路径不变。
- 新checkout `P/repos/dreamlite-full-unet-20260913`，新run `P/runs/dreamlite-official-alignment/b0a06f5-full512-single-20260913`，launcher日志为run路径加`.log`，阶段日志 `stage-0-1789238966133268987.log`。已调度，GPU实际模型进程PID40541；02:50正在执行未训练baseline，尚未确认首个优化步骤。
- 调度前预注册 `reports/official-full-unet-preregistration-20260913.md`。512更新、accum4、相同teacher/2048draw/seed/lr/wd/clip/native28CFG7.5；仅全量U-Net替代rank16 LoRA，其他模型冻结。worker deadline1789251300（06:15），早于平台06:40停止。
- 训练前必须生成 `train/baseline-reference-check.json`：与原Base512 runtime相同、8个latent/image和全部28步轨迹逐位相同、50条raw Reader记录逐字段相同，原resultsha afd3d93c222654c2754060610a93c072449288752db3733867f8f4e1550b92b4。不通过则0步失败，调查差异，不静默放宽容差。
- 全量checkpoint每16步、首步/末步/正常停止均保存，完整optimizer+RNG；硬中断保留physical log并重放未保存更新。默认LoRA仍每步保存。最终collector请用新checkout版本，读取通用unet_parameter_delta_l2；旧collector仅认识adapter字段。
- 接下来先确认full baseline gate及首步/16步checkpoint，随后收集两轮最终raw答案+EOS与哈希，再决定功能迭代；goal仍active。

**02:53实测更新：full baseline gate已经通过，8张初始latent/image、全部轨迹和50条raw记录均与Base512逐位/逐字段一致。核验JSON已下载为reports/official-alignment-results-20260913/full512-baseline-reference-check.json。PID40541占用45840MiB，full已31/512步，checkpoint-latest.pt为4.4GiB，已跨过16步周期保存；约1.9秒/步。主LoRA已观察2152/3500，预计端点约03:30；full端点预计03:10左右。不要重复启动。**

**03:00更新：full258/512，LoRA2447/3500，均实际确认GPU进程。主LoRA的nvidia-smi宿主PID2622580对应容器内PID411798；用pgrep完整命令核验，不能因容器ps找不到宿主PID误报退出。Full节点PID40541仍有效。**

- 条件诊断代码 `712c00d56323bc5c628d1766bf377beecb17c037` 已推送并部署到 `P/repos/dreamlite-condition-control-20260913`。53项针对性测试通过。预注册在reports/official-base-condition-control-20260913.md。
- 复用scripts/probes/official_base_guidance.py，默认native CFG1；新`--condition-style training_raw`在原生28步循环中使用训练缓存condition/mask，三分支重复同一条件、CFG和imageCFG均1，恢复hook并拒绝未消费override。加载LoRA/full权重之后全部冻结，结束检查无梯度/版本不变；修复旧probe仅能审计LoRA的问题。
- **已经排好本次full退出后的串行无训练对照，不要重复提交**：GPU上实际queue PID112379，脚本 `P/runs/dreamlite-official-alignment/run-full-condition-controls-712c00d.sh`，日志 `P/runs/dreamlite-official-alignment/712c00d-full-condition-queue.log`。检查/proc/40541/cmdline必须仍指向full run/train才等待；父进程退出后各probe各自核验completed/result/checkpoint。队列等待上限06:00（1789250400）。
- 固定两输出：`P/runs/dreamlite-official-alignment/712c00d-full-native-guidance1` 和 `P/runs/dreamlite-official-alignment/712c00d-full-training-raw-guidance1`。每个50raw generations、8图/轨迹和complete.json；第二个phase叫training_raw_guidance1，第一个guidance1。不要使用只找train/trained的主训练collector解析这两probe，应独立读phase complete/summary/generations并验hash。
- 队列原脚本本地.cache/run-full-condition-controls.sh；单卡H200仍只本任务使用，不修改运行中的full或LoRA源码。完成图预览标题已增加全U-Net/LoRA rank标识。

## 03:10 首个功能正结果

**Full512正式completed**，resultSHA `51332a99686e4de865bdfc344e68f371b132e0d9a741c1f0410c5852563f485b`；CPU新collector重新核验result和最终checkpoint通过。全部5问法8/8正确且立即EOS，raw40条均ambient、tokens[59614,151645]，baseline全部错误，blank/donor10条均错误。389,968,388参数，deltaL2=19.940394，末64FMloss0.054692 vs相同draw LoRA0.436569。完整512updates/2048draw已对齐。8生成距训练teacher RMS0.05682–0.06772。**这只证明单题可学，不标goal完成。**

- 已下载 `reports/official-alignment-results-20260913/full512-evidence.tgz`、full512-summary.json、full512-preview.png；文本展开到full512/，本地phase文本SHA和result再次通过，PNG已view。PT和4.4GiBcheckpoint留远端。训练配对报告full512-draw-comparison.json。
- Full训练PID40541已结束。Queue112379已自动进入第一臂：native CFG1 wrapper PID190916、GPU worker PID191076（显存22624MiB）。两臂既已预注册，继续收集，不重启父实验、不取消来替换成功端点。
- 主LoRA3500仍活跃，最后观察2786步。容器PID411798/宿主GPU PID2622580。预计03:30附近结束。
- 完成图生成器最新共享副本 `P/runs/dreamlite-official-alignment/render_alignment_preview_v3.py`，title正确区分fullU-Net/LoRA rank；后续用它。
- 下一步：收集两条件臂和主LoRA；事前固定全新噪声的full最终权重确认测试（继续原生CFG7.5，不基于待出对照结果挑配置），并用相同实体的事件值替换/清除检验是否仅恒定输出ambient。随后基于该证据准备多事件teacher与共享Writer训练，不能把当前单题结果充当可用版本。
- 旧16题数据有`mixed`类型，既含event_text也含query；后续提取事件必须包括它，不能只筛type==event。之前读取前缀出现gold与首个事件不一致就是漏看mixed更新，并非原始数据错误。旧BF16 teacher仍需当前FP32读取验证，不能直接导入。

## 03:22 新噪声确认已启动

- 两个712c00d条件对照和queue112379都已经结束。两臂各5问法8/8正确+EOS，40matched raw均ambient。CPU已逐个核验全部PT/JSON哈希与原始token评分，下载full-condition-evidence.tgz和full-condition-summary.json，文本展开full-conditions/并再次验SHA。不要再启动这两臂。
- 确认源码提交 `cb41fa1bc9c1cc50a6c4d178dacd6f9a89ec2765`，checkout `P/repos/dreamlite-writer-confirmation-20260913`，run `P/runs/dreamlite-official-alignment/cb41fa1-full-confirmation`，日志为run加`.log`。单卡实例dl-align-full-h200x1-20260913实际wrapper PID249356、GPU worker PID249886（03:22显存16960MiB，正在加载/生成），后来日志已见28步进度。
- scripts/probes/official_writer_confirmation.py绑定成功full512 resultSHA51332a...，检查完整parent manifest/cursor和checkpoint；加载后所有模型冻结。16个新namespace噪声对原始事件、同前4噪声对jazz/clear事件，blank/donor各一次；24张图、130条raw答题，原生CFG7.5/28步，deadline1789251300（06:15）。无训练、不改父模型、不选择CFG对照结果。已编译并验证固定/配对/无重叠seed计划，调度前note+exactplan JSON已提交。
- 主LoRA3500最后观察3320/3500，仍等待端点及最终校验，容器PID411798。继续收集，不能因剩余少而提前停。
- .gitattributes新增结果目录 -text，实测即使Windows core.autocrlf=true，Git checkout过滤后仍保持full result原始SHA。证据tar仍为原始bytes来源。
- 旧16题FP32兼容性还没有执行。若要复用：B/seed0/step256在各lane/runs/target-XXX-seed-00-B/checkpoints/step-256.pt，payload键latent_fp32，checkpoint_index.jsonl有SHA；原训练是BF16VAE、仅original_open CE+EOS。旧三问法的instruction行也不同于当前五问法，不能只换dtype后就宣称满足新bank合同。后续应固定新五问法做FP32读取，或按已验证FP32三训练问法+两留出问法重新准备同实体ambient/jazz/clear的独立目标，再训练共享Writer；取决于当前确认的条件敏感性结果。

## 03:31 全部现有训练/确认已完成；下一步多状态

**当前GPU上已无活跃计算进程**，两个实例都实际查询nvidia-smi确认空闲（2卡r2与1卡full）。保留供紧接的目标构建/共享Writer训练使用，不再等待/重启旧PID或队列。CPU仍可用。主LoRA3500正式completed，全部五问法0/8；resultSHA `dc5b38055139bf202b92ad860300e984dee29a075294a419722589a7d4516604`，末64loss0.332762，deltaL2=33.412885。CPU重验result/checkpoint通过，base3500-evidence.tgz、summary、preview均下载、文本展开base3500/复核，图已view。

Full确认cb41fa1也completed：原事件16新噪声×5问法80/80正确立即EOS；jazz和clear各20条全失败，仍raw ambient/[59614,151645]，blank/donor各5失败。全部artifact哈希远端复核，full-confirmation-evidence.tgz/summary下载，展开full-confirmation/，文本/PNG再次验SHA，paired事件图full-confirmation-preview.png已生成并view。这证明当前权重是单状态记忆，并未实现事件值变化；goal仍active，下一步必须共享Writer多状态学习。

已查明更合适的已验证oracle配方（不必先迁移旧BF16多题）：当前ambient teacher来自 `P/runs/direct-multiprompt-eos/46cd36b-20260909-r01/lane-1/runs/direct-gaussian-s07-a0.5`，源码commit46cd36b在本地git可读。其FP32 VAE+bf16 Reader、Adam lr.05、256更新、EOS权重1、三训练问法original_open/paraphrase_1/paraphrase_2按zero-based round_robin；paraphrase_3/4只用于评测。不是每步累积三个问法。`git show 46cd36b:scripts/experiments/direct_geometry_eos_training.py`已读，可据此恢复显式可选multiprompt能力，当前worktree该helper仍为仅original版本，不能误调用默认就声称复现三问法。

建议下一步具体执行：在已空闲的2-H200 r2上准备同实体jazz/clear两个FP32 latent目标，与既有ambient teacher一起构成3状态bank，再用1-H200 full实例训练共享官方FM U-Net。初始化固定复用上述hash选定teacher的原始高斯seed7/scale.5起点；原始`latents/step-000.pt`(payload latent_fp32)可从source_run读取并用latent_index.jsonl及bank记录的index SHA验证，避免按新状态结果选初始化。每个新目标256更新、三问法轮转、两问法留出；所有5问法成功+EOS后才能封存，不能失败后换seed择优。仍须事前确定新实验配置后再派发。

原ambient bank teacher记录（.cache/teacher-bank-manifest.json内）有完整来源：source manifest SHA d331859338726a6c90cbca1448f74c47840a264b27603a0d60a5af2792bb201e，latent_index SHA2ee4b8c90d254649bb2b68324b20e03cc492963acad5a4c238190aa2e5b48d8e，endpoint SHAa58c978cf78dc2955705a8d370f7d60bf9e1805343a6e563d108be9f455e6cdd。当前独立teacher tensor SHA e910e9b89ed3861ef7c36176c762bccde7d9d809e2510618b5318202313777f6。

新bank若按状态使用不同group ID，必须明确这是同一语义问题的3种条件状态，不能报成3道独立题；问题5种问法沿用原组，Writer只接事件和source，不接query/gold标签。沿用source_kind blank_gray_1024、原封存source tensor及PIL128条件差异约定。不同答案donor仍可用原orange RGB control。暂未实现或调度多状态oracle/Writer，不要把建议写成已运行。

## 03:56 三状态目标完成，共享 Writer 实际开始优化

上述03:31建议已执行。代码c3bee0e8564ca379300326ed3f2d2ebe9ec8d8d5、checkout `P/repos/dreamlite-state-oracles-20260913`、run `P/runs/dreamlite-official-alignment/c3bee0e-state-oracles` 已完成。ambient复用固定原目标；jazz和clear各从已核验的原seed7/scale.5初态训练256更新，三问法计数86/85/85，p3/p4不参加梯度。三个目标都五问法正确且立即EOS。CPU核验45条raw、两轮512更新、所有中间latent/checkpoint文件SHA；本地再验三个bank tensor canonical SHA。证据state-oracles-evidence.tgz、state-oracles-summary.json、state-oracles/已下载。三目标pairwise RMS分别0.36555954、0.43353733、0.42337517。这些是oracle目标，不是共享Writer输出。

原bank SHA4e68df0fe38a4eb38c6cbf1903229ac81598d7b582c480e1700dbcf69e15b2f9遗漏snapshots字段，首次共享Writer `63483ac-three-state-full1536-20260913` 在模型加载前KeyError、0更新退出，three-state-first-attempt.tgz保留。修复exporter并用scripts/reporting/complete_state_bank_metadata.py产生新manifest；该工具验证原始bank和parent bank SHA、目标及raw hashes，只复制原有snapshot绑定且记录provenance，原manifest不变。新bank `P/runs/dreamlite-official-alignment/c90896c-state-bank/manifest.json`，SHA **5165c059a0a4c0760cd4b0df1663c642132e52bf55a13031cd99207fd200bb86**，本地state-bank-complete-manifest.json再验SHA及groups/teachers逐字段不变。

**当前唯一活跃GPU训练**：源码c90896c55c7b6cd6f489ccb7848ade2cff587daa，checkout `P/repos/dreamlite-three-state-writer-20260913-r2`，run `P/runs/dreamlite-official-alignment/c90896c-three-state-full1536-20260913`，stage-0-1789242667553720835.log。单H200实例 `dl-align-full-h200x1-20260913` 实际GPU PID396075、45842MiB；03:56观察28/1536更新，约1.9秒/步，已出现checkpoint。不是仅提交/加载。预注册reports/official-three-state-writer-20260913.md：fresh Base full U-Net、lr5e-5/accum4/clip1/wd1e-4、三状态各2048训练draw、原生28CFG7.5/image1；没有继承ambient-only权重。1536结束后还要完成3组baseline/trained的真实Reader评价。deadline1789251300（06:15），实例约06:40到期。55项相关测试通过。不要重启这个run。

2-H200实例 `dl-align-h200x2-20260913-r2` 已实际核验无GPU计算进程，目标/旧训练证据均在共享盘且本地下载，现已stop并delete。保留上述单H200和CPU实例；不要等待已删除实例的历史PID。三状态Writer尚未功能验证，goal继续active。正在预注册新噪声及两种留出事件表达的确认测试；只有固定1536端点的每个状态/问法/噪声全部通过，才进入该确认。

## 04:10 确认代码已准备，主训练继续

本地新commit **1e4acd2bcd9cbf9f0b2878e51694c17bebe06110** 已产生，包含所有三状态目标证据、修复后manifest、首次失败记录、scripts/probes/official_three_state_confirmation.py、固定JSON计划和预注册。57项原套件/确认门槛测试加1项按组统计回归，共58 passed。确认方案：每状态原始事件16全新配对噪声、两种未训练事件表达各4噪声；72张图/360真实答题，加按三答案重复评分的blank/donor30行。确认脚本在任何模型加载前要求父commit/bank/1536步设计和完整已成功的120个development matched单元，拒绝重复、漏项、失败、额外token；保存全部轨迹、raw和冻结边界。不是自动调参或挑选最佳端点。

本地exec session72824的git commit/gc/push已全部exit0，1e4acd2已推到origin。不要重新等待或重启它。Git common dir为D:/2026WorkExperience/VisonLearnableMemory/.git。后续audit、此续接记录、render_alignment_preview.py多状态修复另行提交。新render用(question_id,noise_seed)映射，避免三组同噪声答案覆盖，按组排版。

**确认队列已实际启动，不要重复提交**。CPU已经fetch并创建 `P/repos/dreamlite-three-state-confirmation-20260913` 锁定1e4acd2，shell位于 `P/runs/dreamlite-official-alignment/run-three-state-confirmation-1e4acd2.sh`（本地.cache/run-three-state-confirmation.sh）。单H200实际bash PID540574，GPU上仍只有父训练PID396075/45842MiB。队列每20秒等父terminal到06:00，随后固定调用确认脚本，deadline06:15；失败门槛会在加载模型前拒绝。固定输出 `P/runs/dreamlite-official-alignment/1e4acd2-three-state-confirmation`，worker日志同路径加.log，queue.json记录退出码，外层nohup日志同路径加-queue.log。仅退出0还不等于功能成功，须检查complete.json、各state/style/prompt的raw/EOS和所有hash。父训练仍为c90896c，不修改其checkout。

主训练最近实际观察335/1536（04:06–04:07），1536预计04:45前后结束优化，其后还有3组原生采样/Reader评测；具体以terminal/raw为准。没有新功能结果。已上传新版CPU报告器为 `P/runs/dreamlite-official-alignment/collect_groups_1e4acd2.py`（scp session73539已exit0）；其per-conditional-group计数避免constant-state被aggregate掩盖。它可先采进度，也可在端点后核验checkpoint/result并打包。原state bank遗留descriptive planned_count/successful_run_count=96来自parent，但实际每组teacher列表只有1，训练完全不读这两个字段；future exporter已改为1，正在运行的manifest保持不变，预注册也公开此点。

## 04:33 非灰图源条件已真实验证，主训练1183/1536

源码 **6419eda2270e853e6d18a21b99e7f509069aa130** 已推送；加入sealed_rgb_1024路径和15组source-dependent转移准备，62项相关测试通过。source_images.py严格核验PNG字节SHA/RGB/1024尺寸，不隐式缩放或重绘；official_base_runtime实际编码必须等于bank source tensor，完成再验PNG SHA。blank control始终用灰图，旧灰图分支的PIL128与float0.5差别仍显式保留。所有在跑的旧checkout/队列继续用其固定旧代码。

临时新单H200 `dl-align-sources-h200x1-20260913`（同workspace/project/group/NGC25.02，1GPU20CPU200GiB/shm64，120分钟限制）实际worker PID22200已完成，nvidia-smi确认无计算进程，证据持久化/下载后stop并delete，不要等待其旧PID。checkout `P/repos/dreamlite-source-transitions-20260913`；run **P/runs/dreamlite-official-alignment/6419eda-source-transitions**，日志同路径+.log；deadline1789247700（05:15）。现在只保留主单H200和CPU实例。

结果：三个oracle目标只经VAE decode/nearest uint8/PNG成为输入源；PNG真实Reader五问法15/15正确立即EOS，无新优化。实际加载15个新group，三个no-op同文本下的native28完整轨迹捕获通过，source均逐位等于训练输入；三个image-aware condition SHA不同。新bank **P/runs/dreamlite-official-alignment/6419eda-source-transitions/manifest.json**，SHA **3d89168c9d36e6df5ded067e54d13128dda1a3cf2c5516600901b017ebe377f2**。3个gray→目标、9个已有状态→目标、3个noop；一个semantic question、三个unique target tensors、15组/teacher记录（重复目标只为条件身份，不是15个独立目标）。完整权重冻结检查通过。新的15组bank尚未训练，不把pretrained native source检查图算作功能成功。

CPU报告器 `P/runs/dreamlite-official-alignment/collect_source_transitions.py`（本地.cache/同名）已验证全部产物SHA、15raw/EOS和noop设计。已下载source-transitions-summary.json、source-transitions-evidence.tgz，展开source-transitions/；本地再次核验下载文件、三个canonical source tensor、12个source binding RMS=0及15raw。source-transitions-local-verification.json列出三条留远端的完整native PT轨迹，其远端SHA已验证。jazz-source.png已view，只有模型产生的纹理，没有外部答案栅格化。候选candidate-manifest.json保留为中间产物，后续仅使用正式manifest.json及complete封存绑定。

主c90896c Writer最新观察 **1183/1536（04:33）**，PID396075；1e4acd2确认队列PID540574等待它。预计04:45附近结束优化，再有最终评测。先收齐该端点按状态raw/EOS：若主CFG7.5失败，使用已有scripts/probes/official_base_guidance.py固定native CFG1与training_raw CFG1诊断条件/引导差异，不重启或只加预算；若通过，既有确认队列自动执行新噪声与改写。15组bank为后续source-dependent保留/更新提供数据，但训练预算/条件设置应基于当前端点结果确定。后续还需实际生成图→PNG→官方再编码的链式事件测试；现有确认仅从灰图开始，不能当作完整可用更新器。goal仍active。
