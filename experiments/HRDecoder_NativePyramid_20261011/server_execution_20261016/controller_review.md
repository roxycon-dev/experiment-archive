# 服务器执行控制器只读审查（旧版历史记录）

**旧版审查已失效：用户明确要求准备、预检和排错失败时保留服务器。第二版已取消这些阶段的关机，详见文末修订记录。旧版“一切异常都关机”的结论不得用于部署。**

2026-10-16初审，2026-10-17闭环记录（Asia/Shanghai）。范围：本地 `controller.py`，补读关机包装脚本准备逻辑。没有连接服务器，没有执行 shell 命令、训练、测试或关机。初读发现的五项问题已由主代理修复，本子代理重新完整读取了194行落盘控制器，闭环如下。

## 初读发现

1. **关机不能被结果持久化异常跳过。** 初读 `finish_and_shutdown` 第101–116行先read/JSON/write/fsync，再调用包装脚本；其中任何异常都会直接退出该函数，导致不请求关机。应将关机动作放在最外层finally，结果尽力保存，保留保存失败和计算不确定性的记录，不将异常写成训练完成。
2. **控制器异常时须先处理还活着的child。** 初读stage第49–60行启动child后，监控read/write或KeyboardInterrupt异常会离开stage，但没有终止并等待child；随后控制器立即走关机，可能把尚未保存完整边界的训练进程直接中断。应先请求child正常中断、等待其保存终态/不确定更新证据；不能退出则如实标记强制终止及未确认成本，再请求关机。
3. **记录的源码SHA没有核验。** 初读SOURCE_SHA只是写入状态，verify PASS后未要求preflight.source_sha256等于该值。train/evaluate自身有更完整的身份门，但控制器不能无核验地把另一个实际源码身份报告成这个固定SHA。应在开始正式训练前核验。
4. **marker写入失败也须纳入自有finally。** 初读main的controller_started独占创建、JSON写入、fsync在try外。marker创建成功后如果dump/fsync失败，不会执行关机。应保留“重复启动失败不关掉另一个活控制器”的行为，同时让本进程成功取得独占后发生的失败进入自身收尾。
5. **关键运行前置条件不应仅用assert。** 初读run_job第67–68、88行环境/全新目录/终点判定使用assert，可能被PYTHONOPTIMIZE关闭。应使用显式异常。训练自身仍有新鲜目录保护，但控制器应保持同样约束。

## 初读已正确的行为

- 只有verify子进程exit0且preflight.status为PASS才尝试启动train；正式命令只启动一次、固定micro1，未使用resume、改超参数、回滚、阈值搜索或追加训练。
- train返回PAUSED时保留运行产物并结束本次尝试，不自动续接；outcome写FORMAL_TRAINING_PAUSED，不能将其说成完成40k。
- 40k终点锁的test_permitted为true才调用独立evaluate；该固定源码的evaluate还须核验完整40k/内部val/身份/真实test文件SHA后才读测试像素。
- 控制器marker和各stage日志独占创建；重复控制器启动被拒绝且不进入第二次shutdown流程，避免第二进程误关正在执行的第一进程。
- 关机状态使用SHUTDOWN_REQUESTED，调用成功本身不证明实例已实际停止；对外只能按已证实的状态报告。
- 已看到控制器改用`receipts/provider_shutdown_preserve_files.sh`。准备脚本从原provider wrapper仅移除指定Trash清理行并保留SHA收据；不得调用原带清理副作用的脚本。

## 修复后复核

|初读问题|当前源码|闭环判断|
|---|---|---|
|保存结果异常阻断关机|166–177行：save_final_result失败尽力写finalization_failure，最外层finally仍调用request_shutdown；135–163行：关机收据失败不阻断调用，shutdown.log打不开改DEVNULL。|已修复。异常时不会因JSON/报告/日志写入失败直接跳过关机请求。|
|控制器异常后仍有活child|50–78行：child为新session，stage的finally先SIGINT自有child并wait45秒，必要时仅终止其自有process group，并记录forced_stop_boundary_uncertain。|已修复。先给训练/预检持久化失败和in-flight的机会；超时强制终止明确承认边界不确定，不回滚重试。|
|固定源码SHA未验证|92–97行：verify exit0、preflight.status=PASS后，必须preflight.source_sha256等于固定SOURCE_SHA才可启动正式训练。|已修复。下游train/evaluate自身仍核更完整源码、数据、环境与预算身份。|
|marker取得后写入失败不走finally|184行独占取得marker；185–193行将dump/fsync及run_job纳入自有finally。|已修复。重复取得失败不会关掉另一个owner；本进程成功取得后的失败会执行自身收尾。|
|assert可被优化关闭|84–87、109–110行为显式if/raise。|已修复。|

正式路径只有一次 `verify → train → eligible evaluate`，没有训练重试、resume、回滚、配方修改或追加预算。verify非PASS/非0退出不启动正式；train的PAUSED、异常与成功分别保留outcome，不把暂停或失败说成完成40k。test仍依赖固定40k锁与原evaluate的完整门控，不能靠控制器的test_permitted字段跳过原评估器身份/val检查。

控制器使用去掉无关Trash清理副作用的provider包装脚本；本次没有执行该脚本或实际验证provider关机服务可达。shutdown最多三次请求，状态为SHUTDOWN_REQUESTED或SHUTDOWN_COMMAND_FAILED；不能只凭命令返回0就宣称付费实例已实际停止。主代理报告6个纯控制流mock测试通过，本子代理没有自行运行，测试证据以主代理实际报告为准。

**当前结论：初读五项问题已源码闭环，没有发现阻止按既定流程部署启动的剩余控制器缺陷。** 这是一份只读代码审查，不能作为预检已PASS、训练成功、目标达标或实例已关机的证据；这些均须由后续真实运行和实际状态证明。

本次回读controller.py字节SHA256：`68c68ba88cb5dadfb66ff039c7fb8146933907144ccb47209c52cc690fbb162f`。
# 2026-10-17 用户纠正后的状态

上文为第一版控制器的历史审查，不能作为当前关机策略的依据。第一版把预检失败也当作关机条件，违背用户“任务完成后关机”的要求，已由主代理明确承认并撤销。

第二版控制器默认不启用关机；仅观测本源码正式训练完成至少一次SGD更新后启用，并且仅在固定40000更新以及所需评估成功完成后允许请求关机。环境准备、预检、工程排错、正式训练失败或PAUSED均保留实例，没有修复超时关机或空闲关机。实际provider调用及外层finally均有条件保护。

本地11项控制流程mock测试通过；当前SSH端口拒绝连接，第二版尚未部署到原服务器。上文对旧版“一切异常finally关机”的肯定已失效。

