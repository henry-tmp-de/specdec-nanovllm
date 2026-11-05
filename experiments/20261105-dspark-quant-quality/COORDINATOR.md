# DSpark量化条件质量诊断：协调记录

## 目标与授权

2026-11-05用户要求尝试哪些DSpark层量化时精度损失较小，并询问DFlash非独立draft能否测PPL。本批只做固定参考下的条件质量/漂移诊断，不训练、不新增W4、不重跑吞吐矩阵。上一轮W8A16性能测量已经完整结束，原结果不覆盖或改名。

新工作区D:/学习/specdec-dspark-quant-quality，分支feat/dspark-quant-quality-20261105，起点49265569d2cb21015b4025148466ce21ff134fd0。Terra Max实现与数据工具，GPT-6.1 Sol Medium独立审核，主协调亲审并操作GPU。预算单卡累计阶段墙钟1200秒，其中smoke最多300秒，含加载、量化、预热与清理；编码与CPU准备阶段不占GPU。无训练、sudo、安装、模型下载或其他实验启动。

## 指标含义

不能将模型单独送入普通LM PPL脚本：DSpark需要target历史features。本批固定完整因果prefix、anchor、窗口与7位置，使用真实前驱进行Markov条件评分，计算条件token NLL及exp(mean NLL)、量化相对BF16的exp(ΔNLL)。这是所选参考位置上的条件分数，不是标准自然语料LM PPL，也不与原0.6B PPL直接比较。量化损失小通常指NLL/PPL升幅小；负ΔNLL不等于泛化质量改善。

同时记录KL(q_BF16||q_quant)、KL(p_target||q_quant)、sum(min(p_target,q_quant))及top-1一致性，按面板/位置/整体复算。overlap是同一teacher-forced历史下的单步接受代理，不是实际接受前缀比例/每轮交付；不将七个位置相乘推算真实tau。

## 共同参考与因果边界

复用此前普通target Graph解码、T=1、seed20261105的实际输出，不重新生成参考：旧final-all-01中formal4a085-35-nospec-b8-c1024.jsonl，三面板各8条、repeat0，共24条256-token轨迹；通过input_metadata真实hash映射workload-1k-original.json的prompt IDs。

每条轨迹选offset0/64/128/192：prefix=prompt+output[:offset]，anchor=output[offset]，labels=output[offset+1:offset+8]。共96个锚点、每variant672个评分位置；不是96篇独立文章或人类gold。原prompt含Wiki填充，相关来源、标签和hash保留。

target为同权重Qwen3-4B BF16的HF参考实现。prefix-only前向得到完整target cache及hidden_states；从layer_id+1（2/10/18/26/34）取最后512行features。anchor和labels不得进入生成features的输入。随后target cache续算anchor+7labels，以前7个causal logits评分labels，完整prefix长度C必须体现在position_ids与mask/cache中。

DSpark复用原模型代码/官方权重/target共享embedding及LM head，窗口512，显式chronological slots覆盖同样的512个历史位置；每variant都从相同features重建KV。此独立数学诊断不检验native环池owner生命周期，也不测性能。HF与nano后端有数值差异，不声明逐位等价。

## 15个预设与执行门槛

none；q/k/v/o/gate/up/down各一组（均选5层相应投影）；q_o、kv、gate_up三组合；ffn、attention、all三组合；增加q_o_down（5层q/o/down共15模块）。fc/Markov/shared embedding/head/norm与KV保持BF16。量化bank与BF16原模块同时保留，不统计bank模式的显存节省；切换后不可重放旧CUDA Graph，诊断用eager。

每个位置q由base logits与true predecessor的Markov修正组成：j0用anchor，j>0用前一个label，原生BF16相加后转FP32。不能把当前或未来label送入backbone。

先CPU因果/切分/数学/恢复契约与Sol代码审核；smoke使用2个case核有限性、概率归一化、位置对应、prefix cache长度、每组重建KV、none首末实际输出恢复和feature hash不变；通过后完整96case另写文件，不重复计入smoke。full完成复算全部raw并确认自有GPU进程退出后停在下一门禁。数值、OOM、外来占用或预算问题保存证据，不自动扩训练/改模型或重复实验。

最新用户进一步明确：最终目标是速度，不是只找最低条件NLL。此前B8图模式的Q/O/down投影有正收益，而gate投影明显变慢；因此增加q_o_down候选。质量仅是选择条件之一；必须固定所选权重与地址、重新捕获Graph并验证真实请求速度，才可能宣称加速。当前只批准诊断新增组合，尚未启动生产preset修改或新的吞吐矩阵，等质量结果由主协调决定。没有新NCU/完整GPU分解，不编造实际瓶颈占比；KV池差异也不能直接解释无抢占固定B测试的速度差。

## 当前状态

本批质量及固定部署速度测试均已完成，资源释放，无后续GPU任务，最终入口FINAL_REPORT.md。现场初始GPU0至5空闲，6/7有compute；不得用该历史快照启动新任务。

## 已完成质量诊断

冻结a4e5a9c（评估器0ddf734，数据/数学bbfbae0）经过主协调亲读与Sol Medium独立审。主协调独立联合CPU21项通过；smoke实际2case×15组共30条评分，none首末NLL/KL差均为0，实际guard97.718113秒。完整96case/24trajectory×15组形成1440条评分，每组672位置，96次首末恢复检查通过。full守护耗时158.856660秒，两阶段总计256.574773秒，均ok/exit0，GPU0阶段间已释放。

完整raw与报告经Sol独立复算。q_o/down/q_o_down相对BF16的条件指数分数比为0.99777730/1.00123541/0.99854076，KL(qBF16||qQuant)均值0.00046535/0.00042223/0.00052563。不能只看overall：局部panel×position有约1.4–1.6%的分数升幅，且每格只有32个位置；这些数据是候选screen，不是质量无损认证。完整报告在analysis/quality-full-01，smoke独立analysis/smoke-01，不合并样本。

## 最新速度方向与CPU门禁

用户再次强调量化是为了速度后，主协调批准准备固定部署q_o/down/q_o_down三个生产可选预设。d878e86只扩config/expected_names与新native bench，不改量化内核、target、旧四预设或默认none。权重在加载后、建图前一次固定替换，质量bank不会进入性能测试。主协调独立CPU43项通过，Sol也审实际diff、CPU33+96日志及独立43项通过。selected审计396a8d6单独复用原native compute_run与配对，不能使用旧硬编码四preset审计静默混组。

质量screen已支持继续验证上述三候选。下一阶段由主协调在新clean clone重新现场双查后，只运行B1/B8×none/q_o/down/q_o_down共8个模型进程组，每组三面板三重复、1K输入/128输出/W512/0.85/精确Graph；预计72run、324请求、41472输出。继续同一1200秒总账，当前剩余943.425秒，不扩预算。若异常或余额不足，保存raw停止，不自动改矩阵/重跑或训练。

## 最终native验收与停止

上述八组已全部完成，执行冻结d878e8683a35ffafab3b6b34d72820368f23692d；质量执行冻结a4e5a9c。source、HF/native、质量与速度数据均按阶段分开，不把bank模式当速度。实际72run/324请求/41472输出全部有效；24组三重复、18组54对输入和配置完整匹配。主协调独立核全部raw的官方35shape/字节/模块数、计量/输入和stage归属，Sol独立逐字段重建CSV、配对和Markdown，均通过。正式native审计工具最终34368c0，汇总产物c026071。

Q/O+down在B1代码+7.14%，但B8通用-12.65%；后者tau由3.564912降至3.175、全局decode步骤51→66。Q/O-only六格仅+0.09%至+1.50%。不宣称量化精度无损或普遍加速，默认none，不合main，完整正负表和范围见FINAL_REPORT.md。

完整同一账10stage全部ok/exit0，累计626.458493959秒，唯一smoke97.718113秒；274个自有PID/starttick身份及三个controller均退出，最终GPU0为15MiB/0%/无compute。证据在evidence/final-01，完整压缩归档在本机D:/tools/codex-runs/dspark-quant-quality-20261105及服务器本批根同名final-evidence-01.tar.gz，约1.22MB。没有后续GPU任务，未触碰其他作业、全局权限或旧PD等任务。
