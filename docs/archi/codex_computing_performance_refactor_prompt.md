# Codex Task — Audit-Driven Computing Performance Refactor and Code Documentation

请在当前 `e_commerce_recsys` 工作区内直接实施本任务，**不是只输出优化方案、诊断报告或新 prompt**。本次交付必须包含真实源代码优化、逐行注释、仓库根目录的详细 `README.md`，以及 `docs/archi/computing_opt_report.md`。先阅读真实代码和审计报告再动手，不要仅凭此任务中的示例假定瓶颈成立。

## 0. 权威资料、优先级和工作区保护

开始前依次阅读，并核实真实存在的文件位置及当前代码版本：

1. `docs/computing_performance_architecture_audit_report.md` —— **本次性能架构审计的主要依据**，逐条提取问题、证据、风险、建议、文件/函数定位；不可虚构尚未阅读的结论。
2. `AGENTS.md` —— Google Python Style、英文注释/docstrings、Notebook/Markdown、Ruff、Pyright、pytest 等规范。
3. 当前有效的 `proposal.md` 或 `docs/proposal.md`、`docs/archi/architecture.md`、`docs/archi/evaluation_protocol.md`、`docs/archi/implementation_decisions.md` 和最新 full/pilot runbook（以实际仓库及新架构为准）。
4. `src/`、`scripts/`、`tests/`、`configs/`、`notebooks/`、`pyproject.toml`、`uv.lock`；尤其是目前真实的 Two-Tower、数据加载、内容向量、召回、特征构造、LambdaMART、Colab 阶段 runner/checkpoint/reload 实现。

若审计报告缺失、无法读取或与实际源码不符，不要编造结论：记录具体路径/差异，完成能够安全执行的审计/实现，明确标记 blocked 项。**最新明确的用户需求及有效架构优先于历史方案/旧 Notebook；审计中涉及更改模型语义的建议必须单独标记，不得暗中实施。**

先记录工作区修改情况与已有产物；保护用户现有改动。不要初始化 Git、commit、push、删除或覆盖 `data/original data/`、`data/text_features/`、`data/image_features/`、历史 runs、模型或 checkpoint；不要执行旧 `docs/achive/` Notebook，也不要重提取已有图文 embeddings。训练、临时产物使用 `data/` 现有约定，文档放在用户指定位置。

## 1. 审计驱动的实施计划和真实基线

在修改核心代码前，建立审计行动矩阵（最终写进 `docs/archi/computing_opt_report.md`）：

- 审计报告的原文问题/章节；对应当前文件、函数、关键行；是否在当前版本复现。
- 分类：功能错误、时间泄露/训练推理不一致、计算瓶颈、I/O、重复计算、内存、GPU 利用率、checkpoint/reload、可维护性。
- 具体修改方式、是否保持算法等价、预计影响的阶段、需要验证的测试；标识未解决/不适用及原因。

对可测路径建立 **before baseline**（使用当前可运行环境中的安全小型 fixture/受限 benchmark），记录：

- Two-Tower：target 读取、past-only history 查询、same-day positive 查询、输入组 batch、原始向量取数、H2D 传输、forward/loss/backward/optimizer、内部验证、完整商品编码、checkpoint 序列化/校验/回存。
- 六路召回和每个 cutoff 的全局统计/ItemCF；候选合并；每个 query 的特征构造；完整 Validation 重载生成；LambdaMART 数据准备、实际 fit、预测。
- 实际输入行数、query 数、候选行数、feature 数、batch 数、wall time、CPU/RSS、PyTorch CUDA allocated/reserved（若有 CUDA）、正确采样的吞吐/step p50/p95。

GPU 低显存占用不等于 GPU 空转；不要猜测 95% idle、保证 20–50x 提升或臆测 Colab 的 CPU 型号。CUDA 异步计时用 CUDA events 或恰当同步的 benchmark，正式热路径不能每行强制 synchronize。不能把 Notebook 包装/Drive 复制时间混成纯训练时间。CPU 和 CUDA 证据分开报告。若只有 Mac CPU，则 CUDA benchmark 标记 **not run**，不可伪造。

## 2. 真正改进执行速度（以审计结论及实测为优先）

在保持可验证的训练/评估等价前提下，**实际修改共享执行代码**，而非只新增 profiler、配置、文档或把 batch size 调大：

1. **Two-Tower 有界批量输入**：消除不必要的逐行 Python/字典/Pandas 装配和同一 epoch/不同 epoch 重复 SQL 查询；将事件历史、完整多正例集合、用户属性和商品映射预索引/列式化，按 batch 获取。保留截止日 `< anchor/event date` 的定义、用户历史顺序/权重、同日多个正例、重复交互语义、固定随机种子和完整恢复状态。不得把全量交易、全部事件展开成无界 Python 列表。
2. **静态商品特征银行**：一次构建 `article_id → row_index`、类别 ID、原始 text/image embedding 和 mask；读取实际维度、校验 ID 顺序与缺失。经实际 GPU 内存预算确认可行时，允许冻结的原始输入表一次驻留 GPU，训练 batch 仅传 ID/row index；否则实现受控分批读取、预取和 pinned CPU memory。不能把随梯度更新的 ID embedding、text/image projection 输出、learned item vectors 或用户历史 pooled learned vectors 缓存成跨-step 常量。
3. **不改变现有损失**：保留现有 multi-positive LogQ/in-batch negative、mask、Q、有效行过滤、batch 边界、优化器、学习率、scheduler、训练期与 epochs；禁止把多正例损失换成单对角线 CE。不得为了提高 GPU 占用默认改成 batch 8192、AMP/TF32/torch.compile 或 GPU 负采样。此类会影响实验语义的改动只能作为单独显式对照实验，默认禁用。
4. **消除特征重复计算**：如审计/源码确认，对冻结的 K=128 visual centers 为每个 article 预计算 cluster 一次；热门榜按 `(cutoff, age_bucket)` 缓存；用户级历史计数、价格/类别偏好和内容相似度在 `(user, cutoff)` 层计算一次并批量映射到候选；复用 Repurchase/Sibling/召回来源信息。缓存键必须包含 cutoff、数据/模型/中心/特征版本，正确失效。不要跨未来时间共享动态历史。
5. **候选与检索**：减少重复 active-item 过滤、ID 字符串转换、排序和单用户循环；采用有界 vectorized/batched 精确检索，保留 score、可用候选、来源排名和稳定 tie-breaking。不能未经消融和同口径召回校验就把精确检索改成近似 ANN。
6. **checkpoint 和阶段 I/O**：区分本地 checkpoint 与 Drive 持久化频率，量化复制、hash 和上传成本；保留最后一个有效远端 checkpoint，状态至少包括模型、optimizer、scheduler、RNG、epoch/step、target-order identity、best epoch。原子写入和哈希校验不可删除；失败和新 Colab runtime 恢复必须有测试。调整回存节奏须明确恢复时最多损失的训练进度。
7. **WALS / ItemCF / LambdaMART**：若审计显示为热点，优化重复 Gram 计算、全局聚合、查询索引、query-group/特征 matrix 构造和列式写入；保持真正的 WALS folding-in、ItemCF 公式和 LambdaMART 的单次 fit。**LambdaMART 仍为既定 49 列 feature schema**（以当前源码及测试核实）、排序目标、负采样与 early stopping 规则，不得将原始图文向量展开为几百列，不得用对各分片多次 `init_model` 冒充单次完整训练。

按“影响最大、实测有效、语义风险最低”的顺序修改；没有实测依据时先加测量，**必须落实可行的代码变更**。对不安全或需重新定义训练目标的审计建议说明原因并保留为独立实验，不可悄悄改变线上/离线语义。

## 3. 强制源代码逐行注释（本次新增的明确用户要求）

**不是只写文件头、类/函数 docstring 或少量段落注释。请对仓库内自有、维护中的 Python 源代码（至少 `src/`、`scripts/`，以及本次新增/修改的 `tests/` 和 Notebook code cells）逐行审查：对每一条非空、有语义的执行语句提供紧邻的英文注释，解释该语句的目的、输入输出、数据/时间约束或必要的行为。** 包括重要的导入依赖、赋值、过滤、SQL、条件、循环、张量变形、设备迁移、缓存、异常、序列化、return 和 side effect。多行表达式允许用紧邻的分层前置注释覆盖每个逻辑步骤；空行、纯闭合括号、无法合法添加注释的语法续行除外。

要求：

- 注释必须专业英文、准确、具体，不允许自动批量插入 `# Set x to x` 等无意义填充，也不能只把 Python 语句机械翻译成英文。
- 解释每条关键语句在系统中的作用，尤其是 **why/how、shape/dtype、point-in-time cutoff、mask、归一化、负例、正例、group、设备、缓存失效、I/O/复杂度、错误保护**；同时保留 Google-style docstring 与公共接口类型注解。
- 新指令的“逐行说明”在本次任务中覆盖 AGENTS.md 中“避免冗余逐行注释”的一般偏好；其他规范仍遵守。若发现字面上的逐行尾注释会破坏代码语法、Ruff 格式或可读性，使用**紧邻前置英文注释**，不可为了注释而改变算法行为。
- 不能只对本次修改的 1–2 个文件注释，最终报告列出覆盖的 Python 文件、尚未覆盖的文件和原因；旧 `docs/achive/`、第三方代码、自动生成文件、`.npy/.parquet/.index`、JSON 配置文件不要强行加入非法注释。
- 修改代码后必须确保注释准确；不要通过压缩复杂语句、合并多行来逃避说明要求。

## 4. 详细重写根目录 `README.md`

生成可供新工程师在不反复打开源文件的情况下理解、维护、调试系统的**专业英文 README**。必须以真实仓库为准，不得按想象虚构入口：

- 项目目标、H&M purchase-only 隐式反馈的边界、六路召回、单一 LambdaMART、阶段图（可用 Mermaid）、训练/推理实际数据流。
- **开发环境**：macOS M1 Max 小内存环境的安装与受限测试、Colab Linux/CUDA 的独立环境、`uv`/锁文件/Python、Ruff/Pyright/pytest、可复制的 CLI 命令、运行配置、日志/产物位置、调试和恢复流程；不要包含真实凭据。
- **真实数据目录**：`data/original data/`、`data/text_features/`、`data/image_features/` 的实际文件和映射约定；Drive 数据复制到 `/content/e_commerce_recsys/data/`；原始 embedding 已预处理且只读。
- **代码树逐文件目录**：必须覆盖当前仓库中每个自有代码文件（至少 `src/`、`scripts/`、`tests/`、`configs/`、现用 notebooks），逐项说明 `relative path → purpose → major APIs/CLI → inputs → outputs → upstream/downstream → responsible stage`，不是只介绍顶层目录。文件过多时可分表按包/阶段列出，但不得遗漏。
- 清楚解释数据 grain（purchase line、user–item、user–date–article、query、candidate row）、训练与推理 I/O、六路具体 quota 和作用、Two-Tower/LogQ、WALS folding-in、LambdaMART 49 特征及 group/label、四个不重叠 folds、完整 Validation/MAP@12、Train-only 负采样和防泄露规则。
- 说明 `smoke`/`pilot`/`full` 的真实差异、显式授权、资源预算、checkpoint/resume、已知限制与故障排查（CPU/GPU、性能、磁盘、重复 run ID、产物身份），给出导航到主要文档的相对链接。
- 长文档要有目录、单一 H1、正确 heading hierarchy、相对链接和语法有效的 Mermaid；真实测试结果与尚未执行的结果分开标示。仅介绍当前真实存在的命令/文件。

## 5. 必交修改报告：`docs/archi/computing_opt_report.md`

报告使用**专业英文**，必须详细、可审查、可复现：

1. Executive summary：实际优化范围、哪些已测有效、哪些未验证。
2. Audit-to-code traceability：审计报告每个发现/建议的编号或标题、状态（implemented/verified/deferred/rejected）、依据和原因。
3. **逐文件修改表**：真实相对路径、函数/类/代码段、修改前问题、具体改法、为何不改变语义、源代码行号或可搜索符号、配套测试及结果。不要只写“optimized pipeline”。
4. Two-Tower 的 before/after 分段 benchmark（数据规模、环境、批量、真实 step 数、数据准备/H2D/GPU/验证/checkpoint、吞吐、p50/p95、RSS/GPU memory）；区别冷启动建缓存与稳态吞吐，不可只用不同硬件或不同样本得出 speedup。
5. 候选构造、ItemCF、特征生成、LambdaMART 的 before/after，明确测得数字、估算、未测量；含资源与磁盘开销、缓存命中与失效策略。
6. Parity/correctness：相同 seeds/输入/配置下的样本、历史 cutoffs、multi-positive mask/Q/loss、梯度/权重容差、每路候选与 ties、49 列特征、group/label、Top-12、MAP@12/指标分母、checkpoint 中断恢复；记录允许的数值容差及原因。
7. 逐行注释覆盖清单；README 文件目录覆盖检查；Ruff/format、Pyright、pytest、受限运行的实际命令、退出码、failed/skipped 项。
8. 真实 Colab/CUDA/full 尚未验证的工作与风险、复现步骤、变更/版本身份；不要宣称未运行的 GPU/full 加速。

文档必须直接落在 **`docs/archi/computing_opt_report.md`**，不是根目录，也不能只贴在 Codex 聊天输出里。

## 6. 测试、资源门禁与验收

- 不改变既定架构：Trend 60、Repurchase 30、Sibling 30、ItemCF 50、WALS 50、Two-Tower 50；一个 LambdaMART；ranker Train F1–F4 与完整 Validation **标签窗口不得重叠**，动态行为/特征一律取 cutoff 之前。
- 采用 reference vs optimized 路径的同口径测试：同历史、标签、训练顺序、batch size、模型初值、seed、设备。不要以更低采样率、更少 epochs、跳过 internal validation、跳过 checkpoint 或修改 loss 来制造速度提升。
- 加入性能回归测试和正确性测试：缓存冷/热命中、数据版本失效、跨 cutoff 泄露、缺失向量、不同商品 ID 映射、负例/多正例、GPU 不可用、资源不足、中断恢复、完整 Validation 的零命中 query。避免 flaky 的“必须快 X 倍”单元测试，性能数字由可重复 benchmark 报告。
- 在本地 M1 Max 只有约 7–8 GB 可用 RAM 的条件下，不准运行真实 full 数据训练、全局 ItemCF 重建或大规模向量复制。只能做合成/受限 smoke 与性能测试；Colab CUDA benchmark 提供明确手工运行入口，用户未授权不得启动真正 full 或最终 Test。
- 必须运行可行的 `uv run --locked pytest -q`、`ruff check`、`ruff format --check`、`pyright`，记下实测退出码。旧 Notebook 的历史问题单独列明，不许顺手重排/改写归档文件，也不许降低测试或关闭警告掩盖错误。
- 使用新 release 和新 run ID；不要改写正在进行或已完成实验的 code/config/input identity，不要因提速牺牲 Train/Inference parity。若更新了 Notebook/CLI，运行相关契约检查并生成匹配 ZIP、SHA256、manifest；本次可将构建包作为交付项，但不得声称云端已跑通。

## 7. 最终回复形式

完成实际代码和文档后，简要报告以下内容，并给出仓库内真实路径：

1. **改了哪些文件/函数**，审计问题如何被逐项解决。
2. **哪些性能变好**，在什么受限环境、什么真实测量口径下变化多少；若没有明显提速则如实报告。
3. **逐行英文注释**覆盖了哪些源文件，未覆盖哪些及原因。
4. 根目录 `README.md` 与 `docs/archi/computing_opt_report.md` 的完整路径。
5. 实际执行的测试、退出码、剩余失败/跳过、CUDA/Colab/full 的 not-run 状态。
6. 下一步 Colab 基准如何运行，但不要代替用户启动 full。

**不要只写计划或 Markdown：本任务的核心验收是源码性能优化 + 真实验证 + 逐行注释 + 详细 README + 精确的逐文件修改报告。**
