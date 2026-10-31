# HRDecoder 逐实验事实档案（截至 2026-09-20）

> **2026-10-02 补充：当前最佳 test mIoU 为44.7809%，配方为旧冠军75%＋历史U0 25%。本轮详情见 [实验补充](D:/dataset/experiment/_全实验档案_20260920/04_本轮实验补充_20261002.md)。以下覆盖计数及旧CSV/JSON保留2026-09-20扫描快照；新增结构化事实单独存于04_本轮实验补充_20261002.json。**


本档案只整理本地原文、脚本、配置、日志、CSV/JSON 与 checkpoint 资产；没有把计划、历史引用、口头数字或推断写成实际运行结果。百分数保留原文百分号口径，小数保留 JSON/配置原值。`未记载` 表示在已扫描材料中没有该字段，`未执行` 表示原文明确没有启动，`无本地证据` 表示数字没有找到可核验的本地原始文件。

## 覆盖范围

已发现并扫描的 HRDecoder 主目录/目录族：

1. `D:\\dataset\\experiment\\HRDecoder\_诊断调研\_20260910`（调查、历史日志与证据归档，未启动新训练）；
2. `D:\\dataset\\experiment\\autodl100`（早期 100 元预算实验，含 A/B、类别路由与集成）；
3. `D:\\dataset\\experiment\\hrdecoder\_plus3`（D 方案及 D 测试补充）；
4. `D:\\dataset\\experiment\\HRDecoder\_MA残差头与集成\_20260915-16`（E/E2/L、TTA、集成矩阵）；
5. `D:\\dataset\\experiment\\HRDecoder\_审查轮\_20260916`（audit\_20260916\_A、H3–H13、冻结/契约/诊断）；
6. `D:\\dataset\\experiment\\HRDecoder\_FusionAware\_20260917`（R1、R5、R6、R6b 及执行/回收目录）；
7. `D:\\dataset\\experiment\\HRDecoder\_FullCompetition\_20260920`（H0/H1/U0/U1 锁定配置、训练核心、U0 本地实际 run、互补性分析）。

另见 `autodl100\\work\\HRDecoder`、`hrdecoder\_plus3\\repo` 等代码快照，以及 `HRDecoder\_诊断调研\_20260910.zip`、`HRDecoder\_FullCompetition\_20260920\_upload.tar.gz`；压缩包/代码快照作为资产索引，不另算训练 run。

## 逐实验事实

### 早期诊断与历史实验（2026-09-10）

|ID|实验/状态|为什么做（原文）|数据、初始化、训练/冻结、损失、采样|推理、阈值、选点|val（EX/HE/SE/MA；mIoU）|test（EX/HE/SE/MA；mIoU）|耗时/资产/完成度|证据|
|-|-|-|-|-|-|-|-|-|
|DGN-HIST-BASE|历史 baseline，引用|建立同划分基准；调查记录说明本轮未重训|454/76/227；seed 42、batch 2、lr 0.005、40k、BN、norm\_eval=False、smooth=1e-5；具体采样未在该汇总行给出|历史协议；最佳 val 选点记录为 32k；阈值未在该表单独记载|39.390；逐类未在该表记载|42.20；逐类未在该表记载|历史资产不在该目录中；本轮调查未执行训练|`HRDecoder\_诊断调研\_20260910\\调研记录与证据分级.md:15-19,26-30`；`HRDecoder\_诊断调研\_20260910\\HRDecoder\_诊断与押注建议.md:55-64`|
|DGN-CE-DICE|CE+Dice 历史训练，引用|诊断损失/验证选点；不是本轮新训练|454/76/227；seed 42、batch 2、lr 0.005、40k；CE+Dice；冻结范围未记载|最佳 val @12k；测试加载 12k；阈值未在该汇总表单独记载|39.970；逐类未在该表记载|39.43；逐类未在该表记载|原始训练 JSON/日志在历史仓库，调查本身未启动|`HRDecoder\_诊断调研\_20260910\\调研记录与证据分级.md:17,26-30`；`HRDecoder\_诊断调研\_20260910\\HRDecoder\_诊断与押注建议.md:55-64`|
|DGN-E4|E4 历史训练，引用|诊断细节分支/高分辨率路径|454/76/227；seed 42、batch 2、lr 0.005、40k；E4；backbone BN 持续更新（norm\_eval=False）；细节 scale 零初始化；冻结范围未记载|最佳 val @36k；测试加载 36k；部署路径与融合细节见审查轮诊断|38.766；逐类未在该表记载|41.63；逐类未在该表记载|历史 checkpoint/日志；调查本身未启动|`HRDecoder\_诊断调研\_20260910\\调研记录与证据分级.md:17,26-30`；`HRDecoder\_诊断与押注建议.md:33-41,55-70`|
|DGN-E2E3|E2E3 联合历史训练，引用|诊断 E2/E3 联合干预；原文说明不能独立归因|454/76/227；seed/预算沿历史配置；联合 E2+E3；具体冻结/采样未记载|最佳 val @20k；测试加载 20k|39.206；逐类未在该表记载|39.98；逐类未在该表记载|历史日志；调查本身未启动|`HRDecoder\_诊断与押注建议.md:55-64`|
|DGN-E4-CD|E4+CE+Dice 历史训练，引用|诊断结构与损失组合|454/76/227；历史配置；恢复训练链；冻结/采样未记载|最佳 val @32k；测试加载 32k|39.871；逐类未在该表记载|41.08；逐类未在该表记载|历史日志；调查本身未启动|`HRDecoder\_诊断与押注建议.md:55-66`|
|DGN-E4V2|E4v2 历史训练，引用|诊断 E4 辅助监督版本|454/76/227；历史配置；辅助损失权重与完整冻结范围未在此表记载|最佳 val @36k；测试加载 36k|38.667；逐类未在该表记载|40.06；逐类未在该表记载|历史日志；调查本身未启动|`HRDecoder\_诊断与押注建议.md:55-66`|

### autodl100 早期实验

|ID|实验/状态|为什么做（原文）|数据、初始化、训练/冻结、损失、采样|推理、阈值、选点|val（EX/HE/SE/MA；mIoU）|test（EX/HE/SE/MA；mIoU）|耗时/资产/完成度|证据|
|-|-|-|-|-|-|-|-|-|
|A|语义条件 MA 残差分支，停止|预算内测试“语义引导 MA 残差”；从旧 E4 最佳起步|454/76/227；从 E4@36000；保留 E4 分支并加语义引导 MA 残差；MA 正像素/难负样本辅助损失 0.02；BN running stats 冻结、affine 可训练；低 lr；8k 步实际完成|val 选 @1000；未做 test|38.632@1000；最终 38.265；逐类未在结论表记载|未测试|2026-09-12 00:11 开始；每步约 0.53 s；A 8000 步完成；`results/trial\_a/selected\_weights\_only.pth`，SHA256 `82689516488846d3129f8576fa0533bad954cfd546f8c2187f64ca9c8fb903b0`|`autodl100\\实施记录.md:3,11-14`；`autodl100\\最终实验结论.md:5-14`；`autodl100\\docs 不存在，资产路径见` `autodl100\\results\\trial\_a`|
|B|低 lr/冻结 BN/MA 辅助损失对照，完成|与 A 对照，区分新增 MA 分支是否必要|同 A 起点 E4@36000；无新分支；MA 辅助损失 0.02；BN running stats 冻结、affine 可训练；低 lr；8k 步|val 锁定 @3000；test 固定 sigmoid 0.5、无 TTA|39.205@3000；逐类未在结论表记载|41.92：EX52.65/HE44.37/SE48.91/MA21.74|B 8000 步完成；`autodl100\\results\\trial\_b\\selected\_weights\_only.pth`，263,479,238 bytes；SHA256 `0b3cd72e740fc5c69242994c12cfd71a49ac892aff37d09c50bb9fb6931a8626`|`autodl100\\实施记录.md:3,15-19`；`autodl100\\最终实验结论.md:10-16`|
|ROUTE|按 validation 选类别路由，完成探索|验证不同类别由不同模型负责|E4/B；按类选择，未重训；具体冻结/损失/采样沿模型历史设置|validation 选择模型；test 不反向调参|39.90；逐类未记载|41.82：EX52.65/HE41.61/SE48.91/MA24.13|已评估；保留为探索性失败|`autodl100\\实施记录.md:21`；`autodl100\\最终实验结论.md:12-24`|
|ENS-E4-B|固定 50/50 原图概率集成，完成|减少类别路由自由度、检查模型预测互补|E4@36000 + B@3000；不重训；等权概率平均；454/76/227|val 先过 39.40 门槛；test 固定 sigmoid 0.5、无 TTA、背景不计入|39.54；逐类未记载|42.35：EX52.29/HE43.95/SE49.93/MA23.22|2026-09-11 完成；E4 权重 `autodl100\\results\\e4\_reference\_weights.pth`；B 副本同上；两权重哈希在 `ensemble\_weights\_verified.json`|`autodl100\\最终实验结论.md:3-16,24-30,42-44`；`autodl100\\复现说明.md:1-17`|

### hrdecoder\_plus3：D 方案

|ID|实验/状态|为什么做（原文）|数据、初始化、训练/冻结、损失、采样|推理、阈值、选点|val（EX/HE/SE/MA；mIoU）|test（EX/HE/SE/MA；mIoU）|耗时/资产/完成度|证据|
|-|-|-|-|-|-|-|-|-|
|D-TRAIN|D student-only 正式训练，完成|测试真实高分辨率局部编码，目标验证提升；连续两次 42.4 才触发 test|454/76/227；两路以 B@3000 初始化；仅 Dice；BN running stats 冻结、affine 可训练；50% 随机 + 50% 图内阳性类均匀中心采样；SGD lr 0.0005，poly power=1 至 0.00005，warmup 200；20,000 步|真实 HRNet 裁块 + 冻结 B 全局概率各 0.5；统一缩放 2048²；1024 crop、stride 768；重叠 logits 均值后回原图 sigmoid；阈值 0.5；无 TTA；val 最优 @2000|44.04/44.73/55.08/22.62；41.615@2000；最终 40.476|本报告明确本轮未做 test（后有 D\_TEST 补充）|正式训练约 0.34 s/步，约 2 h14；`results/D\_0913\_retry/backup\_student\_2000.pth`，263,419,124 bytes，SHA256 `5fe18717b62d8fc3ab4846a50c046588e2af05abbbfdb3a958f4794018b319ea`；训练完成|`hrdecoder\_plus3\\本轮实验报告\_20260913.md:5-19,23-48,50-80`|
|D-TEST|D@2000 测试补充，完成|用户明确要求补做 test，不再让验证门槛阻止评估|使用 D@2000；训练设置继承 D-TRAIN；无重训|227 test；阈值 0.5；两路概率等权平均；无 TTA；不改 checkpoint/阈值/融合|—|53.68/41.67/50.23/24.73；42.58%|`results/D\_test/d\_test.log`；平台最终关机与新增费用在补充报告中记为尚需核实|`hrdecoder\_plus3\\D方案\_测试集补充报告.md:1-16`|
|L-PLAN|Dice+0.1 Lovasz，准备但未执行|作为 D 的单变量损失对照|与 D 完全一致，唯一计划差异 Lovasz 权重 0.1；未启动训练|计划沿用 D 推理协议；未选点|未执行|未执行|`repo/configs/lesion/exp/hrplus\_patch\_lovasz.py`；原文明确未启动|`hrdecoder\_plus3\\本轮实验报告\_20260913.md:17-20,58-72`|

### MA 残差头与集成（2026-09-15—16）

基础协议：train/val/test=454/76/227；test 227 张；四类 IoU 不含背景；sigmoid 阈值 0.5；报告评估器复现 D=42.576、B=41.918、E4=41.626，偏差小于 0.005（`最终报告\_两轮实验汇总.md:1-5`）。

|ID|实验/状态|为什么做（原文）|数据、初始化、训练/冻结、损失、采样|推理、阈值、选点|val（EX/HE/SE/MA；mIoU）|test（EX/HE/SE/MA；mIoU）|耗时/资产/完成度|证据|
|-|-|-|-|-|-|-|-|-|
|E|B 基准+MA 残差头，完成|在 B 起点测试语义条件 MA 残差头|B 起点；仅新头训练；残差零初始化；头细节见审查轮 H4 结构；其余主干/参考冻结；训练步数文档记录 E@1500|无 TTA 单模型；阈值 0.5；val/test 由原始 eval JSON|40.255@E1500；EX44.98/HE41.39/SE51.84/MA22.81（汇总表同时把该行作为 B 起点记录，须按来源分别核对）|42.559：EX53.64/HE43.83/SE47.63/MA25.13|`HRDecoder\_MA残差头与集成\_20260915-16\\02\_原始数据\\eval\_json\\E\_0916\_\_eval\_TEST\_E\_head.json`；本地资产 `HRDecoder\_FusionAware\_20260917\\assets\\E\_1500.pth`，527,412,743 bytes，SHA256 `a18ff92b4da24d870554f17856983d4be042386cacbcaba0d4e13f3f248ed5e5`|`最终报告\_两轮实验汇总.md:31-32,40-46`；`champion\_asset\_inventory\_workbuddy.md:33-35,68,96`|
|E2|D 基准+MA 残差头，完成|检验把 MA 残差头加在 D 基准上的效果|D 起点；MA 头训练；主干/参考冻结；具体步数未在汇总表给出|无 TTA 单模型；阈值 0.5|未在汇总表记载|42.544：EX53.68/HE41.67/SE50.23/MA24.60|`02\_原始数据\\eval\_json\\E2\_0916\_\_eval\_TEST\_E2\_head.json`；是否产生独立 checkpoint 未在汇总表记载|`最终报告\_两轮实验汇总.md:31-35`|
|L|Dice+0.1 Lovasz，对照完成|测试 Lovasz 作为 D 的单变量损失对照|D 配方，仅损失改 Dice+0.1 Lovasz；最佳 checkpoint @6000（文档）；其余冻结/采样沿 D|无 TTA/翻转 TTA 两种；阈值 0.5|41.056（单变量报告）；逐类未在该表记载|无 TTA 41.777；翻转 TTA 41.645；逐类未在该表记载|`05\_日志\\L\_0915\_\_l\_train.log`；`eval\_json\\L\_0915\_\_eval\_L\_notta\_test.json` / `...L\_flip\_test.json`|`第一轮\_L对照与TTA结果.md:14-27,49-54,71-78`|
|ENS-MATRIX|D/E/B/E4/L 多模型矩阵，完成|测试翻转 TTA、模型数量、固定/加权集成|成员 checkpoint：D@2000、L@6000、B@3000、E4@36000；全冻结，仅推理融合；D 权重 1/2/3；均匀权重见配置|概率级融合；翻转 TTA；阈值 0.5；按 val/预注册方案记录，test 表无再调参|汇总 val 表仅列 D=41.615、E=40.255、B 起点=38.937 等诊断值|见下一列，各配置 mIoU：D×3+B+E4=43.149；D×2+B+E4=43.122；D+B+E4=43.005；D+L+B+E4=42.971；D+L=42.923；D+E+flip+multiscale=42.805；D multiscale=42.797；D flip=42.717|22 个 test 条目完整列于报告；checkpoint 加载缺失/多余均为 0（D/L/B/E4）|`最终报告\_两轮实验汇总.md:11-36,38-46,58-78`；`第一轮\_L对照与TTA结果.md:12-27,65-78`|

### 审查轮 audit\_20260916\_A

|ID|实验/状态|为什么做（原文）|数据、初始化、训练/冻结、损失、采样|推理、阈值、选点|val（EX/HE/SE/MA；mIoU）|test（EX/HE/SE/MA；mIoU）|耗时/资产/完成度|证据|
|-|-|-|-|-|-|-|-|-|
|AUDIT-PHASE|证据/契约审查，完成部分阶段|先证明加载、冻结、推理融合与指标协议，再决定是否继续训练；原文列 Phase A PASS、Phase B PARTIAL、C1/C3/C4 完成，C2/E/F 初始未做|未启动两臂 pilot；检查 454/76/227、split identity、255 标签、checkpoint API、几何/TTA、指标 oracle；未训练新模型|固定严格 `>`、阈值 0.5、数据集累计 TP/FP/FN、背景不计入|student 0.393351/EX0.402364/HE0.430491/SE0.512710/MA0.227840；reference 0.392050/0.442479/0.415482/0.516243/0.193999；blend 0.416148/0.440374/0.447273/0.550773/0.226174|未做新方案比较；D 回归 test 42.576 仅为复现|RTX4090 约 20:11–20:40（<1 h）；审计资产 `audit\_runs\\audit\_20260916\_A`|`HRDecoder\_审查轮\_20260916\\audit\_runs\\audit\_20260916\_A\\final\_report.md:9-28,43-67`；`diagnostics\\val\_blend\_decomposition.json:1-103`|
|H3|MA 融合权重扫描，完成|检验 MA 通道融合权重是否可改善|冻结 BASE\_0/学生/参考；w∈{0.5,0.6,0.7,0.8,1.0}；无训练|val 扫描，严格阈值 0.5|MA 峰值增量 +0.0047；mIoU +0.0012；逐类绝对值未单列|未跑新 test|`experiments`/`diagnostics` 下 H3 相关扫描产物；完成|`FINAL\_本轮最终报告.md:105-118`|
|H4|后融合 MA 校正头训练，完成|依据融合对 MA 微负，测试只修正 MA 的后融合头|D@2000 起点；输入 7ch=\[RGB, fused MA logit, student MA logit, reference MA logit, student-reference 概率差]；3×3(7→16)-ReLU-3×3(16→16)-ReLU-1×1(16→1)；零初始化；仅 3,361 参数；Adam lr 1e-3 poly；BCE 0.5 + Dice 0.5；主干全冻结；2000 步跑完，best@1600|crop 监督；部署整图融合；严格阈值 0.5；best 以 val corrected@1600 选|corrected MA 0.252207；student 0.227840；fused 0.226174；mIoU（val 头对应 H4）0.426686；逐类非 MA 恒取 BASE\_0 0.457705/0.437552/0.551806|H4 单独 test confirmation：fused mIoU 0.425764，corrected 0.434287；corrected EX/HE/SE/MA=0.536760/0.416678/0.502328/0.281383|`h4\_best.pth` 15,447 bytes；SHA256 `14757e02de49f6f065d4ef82cb8b1c4ee8a6a7f3f66cb1867276993950f80c50`；训练 history elapsed 676.1 s@2000|`experiments\\h4\_post\_fusion\_corrector.json:1-75`；`experiments\\h4\_test\_confirmation.json:1-75`；`FINAL\_本轮最终报告.md:32-45`|
|H5/H7|H4 接完整冠军集成并独立 test 复现，完成|验证 MA 头接入完整 D/E/B/E4 集成后是否保持增益|D@2000×3、E@1500×2、B@3000×1、E4@36000×1；四主干冻结；H4 head@1600；无外部数据/重训|概率加权 3:2:1:1；horizontal flip TTA；严格 p>0.5；H5/H7 test 各独立执行|champion val 42.6686（p0）或 42.6694（后续 fp32 remote）；逐类 45.7715/43.7583/55.1807/25.9672（remote 原始）|fused 无头 0.432289/EX0.538611/HE0.445130/SE0.498161/MA0.247255；final 0.442115/EX0.538611/HE0.445130/SE0.498161/MA0.286558；两次逐位相同|`h5\_test.json`、`h7\_test.json`；两次 test 均 n=227；完成|`experiments\\h5\_test.json:1-49`；`experiments\\h7\_test.json:1-95`；`champion\_asset\_inventory\_workbuddy.md:31-48,103-144`|
|H6|HE/SE 校正头，完成|把 H4 方案迁移到非 MA 类|冻结冠军/输入概率；头结构沿 H4；仅目标类头训练；损失细节未在汇总行单独记载|val 选点；严格阈值 0.5|HE best 0.4356 < fused 0.4473；SE best 0.5144 < fused 0.5508|未作为新冠军跑 test|审查轮产物；完成|`FINAL\_本轮最终报告.md:105-118`|
|H8|延长 H4 头训练，完成|检验更长训练/更大学习率|H4 头；6000 步、lr 2e-3；主干冻结|val 选点，随后 test 观察|val MA 0.2569|test MA 0.2858；完整 mIoU 未在该行单列|审查轮产物；完成|`FINAL\_本轮最终报告.md:105-118`|
|H9|BCE 主导 HE 头，完成|排除损失配比噪声|HE 校正头；BCE/Dice=0.9/0.1；冻结其余|val 选点；阈值 0.5|HE best 0.4382 < fused 0.4473|未跑新 test|审查轮产物；完成|`FINAL\_本轮最终报告.md:105-118`|
|H11|BCE 主导 MA 头，完成|对 H4 改损失配比|MA 头；BCE 主导；其余冻结|val 选点；阈值 0.5|MA 0.2510 < H4 0.2522|未跑新 test|审查轮产物；完成|`FINAL\_本轮最终报告.md:105-118`|
|H12/H13|整图级监督及 test 确认，完成|针对训练 crop 与部署整图/融合监督不一致进行验证|H12 用整图级监督；其他训练细节以脚本/报告为准；H13 为 test confirm|val 选点；阈值 0.5；H13 test|H12 val MA 0.2565|H12 test MA 0.2803；H13 full mIoU 44.05 < 44.21|审查轮产物；完成|`FINAL\_本轮最终报告.md:105-118`|
|P5|多种子 H4 头复核，完成|检查头配方的 seed 散布|seed 1/2/3；头训练，其余冻结；细节损失沿 H4|val 选点|MA val 0.2510/0.2537/0.2514|未做 test|3 个 seed；完成|`FINAL\_本轮最终报告.md:119-131`|

### FusionAware（2026-09-17—19）

|ID|实验/状态|为什么做（原文）|数据、初始化、训练/冻结、损失、采样|推理、阈值、选点|val（EX/HE/SE/MA；mIoU）|test（EX/HE/SE/MA；mIoU）|耗时/资产/完成度|证据|
|-|-|-|-|-|-|-|-|-|
|R1-A|融合感知受控对照 A，完成|比较纯 student 监督与将 frozen B 概率加入训练目标|454/76/227；D2000 初始化；Dice(student)；训练 4000 updates；参考 B frozen；采样与协议由 runner 固定|val 每 1000；阈值 0.5；未过 gate 不 test|it1000/2000/3000/4000=0.4152/0.4162/0.4167/0.4131；逐类未单列|未测试|R1 09-17 23:48→03:25；完成|`HRDecoder\_FusionAware\_20260917\\HRDecoder-DDR-阶段成果报告-20260917至0918.md:29-39`|
|R1-B|融合感知臂 B，完成|测试参考模型概率是否应进入学生训练目标|同 R1-A；loss=0.5 Dice(student)+0.5 Dice(0.5 student+0.5 ref)；4000 updates；reference frozen|val 每 1000；gate 要求 B≥A+0.003；未过 gate 不 test|0.4162/0.4169/0.4175/0.4142；逐类未单列|未测试|同 R1；完成；参考概率缓存 454 张构建|`HRDecoder\_FusionAware\_20260917\\HRDecoder-DDR-阶段成果报告-20260917至0918.md:29-39`|
|R5-E0|第 6 成员集成扩展，完成|检查 R1 学生加入冠军集成的边际收益|冠军五成员 + R1 BNEW@3000/@4000；w=1/2；不重训|val 扫描；G0=冠军+0.002；未过不 test|champion 0.4267；+BNEW@3000 w1 0.4273；w2 0.4278；@4000 w1 0.4272|未测试|09-18 09:01–09:17；完成|`ROUND5\_RESULT\_20260918.md:22-34`|
|R5-E1-V1|纯融合监督 w=1，完成|强化融合监督，寻找可传导的学生改进|D 起点；纯融合 Dice，w=1.0；T=3000；seed42；其余 runner 固定|val @1000/2000/3000；G1≥0.4185|0.4170/0.4181/**0.4187**；逐类未单列|未测试|09-18 09:18–11:16；完成|`ROUND5\_RESULT\_20260918.md:36-46`|
|R5-E1-V2|课程融合监督，完成|比较 0→0.5→1.0 课程与直接纯融合|同 V1；w 课程 0→0.5→1.0；T=3000；seed42|val @1000/2000/3000；G1|0.4152/0.4166/0.4181；未 test|未测试|同 E1；完成|`ROUND5\_RESULT\_20260918.md:36-46`|
|R5-E2|V1 seed43 方向复核，完成|对胜出 V1 换 seed 复核|V1 配方；seed43；T=3000|val @1000/2000/3000；未 test|0.4176/0.4179/0.4178|未测试|11:20–11:59；完成|`ROUND5\_RESULT\_20260918.md:47-53`|
|R5-E3|V1 学生加入冠军，完成|检查学生改进能否传导到集成|冠军 + V1(s42@3000) w1/w2；无重训新成员|val 扫描；G3=冠军+0.002；未 test|w1 0.4273，w2 同量级，约 +0.0006|未测试|11:59–12:11；完成|`ROUND5\_RESULT\_20260918.md:55-62`|
|R6-P0|冠军复现/错误画像，完成|先固定可比基线并量化类级错误|冠军 D×3+E×2+B+E4+H4；全冻结；计数审计|val/test 严格阈值 0.5、翻转 TTA；逐图逐类累计|0.4267：EX0.4577/HE0.4376/SE0.5518/MA0.2597|0.4421：EX0.5386/HE0.4451/SE0.4982/MA0.2866|13:22–17:19 窗口；两次复现逐位一致|`HRDecoder-DDR-阶段成果报告-20260917至0918.md:56-60,20-25`|
|R6-P1-HE|H4 式头迁移到 HE，拒绝|预注册按非 MA 类中 val IoU 最低类攻击（HE）|HE 头沿 H4；主干/集成冻结；训练曲线多点评估|val 选点；gate 相对基线；未 test|best 0.4360 vs baseline 0.4376，Δ−0.0015|未测试|EX 分支缓存建到 50/76 后预算到期，跳过；HE 完成|`HRDecoder-DDR-阶段成果报告-20260917至0918.md:62-70`；`ROUND6\_RESULT\_20260918.md:11-18`|
|R6-P2-M1|MA native-res 专家，拒绝|测试从 RGB 从零学习 MA 专家的可行性|512² patch；Dice+0.5 BCE；训练 3000；冠军概率融合扫描 α；细节冻结范围以专家脚本为准|patch IoU 与 α∈{0.3,0.5,0.7,1.0}；gate MA val +0.010；未 test|patch IoU=0/0/0.0052/0.0220/0.0100/0.0137；α=0.3 时 MA0.2438 vs 冠军0.2597|未测试|13:22–17:19；完成；专家 pth 在 pulled/归档|`HRDecoder-DDR-阶段成果报告-20260917至0918.md:71-80`|
|R6-P3|终局对照，完成|在 P1/P2 均拒后记录冠军 test|不训练新模型；冠军保持|固定冠军推理|0.4267（复算）|0.4421，与 P0 一致|`no\_gain`；完成|`HRDecoder-DDR-阶段成果报告-20260917至0918.md:82-83`|
|R6b|256 patch + Tversky MA 专家，拒绝|依据 P2 退化诊断做授权配方修订|256² patch；Tversky(0.3,0.7)；2000 steps；其余 U-Net/patch/α/gate 不变|α∈{0.3,0.5,0.7,1.0}；gate +0.010；未 test|patch IoU .0081/.0045/.0015/.0563/.0038/.0066/.0432/.0402；α=.3 MA .2431，Δ−.0166|未测试|17:26–18:10；完成；`status=rejected`|`ROUND6\_RESULT\_20260918.md:42-69`|
|FA-MAIN-PLAN|2026-09-19 主方案，计划未执行|旧版被替代；执行优先级改为核清初始化、受控 C/S/P、胜出方向第二 seed、再决定同域预训练/HACDR|计划中的 H0/H1/U0/U1 等候后续；具体实际训练未启动|计划仅 val 锁候选后一次 test；未执行|未执行|未执行|`EXECUTION\_PLAN\_CURRENT.md` 明确唯一主方案；无 GPU 实验记录|`HRDecoder\_FusionAware\_20260917\\EXECUTION\_PLAN\_CURRENT.md:1-16`；`NEXT\_ROUND\_PLAN\_20260919\_REVISED.md:1-14`|

### FullCompetition（2026-09-20）

配置共同事实：固定 DDR 454/76/227，输入 FOV 2048、train crop 1024；四通道 EX/HE/SE/MA；整图累计 TP/FP/FN；严格 p>0.5；AdamW、warmup+cosine、有效 batch 8、ce+dice、smooth 1e-5、valid=0 整类屏蔽。H0/H1 使用 HRNet-W48 适配器并冻结 backbone BN running stats、affine 可训练；U0/U1 为 ConvNeXt-S+U-Net、LayerNorm。具体配置字段见各 YAML。

|ID|实验/状态|为什么做（原文）|数据、初始化、训练/冻结、损失、采样|推理、阈值、选点|val（EX/HE/SE/MA；mIoU）|test（EX/HE/SE/MA；mIoU）|耗时/资产/完成度|证据|
|-|-|-|-|-|-|-|-|-|
|H0|HRDecoder-W48 DDR-only 对照，locked-intent/未正式训练|回答“HRDecoder 的可比较完整训练基准”|seed42；12,000 updates；effective batch8；backbone lr5e-5、decoder lr2e-4；ImageNet 初值；全 stage 开放；sampling random\_plus\_foreground\_mixed；BN stats 冻结、affine trainable；ce+dice masked；micro\_batch/grad\_accum/gradient\_checkpointing/权重均 REQUIRED|整图/滑窗协议；阈值0.5；仅 val 选点；test 只允许 full lock 后一次|未执行；用户给定 H0@6000=36.9225（EX41.88/HE43.52/SE41.45/MA20.84）本地无原始出处，不能当实际本地 run|未执行|`configs/H0\_hrdecoder\_ddr\_only.yaml` status locked-intent；formal preflight 仍有占位符；H 臂本机依赖 blocked|`configs\\H0\_hrdecoder\_ddr\_only.yaml:1-105`；`docs\\TRAINING\_IMPLEMENTATION\_STATUS.md:8-15,100-130`；`docs\\champion\_asset\_inventory\_workbuddy.md:155-158`|
|H1|HRDecoder-W48 外部真实病灶预训练→DDR，locked-intent/未训练|回答“真实病灶预训练是否有用”|外部 pretrain 8,000 updates；external train-only manifest、dev\_ratio .1、source-balanced；seed42 DDR 12,000 updates；其余 DDR 日程与 H0 逐字段相同；ImageNet 初值；微批/累积/外部 checkpoint/manifest 全 REQUIRED|仅 val 选点；test full lock 后一次|未执行|未执行|status locked-intent；未联网/未下载权重；无外部 manifest/checkpoint|`configs\\H1\_hrdecoder\_ext\_pretrain.yaml:1-129`；`docs\\TRAINING\_IMPLEMENTATION\_STATUS.md:3-15`|
|U0-REAL|ConvNeXt-S + U-Net DDR-only，本地真实训练，完成|回答“异构特征和解码路径能否提供增量”|seed42；12,000 updates；runtime micro\_batch4、grad\_accum2、effective batch8、workers4；ConvNeXt ImageNet 字段在配置仍有 REQUIRED 但本地 builtin 模型实际训练；LayerNorm；random\_plus\_foreground\_mixed；ce+dice masked；AdamW；无外部预训练；不含 detail branch|整图 formal val；严格 p>0.5；eval @3000/6000/9000/12000；best val @12000|43.6760/38.4266/50.9362/20.7468；mIoU 0.3844641257662971（38.4464%）；@6000 mIoU 0.3500054425872542，@9000 0.37435898404454215|未执行|status complete；elapsed 13,758 s；best update12000；`runs/ckpt\_best.pth` 636,326,835 bytes；SHA256 `98d98a5d02dc2615062c3c31cd15d061daf3d7ab41cb8f25cd90b062cd53bb34`; `eval\_history.csv` 保存四个 val 点|`runs\\completion\_summary.json:1-40`；`runs\\eval\_history.csv:1-5`；`configs\\U0\_convnext\_unet\_ddr\_only.yaml:1-124`|
|U1|ConvNeXt-S + U-Net 外部预训练→DDR，locked-intent/未训练|回答“外部预训练对异构 U 臂是否有用”|external pretrain 8,000 updates、train-only manifest、dev\_ratio .1；DDR seed42 12,000 updates；与 U0 逐字段锁定；external checkpoint/manifest/weights/micro-batch 等 REQUIRED|仅 val 选点；test full lock 后一次|未执行|未执行|status locked-intent；U1 readiness/calibration blockers 未清空|`configs\\U1\_convnext\_unet\_ext\_pretrain.yaml:1-124`；`docs\\READINESS\_RECHECK.json:5-108`|
|FC-EXPERT-COMP|expert complementarity 分析，分析执行完成；不是新训练|检查 fullmodel/HRPlus 专家逐像素互补性与面积分桶|使用现有冠军/候选输出契约与 shared GT；未训练、未生成新 checkpoint；分析脚本导出 masks/contract|依据现有评估契约；分析对象为 val/候选输出；未使用 test 选点|数值汇总见 `analysis\_outputs\\expert\_complementarity.json`；未在本档案混入为新模型 val/test IoU|未执行新 test|`analysis\_results\\expert\_complementarity\_20260920`，含 per-class、per-image、area buckets CSV/JSON/MD、contract manifests|`analysis\_results\\expert\_complementarity\_20260920\\EXECUTION\_REPORT\_20260920.md`；`COLLECTION\_MANIFEST.json`；`analysis\_outputs\\EXPERT\_COMPLEMENTARITY.md`|

## 冠军资产、融合与哈希记录

以下是本地资产盘点中可逐位核对的 checkpoint；路径均相对于 `D:\\dataset\\experiment` 的原文路径，哈希为 SHA256：

|代号|绝对路径|bytes|SHA256|结构/用途|
|-|-|-:|-|-|
|D|`D:\\dataset\\experiment\\hrdecoder\_plus3\\results\\D\_0913\_retry\\backup\_student\_2000.pth`|263,419,124|`5fe18717b62d8fc3ab4846a50c046588e2af05abbbfdb3a958f4794018b319ea`|student；含 detail\_scale/detail\_head|
|B|`D:\\dataset\\experiment\\hrdecoder\_plus3\\repo\\weights\\b\_weights.pth`|263,479,238|`0b3cd72e740fc5c69242994c12cfd71a49ac892aff37d09c50bb9fb6931a8626`|reference/B；`autodl100\\results\\trial\_b\\selected\_weights\_only.pth` 副本逐位相同|
|E|`D:\\dataset\\experiment\\HRDecoder\_FusionAware\_20260917\\assets\\E\_1500.pth`|527,412,743|`a18ff92b4da24d870554f17856983d4be042386cacbcaba0d4e13f3f248ed5e5`|student+reference+MA refiner，含 optimizer|
|E4|`D:\\dataset\\experiment\\autodl100\\results\\e4\_reference\_weights.pth`|263,477,167|`c5a3d0f0b990d2430e545146745bc809538bbe764979baf54f62b450785268e0`|E4 reference|
|H4|`D:\\dataset\\experiment\\HRDecoder\_审查轮\_20260916\\audit\_runs\\audit\_20260916\_A\\h4\_best.pth`|15,447|`14757e02de49f6f065d4ef82cb8b1c4ee8a6a7f3f66cb1867276993950f80c50`|3,361-param MA correction head, best step1600|
|A|`D:\\dataset\\experiment\\autodl100\\results\\trial\_a\\selected\_weights\_only.pth`|263,709,987|`82689516488846d3129f8576fa0533bad954cfd546f8c2187f64ca9c8fb903b0`|A 非冠军实验|
|C|`D:\\dataset\\experiment\\HRDecoder\_FusionAware\_20260917\\execution\_20260919\\remote\_recovery\\20260920\\runs\\cs4000\_v2\\C\_step\_2000.pth`|718,620,759|`119793766067d872ecc8d420cda4f2ed36d359480ee4581f85d3cd37b74f928e`|回收资产|
|S|`D:\\dataset\\experiment\\HRDecoder\_FusionAware\_20260917\\execution\_20260919\\remote\_recovery\\20260920\\runs\\cs4000\_v2\\S\_step\_4000.pth`|718,620,823|`abb2f078e57fed2444b74f2d81d3aa22049fbc17ea5d7fc4e8fd5f1a94c52179`|回收资产|
|U0-best|`D:\\dataset\\experiment\\HRDecoder\_FullCompetition\_20260920\\runs\\ckpt\_best.pth`|636,326,835|`98d98a5d02dc2615062c3c31cd15d061daf3d7ab41cb8f25cd90b062cd53bb34`|U0 @ update12000|

冠军融合事实：D:E:B:E4 权重为 3:2:1:1，概率加权平均（不是 logit 平均），开启 horizontal-flip TTA；H4 在融合 MA 上执行 `sigmoid(logit(P\_fused\_MA)+6\*tanh(head\_7ch))`；输入 7ch 为 RGB、fused/student/reference MA logits、student-reference 概率差；主干冻结。来源：`HRDecoder\_FullCompetition\_20260920\\docs\\champion\_asset\_inventory\_workbuddy.md:25-48`。

## 未解析项与明确边界

* `H0@6000 = 41.88/43.52/41.45/20.84`（mIoU 36.9225）在本地全域没有逐字或近似原始出处；登记为外部/口头数字，不能当本地实际训练结果。来源：`champion\_asset\_inventory\_workbuddy.md:155-158`。
* FullCompetition H0/H1/U1 配置仍有 `REQUIRED/TBD`；正式 preflight 记录 60 个 blocker；不能把 locked-intent 当已训练。来源：`docs\\IMPLEMENTATION\_STATUS.md:14-27`、`docs\\REAL\_PREFLIGHT\_RECHECK.json`。
* FusionAware 服务器概率缓存机制在代码中存在，但本地没有 `.npy/.npz/.pt/.ckpt` 概率缓存；回收清单不含概率图。来源：`champion\_asset\_inventory\_workbuddy.md:162-180`。
* `HRDecoder\_审查轮\_20260916\\audit\_runs\\audit\_20260916\_A\\experiments\\experiment\_decision.md` 为空，未从该文件补写实验结论。
* FullCompetition 的 test 四类 IoU 对 H0/H1/U1/U0 均未执行；U0 仅有 val 训练曲线。
* 原始日志/压缩包/代码快照中大量缓存、`.pyc`、合成测试资产未逐个列为 run；它们已计入目录覆盖，但未伪造成训练实验。

## 覆盖计数

本档案登记 **7 个主目录族、45 个实验/阶段记录**（其中包含计划/审计/资产分析记录，实际训练或推理 run 少于该数）。明确未完成或未执行项：D-TRAIN 初始报告阶段 test、L-PLAN、H0、H1、U1、R1/R5/R6 未过 gate 的 test、FusionAware 主方案计划。所有来源资产均保持原样，未修改原资料。



<!-- HRD_APPEND_20261002 -->
## 2026-10-02 追加事实（原45条统计保持2026-09-20快照）

下面9条为本轮执行/复用/融合评价记录，不把未触发扩展计作已运行实验。U1/H1编号属于2026-10-02这一轮，不能与FullCompetition旧同名计划混用。

|ID|实际执行|选点/结果|test|耗时/资产|
|---|---|---|---|---|
|READY-CHAMP-REPRO|旧冠军FP32验证复现，未重训|76图val42.6691%，保留D/E/B/E4 3:2:1:1、H4及翻转|在唯一锁定test中同次计算44.2110%|原概率导出和原锁保留|
|READY-U0-REUSE|历史ConvNeXt-S＋U-Net 12k best复用|原尺寸整图76图val38.4469%，通过历史一致性检查|不单独查看候选test|权重本机和服务器保留|
|READY-U1-EXT|多边形IDRiD 48train/6dev，2.4k更新|外部dev27.1605%，不是DDR val|未执行外部test|23分20秒，本机best保留|
|READY-U1-DDR|外部完整模型迁移后DDR12k|DDR val36.6934%@12k|未单独评价|1小时40分30秒，本机best保留|
|READY-H1-EXT|HRNet-W48＋FCN适配，外部2.4k|外部dev42.9998%，不是DDR val|未执行外部test|1小时2分42秒，本机best保留|
|READY-H1-DDR|DDR完整12k预算|DDR val35.5950%@9k；末次12k35.4032%，停止追加|未单独评价|4小时36分49秒，本机best保留|
|READY-A0-DDR|HACDR官方骨干＋项目解码，6197步后停止|76图最佳val9.6781%@6k；HE/MA检出塌陷|未单独评价|2小时29分25秒；省5803更新；本机best保留|
|READY-FUSION-OOF|最多2家族提名，固定三折，候选权重0/.125/.25/.5|统一U0 .25规则；OOF43.2483%，锁定配方val43.4877%|选模阶段未看test|checkpoint已用val选点，OOF不是独立外层验证|
|READY-LOCKED-TEST|旧冠军.75＋U0 .25，四类统一，严格p>.5|锁文件SHA 644974f4a888835f838472199dcb623e879be2d62e73dab502394759551ceac4|227图44.7809%；EX54.5977/HE45.2021/SE50.0838/MA29.2400%；+0.5699个百分点|唯一锁定管线；逐图计数已复算一致|

旧冠军历史报告44.2115%与同次复算44.2110%相差0.0005个百分点；本轮提升按同次复算计算。新模型未入最终部署，不能把有效提升归因于新增外部预训练。高分辨率、第二seed、半监督的条件与跳过原因见本轮补充；清理后数据盘剩余20.09GiB，可供下一轮新实验使用，本轮test锁不重选。

来源：[本轮完整MD](D:/dataset/experiment/_全实验档案_20260920/04_本轮实验补充_20261002.md)、[结构化事实JSON](D:/dataset/experiment/_全实验档案_20260920/04_本轮实验补充_20261002.json)、[最终test原始JSON](D:/dataset/experiment/HRDecoder_Ready_20261002/server_results/reports/finish/final_test.json)、[权重清理记录](D:/dataset/experiment/HRDecoder_Ready_20261002/reports/server_weight_cleanup_result.json)。


<!-- HRD_ROUND2_RUNNING_20261002 -->
## 第二轮高分辨率配对实验已启动（2026-10-02）

启动时间23:42:08（北京时间）；服务器supervisor PID41075、初始训练PID41076。顺序为历史U0最佳权重→2048基准尺度控制臂→2048/2560基准尺度混合臂；均保留0.5–2.0随机缩放，每臂最多4000更新，1000/2000/4000完整val。采用既有最快已校准的micro4×accum2、workers4、AMP路径；同seed42、起始权重、LR与GT暴露。

新增完整验证同时计算固定旧冠军75%＋候选25%融合指标，以此选best；2k既无收益也无学习趋势才停，避免遗漏互补候选。预算不同时只在共同节点作配对比较。此次追加是清理空间后继续探索原方案未执行路线；上一轮test44.7809%及原锁保留，本轮不再评价该test。

协议：[protocol.json](D:/dataset/experiment/HRDecoder_Ready_20261002/round2_highres/protocol.json)；当前进度：[进度报告](D:/dataset/experiment/HRDecoder_Ready_20261002/round2_highres/reports/进度.md)。后台每分钟采集健康与同步结果，模型巡检每小时一次。登记时间2026-10-02 23:51:12。


<!-- HRD_ROUND2_COMPLETE_20261003 -->

高分辨率配对续训已完成，原 test 保持锁定。实际更新：U0_continuation2048 2000 步；U0_highres 4000 步。结果与权重备份校验见 [05_高分辨率配对续训_20261003.md](D:/dataset/experiment/_全实验档案_20260920/05_高分辨率配对续训_20261003.md)。


<!-- HRD_ROUND2_CLOSURE_20261003 -->

本轮训练、备份校验和实验汇总已完成；关机指令已通过 bash 执行，随后 SSH 连接断开且复查无法连接，平台最终状态未独立核验。小时模型监控已停止。收尾证据见 [收尾与关机记录](D:/dataset/experiment/HRDecoder_Ready_20261002/round2_highres/reports/收尾与关机记录.md)。
