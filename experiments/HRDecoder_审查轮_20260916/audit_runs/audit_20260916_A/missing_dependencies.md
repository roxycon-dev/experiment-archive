# missing_dependencies.md

> 生成时间：2026-09-16（CST）　run_id: `audit_20260916_A`
> 规则（prompt §3.1）：文件不存在就列入本表。**不得按照名字猜着重写父类，然后宣称复现原方法。**

## 1. 本地缺失、只在服务器上的资产（→ 需开机才能取回）

| 资产 | 服务器路径 | 用途 | 状态 |
|---|---|---|---|
| **D@2000 完整 checkpoint（789,655,426 B，含 `reference.*`）** | `/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth` | 基线参照 D 的权威权重 | BLOCKED |
| **E@1500 完整 checkpoint（含 `ma_refiner.*`）** | `/root/autodl-tmp/hrplus/results/E_0916/best_mIoU_iter_1500.pth` | 本轮残差头模型 | BLOCKED |
| E2@1500 / E3@2000 checkpoint | `/root/autodl-tmp/hrplus/results/E2_0916/`、`E3_0916/` | 消融对照 | BLOCKED |
| L@6000 checkpoint | `/root/autodl-tmp/hrplus/results/L_0915/best_mIoU_iter_6000.pth` | L 对照 | BLOCKED |
| B@3000 **完整** checkpoint（526,178,711 B，含 optimizer） | `/root/autodl-tmp/hr100/trial_b/best_mIoU_iter_3000.pth` | 与本地 weights-only 版做一致性核对 | BLOCKED（本地有 weights-only 版，见下） |
| `hrplus_core.py` / `ma_residual.py` 等**服务器实际版本** | `/root/autodl-tmp/hrplus/repo/` | 与本地副本做逐文件 SHA-256 比对 | BLOCKED（本地有副本，但**未证明与服务器一致**） |
| `ops/*.py` 本轮全部运维脚本 | `/root/autodl-tmp/hrplus/ops/` | 审计旧 guard / 旧评估器 | BLOCKED |
| E / L / D 的**原始训练日志** | `/root/autodl-tmp/hrplus/results/*/e_train.log` | 核对报告中的 val 曲线 | BLOCKED（本地仅有一轮 L 的日志副本） |

## 2. 本地存在但**不能当作已获取**的资产

| 资产 | 本地路径 | 为什么不算已获取 |
|---|---|---|
| 仓库源码副本 | `D:\dataset\experiment\hrdecoder_plus3\repo\` | **不是 git 仓库**（无 `.git`），无分支/提交/未提交改动可记录（§3.1 要求）。也不等于服务器当前版本 |
| `ma_residual.py` 等新增代码 | 工作区 `C:\...\2026-09-15-22-20-12\ma_residual.py` 与 `HRDecoder_MA残差头与集成_20260915-16\03_代码\` | 已取 SHA-256，但**服务器上那份是否一致未验证** |
| D@2000 **student-only**（263,419,124 B） | `hrdecoder_plus3\results\D_0913_retry\backup_student_2000.pth` | 只是 student 分支，**不是完整 D 模型**。参考分支需另外提供 |
| B@3000 **weights-only**（263,479,238 B） | `hrdecoder_plus3\repo\weights\b_weights.pth` | 与服务器上的软链目标（含 optimizer 的 526 MB 版）**不是同一文件** |
| 数据集 tar（684,257,280 B） | `autodl100\bundle\ddr_split613.tar` | 未解压；标签值域（是否含 255）**尚未实测** → 需验证项 V1 |

## 3. 环境缺失（prompt §0.1：不安装依赖，只报告）

| 能力 | 本地 | 影响 |
|---|---|---|
| `mmcv 1.7.1` / `mmseg 0.16.0` | ❌ 缺失 | **无法本地构建 dataset / config / model** → Phase C 及之后全部 BLOCKED |
| Python 3.8.10 训练栈 | ❌ 本地是 3.13.11（conda）/ 3.13.14（venv） | 与服务器解释器不同；`.pyc` 与 pickle 兼容性不保证 |
| torch 2.0.1+cu117 | ❌ 本地是 2.6.0+cu124 | **`torch.load` 默认值不同**：2.6 起 `weights_only=True`，会拒绝含 `numpy.core.multiarray.scalar` 的 mmseg checkpoint。<br>本轮的处置：显式 `weights_only=False`（本地可信文件），并把该差异记入 `manifest.json`。<br>**服务器不受影响**（2.0.1 默认 False） |
| CUDA 训练能力 | ✅ conda 侧 `torch.cuda.is_available()=True`（cu124） | 但缺 mmcv，仍跑不了真实模型 |

## 4. 需要用户/上游提供才能推进的授权

| 项目 | 状态 |
|---|---|
| 当前**有效的服务器开机授权** | ❌ **未提供**（上一轮凭据对应的实例已关机，且 prompt §0.1 要求"当前"授权） |
| 当前**预算 / 最长用时 / 绝对截止时间（含时区）** | ❌ **未提供** |
| 云平台侧**停止计费的独立确认手段** | ❌ 未提供（prompt §4.3-12 要求；TCP 拒绝只能作辅助信号） |
| **test 是否解封** | ❌ 未批准（prompt §11.4：默认保持封存） |
| 是否允许使用公开预训练权重 | ❌ 未定义（prompt §0.1 明确"不得自行决定"，且这是对用户约束的解释问题） |
| 数据集自带的**患者映射表** | ❌ 本地无可用的患者 ID 映射 → 患者级审计 BLOCKED（见 A4 报告） |

## 5. 明确**不能**靠猜测补齐的东西

- ❌ 不重写 `HR100Refinement` / `HRDecoder` / `LesionEncoderDecoder` 的父类行为
- ❌ 不按 `hrplus_patch.py` 的名字重建平铺/融合实现
- ❌ 不用本地 weights-only 文件冒充服务器完整 checkpoint
- ❌ 不在缺 mmcv 的情况下"近似"解析 config 并宣称拿到解析后哈希（→ V3）
