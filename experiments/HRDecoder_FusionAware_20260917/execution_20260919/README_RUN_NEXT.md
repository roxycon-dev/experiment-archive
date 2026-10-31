# README_RUN_NEXT — cs4000_v2 训练收尾与 val 评估操作说明

适用目录：`execution_20260919/`（`finalize_after_training.sh`、`eval_candidates.py`、`phase_cs.py` 同目录部署到服务器）。
适用对象：`cs4000_v2` 一次 4000 步 C/S 匹配续训运行及其后的 val 评估收尾。

---

## 1. 脚本做什么（一句话）

`finalize_after_training.sh` 等训练进程退出 → 核验 `cs4000_v2/` 产物齐全 → **串行**（每次仅 1 个，控制显存）调用 `eval_candidates.py` 在 **val** 上评估 4 个 candidate（champion 概率缓存目录固定）→ 全部成功后把原始输出 + 训练日志打包 `tar.gz` 到 backup 目录并做 sha256 校验 → 校验通过才**原子创建** `READY_TO_SHUTDOWN`。任何一步失败：非零退出，**绝不创建标志**。

> ⚠️ 本脚本**绝不运行 test、绝不 shutdown / reboot / poweroff**。`READY_TO_SHUTDOWN` 只是标志文件，关机由人工或外部看门狗另行决定。

---

## 2. 前置条件

- Linux 服务器（脚本依赖 `/proc`、GNU `tar`、`sha256sum`、`gzip`、`bash`）；`python3` 可用（可用环境变量 `PYTHON=` 覆盖）。
- 部署目录中已有：`phase_cs.py`、`eval_candidates.py`、`finalize_after_training.sh`、完整 settings JSON。
- `eval_candidates.py` 为冻结协议脚本，**不得修改**；收尾脚本只调用它。
- 收尾脚本与训练进程以**同一用户**运行（需要读 `/proc/<PID>/fd` 与 `kill -0`）。
- GPU 显存：同一时刻只有 1 个评估进程（4 个 candidate 严格串行）。

---

## 3. 精确启动命令

### 3.1 训练 + 收尾一起挂后台（推荐）

```bash
cd <部署目录>                      # 含 phase_cs.py / eval_candidates.py / finalize_after_training.sh
BASE=/path/to/run_base             # 其下将出现 cs4000_v2/ 、evalcand_val/ 、READY_TO_SHUTDOWN
SETTINGS=$BASE/settings_full.json  # 完整 settings JSON
BACKUP=/path/to/backup
mkdir -p "$BASE" "$BACKUP"

# 1) 训练（4000 步，C/S 两臂）
nohup python3 phase_cs.py \
    --settings "$SETTINGS" \
    --run-dir "$BASE/cs4000_v2" \
    --mode train --arms C S --steps 4000 --seed 42 \
    > "$BASE/cs4000_v2_train.log" 2>&1 &
TRAIN_PID=$!

# 2) 收尾看护（会一直等到 TRAIN_PID 退出后再干活）
nohup bash finalize_after_training.sh "$TRAIN_PID" "$BASE" "$SETTINGS" "$BACKUP" \
    > "$BASE/finalize_launcher.log" 2>&1 &
```

### 3.2 训练已在跑、后补启动收尾

```bash
TRAIN_PID=$(pgrep -f "phase_cs.py.*cs4000_v2")   # 确认是唯一命中
nohup bash finalize_after_training.sh "$TRAIN_PID" "$BASE" "$SETTINGS" "$BACKUP" \
    > "$BASE/finalize_launcher.log" 2>&1 &
```

参数依次为：**训练PID、base目录、settings JSON 路径、backup目录**（恰好 4 个位置参数）。收尾脚本自身的完整运行日志在 `$BASE/finalize_after_training_<时间戳>.log`。脚本可重复启动前由锁目录 `$BASE/.finalize_after_training.lock` 防止双实例。

---

## 4. 收尾脚本内部流程

1. **等待**：每 30 s（`FINALIZE_POLL_SEC` 可调）轮询 PID；带 PID 复用防护与僵尸态识别。启动时 PID 已不存在则告警并直接进入产物核验（核验会 fail closed）。
2. **产物核验**（缺任一 → 退出码 4，无标志）：
   - `cs4000_v2/C_summary.json`、`S_summary.json`（并校验 JSON、arm、steps=4000、history 非空）
   - `cs4000_v2/C_step_2000.pth`、`C_step_4000.pth`、`S_step_2000.pth`、`S_step_4000.pth`
   - 四个 counts JSON：`C_val_2000_counts.json`、`C_val_4000_counts.json`、`S_val_2000_counts.json`、`S_val_4000_counts.json`
3. **val 评估**（4 次串行；单次失败 → 退出码 5，无标志）。实际执行的 4 条命令形如：

```bash
python3 eval_candidates.py \
    --settings "$SETTINGS" \
    --out-dir "$BASE/evalcand_val/<NAME>" \
    --prefix evalcand --split val \
    --candidate "<NAME>=$BASE/cs4000_v2/<NAME>.pth" \
    --candidate-arm "<NAME>=<C|S>" \
    --champion-cache-dir "$BASE/champ_prob_cache_val"
# <NAME> ∈ {C_step_2000, C_step_4000, S_step_2000, S_step_4000}
```

   - 固定 champion cache：`$BASE/champ_prob_cache_val`。第 1 次运行构建（最慢，含 76 张 val 图的 champion TTA 前向），后 3 次复用（脚本内 identity/sha256 校验，损坏即 fail closed）。
   - ⚠️ cache 与 settings JSON 强绑定：**若更换 settings / 权重，必须先删空 `$BASE/champ_prob_cache_val`**，否则评估会因 identity 不匹配直接失败（这是有意的 fail closed）。
   - 命令永远 `--split val`，且从不传 `--allow-test` / `--locked-candidate` / `--write-lock`。
4. **打包与校验**（失败 → 退出码 6，无标志）：写临时文件 `$BACKUP/finalize_<ts>.tar.gz.tmp` → `gzip -t` + `tar -t` + 条目数下限 → sha256 计算两次比对 → 原子改名 `finalize_<ts>.tar.gz` + `.sha256` sidecar。
5. **终检 + 原子建标志**：再核验全部关键文件仍在 → 写 `$BASE/.READY_TO_SHUTDOWN.<pid>` 临时文件 → 同目录 `mv` 原子改名为 `$BASE/READY_TO_SHUTDOWN`（内容含归档路径与 sha256）。

---

## 5. 产物清单

| 路径 | 内容 |
|---|---|
| `$BASE/cs4000_v2/…` | 训练产物：`{C,S}_summary.json`、`{C,S}_step_{2000,4000}.pth`、`{C,S}_val_{2000,4000}_counts.json`、`{C,S}_first_update.json`、`{C,S}_final.pth` |
| `$BASE/evalcand_val/<NAME>/` | 每个 candidate 一组：`evalcand_summary.json`、`evalcand_champion_counts.json`、`evalcand_cand_<NAME>_counts.json`、`evalcand_mix_<NAME>_counts.json`、`run.log` |
| `$BASE/champ_prob_cache_val/` | champion 概率缓存（manifest + 76 个 npy，settings 变化即失效） |
| `$BACKUP/finalize_<ts>.tar.gz` (+`.sha256`) | 归档：`evalcand_val/` 全部原始输出、`cs4000_v2/*.json`、训练日志（从 `/proc` 捕获的真实 stdout/stderr 目标 + 常规 `*.log`/`nohup.out`）、收尾脚本日志快照 |
| `$BASE/READY_TO_SHUTDOWN` | 全部成功后才原子创建的标志文件 |
| `$BASE/finalize_after_training_<ts>.log` | 收尾脚本全程日志 |

**归档明确排除**：`*.pth` 检查点与 champion 概率缓存（体积大，留在磁盘 `$BASE` 下；如需异地备份请单独处理）。

---

## 6. 失败行为

任何失败：脚本立即非零退出并清理现场（锁、暂存目录），**绝不创建 `READY_TO_SHUTDOWN`**；已完成的中间产物保留在磁盘上，可直接排查后重跑收尾脚本（评估输出会原子覆盖，安全）。

| 退出码 | 含义 |
|---|---|
| 0 | 全部成功，标志已创建 |
| 2 | 用法/参数错误，或 settings JSON 缺失/不可解析/缺必需键 |
| 3 | 环境错误（非 Linux、缺工具、缺 eval_candidates.py）、锁冲突、收到终止信号 |
| 4 | 训练产物核验失败（10 个必需文件缺失/为空/JSON 无效/steps≠4000） |
| 5 | 某次 `eval_candidates.py` 失败或输出不全（对应 `evalcand_val/<NAME>/run.log`） |
| 6 | 打包/sha256 校验/终检失败 |

---

## 7. test 的正确姿势（重要）

收尾脚本**永远不会跑 test**。test 必须走独立的人工流程：

1. 四次 val 评估完成后，**人工**查看 4 个 `$BASE/evalcand_val/<NAME>/evalcand_summary.json` 中 `results`（champion / cand_* / mix_* 的 mIoU），**人工锁定 val 赢家**（平票按协议归 champion；若赢家就是 champion 本身，通常无需再跑 test）。
2. 另开终端、**单独**执行一次 test（赢家名字必须与 val 锁定一致，脚本会与 `locked_candidate.json` 交叉校验）：

```bash
cd <部署目录>
python3 eval_candidates.py \
    --settings "$SETTINGS" \
    --out-dir "$BASE/evalcand_test" \
    --prefix evalcand --split test \
    --candidate "<赢家NAME>=$BASE/cs4000_v2/<赢家NAME>.pth" \
    --candidate-arm "<赢家NAME>=<C|S>" \
    --allow-test --locked-candidate "<赢家NAME>"
```

3. test 与 `READY_TO_SHUTDOWN`、关机完全无关。

---

## 8. 常见问题

- **启动时提示 PID 不存在**：训练可能已退出或 PID 抄错；脚本会继续做产物核验，产物不全则以退出码 4 结束（fail closed）。
- **换 settings/权重后评估报 champion cache 相关错误**：删除 `$BASE/champ_prob_cache_val` 后重跑收尾脚本。
- **重跑收尾脚本**：安全。评估 JSON 均为原子写入，归档按新时间戳另起文件；先确认旧实例已退出（锁目录会被其自动清理）。
- **Windows 上编辑**：必须保持 LF 行尾（本文件与 `.sh` 均为 LF）。
