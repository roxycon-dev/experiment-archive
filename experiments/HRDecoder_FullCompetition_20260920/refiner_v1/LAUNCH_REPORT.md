# LAUNCH REPORT — HRDecoder refiner_v1 formal run

Date: **2026-09-21** · Server: `root@connect.nmb2.seetacloud.com:48301` (RTX 3090)
Root: `/root/autodl-tmp/HRDecoder_FullCompetition_20260920`
Local mirror: `D:\dataset\experiment\HRDecoder_FullCompetition_20260920\refiner_v1`

**Status: LAUNCHED and RUNNING.** The run is detached from the launching
session and is monitored by reading `status.json` — no long-lived agent session
is required. Test was never read; nothing was deleted; the server was not shut
down; 44795 / 4090 were never touched.

---

## 1. The one-time design decision (R1 closed)

```
L = 30.0 * focal_bce(alpha=0.75, gamma=2.0) + 1.0 * soft_dice
```

Fixed by the main agent **before the first formal step**, because the learning
smoke measured focal at only ≈ 0.5 % of the total loss at `w_focal = 1.0`
(the objective was effectively Dice-only). This is frozen and must never be
revised afterwards, and never on val curves.

Enforced in code, not just in prose:

| where | what |
|---|---|
| `refiner/losses.py` | `FROZEN_TERMS / FROZEN_WEIGHTS / FROZEN_ALPHA / FROZEN_GAMMA`, `assert_frozen_objective()`, `weighted_parts()` |
| `configs/refiner_v1.json` | `loss.weights = {focal_bce: 30.0, soft_dice: 1.0}` + `objective_frozen_at` + `_weight_decision` |
| `configs/stage1.json`, `configs/stage2.json` | verbatim mirror of the same block |
| `train_refiner.py` | calls `assert_frozen_objective` before any compute **and** re-checks that the stage override did not change the main config's loss |
| `tests/test_refiner_units.py` | `objective_30focal_1dice`, `objective_frozen_fail_closed`, `loss_total_and_focal_share` |

## 2. Tests and smoke (executed on this exact revision)

| check | result | log |
|---|---|---|
| CPU unit tests | **47/47** (candidate 15/15, refiner 32/32) | `logs/test_candidate_rules_20260921.log`, `logs/test_refiner_units_20260921.log` |
| 40-step learning smoke (`--train-images 4 --overfit-steps 40`) | exit 0; loss 0.12695–1.00005, all rows `finite=1`; `gnorm` 0.08–174.19 (clip 5.0); `focal_share` mean **0.1569** (was ≈0.005 at `w=1.0`); `identity` **bitwise, max|d| = 0.0** | `logs/learning_smoke_w30_20260921.log`, `runs/smoke_w30_20260921/{steps,epochs}.csv`, `done.json` |
| strict checkpoint round-trip (`torch.load(weights_only=True)` + `load_state_dict(strict=True)`) | OK, 25 394 params | `runs/formal_20260921/logs/preflight_tests*.log` |
| supervisor plan (`run_formal_v1.py --dry-run`) | 12 stages, 6 pre-registered stage-2 nodes + 1 stage-1 node | `runs/formal_20260921/supervisor.out` |

New tests added this round (4): the objective is exactly the two declared terms
with frozen weights, drift fails closed, the total and recorded focal share are
consistent, and the validation-node list is frozen and consistent with
`eval_every_epochs`.

## 3. The detached supervisor

`refiner_v1/run_formal_v1.py`, launched by `refiner_v1/tools/start_formal_v1.sh`.

**PID: `197611`** (authoritative file `runs/formal_20260921/supervisor.pid`,
written by the supervisor itself; `PPID = 1`, own session ⇒ it survives the
launching SSH/agent session).

Stage order — strictly sequential, **one GPU job at a time**:

```
0 preflight_tests   CPU unit tests, exit 0 required
1 cache_train       454 images      (GPU)   ~35 min measured 4.6 s/img
2 cache_val          76 images      (GPU)   ~6 min
3 stage1_train       8 images, 200 epochs  (GPU)
4 eval_stage1_ep200  pre-registered stage-1 node (GPU)
5 stage2_train       454 images, 30 epochs (GPU)   <- dominates the wall clock
6..11 eval_stage2_ep{005,010,015,020,025,030}      <- the 6 nodes budgeted in R8
12 selection         frozen rule -> selection.json (CPU)
```

Selection rule (frozen): promote only if val MA IoU **and** val HE IoU both beat
the champion at the fixed `p > 0.5`; among winners take the highest mIoU. With
no winner, the highest-mIoU node is kept as `best_for_analysis.pth` and is
**not** called a win. `selection.json` records `verdict = WIN | NO_WIN`,
per-node deltas and the win flags.

Safety guards — each writes `state=FAILED` + an `alerts` entry and kills the
running child process group:

* free disk < 4 GB on `/root/autodl-tmp`
* non-finite loss / grad-norm (from the trainer's own `steps.csv`, and the
  trainer aborts itself with `TRAIN_FAILED.json` + exit 3)
* identity / bitwise-invariant violation (trainer identity guard; and
  `eval_refiner` refuses to report unless EX/SE and outside-candidate changes
  are 0 and the champion contract self-check passes)
* strict checkpoint load failure (any stage non-zero exit)
* GPU occupied by another task before a GPU stage starts
* no output progress for 30 min (stall guard)

Per-epoch/per-step artefacts: `steps.csv` (incl. `focal_share`), `epochs.csv`,
`history.json`, `heartbeat.json`, `nodes/node_ep###.pth` (the pre-registered
nodes), `last.pth` + `last_opt.pth` (resume), `done.json`. Epoch checkpoints are
retained with a 3-epoch window; node checkpoints are never pruned. Cache ≈ 800
MB, checkpoints ≈ 100 KB each — the disk guard has ~15 GB of headroom.

## 4. Launch verification (observed, not assumed)

```
state=RUNNING  stage=cache_train (1/12)  stages_done=1
pid 197611 ALIVE, PPID=1, SID=197611 (detached)
gpu_apps = ['198091, /root/miniconda3/bin/python, ~1.9 GiB']   <- child of 197611
GPU = RTX 3090, 1913 MiB, 24 GiB total
disk_free = 15.84 GB (floor 4.0)
cache_train: 20/454 images in 92 s, 28.8 MB written
alerts = []
```

## 5. Monitoring and safe stop (status.json is enough)

```bash
R=/root/autodl-tmp/HRDecoder_FullCompetition_20260920/refiner_v1/runs/formal_20260921

# one-screen status
python - <<'EOF'
import json; d=json.load(open("/root/autodl-tmp/HRDecoder_FullCompetition_20260920/refiner_v1/runs/formal_20260921/status.json"))
print(d["state"], d["current_stage"], f'{d["stage_index"]}/{d["n_stages"]}', "done", d["stages_done"])
print("disk", d["disk_free_gb"], "gpu", d["gpu"], "apps", d["gpu_apps"])
print("eta_end", d["eta_end_rough"], "alerts", d["alerts"], "selection", d["selection"])
for s in d["stages"]: print(" ", s["name"], s["state"], s["exit_code"], s["note"][:80])
EOF

# alive?
kill -0 $(cat $R/supervisor.pid) && echo ALIVE

# progress of the running stage
tail -n 3 $R/logs/stage2_train.log ; cat $R/stage2/heartbeat.json

# graceful stop (deletes nothing; state becomes STOPPED)
bash /root/autodl-tmp/HRDecoder_FullCompetition_20260920/refiner_v1/tools/stop_formal_v1.sh
# or: touch $R/STOP      or:  kill -TERM $(cat $R/supervisor.pid)

# final outcomes
ls $R/selection.json $R/RUN_REPORT.md $R/best*.pth
```

Reading `state` alone is sufficient:

| `state` | meaning |
|---|---|
| `RUNNING` | healthy, check `current_stage` / `stages_done` / `heartbeat_age_s` |
| `DONE` | all stages finished; read `selection.json` / `RUN_REPORT.md` |
| `FAILED` | stopped by a guard — read `alerts[-1]` (reason + the offending stage) |
| `STOPPED` | stopped because the `STOP` sentinel (or SIGTERM) was received |

## 6. Estimated finish

Server time at launch **2026-09-21 00:19**. Wall-clock is dominated by
`stage2_train` (≈ 19 600 steps, 30 epochs × ≈ 654 steps), which has not been
measured at batch 16 yet, so:

* **estimate: 2026-09-21 ≈ 04:00 – 07:00 server time** (≈ 3.7 – 7 h from launch)
* `status.json.eta_seconds_rough` / `eta_end_rough` are refreshed on every poll
  and get real (measured) once each stage completes; after `stage2_train`
  finishes the remaining work is a fixed 6 × ≈ 5.5 min of val evaluation.

Nothing is stopped early because of an unattractive curve: training runs to the
last pre-registered epoch unless one of the fail-closed guards fires.

## 7. Files mirrored back to the local `refiner_v1/`

```
run_formal_v1.py                     the supervisor
tools/start_formal_v1.sh             detached launch
tools/stop_formal_v1.sh              graceful stop (STOP sentinel)
configs/{refiner_v1,stage1,stage2}.json   frozen objective + nodes
refiner/losses.py, train_refiner.py, tests/test_refiner_units.py
GO_NO_GO.md, README.md               updated with the R1 decision
logs/test_candidate_rules_20260921.log      15/15
logs/test_refiner_units_20260921.log        32/32
logs/learning_smoke_w30_20260921.log        40-step smoke + readings
runs/smoke_w30_20260921/{steps,epochs,history,heartbeat,done}.json, refiner_last.pth
runs/formal_20260921/supervisor.pid          197611
runs/formal_20260921/status.json             launch snapshot
runs/formal_20260921/supervisor.out          supervisor stdout
runs/formal_20260921/logs/{preflight_tests,preflight_tests_units,cache_train}.log
```

The server keeps running; the run continues unsupervised and self-monitoring.
