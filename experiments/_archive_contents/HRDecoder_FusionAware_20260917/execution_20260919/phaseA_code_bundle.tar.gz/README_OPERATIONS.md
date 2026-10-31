# Operations Guide (Budget, Watchdog, Asset Audit)

## Scope
- Platform: production on Windows CPU; deadline watchdog and real shutdown only on Linux.
- This repo performs a **budget-guarded run** and an **asset audit**. It is NOT root training.
- Output is written only into a new run/output directory; scoped source paths are read-only.

## 1. Initialize budget ledger
```bash
python3 ops.py init --started-at 2026-09-16T00:00:00+00:00 --other-cost 5.5 --run-dir ./run
```
- `--started-at` MUST include a timezone (UTC expected). Naive datetimes are rejected.
- Deadline = min(start + 24h, start + (50 - other_cost) / 1.98 hours). Budget 50 RMB, rate 1.98 RMB/h.
- The original billing start is persisted and **never reset on restart**; re-running `init` on an existing run-dir is a no-op.

## 2. Watchdog (Linux only for real shutdown)
```bash
python3 ops.py watchdog --run-dir ./run            # real: shutdown -h now at deadline (Linux only)
python3 ops.py watchdog --run-dir ./run --dry-run  # safe everywhere, incl. Windows/tests
```
Real shutdown on Windows is forbidden and raises. Tests never trigger real shutdown.

## 3. Guarded run (Linux)
```bash
bash run_guarded.sh <settings-file> <run-dir> [backup-seconds]
```
- Requires an existing ledger.
- Starts watchdog via `nohup` before the workload.
- `trap` on EXIT/ERR/INT/TERM triggers shutdown (Linux).
- Backup is bounded by `timeout` and capped so it cannot exceed the deadline.
- Finally invokes `python3 run_protocol.py --settings <file> --run-dir <dir>`.

## 4. Protocol helpers (`ops.py`)
- `choose_steps(train_s_per_step, eval_s, overhead_s=600)`: largest T in [1000,4000], multiple of 4, with `2*T*train + 8*eval + overhead <= 21600`; else error.
- `gate_first(a4, b4, start_miou)`: pass iff `max(B)-max(A) >= 0.003`, at least 2 paired points with B>A, and `max(B) >= start_miou`.
- `choose_checkpoint(points)`: highest mIoU; ties resolved to the earliest step.
- `metrics_from_counts(counts)`: image-level TP/FP/FN are summed per class **before** IoU; a class with zero union raises an explicit error.

## 5. Asset audit (`asset_inventory.py`)
```bash
python3 asset_inventory.py ./new_audit_out
```
- Fixed scoped paths only (no full-disk scan):
  - Champion root: `D:/dataset/experiment/hrdecoder_plus3` (heads D/E/B/E4/MA; D2000 student backup).
  - Audit run: `D:/dataset/experiment/HRDecoder_审查轮_20260916/audit_runs/audit_20260916_A` (`h4_best.pth`, `experiments/h5_test.json`, `experiments/h7_test.json`).
  - Data root: `D:/HRDecoder/data/DDR_split_613`.
- SHA-256 hashes + missing flags per asset; h5/h7 JSON compared numerically (no pickle loading anywhere).
- Split: reads `split_assignment.json` defensively (inspects actual dict/list structure), else falls back to files on disk. Verifies counts 454/76/227, no duplicates/overlaps, paired image/label presence; otherwise status `BLOCKED`.
- Results written only into the new output directory (`asset_inventory.json`).

## 6. Tests
```bash
python -m pytest tests/test_ops.py -v
```
Covers budget deadline math, restart persistence, dry-run watchdog, Windows shutdown prohibition, step planning, gates/checkpoint, metrics, and split-assignment structure parsing. Tests are offline and never issue a real shutdown; they do not depend on the Windows data paths existing.
