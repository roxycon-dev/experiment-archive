#!/usr/bin/env python3.8
"""ops/compact_monitor.py — read-only local monitor for legacy mmseg runs.

Reads state.json and *.log.json under --work-dir, summarizes training
progress into --out JSON atomically. No credentials, no network, no training
changes. Prints compact JSON to stdout only on phase/validation changes.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

INTERVAL = 60
MAX_SECONDS = 9 * 3600
TRAIN_RE = re.compile(r'"mode"\s*:\s*"train"', re.I)
VAL_RE = re.compile(r'"mode"\s*:\s*"val"', re.I)
ERR_PATS = ("Traceback", "CUDA out of memory", "NaN")

def safe_json_load(path):
    """Robust partial-JSON skip: return None if file unreadable/truncated."""
    try:
        with open(path, "r") as f:
            txt = f.read()
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            rows = []
            for line in txt.splitlines():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass
            return rows
    except Exception:
        return None

def scan_errors(work_dir):
    errs = []
    try:
        for lp in sorted(glob.glob(os.path.join(work_dir, "*.log")))[-3:]:
            try:
                with open(lp, "r", errors="replace") as f:
                    tail = f.read()[-20000:]
            except Exception:
                continue
            for line in tail.splitlines():
                if any(p in line for p in ERR_PATS):
                    errs.append(line.strip()[:200])
    except Exception:
        pass
    return errs[-5:]

def extract(logs):
    """logs = list of dicts (parsed log.json entries). Returns summary parts."""
    cur_iter = None
    last_val = None      # dict: iter, mIoU, per_class (or None)
    best_val = None
    phase = "unknown"
    val_history = []

    for e in logs:
        if not isinstance(e, dict):
            continue
        mode = e.get("mode")
        it = e.get("iter")
        if mode == "train":
            phase = "train"
            if isinstance(it, int):
                cur_iter = it
        elif mode == "val":
            m = e.get("mIoU")
            if m is not None and isinstance(it, int):
                miou100 = float(m)
                if 0.0 <= miou100 <= 1.0:
                    miou100 *= 100.0
                val_history.append({"iter": cur_iter, "mIoU": round(miou100, 2)})
                last_val = val_history[-1]
                if best_val is None or miou100 > best_val["mIoU"]:
                    best_val = dict(last_val)
    if cur_iter is not None and last_val is not None:
        phase = "train" if cur_iter > last_val["iter"] else "validated"
    per_class = None
    if logs:
        for e in reversed(logs):
            if isinstance(e, dict) and e.get("mode") == "val" and e.get("per_class_iou"):
                per_class = e["per_class_iou"]
                break
    return cur_iter, last_val, best_val, phase, val_history, per_class

def decision_needed(cur_iter, val_history):
    flags = []
    if cur_iter is not None and cur_iter >= 8000:
        recent = val_history
        vals = recent[-4:]
        if len(vals) >= 4 and all(v["mIoU"] < 39.5 for v in vals[-4:]):
            flags.append("stalled_below_39.5_after_8k")
        for a, b in zip(val_history, val_history[1:]):
            if a["mIoU"] >= 42.4 and b["mIoU"] >= 42.4 and b["iter"] > a["iter"]:
                flags.append("two_consecutive_ge_42.4")
                break
    return flags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    start = time.time()
    prev_sig = None

    while time.time() - start < MAX_SECONDS:
        logs = []
        for p in sorted(glob.glob(os.path.join(args.work_dir, "*.log.json"))):
            d = safe_json_load(p)
            if isinstance(d, list):
                logs.extend(x for x in d if isinstance(x, dict))
            elif isinstance(d, dict):
                logs.append(d)
        state = safe_json_load(os.path.join(args.work_dir, "state.json"))
        state_phase = state.get("phase") if isinstance(state, dict) else None
        state_iter = state.get("iter") if isinstance(state, dict) else None

        cur_iter, last_val, best_val, phase, val_hist, per_class = extract(logs)
        if state_phase:
            phase = state_phase
        if state_iter is not None and (cur_iter is None or state_iter >= cur_iter):
            cur_iter = state_iter
        errs = scan_errors(args.work_dir)
        try:
            log_age = round(time.time() - os.path.getmtime(
                max(glob.glob(os.path.join(args.work_dir, "*.log.json")),
                    key=os.path.getmtime)), 1)
        except Exception:
            log_age = None

        summary = {
            "ts": int(time.time()),
            "phase": phase,
            "current_train_iter": cur_iter,
            "last_val": last_val,
            "best_val": best_val,
            "log_age_s": log_age,
            "errors_tail": errs,
            "decision_needed": bool(decision_needed(cur_iter, val_hist)),
        }
        # Atomic write
        tmp = out_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(summary, f)
            os.replace(tmp, out_path)
        except Exception as e:
            print(json.dumps({"error": f"write_failed:{e}"}), flush=True)

        sig = (phase, last_val and last_val["iter"],
               last_val and last_val["mIoU"], len(errs),
               summary["decision_needed"])
        if sig != prev_sig:
            print(json.dumps(summary), flush=True)
            prev_sig = sig

        if phase in ("complete", "error"):
            break
        time.sleep(INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
