#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_formal_v1 — detached supervisor for the FORMAL refiner_v1 run.

Why a supervisor and not a shell script: the run lasts hours, the launching
session must be disposable, and the only monitoring channel that is allowed to
matter is a machine-readable `status.json`.  Everything the operator needs is a
file read away, so no long-lived interactive session is required.

Stage order (strictly sequential, never two GPU jobs at once)
------------------------------------------------------------
  1  preflight_tests   CPU unit tests (47 checks), must exit 0
  2  cache_train       build_patch_cache.py --split train      (GPU)
  3  cache_val         build_patch_cache.py --split val        (GPU)
  4  stage1_train      train_refiner.py --stage-config stage1  (GPU)
  5  eval_stage1_ep200 eval_refiner.py --ckpt nodes/node_ep200 (GPU)
  6  stage2_train      train_refiner.py --stage-config stage2  (GPU)
  7  eval_stage2_ep{005,010,015,020,025,030}                   (GPU)
  8  selection         frozen rule, written to selection.json  (CPU)

The stage list is built from the CONFIGS (`stages.<stage>.checkpoint_nodes`),
so the pre-registered validation nodes cannot drift from what is evaluated.

Fail-closed safety (each writes state=FAILED + an `alerts` entry, and kills the
running child process group before stopping)
-------------------------------------------
  * free disk < 4 GB on /root/autodl-tmp
  * non-finite loss / gradient norm (from the trainer's own steps.csv too)
  * identity or bitwise-invariant violation (trainer identity guard, or
    eval_refiner's `accepted` / `contract_selfcheck` being false)
  * strict checkpoint load failure (eval/train exit codes)
  * GPU occupied by another task before a GPU stage starts
  * a stage that produces no output progress for `--stall-s` seconds

Monitoring / safe stop
----------------------
  watch : cat <run_root>/status.json   (state, current_stage, stage exit codes,
                                        disk_free_gb, heartbeat age, alerts)
  stop  : touch <run_root>/STOP        (graceful: kills the child group and
                                        marks state=STOPPED; nothing is deleted)
  or    : kill -TERM <status.pid>      (same path, via the signal handler)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

PY = "/root/miniconda3/bin/python"
ROOT = "/root/autodl-tmp/HRDecoder_FullCompetition_20260920"
HERE = os.path.join(ROOT, "refiner_v1")
RUN_ROOT = os.path.join(HERE, "runs", "formal_20260921")
CACHE_ROOT = os.path.join(HERE, "patch_cache")
DEFAULT_CONFIG = os.path.join(HERE, "configs", "refiner_v1.json")
STAGE1_CONFIG = os.path.join(HERE, "configs", "stage1.json")
STAGE2_CONFIG = os.path.join(HERE, "configs", "stage2.json")
DISK_PATH = "/root/autodl-tmp"
CONTRACT = {"mIoU": 0.42669428454584546, "MA": 0.259672, "HE": 0.437583}

# rough per-stage budget, used only for the ETA field in status.json
EST_S = {"preflight_tests": 90, "cache_train": 2400, "cache_val": 420,
         "stage1_train": 900, "eval_stage1_ep200": 400, "stage2_train": 7200,
         "eval_stage2_ep005": 400, "eval_stage2_ep010": 400,
         "eval_stage2_ep015": 400, "eval_stage2_ep020": 400,
         "eval_stage2_ep025": 400, "eval_stage2_ep030": 400}

# CUDA caching-allocator policy handed to every spawned stage.
#
# `refiner.dataset.collate` zero-pads every patch in a batch up to that batch's
# maximum (h, w); on the frozen stage-2 cache 223 / 10608 patches (2.1 %) have
# both edges >= 512 px, so ~31 % of shuffled batches of 16 are padded to
# 576x576.  Measured peak for that shape (forward + backward + AdamW step) is
# 21.34 GiB on the 24 GiB RTX 3090 (23.57 GiB visible to torch), i.e. the frozen
# stage-2 batch does fit, but with only ~2.2 GiB of headroom.  The first formal
# run died at step 52 with "18.49 GiB already allocated; 1.95 GiB free;
# 21.29 GiB reserved" -- 2.80 GiB of the reservation was stranded by
# fragmentation built up over the preceding 51 batches of mixed sizes.
#
# `max_split_size_mb:128` stops the allocator from carving blocks bigger than
# 128 MiB into smaller pieces, which is exactly what stranded the reservation.
# It is a pure allocation hint: it cannot change any value the trainer computes.
# Verified by replaying the deterministic batch sequence (the trainer seeds with
# --seed 20260920 before building the DataLoader):
#   * default          -> OOM at step 51, same numbers as the formal failure
#   * max_split_size_mb:128 -> 700 consecutive steps (a full epoch of 663 plus
#                              37) with loss/focal/dice/gnorm IDENTICAL to the
#                              default run for steps 1/20/40 at 0.694 s/step
#                              versus 0.70 s/step -- no slowdown.
#   * roundup_power2_divisions:* -> OOM (they enlarge blocks and make it worse)
# torch here is 2.0.1+cu117, which predates `expandable_segments` (2.1+).
CUDA_ALLOC_CONF = "max_split_size_mb:128"


def iso(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t or time.time()))


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def free_gb(path=DISK_PATH):
    try:
        return shutil.disk_usage(path).free / float(1 << 30)
    except OSError:
        return -1.0


def stage_env(st, base=None):
    """Environment for a spawned stage: inherit, then apply stage policy.

    GPU stages (training and eval both touch CUDA) get the fragmentation-safe
    allocator policy; an explicit value already present in the environment wins.
    Pure function of its arguments so the policy is unit-testable.
    """
    env = dict(os.environ if base is None else base)
    env["PYTHONUNBUFFERED"] = "1"
    if st["kind"] in ("gpu", "eval"):
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", CUDA_ALLOC_CONF)
    return env


def nvidia(fields, apps=False):
    key = "compute-apps" if apps else "gpu"
    try:
        out = subprocess.run(["nvidia-smi", "--query-%s=%s" % (key, fields),
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


class Aborted(RuntimeError):
    def __init__(self, reason, **where):
        super().__init__(reason)
        self.reason = reason
        self.where = where


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.run_root = os.path.abspath(args.run_root)
        self.log_dir = os.path.join(self.run_root, "logs")
        self.status_path = os.path.join(self.run_root, "status.json")
        self.stop_flag = os.path.join(self.run_root, "STOP")
        self.poll_s = float(args.poll)
        self.min_free_gb = float(args.min_free_gb)
        self.stall_s = float(args.stall_s)
        self.cfg = load_json(DEFAULT_CONFIG) or {}
        self.state = "INIT"
        self.exit_code = 0
        self.alerts = []
        self.stages = []
        self.cur = None
        self.last_progress = time.time()
        self.progress_watch = []
        self.t_start = time.time()
        self.selection = None
        self.stop_requested = False

    # ---------------------------------------------------------------- stages
    def stage_cfg(self, path, name):
        c = load_json(path) or {}
        return (c.get("stages") or {}).get(name) or {}

    def nodes(self, path, name):
        return [int(x) for x in (self.stage_cfg(path, name).get("checkpoint_nodes")
                                 or [])]

    def build_stages(self):
        s1_nodes = self.nodes(STAGE1_CONFIG, "stage1_overfit")
        s2_nodes = self.nodes(STAGE2_CONFIG, "stage2_main")
        if not s1_nodes or not s2_nodes:
            raise SystemExit("stage configs carry no checkpoint_nodes")
        gpu = lambda *c: {"kind": "gpu", "cmd": list(c)}
        st = [
            {"name": "preflight_tests", "kind": "cpu",
             "cmd": [PY, os.path.join(HERE, "tests", "test_candidate_rules.py")],
             "cmd2": [PY, os.path.join(HERE, "tests", "test_refiner_units.py")]},
            {"name": "cache_train", "kind": "gpu",
             "cmd": [PY, os.path.join(HERE, "build_patch_cache.py"),
                     "--split", "train", "--cache-root", CACHE_ROOT,
                     "--config", DEFAULT_CONFIG]},
            {"name": "cache_val", "kind": "gpu",
             "cmd": [PY, os.path.join(HERE, "build_patch_cache.py"),
                     "--split", "val", "--cache-root", CACHE_ROOT,
                     "--config", DEFAULT_CONFIG]},
            {"name": "stage1_train", "kind": "gpu", "nodes": s1_nodes,
             "cmd": [PY, os.path.join(HERE, "train_refiner.py"),
                     "--stage-config", STAGE1_CONFIG, "--cache-root", CACHE_ROOT,
                     "--config", DEFAULT_CONFIG,
                     "--out-dir", os.path.join(self.run_root, "stage1")]},
        ]
        for ep in s1_nodes:
            st.append({"name": "eval_stage1_ep%03d" % ep, "kind": "eval",
                       "stage_key": "stage1", "epoch": ep,
                       "ckpt": os.path.join(self.run_root, "stage1", "nodes",
                                            "node_ep%03d.pth" % ep),
                       "cmd": [PY, os.path.join(HERE, "eval_refiner.py"),
                               "--config", DEFAULT_CONFIG,
                               "--ckpt", os.path.join(self.run_root, "stage1",
                                                      "nodes",
                                                      "node_ep%03d.pth" % ep),
                               "--split", "val",
                               "--out", os.path.join(self.run_root, "eval",
                                                     "stage1_ep%03d.json" % ep)]})
        st.append({"name": "stage2_train", "kind": "gpu", "nodes": s2_nodes,
                   "cmd": [PY, os.path.join(HERE, "train_refiner.py"),
                           "--stage-config", STAGE2_CONFIG,
                           "--cache-root", CACHE_ROOT, "--config", DEFAULT_CONFIG,
                           "--out-dir", os.path.join(self.run_root, "stage2")]})
        for ep in s2_nodes:
            st.append({"name": "eval_stage2_ep%03d" % ep, "kind": "eval",
                       "stage_key": "stage2", "epoch": ep,
                       "ckpt": os.path.join(self.run_root, "stage2", "nodes",
                                            "node_ep%03d.pth" % ep),
                       "cmd": [PY, os.path.join(HERE, "eval_refiner.py"),
                               "--config", DEFAULT_CONFIG,
                               "--ckpt", os.path.join(self.run_root, "stage2",
                                                      "nodes",
                                                      "node_ep%03d.pth" % ep),
                               "--split", "val",
                               "--out", os.path.join(self.run_root, "eval",
                                                     "stage2_ep%03d.json" % ep)]})
        for i, s in enumerate(st):
            s.setdefault("state", "PENDING")
            s.setdefault("exit_code", None)
            s.setdefault("t_start", None)
            s.setdefault("t_end", None)
            s.setdefault("elapsed_s", None)
            s.setdefault("pid", None)
            s.setdefault("note", "")
            s.setdefault("est_s", EST_S.get(s["name"]))
            s["index"] = i
            s["log"] = os.path.join(self.log_dir, s["name"] + ".log")
            if "cmd2" in s:
                s["log2"] = os.path.join(self.log_dir, s["name"] + "_units.log")
        # skip rules: an already-complete artefact is not rebuilt, and a
        # training stage that was interrupted is RESUMED instead of restarted.
        for s in st:
            if s["name"] == "cache_train" and self._cache_complete("train"):
                s["skip"] = "index_train.json complete=true"
            if s["name"] == "cache_val" and self._cache_complete("val"):
                s["skip"] = "index_val.json complete=true"
            if s["name"].endswith("_train"):
                d = os.path.join(self.run_root, s["name"].split("_")[0])
                if os.path.isfile(os.path.join(d, "done.json")):
                    s["skip"] = "done.json present"
                elif os.path.isfile(os.path.join(d, "last.pth")):
                    s["cmd"].append("--resume")
                    s["note"] = "resuming from last.pth"
        self.stages = st

    def _cache_complete(self, split):
        idx = load_json(os.path.join(CACHE_ROOT, "index_%s.json" % split)) or {}
        return bool(idx.get("complete")) and int(idx.get("n_images") or 0) > 0

    # ---------------------------------------------------------------- status
    def write_status(self, force=False):
        now = time.time()
        apps = nvidia("pid,process_name,used_memory", apps=True)
        gpu = nvidia("name,memory.used,memory.total,utilization.gpu")
        hb_age = None
        if self.cur and self.cur.get("kind") == "gpu" and \
                self.cur["name"].endswith("_train"):
            p = os.path.join(self.run_root, self.cur["name"].split("_")[0],
                             "heartbeat.json")
            if os.path.isfile(p):
                hb_age = round(now - os.path.getmtime(p), 1)
        pend = [s for s in self.stages if s["state"] in ("PENDING", "RUNNING")]
        eta = sum(int(s.get("est_s") or 0) for s in pend)
        self.status = {
            "schema": "hrdecoder-refiner-supervisor/v1",
            "run": os.path.basename(self.run_root),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "root": self.run_root,
            "state": self.state,
            "resumed": bool(getattr(self.args, "resume", False)),
            "started_at": iso(self.t_start), "updated_at": iso(now),
            "elapsed_s": round(now - self.t_start, 1),
            "current_stage": (self.cur or {}).get("name"),
            "stage_index": (self.cur or {}).get("index"),
            "n_stages": len(self.stages),
            "stages_done": sum(1 for s in self.stages
                               if s["state"] in ("DONE", "SKIPPED")),
            "disk_free_gb": round(free_gb(), 2),
            "disk_min_gb": self.min_free_gb,
            "gpu": gpu, "gpu_apps": apps,
            "heartbeat_age_s": hb_age,
            "eta_seconds_rough": eta,
            "eta_end_rough": iso(now + eta),
            "alerts": self.alerts[-20:],
            "selection": self.selection,
            "exit_code": self.exit_code,
            "stages": [{k: s.get(k) for k in
                        ("index", "name", "kind", "state", "exit_code", "pid",
                         "t_start", "t_end", "elapsed_s", "log", "note",
                         "est_s", "skip")} for s in self.stages],
        }
        atomic_json(self.status_path, self.status)

    def alert(self, kind, msg, **extra):
        self.alerts.append({"ts": iso(), "kind": kind, "msg": msg, "extra": extra})
        print("[sup] ALERT %s: %s %r" % (kind, msg, extra), flush=True)

    # ------------------------------------------------------------ execution
    def kill_group(self, p):
        if p is None or p.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except OSError:
            return
        for _ in range(30):
            if p.poll() is not None:
                return
            time.sleep(1)
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except OSError:
            pass

    def _sig(self, *_a):
        self.stop_requested = True

    def run_stage(self, st):
        st["state"] = "RUNNING"
        st["t_start"] = iso()
        self.cur = st
        self.last_progress = time.time()
        self.write_status()
        print("[sup] == stage %s (idx %d/%d) %s"
              % (st["name"], st["index"], len(self.stages), " ".join(st["cmd"])),
              flush=True)
        if st["kind"] in ("gpu", "eval"):
            apps = nvidia("pid,process_name,used_memory", apps=True)
            if apps:
                raise Aborted("gpu_busy", stage=st["name"], gpu_apps=apps)
        rc = self._spawn(st, st["cmd"], st["log"])
        if rc != 0:
            return rc
        if st.get("cmd2"):
            rc2 = self._spawn(st, st["cmd2"], st["log2"], sub=True)
            if rc2 != 0:
                return rc2
        # keep the official exit code of the primary command
        return st["exit_code"]

    def _spawn(self, st, cmd, log_path, sub=False):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        f = open(log_path, "ab", buffering=0)
        env = stage_env(st)
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True,
                             env=env)
        if not sub:
            st["pid"] = int(p.pid)
        self.progress_watch = [log_path,
                               os.path.join(self.run_root, st["name"].split("_")[0],
                                            "heartbeat.json")
                               if st["name"].endswith("_train") else log_path]
        self.write_status()
        rc = None
        while True:
            rc = p.poll()
            if rc is not None:
                break
            if self.stop_requested or os.path.isfile(self.stop_flag):
                self.kill_group(p)
                f.close()
                self.stop_requested = True
                raise Aborted("stop_requested", stage=st["name"])
            try:
                self.tick(st)
            except Aborted as a:
                self.kill_group(p)
                f.close()
                raise a
            time.sleep(self.poll_s)
        f.close()
        st["exit_code"] = int(rc)
        return int(rc)

    def tick(self, st):
        # progress watch -> stall detection
        cur = [os.path.getmtime(p) if os.path.isfile(p) else 0
               for p in self.progress_watch]
        if cur != getattr(self, "_last_mt", None):
            self._last_mt = cur
            self.last_progress = time.time()
        # disk guard
        fg = free_gb()
        if fg >= 0 and fg < self.min_free_gb:
            raise Aborted("disk_low", stage=st["name"],
                          free_gb=round(fg, 2), min_free_gb=self.min_free_gb)
        # training-curve guard: the trainer's own steps.csv must stay finite
        if st["name"].endswith("_train"):
            self._check_train_files(st)
            fail = os.path.join(self.run_root, st["name"].split("_")[0],
                                "TRAIN_FAILED.json")
            if os.path.isfile(fail):
                raise Aborted("trainer_aborted_by_guard", stage=st["name"],
                              detail=load_json(fail, {}))
        # stall guard
        idle = time.time() - self.last_progress
        if idle > self.stall_s:
            raise Aborted("stalled", stage=st["name"], idle_s=round(idle, 1),
                          stall_s=self.stall_s)
        self.write_status()

    def _check_train_files(self, st):
        d = os.path.join(self.run_root, st["name"].split("_")[0])
        p = os.path.join(d, "steps.csv")
        if not os.path.isfile(p):
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
        except OSError:
            return
        if len(lines) < 2:
            return
        head = lines[0].split(",")
        try:
            row = dict(zip(head, lines[-1].split(",")))
            for k in ("loss", "focal", "dice", "gnorm"):
                v = float(row.get(k, "nan"))
                if v != v or v in (float("inf"), float("-inf")):
                    raise Aborted("non_finite_in_steps_csv", stage=st["name"],
                                  field=k, value=row.get(k))
        except (ValueError, IndexError):
            return

    # ------------------------------------------------------------ postchecks
    def post(self, st):
        n = st["name"]
        if n == "preflight_tests":
            return
        if n.startswith("cache_"):
            split = n.split("_")[1]
            idx = load_json(os.path.join(CACHE_ROOT, "index_%s.json" % split)) or {}
            if not idx.get("complete"):
                raise Aborted("cache_incomplete", stage=n,
                              index=os.path.join(CACHE_ROOT, "index_%s.json" % split))
            st["note"] = ("images=%d tiles=%d cand=%d bytes=%.1fMB MA_ub=%.4f HE_ub=%.4f"
                          % (idx["n_images"], idx["counts"]["n_tiles"],
                             idx["counts"]["n_candidate_tiles"],
                             idx["storage"]["total_npz_bytes"] / 1e6,
                             idx["counts"]["ma_recall_upper_bound"],
                             idx["counts"]["he_recall_upper_bound"]))
            return
        if n.endswith("_train"):
            d = os.path.join(self.run_root, n.split("_")[0])
            fail = load_json(os.path.join(d, "TRAIN_FAILED.json"))
            if fail:
                raise Aborted("trainer_aborted_by_guard", stage=n, detail=fail)
            done = load_json(os.path.join(d, "done.json"))
            if not done:
                raise Aborted("train_done_json_missing", stage=n, dir=d)
            if not (done.get("identity_ok") or {}).get("bitwise"):
                raise Aborted("identity_not_bitwise", stage=n,
                              identity=done.get("identity_ok"))
            want = int(done.get("epochs") or 0)
            hb = load_json(os.path.join(d, "heartbeat.json")) or {}
            if int(hb.get("epoch1") or 0) < want:
                raise Aborted("epochs_not_completed", stage=n,
                              last_epoch1=hb.get("epoch1"), epochs=want)
            for ep in (st.get("nodes") or []):
                cp = os.path.join(d, "nodes", "node_ep%03d.pth" % ep)
                if not os.path.isfile(cp):
                    raise Aborted("node_checkpoint_missing", stage=n, ckpt=cp)
            st["note"] = ("epochs=%s steps=%s loss_first=%s loss_last=%s "
                          "focal_share_mean=%.4f identity_bitwise=%s"
                          % (done.get("epochs"), done.get("steps"),
                             done.get("loss_first"),
                             (done.get("final") or {}).get("loss"),
                             float(done.get("focal_share_mean") or 0.0), True))
            return
        if n.startswith("eval_"):
            out = st["cmd"][st["cmd"].index("--out") + 1]
            rep = load_json(out)
            if not rep:
                raise Aborted("eval_report_missing", stage=n, path=out)
            if not rep.get("accepted"):
                raise Aborted("eval_invariant_violation", stage=n,
                              bitwise=rep.get("bitwise_checks"),
                              contract=rep.get("contract_selfcheck"))
            cs = rep.get("contract_selfcheck")
            if cs and not cs.get("pass"):
                raise Aborted("contract_selfcheck_failed", stage=n, contract=cs)
            st["note"] = ("refined mIoU=%.6f MA=%.6f HE=%.6f  champ mIoU=%.6f "
                          "dMA=%+.6f dHE=%+.6f dIoU=%+.6f"
                          % (rep["refined"]["mIoU"], rep["refined"]["per_class"]["MA"],
                             rep["refined"]["per_class"]["HE"],
                             rep["champion"]["mIoU"],
                             rep["delta"]["per_class"]["MA"],
                             rep["delta"]["per_class"]["HE"],
                             rep["delta"]["mIoU"]))

    # ------------------------------------------------------------ selection
    def build_selection(self):
        evals = []
        for st in self.stages:
            if st["kind"] != "eval" or st["state"] != "DONE":
                continue
            out = st["cmd"][st["cmd"].index("--out") + 1]
            rep = load_json(out) or {}
            evals.append({
                "stage": st["stage_key"], "epoch": st["epoch"], "node": st["name"],
                "ckpt": st["ckpt"], "report": out,
                "champion": rep.get("champion", {}).get("per_class", {}),
                "refined": rep.get("refined", {}).get("per_class", {}),
                "mIoU_champion": rep.get("champion", {}).get("mIoU"),
                "mIoU_refined": rep.get("refined", {}).get("mIoU"),
                "delta": rep.get("delta", {}),
            })
        if not evals:
            self.selection = {"done": True, "verdict": "NO_EVAL",
                              "reason": "no completed eval node"}
            return
        champ = evals[0]["champion"]
        cma, che = champ.get("MA"), champ.get("HE")
        if cma is None or che is None:
            self.selection = {"done": True, "verdict": "NO_EVAL",
                              "reason": "eval reports carry no champion per-class IoU"}
            return
        spread = max(abs(e["champion"].get("MA", 0) - cma) for e in evals)
        if spread > 1e-6:
            self.alert("champion_inconsistent_across_evals", "champion MA drifted",
                       spread=spread)
        for e in evals:
            e["beats_champion_ma"] = bool(e["refined"].get("MA", 0) > cma)
            e["beats_champion_he"] = bool(e["refined"].get("HE", 0) > che)
            e["win"] = bool(e["beats_champion_ma"] and e["beats_champion_he"])
        winners = [e for e in evals if e["win"]]
        best = max(winners or evals, key=lambda e: e["mIoU_refined"] or -1)
        dest = os.path.join(self.run_root,
                            "best.pth" if winners else "best_for_analysis.pth")
        try:
            shutil.copyfile(best["ckpt"], dest)
        except OSError as ex:
            self.alert("best_copy_failed", str(ex), src=best["ckpt"])
            dest = None
        self.selection = {
            "done": True,
            "schema": "hrdecoder-refiner-selection/v1",
            "champion": {"MA": cma, "HE": che,
                         "mIoU": evals[0].get("mIoU_champion")},
            "contract": CONTRACT,
            "rule": ("promote only if val MA IoU AND val HE IoU both exceed the "
                     "champion on the assembled map at the fixed p>0.5; among "
                     "winners take the highest mIoU. If there is no winner, keep "
                     "the highest-mIoU node for failure analysis and do NOT call "
                     "it a win."),
            "verdict": "WIN" if winners else "NO_WIN",
            "n_nodes": len(evals), "n_winners": len(winners),
            "best_node": best["node"], "best_ckpt": best["ckpt"],
            "best_epoch": best["epoch"], "best_stage": best["stage"],
            "best_mIoU_refined": best["mIoU_refined"],
            "best_delta": best["delta"],
            "is_promoted": bool(winners),
            "artefact": dest,
            "note": ("" if winners else
                     "NO node improved BOTH classes: this is NOT a win. The "
                     "highest-mIoU node is preserved for failure analysis only."),
            "nodes": [{k: e[k] for k in
                       ("node", "stage", "epoch", "ckpt", "report", "champion",
                        "refined", "mIoU_champion", "mIoU_refined", "delta",
                        "beats_champion_ma", "beats_champion_he", "win")}
                      for e in evals],
        }
        atomic_json(os.path.join(self.run_root, "selection.json"),
                    self.selection)
        print("[sup] SELECTION verdict=%s best=%s"
              % (self.selection["verdict"], self.selection["best_node"]), flush=True)

    # ---------------------------------------------------------------- main
    def _live_supervisor(self):
        """PID from a previous supervisor, if that process is still alive."""
        p = os.path.join(self.run_root, "supervisor.pid")
        try:
            with open(p, encoding="utf-8") as f:
                pid = int(f.read().strip())
        except (OSError, ValueError):
            return None
        try:
            os.kill(pid, 0)
        except OSError:
            return None
        return pid

    def run(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_root, "eval"), exist_ok=True)
        if self.args.resume:
            prev = self._live_supervisor()
            if prev is not None and prev != os.getpid():
                raise SystemExit("refusing --resume: supervisor pid %d is still "
                                 "alive on %s" % (prev, self.run_root))
            st_prev = load_json(self.status_path) or {}
            print("[sup] RESUME %s: previous state=%s stages_done=%s; stages "
                  "whose artefacts exist are SKIPPED (caches by "
                  "index_*.json complete=true, training by done.json), nothing "
                  "is deleted or rebuilt"
                  % (self.run_root, st_prev.get("state"),
                     st_prev.get("stages_done")), flush=True)
        # authoritative PID file: written by the supervisor itself, so it stays
        # correct even when the launcher used setsid (which may fork).
        self.pid_path = os.path.join(self.run_root, "supervisor.pid")
        with open(self.pid_path, "w", encoding="utf-8") as f:
            f.write("%d\n" % os.getpid())
        self.build_stages()
        if self.args.dry_run:
            for s in self.stages:
                print("%2d %-22s %-4s %s" % (s["index"], s["name"], s["kind"],
                                             " ".join(s["cmd"])))
            return 0
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._sig)
        self.state = "RUNNING"
        self.write_status()
        print("[sup] started pid=%d run_root=%s stages=%d disk_free=%.1fGB"
              % (os.getpid(), self.run_root, len(self.stages), free_gb()), flush=True)
        for st in self.stages:
            if self.stop_requested or os.path.isfile(self.stop_flag):
                self.stop_requested = True
                break
            if st.get("skip"):
                st["state"] = "SKIPPED"
                st["note"] = "already present: %s" % st["skip"]
                print("[sup] skip %s (%s)" % (st["name"], st["note"]), flush=True)
                self.write_status()
                continue
            try:
                rc = self.run_stage(st)
                st["t_end"] = iso()
                st["elapsed_s"] = round(time.time() -
                                        time.mktime(time.strptime(st["t_start"],
                                                                   "%Y-%m-%dT%H:%M:%S")), 1)
                if rc != 0:
                    st["state"] = "FAILED"
                    self.cur = st
                    raise Aborted("stage_exit_nonzero", stage=st["name"], exit_code=rc)
                self.post(st)
                st["state"] = "DONE"
                print("[sup] DONE %s rc=0 %.1fs %s"
                      % (st["name"], st["elapsed_s"], st["note"]), flush=True)
                self.write_status()
            except Aborted as a:
                st["t_end"] = iso()
                self.cur = st
                self.alert(a.reason, "stage %s aborted" % st["name"], **a.where)
                if a.reason == "stop_requested":
                    st["state"] = "STOPPED"
                    self.state = "STOPPED"
                else:
                    st["state"] = "FAILED"
                    self.state = "FAILED"
                self.exit_code = 0 if a.reason == "stop_requested" else 1
                self.write_status()
                return self.exit_code
        self.cur = None
        self.build_selection()
        self.state = "STOPPED" if self.stop_requested else "DONE"
        self.exit_code = 0
        self.write_status()
        if self.state == "DONE":
            self.write_report()
        print("[sup] finished state=%s exit=%d" % (self.state, self.exit_code),
              flush=True)
        return self.exit_code

    def write_report(self):
        sel = self.selection or {}
        ch = sel.get("champion") or {}
        lines = [
            "# refiner_v1 FORMAL RUN — supervisor report",
            "",
            "run root : `%s`" % self.run_root,
            "pid      : %s" % os.getpid(),
            "state    : %s" % self.state,
            "",
            "| stage | kind | exit | elapsed | note |",
            "|---|---|---|---|---|",
        ]
        for s in self.stages:
            lines.append("| %s | %s | %s | %ss | %s |"
                         % (s["name"], s["kind"], s["exit_code"],
                            s["elapsed_s"], (s["note"] or "").replace("|", "/")))
        lines += ["", "## selection", "",
                  "verdict: **%s** (best node %s)" % (sel.get("verdict"),
                                                      sel.get("best_node")),
                  "",
                  "`%s`" % sel.get("rule", ""), "",
                  "champion val: MA=%s HE=%s mIoU=%s"
                  % (ch.get("MA"), ch.get("HE"), ch.get("mIoU")),
                  "",
                  "objective: %s"
                  % ((self.cfg.get("loss") or {}).get("objective_string", "?"),), ""]
        with open(os.path.join(self.run_root, "RUN_REPORT.md"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=RUN_ROOT)
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--min-free-gb", type=float, default=4.0)
    ap.add_argument("--stall-s", type=float, default=1800.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: refuse if another "
                         "supervisor is alive, otherwise skip every stage whose "
                         "artefact already exists (caches, trained stages) and "
                         "restart at the first unfinished stage")
    args = ap.parse_args()
    return Supervisor(args).run()


if __name__ == "__main__":
    sys.exit(main())
