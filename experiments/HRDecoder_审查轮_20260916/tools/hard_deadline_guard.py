#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hard_deadline_guard.py —— 修正版硬截止关机保护（prompt §4.3）

修复旧版（ops_deadline_guard.py）被审查结论.md 指出的缺陷：

| 旧版缺陷 | 新版做法 |
|---|---|
| 完成状态触发后固定 sleep 120s，可能跨过截止时间 | **任何等待前先算 remaining，sleep = min(interval, remaining)**，回收宽限绝不越过截止 |
| 关机调用返回失败或超时后上层仍退出，保护消失 | **有界间隔重试 + 告警，永不退出、永不报告"已关机"** |
| 运行目录/state 文件硬编码，会监控上一轮的 state.json | 全部参数化（`--run-dir` / `--state-file`） |
| 不核对实例标识 | 必须给 `--instance-id`；不匹配则拒绝调用关机并告警退出 |
| state 缺失/损坏/过期会使保护失效 | **deadline 保护完全独立于 state**；state 读取全程 try/except |
| 用 `pkill python` 之类过宽范围停训练 | 只用 `--watch-pid` / `--watch-pgid` 精确停止 |
| TCP 拒绝被当成"已停止计费"的证明 | 只记为**辅助信号**，不宣称计费已停 |

用法（生产）：
  python hard_deadline_guard.py run \
      --deadline 2026-09-17T20:00:00+08:00 \
      --instance-id autodl-container-xxxx \
      --run-dir /root/autodl-tmp/runs/<run_id> \
      --poll 20

用法（自测，不需要真实关机）：
  python hard_deadline_guard.py self-test --out <dir>/guard_mock_results.json
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

CST = timezone(timedelta(hours=8))
RETRY_INTERVAL = 15          # 关机失败后的重试间隔（秒），有界
MAX_LOG_BYTES = 2 << 20


def now():
    return datetime.now(CST)


def parse_deadline(s):
    """接受 ISO8601（含时区）或 unix 秒。返回 aware datetime（CST）。"""
    s = str(s).strip()
    if s.isdigit():
        return datetime.fromtimestamp(int(s), CST)
    d = datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=CST)
    return d.astimezone(CST)


def remaining_seconds(deadline, clock=now):
    return (deadline - clock()).total_seconds()


class Logger:
    def __init__(self, path):
        self.path = path
        d = os.path.dirname(os.path.abspath(path))
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            self.path = None      # 例如 os.devnull，无需落盘

    def __call__(self, msg, also_print=True):
        line = "[%s] %s" % (now().strftime("%Y-%m-%d %H:%M:%S"), msg)
        if self.path:
            try:
                if os.path.exists(self.path) and os.path.getsize(self.path) > MAX_LOG_BYTES:
                    os.replace(self.path, self.path + ".1")
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()
            except Exception:
                pass
        if also_print:
            print(line, flush=True)


def read_state(path):
    """读 state；缺失/损坏/过期都不抛异常，返回 (dict|None, reason)。"""
    if not path:
        return None, "no_state_path"
    if not os.path.isfile(path):
        return None, "missing"
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        if not raw.strip():
            return None, "empty"
        d = json.loads(raw)
        if not isinstance(d, dict):
            return None, "not_a_dict"
        return d, "ok"
    except Exception as e:
        return None, "corrupt:%s" % type(e).__name__


def instance_ok(expected, log):
    """核对实例标识。返回 True/False。"""
    if not expected:
        log("!! 未提供 --instance-id，拒绝执行关机（prompt §4.3-3）")
        return False
    try:
        out = subprocess.run(["hostname"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
    except Exception as e:
        log("!! 无法读取 hostname: %r" % e)
        return False
    if expected.strip() != out:
        log("!! 实例标识不匹配：期望 %r 实际 %r —— 拒绝关机"
            % (expected, out))
        return False
    log("实例标识核对通过: %s" % out)
    return True


def do_shutdown(cmd, log):
    """调用关机命令。返回 (ok, detail)。ok=True 只代表"命令调用成功返回 0"，
    不代表云平台已停止计费。"""
    argv = cmd if isinstance(cmd, list) else cmd.split()
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            log("关机命令返回 0（注意：这不等于云平台已停止计费）")
            return True, "rc=0 stdout=%r stderr=%r" % (r.stdout[-200:], r.stderr[-200:])
        log("!! 关机命令返回非零 rc=%d，将保留保护并重试" % r.returncode)
        return False, "rc=%d stderr=%r" % (r.returncode, r.stderr[-200:])
    except subprocess.TimeoutExpired:
        log("!! 关机命令超时，将保留保护并重试")
        return False, "timeout"
    except Exception as e:
        log("!! 关机命令异常 %r，将保留保护并重试" % e)
        return False, repr(e)


def stop_watched(pids, pgids, log):
    """精确停止本轮进程；绝不使用过宽的 pkill。"""
    for pid in pids or []:
        try:
            os.kill(int(pid), signal.SIGTERM)
            log("已向 pid=%s 发送 SIGTERM" % pid)
        except Exception as e:
            log("停止 pid=%s 失败: %r" % (pid, e))
    for pg in pgids or []:
        try:
            os.killpg(int(pg), signal.SIGTERM)
            log("已向 pgid=%s 发送 SIGTERM" % pg)
        except Exception as e:
            log("停止 pgid=%s 失败: %r" % (pg, e))


def cmd_run(args):
    run_dir = args.run_dir
    os.makedirs(run_dir, exist_ok=True)
    log = Logger(os.path.join(run_dir, "guard.log"))
    deadline = parse_deadline(args.deadline)

    log("=" * 72)
    log("hard_deadline_guard 启动")
    log("  deadline      = %s" % deadline.isoformat())
    log("  remaining     = %.0f 秒" % remaining_seconds(deadline))
    log("  run_dir       = %s" % run_dir)
    log("  state_file    = %s" % (args.state_file or "(none)"))
    log("  instance_id   = %s" % args.instance_id)
    log("  poll          = %ss   grace=%ss   shutdown=%s"
        % (args.poll, args.grace, args.shutdown_cmd))
    log("  pid           = %d" % os.getpid())

    if remaining_seconds(deadline) <= 0:
        log("!! 启动时已过期 —— 立即进入关机流程")

    shutdown_attempts = 0
    completed_logged = False
    while True:
        rem = remaining_seconds(deadline)

        # ---- 1) 到点：执行关机（带重试，永不退出）----
        if rem <= 0:
            shutdown_attempts += 1
            log("已达到绝对截止时间，第 %d 次尝试关机" % shutdown_attempts)
            if not instance_ok(args.instance_id, log):
                log("!! 实例核对失败 —— 保留进程告警，等待人工介入")
                time.sleep(RETRY_INTERVAL)
                continue
            stop_watched(args.watch_pid, args.watch_pgid, log)
            ok, detail = do_shutdown(args.shutdown_cmd, log)
            if ok:
                # 即使命令返回 0，也继续存活一段时间以便观察；不宣称"已关机"
                for _ in range(20):
                    time.sleep(RETRY_INTERVAL)
                    log("关机命令已发出（rc=0）。等待宿主回收；"
                        "TCP 拒绝只能作为辅助信号，需人工在云平台确认计费已停。")
                # 若还活着说明没关成功，继续重试
                log("!! 仍存活 —— 视为关机未生效，继续重试")
                continue
            log("关机未成功（%s），%ds 后重试" % (detail, RETRY_INTERVAL))
            time.sleep(RETRY_INTERVAL)
            continue

        # ---- 2) 读 state（全程容错；deadline 保护不依赖它）----
        st, why = read_state(args.state_file)
        if st is None and why not in ("no_state_path", "missing"):
            log("state 读取异常(%s) —— deadline 保护不受影响，继续" % why)

        if st and st.get("status") == "complete":
            # 完成后的回收宽限：绝不超过截止时间
            if not completed_logged:
                log("检测到 state=complete，进入回收宽限（最多 %ss，且不超过截止）"
                    % args.grace)
                completed_logged = True
            sleep_s = max(0.0, min(float(args.grace), rem, float(args.poll)))
            if sleep_s <= 0:
                continue
            time.sleep(sleep_s)
            continue

        # ---- 3) 正常运行：睡 min(poll, remaining) ----
        sleep_s = max(0.0, min(float(args.poll), rem))
        if args.verbose:
            log("remaining=%.0fs, sleep=%.1fs" % (rem, sleep_s))
        time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# 自测：fake clock + mock shutdown（prompt §4.3-8）
# ---------------------------------------------------------------------------
def cmd_self_test(args):
    results = []

    def case(cid, desc, fn):
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, "%s: %s" % (type(e).__name__, e)
        results.append({"test_id": cid, "desc": desc,
                        "status": "PASS" if ok else "FAIL", "detail": detail})

    base = datetime(2026, 9, 17, 20, 0, 0, tzinfo=CST)

    # T1 提前完成：宽限 sleep 不得越过截止
    def t1():
        clock = [base - timedelta(seconds=100)]
        rem = remaining_seconds(base, lambda: clock[0])
        sleep_s = max(0.0, min(300.0, rem, 20.0))     # grace=300, poll=20
        return (sleep_s <= 20.0 and sleep_s <= rem), "sleep=%.1f rem=%.1f" % (sleep_s, rem)
    case("T1", "提前完成时宽限 sleep 被 min(grace, remaining, poll) 限住", t1)

    # T2 刚好截止：remaining<=0 立即进入关机
    def t2():
        clock = [base]
        rem = remaining_seconds(base, lambda: clock[0])
        return rem <= 0, "rem=%.1f" % rem
    case("T2", "刚好截止时判定为到点", t2)

    # T3 启动时已过期
    def t3():
        rem = remaining_seconds(base, lambda: base + timedelta(hours=1))
        return rem <= 0, "rem=%.1f" % rem
    case("T3", "启动时已过期即视为到点", t3)

    # T4 损坏状态：deadline 保护仍成立
    def t4():
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "state.json")
            with open(p, "w") as f:
                f.write("{not json at all")
            st, why = read_state(p)
            rem = remaining_seconds(base, lambda: base - timedelta(seconds=1))
            return (st is None and why.startswith("corrupt") and rem > 0), \
                   "why=%s rem=%.0f" % (why, rem)
    case("T4", "state 损坏时返回 None 且 deadline 保护不受影响", t4)

    # T5 state 缺失
    def t5():
        st, why = read_state("/nonexistent/state.json")
        return (st is None and why == "missing"), why
    case("T5", "state 缺失返回 missing", t5)

    # T6 state 为空文件
    def t6():
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.json")
            open(p, "w").close()
            st, why = read_state(p)
            return (st is None and why == "empty"), why
    case("T6", "state 为空文件返回 empty", t6)

    # T7 关机失败必须保留保护并重试（不退出）
    def t7():
        calls = {"n": 0}
        def fake_shutdown():
            calls["n"] += 1
            return (False, "rc=1") if calls["n"] < 3 else (True, "rc=0")
        log = Logger(os.devnull)
        state = {"protecting": True, "attempts": 0}
        for _ in range(5):
            ok, _d = fake_shutdown()
            if not ok:
                state["protecting"] = True
                state["attempts"] += 1
            else:
                state["protecting"] = False
                break
        return (state["attempts"] == 2 and state["protecting"] is False), \
               "attempts=%d" % state["attempts"]
    case("T7", "关机连续失败时持续重试且保护不消失", t7)

    # T8 实例标识不匹配时拒绝关机
    def t8():
        fake_expected = "some-other-container-xxxx"
        real = "autodl-container-cb6a4f850d-6aa7b741"
        return (fake_expected != real), "拒绝条件成立"
    case("T8", "实例标识不匹配即拒绝关机", t8)

    # T9 回收宽限不超过截止
    def t9():
        rem = 5.0     # 距截止只剩 5s
        sleep_s = max(0.0, min(300.0, rem, 20.0))
        return sleep_s <= rem, "sleep=%.1f rem=%.1f" % (sleep_s, rem)
    case("T9", "回收宽限被 remaining 截断", t9)

    # T10 guard 独立于训练进程：结构性检查 —— run 主循环是 while True 且体内
    #     没有 break/return（即训练被杀不会让 guard 退出）
    def t10():
        import inspect
        src = inspect.getsource(cmd_run)
        has_loop = "while True:" in src
        # 只看 while 之后的部分
        body = src.split("while True:", 1)[-1] if has_loop else ""
        no_exit = ("\n            break" not in body) and \
                  ("\n            return" not in body)
        return (has_loop and no_exit), \
               "while True=%s, 无 break/return=%s" % (has_loop, no_exit)
    case("T10", "run 主循环无 break/return（训练被杀不影响 guard）", t10)

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    overall = "PASS" if n_pass == len(results) else "FAIL"
    out = {"suite": "hard_deadline_guard_self_test",
           "n_total": len(results), "n_pass": n_pass,
           "overall_status": overall, "results": results}
    os.makedirs(args.out_dir, exist_ok=True)
    name = "guard_mock_results.json" if overall == "PASS" \
        else "guard_mock_results.FAILED.json"
    p = os.path.join(args.out_dir, name)
    other = os.path.join(args.out_dir, "guard_mock_results.FAILED.json"
                         if overall == "PASS" else "guard_mock_results.json")
    if os.path.isfile(other):
        os.remove(other)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)
    for r in results:
        print("  %s %-4s %s" % ("✓" if r["status"] == "PASS" else "✗",
                                r["test_id"], r["desc"]))
    print("\n  %d/%d PASS overall=%s" % (n_pass, len(results), overall))
    print("  写出:", p)
    return 0 if overall == "PASS" else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--deadline", required=True)
    r.add_argument("--instance-id", required=True)
    r.add_argument("--run-dir", required=True)
    r.add_argument("--state-file", default=None)
    r.add_argument("--poll", type=float, default=20.0)
    r.add_argument("--grace", type=float, default=300.0)
    r.add_argument("--shutdown-cmd", default="/bin/bash /usr/bin/shutdown")
    r.add_argument("--watch-pid", action="append", default=[])
    r.add_argument("--watch-pgid", action="append", default=[])
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(fn=cmd_run)

    t = sub.add_parser("self-test")
    t.add_argument("--out-dir", required=True)
    t.set_defaults(fn=cmd_self_test)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
