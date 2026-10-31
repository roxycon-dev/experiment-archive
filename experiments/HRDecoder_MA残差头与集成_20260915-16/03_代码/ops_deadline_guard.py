#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立硬截止关机保护。

与任何训练进程无关：到点必然关机。
- 支持绝对截止时间（本地时区字符串或 unix 秒）
- 支持"训练完成即关机"：监测 state.json 的 phase == complete
- 每次判定都写日志，便于事后审计
- 关机通过 `bash /usr/bin/shutdown`（AutoDL 容器脚本，必须 bash 调用）
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

CST = timezone(timedelta(hours=8))
LOG = "/root/autodl-tmp/hrplus/ops/deadline_guard.log"
STATE = "/root/autodl-tmp/hrplus/results/L_0915/state.json"
POLL = 30


def log(msg):
    line = "[%s] %s" % (datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S"), msg)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def shutdown(reason):
    log("SHUTDOWN_TRIGGERED reason=%s" % reason)
    try:
        r = subprocess.run(["/bin/bash", "/usr/bin/shutdown"],
                           timeout=60, check=False)
        log("shutdown rc=%d" % r.returncode)
    except Exception as exc:
        log("shutdown error %r" % (exc,))


def main():
    deadline = float(sys.argv[1])          # unix seconds
    watch_complete = len(sys.argv) > 2 and sys.argv[2] == "watch"
    log("guard start deadline=%s watch_complete=%s pid=%d" % (
        datetime.fromtimestamp(deadline, CST).strftime("%Y-%m-%d %H:%M:%S"),
        watch_complete, os.getpid()))

    while True:
        now = time.time()
        if now >= deadline:
            shutdown("deadline")
            return
        if watch_complete and os.path.isfile(STATE):
            try:
                with open(STATE, encoding="utf-8") as f:
                    st = json.load(f)
                if st.get("phase") == "complete":
                    # 完成后留 120s 给备份与日志落盘
                    log("state complete detected, grace 120s")
                    time.sleep(120)
                    if time.time() < deadline:
                        shutdown("train_complete")
                        return
            except Exception:
                pass
        time.sleep(POLL)


if __name__ == "__main__":
    main()
