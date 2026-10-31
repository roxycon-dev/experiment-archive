"""Local helper: push files / run commands on the single allowed RTX3090 box.

Usage:
  python refiner_v1/tools/rc.py mkdir <remote_dir> ...
  python refiner_v1/tools/rc.py put <local> <remote> [<local2> <remote2> ...]
  python refiner_v1/tools/rc.py get <remote> <local> [...]
  python refiner_v1/tools/rc.py run "<shell command>"
  python refiner_v1/tools/rc.py py <remote_script_abs_path> [args...]
"""
from __future__ import annotations

import os
import stat
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis"))
from _remote import connect, run  # noqa: E402

PY = "/root/miniconda3/bin/python"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == "mkdir":
        c = connect()
        for d in sys.argv[2:]:
            run("mkdir -p %s" % d, client=c)
        c.close()
        print("mkdir ok")
        return 0
    if cmd == "put":
        args = sys.argv[2:]
        if len(args) % 2:
            raise SystemExit("put needs local/remote pairs")
        c = connect(); sf = c.open_sftp()
        for i in range(0, len(args), 2):
            sf.put(args[i], args[i + 1])
            print("put %s -> %s" % (args[i], args[i + 1]))
        sf.close(); c.close()
        return 0
    if cmd == "putd":
        # putd <local_dir> <remote_dir>  (recursive, text/bin)
        ld, rd = sys.argv[2], sys.argv[3]
        c = connect(); sf = c.open_sftp()

        def ensure(d):
            parts, cur = d.strip("/").split("/"), ""
            for p in parts:
                cur += "/" + p
                try:
                    sf.stat(cur)
                except IOError:
                    sf.mkdir(cur)

        for root, _dirs, files in os.walk(ld):
            rel = os.path.relpath(root, ld).replace("\\", "/")
            target = rd if rel == "." else rd + "/" + rel
            ensure(target)
            for f in files:
                if f.endswith(".pyc"):
                    continue
                sf.put(os.path.join(root, f), target + "/" + f)
        sf.close(); c.close()
        print("putd %s -> %s" % (ld, rd))
        return 0
    if cmd == "get":
        args = sys.argv[2:]
        c = connect(); sf = c.open_sftp()
        for i in range(0, len(args), 2):
            try:
                sf.get(args[i], args[i + 1])
                print("get %s -> %s" % (args[i], args[i + 1]))
            except IOError as e:
                print("MISS %s (%s)" % (args[i], e))
        sf.close(); c.close()
        return 0
    if cmd == "getd":
        rd, ld = sys.argv[2], sys.argv[3]
        c = connect(); sf = c.open_sftp()

        def walk(rdir, ldir):
            os.makedirs(ldir, exist_ok=True)
            for a in sf.listdir_attr(rdir):
                rp, lp = rdir + "/" + a.filename, os.path.join(ldir, a.filename)
                if stat.S_ISDIR(a.st_mode):
                    walk(rp, lp)
                else:
                    try:
                        sf.get(rp, lp)
                        print("get %s" % lp)
                    except IOError as e:
                        print("MISS %s (%s)" % (rp, e))

        walk(rd, ld)
        sf.close(); c.close()
        return 0
    if cmd == "run":
        rc, out, err = run(sys.argv[2], timeout=7200)
        sys.stdout.write(out)
        if err.strip():
            sys.stderr.write("--- STDERR ---\n" + err)
        print("\n[remote_exit=%d]" % rc)
        return 0
    if cmd == "py":
        script = sys.argv[2]
        rest = " ".join(sys.argv[3:])
        rc, out, err = run("cd /root/autodl-tmp/HRDecoder_FullCompetition_20260920 && "
                           "%s %s %s" % (PY, script, rest), timeout=7200)
        sys.stdout.write(out)
        if err.strip():
            sys.stderr.write("--- STDERR ---\n" + err)
        print("\n[remote_exit=%d]" % rc)
        return 0
    raise SystemExit("unknown cmd %r" % cmd)


if __name__ == "__main__":
    sys.exit(main())
