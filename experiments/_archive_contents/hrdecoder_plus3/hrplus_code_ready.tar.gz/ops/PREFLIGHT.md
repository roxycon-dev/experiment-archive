# Preflight — mandatory BEFORE every guard launch

## Hard rules
- **Platform absolute OFF before launch.** Verify the machine is fully powered off
  before launch. An SSH disconnect is **NOT proof of OFF** — confirm on the
  platform console (provider status / forced stop), not via network probes.
- **External platform timer is separate and mandatory.** Set the platform's own
  auto-stop timer for the same deadline. The guard is a second line of defense,
  not a substitute. If the network drops, only the platform timer can guarantee OFF.
- **Stop on idle or failure.** Do not keep a session running while idle or after
  a failed stage. Terminate and save, then power off.
- **Backup barrier <= 90 seconds.** Weights and logs are saved via the separate
  backup step with a hard 90s barrier — never indefinite. If backup exceeds 90s,
  abort the backup and power off (partial checkpoint > runaway session).
- **Each launch max 12h** (budget cap 23.76/card-launch; cumulative plan cap 44
  new card-hours, 87.12 max). Guard enforces 12h; platform timer is the backstop.
- No background schedulers, no cloud APIs, no paid storage, no auto installs or
  paid services. Guard saves no credentials and performs no network calls.

## Arming
- Guard is dry-run unless `--execute-shutdown` is passed. Only arm when the
  platform timer is confirmed set.
- Deadline must be ISO with timezone: `--deadline-utc 2024-01-01T00:00:00+00:00`.
- Preflight rejects: missing/naive/expired deadline, deadline >12h from now.

## Stage plan
1. **Stage 1 smoke: <= 15 min.** Validate data pipeline + one training step +
   checkpoint save before any long run.
2. **Train D 20k hard: 10h incl. val.**
3. **Train L 20k ONLY if remaining time budget permits and reviewed.**
4. **No test run until val gate 42.4 passes and supervisor locks the winner.**
   (Target test 45.20 is not guaranteed.)

## Remaining GPU checks before launch
- [ ] `nvidia-smi`: single 4090 visible, no stray processes, ECC/temp OK
- [ ] driver/CUDA runtime versions match the training image
- [ ] GPU memory headroom confirmed for batch size
- [ ] `nvidia-smi -q`: power limit and persistence mode sane
- [ ] Disk: checkpoint + log space >= expected run needs
- [ ] Clock synced (UTC) so the ISO deadline is meaningful
- [ ] Platform auto-stop timer set and verified (screenshot/console check)
- [ ] Backup target writable; backup barrier timeout configured <= 90s

## Launch
```bash
python3 ops/guard.py \
  --deadline-utc <NOW+<=12h, ISO UTC> \
  --log /var/log/guard/run.log \
  [--execute-shutdown]  # ONLY if platform timer verified \
  -- <command and args>
```
Exit codes: child exit propagated; 124 timeout; 130 interrupted; 2 preflight reject;
125 wrapper error.
