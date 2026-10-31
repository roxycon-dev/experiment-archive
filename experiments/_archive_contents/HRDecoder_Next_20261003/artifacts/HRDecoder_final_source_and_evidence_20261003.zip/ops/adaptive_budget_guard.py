"""Optional authorized budget decision at a completed, matched validation node.

No training code or protocol identity changes. All model/optimizer checkpoints
must be intact before the old controller is replaced by the same controller.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nextlib.common import read_json, write_json, digest, sha, trusted_load, source_identity
from nextlib.decision import best_candidate, seed_gate
from analyze_cache import image_bootstrap


def processes():
    found = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            status = (p/'status').read_text()
            parent = int(next(x for x in status.splitlines() if x.startswith('PPid:')).split()[1])
            cmd = (p/'cmdline').read_bytes().replace(b'\0', b' ').decode()
            found[int(p.name)] = (parent, cmd)
        except (OSError, StopIteration, ValueError):
            pass
    return found


def descendants(table, parent):
    result = {parent}
    while True:
        additions = {pid for pid, (ppid, _) in table.items() if ppid in result}
        if additions <= result:
            return result
        result |= additions


def main():
    policy = read_json(ROOT/'reports/adaptive_budget_policy.json')
    expected_source = policy['source_sha256']
    assert source_identity()['sha256'] == expected_source
    root = ROOT/'work/M_seed42'
    controller = int(read_json(ROOT/'reports/continuation_pid.json')['pid'])
    checkpoint_verified = False
    paused = []
    receipt_path = ROOT/'reports/adaptive_budget_decision.json'
    deadline = time.time()+3600
    while time.time() < deadline:
        if (root/'selection.json').exists():
            write_json(receipt_path, dict(status='NO_ACTION_ALREADY_COMPLETE', time=time.time()))
            return
        histories = {arm: read_json(root/arm/'history.json') for arm in ('M0', 'M1')}
        if not all(any(row['update'] == 2000 for row in h) for h in histories.values()):
            time.sleep(1)
            continue
        # Last is rewritten AFTER full validation, selection and cache pruning.
        if not all((root/arm/'last.pth').stat().st_mtime_ns >
                   (root/arm/'history.json').stat().st_mtime_ns for arm in histories):
            time.sleep(.25)
            continue
        baseline = read_json(ROOT/'work/G/current_deployment/metrics.json')
        base_counts = {x['sid']: x['counts'] for x in baseline['per_image']}
        evidence = {}
        for arm, history in histories.items():
            before = next(x for x in history if x['update'] == 1000)
            current = next(x for x in history if x['update'] == 2000)
            mode = max(('solo', 'fixed_half'), key=lambda k: current[k]['mIoU_lesion4'])
            now_counts = {x['sid']: x['counts'] for x in current[mode]['per_image']}
            assert set(now_counts) == set(base_counts) and len(now_counts) == 76
            ids = sorted(now_counts)
            ci = image_bootstrap([now_counts[x] for x in ids], [base_counts[x] for x in ids])
            best = best_candidate(history)
            previous = max(before[k]['mIoU_lesion4'] for k in ('solo', 'fixed_half'))
            now = current[mode]['mIoU_lesion4']
            qualifies = (best['score'] <= baseline['mIoU_lesion4']-policy['minimum_gap'] and
                         now <= previous+policy['trend_epsilon'] and ci['percentile95'][1] < 0)
            evidence[arm] = dict(best=best, current=now, recent_delta=now-previous,
                                 image_uncertainty_vs_old=ci, qualifies=qualifies)
        if not all(x['qualifies'] for x in evidence.values()):
            write_json(receipt_path, dict(status='CONTINUE_ORIGINAL_4000_GATE', time=time.time(),
                                         evidence=evidence, policy_sha256=sha(ROOT/'reports/adaptive_budget_policy.json')))
            return
        table = processes()
        assert 'ops/continue_campaign.py' in table[controller][1]
        train = [pid for pid, (ppid, cmd) in table.items()
                 if ppid == controller and 'run_campaign.py --stage M' in cmd]
        assert len(train) == 1
        owned = descendants(table, controller)
        assert os.getpid() not in owned
        try:
            # Stop the controller first: no failure handling or new test can race this audit.
            for pid in [controller, train[0]]+sorted(owned-{controller, train[0]}):
                try:
                    os.kill(pid, signal.SIGSTOP)
                    paused.append(pid)
                except ProcessLookupError:
                    pass
            time.sleep(.3)
            # Include workers created between the first inventory and parent pause.
            owned = descendants(processes(), controller)
            for pid in sorted(owned-set(paused)):
                try:
                    os.kill(pid, signal.SIGSTOP)
                    paused.append(pid)
                except ProcessLookupError:
                    pass
            protocol = read_json(ROOT/'configs/protocol.json')
            lock = read_json(ROOT/'work/geometry_lock.json')
            stream = read_json(ROOT/'work/stream_seed42.json')
            assert lock['source_sha256'] == expected_source
            pair_identity = dict(protocol_sha256=digest(protocol), stream_sha256=stream['rows_sha256'],
                                 geometry_sha256=digest(lock), seed=42)
            checkpoints = {}
            for arm in histories:
                ck = trusted_load(root/arm/'last.pth')
                identity = read_json(root/arm/'identity.json')
                revision = identity.pop('statistics_compatibility_revision', None)
                assert revision == dict(original_producer_source_sha256='56fa846972da1fbf4acb366892eb5b038dd5999d12b1fbb53e00d4d3c77fea9d',
                    consumer_source_sha256=expected_source,
                    change='Only foreground diagnostic reduction; model, losses, geometry and predictions unchanged')
                assert ck['identity'] == identity
                assert identity['protocol_sha256'] == digest(protocol)
                assert identity['stream_sha256'] == stream['rows_sha256']
                assert identity['source_sha256'] == expected_source
                assert ck['update'] == 2000 and ck['stream_cursor'] == 16000
                assert [x['update'] for x in ck['history']] == [0, 1000, 2000]
                assert [x['update'] for x in histories[arm]] == [0, 1000, 2000]
                best_ck = trusted_load(root/arm/'best.pth')
                assert best_ck['selected'] == best_candidate(histories[arm])
                checkpoints[arm] = dict(update=ck['update'], stream_cursor=ck['stream_cursor'],
                    amp_skips=ck['amp_skips'], last_sha256=sha(root/arm/'last.pth'),
                    best_sha256=sha(root/arm/'best.pth'), best_update=best_ck['update'])
                del ck, best_ck
            gate, winner = seed_gate(histories, lock['baseline_val_miou'])
            assert not gate
            decision = dict(status='PAIRED_COST_STOP_AT_2000', time=time.time(), evidence=evidence,
                checkpoints=checkpoints, policy_sha256=sha(ROOT/'reports/adaptive_budget_policy.json'),
                original_planned_updates=8000, actual_matched_updates=2000,
                not_a_convergence_claim=True, old_source_unchanged=True,
                reason='Both complete arms remain over 1 percentage point below old deployment; no recent rise; paired-image 95% upper bounds below old deployment',
                discarded_unscored_partial_segment='The paused process may have begun constructing the next segment. No later complete result is used; preserved last checkpoints are both exactly 2000.')
            write_json(receipt_path, decision)
            write_json(root/'paired_state.json', dict(node=2000, stop=True, reason=decision['reason'],
                best={a:best_candidate(h) for a,h in histories.items()}, adaptive_budget_decision=str(receipt_path)))
            write_json(root/'selection.json', dict(identity=pair_identity, winner=winner,
                second_seed_gate=False, engineering_gain_threshold=.005,
                histories=[str(root/arm/'history.json') for arm in histories], adaptive_stop=decision))
            checkpoint_verified = True
            # All durable evidence is committed. Kill only this audited job tree.
            for pid in sorted(owned-{controller}, reverse=True)+[controller]:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(1)
            with (ROOT/'reports/continuation.log').open('ab') as log:
                proc = subprocess.Popen([sys.executable, 'ops/continue_campaign.py'], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    env=dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'))
            decision['resumed_controller_pid'] = proc.pid
            write_json(receipt_path, decision)
            error_path = ROOT/'reports/adaptive_budget_guard_error.json'
            if error_path.exists():
                old_error = read_json(error_path)
                old_error.update(status='RESOLVED_COMPATIBILITY_METADATA_ONLY',resolved_at=time.time(),
                    explanation='Known zero-update statistics compatibility annotation is preserved; all core checkpoint identity fields compared exactly',
                    checkpoint_unchanged=True)
                write_json(error_path,old_error)
            print('PAIRED_COST_STOP_AT_2000', proc.pid, flush=True)
            return
        except Exception as exc:
            write_json(ROOT/'reports/adaptive_budget_guard_error.json', dict(error=repr(exc),
                time=time.time(), checkpoint_verified=checkpoint_verified))
            if not checkpoint_verified:
                for pid in reversed(paused):
                    try:
                        os.kill(pid, signal.SIGCONT)
                    except ProcessLookupError:
                        pass
            raise
    raise TimeoutError('Matched node not complete within bounded guard window')


if __name__ == '__main__':
    main()
