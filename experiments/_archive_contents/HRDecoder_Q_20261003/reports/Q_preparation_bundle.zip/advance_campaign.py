"""Persist only preregistered decisions; this utility never starts training by itself."""
from pathlib import Path

from qlib.common import read_json, utc_now, write_json
from qlib.decision import serious_failure, extension_required, promotion_signals, lock_selection


ROOT = Path(__file__).resolve().parent


def main():
    runtime = read_json(ROOT/'configs/runtime.server.json')
    arms = ['Q_DDR']
    if runtime['external_manifest'] and read_json(runtime['external_manifest'])['status'] == 'READY':
        arms.append('Q_MIX')
    results = {}
    statuses = {}
    for arm in arms:
        results[arm] = {}
        for seed in (42, 2024):
            directory = ROOT/'runs'/f'{arm}_seed{seed}'
            nodes = {int(p.parent.name.removeprefix('val_u')): read_json(p) for p in directory.glob('val_u*/metrics.json')}
            if nodes:
                results[arm][seed] = nodes
            if (directory/'status.json').exists():
                statuses[f'{arm}_seed{seed}'] = read_json(directory/'status.json')
    discovery = {arm: seeds.get(42, {}) for arm, seeds in results.items()}
    stop = any(serious_failure(n) for n in discovery.values())
    endpoint = 24000 if extension_required(discovery) else 16000
    # Monotonic common endpoint: never undo an already triggered extension.
    previous = read_json(ROOT/'reports/campaign_state.json') if (ROOT/'reports/campaign_state.json').exists() else {}
    endpoint = max(endpoint, previous.get('common_endpoint', 16000))
    discovery_complete = all(endpoint in discovery[a] for a in arms)
    signals = promotion_signals(discovery, endpoint)
    review = discovery_complete and bool(signals) and not stop
    state = {'arms': arms, 'common_endpoint': endpoint, 'architecture_stopped': stop,
             'discovery_complete': discovery_complete, 'promotion_signals': signals,
             'review_seed_authorized': review, 'automatic_extra_experiments': False,
             'statuses': statuses, 'updated_at': utc_now()}
    if stop:
        state['next_action'] = 'STOP_ARCHITECTURE_BACKUP_REPORT_SHUTDOWN'
    elif not discovery_complete:
        state['next_action'] = 'COMPLETE_DISCOVERY_AT_COMMON_ENDPOINT'
    elif not review:
        state['next_action'] = 'NO_SUSTAINED_SIGNAL_STOP_BACKUP_REPORT_SHUTDOWN'
    elif not all(set(results[a]) == {42, 2024} and endpoint in results[a][2024] for a in arms):
        state['next_action'] = 'REVIEW_ALL_AVAILABLE_ARMS_WITH_SEED_2024'
    else:
        selection = lock_selection(results, endpoint)
        state['selection'] = selection
        state['next_action'] = 'FREEZE_SELECTION_THEN_FINAL_TEST' if selection['selected']['test_eligible'] else 'INSUFFICIENT_REPLICATION_STOP_BACKUP_REPORT_SHUTDOWN'
    write_json(ROOT/'reports/campaign_state.json', state)
    print(state['next_action'])


if __name__ == '__main__':
    main()
