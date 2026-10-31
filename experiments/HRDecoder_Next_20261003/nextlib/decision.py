"""Predeclared paired stopping and node/output selection."""


def best_candidate(history):
    rows = []
    for node in history:
        for mode in ('solo', 'fixed_half'):
            rows.append(dict(update=node['update'], output=mode,
                             score=node[mode]['mIoU_lesion4'], arm=node.get('arm')))
    # Equal scores -> earlier node -> solo, as specified in the plan.
    return sorted(rows, key=lambda r: (-r['score'], r['update'], r['output'] != 'solo'))[0]


def paired_stop(histories, baseline, trend_epsilon=.0001):
    if any(not h or h[-1]['update'] < 4000 for h in histories.values()):
        return False, 'Need both complete 4000-update validation nodes'
    no_gain, no_trend = [], []
    for arm, history in histories.items():
        start = max(baseline, history[0]['solo']['mIoU_lesion4'],
                    history[0]['fixed_half']['mIoU_lesion4'])
        no_gain.append(best_candidate(history)['score'] <= start+trend_epsilon)
        before, now = history[-2:]
        before_score = max(before[k]['mIoU_lesion4'] for k in ('solo', 'fixed_half'))
        now_score = max(now[k]['mIoU_lesion4'] for k in ('solo', 'fixed_half'))
        no_trend.append(now_score <= before_score+trend_epsilon)
    stopped = all(no_gain) and all(no_trend)
    return stopped, ('Both arms have no deployment gain and no recent full-val rise' if stopped
                     else 'At least one arm has gain or a continuing rise; keep matched budget')


def seed_gate(histories, baseline, minimum_gain=.005):
    winner = max((best_candidate(h) for h in histories.values()), key=lambda r: r['score'])
    return winner['score'] >= baseline+minimum_gain, winner
