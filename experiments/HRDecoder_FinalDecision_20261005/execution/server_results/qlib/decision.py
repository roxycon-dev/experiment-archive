def serious_failure(nodes, p0=43.48770322103657):
    """Only apply after complete 4k/8k evaluations; no loss-based stopping."""
    if 4000 not in nodes or 8000 not in nodes:
        return False
    a, b = nodes[4000], nodes[8000]
    return (b['single']['miou_pct'] < 35. and b['fixed_fusion']['miou_pct'] < p0-1.
            and b['single']['miou_pct']-a['single']['miou_pct'] < .2)


def extension_required(arms):
    return any(12000 in n and 16000 in n and n[16000]['single']['miou_pct']-n[12000]['single']['miou_pct'] >= .3
               for n in arms.values())


def promotion_signals(arms, common_endpoint=16000, p0=43.48770322103657):
    signals = []
    for arm, nodes in arms.items():
        late = sorted(u for u in nodes if 12000 <= u <= common_endpoint)
        for formula in ('single', 'fixed_fusion'):
            for first, second in zip(late, late[1:]):
                if nodes[first][formula]['miou_pct'] >= p0+.5 and nodes[second][formula]['miou_pct'] >= p0+.5:
                    signals.append({'arm': arm, 'formula': formula, 'nodes': [first, second]})
    return signals


def lock_selection(results, common_endpoint, p0=43.48770322103657):
    # results[arm][seed][update]; do not accept a missing/bad seed selectively.
    candidates = []
    for arm, seeds in results.items():
        if set(seeds) != {42, 2024}:
            raise ValueError('Both prespecified seeds are required')
        common = set(seeds[42]) & set(seeds[2024])
        if common_endpoint not in common:
            raise ValueError('Paired runs have not completed the common endpoint')
        for u in sorted(x for x in common if 12000 <= x <= common_endpoint):
            for formula in ('single', 'fixed_fusion'):
                scores = [seeds[s][u][formula]['miou_pct'] for s in (42, 2024)]
                candidates.append({'arm': arm, 'update': u, 'formula': formula,
                                   'seed_scores_pct': scores, 'mean_val_pct': sum(scores)/2})
    # Mean first; exact tie -> earlier node -> single; then fixed arm ordering.
    candidates.sort(key=lambda x: (-x['mean_val_pct'], x['update'], x['formula'] != 'single', x['arm']))
    best = candidates[0]
    best = {**best, 'test_eligible': best['mean_val_pct'] >= p0+.5 and all(s > p0 for s in best['seed_scores_pct']),
            'deployment_seed': 42, 'candidate_count': len(candidates)}
    return {'selected': best, 'all_candidates': candidates, 'common_endpoint': common_endpoint,
            'test_selection_forbidden': True}
