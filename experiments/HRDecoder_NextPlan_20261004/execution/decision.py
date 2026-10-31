"""All decisions use full validation, registered nodes, and unrounded scores."""
FORMULAS = ('single', 'fixed_fusion', 'half_fusion')
ENDPOINT = {'E_NEG': 24000, 'F4': 12000, 'F12': 12000}
NODES = {'E_NEG': (2000,4000,8000,12000,16000,20000,24000),
         'F4': (2000,4000,8000,12000), 'F12': (2000,4000,8000,12000)}
LATE = {'E_NEG': (20000,24000), 'F4': (8000,12000), 'F12': (8000,12000)}


def promoted(arm, nodes, seed, references):
    reference = references['r_val_by_seed'][str(seed)]
    return [f for f in FORMULAS if all(n in nodes and nodes[n][f]['miou_pct']-reference >= .5
                                      for n in LATE[arm])]


def catastrophic(arm, nodes, seed, references):
    if arm == 'E_NEG':
        return False
    q = references['q_single_val_by_seed'][str(seed)]
    r = references['r_val_by_seed'][str(seed)]
    return all(n in nodes and nodes[n]['single']['miou_pct'] <= q-2 and
               nodes[n]['fixed_fusion']['miou_pct'] <= r-1 for n in (2000,4000))


def select(runs, references):
    candidates = []
    for arm in ('E_NEG','F4','F12'):
        if not all((arm,s) in runs and runs[arm,s].get('completed') == ENDPOINT[arm] and
                   runs[arm,s]['stage'] == 'COMPLETE' for s in (42,2024)):
            continue
        for formula in FORMULAS:
            values = [runs[arm,s]['endpoint_metrics'][formula]['miou_pct'] for s in (42,2024)]
            gains = [v-references['r_val_by_seed'][str(s)] for s,v in zip((42,2024),values)]
            candidates.append({'arm':arm,'formula':formula,'val_by_seed':dict(zip(('42','2024'),values)),
                               'val_mean':sum(values)/2,'gain_by_seed_pp':dict(zip(('42','2024'),gains)),
                               'mean_gain_pp':sum(gains)/2,'test_eligible':min(gains)>0 and sum(gains)/2 >= .5})
    order = {'F4':0,'F12':1,'E_NEG':2}
    candidates.sort(key=lambda c:(-c['val_mean'],FORMULAS.index(c['formula']),order[c['arm']]))
    # Lock the best mean candidate once; no test-guided fallback to another recipe.
    selected = candidates[0] if candidates else None
    return {'candidates':candidates,'selected':selected,
            'test_authorized':bool(selected and selected['test_eligible']),
            'deployment_seed':42,'candidate_limit':9,'test_driven_changes_forbidden':True}
