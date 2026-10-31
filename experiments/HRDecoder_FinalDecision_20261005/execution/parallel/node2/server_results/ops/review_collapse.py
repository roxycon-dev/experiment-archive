"""Only an explicitly reviewed, recorded collapse can end an incomplete run."""
import sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,sha,utc_now
from train_run import review_trigger

def main(run_name,reason):
    assert reason.strip() and len(reason.strip())>=20,'Record concrete scientific evidence, not a generic no-hope slogan'
    path=ROOT/'runs'/run_name;assert path.resolve().is_relative_to((ROOT/'runs').resolve())
    state=read_json(path/'status.json');assert state['stage']=='CURVE_REVIEW_REQUIRED'
    identity=read_json(path/'identity.json');seed=identity['seed']
    nodes={int(p.parent.name[5:]):read_json(p) for p in path.glob('val_u*/metrics.json')}
    parent=read_json(ROOT/f'reports/parent_validation_seed{seed}.json')['single']['miou_pct']
    reference=read_json(ROOT/'configs/protocol.json')['evaluation']['R_val_by_seed'][str(seed)]
    assert review_trigger(nodes,parent,reference)
    receipt={'stage':'CONFIRMED_COLLAPSE','run':run_name,'completed':state['completed'],'reason':reason,
        'formal_node_hashes':{str(u):sha(path/f'val_u{u}/metrics.json') for u in sorted(nodes)[-2:]},
        'identity_sha256':sha(path/'identity.json'),'reviewed_at':utc_now(),'paired32k_conclusion_allowed':False}
    write_json(path/'collapse_review.json',receipt)
    write_json(path/'status.json',dict(state,stage='SCIENTIFIC_STOP_INCOMPLETE',collapse_review=receipt,updated_at=utc_now()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--reason',required=True);a=p.parse_args();main(a.run,a.reason)
