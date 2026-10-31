"""Read-only cost-benefit diagnostic from completed validation nodes, no new inference."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mirror_remote import ROOT,connect,remote_exec,write_json
CODE=r'''
from pathlib import Path
import json,sys
r=Path('/root/autodl-tmp/HRDecoder_Next_20261003');sys.path.insert(0,str(r))
from analyze_cache import image_bootstrap
base=json.loads((r/'work/G/current_deployment/metrics.json').read_text())
bc={x['sid']:x['counts'] for x in base['per_image']};result={}
for arm in ['M0','M1']:
 p=r/'work/M_seed42'/arm/'history.json';hist=json.loads(p.read_text())
 prev=next((x for x in hist if x['update']==1000),None)
 now=next((x for x in hist if x['update']==2000),None)
 if not now:result[arm]=dict(status='waiting_for_complete_2000');continue
 mode=max(['solo','fixed_half'],key=lambda k:now[k]['mIoU_lesion4'])
 nc={x['sid']:x['counts'] for x in now[mode]['per_image']};assert set(nc)==set(bc) and len(nc)==76
 ids=sorted(nc);ci=image_bootstrap([nc[x] for x in ids],[bc[x] for x in ids])
 best=max(row[k]['mIoU_lesion4'] for row in hist for k in ['solo','fixed_half'])
 before=max(prev[k]['mIoU_lesion4'] for k in ['solo','fixed_half'])
 result[arm]=dict(status='complete',output=mode,current=now[mode]['mIoU_lesion4'],best=best,
 baseline_gap=base['mIoU_lesion4']-best,recent_delta=now[mode]['mIoU_lesion4']-before,
 image_uncertainty_vs_old=ci,
 strong_underperformance=best<=base['mIoU_lesion4']-.01 and now[mode]['mIoU_lesion4']<=before+.0001 and ci['percentile95'][1]<0,
 per_class=now[mode]['per_class'])
print(json.dumps(dict(arms=result,rule='Read-only diagnostic: best >=1pp below old, no recent full-val rise, 95% paired-image interval below old',
 not_a_convergence_claim=True)))
'''
def main():
 c=connect()
 try:data=json.loads(remote_exec(c,"/root/miniconda3/bin/python - <<'PY'\n"+CODE+"\nPY",30))
 finally:c.close()
 write_json(ROOT/'reports/early_signal_2000.json',data)
 print(json.dumps(data,ensure_ascii=True,indent=2))
if __name__=='__main__':main()
