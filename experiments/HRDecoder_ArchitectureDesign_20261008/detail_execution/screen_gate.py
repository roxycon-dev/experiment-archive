"""Registered4k full-val gate; never test or warm-probe evidence."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'sacd_micro_execution'))
from gate import verify

def screen_gate(work,plan_path,digest):
 work,plan_path=Path(work),Path(plan_path)
 plan=json.loads(plan_path.read_text());p=plan['criteria']
 protocol=json.loads((work/'protocol.json').read_text());smoke=json.loads((work/'GPU_SMOKE_PASS.json').read_text())
 assert protocol['identity_SHA']==smoke['identity_SHA']==digest and smoke['status']=='PASS'
 assert smoke['fresh_initialization_verified'] and protocol['initialization'].startswith('fresh ImageNet')
 gt=json.loads((plan_path.parent.parent/'sacd_execution/val_label_reference.json').read_text())['ground_truth_positive_pixels']
 final=verify(json.loads((work/'val_04000.json').read_text()),gt);base=verify(final['base_head_metrics'],gt)
 assert final['completed']==4000
 calibration={}
 for c,name in ((1,'HE'),(3,'MA')):
  tp,fp,fn=final['counts'][c];btp,bfp,bfn=base['counts'][c]
  calibration[name]={'FN_ratio_final_vs_base':fn/max(bfn,1),'FP_ratio_final_vs_base':fp/max(bfp,1),
    'pass':not(fn>p['HE_MA_reject_if_FN_growth_over']*bfn and fp>p['HE_MA_requires_FP_ratio_when_FN_grows_at_most']*bfp)}
 decisions={'4k_floor':final['mIoU_percent']>=p['4k_mIoU_at_least'],
  'private_net_gain_at_least1pp':final['mIoU_percent']-base['mIoU_percent']>=p['final_minus_base_4k_pp_at_least'],
  'all_classes_nonzero':all(x>0 for x in final['IoU_percent']),
  'no_class_IoU_loss_over2pp':all(a>=b-p['max_class_IoU_loss_final_vs_base_pp'] for a,b in zip(final['IoU_percent'],base['IoU_percent'])),
  'HE_MA_FN_FP_tradeoff':all(x['pass'] for x in calibration.values())}
 return {'status':'PASS' if all(decisions.values()) else 'REJECTED','identity_SHA':digest,
  'decisions':decisions,'time_unix':time.time(),'screen_updates':4000,'full_val_counts_verified':True,
  'mIoU_4k':final['mIoU_percent'],'base_mIoU_4k':base['mIoU_percent'],
  'private_gain_pp':final['mIoU_percent']-base['mIoU_percent'],'calibration':calibration,
  'criterion_registered_before_results':True,'test_accessed':False,'accuracy_guaranteed':False,'goal_achieved':False}
