"""Validator-only contract repair; does not import torch or modify training state."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'sacd_micro_execution'))
from gate import verify

def main():
    remote_work=ROOT/'training_runs/residual'
    local_work=ROOT/'cache_execution/server_receipts/residual/training_runs/residual'
    work=remote_work if (remote_work/'val_04000.json').exists() else local_work
    value=json.loads((work/'val_04000.json').read_text())
    gt=json.loads((ROOT/'sacd_execution/val_label_reference.json').read_text())['ground_truth_positive_pixels']
    assert verify(value,gt) is value
    assert verify(value['base_head_metrics'],gt) is value['base_head_metrics']
    corrupt=[]
    for mode in ('IoU','count','missing_deployment_flag','missing_base_head','negative_count','float_count'):
        row=copy.deepcopy(value)
        if mode=='IoU':row['base_head_metrics']['IoU_percent'][0]+=.01
        if mode=='count':row['counts'][0][2]+=1
        if mode=='missing_deployment_flag':del row['teacher_or_GT_input']
        if mode=='missing_base_head':del row['base_head_metrics']
        if mode=='negative_count':row['counts'][0][0]=-1
        if mode=='float_count':row['counts'][0][0]=float(row['counts'][0][0])
        try:verify(row,gt)
        except (AssertionError,KeyError,ValueError,TypeError):corrupt.append(mode)
        else:raise AssertionError('Corruption accepted: '+mode)
    digest=hashlib.sha256((ROOT/'sacd_micro_execution/gate.py').read_bytes()).hexdigest()
    receipt={'status':'PASS','validation_helper_SHA256':digest,'full_record_and_standalone_metrics_validated':True,
      'negative_controls_rejected':corrupt,'GPU_used':False,'optimizer_updates':0,
      'thresholds_changed':False,'counts_or_raw_metrics_changed':False,'training_math_changed':False}
    (Path(__file__).resolve().parent/'METRIC_ADAPTER_CPU_PASS.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
