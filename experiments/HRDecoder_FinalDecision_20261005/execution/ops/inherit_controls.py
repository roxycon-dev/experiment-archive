"""Reuse immutable historical integer evidence; no new test prediction access."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from qlib.common import read_json,write_json,sha,utc_now
from qlib.metrics import summarize

def main():
    old=Path(read_json(ROOT/'configs/runtime.local.json')['old_root']);protocol=read_json(ROOT/'configs/protocol.json');receipt={}
    for seed in (42,2024):
        path=old/f'server_results/reports/final_test/seed{seed}.json';source=read_json(path)
        assert source['status']=='COMPLETE' and source['formula']=='fixed_fusion' and source['update']==16000
        rows=[{'sid':sid,'R':source['images'][sid]['selected']} for sid in sorted(source['images'])]
        assert len(rows)==227
        counts=summarize(np.asarray([r['R'] for r in rows]).sum(0))
        assert abs(counts['miou_pct']-protocol['target']['current_R_test_by_seed'][str(seed)])<1e-12
        write_json(ROOT/f'configs/R_test_counts_seed{seed}.json',{'per_image':rows,'R':counts,'historical_source_sha256':sha(path),'historical_only':True})
        valpath=old/f'server_results/runs/Q_DDR_seed{seed}/val_u16000/metrics.json';val=read_json(valpath)
        assert abs(val['fixed_fusion']['miou_pct']-protocol['evaluation']['R_val_by_seed'][str(seed)])<1e-12
        write_json(ROOT/f'configs/R_val_counts_seed{seed}.json',{'per_image':val['per_image'],'historical_source_sha256':sha(valpath),'R':val['fixed_fusion']})
        receipt[str(seed)]={'test_source_sha256':sha(path),'val_source_sha256':sha(valpath)}
    write_json(ROOT/'reports/inherited_control_identity.json',{'status':'READY','seeds':receipt,'time':utc_now()})

if __name__=='__main__':main()
