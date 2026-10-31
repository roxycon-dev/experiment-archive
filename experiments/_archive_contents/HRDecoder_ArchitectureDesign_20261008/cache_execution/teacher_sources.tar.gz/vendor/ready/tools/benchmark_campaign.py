"""Small throughput sweep. No validation, checkpoint, or model selection."""
import json
import argparse
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CASES=[('U1',2,'external_pretrain',40),('U1',4,'external_pretrain',40),
       ('U0',4,'ddr_finetune',24),('H1',2,'external_pretrain',16),
       ('H1',4,'external_pretrain',16),('A0',2,'ddr_finetune',16),
       ('A0',4,'ddr_finetune',16)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--extra',action='store_true');a=ap.parse_args()
    paths=json.loads((ROOT/'configs/resolved/resolved_paths.json').read_text())
    summary=ROOT/'reports/speed/summary.json'
    results=json.loads(summary.read_text()) if a.extra else []
    cases=[('H1',8,'external_pretrain',16),('A0',8,'ddr_finetune',16),
           ('U1',2,'external_pretrain',64)] if a.extra else CASES
    for arm,micro,stage,updates in cases:
        workers=8 if a.extra and arm=='U1' else 4
        tag=f'{arm}_{stage}_m{micro}_w{workers}'
        output=ROOT/f'reports/speed/{tag}.json';output.parent.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-u','tools/gpu_preflight.py','--config',paths[arm],
             '--stage',stage,'--out',str(output),'--micro-batch',str(micro),
             '--updates',str(updates),'--fast-kernels','--workers',str(workers)]
        with output.with_suffix('.log').open('w') as f:
            rc=subprocess.run(cmd,cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT).returncode
        item=dict(arm=arm,stage=stage,micro_batch=micro,workers=workers,exit_code=rc)
        if rc==0:
            r=json.loads(output.read_text());times=r['timings_seconds'][max(4,updates//2):]
            item.update(mean_seconds=statistics.mean(times),median_seconds=statistics.median(times),
                        peak_mb=r['peak_allocated_mb'],all_seconds=r['timings_seconds'],
                        total_data_wait_seconds=sum(r['data_wait_seconds']))
        else:
            item['error_tail']=output.with_suffix('.log').read_text()[-4000:]
        results.append(item)
        (ROOT/'reports/speed/summary.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(item),flush=True)

if __name__=='__main__':main()
