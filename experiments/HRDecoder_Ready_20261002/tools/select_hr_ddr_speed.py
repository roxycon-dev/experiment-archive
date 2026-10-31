"""One bounded H DDR throughput choice before its first DDR validation.

Only 4 versus 8; hold effective batch 8, LR, crop and schedule. No validation
or checkpoint. A successful choice is recorded into the actual config before
its formal digest is computed. Do not use this on a started/resumed run.
"""
import json,os,statistics,subprocess,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]

def select(config,out_dir):
    cfg=yaml.safe_load(Path(config).read_text())
    if (Path(out_dir)/'ckpt_last.pth').exists():raise RuntimeError('Speed selection must precede formal training')
    results=[]
    for micro in [4,8]:
        out=ROOT/f'reports/speed/H1_ddr_finetune_m{micro}_w4.json';out.parent.mkdir(parents=True,exist_ok=True)
        if not out.exists():
            args=[sys.executable,'-u','tools/gpu_preflight.py','--config',config,'--stage','ddr_finetune',
                  '--micro-batch',str(micro),'--workers','4','--updates','16','--fast-kernels','--out',str(out)]
            with out.with_suffix('.log').open('w') as f:
                rc=subprocess.run(args,cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT).returncode
            if rc:
                results.append(dict(micro_batch=micro,error='calibration failed; see log'));continue
        data=json.loads(out.read_text());times=data['timings_seconds'][8:]
        results.append(dict(micro_batch=micro,mean_seconds=statistics.mean(times),median_seconds=statistics.median(times),
                            peak_mb=data['peak_allocated_mb']))
    passed=[r for r in results if 'mean_seconds' in r]
    if not passed:raise RuntimeError('Both H DDR speed configurations failed; repair engineering')
    chosen=min(passed,key=lambda r:r['mean_seconds'])
    cfg['ddr_finetune_stage'].update(micro_batch=chosen['micro_batch'],grad_accum=8//chosen['micro_batch'])
    cfg.pop('calibrate_hr_ddr_once',None)
    Path(config).write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    record=dict(cases=results,chosen=chosen,validation_used=False,effective_batch=8,
                inference_geometry_unchanged=True,reason='lowest post-warmup mean including actual DDR data loading')
    (ROOT/'reports/speed/H_DDR_selection.json').write_text(json.dumps(record,indent=2))
    print('[speed selection] '+json.dumps(record),flush=True)
