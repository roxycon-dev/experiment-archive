"""Advance the authorized finite campaign without idle gaps; never shuts down itself."""
from pathlib import Path
import os
import sys
import subprocess
import time
import shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nextlib.common import read_json,write_json,sha,digest,source_identity
from nextlib.data import DDR
from nextlib.evaluate import fusion_cache
from nextlib.decision import best_candidate
ENV=dict(os.environ,PATH='/root/miniconda3/bin:'+os.environ.get('PATH',''),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')

def command(stage,args):
    write_json(ROOT/'reports/server_stage.json',dict(stage=stage,status='running',pid=os.getpid(),time=time.time()))
    with (ROOT/f'reports/{stage}.log').open('a') as log:
        result=subprocess.run([sys.executable]+args,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        write_json(ROOT/'reports/server_stage.json',dict(stage=stage,status='failed',exit_code=result.returncode,time=time.time()))
        raise RuntimeError(stage+' failed; inspection required, no final test or shutdown')

def main():
    write_json(ROOT/'reports/continuation_pid.json',dict(pid=os.getpid(),time=time.time()))
    while not (ROOT/'work/geometry_lock.json').exists():
        stage=read_json(ROOT/'reports/server_stage.json')
        if stage['status']=='failed':raise RuntimeError('G requires engineering repair')
        time.sleep(15)
    while not (ROOT/'reports/G_analysis_complete.json').exists() and read_json(ROOT/'reports/server_stage.json')['status']!='complete':time.sleep(5)
    settings=read_json(ROOT/'configs/settings.json');lock=read_json(ROOT/'work/geometry_lock.json')
    assert lock['source_sha256']==source_identity()['sha256']
    ds=DDR(settings['ddr_root'],'val',read_json(settings['val_contract'])['stems']);g=ROOT/'work/G'
    high=g/f'U0_highres_{lock["canvas"]}_{lock["G_inference"]}'
    candidates=[]
    for key,value in read_json(g/'matrix.json').items():
        if 'fixed_25' not in value:continue
        name,canvas,mode=key.split('_')
        candidates.append(dict(kind='G_fixed25',asset=name,canvas=canvas,inference=mode,
            score=value['fixed_25']['mIoU_lesion4'],cache=str(g/(key+'_fixed25')),key=key))
    if (high/'index.json').exists():
        report=fusion_cache(ds,g/'champion',high,g/'highres_fixed25',.75)
        candidates.append(dict(kind='G_fixed25',asset='U0_highres',canvas=lock['canvas'],
            inference=lock['G_inference'],score=report['mIoU_lesion4'],cache=str(g/'highres_fixed25'),key=high.name))
    best_g=max(candidates,key=lambda x:x['score'])
    write_json(ROOT/'reports/G_deployment_candidates.json',dict(candidates=candidates,best=best_g,
        baseline_val=lock['baseline_val_miou'],criterion='fixed25, finite declared geometry matrix; highres only at locked geometry'))
    if not (ROOT/'reports/G_analysis_complete.json').exists():
        command('G_calibration',['analyze_cache.py','--cache',lock['baseline_cache'],'--baseline',lock['baseline_cache'],
            '--out',str(ROOT/'reports/baseline_calibration_OOF.json')])
        command('G_best_calibration',['analyze_cache.py','--cache',best_g['cache'],'--baseline',lock['baseline_cache'],
            '--out',str(ROOT/'reports/G_best_calibration_OOF.json')])
        command('G_gallery',['ops/error_review.py','--cache',best_g['cache'],'--baseline',lock['baseline_cache'],
            '--out',str(ROOT/'reports/G_error_gallery')])
        write_json(ROOT/'reports/G_analysis_complete.json',dict(status='complete',time=time.time(),best_G=best_g,
            choice='Proceed to matched main representation experiment; calibration is a diagnostic control, not an oracle deployment score',
            E='Official IDRiD raw/pixel GT or MAPLES paired originals not present in audited assets; no external training',
            mechanism='Do not spend GPU on paper-only capacity/lowpass proof while searching for performance improvements'))
    cold=ROOT/'cold_storage/G';cold.mkdir(parents=True,exist_ok=True)
    moved=[]
    protected={'champion','U0_native_whole',best_g['key'],high.name}
    for folder in ([] if (ROOT/'reports/cold_storage_moves.json').exists() else list(g.iterdir())):
        if not folder.is_dir() or folder.is_symlink() or folder.name in protected:continue
        if not (folder/'index.json').exists():continue
        files=list(folder.glob('*.npz'))
        if len(files)!=76:continue
        size=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
        if shutil.disk_usage(cold).free<size+1024**3:continue
        destination=cold/folder.name
        assert not destination.exists()
        shutil.move(str(folder),str(destination));folder.symlink_to(destination,target_is_directory=True)
        moved.append(dict(cache=folder.name,bytes=size,destination=str(destination)))
        if len(moved)>=3:break
    if not (ROOT/'reports/cold_storage_moves.json').exists():
        write_json(ROOT/'reports/cold_storage_moves.json',dict(moved=moved,
            reason='Free RAM-disk space for main validation while verified network backups continue',
            no_original_files_moved=True,no_probability_precision_change=True))
    command('M',['run_campaign.py','--stage','M'])
    results=read_json(ROOT/'work/M_selections.json')
    m_candidates=[]
    for run in (ROOT/'work').glob('M_seed*/M*'):
        if not (run/'history.json').exists():continue
        history=read_json(run/'history.json');best=best_candidate(history)
        seed=int(run.parent.name.replace('M_seed',''))
        m_candidates.append(dict(kind='M',seed=seed,arm=run.name,**{k:v for k,v in best.items() if k!='arm'}))
    winner=max(candidates+m_candidates,key=lambda x:x['score'])
    write_json(ROOT/'reports/val_final_selection.json',dict(winner=winner,candidates=candidates+m_candidates,
        basis='Complete 76-image val only',baseline=lock['baseline_val_miou'],original_winner_retained=True))
    if winner['score']>lock['baseline_val_miou']:
        if winner['kind']=='M':
            command('freeze',['freeze_final.py','--seed',str(winner['seed']),'--arm',winner['arm']])
            command('final',['final_test.py'])
        else:
            final=dict(version='final-G-20261003-v1',kind='G_fixed25',basis='validation_only',
                **{k:v for k,v in winner.items() if k not in ['kind','cache','key']},val_miou=winner['score'],
                runner_sha256=sha(ROOT/'ops/final_g.py'),source_sha256=source_identity()['sha256'],
                champion_settings_sha256=sha(settings['champion_settings']),protocol_sha256=sha(ROOT/'configs/protocol.json'),
                assets={k:dict(path=v['path'],sha256=v['sha256']) for k,v in settings['assets'].items()},
                threshold=.5,strict_greater=True,old_pipeline='0.75*champion+0.25*historical_U0',target=.472115,
                historical_test_note='227-image test inspected previously; exploratory continuation')
            path=ROOT/'work/final_lock.json'
            if path.exists():assert read_json(path)==final
            else:assert not (ROOT/'work/test_progress.json').exists();write_json(path,final)
            command('final',['ops/final_g.py'])
    required={}
    for p in [ROOT/'work/geometry_lock.json',ROOT/'work/campaign_complete.json',ROOT/'configs/protocol.json',
              ROOT/'configs/settings.json',ROOT/'reports/val_final_selection.json']:
        required[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in (ROOT/'work').glob('M_seed*/M*/best.pth'):required[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in (ROOT/'work').glob('M_seed*/M*/last.pth'):required[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in [ROOT/'work/final_lock.json',ROOT/'work/final_test.json',ROOT/'work/test_progress.json']:
        if p.exists():required[p.relative_to(ROOT).as_posix()]=sha(p)
    write_json(ROOT/'reports/campaign_completion.json',dict(status='EXPERIMENTS_COMPLETE_BACKUP_REVIEW_PENDING',
        time=time.time(),winner=winner,required_files=required,original_winner_retained=True,
        E='official assets gate not satisfied; no GPU time allocated',
        mechanism='Not run: user requests discovering improvements, not repeated proof of efficacy',
        target=.472115,test_run=(ROOT/'work/final_test.json').exists(),
        next='Verify local backup, finish report and manual error review, then authorized shutdown'))
    write_json(ROOT/'reports/server_stage.json',dict(stage='experiments',status='complete',time=time.time()))

if __name__=='__main__':main()
