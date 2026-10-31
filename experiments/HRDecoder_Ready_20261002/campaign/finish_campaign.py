"""Finish the exploratory campaign with declared, bounded adaptive rules.

Wait for the serial main queue; cache val predictions; nominate two families;
cross-fit; extend a useful direction; lock exactly one final test pipeline.
No repeated historical training, no significance gate, no test-based selection.
"""
import copy
import atexit
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.configure_server import sha
from tools.probability_cache import load_probability,save_probability
from tools.crossfit_fusion import iou

STATE=ROOT/'work_dirs/finish_state.json'

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    p=path.with_suffix('.tmp');p.write_text(json.dumps(obj,indent=2),encoding='utf-8');p.replace(path)

def command(args,tag):
    log=ROOT/f'reports/finish/{tag}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        rc=subprocess.run([sys.executable,'-u']+list(map(str,args)),cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT).returncode
    if rc:raise RuntimeError(f'{tag} failed ({rc}); inspect {log}; engineering failure, not method rejection')

def counts(p,gt):
    valid=gt!=255
    return np.array([[np.count_nonzero((p[c]>.5)&(gt==c+1)&valid),
        np.count_nonzero((p[c]>.5)&(gt!=c+1)&valid),
        np.count_nonzero((p[c]<=.5)&(gt==c+1)&valid)] for c in range(4)],dtype=np.int64)

def export(spec):
    cache=ROOT/f'cache/val/{spec["id"]}';idx=cache/'index.json'
    config=spec['members'][0]['config'];ck=spec['members'][0]['checkpoint']
    if spec.get('kind','model')=='model':
        if idx.exists():
            d=json.loads(idx.read_text())
            if d['checkpoint_sha256']==sha(ck) and d['config_sha256']==sha(config):
                spec['cache']=str(cache);return spec
        command(['tools/export_predictions.py','--config',config,'--checkpoint',ck,'--out',cache],f'export_{spec["id"]}')
    else:
        mean_identity=hashlib.sha256(json.dumps([dict(checkpoint_sha256=sha(m['checkpoint']),
            config_sha256=sha(m['config']),weight=m['weight']) for m in spec['members']],sort_keys=True).encode()).hexdigest()
        if not idx.exists() or json.loads(idx.read_text())['checkpoint_sha256']!=mean_identity:
            a,b=spec['components'];ia=json.loads((Path(a['cache'])/'index.json').read_text())
            cache.mkdir(parents=True,exist_ok=True)
            ia.update(checkpoint_sha256=mean_identity,
                      components=[a['id'],b['id']],config_sha256='fixed equal probability average',
                      storage='lazy_fixed_half_lossless_float32_mean',components_cache=[a['cache'],b['cache']])
            write(idx,ia)
    spec['cache']=str(cache);return spec

def assess(specs,champ,labels,ids):
    base=np.zeros((len(ids),4,3),np.int64)
    evaluations={s['id']:dict(spec=s,alone=np.zeros_like(base),fixed25=np.zeros_like(base)) for s in specs}
    for i,sid in enumerate(ids):
        gt=np.asarray(Image.open(labels/f'{sid}.png'));p=load_probability(champ,sid);base[i]=counts(p,gt)
        for s in specs:
            q=load_probability(s['cache'],sid)
            if q.shape!=p.shape:raise ValueError('Cache coordinate mismatch')
            evaluations[s['id']]['alone'][i]=counts(q,gt)
            evaluations[s['id']]['fixed25'][i]=counts(.75*p+.25*q,gt)
    biou=iou(base.sum(axis=0));bm=float(np.mean(biou));rows=[]
    for item in evaluations.values():
        f=iou(item['fixed25'].sum(axis=0));a=iou(item['alone'].sum(axis=0))
        # The same fixed 25% blend ranks each variant; a useful class may
        # qualify a weaker standalone network without unrestricted weights.
        score=max(float(np.mean(f))-bm,float(np.max(f-biou))/4)
        rows.append(dict(spec=item['spec'],score=score,alone_miou=float(np.mean(a)),
                         alone_iou=a.tolist(),fixed25_miou=float(np.mean(f)),fixed25_iou=f.tolist()))
    rows.sort(key=lambda r:(r['score'],r['alone_miou']),reverse=True)
    winners={}
    for row in rows:winners.setdefault(row['spec']['family'],row)
    nominated=list(winners.values())[:2]
    return dict(base_miou=bm,base_iou=biou.tolist(),rows=rows,nominated=nominated),base

def fusion(assessment,champ,labels,tag):
    output=ROOT/f'reports/finish/{tag}.json'
    args=['tools/crossfit_fusion.py','--folds',ROOT/'reports/fusion_folds.json',
          '--champion',champ,'--labels',labels,'--out',output]
    for row in assessment['nominated']:
        s=row['spec'];args+=['--candidate',s['family']+'='+s['cache']]
    command(args,tag)
    result=json.loads(output.read_text())
    # Compute the already-locked vote's descriptive val result, without
    # scanning or selecting weights again on all val images.
    with np.load(output.with_suffix('.counts.npz'),allow_pickle=False) as z:
        tensor=z['counts'];options=json.loads(str(z['options']));ids=z['ids'].tolist()
        selected=[]
        for c,opt in enumerate(result['deploy']):
            j=next(i for i,o in enumerate(options) if o==opt)
            selected.append(tensor[j,:,c])
        deployed=np.stack(selected,axis=1);base=tensor[0]
    result['locked_val_miou']=float(np.mean(iou(deployed.sum(axis=0))))
    result['selected_oof_miou']=result['oof_class_miou'] if result['selected_mode']=='per_class' else result['oof_global_miou']
    rng=np.random.RandomState(42);delta=[]
    for _ in range(1500):
        draw=rng.randint(0,len(ids),len(ids))
        delta.append(float(np.mean(iou(deployed[draw].sum(axis=0)))-np.mean(iou(base[draw].sum(axis=0)))))
    result['descriptive_paired_bootstrap_delta_ci95']=np.quantile(delta,[.025,.975]).tolist()
    result['bootstrap_is_not_outer_validation']=True
    write(output,result);return result

def train_variant(source,kind):
    src=source['members'][0];cfg=yaml.safe_load(Path(src['config']).read_text())
    tag=source['id']+'_'+kind;path=ROOT/f'configs/followups/{tag}.yaml';path.parent.mkdir(parents=True,exist_ok=True)
    if kind=='seed142':
        original_paths=json.loads((ROOT/'configs/resolved/resolved_paths.json').read_text())
        cfg=yaml.safe_load(Path(original_paths[cfg['arm_id']]).read_text())
        cfg['initialization'].pop('ddr_warm_start_checkpoint',None)
        cfg['ddr_finetune_stage']['seed']=142
        cfg['ddr_finetune_stage']['updates']=12000
        cfg['ddr_finetune_stage']['eval_progress_points']=[.25,.5,.75,1]
        cfg['ddr_finetune_stage']['augmentation'].pop('resize_long_sides',None)
    else:
        cfg['initialization']['ddr_warm_start_checkpoint']=dict(file=src['checkpoint'],sha256=sha(src['checkpoint']))
        st=cfg['ddr_finetune_stage'];st['updates']=4000
        st['eval_progress_points']=[.25,.5,1]
        st['backbone_lr']*=.2;st['decoder_lr']*=.2
        st['augmentation']['resize_long_sides']=[2048,2560] if kind=='highres' else [2048]
        st['screening']=dict(enabled=True,mode='continuation_plateau',min_progress=.5,reference_miou=source['alone_miou'])
        if kind=='semi':
            st['pseudo_manifest']=source['pseudo_manifest'];st['pseudo_manifest_sha256']=sha(source['pseudo_manifest'])
            st['pseudo_micro_batch']=min(2,st['micro_batch'])
    # Retain the serialized config exactly on resume; never silently weaken
    # config or manifest hash checks.
    if not path.exists():path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    out=ROOT/f'work_dirs/{tag}';summary=out/'completion_summary.json'
    if not summary.exists():
        args=['train.py','--config',path,'--stage','ddr_finetune','--mode','formal','--out-dir',out]
        if (out/'ckpt_last.pth').exists():args+=['--resume',out/'ckpt_last.pth']
        command(args,tag)
    done=json.loads(summary.read_text())
    return dict(id=tag,family=source['family'],kind='model',members=[dict(config=str(path),checkpoint=str(out/'ckpt_best.pth'),weight=1.)],
                updates_done=done['updates_done'])

def recover_duplicate_storage():
    # Keep best and last (including full optimizer/RNG). A completed final
    # snapshot is redundant only after exact equality of its model tensors
    # and update with last has been checked. Active stages are never touched.
    import torch
    from trainer.checkpoint import load_checkpoint
    records=[]
    for directory in (ROOT/'work_dirs').iterdir():
        if not directory.is_dir() or ROOT.resolve() not in directory.resolve().parents:continue
        summary=directory/'completion_summary.json';final=directory/'ckpt_final.pth';last=directory/'ckpt_last.pth'
        if not (summary.exists() and final.exists() and last.exists()):continue
        if json.loads(summary.read_text()).get('status') not in ['complete','stopped']:continue
        a=load_checkpoint(final);b=load_checkpoint(last)
        identical=(a['update']==b['update'] and a['config_sha256']==b['config_sha256'] and
            a['model_state'].keys()==b['model_state'].keys() and
            all(torch.equal(a['model_state'][k],b['model_state'][k]) for k in a['model_state']))
        del a,b
        if identical:
            records.append(dict(removed=str(final),retained=str(last),bytes=final.stat().st_size,
                                reason='same update and exactly equal model tensors; best and full last retained'))
            final.unlink()
    path=ROOT/'reports/finish/space_recovery.json'
    prior=json.loads(path.read_text()) if path.exists() else []
    write(path,prior+records)

def storage_need(source,copies=1,extra=0):
    cache=sum(p.stat().st_size for p in Path(source['cache']).glob('*.npz'))
    checkpoint=Path(source['members'][0]['checkpoint']).stat().st_size
    return int(copies*(3*checkpoint+cache)*1.1+1024**3+extra)

def useful(f):
    return f['selected_oof_miou']>f['oof_base_miou']+.001 and f['locked_val_miou']>f['oof_base_miou']+.001

def main():
    lock=ROOT/'work_dirs/finish.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    if lock.exists():
        try:os.kill(int(lock.read_text()),0)
        except (ProcessLookupError,ValueError):lock.unlink()
        else:raise RuntimeError('Finish driver already running')
    lock.write_text(str(os.getpid()))
    atexit.register(lambda lock_path=lock:lock_path.unlink() if lock_path.exists() else None)
    status=dict(status='waiting_for_main_training',rules=dict(
        highres='OOF and locked-val gain >0.001; best fixed25 nominee only; paired 4000-update continuation',
        second_seed='OOF gain >=0.005 and standalone >= champion-0.015; ensemble exploration, not confirmation',
        final_test='one locked pipeline only if OOF and locked-val gain >0.001; otherwise retain champion'))
    write(STATE,status)
    while True:
        p=ROOT/'work_dirs/campaign_state.json'
        if p.exists():
            queue=json.loads(p.read_text())
            if queue.get('status')=='main_training_done':break
            pid=queue.get('pid')
            if pid:
                proc=Path('/proc')/str(pid)/'cmdline'
                if not proc.exists() or b'run_campaign.py' not in proc.read_bytes():
                    raise RuntimeError('Main queue exited before completion; inspect logs and resume saved checkpoints')
        time.sleep(30)
    paths=json.loads((ROOT/'configs/resolved/resolved_paths.json').read_text())
    recover_duplicate_storage()
    cfg=yaml.safe_load(Path(paths['U0']).read_text());labels=Path(cfg['data']['ddr_root'])/'labels/val'
    ids=sorted(json.loads((ROOT/'reports/fusion_folds.json').read_text())['folds'])
    champ=ROOT/'cache/val/champion';status['status']='exporting';write(STATE,status)
    if not (champ/'index.json').exists():
        command(['tools/export_champion.py','--settings',ROOT/'configs/champion_server.json','--out',champ],'export_champion')
    actual=json.loads((champ/'index.json').read_text())['metrics']['mIoU_lesion4']
    if abs(actual-.426694)>.0002:raise RuntimeError(f'Champion reproduction mismatch: {actual}; repair before fusion')
    specs=[]
    for arm in ['U0','U1','H1','A0']:
        ck=ROOT/'assets/U0_historical_best.pth' if arm=='U0' else ROOT/f'work_dirs/{arm}_ddr_finetune/ckpt_best.pth'
        if ck.exists():
            if arm=='U0' and sha(ck)!='98d98a5d02dc2615062c3c31cd15d061daf3d7ab41cb8f25cd90b062cd53bb34':
                raise ValueError('Historical U0 file hash mismatch')
            c=yaml.safe_load(Path(paths[arm]).read_text())
            s=dict(id=arm,family=c['structure']['family'],kind='model',members=[dict(config=paths[arm],checkpoint=str(ck),weight=1.)])
            specs.append(export(s))
    if not specs:raise RuntimeError('No candidate checkpoints; inspect engineering failures')
    a,base=assess(specs,champ,labels,ids);write(ROOT/'reports/finish/initial_nomination.json',a)
    f=fusion(a,champ,labels,'initial_fusion');status.update(status='initial_fusion_done',fusion=f);write(STATE,status)
    if useful(f) and shutil.disk_usage(ROOT).free>storage_need(a['nominated'][0]['spec'],2):
        winner=copy.deepcopy(a['nominated'][0]['spec']);winner['alone_miou']=a['nominated'][0]['alone_miou']
        for kind in ['continuation2048','highres']:
            specs.append(export(train_variant(winner,kind)));recover_duplicate_storage()
        a,base=assess(specs,champ,labels,ids);write(ROOT/'reports/finish/highres_nomination.json',a)
        f=fusion(a,champ,labels,'highres_fusion')
        status.update(status='highres_done',fusion=f);write(STATE,status)
    else:status['highres_skipped']='no useful cross-fit signal or insufficient cache/checkpoint space'
    if (useful(f) and f['selected_oof_miou']-f['oof_base_miou']>=.005 and
        a['nominated'][0]['alone_miou']>=a['base_miou']-.015 and
        shutil.disk_usage(ROOT).free>storage_need(a['nominated'][0]['spec'])):
        winner=a['nominated'][0]['spec'];seed=export(train_variant(winner,'seed142'))
        recover_duplicate_storage()
        specs.append(seed)
        mean=dict(id=winner['id']+'_seedmean',family=winner['family'],kind='mean',components=[winner,seed],
                  members=[dict(m,weight=m['weight']*.5) for s in [winner,seed] for m in s['members']])
        specs.append(export(mean));a,base=assess(specs,champ,labels,ids)
        write(ROOT/'reports/finish/seed_nomination.json',a);f=fusion(a,champ,labels,'seed_fusion')
    else:status['second_seed_skipped']='no large enough promising ensemble signal; avoid confirmation-only rerun'
    auxiliary=ROOT/'assets/auxiliary/train_manifest.jsonl'
    teacher=max((r for r in a['rows'] if r['spec'].get('kind','model')=='model'),key=lambda r:r['alone_miou'])
    control_exists=(ROOT/f'work_dirs/{teacher["spec"]["id"]}_continuation2048/completion_summary.json').exists()
    if (auxiliary.exists() and teacher['alone_miou']>=a['base_miou']-.02 and
        shutil.disk_usage(ROOT).free>storage_need(teacher['spec'],1 if control_exists else 2,1024**3)):
        teacher=copy.deepcopy(teacher);teacher['base_iou']=a['base_iou']
        teacherfile=ROOT/'reports/finish/semi_teacher.json';write(teacherfile,teacher)
        pseudo=ROOT/f'cache/pseudo/{teacher["spec"]["id"]}'
        command(['tools/build_pseudo_labels.py','--teacher',teacherfile,'--auxiliary',auxiliary,'--out',pseudo],'pseudo_cache')
        quality=json.loads((pseudo/'quality.json').read_text());status['semisupervision']=quality
        if quality['qualified']:
            source=copy.deepcopy(teacher['spec']);source['alone_miou']=teacher['alone_miou']
            source['pseudo_manifest']=str(pseudo/'manifest.jsonl')
            for kind in ['continuation2048','semi']:
                result=export(train_variant(source,kind))
                recover_duplicate_storage()
                if result['id'] not in {s['id'] for s in specs}:specs.append(result)
            a,base=assess(specs,champ,labels,ids);write(ROOT/'reports/finish/semi_nomination.json',a)
            f=fusion(a,champ,labels,'semi_fusion')
    else:status['semisupervision']='skipped: teacher not within 0.02 of champion, pool absent, or insufficient space'
    if useful(f):
        by_family={row['spec']['family']:row['spec'] for row in a['nominated']}
        lock=dict(champion_settings=str(ROOT/'configs/champion_server.json'),ddr_root=cfg['data']['ddr_root'],
                  candidates=by_family,deploy=f['deploy'],threshold=.5,strict_greater=True,
                  source_fusion=f,locked_at_epoch=time.time())
        for s in lock['candidates'].values():
            for m in s['members']:m.update(checkpoint_sha256=sha(m['checkpoint']),config_sha256=sha(m['config']))
        lockpath=ROOT/'reports/finish/deployment_lock.json'
        if not lockpath.exists():write(lockpath,lock)
        status['status']='locked_final_test';write(STATE,status)
        command(['tools/final_eval_locked.py','--lock',lockpath],'final_test')
        status['final_test']=json.loads((ROOT/'reports/finish/final_test.json').read_text())
    else:status['final_test']='skipped: no useful locked val/OOF signal; retain historical champion 44.2115%'
    status.update(status='complete',completed_at_epoch=time.time(),final_fusion=f)
    write(STATE,status);write(ROOT/'reports/finish/campaign_result.json',status)
    print(json.dumps(status,indent=2))

if __name__=='__main__':
    try:main()
    except Exception as e:
        current=json.loads(STATE.read_text()) if STATE.exists() else {}
        current.update(status='engineering_attention_required',error=str(e));write(STATE,current);raise
