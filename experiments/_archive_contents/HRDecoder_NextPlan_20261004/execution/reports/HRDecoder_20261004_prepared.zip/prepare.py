"""Local assets only. Never connect, allocate a server, train, or read test labels."""
import copy
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from PIL import Image
from qlib.common import digest, read_json, sha, utc_now, write_json

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parents[1]/'HRDecoder_Q_20261003'
REMOTE_OLD = '/root/autodl-tmp/HRDecoder_Q_20261003'
REMOTE_NEW = '/root/autodl-tmp/HRDecoder_Finetune_20261004'


def repair(path, audit):
    original = read_json(path)
    fixed = copy.deepcopy(original)
    allowed = {(e['sid'],e['class']) for e in audit['affected_entries']}
    changed = []
    for old, new in zip(original['samples'], fixed['samples']):
        for name, entry in new['classes'].items():
            if (new['sid'],name) not in allowed:
                continue
            assert new['source']=='IDRiD' and new['split']=='train' and not new['excluded_duplicate']
            assert entry['review_status']=='unknown' and entry['positive_pixels']==0
            pixels = entry['ignore_pixels']
            entry.update(review_status='confirmed_negative',valid_pixels=pixels,ignore_pixels=0,
                         completeness_basis='User confirms lossless complete official GT conversion, 2026-10-04')
            changed.append({'sid':new['sid'],'class':name,'pixels_restored':pixels,
                            'image_sha256':new['image_sha256'],'mask_sha256':entry['sha256']})
        assert old['image']==new['image'] and old['image_sha256']==new['image_sha256']
        for name in old['classes']:
            assert old['classes'][name]['mask']==new['classes'][name]['mask']
            assert old['classes'][name]['sha256']==new['classes'][name]['sha256']
            if (new['sid'],name) not in allowed:
                assert old['classes'][name]==new['classes'][name]
        if new['source']=='MAPLES':
            assert old==new
    assert len(changed)==29 and sum(x['class']=='SE' for x in changed)==28
    return fixed, changed


def main():
    audit = read_json(ROOT.parent/'IDRiD_negative_supervision_audit.json')
    local_original = OLD/'reports/external_acceptance/manifest.local.accepted.json'
    assert sha(local_original)==audit['source_sha256']
    historical = read_json(OLD/'reports/source_identity.json')
    # Verify only the implementation actually needed for paired reconstruction.
    needed = ['train_run.py','qlib/model.py','qlib/data.py','qlib/geometry.py','qlib/loss.py',
              'qlib/optim.py','qlib/checkpoint.py','qlib/common.py','qlib/metrics.py']
    needed += [rel for rel in historical['files'] if rel.startswith('vendor/dinov3/') and rel.endswith('.py')]
    for rel in needed:
        assert sha(OLD/rel)==historical['files'][rel], 'Historical implementation changed: '+rel
    source_receipt = {rel:historical['files'][rel] for rel in needed}
    write_json(ROOT/'reports/historical_implementation.json',source_receipt)
    for variant in ('local','server'):
        source = OLD/f'reports/external_acceptance/manifest.{variant}.accepted.json'
        fixed, changes = repair(source,audit)
        target = ROOT/f'reports/manifest.{variant}.negative_fixed.json'
        write_json(target,fixed)
        write_json(ROOT/f'reports/negative_diff.{variant}.json',{'source_sha256':sha(source),
                   'output_sha256':sha(target),'changed_entries':changes,'unknown_elsewhere_preserved':True,
                   'images_masks_and_foreground_pools_unchanged':True})
    fixed = read_json(ROOT/'reports/manifest.local.negative_fixed.json')
    mask_checks = []
    for item in changes:
        row = next(r for r in fixed['samples'] if r['source']=='IDRiD' and r['sid']==item['sid'])
        entry = row['classes'][item['class']]
        assert sha(row['image'])==row['image_sha256']
        assert sha(entry['mask'])==entry['sha256']
        with Image.open(entry['mask']) as im:
            assert not np.asarray(im).any(), 'Restored-negative mask is not empty'
        mask_checks.append({'sid':item['sid'],'class':item['class'],'zero_mask_verified':True,
                            'image_sha256':row['image_sha256'],'mask_sha256':entry['sha256']})
    write_json(ROOT/'reports/negative_input_checks.json',mask_checks)
    # Reuse a pre-existing legal DDR-only stream suffix as the fixed NEW-stage stream.
    # This is 12k exposures at old stream positions 12001..24000, without recataloging data.
    # Both F arms replay exactly the same windows/augmentation; some windows were seen before Q16k.
    windows = {}
    references = copy.deepcopy(read_json(ROOT/'configs/protocol.json')['reference'])
    references['q_single_val_by_seed'] = {}
    initial = {}
    for seed in (42,2024):
        src = OLD/f'server_results/reports/stream_Q_MIX_seed{seed}.jsonl'
        identity = read_json(OLD/f'server_results/runs/Q_MIX_seed{seed}/identity.json')
        assert sha(src)==identity['stream_sha256']
        target = ROOT/f'reports/stream_E_NEG_seed{seed}.jsonl'
        shutil.copyfile(src,target)
        counts = {'SE':0,'HE':0}
        affected_rows = ROOT/f'reports/affected_windows_seed{seed}.jsonl'
        allowed = {(e['sid'],e['class']) for e in audit['affected_entries']}
        source_effective = {}
        with affected_rows.open('w',encoding='utf-8',newline='\n') as out:
            for line in src.open(encoding='utf-8'):
                row = json.loads(line)
                if row['source']!='IDRiD':
                    continue
                for c in ('SE','HE'):
                    if (row['sid'],c) in allowed:
                        counts[c]+=1
                        out.write(json.dumps({'class':c,**row},sort_keys=True)+'\n')
        assert counts['SE']=={42:10761,2024:10762}[seed]
        windows[str(seed)] = {'counts':counts,'stream_sha256':sha(target),
                             'affected_window_file_sha256':sha(affected_rows)}
        ddr = OLD/f'server_results/reports/stream_Q_DDR_seed{seed}.jsonl'
        qidentity = read_json(OLD/f'server_results/runs/Q_DDR_seed{seed}/identity.json')
        assert sha(ddr)==qidentity['stream_sha256']
        fpath = ROOT/f'reports/stream_F_seed{seed}.jsonl'
        with fpath.open('w',encoding='utf-8',newline='\n') as out:
            for index,line in enumerate(ddr.open(encoding='utf-8')):
                if index>=12000*8:
                    out.write(line)
        assert sum(1 for _ in fpath.open())==12000*8
        checkpoint = OLD/f'server_results/runs/Q_DDR_seed{seed}/checkpoint_u16000.pth'
        initial[str(seed)] = {'local':str(checkpoint),'server':f'{REMOTE_OLD}/runs/Q_DDR_seed{seed}/checkpoint_u16000.pth',
                              'sha256':sha(checkpoint),'identity':qidentity,'completed':16000}
        metrics = read_json(OLD/f'server_results/runs/Q_DDR_seed{seed}/val_u16000/metrics.json')
        references['q_single_val_by_seed'][str(seed)] = metrics['single']['miou_pct']
        assert abs(metrics['fixed_fusion']['miou_pct']-references['r_val_by_seed'][str(seed)])<1e-10
        for arm in ('Q_MIX','Q_DDR'):
            dest = ROOT/f'historical/{arm}_seed{seed}'
            dest.mkdir(parents=True,exist_ok=True)
            for name in ('identity.json','initialization.json'):
                shutil.copyfile(OLD/f'server_results/runs/{arm}_seed{seed}/{name}',dest/name)
            node = 24000 if arm=='Q_MIX' else 16000
            shutil.copyfile(OLD/f'server_results/runs/{arm}_seed{seed}/val_u{node}/metrics.json',dest/'metrics.json')
        historical_test=read_json(OLD/f'server_results/reports/final_test/seed{seed}.json')
        assert historical_test['formula']=='fixed_fusion' and historical_test['update']==16000
        write_json(ROOT/f'historical/R_test_seed{seed}.json',{'per_image':[
            {'sid':sid,'fixed_fusion':r['selected']} for sid,r in historical_test['images'].items()],
            'source_sha256':sha(OLD/f'server_results/reports/final_test/seed{seed}.json')})
    write_json(ROOT/'reports/affected_window_counts.json',windows)
    write_json(ROOT/'configs/references.json',references)
    write_json(ROOT/'configs/initial_weights.json',initial)
    for name in ('data_identity.json','pretraining_download.json'):
        shutil.copyfile(OLD/'reports'/name,ROOT/'reports'/name)
    shutil.copyfile(OLD/'server_results/reports/pretraining_identity.json',ROOT/'reports/pretraining_identity.json')
    shutil.copyfile(OLD/'configs/p0_identity.json',ROOT/'configs/p0_identity.json')
    for variant in ('local','server'):
        runtime = read_json(OLD/('configs/runtime.local.json' if variant=='local' else 'configs/runtime.qmix.server.json'))
        runtime['external_manifest'] = str(ROOT/'reports/manifest.local.negative_fixed.json') if variant=='local' else REMOTE_NEW+'/reports/manifest.server.negative_fixed.json'
        runtime['dinov3_repo'] = str(OLD/'vendor/dinov3') if variant=='local' else REMOTE_OLD+'/vendor/dinov3'
        runtime.update(old_root=str(OLD) if variant=='local' else REMOTE_OLD,execution_root=str(ROOT) if variant=='local' else REMOTE_NEW,
                       historical_init_weights_file='configs/initial_weights.json',final_test_authorized=False,
                       f_resources=copy.deepcopy(runtime['resources']),stage='LOCAL_PREPARATION_ONLY_GPU_GATE_PENDING',
                       server_python='/root/miniconda3/envs/hrdecoder_q_20261003/bin/python')
        runtime.update(legacy_python='/root/miniconda3/bin/python',
                       test_p0_cache='/dev/shm/hrdecoder_finetune_p0_20261004',
                       old_p0_driver_sha256=sha(OLD/'reports/final_evaluation.py'),
                       old_p0_lock_sha256=sha(OLD/'configs/p0_original_lock.json'))
        runtime['server']['remote_root']=REMOTE_NEW
        runtime['budget']={'hourly_rate_cny':None,'approval':'User requested execution 2026-10-04; server startup by user'}
        write_json(ROOT/f'configs/runtime.{variant}.json',runtime)
    write_json(ROOT/'reports/preparation.json',{'stage':'LOCAL_ASSETS_PREPARED','created_at':utc_now(),
       'changed_entries':29,'affected_windows':windows,'formal_gpu_started':False,'test_read':False,
       'initialization_evidence':'Reconstruct original seed->official pretrained->Q head from verified historical source; no archived u0 tensor digest exists. GPU gate must compare independently reconstructed state fingerprints before claiming strict pairing.',
       'new_F_stream':'Archived DDR stream positions12001..24000 as a fixed new 12k stage, same for F4/F12. Repeated windows permitted; no disjoint/new-image claim.',
       'gpu_gate_required':['E_NEG original initialization reconstruction','F4/F12 each100 full effective updates plus full76val','freeze shared micro scheme before formal F runs']})
    print(json.dumps({'stage':'LOCAL_ASSETS_PREPARED','windows':windows,'formal_gpu_started':False},ensure_ascii=False))


if __name__=='__main__':
    main()
