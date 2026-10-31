"""Server-only import with full original-coordinate count/diagnostic verification."""
import sys
from pathlib import Path
import shutil
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nextlib.common import ROOT,read_json,write_json,sha,digest,source_identity
from nextlib.data import DDR
from nextlib.evaluate import evaluate,legacy_logits,counts
from nextlib.g_stage import load_old
from tools.probability_cache import load_probability

def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    settings=read_json(ROOT/'configs/settings.json')
    pf=read_json(ROOT/'reports/server_preflight.json');assert pf['status']=='PASS'
    ds=DDR(settings['ddr_root'],'val',read_json(settings['val_contract'])['stems'])
    old=Path('/root/autodl-tmp/HRDecoder_Ready_20261002/cache/val')
    root=ROOT/'work/G';root.mkdir(parents=True,exist_ok=True)
    receipt=[]
    for name in ['champion','U0','U1']:
        src=old/name;j=read_json(src/'index.json')
        assert j['n_images']==76 and j['dtype']=='float32'
        assert j['coordinates']=='original_image' and j['channel_order']==['EX','HE','SE','MA']
        assert {row['sample_id'] for row in j['samples']}==set(ds.stems)
        if name=='U0':
            assert j['checkpoint_sha256']==settings['assets']['U0']['source_sha256']
            assert j['resize_to'] is None and not j['flip_tta'] and not j['fast_kernels']
            model=load_old(settings,'U0')
            sid=ds.stems[0];rgb,label=ds.load(sid)
            p=legacy_logits(model,rgb,'native','whole').sigmoid().numpy()[0]
            saved=load_probability(src,sid)
            error=float(np.abs(p-saved).max())
            if error>1e-5 or not np.array_equal(counts(p,label),counts(saved,label)):
                raise ValueError('Existing U0 cache does not reproduce the exact forward')
            del model;torch.cuda.empty_cache()
            dst=root/'U0_native_whole'
            identity=dict(asset_sha256=settings['assets']['U0']['sha256'],geometry='native',
                inference='whole',H_ordinary=False,normalization='local_imagenet',
                source_sha256=source_identity()['sha256'],split='val',data_sha256=pf['data_sha256'])
            for sid in ds.stems:
                dst.mkdir(parents=True,exist_ok=True)
                path=dst/(sid+'.npz')
                if not path.exists():path.symlink_to(src/(sid+'.npz'))
            report=run_diagnostics(ds,src,dst,identity)
        elif name=='champion':
            expected=read_json(settings['champion_settings'])['expected_sha256']
            archived=j['champion_expected_hashes']
            for path,h in expected.items():
                matched=[v for k,v in archived.items() if Path(k).name==Path(path).name]
                assert matched==[h],path
            assert j['includes_ma_head'] and j['flip_tta']
            dst=root/'champion';dst.mkdir(parents=True,exist_ok=True)
            for sid in ds.stems:
                target=dst/(sid+'.npz')
                if not target.exists():target.symlink_to(src/(sid+'.npz'))
            shutil.copy2(src/'index.json',dst/'index.json')
            write_json(dst/'data_identity.json',dict(data_sha256=pf['data_sha256']))
            report=run_diagnostics(ds,src,root/'champion_diagnostics',dict(
                operation='existing_champion_cache_full_verification',source_index_sha256=sha(src/'index.json'),
                data_sha256=pf['data_sha256']))
        else:
            assert j['checkpoint_sha256']==settings['assets']['U1']['source_sha256']
            assert j['resize_to']==[2048,2048] and not j['flip_tta']
            dst=root/'U1_original_reference_diagnostics'
            report=run_diagnostics(ds,src,dst,dict(operation='saved_U1_original_reference',
                source_index_sha256=sha(src/'index.json'),data_sha256=pf['data_sha256']))
            report.update(reference_asset_source_sha256=j['checkpoint_sha256'],
                reference_data_sha256=pf['data_sha256'],reference_fast_kernels=j['fast_kernels'],
                note='Saved original square2048/slide reference retains its actual fast-kernel protocol; no duplicate inference')
            write_json(root/'U1_original_reference.json',report)
        ct=j['metrics'].get('counts') or [[j['metrics']['per_class'][c][k] for k in ['tp','fp','fn']]
                                        for c in ['EX','HE','SE','MA']]
        if report['counts']!=ct:raise ValueError('Original-coordinate historical counts differ')
        receipt.append(dict(name=name,source=str(src),destination=str(dst),
            source_index_sha256=sha(src/'index.json'),probability_files_sha256={sid:sha(src/(sid+'.npz')) for sid in ds.stems},
            full_76_counts_equal=True,mIoU=report['mIoU_lesion4'],U0_first_image_max_error=error if name=='U0' else None))
        write_json(ROOT/'reports/cache_reuse_verification.json',receipt)

def run_diagnostics(ds,src,dst,identity):
    partial=Path(dst)/'partial.json'
    completed=read_json(partial)['images'] if partial.exists() else {}
    iterator=iter(sid for sid in ds.stems if sid not in completed)
    return evaluate(ds,lambda rgb:load_probability(src,next(iterator)),dst,identity,save_probs=False)

if __name__=='__main__':main()
