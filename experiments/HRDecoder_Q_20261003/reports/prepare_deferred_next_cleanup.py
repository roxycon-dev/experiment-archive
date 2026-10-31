"""Prepare an exact SHA allowlist for retired U1 probability caches only."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qlib.common import read_json,sha,utc_now,write_json

NEXT = ROOT.parent / 'HRDecoder_Next_20261003'
MAINT = ROOT.parent / '_maintenance'
PLAN = MAINT / 'next_retired_U1_cleanup_plan_20261004.json'


def main():
    assert not PLAN.exists(), 'Review an existing deletion allowlist instead of replacing it'
    policy=read_json(MAINT/'deferred_cleanup_policy_20261004.json')
    report=read_json(ROOT/'reports/original_plan_report_receipt.json')
    final=read_json(ROOT/'reports/continuation_final_evidence_backup_receipt.json')
    assert report['stage']=='COMPLETE_LOCAL_SHA_VERIFIED' and final['stage']=='COMPLETE_SHA_VERIFIED'
    assert report['lock_sha256']==final['lock_sha256']
    for row in report['files']:
        assert sha(ROOT/row['path'])==row['sha256']
    assert read_json(ROOT/'reports/immutable_campaign_asset_backup_receipt.json')['stage']=='COMPLETE_SHA_VERIFIED'
    for arm in ('Q_DDR','Q_MIX'):
        for seed in (42,2024):
            backup=read_json(ROOT/('reports/run_backup_%s_seed%s_24k.json'%(arm,seed)))
            assert backup['complete'] and len(backup['files'])==327
    selected=read_json(ROOT/'server_results/continuation_evidence/reports/continuation_final_selection.json')
    assert selected['selection']['selected']['arm']=='Q_DDR'
    assert selected['selection']['selected']['formula']=='fixed_fusion'
    winner=read_json(NEXT/'server_results/reports/val_final_selection.json')['winner']
    assert winner['asset']!='U1'
    assets=read_json(NEXT/'configs/assets.json')
    models=list((NEXT/'assets').rglob('U1.pth'))
    assert len(models)==1 and sha(models[0])==assets['U1']['sha256']
    model_snapshot=[dict(path=str(p.resolve()),bytes=p.stat().st_size)
                    for p in sorted(NEXT.rglob('*')) if p.is_file() and p.suffix.lower() in ('.pth','.pt','.ckpt')]
    assert model_snapshot
    groups=[('U1_native_whole',NEXT/'server_results/cold_storage/G/U1_native_whole'),
            ('U1_long2048_slide',NEXT/'server_results/work/G/U1_long2048_slide'),
            ('U1_long2048_whole',NEXT/'server_results/work/G/U1_long2048_whole')]
    mirror=read_json(NEXT/'reports/mirror_manifest.json')
    cold=read_json(ROOT/'reports/closed_u1_cache_offload_receipt.json')
    assert cold['stage']=='COMPLETE_BACKUP_SHA_VERIFIED_REMOTE_NPZ_ONLY_OFFLOADED'
    assert cold['models_and_original_metadata_retained'] and not cold['local_deletion_performed']
    cold_files={row['name']:row for row in cold['files']}
    files=[];metadata=[];excluded=[]
    for tag,directory in groups:
        base=NEXT/'server_results/work/G'/tag
        metric=read_json(base/'metrics.json');index=read_json(base/'index.json')
        assert metric['n_images']==index['n_images']==76 and len(metric['per_image'])==76
        assert metric['identity']['asset_sha256']==assets['U1']['sha256']
        image_ids={row['sid'] for row in metric['per_image']}
        assert len(image_ids)==76 and image_ids==set(index['samples'])
        aggregate=[[sum(row['counts'][c][k] for row in metric['per_image']) for k in range(3)] for c in range(4)]
        assert aggregate==metric['counts'], 'Keep exact per-image count evidence before retiring probabilities'
        for p in base.glob('*.json'):
            metadata.append(dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)))
        for p in sorted(directory.glob('*.npz')):
            resolved=p.resolve()
            assert resolved.is_relative_to(NEXT.resolve()) and resolved.parent==directory.resolve() and not p.is_symlink()
            assert p.stem in image_ids
            expected=cold_files.get(p.name) if tag=='U1_native_whole' else mirror.get('work/G/'+tag+'/'+p.name)
            checksum=sha(p)
            if not expected or checksum!=expected['sha256'] or p.stat().st_size!=expected.get('bytes',expected.get('size')):
                excluded.append(dict(path=str(resolved),reason='Prior verified checksum unavailable or differs; preserve'))
                continue
            files.append(dict(path=str(resolved),bytes=p.stat().st_size,sha256=checksum,
                reason='Closed U1 nonwinner validation probability; not used by P0/Q or any remaining original-plan task. Full per-image counts, model, geometry, source and metadata remain for regeneration.'))
    assert len(files)==len({x['path'] for x in files})
    protected_caches=[]
    for tag in ('champion','U0_native_whole','U0_highres_native_slide'):
        p=NEXT/'server_results/work/G'/tag
        protected_caches.append(dict(path=str(p.resolve()),files=[dict(name=x.name,bytes=x.stat().st_size) for x in sorted(p.glob('*.npz'))]))
    assert all(len(x['files'])==76 for x in protected_caches)
    plan=dict(stage='REVIEWED_EXACT_ALLOWLIST_READY',created_at=utc_now(),root=str(NEXT.resolve()),
        original_plan_experiments_and_evaluation_complete=True,necessary_local_results_SHA_verified=True,
        policy_sha256=sha(MAINT/'deferred_cleanup_policy_20261004.json'),
        selected_lock_sha256=final['lock_sha256'],report_receipt_sha256=sha(ROOT/'reports/original_plan_report_receipt.json'),
        allowed_parent_directories=[str(x.resolve()) for _,x in groups],files=files,total_bytes=sum(x['bytes'] for x in files),
        preserved_model_files=model_snapshot,preserved_U1_model=dict(path=str(models[0].resolve()),sha256=assets['U1']['sha256']),
        preserved_metadata=metadata,protected_baseline_caches=protected_caches,excluded_unknown_files=excluded,
        other_baselines_and_all_ambiguous_assets_retained=True,recursive_directory_delete_forbidden=True)
    write_json(PLAN,plan)
    print('READY',len(files),plan['total_bytes'],'bytes; models kept',len(model_snapshot),'excluded',len(excluded))


if __name__=='__main__':
    main()
