"""CPU-only terminal model/EMA/optimizer/RNG/stream validation after verified local backup."""
import hashlib
import json
from pathlib import Path
import torch
from verified_ssh_download import ROOT,sha,write

PARENT='68d7b5821fab1210b3336cf2a63b9de893f24fc32876f5255d7d662084b5e728'
torch.set_num_threads(2)

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()

def finite_tensors(value):
    if isinstance(value,torch.Tensor):
        assert value.device.type=='cpu'
        if value.is_floating_point():assert torch.isfinite(value).all(), 'Nonfinite checkpoint tensor'
        return 1
    if isinstance(value,dict):return sum(finite_tensors(v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(finite_tensors(v) for v in value)
    return 0

def main():
    receipts=ROOT/'evidence/terminal_backups';out=ROOT/'evidence/terminal_checkpoint_validation';out.mkdir(exist_ok=True)
    verified=[]
    for seed in (2024,3407):
        for arm in ('C','S'):
            name=arm+'_seed'+str(seed);full_receipt=receipts/(name+'_full_u20000.json');ema_receipt=receipts/(name+'_EMA20k.json')
            if not (full_receipt.exists() and ema_receipt.exists()):continue
            full=json.loads(full_receipt.read_text());export=json.loads(ema_receipt.read_text());target=out/(name+'.json')
            if target.exists():
                previous=json.loads(target.read_text());assert previous['full_checkpoint_sha256']==full['sha256'] and previous['EMA_export_sha256']==export['sha256'];verified.append(name);continue
            f=Path(full['path']);e=Path(export['path']);assert sha(f)==full['sha256'] and sha(e)==export['sha256']
            state=torch.load(f,map_location='cpu',weights_only=False,mmap=True);compact=torch.load(e,map_location='cpu',weights_only=False,mmap=True)
            assert state['completed']==state['ema_completed']==state['scheduler']['completed']==20000 and state['scheduler']['horizon']==32000
            identity=state['identity'];assert identity['arm']==arm and identity['seed']==seed and identity['parent_full_sha256']==PARENT
            assert digest(identity)==state['identity_sha256'] and state['stream_cursor']=={'DDR':160000,'TJDR':0}
            source=ROOT/'recovered_training_provenance/training_v1'
            for rel,expected in identity['source_files'].items():assert sha(source/rel)==expected
            assert sha(source/'protocol.json')==identity['protocol_sha256']
            assert sha(source/'reports/data_identity.json')==identity['data_manifest_sha256']
            parent_receipt=json.loads((ROOT/'evidence/training_parent_export.json').read_text())
            assert parent_receipt['source_sha256']==PARENT and parent_receipt['model_tensors_bitwise_equal']
            assert identity['parent_compact_sha256']==parent_receipt['compact_sha256']
            stream=source/f'reports/new_stream_DDR_seed{seed}.jsonl';assert sha(stream)==identity['stream_sha256']
            next_rows=[];n=0
            with stream.open('r',encoding='utf-8') as handle:
                for n,text in enumerate(handle,1):
                    if n>160000:next_rows.append(json.loads(text))
            assert n==160008 and len(next_rows)==8 and digest({'DDR':next_rows,'TJDR':[]})==state['next_rows_sha256']
            assert set(state['rng'])=={'python','numpy','torch','cuda'} and len(state['rng']['cuda'])==1
            assert state['rng']['torch'].dtype==torch.uint8 and state['rng']['cuda'][0].dtype==torch.uint8
            assert len(state['rng']['python'])==3 and len(state['rng']['numpy'])==5
            assert state['rng']['torch'].numel()>0 and state['rng']['cuda'][0].numel()>0
            assert state['optimizer']['state'] and state['optimizer']['param_groups']
            tensors=finite_tensors({'model':state['model'],'ema':state['ema'],'optimizer':state['optimizer']})
            assert compact['completed']==compact['ema_completed']==20000
            assert set(compact['ema'])==set(state['ema']) and all(torch.equal(v,compact['ema'][k]) for k,v in state['ema'].items())
            write(target,{'status':'PASS','arm':arm,'seed':seed,'completed':20000,'EMA_completed':20000,'schedule_horizon':32000,
                 'parent_full_sha256':PARENT,'full_checkpoint_sha256':full['sha256'],'EMA_export_sha256':export['sha256'],
                 'all_floating_model_EMA_optimizer_tensors_finite':True,'tensors_checked':tensors,'all_EMA_export_tensors_bitwise_equal':True,
                 'source_files_protocol_and_stream_SHA':'PASS','next8sample_rows_and160000cursor':'PASS','complete_RNG_and_optimizer_states':'PASS',
                 'CPU_only_no_training_or_inference':True})
            del state,compact;verified.append(name)
    original=(ROOT/'evidence/recovered_original_baseline_identity.json').exists()
    corrected=(ROOT/'plan/baseline_reconstruction_amendment.json').exists()
    prefix='baseline_v2' if corrected else 'baseline'
    baseline_selection=receipts/(prefix+'_selection.json');baseline_complete=False
    if original:
        identity=json.loads((ROOT/'evidence/recovered_original_baseline_identity.json').read_text(encoding='utf-8'))
        strict=json.loads((ROOT/'evidence/recovered_original_baseline_strict_load.json').read_text())
        assert identity['status']=='PASS_RECOVERED_HISTORICAL_BASELINE_IDENTITY_AND_LOCAL_BACKUP'
        assert strict['status']=='PASS_STRICT_ALL_1844_STATE_KEYS'
        assert sha(Path(identity['local_checkpoint']))==identity['source_checkpoint_sha256']
        assert sha(Path(identity['compact_checkpoint']))==identity['compact_sha256']
        export=json.loads((ROOT/'evidence/recovered_original_baseline_export.json').read_text())
        assert export['all_state_tensors_bitwise_equal'] and export['compact_sha256']==identity['compact_sha256']
        write(out/'baseline_original_seed42.json',{'status':'PASS','identity':'Recovered historical split613 seed42 best32000',
            'selected_node':32000,'source_checkpoint_sha256':identity['source_checkpoint_sha256'],
            'compact_sha256':identity['compact_sha256'],'full_model_and_optimizer_finite':True,
            'original_config_and_log_match':True,'strict_full_state_load':True,'new_baseline_training_required':False})
        baseline_complete=True
    elif baseline_selection.exists():
        job=json.loads(baseline_selection.read_text());names=[prefix+'_best_u'+str(job['selected_node']),prefix+'_endpoint_u40000']
        if corrected:
            amendment=json.loads((ROOT/'plan/baseline_reconstruction_amendment.json').read_text())
            assert job['kind']=='baseline_v2' and job['config_sha256']==amendment['corrected_config_sha256']
            assert job['restored_scale_ratio']==[.75,1.25]
        if all((receipts/(name+'.json')).exists() for name in names):
            target=out/(prefix+'_seed42.json')
            if not target.exists():
                summaries={}
                for name in names:
                    receipt=json.loads((receipts/(name+'.json')).read_text());path=Path(receipt['path']);assert sha(path)==receipt['sha256']
                    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
                    if '_best_u' in name:
                        assert saved['meta']['iter']==job['selected_node']-1
                        assert Path(saved['meta']['hook_msgs']['best_ckpt']).name==Path(job['checkpoint_path']).name
                        assert abs(saved['meta']['hook_msgs']['best_score']-max(r['original_log_record']['mIoU'] for r in job['all_validation_records']))<1e-5
                    else:
                        # Original MMCV saves after the update and before incrementing
                        # IterBasedRunner._iter: filename40000, zero-based meta.iter39999.
                        semantics=json.loads((ROOT/'evidence/baseline_checkpoint_iteration_semantics.json').read_text())
                        assert semantics['status']=='PASS_ORIGINAL_ZERO_BASED_METADATA_SEMANTICS'
                        assert 'filename_tmpl.format(self.iter + 1)' in semantics['save_checkpoint']
                        assert 'iter=self.iter)' in semantics['save_checkpoint']
                        assert saved['meta']['iter']==39999
                        assert receipt['sha256']==job['endpoint_checkpoint_sha256']
                    assert saved['optimizer']['state']
                    summaries[name]={'sha256':receipt['sha256'],'state_tensors':finite_tensors(saved['state_dict']),'optimizer_tensors':finite_tensors(saved['optimizer'])}
                    del saved
                assert {r['training_node'] for r in job['all_validation_records']}==set(range(4000,40001,4000))
                write(target,{'status':'PASS','training_endpoint':40000,'endpoint_zero_based_metadata_iter':39999,
                     'original_MMCV_iteration_semantics_verified':True,'selected_by_original_validation_hook':True,
                     'selected_node':job['selected_node'],'all10predeclared_validation_nodes':True,'artifacts':summaries,
                     'identity':'New baseline reproduction, not historical42.20 model recovery','CPU_only':True})
            baseline_complete=True
    result={'status':('ALL_REQUIRED_TERMINALS_AND_ORIGINAL_BASELINE_VALIDATED' if original else 'ALL_FIVE_NEW_TRAINING_TERMINALS_LOCALLY_VALIDATED') if len(verified)==4 and baseline_complete else 'PARTIAL',
            'new_adaptation_runs_validated':verified,'baseline_validated':baseline_complete,
            'required_baseline_version':'recovered_original32k' if original else (2 if corrected else 1),'overall_campaign_complete':False}
    write(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
