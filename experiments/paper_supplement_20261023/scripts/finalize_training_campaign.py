"""Strict evidence gate and data-derived final tables. Never fills missing results or selects checkpoints."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from verified_ssh_download import ROOT, sha, write

NAMES=[a+'_seed'+str(s) for s in (42,2024,3407) for a in ('C','S')]
LABELS={'baseline':'原HRDecoder','full':'完整系统','no_h4':'去H4','no_convnext':'去ConvNeXt','matched_c20':'匹配C20替换S20'}
ENGLISH={'baseline':'Original HRDecoder','full':'Full system','no_h4':'Without H4','no_convnext':'Without ConvNeXt','matched_c20':'Matched C20 replacement'}


def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))


def gate():
    requirements={
      'original_baseline_phase':('results/original_baseline_phase_receipt.json','status','ORIGINAL_BASELINE_PHASE_COMPLETE'),
      'phase1':('results/phase1_evaluation_receipt.json','status','COMPLETE'),
      'full_checkpoints':('evidence/terminal_checkpoint_validation/summary.json','status','ALL_REQUIRED_TERMINALS_AND_ORIGINAL_BASELINE_VALIDATED'),
      'full_backups':('results/terminal_backup_status.json','all_terminal_backups_complete',True),
      'three_pairs':('results/paired_seed_summary/summary.json','status','COMPLETE_THREE_PAIRED_SEEDS'),
      'baseline_masks':('results/baseline_original_local/analysis/verified_summary.json','status','PASS_LOCAL_MASK_AND_GT_SHA_RECOUNT'),
      'baseline_MA':('results/baseline_original_threeway/summary.json','status','COMPLETE_THREEWAY_BASELINE_MA_COMPARISON'),
      'baseline_same_hardware_cost':('results/baseline_original_profile_local/summary.json','status','COMPLETE'),
      'baseline_training_history':('evidence/recovered_original_baseline/history_validation_nodes.json','status','PASS_ORIGINAL_LOG_RETAINED'),
      'baseline_real_figure_QA':('evidence/baseline_original_threeway_visual_review.json','status','PASS'),
      'superseded_attempt_preservation':('evidence/superseded_baseline_v2_recovery.json','status','PASS_SUPERSEDED_PARTIAL_ATTEMPT_PRESERVED_LOCALLY'),
    }
    missing=[]
    for label,(path,key,expected) in requirements.items():
        if not (ROOT/path).exists():missing.append({'gate':label,'path':path,'reason':'not yet available'})
        elif read(path).get(key)!=expected:missing.append({'gate':label,'path':path,'reason':'required completion state not reached'})
    for name in NAMES:
        path='results/paired_seed59/'+name+'/analysis/verified_summary.json'
        if not (ROOT/path).exists() or read(path).get('status')!='PASS_LOCAL_MASK_AND_GT_SHA_RECOUNT':
            missing.append({'gate':'local_prediction_recount','name':name})
    value={'status':'READY_FOR_FINAL_INTEGRITY_AUDIT' if not missing else 'WAITING_REQUIRED_EXPERIMENT_EVIDENCE',
           'missing':missing,'overall_campaign_complete':False,'updated_unix':time.time()}
    write(ROOT/'results/campaign_gate.json',value)
    return value


def audit(require_all_pairs=True):
    files={}
    def record(path,expected=None):
        p=Path(path);p=p if p.is_absolute() else ROOT/p
        digest=sha(p)
        if expected is not None:assert digest==expected,('Changed evidence',p)
        files[str(p.relative_to(ROOT))]=digest
    phase=read('results/phase1_evaluation_receipt.json')
    for key,path in [('ddr_analysis_sha256','results/ddr_test_local_v1/analysis/summary.json'),
                     ('external_analysis_sha256','results/idrid_holdout_v1/analysis/summary.json'),
                     ('profile_analysis_sha256','results/full_system_profile_v1/analysis/summary.json'),
                     ('remote_integrity_audit_sha256','evidence/remote_independent_integrity_audit.json')]:record(path,phase[key])
    for p in sorted((ROOT/'evidence/terminal_backups').glob('*.json')):
        row=json.loads(p.read_text(encoding='utf-8'));record(p)
        if 'path' in row and 'sha256' in row:record(row['path'],row['sha256'])
    names=NAMES if require_all_pairs else [n for n in NAMES if (ROOT/('results/paired_seed59/'+n+'/analysis/verified_summary.json')).exists()]
    for name in names+['baseline_original_seed42']:
        receipt=read('evidence/recovered_evaluations/'+name+'.json');record(receipt['local_archive'],receipt['archive_sha256'])
        run=ROOT/('results/baseline_original_local' if name=='baseline_original_seed42' else 'results/paired_seed59/'+name)
        index=json.loads((run/'recovery_manifest.json').read_text(encoding='utf-8'))
        for row in index['files']:record(row['path'],row['sha256'])
        verified=json.loads((run/'analysis/verified_summary.json').read_text(encoding='utf-8'))
        assert verified['source_archive_sha256']==receipt['archive_sha256']
        record(run/'analysis/verified_summary.json');record(run/'analysis/pixel_counts.npz')
        if name!='baseline_original_seed42':
            seed=int(name.split('seed')[1]);arm=name[0]
            manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
            assert manifest['job']['parent_full_sha256']=='68d7b5821fab1210b3336cf2a63b9de893f24fc32876f5255d7d662084b5e728'
            assert manifest['job']['runtime_protocol_sha256']==sha(ROOT/'plan/paired_evaluation_runtime.md')
            if seed!=42:
                terminal=read('evidence/terminal_checkpoint_validation/'+name+'.json')
                assert verified['model_checkpoint_SHA_from_run']==terminal['EMA_export_sha256']
    # Phase1 integer totals and hashes are checked again without repeating inference.
    from analyze_recovered_training_results import score
    for cohort,runname in [('ddr','ddr_test_local_v1'),('idrid','idrid_holdout_v1')]:
        run=ROOT/'results'/runname;summary=read('results/'+runname+'/analysis/summary.json')
        records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((run/'per_image').glob('*.json'))]
        assert len(records)==(227 if cohort=='ddr' else 27)
        for row in records:record(run/'predictions'/(row['sid']+'.npz'),row['mask_archive_sha256'] if 'mask_archive_sha256' in row else row['mask_sha256'])
        for row in summary['pixel_table']:
            computed=score([r['counts'][row['method']] for r in records])
            assert abs(computed['mIoU_pct']-row['miou_pct'])<1e-10
    original=read('evidence/recovered_original_baseline_identity.json')
    assert original['status']=='PASS_RECOVERED_HISTORICAL_BASELINE_IDENTITY_AND_LOCAL_BACKUP'
    record(original['local_checkpoint'],original['source_checkpoint_sha256'])
    record(original['inference_config'],original['inference_config_sha256'])
    for row in original['source_files']:record(row['local'],row['sha256'])
    history=read('evidence/recovered_original_baseline/history_validation_nodes.json')
    record('evidence/recovered_original_baseline/01_板块一_baseline.json',history['log_sha256'])
    assert {r['training_node'] for r in history['validation_nodes']}==set(range(4000,40001,4000))
    stage3=read('results/original_baseline_phase_receipt.json')
    record('阶段3_原基线找回与复核.md',stage3['report_sha256'])
    for path,digest in stage3['input_evidence_sha256'].items():record(path,digest)
    review=read('evidence/baseline_original_threeway_visual_review.json');examples=read('results/baseline_original_threeway/example_provenance.json')['examples']
    assert len(examples)==9 and len(review['figures'])==9
    lookup={Path(r['path']).resolve():r['sha256'] for r in review['figures']}
    for row in examples:record(row['figure'],lookup[Path(row['figure']).resolve()])
    cost=read('results/baseline_original_profile_local/summary.json');fullcost=read('results/full_system_profile_v1/analysis/summary.json')
    assert cost['hardware']==fullcost['hardware'] and cost['sample_ids']==fullcost['profile_samples']
    assert cost['warmup_images_excluded']==fullcost['warmup_images_excluded']==1
    assert np.isfinite(cost['pipeline_to_native_binary_mask_seconds']['per_image']).all()
    export=read('evidence/recovered_original_baseline_export.json')
    assert export['status']=='PASS' and export['all_state_tensors_bitwise_equal']
    assert export['source_checkpoint_sha256']==cost['source_checkpoint_sha256']
    assert cost['source_checkpoint_sha256']==original['source_checkpoint_sha256']
    assert read('results/baseline_original_local/manifest.json')['job']['weight_sha256']==export['compact_sha256']
    record('models/recovered_hrdecoder_baseline_inference.pth',export['compact_sha256'])
    record('evidence/recovered_original_baseline_export.json')
    record('evidence/baseline_checkpoint_iteration_semantics.json')
    record('evidence/recovered_original_baseline_strict_load.json')
    record('evidence/baseline_v2_superseded_by_recovered_original.json')
    partial=read('evidence/superseded_baseline_v2_recovery.json')
    assert partial['not_a_completed40k_run'] and partial['last_logged_training_update']==2980
    record(partial['checkpoint_local'],partial['checkpoint_sha256'])
    record(partial['history_archive'],partial['history_archive_sha256'])
    for row in partial['history_files']:record(ROOT/'evidence/superseded_baseline_v2_history'/row['path'],row['sha256'])
    record('evidence/superseded_baseline_v2_recovery.json')
    record('evidence/recovered_project_checkpoint_inventory.json')
    record('results/baseline_original_threeway/FIGURE_NOTES.md')
    record('results/baseline_original_profile_local/summary.json');record('results/baseline_original_threeway/summary.json')
    pairs=read('results/paired_seed_summary/summary.json')
    available=pairs['paired_seeds_available']
    if require_all_pairs:assert available==[42,2024,3407]
    assert len(pairs['per_seed'])==4*len(available) and all(r['paired_seeds']==len(available) for r in pairs['mean_and_sample_SD'])
    # Recompute reported mean/SD directly from all three fixed paired outcomes.
    for row in pairs['mean_and_sample_SD']:
        members=[r for r in pairs['per_seed'] if r['cohort']==row['cohort'] and r['mode']==row['mode']]
        for key in ('C_mIoU_pct','S_mIoU_pct','S_minus_C_pp'):
            values=np.array([r[key] for r in members]);assert len(values)==len(available)
            assert abs(values.mean()-row[key+'_mean'])<1e-10 and abs(values.std(ddof=1)-row[key+'_sample_SD'])<1e-10
    record('results/paired_seed_summary/summary.json')
    for path in ['plan/additional_training_protocol.json','plan/paired_evaluation_runtime.md','plan/protocol.md',
                 'plan/original_baseline_recovery_amendment.json','plan/baseline_reconstruction_amendment.json',
                 'evidence/external_holdout_manifest.json','evidence/matched_lineage_audit.json']:
        record(path)
    for name in ['analyze_recovered_training_results.py','analyze_original_baseline_threeway.py',
                 'lesion_metrics.py','verify_terminal_checkpoints.py','evaluate_original_baseline_local.py',
                 'evaluate_paired_seed59.py','evaluation59_common.py','recover_original_baseline.py']:
        record('scripts/'+name)
    return files


def mdtable(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']+['---:']*(len(headers)-1))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'


def csv_table(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--audit-available',action='store_true');args=parser.parse_args()
    state=gate()
    if args.audit_available:
        assert not args.finalize,'Partial preflight cannot publish final outputs'
        files=audit(require_all_pairs=False)
        result={'status':'PASS_AVAILABLE_EVIDENCE_ONLY','input_sha256':files,'overall_campaign_complete':False,
                'missing_final_gates':state['missing'],'updated_unix':time.time()}
        write(ROOT/'evidence/available_campaign_integrity_audit.json',result)
        print(json.dumps({'status':result['status'],'inputs_audited':len(files),'overall_campaign_complete':False}));return
    if state['missing'] or not args.finalize:
        print(json.dumps(state,ensure_ascii=False));return
    assert read('plan/original_baseline_recovery_amendment.json')['baseline_primary']=='Original historical DDR_split613 seed42 best32000, not retrained v1/v2'
    files=audit();out=ROOT/'results/final_campaign';out.mkdir(exist_ok=True)
    d=read('results/ddr_test_local_v1/analysis/summary.json');e=read('results/idrid_holdout_v1/analysis/summary.json')
    ma=read('results/baseline_original_threeway/summary.json');base=read('results/baseline_original_local/analysis/verified_summary.json')
    pairs=read('results/paired_seed_summary/summary.json');cost=read('results/baseline_original_profile_local/summary.json')
    fullcost=read('results/full_system_profile_v1/analysis/summary.json');selection=read('evidence/terminal_backups/baseline_original_selection.json')
    pixel=[]
    for cohort,summary in [('ddr',d),('idrid',e)]:
        b=base['cohorts'][cohort]['baseline'];pixel.append({'cohort':cohort,'method':'baseline',**{k:b[k] for k in (
            'EX_IoU_pct','HE_IoU_pct','SE_IoU_pct','MA_IoU_pct','mIoU_pct','MA_pixel_precision_pct','MA_pixel_recall_pct')}})
        for r in summary['pixel_table']:
            pixel.append({'cohort':cohort,'method':r['method'],**{k:r[k] for k in ('EX_IoU_pct','HE_IoU_pct','SE_IoU_pct','MA_IoU_pct')},
                'mIoU_pct':r['miou_pct'],'MA_pixel_precision_pct':r['MA_precision_pct'],'MA_pixel_recall_pct':r['MA_recall_pct']})
    csv_table(out/'baseline_and_components.csv',pixel);csv_table(out/'three_seed_pairs.csv',pairs['per_seed'])
    csv_table(out/'three_seed_mean_SD.csv',pairs['mean_and_sample_SD']);csv_table(out/'MA_threeway.csv',ma['MA_lesion_rows'])
    efficiency=[{'method':'baseline','mean_seconds_per_image':cost['pipeline_to_native_binary_mask_seconds']['mean'],
                 'registered_parameters':cost['registered_parameters'],'inference_weight_bytes':cost['inference_weight_artifact_bytes'],
                 'max_single_process_allocated_peak_bytes':cost['max_single_process_allocated_peak_bytes']},
                {'method':'full','mean_seconds_per_image':fullcost['pipeline_to_fused_mask_seconds']['mean'],
                 'registered_parameters':fullcost['registered_parameters']['full_system_total_registered_parameters'],
                 'inference_weight_bytes':fullcost['registered_parameters']['weight_artifact_bytes_full'],
                 'max_single_process_allocated_peak_bytes':fullcost['max_single_process_allocated_peak_bytes']}]
    csv_table(out/'same_hardware_efficiency.csv',efficiency)
    text='# 论文补充实验结果\n\n'
    text+='已完成预定四次新增C/S适配训练，并找回原HRDecoder基线权重进行冻结推理。固定终点、预测掩膜、逐图计数、完整检查点与EMA导出已本地验SHA；三组配对、MA三方比较、实际推理成本和真实图已核验。原基线复算42.1962%，四舍五入复核历史42.20%；不使用此前配置错误的重训结果作为主基线。结果包含预定负结果，未据测试结果修改阈值、融合比例或模型节点。\n\n'
    text+='## 原基线与最终系统组件\n\n'
    text+=f'基线为用户找回的原split613 seed42单卡BN模型，由历史原hook在40k日程的10个val节点选中{selection["selected_node"]}步；本轮未重新选择节点。原验证39.390182%，本轮39.392558%，不同Torch/CUDA环境不声称逐位重放。DDR四类IoU与报告近似吻合，SE49.9876与历史49.98约差0.01个百分点，不修改当前整数计数去凑小数。四类顺序EX/HE/SE/MA，严格>0.5，均按当前原图坐标全局累计TP/FP/FN计算；类别IoU后取均值。\n\n'
    metrics=['EX_IoU_pct','HE_IoU_pct','SE_IoU_pct','MA_IoU_pct','mIoU_pct','MA_pixel_precision_pct','MA_pixel_recall_pct']
    for cohort,label in [('ddr','DDR test 227图'),('idrid','DDR训练→IDRiD-derived保留27图冻结评估')]:
        text+='### '+label+'\n\n'
        text+=mdtable(['模型','EX IoU%','HE IoU%','SE IoU%','MA IoU%','mIoU%','MA像素P%','MA像素R%'],
            [[LABELS[r['method']]]+[f'{r[k]:.4f}' for k in metrics] for r in pixel if r['cohort']==cohort])
    text+='四行系统移除/替换保持历史固定配方：A专家权重3:2:1:1并带原水平TTA，P0=0.75A+0.25U，最终=0.75P0+0.25S；去H4保留其余分支与阈值，去U固定重归一化为9/13 A+4/13 S，匹配C保持0.25比例。它们解释固定系统的依赖与替换效果，不证明各路径独立训练的因果贡献。\n\n'
    text+='阶段1完整系统及对照保留原BF16/reference运行时，恢复的原基线为FP32。相同图像/GT/坐标与指标不等于相同训练或集成预算。以下三种子统计另外统一为2080Ti FP32/reference，不能与阶段1分数混合求均值。\n\n'
    text+='历史权重继承需与基准比较区分：E4原日志记录ImageNet初始化、load_from/resume_from为空，B加载E4，D/E加载B；尚无证据建立原42.20%权重直接续训到E4这一边。找回原基线补齐了性能参照，不能倒推历史加载关系。相关原日志及实际权重元数据见evidence/historical_hr_lineage_audit.json。\n\n'
    text+='## 三组固定20k C/S配对\n\n'
    text+='两个新种子2024/3407各训练C/S，均从同一个F12_seed42原始model状态开始，保留原32k学习率日程运行至预定20k，统一取EMA。历史42也按相同FP32/reference协议重评。每种子共享采样流；融合统一为0.75P0+0.25模型，共享逐图P0 SHA，无逐种子选最好节点。\n\n'
    text+='C为匹配日程的普通续训，S为空间适配；F12是这一Transformer分支共同父模型的项目名，不是HRDecoder。三组比较只检验后续适配阶段。\n\n'
    text+=mdtable(['数据','模式','seed','C mIoU%','S mIoU%','S−C pp','图像bootstrap 95%CI'],
        [[r['cohort'],r['mode'],r['seed'],f'{r["C_mIoU_pct"]:.4f}',f'{r["S_mIoU_pct"]:.4f}',f'{r["S_minus_C_pp"]:+.4f}',
          '['+', '.join(f'{x:.4f}' for x in r['paired_image_bootstrap_CI95_pp'])+']'] for r in pairs['per_seed']])
    text+=mdtable(['数据','模式','seed','C MA IoU%','S MA IoU%','MA差值 pp'],
        [[r['cohort'],r['mode'],r['seed'],f'{r["C_MA_IoU_pct"]:.4f}',f'{r["S_MA_IoU_pct"]:.4f}',
          f'{r["S_MA_IoU_pct"]-r["C_MA_IoU_pct"]:+.4f}'] for r in pairs['per_seed']])
    decreases=[r for r in pairs['per_seed'] if r['S_MA_IoU_pct']<r['C_MA_IoU_pct']]
    if decreases:
        text+=('总体mIoU改善不能直接解释为MA改善。以下预定比较中S的MA IoU更低：'+
            '；'.join(str(r['seed'])+' / '+r['cohort']+' / '+r['mode']+
                     f'（{r["S_MA_IoU_pct"]-r["C_MA_IoU_pct"]:+.4f}个百分点）' for r in decreases)+'。\n\n')
    text+=mdtable(['数据','模式','C 均值±样本SD%','S 均值±样本SD%','配对差均值±样本SD pp'],
        [[r['cohort'],r['mode']]+[f'{r[k+"_mean"]:.4f} ± {r[k+"_sample_SD"]:.4f}' for k in ('C_mIoU_pct','S_mIoU_pct','S_minus_C_pp')] for r in pairs['mean_and_sample_SD']])
    for cohort,label in [('ddr','DDR'),('idrid','IDRiD-derived')]:
        rows=[r for r in pairs['per_seed'] if r['cohort']==cohort and r['mode']=='fixed_fusion']
        avg=next(r for r in pairs['mean_and_sample_SD'] if r['cohort']==cohort and r['mode']=='fixed_fusion')
        crossing=sum(r['paired_image_bootstrap_CI95_pp'][0]<=0<=r['paired_image_bootstrap_CI95_pp'][1] for r in rows)
        text+=(label+'固定融合的三组S−C分别为'+
               '、'.join(f'{r["S_minus_C_pp"]:+.4f}' for r in rows)+
               f'个百分点，均值{avg["S_minus_C_pp_mean"]:+.4f}、样本SD {avg["S_minus_C_pp_sample_SD"]:.4f}；'+
               f'{crossing}/3个图像bootstrap区间包含零。'+
               ('DDR三组方向不一致，平均差接近零，本次补证不支持空间分支在最终固定融合中稳定提高DDR mIoU。' if cohort=='ddr' else '该外部子集三组点估计均为正，但只有3组且部分区间跨零，不能由此证明普遍泛化改善。')+'\n\n')
    text+='只有3组，均值与样本SD描述该共同父模型条件下的适配波动，不能称完整系统从头训练的重复性。图像bootstrap使用2000次、seed20261023，和训练种子SD分开；没有患者身份，不能声称患者级重采样。若逐组方向不一致或区间跨零，保留原样，不概括为稳定优越。\n\n'
    text+='## MA病灶三方比较\n\n'
    text+='8连通、保留单像素、IoU≥0.10一对一匹配，先最大匹配数量再最大IoU；合并预测最多对应一个GT，分裂多余预测计FP。大小边界来自训练集6793组件：≤32、33–74、>74像素。FP分母包括所有图；像素recall不作为病灶检出率。外部图像尺度不同，像素分组不能当等价物理尺度。\n\n'
    text+=mdtable(['数据','系统','大小','GT','检出','检出率%','FP/图','病灶P%'],
        [[r['cohort'],LABELS[r['method']],r['size'],r['GT'],r['detected'],f'{r["detection_pct"]:.4f}',
          '—' if r['false_positives_per_image'] is None else f'{r["false_positives_per_image"]:.4f}',
          '—' if r['lesion_precision_pct'] is None else f'{r["lesion_precision_pct"]:.4f}'] for r in ma['MA_lesion_rows']])
    text+='完整系统较去H4恢复更多MA，也增加FP并降低病灶precision；必须同时展示这两方面。恢复的原基线复用原先按sid/组件ID固定选出的9例，成功恢复、仍漏检、新增误检各3例，未按基线结果重选。类别由原完整系统相对去H4定义，恢复不一定是相对原基线的恢复；未匹配预测的裁剪内可包含IoU不足的邻近GT。图及其逐项SHA/选例依据见 results/baseline_original_threeway/example_provenance.json 与 FIGURE_NOTES.md；区间与逐图匹配见同目录summary及基线analysis。\n\n'
    text+='## 相同硬件实际成本与训练预算\n\n'
    text+=mdtable(['系统','秒/图均值','注册参数M','部署权重GiB','单进程allocated峰值GiB'],[
      ['原HRDecoder',f'{cost["pipeline_to_native_binary_mask_seconds"]["mean"]:.4f}',f'{cost["registered_parameters"]/1e6:.3f}',f'{cost["inference_weight_artifact_bytes"]/1024**3:.3f}',f'{cost["max_single_process_allocated_peak_bytes"]/1024**3:.3f}'],
      ['完整系统',f'{fullcost["pipeline_to_fused_mask_seconds"]["mean"]:.4f}',f'{fullcost["registered_parameters"]["full_system_total_registered_parameters"]/1e6:.3f}',f'{fullcost["registered_parameters"]["weight_artifact_bytes_full"]/1024**3:.3f}',f'{fullcost["max_single_process_allocated_peak_bytes"]/1024**3:.3f}']])
    text+='相同RTX4060 Laptop、batch1、相同三张固定val原图，另外一次warmup排除。计时含图像读入/变换、全部实际前向、搬移、原图对齐及最终掩膜，完整系统还含跨环境IPC和融合；不含模型加载、GT计数及压缩落盘。注册参数包括复制reference/辅助模块，不是去重参数或FLOPs；单进程allocated峰值不含驱动上下文，完整系统顺序执行，不能当整机峰值。整卡200ms采样峰值及每图时间另见两份profile summary。三张图仅描述当前机器成本，未测多卡吞吐。\n\n'
    text+='新增适配预算为4×20000更新、每更新8个原像素窗口。另有v1基线40k配置错误尝试、v2修正尝试至2980步（最后完整保存2k），找回原权重后终止v2；全部日志/已存权重保留，不把失败尝试隐去，也不称v2完成40k。原基线本轮只推理。最终系统另依赖既有专家、ConvNeXt、父模型和H4训练，不能宣称总训练预算相等。四个适配完整状态含优化器/EMA/RNG/采样身份；原历史基线保存模型和优化器及原训练日志，不能声称其也含完整RNG或从本轮成本倒推全部历史成本。\n\n'
    text+='## 独立验证与解释边界\n\n'
    text+='用户明确确认IDRiD_55–81从未用于训练、选型、调参或比较方案；全流程冻结、没有微调。选中C/S血缘为DDR-only Q_DDR→F12；其他未选中项目探索使用过IDRiD training54。27图/TXT/108掩膜SHA、重新栅格化与1036图历史重复审计通过，未发现重复；无患者身份，不能证明患者级隔离。当前3415–4033方形JPEG及用户确认官方GT无损转换的标签缺少原始官方GT独立逐像素比对，结果应标为IDRiD-derived保留子集，不作为官方原始几何基准分数。\n\n'
    text+='预定负结果全部保留：阶段1外部去ConvNeXt高0.3050 mIoU百分点，差异区间跨零；H4同时增加病灶误检；固定九例中部分仍漏检的目标曾被原基线检出，均如实保留；seed42的DDR匹配C/S替换差区间跨零。外部≤32像素组仅4个GT，不能将这个分组百分比概括为稳健外部小病灶性能。最终新种子结果逐组呈现，不据此继续搜索权重。DDR test历史上已用于开发，补跑不能撤销这段历史。空间/细节/学生路径有继承关系，这些补证不能证明每条路径独立训练贡献，也不自动建立重大模型创新或临床有效性。\n\n'
    text+='## 可复核文件\n\n'
    text+='- results/final_campaign：总体CSV、三组逐项及均值/SD、MA三方表、LaTeX表及输入SHA索引。\n- results/paired_seed59、baseline_original_local：原图二值预测、逐图整数计数与本地GT重算。\n- models、evidence/terminal_backups、terminal_checkpoint_validation：四个完整终点/EMA、恢复的原基线best32k及配置不匹配旧重训资产、全状态/哈希核验。\n- evidence/recovered_original_baseline：原历史训练日志、全部val节点及原选择依据。\n- 阶段1_推理与离线分析.md：已冻结阶段1详细结果；其中“待新增训练”属于当时状态，本报告补齐。\n\n两台服务器保持开机，未经用户另行指示不关闭。本报告仅表示上述实验与证据交付完成，不表示论文已投稿或达到期刊创新门槛。\n'
    report=ROOT/'论文补充实验结果.md';report.write_text(text,encoding='utf-8')
    tex=['% Requires booktabs. Percent values; see report for runtime and cohort limitations.']
    for cohort in ('ddr','idrid'):
        tex += [r'\begin{table}[t]',r'\centering\small',r'\caption{Original baseline and fixed-system substitutions on '+('DDR test (227 images)' if cohort=='ddr' else 'the DDR-trained frozen IDRiD-derived holdout (27 images)')+r'. All values are percentages.}',
                r'\begin{tabular}{lrrrrrrr}',r'\toprule',r'System & EX & HE & SE & MA & mIoU & MA P & MA R \\',r'\midrule']
        tex += [ENGLISH[r['method']]+' & '+' & '.join(f'{r[k]:.2f}' for k in metrics)+r' \\' for r in pixel if r['cohort']==cohort]
        tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    tex += [r'\begin{table}[t]',r'\centering\small',r'\caption{Three conditional paired seeds: mean and sample SD.}',
            r'\begin{tabular}{llrrr}',r'\toprule',r'Cohort & Mode & C & S & S minus C \\',r'\midrule']
    for r in pairs['mean_and_sample_SD']:
        tex.append(r['cohort']+' & '+r['mode'].replace('_',' ')+ ' & '+' & '.join(
            f'${r[k+"_mean"]:.4f} \\pm {r[k+"_sample_SD"]:.4f}$' for k in ('C_mIoU_pct','S_mIoU_pct','S_minus_C_pp'))+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    tex += [r'\begin{table}[t]',r'\centering\small',
            r'\caption{All fixed 20k paired outcomes. Intervals are paired image bootstrap intervals, not seed variability.}',
            r'\begin{tabular}{llrrrrl}',r'\toprule',r'Cohort & Mode & Seed & C & S & Difference & 95\% CI \\',r'\midrule']
    for r in pairs['per_seed']:
        lower,upper=r['paired_image_bootstrap_CI95_pp']
        tex.append(r['cohort']+' & '+r['mode'].replace('_',' ')+' & '+str(r['seed'])+' & '+
                   ' & '.join(f'{r[k]:.4f}' for k in ('C_mIoU_pct','S_mIoU_pct','S_minus_C_pp'))+
                   f' & [{lower:.4f}, {upper:.4f}]'+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    for cohort in ('ddr','idrid'):
        tex += [r'\begin{table}[t]',r'\centering\small',
                r'\caption{MA lesion detection on '+cohort+r'. Pixel area bins are defined on training data; they are not physical-size bins.}',
                r'\begin{tabular}{llrrrlrr}',r'\toprule',
                r'System & Size & GT & Detected & Recall (\%) & 95\% CI & FP/image & Precision (\%) \\',r'\midrule']
        for r in ma['MA_lesion_rows']:
            if r['cohort']!=cohort:continue
            lower,upper=r['detection_image_CI95_pct']
            fp='--' if r['false_positives_per_image'] is None else f'{r["false_positives_per_image"]:.2f}'
            precision='--' if r['lesion_precision_pct'] is None else f'{r["lesion_precision_pct"]:.2f}'
            tex.append(ENGLISH[r['method']]+' & '+r['size']+f' & {r["GT"]} & {r["detected"]} & {r["detection_pct"]:.2f} & [{lower:.2f}, {upper:.2f}] & '+fp+' & '+precision+r' \\')
        tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    tex += [r'\begin{table}[t]',r'\centering\small',
            r'\caption{Actual inference cost on the same RTX 4060 Laptop GPU. Three fixed validation images and one excluded warmup; registered counts include auxiliary/reference copies.}',
            r'\begin{tabular}{lrrrr}',r'\toprule',r'System & Seconds/image & Parameters (M) & Weights (GiB) & Allocated peak (GiB) \\',r'\midrule']
    for r in efficiency:
        tex.append(ENGLISH[r['method']]+f' & {r["mean_seconds_per_image"]:.3f} & {r["registered_parameters"]/1e6:.3f} & {r["inference_weight_bytes"]/1024**3:.3f} & {r["max_single_process_allocated_peak_bytes"]/1024**3:.3f}'+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (out/'paper_tables.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    write(out/'numeric_table_sources.json',{'status':'ALL_REQUIRED_EVIDENCE_AUDITED','input_sha256':files,
        'selectors':{'baseline_and_components.csv':'baseline verified_summary cohorts and phase1 analysis pixel_table',
            'three_seed_pairs.csv':'paired_seed_summary summary.per_seed','three_seed_mean_SD.csv':'summary.mean_and_sample_SD',
            'MA_threeway.csv':'baseline_original_threeway summary.MA_lesion_rows',
            'same_hardware_efficiency.csv':'baseline_original_profile_local summary and full_system_profile_v1 analysis summary'},'all_predeclared_negative_outcomes_retained':True})
    outputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    outputs[str(report.relative_to(ROOT))]=sha(report)
    # Generation is not delivery: bind the actual report/table review to every output hash.
    review_path=ROOT/'evidence/final_report_review.json'
    review=read('evidence/final_report_review.json') if review_path.exists() else None
    reviewed=bool(review and review.get('status')=='PASS_FINAL_REPORT_REVIEW' and
                  review.get('reviewed_outputs_sha256')==outputs)
    final_status='COMPLETE_ALL_AUTHORIZED_EXPERIMENTS_AND_LOCAL_EVIDENCE' if reviewed else 'RESULTS_GENERATED_AWAITING_REPORT_REVIEW'
    write(ROOT/'results/campaign_completion_receipt.json',{'status':final_status,
        'overall_campaign_complete':reviewed,'new_adaptation_training_runs':4,'baseline_training_attempts':2,'completed_rejected_baseline_attempt_updates':40000,'superseded_v2_attempt_updates':2980,'primary_baseline_recovered':True,'primary_baseline_new_training_required':False,'paired_seeds':[42,2024,3407],
        'prediction_sets_locally_verified':7,'output_sha256':outputs,'inputs_audited':len(files),'finalizer_sha256':sha(Path(__file__)),
        'report_review_sha256':sha(review_path) if reviewed else None,
        'servers_shutdown':False,'updated_unix':time.time()})
    print(json.dumps({'status':final_status,'report':str(report),'overall_campaign_complete':reviewed},ensure_ascii=False))


if __name__=='__main__':main()
