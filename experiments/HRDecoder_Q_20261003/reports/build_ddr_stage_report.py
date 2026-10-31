"""Summarize the fixed two-seed result from recovered counts; no inference or selection."""
from datetime import datetime
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,sha,utc_now,CLASSES
from qlib.metrics import summarize

def read(rel): return read_json(ROOT/rel)
def score(a):
    a=np.asarray(a)
    return (a[...,0]/np.maximum(a.sum(axis=-1),1)).mean(axis=-1)*100
def ci(a): return np.quantile(a,[.025,.975]).tolist()
def timestamp(s): return datetime.fromisoformat(s)

summary=read('server_results/reports/final_test/summary.json')
lock=read('reports/final_selection.json')
receipt=read('reports/final_evidence_backup_receipt.json')
assert receipt['local_backup_state']=='SHA_VERIFIED' and receipt['lock_sha256']==sha(ROOT/'reports/final_selection.json')
assert summary['status']=='COMPLETE' and summary['formula']=='fixed_fusion' and summary['update']==16000
assert not summary['selection_changed_after_test']
results={s:read(f'server_results/reports/final_test/seed{s}.json') for s in (42,2024)}
ids=sorted(results[42]['images'])
assert len(ids)==227 and set(ids)==set(results[2024]['images'])
old=np.asarray([results[42]['images'][sid]['p0'] for sid in ids],np.int64)
assert np.array_equal(old,np.asarray([results[2024]['images'][sid]['p0'] for sid in ids]))
p0=summarize(old.sum(0))
rng=np.random.default_rng(100303)
indices=rng.integers(0,len(ids),(2000,len(ids)))
p0_boot=score(old[indices].sum(1))
per_seed={};boot=[];leave=[]
for seed,result in results.items():
    assert result['status']=='COMPLETE' and result['lock_sha256']==receipt['lock_sha256']
    ct=np.asarray([result['images'][sid]['selected'] for sid in ids],np.int64)
    aggregate=summarize(ct.sum(0))
    assert aggregate==result['selected'] and result['p0']==p0
    sampled=score(ct[indices].sum(1));boot.append(sampled)
    loo=score(ct.sum(0)[None]-ct)-score(old.sum(0)[None]-old);leave.append(loo)
    old_bucket=np.asarray(result['p0_lesion_buckets'],np.int64)
    new_bucket=np.asarray(result['selected_lesion_buckets'],np.int64)
    assert np.array_equal(old_bucket[:,:,3].sum(1),old.sum(0)[:,2])
    assert np.array_equal(new_bucket[:,:,3].sum(1),ct.sum(0)[:,2])
    assert np.array_equal(new_bucket[:,:,2].sum(1),ct.sum(0)[:,0]+ct.sum(0)[:,2])
    area=[]
    for i,c in enumerate(CLASSES):
        for b,upper in enumerate(result['lesion_area_upper_bounds']):
            o,n=old_bucket[i,b],new_bucket[i,b]
            area.append({'class':c,'area_upper_bound':upper,'lesions':int(o[0]),
                'p0_whole_missed':int(o[1]),'selected_whole_missed':int(n[1]),
                'gt_pixels':int(o[2]),'p0_fn':int(o[3]),'selected_fn':int(n[3]),
                'p0_boundary_fn':int(o[4]),'selected_boundary_fn':int(n[4]),
                'p0_body_fn':int(o[3]-o[4]),'selected_body_fn':int(n[3]-n[4])})
    t=np.asarray([result['images'][sid]['q_inference_seconds'] for sid in ids])
    per_seed[str(seed)]={'selected':aggregate,'delta_vs_p0_pp':aggregate['miou_pct']-p0['miou_pct'],
        'absolute_image_percentile95_pct':ci(sampled),'paired_image_percentile95_pp':ci(sampled-p0_boot),
        'q_image_inference_seconds':{'mean':float(t.mean()),'median':float(np.median(t)),
            'p90':float(np.quantile(t,.9)),'total':float(t.sum()),
            'scope':'Q forward and CPU probability stitching; excludes P0, disk SHA and lesion diagnostics'},
        'lesion_area_error_buckets':area,'evaluation_seconds_including_IO_and_diagnostics':result['seconds']}

mean_boot=np.mean(boot,axis=0)
mean_loo=np.mean(leave,axis=0)
preflight=read('server_results/reports/gpu_preflight.json')
state=read('server_results/reports/final_evaluation_state.json')
training_seconds={str(s):read(f'server_results/runs/Q_DDR_seed{s}/status.json')['total_seconds_this_invocation'] for s in (42,2024)}
backups={str(s):read(f'reports/run_backup_Q_DDR_seed{s}.json') for s in (42,2024)}
assert all(x['complete'] and len(x['files'])==169 for x in backups.values())
analysis={'created_at':utc_now(),'scope':'Q_DDR completed stage; original plan still continues with official external supervision',
    'lock_sha256':receipt['lock_sha256'],'selection_changed':False,'deployment_seed':42,
    'p0':p0,'p0_absolute_image_percentile95_pct':ci(p0_boot),'seeds':per_seed,
    'two_seed_mean':{'miou_pct':summary['mean_test_pct'],'delta_vs_p0_pp':summary['mean_test_pct']-p0['miou_pct'],
        'range_pct':summary['range_test_pct'],'seed_spread_pp':max(summary['seed_scores_pct'])-min(summary['seed_scores_pct']),
        'absolute_paired_image_percentile95_pct':ci(mean_boot),
        'delta_paired_image_percentile95_pp':ci(mean_boot-p0_boot),
        'leave_one_image_out_delta_range_pp':[float(mean_loo.min()),float(mean_loo.max())]},
    'bootstrap':{'draws':2000,'seed':100303,'unit':'paired image','mean_definition':'Mean of two separately aggregated seed mIoUs for the same sampled image IDs; not pooled counts',
        'includes_seed_variance':False},
    'target':{'pct':47.2115,'both_seeds_met':False,'mean_met':False,'mean_remaining_pp':47.2115-summary['mean_test_pct']},
    'cost':{'formal_training_invocation_seconds':training_seconds,'training_plus_val_and_checkpoint_wall_hours':sum(training_seconds.values())/3600,
        'locked_final_evaluation_wall_hours':(timestamp(state['completed_at'])-timestamp(state['started_at'])).total_seconds()/3600,
        'preflight_wall_hours':preflight['total_preflight_seconds']/3600,
        'full_system_inference_peak_memory':'Not separately instrumented; accepted training preflight memory reported below',
        'training_preflight_peak_allocated_bytes':preflight['performance']['peak_allocated_bytes'],
        'training_preflight_peak_reserved_bytes':preflight['performance']['peak_reserved_bytes'],
        'new_Q_parameters':preflight['loading']['total_parameters'],'system_weight_count':7,
        'system_weights':['D','E','B','E4','H4 learned residual head','historical U0','Q seed42'],
        'actual_charged_cny':None,'actual_rate_cny':None,'billing_limitation':'No platform billing receipt or rate available; process wall hours are not the final bill'},
    'backups':{s:{'files':len(x['files']),'bytes':x['total_bytes'],'sha_verified_at':x['verified_at']} for s,x in backups.items()},
    'final_evidence_verified_files':receipt['verified_files'],
    'historical_limitation':summary['historical_limitation'],'original_plan_complete':False,'shutdown_allowed':False}
write_json(ROOT/'reports/Q_DDR_stage_analysis.json',analysis)

val=read('reports/paired_validation_summary.json')
lines=['# Q_DDR 两种子阶段结果（2026-10-04）','',
    f"固定融合的两种子平均 test mIoU 为 **{summary['mean_test_pct']:.6f}%**，相对旧 P0 **+{analysis['two_seed_mean']['delta_vs_p0_pp']:.6f} 点**；未达到 47.2115% 目标，仍差 {analysis['target']['mean_remaining_pp']:.6f} 点。",
    '这是 P0+Q 系统收益；Q 单模test尚未单独测量，不能将系统成绩归因成某个模块提升。原方案的 Q_MIX 外部监督阶段尚缺官方配套原图，整个原方案未完成；临时开停机状态见 execution_scope_override.json 和 preparation_status.json。','',
    '|模型|锁定 val，%|test，%|相对 P0，点|', '|---|---:|---:|---:|',
    f"|P0|43.487703|{p0['miou_pct']:.6f}|—|"]
for s in (42,2024):
    m=per_seed[str(s)]
    lines.append(f"|固定融合 seed{s}@16000|{val['nodes'][str(s)]['16000']['fixed_fusion']['miou_pct']:.6f}|{m['selected']['miou_pct']:.6f}|+{m['delta_vs_p0_pp']:.6f}|")
lines+= [f"|两种子平均|{lock['selection']['selected']['mean_val_pct']:.6f}|{summary['mean_test_pct']:.6f}|+{analysis['two_seed_mean']['delta_vs_p0_pp']:.6f}|",'',
    f"test 范围 [{min(summary['seed_scores_pct']):.6f}, {max(summary['seed_scores_pct']):.6f}]%，两种子差 {analysis['two_seed_mean']['seed_spread_pp']:.6f} 点。预定部署 seed42，未按 test 更换种子或配方。单模16k val：seed42 37.893719%，seed2024 37.295592%；单模 test 未单独推理。",'',
    '冻结选择仅依据两种子共同晚期 val：4个候选（12k/16k × 单模/固定融合），选择16k固定融合。公式为 0.75P0+0.25Q = 0.5625C+0.1875U0+0.25Q；C保留D/E/B/E4的3/2/1/1和学习的H4残差。',
    f"冻结时间 `{lock['created_at']}`；冻结 SHA256 `{receipt['lock_sha256']}`。阈值严格 >0.5，原坐标全图累计四类TP/FP/FN后宏平均，无阈值或TTA扫描。历史P0的227张逐图计数和总计数完全重现。",'',
    '## 四类测试计数与 precision/recall','',
    '|身份|类别|TP|FP|FN|IoU，%|precision，%|recall，%|','|---|---|---:|---:|---:|---:|---:|---:|']
for name,metrics in [('P0',p0)]+[(f'seed{s}',per_seed[str(s)]['selected']) for s in (42,2024)]:
    for c in CLASSES:
        x=metrics['per_class'][c]
        lines.append(f"|{name}|{c}|{x['tp']}|{x['fp']}|{x['fn']}|{x['iou_pct']:.6f}|{100*x['precision']:.4f}|{100*x['recall']:.4f}|")
lines+=['','两种子每类IoU增量（百分点）：']
for c in CLASSES:
    d=[per_seed[str(s)]['selected']['per_class'][c]['iou_pct']-p0['per_class'][c]['iou_pct'] for s in (42,2024)]
    lines.append(f"- {c}：seed42 {d[0]:+.6f}，seed2024 {d[1]:+.6f}，平均 {np.mean(d):+.6f}。")
lines+=['','## 图像抽样不确定性与种子差异','']
for s in (42,2024):
    x=per_seed[str(s)]
    lines.append(f"- seed{s}：绝对test图像bootstrap95%区间 {x['absolute_image_percentile95_pct']}%；相对P0配对差异区间 {x['paired_image_percentile95_pp']} 点。")
lines+= [f"- 两种子平均相对P0的配对图像95%区间 {analysis['two_seed_mean']['delta_paired_image_percentile95_pp']} 点；绝对均值区间 {analysis['two_seed_mean']['absolute_paired_image_percentile95_pct']}%。",
    f"- 每次删去一张测试图后，两种子均值相对P0的增量范围 {analysis['two_seed_mean']['leave_one_image_out_delta_range_pp']} 点。", 
    '- 图像ID配对重采样2000次，随机种子100303；每次重新汇总计数，再分别求两种子mIoU并平均。图像区间不包括训练种子方差，不能把它当作未来重复训练的保证。',
    '- 76张val和227张test都有历史使用，本轮属于探索性延续。Q_MIX后续协议必须披露Q_DDR测试已完成，不能把后续DDR结果称为首次未见测试确认。','',
    '## 全部正式验证节点','',
    '|种子|更新数|单模mIoU，%|固定融合mIoU，%|P0，%|','|---|---:|---:|---:|---:|']
for s in (42,2024):
    for node,x in sorted(val['nodes'][str(s)].items(),key=lambda item:int(item[0])):
        lines.append(f"|{s}|{node}|{x['single']['miou_pct']:.6f}|{x['fixed_fusion']['miou_pct']:.6f}|{x['p0']['miou_pct']:.6f}|")
lines+=['','seed42单模12k→16k仅 +0.052758 点，未触发≥0.3点的共同24k延长。两相邻正式晚期固定融合超过P0+0.5点，按原规则完成seed2024复核；没有追加第三种子或其他模型。每种子实际DDR暴露128000窗口，外部暴露0。','',
    '全部节点四类TP/FP/FN、IoU及precision/recall见 `paired_validation_summary.json`；逐图计数见备份的10份 `runs/Q_DDR_seed*/val_u*/metrics.json`。','',
    '## 原坐标误差分桶','',
    'GT按8连通域分组；面积桶≤16、17–64、65–256、257–1024、>1024像素。边界为每个GT连通域经3×3零边界腐蚀得到的一像素内圈，主体为其余GT像素。面积是像素面积，非病灶体积。以下完全漏检病灶数/漏检像素/边界漏检均来自锁定test逐图记录，FN总和已与指标计数一致。','',
    '|种子|类别|面积桶|病灶数|P0完全漏检|融合完全漏检|P0主体FN|融合主体FN|P0边界FN|融合边界FN|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for s in (42,2024):
    for x in per_seed[str(s)]['lesion_area_error_buckets']:
        b=x['area_upper_bound'];label={16:'≤16',64:'17–64',256:'65–256',1024:'257–1024',None:'>1024'}[b]
        lines.append(f"|{s}|{x['class']}|{label}|{x['lesions']}|{x['p0_whole_missed']}|{x['selected_whole_missed']}|{x['p0_body_fn']}|{x['selected_body_fn']}|{x['p0_boundary_fn']}|{x['selected_boundary_fn']}|")
lines+=['','## 成本、模型身份和回收','',
    f"- 新Q参数 {preflight['loading']['total_parameters']:,}；完整选中系统需7份权重：D、E、B、E4、H4、历史U0、Q seed42。未自动增加跨种子集成。",
    f"- 真实训练预检峰值 allocated {preflight['performance']['peak_allocated_bytes']/1024**3:.4f} GiB，reserved {preflight['performance']['peak_reserved_bytes']/1024**3:.4f} GiB；完整P0+Q推理峰值未另行采集，不以此冒充系统峰值。",
    f"- 真实末四层反传预检平均 {preflight['performance']['unfrozen_seconds_per_update_mean']:.6f} 秒/更新，数据等待 {preflight['performance']['data_wait_seconds_mean']:.6f} 秒；GPU利用率不替代吞吐。保持BF16、micro2×accum4。",
    f"- 两条正式训练调用墙钟合计 {analysis['cost']['training_plus_val_and_checkpoint_wall_hours']:.6f} 小时（含val与checkpoint），最终P0+两种子评估墙钟 {analysis['cost']['locked_final_evaluation_wall_hours']:.6f} 小时，预检 {analysis['cost']['preflight_wall_hours']:.6f} 小时。这些是任务耗时，不是当前平台账单；费率和实际收费尚无回执。"]
for s in (42,2024):
    t=per_seed[str(s)]['q_image_inference_seconds'];b=analysis['backups'][str(s)]
    lines.append(f"- seed{s} Q每图推理/拼接平均 {t['mean']:.6f} 秒，中位 {t['median']:.6f} 秒，P90 {t['p90']:.6f} 秒；不含P0、SHA或误差诊断。整条run {b['files']}文件/{b['bytes']}字节已本机SHA核验，回执时间 {b['sha_verified_at']}。")
lines+= [f"- 最终证据另回收 {receipt['verified_files']} 文件并逐一SHA核验，见 `final_evidence_backup_receipt.json`；完整test227的两个种子逐图计数见 `server_results/reports/final_test/seed42.json` 与 `seed2024.json`。",'',
    'Q_MIX尚未训练，不能给出外部监督收益。后续依原方案补齐官方IDRiD原图/像素标签、MAPLES对应MESSIDOR-1原图，完成去重、ignore与20张叠图验收后再启动。相关目录和有用检查点继续保留；按最新“今晚不要空转”指令，数据/许可阻塞时允许保存后临时停机，原方案待办继续保留，不宣称已全部完成。']
(ROOT/'reports/Q_DDR_两种子阶段结果_20261004.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
write_json(ROOT/'reports/Q_DDR_stage_report_receipt.json',{'created_at':utc_now(),
    'report_sha256':sha(ROOT/'reports/Q_DDR_两种子阶段结果_20261004.md'),
    'analysis_sha256':sha(ROOT/'reports/Q_DDR_stage_analysis.json'),'locked_selection_sha256':receipt['lock_sha256'],
    'source_final_evidence_receipt_sha256':sha(ROOT/'reports/final_evidence_backup_receipt.json'),
    'new_inference':False,'new_selection':False,'raw_test_labels_read':False})
print({'report_saved':True,'test_mean_pct':summary['mean_test_pct'],
    'delta_pp':analysis['two_seed_mean']['delta_vs_p0_pp'],
    'mean_image_delta95_pp':analysis['two_seed_mean']['delta_paired_image_percentile95_pp']})
