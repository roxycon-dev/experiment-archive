"""Local historical checkpoint comparisons; no GPU work, hashes or training changes."""
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ARCHIVE=Path('C:/Users/GuanXingRen/AppData/experment/fundus-seg-2026/hrdecoder-experiments')
CURRENT=ROOT/'cache_execution/server_receipts/training_runs/v3/val_04000.json'
current=json.loads(CURRENT.read_text(encoding='utf-8'))
specs=[
    ('原HRDecoder baseline',ARCHIVE/'week1/logs/01_板块一_baseline.json'),
    ('E4 detail',ARCHIVE/'week2/logs/e4_detail_20260817_034605.log.json'),
    ('E4v2',ARCHIVE/'week2/logs/e4v2_20260818_051249.log.json'),
    ('CE+Dice',ARCHIVE/'week2/logs/e1_cedice_20260817_105140.log.json'),
    ('E2+E3',ARCHIVE/'week2/logs/e2e3_20260817_220954.log.json'),
    ('combo',ARCHIVE/'week2/logs/combo_20260817_204954.log.json'),
    ('combo_v2',ARCHIVE/'week2/logs/combo_v2_20260818_040002.log.json'),
    ('combo2',ARCHIVE/'week2/logs/combo2_20260818_082631.log.json')]

def parse_log(path):
    config=''; seed=None; previous=0; vals=[]
    for number,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip():continue
        row=json.loads(line)
        if 'config' in row: config=row['config']; seed=row.get('seed',seed)
        if row.get('mode')=='train': previous=row['iter']
        if row.get('mode')=='val':
            assert row['iter']==76
            vals.append(dict(step=previous,line=number,**{k:row[k]*100 for k in ('mIoU','mF1','mPPV','mS') if k in row}))
    return {'path':str(path),'seed':seed,'validation':vals,
        'initialization_config':[line.strip() for line in config.splitlines() if re.match(r'\s*(load_from|resume_from|pretrained|seed)\s*=',line)],
        'config_facts':{
        'DDR_split_613':'DDR_split_613' in config,'square_validation':'keep_ratio=False' in config,
        'sigmoid':'use_sigmoid=True' in config,'samples_per_gpu_2':'samples_per_gpu=2' in config,
        'SGD':"type='SGD'" in config,'40k_schedule':'max_iters=40000' in config,
        'fp16_in_config':'fp16 =' in config or 'fp16=' in config}}

early=[]; histories=[]
for name,path in specs:
    if not path.exists():continue
    history=parse_log(path); history['name']=name; histories.append(history)
    val=next(v for v in history['validation'] if v['step']==4000)
    early.append({'name':name,'historical_4k_mIoU':val['mIoU'],
        'v3_minus_historical_pp':current['mIoU_percent']-val['mIoU'],
        'source':str(path),'source_line':val['line'],'protocol_facts':history['config_facts']})
baseline=histories[0]
baseline4k=next(v for v in baseline['validation'] if v['step']==4000)
baseline_best=max(baseline['validation'],key=lambda v:v['mIoU'])
baseline_end=next(v for v in baseline['validation'] if v['step']==40000)
macro_precision=sum(current['precision_percent'])/4
macro_recall=sum(current['recall_percent'])/4
modern_source=Path('D:/dataset/experiment/HRDecoder_Q_20261003/reports/paired_validation_all_nodes_24k.json')
modern=json.loads(modern_source.read_text(encoding='utf-8'))
different=[]
gt=[tp+fn for tp,fp,fn in current['counts']]
for name,nodes in modern['nodes'].items():
    record=nodes.get('4000')
    if not record: continue
    single=record['single']
    assert [tp+fn for tp,fp,fn in single['counts']]==gt
    different.append({'name':name,'step':4000,'mIoU':single['miou_pct'],
        'v3_minus_historical_pp':current['mIoU_percent']-single['miou_pct'],
        'per_class_IoU':[single['per_class'][c]['iou_pct'] for c in ('EX','HE','SE','MA')],
        'GT_positive_counts_match':True,'source':str(modern_source),
        'comparability':'Same DDR val labels/counts, native sliding-window DINOv3 pretrained model, batch8; context only, not paired v3 comparison'})
for name in ('C_seed42','S_seed42','D_seed42','SD_seed42'):
    path=Path('D:/dataset/experiment/HRDecoder_FinalDecision_20261005/execution/server_results/runs')/name/'val_u4000/metrics.json'
    if not path.exists(): continue
    record=json.loads(path.read_text(encoding='utf-8')); single=record['single']
    assert record['n_images']==76 and [tp+fn for tp,fp,fn in single['counts']]==gt
    different.append({'name':name,'step':4000,'mIoU':single['miou_pct'],
        'v3_minus_historical_pp':current['mIoU_percent']-single['miou_pct'],
        'per_class_IoU':[single['per_class'][c]['iou_pct'] for c in ('EX','HE','SE','MA')],
        'GT_positive_counts_match':True,'source':str(path),
        'comparability':'Continuation from Q/F12 trained parent; 4k additional updates, not from ImageNet; native sliding windows/batch8'})
data=json.loads((ROOT/'cache_execution/server_receipts/metadata/data_identity.json').read_text(encoding='utf-8'))
assignment=json.loads(Path('D:/dataset/experiment/autodl100/bundle/split_assignment.json').read_text(encoding='utf-8'))
legacy_val={Path(name).stem for name,row in assignment.items() if row['split']=='val'}
current_val={row['sid'] for row in data['splits']['val']['samples']}
assert legacy_val==current_val and len(current_val)==76
legacy_gt=[sum(row['class_pixels'][c] for row in assignment.values() if row['split']=='val') for c in ('EX','HE','SE','MA')]
print(json.dumps({'historical_assignment_GT':legacy_gt,'actual_v3_GT':gt,'val_IDs_equal':legacy_val==current_val}))
manifest=json.loads(Path('D:/dataset/experiment/autodl100/bundle/data_manifest.json').read_text(encoding='utf-8'))
saved_identities={entry['path']:entry['sha256'] for entry in manifest}
label_matches=sum(saved_identities['data/DDR_split_613/labels/val/'+row['sid']+'.png']==row['label_sha256'] for row in data['splits']['val']['samples'])
assert label_matches==76
result={'current_step':4000,'current_mIoU':current['mIoU_percent'],'early_same_stage':early,
    'baseline_history':baseline['validation'],'historical_source_metadata':histories,
    'baseline_best':baseline_best,'baseline_fixed40k':baseline_end,
    'current_macro_precision':macro_precision,'current_macro_recall':macro_recall,
    'baseline_macro_precision_4k':baseline4k.get('mPPV'),'baseline_macro_recall_4k':baseline4k.get('mS'),
    'different_protocol_context':different,'current_GT_positive_counts':gt,
    'current_data_identity_top_level_keys':list(data),'historical_val_IDs_match':True,
    'historical_assignment_GT_positive_counts':legacy_gt,'historical_assignment_GT_matches':legacy_gt==gt,
    'historical_saved_label_identities_match':label_matches,'file_hashes_recomputed':False,
    'limitations':['Historical metrics are rounded training logs, not a new paired baseline run',
        'v3 BF16, KD, architecture, valid masks and exposure RNG differ; early gain is not causal or endpoint proof',
        'Do not compare validation score with test score or continuation step with total from-start step']}
(HERE/'historical_checkpoint_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text=['# v3 4k与历史检查点比较','',
    f"当前v3：4k，全76张val，mIoU {current['mIoU_percent']:.4f}%。以下为读取已保存结果，未新跑模型或使用GPU。",'',
    '## 同一训练阶段的历史HRDecoder','',
    '历史原始JSON日志中val的iter=76是图像计数，实际训练步用前一条train记录恢复。各项差值为当前v3减历史，单位为百分点。','',
    '| 历史模型 | 历史4k mIoU% | v3变化（百分点） | 原始证据 |',
    '|---|---:|---:|---|']
for row in early:
    text.append(f"| {row['name']} | {row['historical_4k_mIoU']:.3f} | {row['v3_minus_historical_pp']:+.3f} | [{Path(row['source']).name}]({Path(row['source']).as_posix()}:{row['source_line']}) |")
text+=['',f"v3相对原baseline的4k提高{current['mIoU_percent']-baseline4k['mIoU']:.2f}个百分点，但未超过CE+Dice和combo等所有历史4k点。这是早期信号，不是最终架构获胜结论。",'',
    f"原baseline 4k宏平均precision/recall为{baseline4k['mPPV']:.2f}%/{baseline4k['mS']:.2f}%；当前为{macro_precision:.2f}%/{macro_recall:.2f}%。精确率明显更高、平均召回率更低，尚无原baseline4k逐类计数来证明各类均提高。",'',
    '## 原baseline后续门槛','',
    '| 步骤 | 原baseline val mIoU% |','|---:|---:|']
for row in baseline['validation']:text.append(f"| {row['step']} | {row['mIoU']:.3f} |")
text+=['',f"原baseline固定40k端点为{baseline_end['mIoU']:.3f}%，历史best-val为{baseline_best['mIoU']:.3f}%@{baseline_best['step']}。本轮还只有4k验证，不能判断40k最终是否进步。",'',
    '## 其他架构与续训结果：仅作背景','',
    '下面模型与本轮的76张val四类GT正像素总计一致；原图推理、初始化、batch与训练预算不同，4k续训并非从头4k。','','| 历史模型 | 4k val mIoU% | 口径 |','|---|---:|---|']
for row in different:text.append(f"| {row['name']} | {row['mIoU']:.3f} | {row['comparability']} |")
text+=['','## 可比性核对','',
    f'已用保存的split_assignment与当前data_identity核对：76张val的图像ID集合完全相同；历史data_manifest与当前data_identity记录的76个标签文件身份全部一致。未重新计算文件哈希或读取图像大文件。旧划分附带class_pixels总计{legacy_gt}，本轮实际GT计数为{gt}；这份附带class_pixels有小差异，不能替代实际评估GT计数。',
    '原HRDecoder配置与代码记录2048×2048整图缩放、keep_ratio=False、相同均值/标准差、无翻转TTA、sigmoid严格>0.5；按整套验证图累计统计后平均四病灶IoU。当前按相同阈值和四病灶计数汇总。原baseline日志显示seed42、batch2、SGD和40k。',
    '这仍是历史检查点参照：没有重跑当前B0配对对照，v3新增模块、蒸馏、BF16、有效域与每曝光随机流不同；不能把+7.93百分点全归因于某个机制。不同架构与预先训练过的续训节点不作同预算结论。',
    '历史test42.20%、最终教师test45.86%等测试结果不与本轮val29.49%混算。']
(HERE/'历史检查点对比.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('historical_source_metadata','current_data_identity_top_level_keys')},ensure_ascii=False))
print(json.dumps({h['name']:{'seed':h['seed'],'initialization_config':h['initialization_config']} for h in histories},ensure_ascii=False))
