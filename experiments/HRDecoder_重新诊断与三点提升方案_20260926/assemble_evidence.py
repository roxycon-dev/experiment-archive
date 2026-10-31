"""Read local records and assemble provenance; no training or remote access."""
from pathlib import Path
import json, re, os, tarfile, hashlib, collections

root=Path(__file__).resolve().parent
exp=root.parent
def dump(name,obj):
    (root/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

raw=(root/'delegation/history_response.md').read_text(encoding='utf-8')
blocks=re.findall(r'```json\s*([\s\S]*?)```',raw)
hist=json.loads(blocks[-1])
dump('history_ledger_workbuddy_unreviewed.json',hist)
raw2=(root/'delegation/complement_response.md').read_text(encoding='utf-8')
comp=json.loads(re.findall(r'```json\s*([\s\S]*?)```',raw2)[-1])
dump('complement_numbers_workbuddy_unreviewed.json',comp)
print('history keys',list(hist))

assets=[exp/'hrdecoder_plus3/results/D_0913_retry/backup_student_2000.pth',
exp/'hrdecoder_plus3/repo/weights/b_weights.pth',
exp/'HRDecoder_FusionAware_20260917/assets/E_1500.pth',
exp/'autodl100/results/e4_reference_weights.pth',
exp/'HRDecoder_审查轮_20260916/audit_runs/audit_20260916_A/h4_best.pth',
exp/'HRDecoder_FullCompetition_20260920/runs/ckpt_best.pth']
dump('asset_presence.json',[{'path':str(p),'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else None,'hash_checked_this_audit':False} for p in assets])

counts=collections.Counter(); exts=collections.Counter(); total=0
with (root/'local_file_catalog.jsonl').open('w',encoding='utf-8') as f:
    for base,dirs,files in os.walk(exp):
        dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','node_modules','_removed-armor',root.name)]
        for name in files:
            p=Path(base)/name
            try: rel=p.relative_to(exp);st=p.stat()
            except OSError: continue
            counts[rel.parts[0]]+=1;exts[p.suffix.lower()]+=1;total+=1
            f.write(json.dumps({'path':str(rel),'bytes':st.st_size},ensure_ascii=False)+'\n')
dump('coverage.json',{'date':'2026-09-26','root':str(exp),'files_catalogued':total,'top_level_file_counts':dict(sorted(counts.items())),
'extensions':dict(exts),'excluded':['.git','__pycache__','node_modules','_removed-armor',root.name],
'scope_note':'目录级全量索引不等于每个文件逐字阅读。重点读取报告、原始指标、配置、加载日志及异常证据；图片和权重不逐个推理。此索引脚本不联系服务器；本次整体审查曾尝试连接3090但不可达，未访问4090。'})

tarpath=exp/'HRDecoder_FullCompetition_20260920/runs/H0_stopped6000_metadata.tar.gz'
with tarfile.open(tarpath) as tar:
    meta={}
    for m in tar.getmembers():
        if m.isfile() and m.size<500000 and m.name.endswith(('.log','.json','.csv','.txt')):
            meta[m.name]=tar.extractfile(m).read().decode('utf-8',errors='replace')
dump('H0_metadata_readback.json',meta)
print('H0 archive',[(m,len(v)) for m,v in meta.items()])
print('files catalogued',total)

old=Path(r'C:\Users\GuanXingRen\AppData\experment\fundus-seg-2026\hrdecoder-experiments\week2\logs\e4_detail_20260817_034605.log.json')
with old.open(encoding='utf-8') as f:header=json.loads(f.readline())
cfg=header['config']
dump('E4_initialization_evidence.json',{'source':str(old),'seed':header['seed'],
'initialization_lines':[line for line in cfg.splitlines() if any(s in line for s in ('pretrained=','load_from','resume_from'))],
'interpretation':'运行日志配置为 ImageNet HRNet 初始化、load_from=None、resume_from=None。不能凭交接文档中存在官方 DDR 权重就认定实验使用了它。'})

audit=exp/'HRDecoder_FullCompetition_20260920/analysis_results/expert_complementarity_20260920/analysis_outputs/expert_complementarity.json'
d=json.loads(audit.read_text(encoding='utf-8'))
dump('expert_key_evidence.json',{k:v for k,v in d.items() if k in ('all_models_per_class_iou','direct_four_class_combination','area_bucket_recall','gt_global_positive_pixels')})
print('area bucket',json.dumps(d.get('area_bucket_recall'),ensure_ascii=False)[:3500])
