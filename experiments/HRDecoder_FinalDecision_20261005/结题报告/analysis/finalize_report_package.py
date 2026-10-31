"""Offline delivery QA and evidence map. No SSH, model loading or new experiments.
Run --visual-review-complete only after inspecting every final rendered page.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, re, sys
from docx import Document
from pypdf import PdfReader

OUT=Path(__file__).resolve().parents[1]
EX=OUT.parent/'execution'; ROOT=OUT.parent.parent; E=OUT/'evidence'; SR=EX/'server_results'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=datetime.now(timezone.utc).isoformat()
t=read(E/'final_locked_test_evidence.json');v=read(E/'four_arm_saved_validation.json')
source_index=read(E/'workspace_source_hashes.json'); sources=source_index['sources']
assert not source_index['unresolved_exact_paths']
for z in sources:
    p=Path(z['path']);assert p.is_file() and p.stat().st_size==z['bytes'] and sha(p)==z['sha256'],z['path']
for z in v['records']:
    assert sha(z['source_path'])==z['source_sha256'] and z['scores_recomputed_from_saved_per_image_counts']
assert len(v['records'])==34
test_count=sum(100*z['tp']/(z['tp']+z['fp']+z['fn']) for z in t['test_per_class'].values())/4
assert abs(test_count-t['locked_test_miou_pct'])<1e-12
assert abs(t['gain_vs_R_pp']-(test_count-t['R_test_miou_pct']))<1e-12
assert abs(t['approx_gain_vs_baseline_pp']-(test_count-42.20))<1e-12
audit=read(E/'completion_audit_status.json'); assert audit['stage']=='PASS'
assert audit['native_data_identities_verified']==757
lock=read(SR/'reports/final_selection_lock.json')
assert lock['lock_sha256']==t['locked_recipe']['lock_sha256']
assert lock['checkpoints']['42']['sha256']==t['locked_recipe']['checkpoints']['42']['sha256']
release=read(EX/'reports/publication_report_server_release.json')
assert release['stage']=='SERVER_NONTRAINING_EVIDENCE_WORK_COMPLETE'
assert release['all_outputs_recovered_with_SHA'] and release['no_new_training_or_selection_or_threshold_scan']
closure=read(EX/'reports/server_shutdowns_verified_dispatch_20261006.json')
assert closure['primary_shutdown']['stage']==closure['second_shutdown']['stage']=='SHUTDOWN_DISPATCHED'
assert closure['primary_shutdown']['backup_SHA_verified'] and closure['second_shutdown']['backup_SHA_verified']
backups=[read(EX/'reports/final_backup_receipt.json'),read(EX/'parallel/node2/reports/final_backup_receipt.json')]
assert all(z['stage']=='FULL_BACKUP_VERIFIED' for z in backups)

word=OUT/'HRDecoder项目结题报告.docx';pdf=OUT/'HRDecoder项目结题报告.pdf';md=OUT/'HRDecoder项目结题报告.md'
doc=Document(word); reader=PdfReader(pdf)
assert len(doc.tables)==11 and len(doc.inline_shapes)==11
captions=[p.text for p in doc.paragraphs if p.style.name=='Caption' and re.match(r'^图\d+',p.text)]
assert [int(re.match(r'^图(\d+)',z)[1]) for z in captions]==list(range(1,12))
text='\n'.join(p.extract_text() or '' for p in reader.pages)
clean=re.sub(r'\s+|\u200b','',text)
assert '\ufffd' not in text
for phrase in ['45.8604','42.20','3.6604','0.1388','45.3218','47.2115','29.2435','0.2626','0.6519']:
    assert phrase in clean,phrase
for i,page in enumerate(reader.pages,1):
    txt=re.sub(r'\s+','',page.extract_text() or '')
    assert f'第{i}页' in txt,(i,'footer')
    assert len(txt)>80,(i,'unexpected blank page')
assert reader.outline,'PDF heading navigation missing'
pages=sorted((OUT/'qa/final').glob('page-*.png'))
assert len(pages)==len(reader.pages)==29
assert '--visual-review-complete' in sys.argv,'Human/agent visual inspection acknowledgment is required'
reviewed_page_hashes=[{'page':i,'path':str(p),'sha256':sha(p)} for i,p in enumerate(pages,1)]
table_sources={
 'T1':[SR/'reports/data_identity.json',EX/'reports/TJDR_manifest.json',EX/'reports/TJDR_visual_review.json',EX/'reports/TJDR_duplicate_pairs.json',ROOT/'HRDecoder_Q_20261003/reports/original_plan_experiment_result.json'],
 'T2':[OUT.parent/'01_执行协议.json',OUT.parent/'02_空间交互与随机流接口.md',SR/'spatial.py',SR/'final_model.py',SR/'engine.py',SR/'train_run.py',SR/'inputs.py',EX/'reports/stream_identity.json'],
 'T3':[Path(z['path']) for z in sources if any(k in z['path'] for k in ['全实验档案','autodl100','hrdecoder_plus3','hrdecoder_fcn','baseline','h7_test','最终报告_两轮','original_plan_experiment','最终实验报告'])],
 'T4':[Path(z['source_path']) for z in v['records'] if z['selection_node']],
 'T5':[Path(z['source_path']) for z in v['records'] if z['selection_node']],
 'T6':[Path(z['path']) for z in t['sources']]+[Path(z['path']) for z in read(E/'original_HRDecoder_baseline_evidence.json')['source_files']],
 'T7':[Path(z['path']) for z in t['sources']]+[Path(z['path']) for z in read(E/'original_HRDecoder_baseline_evidence.json')['source_files']],
 'T8':[SR/'runs/LOCKED_TEST_seed42/metrics.json'],
 'T9':[EX/'reports/publication_server_evidence_plan.json',SR/'reports/publication_server_evidence/derivation.json'],
 'T10':[SR/'reports/gpu_readiness.json'],
 'T11':[Path(z['path']) for z in t['sources']]+[Path(z['source_path']) for z in v['records'] if z['selection_node']]
}
by_path={str(Path(z['path']).resolve()):z for z in sources}
maps=[]
for table,paths in table_sources.items():
    entries=[]
    for p in dict.fromkeys(paths):
        entry=by_path.get(str(p.resolve()));assert entry is not None,str(p)
        entries.append(entry)
    maps.append({'table':table,'report_table_json_pointer':f'/'+table,
                 'source_csv':f'evidence/{table}_source.csv','original_sources':entries})
save(E/'table_claim_source_map.json',{'generated_at':now,'tables':maps,
 'identity_distinction':{'recipe_content_hash':lock['lock_sha256'],'selection_lock_file_sha256':sha(SR/'reports/final_selection_lock.json'),
 'selected_checkpoint_sha256':lock['checkpoints']['42']['sha256'],'not_equal_hash_scopes_are_not_a_corruption':True}})
ledger=read(E/'historical_experiment_ledger.json')['records']
ledger_missing=[{'record':z['id'],'path':p} for z in ledger for p,exists in z.get('referenced_source_exists',{}).items() if not exists]
save(E/'final_report_qa.json',{
 'stage':'PASS_RENDERED_AND_VISUALLY_VERIFIED','time':now,'pdf_pages':len(reader.pages),'word_tables':11,'word_figures':11,
 'visual_review':'All 29 final Poppler-rendered pages inspected at original resolution; caption order, glyphs, table continuity, figures, margins and references checked',
 'rendered_page_hashes':reviewed_page_hashes,'figure_caption_numbers':list(range(1,12)),
 'primary_sources_SHA_rechecked':len(sources),'historical_ledger_records':len(ledger),'historical_snapshot_unresolved_references':ledger_missing,
 'native_data_identities_prior_exact_audit':757,'validation_records_exactly_audited':34,'locked_test_counts_recomputed':True,
 'document_metrics_match_saved_results':True,'no_new_training_inference_selection_or_scan':True,'servers_contacted':False,
 'word_sha256':sha(word),'pdf_sha256':sha(pdf),'markdown_sha256':sha(md)})

checklist=f'''# 内部证据核验清单

核验日期：2026年10月6日。报告定稿为29页、11图、11表。此清单用于复核，不将缺失证据补写为结果。

已核实：

- 最终配方是 S@20k EMA 与冻结P0的固定25%概率融合；由锁定文件确认，未重选模型、节点、阈值或融合权重。
- 227张图的逐图TP/FP/FN累加后，测试mIoU={test_count:.12f}%；验证45.321831227388%。摘要、正文、表格、结论一致。
- 原HRDecoder测试42.20%来自原始项目报告，保留两位小数；累计约+{t['approx_gain_vs_baseline_pp']:.12f}个百分点，约+{t['approx_relative_gain_vs_baseline_pct']:.12f}%相对增幅。没有虚构基线整数。
- 上一系统R测试{t['R_test_miou_pct']:.12f}%，本轮增量{t['gain_vs_R_pp']:.12f}个百分点；保存的条件bootstrap区间含零，未声称统计显著。
- 四臂32个正式节点及F12/R参考共34份保存记录复算通过；20k/32k全部24个允许配方均来自真实端点，16k诊断高值未冒充部署结果。
- 757个现有DDR图像/标注身份及划分核验通过；TJDR448对/439保留/9排除/23实际叠加检查有独立记录。TEST隔离，未用于报告再推理。
- S父链来自Q_DDR与F12，非Q_MIX、TJDR、IDRiD或MAPLES像素监督；C_old/U0是推理伙伴，不是S初始化参数。
- D_old保留冻结B参考；H4是MA非线性后融合头；历史名称与当前C/S/D/SD分开。匹配差值仅解释相同节点和公式。
- {len(ledger)}条台账保留早期45条原始归档快照与后续实际记录。{len(sources)}份主要文档/配置/指标SHA复核，图表与原始证据映射见table_claim_source_map.json。
- DOCX由隔离隐藏Word实例原生导出PDF，29页逐页渲染并查看；11张图具PNG/SVG/PDF，表格具有CSV源数据。最终文档与渲染页SHA见final_report_qa.json。
- 两节点FULL_BACKUP_VERIFIED与SHUTDOWN_DISPATCHED收据均存在；关机仅发出一次。报告取证已完成且SHA恢复，报告工作在本地进行。

仍待资料或运营确认：

- 两台平台实际OFF尚待已登录平台状态或用户直接确认；SSH断开不是证明。此项不改变实验计数或报告成绩。
- 原基线当年的完整字节身份快照与原始测试逐图整数未在现有资料中获得，不能把现在的数据核验反推为当年完整证明。
- 患者标识及跨中心独立性没有被当前图像身份核验替代；临床敏感性、跨中心泛化和端到端系统在线延迟没有在本项目中测得。
- 历史台账若只留下摘要或有限字段，明确标注未提取/见源记录，不从计划、命名或未完成状态推断已训练模型及成绩。
- 条件bootstrap基于固定模型和既有图像，不重新执行候选选择；当前项目多阶段已使用此测试集，不称为全新独立留出。

配方内容身份：{lock['lock_sha256']}。
锁定文件字节SHA：{sha(SR/'reports/final_selection_lock.json')}。二者哈希范围不同。
最终S20k检查点SHA：{lock['checkpoints']['42']['sha256']}。
本成果包未复制大体积训练权重；权重及可恢复状态沿用execution/server_results规范存储及已验证保管收据。
'''
(OUT/'内部证据核验清单.md').write_text(checklist,'utf8')
index='\n'.join(f'| {x["table"]} | [{x["source_csv"]}]({x["source_csv"]}) | '+', '.join(z['id'] for z in x['original_sources'])+' |' for x in maps)
(E/'正文表格证据索引.md').write_text('# 正文表格与证据索引\n\n每个E编号的绝对路径、字节数与SHA见workspace_source_hashes.json。完整映射见table_claim_source_map.json。\n\n| 表 | 源数据 | 原始证据编号 |\n|---|---|---|\n'+index+'\n','utf8')
figure_index=[{'report_figure':i,'artifact_stem':m[1],'caption':m[2]} for i,m in enumerate(re.finditer(r'\[\[FIG:([^|]+)\|([^\]]+)\]\]',(OUT/'analysis/report_source.md').read_text('utf8')),1)]
save(E/'figure_artifact_index.json',figure_index)
for z in figure_index:
    for ext in ['png','svg','pdf']:assert (OUT/'figures'/(z['artifact_stem']+'.'+ext)).is_file()
readme=f'''# HRDecoder 项目结题报告成果包

主报告已完成逐页检查：29页，正文约1.66万中文字符（含表与图注），11图、11表。成果与证据保持对应。

- [可编辑Word报告](HRDecoder项目结题报告.docx)
- [阅读与打印PDF](HRDecoder项目结题报告.pdf)
- [Markdown全文](HRDecoder项目结题报告.md)
- [内部证据核验清单](内部证据核验清单.md)
- [正文表格证据索引](evidence/正文表格证据索引.md)
- [79条历史成果台账CSV](evidence/historical_experiment_ledger.csv)及[JSON](evidence/historical_experiment_ledger.json)

最佳成果：S20k EMA与P0按固定25%融合，验证45.3218%，锁定测试45.8604%。相对同新DDR划分原HRDecoder报告42.20%，累计约+3.6604个百分点；相对上一系统R本轮+0.1388个百分点。报告完整区分参数父链、历史预测组成、匹配消融及各阶段未采纳的结果。

`figures`含全部图的300dpi PNG、SVG、PDF；`evidence`含图表源数据、{len(sources)}份主要来源SHA、逐图计数派生数据及精确证据映射；`analysis`保留仅离线使用的收集、复算、制图和文档生成脚本。`qa/final`是与当前PDF对应的29页检查图。文件名F编号沿用原分析图身份，正式图1–11与文件名映射见figure_artifact_index.json。

复算已有数据：使用本机Conda Python执行analysis/audit_completion_evidence.py；制图执行analysis/build_scientific_figures.py。文档生成使用bundled Python执行analysis/build_completion_artifacts.py；已有Word导出使用analysis/export_with_word.ps1。生成脚本读取原工作区实际证据路径，需要保留工作区目录结构。不得把离线复算当成新训练或新测试。

最终QA与文件身份见evidence/final_report_qa.json及report_delivery_manifest.json。大体积模型、原始/EMA/优化器/RNG状态继续保存在原规范server_results与保管链，不为报告包另复制一套。

服务器端训练、锁定测试、报告必要取证及两节点全备份均已完成，关机请求已发出；平台实际OFF单独等待确认。报告写作已全部在本地完成。
'''
(OUT/'README.md').write_text(readme,'utf8')
save(OUT/'report_work_state.json',{'stage':'REPORT_RENDERED_AND_VERIFIED_LOCAL_DELIVERY_READY','time':now,
 'server_work_complete':True,'both_full_backup_verified':True,'both_shutdown_dispatched':True,
 'actual_BOTH_OFF_confirmation_pending':not closure['platform_OFF_confirmed'],
 'pdf_pages':29,'figures':11,'tables':11,'ledger_records':len(ledger),'primary_sources':len(sources),
 'final_result':test_count,'no_new_inference_training_or_reselection':True})
files=[]
for p in sorted(OUT.rglob('*')):
    if not p.is_file() or 'qa' in p.relative_to(OUT).parts or p.name=='report_delivery_manifest.json' or '__pycache__' in p.parts:continue
    files.append({'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
save(OUT/'report_delivery_manifest.json',{'stage':'VERIFIED_DELIVERY_PACKAGE','time':now,'files':files,
 'total_bytes_excluding_rendered_QA':sum(z['bytes'] for z in files),'pdf_pages':29,'figure_count':11,'table_count':11,
 'servers_full_backup_verified':True,'shutdown_dispatched_both':True,'platform_OFF_confirmation_pending':not closure['platform_OFF_confirmed'],
 'no_new_inference_training_or_selection':True})
print(f'PASS: 29 rendered pages inspected; 11 figures/11 tables; {len(ledger)} records;{len(sources)} source hashes; exact saved metrics; delivery manifest complete.')
