"""Record the primary agent's completed report/figure review; not a ready flag."""
from pathlib import Path
import json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mirror_remote import ROOT,LOCAL,file_sha,write_json
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
test=read(LOCAL/'work/final_test.json');progress=read(LOCAL/'work/test_progress.json')
assert test['n_images']==len(progress['images'])==227
assert test['target']==.472115 and test['target_met'] is False
assert test['lock_sha256']==progress['lock_sha256']==file_sha(LOCAL/'work/final_lock.json')
for mode,per_image in [('old_complete_pipeline','old_complete'),('selected_complete_pipeline','selected_complete')]:
 counts=[[sum(row[per_image][c][k] for row in progress['images'].values()) for k in range(3)] for c in range(4)]
 assert counts==test[mode]['counts']
 value=sum(tp/(tp+fp+fn) for tp,fp,fn in counts)/4
 assert abs(value-test[mode]['mIoU_lesion4'])<1e-12
assert abs(test['delta']-(test['selected_complete_pipeline']['mIoU_lesion4']-test['old_complete_pipeline']['mIoU_lesion4']))<1e-12
completion=read(LOCAL/'reports/campaign_completion.json')
for rel,h in completion['required_files'].items():assert file_sha(LOCAL/rel)==h,rel
assert read(ROOT/'reports/M_analysis_summary.json')['status']=='REVIEWED_PAIRED_COMPLETE'
assert read(ROOT/'reports/final_analysis_summary.json')['review_status']=='REVIEWED'
report=ROOT/'reports/HRDecoder_严格实验最终报告_20261003.md'
text=report.read_text(encoding='utf-8');assert '44.7809%' in text and '44.6345%' in text and '2.4306' in text
assert '等待主代理审查' not in text
original=ROOT.parent/'HRDecoder_Ready_20261002/best_pipeline/original_deployment_lock.json'
old_sha='644974f4a888835f838472199dcb623e879be2d62e73dab502394759551ceac4'
assert file_sha(original)==old_sha
package=read(ROOT/'reports/reproducibility_package.json');assert package['status']=='PASS'
assert file_sha(package['path'])==package['sha256']
approval=dict(status='PRIMARY_REVIEW_COMPLETE',time=time.time(),report=str(report.resolve()),
 report_sha256=file_sha(report),final_test_sha256=file_sha(LOCAL/'work/final_test.json'),
 metadata_manifest_sha256=file_sha(ROOT/'reports/closed_node_metadata_manifest.json'),
 package_sha256=package['sha256'],model_source_sha256=package['source_sha256'],
 original_deployment_lock=dict(path=str(original.resolve()),sha256=old_sha),
 checked=['Complete paired 0/1k/2k outputs','Cost-stop policy and preserved exact last/best',
 'G 24 and M 12 inspected error crops','M curve figure layout','Pooled 227-image counts and fixed test identity',
 'Image-bootstrap limits and absence of seed validation','Retain original deployed winner','Persistent nonwinner storage and local critical-backup gate'],
 waiting_only_for='Required stable probability backup and final server idle/shutdown verification',
 completion_ready_written=False)
write_json(ROOT/'reports/primary_review_approval.json',approval)
print(json.dumps(dict(status=approval['status'],ready=False)))
