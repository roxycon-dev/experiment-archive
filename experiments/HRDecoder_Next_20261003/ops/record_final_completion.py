"""Recover the exact official acceptance receipt carried by SSH interruption."""
from pathlib import Path
import datetime,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mirror_remote import ROOT,write_json,file_sha
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
r=ROOT/'reports';backup=read(r/'backup_verification.json');ready=read(r/'completion_ready.json')
assert backup['status']=='PASS' and ready['status']=='REVIEWED_COMPLETE'
assert backup['probability_files_locally_sha_verified']==836
assert backup['closed_metadata_files_verified']==110
assert backup['persistent_negative_files_sha_verified']==228
p=r/'shutdown_state.json';old=read(p)
if old['status']=='shutdown_request_accepted':
 receipt=old
else:
 assert old['status']=='shutdown_dispatched_unconfirmed'
 reason=old['reason'];begin=reason.index('{')
 receipt,end=json.JSONDecoder().raw_decode(reason[begin:])
 assert receipt['status']=='shutdown_request_accepted' and receipt['returncode']==0
 assert reason[begin+end:].strip()==''
 write_json(r/'shutdown_transport_interruption.json',old)
 receipt=dict(receipt,receipt_evidence='Exact remote JSON output preserved in the SSH interruption exception',
  transport_closed_after_receipt=True,platform_off_confirmed=False,
  platform_state_note='Official command accepted the request; no authenticated AutoDL platform page was accessible, so OFF is not independently asserted',
  recovered_at=time.time())
 write_json(p,receipt)
test=read(ROOT/'server_results/work/final_test.json')
approval=read(r/'primary_review_approval.json')
assert file_sha(approval['report'])==approval['report_sha256']
assert file_sha(ROOT/'server_results/work/final_test.json')==approval['final_test_sha256']
requested=datetime.datetime.fromtimestamp(receipt['requested_at'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
status=dict(status='COMPLETE_SHUTDOWN_REQUEST_ACCEPTED',completed_at=time.time(),
 experiment_and_review_complete=True,required_local_backup_complete=True,
 shutdown_request_accepted=True,shutdown_requested_at_shanghai=requested,
 platform_off_confirmed=False,no_shutdown_request_repeated=True,
 report=approval['report'],retained_best_test=test['old_complete_pipeline']['mIoU_lesion4'],
 evaluated_new_candidate_test=test['selected_complete_pipeline']['mIoU_lesion4'],
 target=test['target'],target_met=test['target_met'],
 conditional_branches='M paired cost-stop at 2000; E asset gate unmet; no seed or mechanism follow-up',
 local_probability_files=836,persistent_negative_probability_files=228)
write_json(r/'completion_status.json',status)
print(json.dumps(status,ensure_ascii=True))
