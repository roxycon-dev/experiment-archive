"""Reuse the frozen campaign writer with an explicit effective-runtime dependency."""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import advance_campaign
from qlib.common import read_json, write_json, sha, utc_now, assert_source

def main():
    assert_source(ROOT)
    runtime_path=ROOT/'configs/runtime.qmix.server.json'
    runtime=read_json(runtime_path)
    manifest=read_json(runtime['external_manifest'])
    if manifest['status']!='READY': raise ValueError('Q_MIX acceptance is incomplete')
    protocol=read_json(ROOT/'reports/qmix_continuation_protocol.json')
    if sha(runtime['external_manifest'])!=protocol['external_manifest_server_sha256']:
        raise ValueError('Accepted external manifest changed')
    prior=read_json(ROOT/'reports/campaign_state.json')
    if prior['arms']==['Q_DDR']:
        write_json(ROOT/'reports/Q_DDR_completed_campaign_snapshot.json',prior)
    # The original implementation and decisions stay unchanged. Only its runtime
    # reader receives the accepted MIX runtime, instead of overwriting the old
    # runtime file whose SHA belongs to the completed final-evaluation lock.
    original_reader=advance_campaign.read_json
    def effective_read(path):
        return runtime if Path(path).resolve()==(ROOT/'configs/runtime.server.json').resolve() else original_reader(path)
    advance_campaign.read_json=effective_read
    try:
        advance_campaign.main()
    finally:
        advance_campaign.read_json=original_reader
    write_json(ROOT/'reports/qmix_campaign_decision_provenance.json',{
        'created_at':utc_now(),'decision_writer':'advance_campaign.py','decision_writer_sha256':sha(ROOT/'advance_campaign.py'),
        'effective_runtime_sha256':sha(runtime_path),'external_manifest_sha256':sha(runtime['external_manifest']),
        'original_DDR_runtime_unchanged':True,'existing_final_selection_sha256':sha(ROOT/'reports/final_selection.json'),
        'existing_DDR_test_is_completed_and_must_not_be_repeated':True,
        'new_choice_requires_separate_lock_and_explicit_test_identity_check':True})

if __name__=='__main__':main()
