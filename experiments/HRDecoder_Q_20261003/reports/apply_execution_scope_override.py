"""Persist the user's revised completion and shutdown scope; keep frozen protocol intact."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

override = {
    'updated_at_utc': NOW,
    'authority': 'Direct user instruction in this conversation, 2026-10-04 Asia/Shanghai',
    'instruction_verbatim': '这个任务做完后不要关机啊，吧原方案完全做完后再关机',
    'supersedes': 'Shutdown after the currently running Q_DDR-only stage',
    'shutdown_after_q_ddr_only': False,
    'shutdown_allowed_now': False,
    'shutdown_authorized_after_original_plan_complete': True,
    'original_plan_sources': [
        'D:/dataset/experiment/HRDecoder_Breakthrough_Plan_20261003/01_成熟突破方案.md',
        'D:/dataset/experiment/HRDecoder_Breakthrough_Plan_20261003/02_执行与验收规范.md',
    ],
    'completion_gate': [
        'All applicable original-plan work completed and acceptance recorded',
        'No unresolved required external-supervision task treated as completed',
        'All necessary results recovered locally and SHA verified',
        'Dependency audit and authorized deferred cleanup completed',
        'Only then request shutdown and verify platform OFF',
    ],
    'preserve': [
        'Frozen core training protocol, run identities and final Q_DDR selection',
        'Original technical pause, severe failure, extension and promotion gates',
        'No threshold, TTA, loss, model or extra-seed search',
        'No test-driven re-selection or repeated tests for optimization',
        'Needed and ambiguous assets until the whole applicable scope is closed',
    ],
    'remaining_data_dependency': 'Official IDRiD native images/pixel GT and MAPLES paired MESSIDOR-1 originals',
    'late_external_arm': 'Freeze a prospective continuation protocol before training; disclose that Q_DDR selection/test started before official external originals were available. Do not rewrite the existing lock or call a later test a first untouched test.',
    'automatic_shutdown_job_found': False,
    'automation_id': 'hrdecoder-q',
    'monitor_minutes': 45,
}
write(ROOT / 'reports/execution_scope_override.json', override)

status_path = ROOT / 'reports/preparation_status.json'
status = read(status_path)
status.update({
    'shutdown_authorized': True,
    'shutdown_after_q_ddr_only': False,
    'shutdown_allowed_now': False,
    'shutdown_gate': 'ORIGINAL_PLAN_APPLICABLE_WORK_COMPLETE_THEN_LOCAL_SHA_AND_DEPENDENCY_CLEANUP',
    'execution_scope_override': 'reports/execution_scope_override.json',
    'original_plan_complete': False,
    'scope_instruction_updated_at_utc': NOW,
})
write(status_path, status)

completion_path = ROOT / 'reports/original_plan_completion.json'
completion = read(completion_path) if completion_path.exists() else {}
completion.update({
    'updated_at_utc': NOW,
    'original_plan_complete': False,
    'shutdown_allowed': False,
    'scope_override': 'reports/execution_scope_override.json',
    'tasks': {
        'pretraining_and_gpu_acceptance': {'state': 'COMPLETE', 'repeat_not_required': True},
        'Q_DDR_discovery_and_review': {'state': 'COMPLETE', 'seeds': [42, 2024], 'endpoint': 16000},
        'locked_Q_DDR_final_evaluation': {'state': 'STARTED_NOT_YET_ACCEPTED_LOCALLY', 'lock': 'reports/final_selection.json', 'selection_changes_forbidden': True},
        'official_external_supervision': {'state': 'PENDING_REQUIRED_ORIGINAL_IMAGES_AND_ACCEPTANCE', 'readiness_evidence': 'reports/external_readiness.json'},
        'Q_MIX_paired_experiment': {'state': 'PENDING_DATA_AND_PROSPECTIVE_CONTINUATION_PROTOCOL', 'preserve_original_comparison_and_cost_gates': True},
        'original_plan_final_reports_and_local_SHA_backup': {'state': 'PENDING'},
        'dependency_audit_and_deferred_cleanup': {'state': 'DEFERRED_UNTIL_APPLICABLE_ORIGINAL_SCOPE_COMPLETE'},
        'server_shutdown_and_OFF_verification': {'state': 'DEFERRED_UNTIL_ALL_PRIOR_REQUIRED_TASKS_COMPLETE'},
    },
    'optional_research_not_automatically_added': ['Third seed', 'Other model or loss search', 'Paper ablations before confirmed gain', 'External evaluation without proof of unused holdout identity'],
})
write(completion_path, completion)

cleanup_path = ROOT.parent / '_maintenance/deferred_cleanup_policy_20261004.json'
cleanup = read(cleanup_path)
cleanup.update({
    'updated_at_utc': NOW,
    'latest_scope_instruction': override['instruction_verbatim'],
    'completion_scope': 'All applicable tasks in the original two-file plan; Q_DDR-only completion is insufficient',
    'cleanup_gate': [
        'Applicable original-plan experiments and required evaluation complete',
        'All necessary results recovered locally and SHA verified',
        'Check active, upcoming original-plan and final pipeline dependencies',
        'Delete only explicitly confirmed unnecessary files; retain ambiguous assets and evidence',
    ],
})
write(cleanup_path, cleanup)
print(json.dumps({'scope_override_saved': True, 'shutdown_allowed_now': False, 'original_plan_complete': False, 'frozen_training_protocol_modified': False, 'updated_at_utc': NOW}, ensure_ascii=False))
