import json
from pathlib import Path
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[1]
now = datetime.now(timezone.utc).isoformat()
exception = {
    'updated_at': now, 'authority': 'Direct user instruction in this conversation',
    'instruction_verbatim': '这是messidor idrid我那个bestseg就是，虽然格式不同但完全没裁剪过保证可用',
    'accepted_source': 'D:/dataset/best_segmentation',
    'user_assertion': 'Uncropped and usable despite format difference',
    'official_native_IDRiD_requirement_waived_for_this_run': True,
    'independently_verified_uncropped': False,
    'actual_geometry': 'Stored square JPEG coordinates, variable native stored dimensions',
    'supervision': 'Four independent polygon-rasterized 0/255 masks in stored image coordinates',
    'empty_class_without_completeness_evidence': 'unknown; ignore its entire channel for this image',
    'official_GT_restoration_claimed': False,
    'training_split': '54 training images only; 27 testing images quarantined',
    'conversion_receipt': 'reports/idrid_polygon_conversion_receipt.json',
    'remaining_acceptance': ['Exact image/mask pairing and SHA', 'Cross-dataset image dedup',
                             'MAPLES uncertain-region ignore', 'At least 20 actual visual overlay reviews'],
    'architecture_optimizer_microbatch_and_selection_rules_changed': False,
    'MESSIDOR_access': 'User-authorized private access URL, legitimate server cookie, downloading 12 archives',
    'private_download_files': ['assets/messidor_download/access.private', 'assets/messidor_download/links.private',
                               'assets/messidor_download/cookies.private']
}
(root/'reports/external_source_user_override.json').write_text(json.dumps(exception, ensure_ascii=False, indent=2), encoding='utf-8')
def amend(name, action):
    path = root/'reports'/name
    data = json.loads(path.read_text(encoding='utf-8'))
    action(data)
    data['updated_at'] = now
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
def external(x):
    x.update(Q_MIX='NOT_READY_ACCEPTANCE_IN_PROGRESS', IDRiD='User-authorized stored images and derived masks; 54 train only',
             MAPLES='Authorized MESSIDOR originals downloading; official labels available',
             idrid_source_exception='reports/external_source_user_override.json')
amend('external_readiness.json', external)
def prep(x):
    x.update(stage='Q_MIX_EXTERNAL_DATA_PREPARATION', Q_MIX='NOT_READY_ACCEPTANCE_IN_PROGRESS',
             shutdown_allowed_now=False, next_action='ACQUIRE_MATCH_DEDUP_REVIEW_THEN_START_Q_MIX',
             suppress_repeated_ssh_poll_while_stopped=False, temporary_stop_dispatch='NOT_DISPATCHED',
             external_source_override='reports/external_source_user_override.json')
amend('preparation_status.json', prep)
def scope(x):
    x.update(shutdown_allowed_now=False,
             remaining_data_dependency='MESSIDOR originals acquiring; stored IDRiD accepted by direct user override; dataset acceptance remains',
             latest_data_instruction=exception['instruction_verbatim'],
             temporary_stop_reason='No stop dispatched. New authorized data acquisition enables original-plan continuation.',
             resume_condition='External supervision passes acceptance and prospective continuation is frozen',
             external_source_override='reports/external_source_user_override.json')
amend('execution_scope_override.json', scope)
def plan(x):
    x['shutdown_allowed'] = False
    x['tasks']['official_external_supervision'].update(state='ACQUISITION_AND_ACCEPTANCE_IN_PROGRESS',
        local_asset_quality='User-authorized stored image/polygon source; no official GT restoration claimed',
        user_override='reports/external_source_user_override.json')
    x['tasks']['server_shutdown_and_OFF_verification'].update(state='DEFERRED_ORIGINAL_PLAN_CONTINUATION',
        reason='No shutdown dispatched; new user data access is actionable')
amend('original_plan_completion.json', plan)
print('Direct user data override recorded; original Q_DDR selection and core protocol unchanged.')
