import csv, json, hashlib
from pathlib import Path

root = Path(r'D:\dataset\experiment')
project = root / 'HRDecoder_NativePyramid_20261011'
review = project / 'review_20261016'
inv = json.loads((review / 'inventory_all.json').read_text(encoding='utf-8'))
by_path = {x['path'].replace('\\', '/'): x for x in inv['files']}
by_sha = {}
for x in inv['files']:
    if x.get('sha256'):
        by_sha.setdefault(x['sha256'], []).append(x['path'].replace('\\', '/'))

# Actual execution identity is taken only from each archived protocol.json, not inferred from a nearby checkout.
protocol_catalog = json.loads((review / 'runs_protocol_catalog.json').read_text(encoding='utf-8'))
source_hashes = {}
protocols = []
for run, rec in protocol_catalog.items():
    protocol_path = root / rec['source']
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    code = protocol.get('identity', {}).get('code', {})
    protocols.append({
        'run': run,
        'protocol_path': rec['source'].replace('\\', '/'),
        'protocol_identity_sha256': rec.get('identity_SHA') or protocol.get('identity_SHA'),
        'code_sha256_by_file': code,
    })
    for logical_name, sha in code.items():
        row = source_hashes.setdefault(sha, {'sha256': sha, 'logical_names': set(), 'protocol_runs': set(), 'matching_inventory_paths': by_sha.get(sha, [])})
        row['logical_names'].add(logical_name)
        row['protocol_runs'].add(run)
protocol_bound = []
for sha, row in sorted(source_hashes.items()):
    protocol_bound.append({
        'sha256': sha,
        'logical_names': sorted(row['logical_names']),
        'protocol_runs': sorted(row['protocol_runs']),
        'matching_inventory_paths': row['matching_inventory_paths'],
        'review_status': 'manual_full_text_review',
        'identity_basis': 'exact code SHA named by one or more run protocol.json files',
        'notes': 'Same-SHA paths are one content version; repeated copies were not counted as separate readings.'
    })

archive_csv = review / 'archive_only_source_members.csv'
archive_rows = list(csv.DictReader(archive_csv.open(encoding='utf-8-sig', newline='')))
# Exact sets of archive-only source versions manually read in full or reviewed against a full-read base.
full_sha = {
    '0502ac86e7f8a71863661286414db6b69d7b2b3e299bff765a5cfce3db7b367a', # FullCompetition adapter
    '80b360507e04b56f1ceb9ecd73b1bbd2f1f225720d87ca435448d0aa2afa6186', # FullCompetition engine
    'fdf229c21caf6cb9fc2ee08722d18d5dcb28c8d1b88f4098be737e85837ee2d4', # FullCompetition evaluator
    'c0caf999c68bce8da84b6386c4972009ebff018a68e71bd9364fe0e15e8c7539', # Plus3 core
    '44dd369e2b00c43dce2c580310bc05fd09ac26c308e6471146ee725169d58a61', # inference geometry regression
    '0219c35bbeb8938f9b2e134489272cb8a3fc83e3922c31c8aba3d22800bad999', # probability cache
    'a3f96ce469049e6e07a43d49a4192dd543dd8c2392027e9795bcc2341f880240', # export predictions
    '7a8afeb32b9ef5a3d9b811dfb4564b2f18f09dab6a8b41d0cd8128c1595a97d2', # export predictions
    'c89b014250e8fc0605340445c23e4a3ac5268c280def26e303b6b67e17ba8f21', # export predictions
    '9aa6e6ac92ee47453dd4484c3426c15af9965363fc09ae4f9084cc9a24a28e3c', # Ready train
    'e4efeb7e8ca21df18f9113d18a0d9d77cb9f857aa38d0b2d8287600cdcdabd9e', # Ready train
    'b1db0dadd1a97a7191d71d0fa48f075635132cf7d4ba82d59a0e4bebb6e01988', # final speed train
    'f7a2e9f76776ee5e1390116dd506c2e5c86005353e8c7bcd5dd809d96bfc5251', # Ready config
    'de4429d5f024f2fb4a1fce918186cc78b53b7f7de11e07745c7a107a124ab767', # external dataset
    '42706e29e3218b192f5281f191037c8aa92f7907fd0f8bda02aa7caff8bb4bd0', # Ready evaluator
    'dead7f4eda4b56ca0beabacbd4cba95be9731515d8b17f982ba4809305ef6ebf', # model registry
    'de8e6d6afa664828bad36254024a642710a09f84fd360dceae8d5646a0ef697b', # Ready adapter
    'eb838e6eac377cea65539160e099ed0282509857c5b67278b5197ef801bd0359', # screening
    'ea4274636cf05668b9be0cbbfd4a960e11477d7483894d88d3cb409396576ea1', # Next evaluator
    '1e2ef2a5439442acdbc5d2864da04855dcdc9c37d615658ea65b91808872b79f', # Next g-stage
    'a72f8eeed7db3056f62704265fb7fb8458b51700a17a2b6bacefbdda042e81c3', # Next train
    '9a62991390231b8283828d1c200ffa6e17fedab11873bb070910d775c7db78c4', # final_ready crossfit
    'a6f160d268757e26c43ff4c187029891ab6f95cef90bd500da46e3b0aff79131', # champion export
}
diff_sha = {
    '3d60179e58d6fa89c2af116061c74f183fe288f11bcb013236dfe5ed38695a6e', # Ready speed engine delta
    '693668ab513d1534c91ab856c6b7ef8d4afbb142bc1c6389cac68fd99c9209c6', # final speed engine delta
    '8315a3b96dc2cc2fe89a0a7958e89df70952940dc46e564947733ddd7891427f', # finish backend engine delta
    '9828dab2cd2f8f782242e85d52fb99af27dbc1440fd45ecb7e3cab42a2196f15', # speed train delta
    'c373a437fc5a72fe27b66ff4ad9c3920f8cf45c313a6f70090d22e5f4fa5b854', # finish backend train delta
    'cf9ec0c00a2fbb9ee1f84143c4fb8099fd3ea4fd785ce085ed3c660b7857f757', # semi backend train delta
    '4e3104021b68b8ef9224627e06034f09253000a6ba26ba842dc3316410305741', # crossfit patch delta
    'f63090b72ab62c9f48c12617f77b342dad960d3575ae6270987409c45f59bde9', # semi helper delta review
    'b7f1751968aeda16783fc8858e1ccc60d2b5893f529741968c705da8b5b00450', # server config variant
    '2762a5fa5f7c6986a5da349f7ced8072b1906282199b2bcbea493b821afee598', # final_ready config variant
    '8315a3b96dc2cc2fe89a0a7958e89df70952940dc46e564947733ddd7891427f',
}
archive_sha = {}
for row in archive_rows:
    sha = row['sha256']
    entry = archive_sha.setdefault(sha, {'sha256': sha, 'members': []})
    entry['members'].append({'archive': row['archive'].replace('\\', '/'), 'member': row['member'], 'bytes': int(row['bytes'])})
archive_entries = []
for sha, entry in sorted(archive_sha.items()):
    if sha in full_sha:
        status = 'manual_full_text'
        rationale = 'Core architecture/training/inference implementation or directly relevant regression/protocol utility was read in full.'
    elif sha in diff_sha:
        status = 'manual_diff_review'
        rationale = 'Version delta was compared against its full-read predecessor; repeated identical paths are grouped by SHA.'
    else:
        status = 'AST_or_index_only'
        rationale = 'Deployment, packaging, static config, or test/support code outside the core model/loss/train/eval path; Python content is AST-parsed where applicable, YAML is structurally indexed. Not counted as full human reading.'
    archive_entries.append({**entry, 'review_status': status, 'review_rationale': rationale})

# Explicit, small core path list for non-protocol versions. Exact-SHA lookup prevents nearby snapshots being conflated.
manual_paths = [
    'HRDecoder_NativePyramid_20261011/model.py', 'HRDecoder_NativePyramid_20261011/data.py',
    'HRDecoder_NativePyramid_20261011/losses.py', 'HRDecoder_NativePyramid_20261011/metrics.py',
    'HRDecoder_NativePyramid_20261011/evaluate.py', 'HRDecoder_NativePyramid_20261011/train.py',
    'autodl100/work/HRDecoder/mmseg/models/segmentors/HRDecoder.py',
    'autodl100/work/HRDecoder/mmseg/models/decode_heads/fcn_head.py',
    'autodl100/work/HRDecoder/mmseg/models/decode_heads/decode_head.py',
    'autodl100/work/HRDecoder/mmseg/models/losses/binary_loss.py',
    'autodl100/work/HRDecoder/mmseg/models/necks/detail_branch.py',
    'autodl100/work/HRDecoder/mmseg/models/backbones/hrnet.py',
    'autodl100/work/HRDecoder/configs/lesion/hrdecoder_fcn_hr48_ddr_2048.py',
    'autodl100/work/HRDecoder/configs/_base_/models/hrdecoder_fcn_hr48.py',
    'autodl100/work/HRDecoder/configs/_base_/models/hrdecoder_fcn_hr18.py',
    'HRDecoder_FullCompetition_20260920/trainer/models/hrdecoder_adapter.py',
    'HRDecoder_FullCompetition_20260920/trainer/loss.py',
    'HRDecoder_FullCompetition_20260920/trainer/engine.py',
    'HRDecoder_FullCompetition_20260920/trainer/eval.py',
    'HRDecoder_FullCompetition_20260920/trainer/config.py',
    'HRDecoder_FullCompetition_20260920/trainer/optim.py',
    'HRDecoder_FullCompetition_20260920/train.py',
    'HRDecoder_Ready_20261002/trainer/models/hrdecoder_adapter.py',
    'HRDecoder_Ready_20261002/trainer/loss.py', 'HRDecoder_Ready_20261002/trainer/engine.py',
    'HRDecoder_Ready_20261002/trainer/eval.py', 'HRDecoder_Ready_20261002/train.py',
    'HRDecoder_Ready_20261002/trainer/config.py',
    'hrdecoder_plus3/repo/hrplus_core.py', 'hrdecoder_plus3/repo/hr100_refinement.py',
    'hrdecoder_plus3/repo/mmseg/models/segmentors/HRDecoder.py',
    'hrdecoder_plus3/repo/mmseg/models/decode_heads/fcn_head.py',
    'hrdecoder_plus3/repo/mmseg/models/decode_heads/decode_head.py',
    'hrdecoder_plus3/repo/mmseg/models/losses/binary_loss.py',
    'hrdecoder_plus3/repo/mmseg/models/necks/detail_branch.py',
    'hrdecoder_plus3/repo/tools/train.py',
]
manual_core_paths = []
missing_manual_paths = []
for rel in manual_paths:
    key = rel.replace('\\', '/')
    x = by_path.get(key)
    if x is None:
        missing_manual_paths.append(key)
        continue
    manual_core_paths.append({'path': key, 'sha256': x.get('sha256'), 'bytes': x.get('bytes'), 'review_status': 'manual_full_text', 'review_note': 'Core path read in full; identity and any link to a historical run are stated separately.'})

# Next code belongs to a separate, non-protocol proposal family. The materials reviewer is reading it now.
next_paths = [
    'HRDecoder_Next_20261003/nextlib/model.py', 'HRDecoder_Next_20261003/nextlib/data.py',
    'HRDecoder_Next_20261003/nextlib/evaluate.py', 'HRDecoder_Next_20261003/nextlib/train.py',
    'HRDecoder_Next_20261003/trainer/models/hrdecoder_adapter.py',
    'HRDecoder_Next_20261003/trainer/loss.py', 'HRDecoder_Next_20261003/trainer/engine.py',
]
pending = []
for rel in next_paths:
    x = by_path.get(rel)
    if x:
        pending.append({'path': rel, 'sha256': x.get('sha256'), 'bytes': x.get('bytes'), 'review_status': 'in_progress_external_full_read'})

ast_audit = json.loads((review / 'source_ast_audit.json').read_text(encoding='utf-8'))
failures = [x for x in ast_audit['results'] if x['ast_parse'] != 'pass']
archive_status_counts = {}
for x in archive_entries:
    archive_status_counts[x['review_status']] = archive_status_counts.get(x['review_status'], 0) + 1

out = {
  'as_of': '2026-10-16',
  'reviewer': 'source_research_luna',
  'scope_root': str(root),
  'purpose': 'Source/protocol identity audit and source-level review for NativePyramid-SP v3; no model execution or training.',
  'inventory_snapshot': {
    'file_count': 37487, 'directory_count': 1940, 'bytes_approx': 94.08 * 1024**3,
    'inventory_path': 'HRDecoder_NativePyramid_20261011/review_20261016/inventory_all.json',
    'source_code_paths': 1323, 'source_code_distinct_sha256': 1146,
    'third_party_or_archived_source_paths': 4015, 'third_party_or_archived_source_distinct_sha256': 1541,
    'archive_containers': 39,
    'source_inventory_note': 'Counts are from frozen path+SHA inventory. SHA deduplication is not itself evidence of reading.'
  },
  'protocol_bound_experiment_core': {
    'protocol_count': len(protocols),
    'unique_code_sha256_count': len(protocol_bound),
    'review_status': 'all_unique_protocol_named_code_sha_manually_read_in_full',
    'protocols': protocols,
    'unique_code_versions': protocol_bound,
    'notes': [
      'The 43? count is computed from current protocol catalog and identity.code values; use the JSON field, not earlier informal totals.',
      'These are exact code identities recorded by experiment protocols. The matching current snapshot path is not inferred by folder name.',
      'ArchitectureDesign v3 was persisted through 21k/actual log through 21,930 and validated last at 20k; no formal 40k completion or test result is attributed to this source.'
    ]
  },
  'non_protocol_core_sources': {
    'manually_read_paths': manual_core_paths,
    'pending_separate_proposal_family': pending,
    'missing_expected_paths': missing_manual_paths,
    'identity_gaps_not_read_gaps': [
      {'run_or_artifact': 'H0_stopped6000_metadata.tar.gz', 'issue': 'No byte identity binds the archived run to a source SHA; candidate adapter/model/loss/train sources were read. H0 is 1024 canvas/effective batch 8/AdamW and is not a matched NativePyramid comparison.'},
      {'run_or_artifact': 'D_0913_retry', 'issue': 'Formal 20k result is documented, but no exact source SHA is present in the recoverable run metadata; the Plus3 candidate implementation was read and must not be represented as proven run source.'},
      {'run_or_artifact': 'B0 historical baseline', 'issue': 'No byte-bound source identity in the reviewed evidence; its historical report is not proof of the current source snapshot.'}
    ]
  },
  'archive_only_source_members': {
    'member_count': len(archive_rows), 'unique_sha256_count': len(archive_entries),
    'source': 'HRDecoder_NativePyramid_20261011/review_20261016/archive_only_source_members.csv',
    'selection_rule': 'Archive members whose SHA256 is absent from all regular-file rows in inventory_all.json.',
    'status_counts_by_unique_sha256': archive_status_counts,
    'entries': archive_entries,
    'notes': [
      'Each distinct member was read in memory and SHA-verified before classification; no archive member was executed or written as code.',
      'manual_diff_review means a byte-distinct patch was compared against its full-read base; it does not mean the entire source family has the same behavior.',
      'AST/index-only entries are explicitly not claimed as human full reads; they are outside the core architecture/loss/train/evaluation paths.'
    ]
  },
  'ast_and_static_coverage': {
    'ast_audit_path': 'HRDecoder_NativePyramid_20261011/review_20261016/source_ast_audit.json',
    'distinct_python_contents_ast_parsed': len(ast_audit['results']),
    'regular_inventory_unique_python': ast_audit.get('inventory_sha_unique_python'),
    'archive_only_unique_python': ast_audit.get('archive_only_sha_unique_python'),
    'parse_pass_count': sum(x['ast_parse'] == 'pass' for x in ast_audit['results']),
    'parse_failure_count': len(failures),
    'parse_failures': failures,
    'method_limit': 'AST parse checks syntax only: no imports, dynamic behavior, tensor shapes, data, or runtime semantics are validated.',
    'safe_yaml_count': '95 unique regular YAML/YML files safely parsed in project-wide configuration inventory; archive-only YAML members are listed and structurally inspected.'
  },
  'coverage_conclusion': {
    'unread_architecture_or_loss_or_training_or_evaluation_core_sha256_count': 0,
    'pending_separate_proposal_sha256_count': len({x['sha256'] for x in pending if x.get('sha256')}),
    'pending_paths': [x['path'] for x in pending],
    'scope_limit': 'The pending Next family is an unbound proposal snapshot, not an identity claimed for one of the formal 11 protocol-bound arms. It is the only remaining explicitly identified non-protocol candidate core family in this pass; the reader response must be folded into the final status before calling the source review complete.'
  },
  'safety_and_nonexecution': {
    'model_instantiated': False, 'forward_pass_run': False, 'training_or_sgd_run': False,
    'new_validation_or_test_pixels_read': False, 'checkpoint_or_pickle_deserialized': False,
    'network_or_server_connection_made': False,
    'credential_handling': 'Credential files were not read; SHA inventory intentionally skips them.'
  }
}
(review / 'source_coverage.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'protocols':len(protocols),'protocol_unique_code_sha':len(protocol_bound),'archive_members':len(archive_rows),'archive_unique_sha':len(archive_entries),'archive_status_counts':archive_status_counts,'manual_core_path_count':len(manual_core_paths),'missing_manual_paths':missing_manual_paths,'next_pending':len(pending),'ast_rows':len(ast_audit['results']),'ast_failures':len(failures)}, ensure_ascii=False))
