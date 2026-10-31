# -*- coding: utf-8 -*-
"""Server-side: sanitise DEVIATIONS.jsonl (keep only valid JSON lines) and
append the declared D-student restore entry."""
import json
import os

RUN = '/root/autodl-tmp/hrplus/glm/runs/r6'
p = os.path.join(RUN, 'DEVIATIONS.jsonl')
lines = [l for l in open(p, encoding='utf-8').read().splitlines() if l.strip()]
good = []
for l in lines:
    try:
        json.loads(l)
        good.append(l)
    except Exception:
        print('dropping malformed line: %s' % l[:100])
entry = {
    'time': '2026-09-18T14:55:00',
    'kind': 'operator_weight_restore',
    'detail': {
        'asset': 'weights/glm_D_student.pth',
        'cause': ('deleted with D_0913_retry during the 14:00 disk cleanup '
                  '(the weights path was a symlink into that directory)'),
        'restored_from': ('/root/autodl-tmp/audit/audit_20260916_A/tests/'
                          'c2_roundtrip.pth'),
        'method': ('strip 20 ma_refiner.* keys; functional match: val '
                   'per-image per-class counts bit-identical on 76/76 images '
                   'vs the P0 baseline (val 0.426686)'),
        'old_sha256': ('5fe18717b62d8fc3ab4846a50c046588e2af05abbbfdb3a958'
                       'f4794018b319ea'),
        'new_sha256': ('d583548a26ce31a16c9ae29f15377ce440bf14e2436592814a5'
                       '0de388a962166'),
    },
}
good.append(json.dumps(entry, sort_keys=True))
with open(p, 'w', encoding='utf-8') as f:
    f.write('\n'.join(good) + '\n')
print('DEVIATIONS valid lines: %d' % len(good))
for l in good[-2:]:
    print(json.dumps(json.loads(l), sort_keys=True)[:200])
