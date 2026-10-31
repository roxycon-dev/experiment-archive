# -*- coding: utf-8 -*-
"""Confirm: does build_model/constructor mutate cfg so that the dump digest
matches the refcache manifest? Print the exact diff."""
import hashlib
import json
import sys

sys.path.insert(0, '.')
import runtime as rt  # noqa: E402

settings = json.load(open('settings.json'))
rt.inject_repo(settings['repo'])
man = json.load(open('/root/autodl-tmp/hrplus/cache/ref_train_v3/manifest.json'))

cfg = rt.load_config(settings['repo'], settings['config'])
cfg.model['source_checkpoint'] = settings['repo'] + '/' + \
    settings['reference_checkpoint']
before = cfg.dump()

# exactly what build_student does (rp.build_student without cuda)
model = rt.build_model(cfg)
after = cfg.dump()

print('digest before build_model:', hashlib.sha256(
    before.encode()).hexdigest())
print('digest after  build_model:', hashlib.sha256(
    after.encode()).hexdigest())
print('manifest digest          :', man.get('config_sha256'))
print('match after:', hashlib.sha256(
    after.encode()).hexdigest() == man.get('config_sha256'))

if before != after:
    import difflib
    diff = list(difflib.unified_diff(before.splitlines(),
                                     after.splitlines(),
                                     'before', 'after', lineterm=''))
    print('--- cfg diff (%d lines) ---' % len(diff))
    for line in diff[:60]:
        print(line)
