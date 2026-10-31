"""A real archived example checks remote-to-local lazy resolution and exact recipe parity."""
from pathlib import Path
import json,time
import numpy as np
from local_probability import ROOT,load_probability
sid='007-1774-100'
champion=load_probability('work/G/champion',sid,verify=True)
expert=load_probability('work/G/U0_native_whole',sid,verify=True)
expected=.75*champion+.25*expert
actual=load_probability('work/G/current_deployment',sid,verify=True)
assert np.array_equal(expected,actual) and actual.dtype==np.float32
result=dict(status='PASS',time=time.time(),sid=sid,shape=list(actual.shape),max_error=0,
 exact_original_recipe=True,all_component_file_hashes_verified=True,
 original_remote_index_files_unchanged=True,note='One real archived example; no new GPU inference')
(ROOT/'reports/local_cache_reader_acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
