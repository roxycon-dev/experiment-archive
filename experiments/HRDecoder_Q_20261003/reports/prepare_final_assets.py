"""Prepare historical baseline metadata only; never load new test GT pixels."""
from pathlib import Path
import shutil
import sys
import numpy as np

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from qlib.common import read_json,sha,write_json

old=root.parent/'HRDecoder_Next_20261003'
ready=root.parent/'HRDecoder_Ready_20261002'
original=old/'configs/original_deployment_lock.json'
assert sha(original)==read_json(root/'configs/p0_identity.json')['historical_lock_sha256']
shutil.copyfile(original,root/'configs/p0_original_lock.json')
final=read_json(ready/'server_results/reports/finish/final_test.json')
progress=read_json(ready/'server_results/reports/finish/test_progress.json')
per_image={sid:r['selected_pipeline'] for sid,r in progress['images'].items()}
assert progress['lock_sha256']==sha(original) and final['lock_sha256']==sha(original)
assert final['n_images']==227 and len(per_image)==227
assert np.asarray(list(per_image.values())).sum(0).tolist()==final['selected_pipeline_counts']
assert abs(final['selected_pipeline_miou']*100-44.7808914511)<1e-8
write_json(root/'reports/p0_historical_test_reference.json',{'n_images':227,
    'historical_lock_sha256':sha(original),'per_image':per_image,'counts':final['selected_pipeline_counts'],
    'miou_pct':final['selected_pipeline_miou']*100,'threshold':.5,'strict_greater':True,
    'source_final_sha256':sha(ready/'server_results/reports/finish/final_test.json'),
    'source_progress_sha256':sha(ready/'server_results/reports/finish/test_progress.json'),
    'new_test_label_pixels_read':False,'purpose':'exact identity of pre-existing frozen P0; never Q selection'})
print('Historical P0 baseline identity prepared; no new test label pixels read')
