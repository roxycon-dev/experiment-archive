import unittest
import tempfile
import json
import hashlib
from pathlib import Path
import numpy as np
from trainer.fusion_probe import pixel_counts, FixedFusionProbe
from trainer.screening import decide_stop
from tools.probability_cache import save_probability


class Round2Test(unittest.TestCase):
    def test_complete_cache_api_and_baseline_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ids=['image%02d'%i for i in range(76)]
            base=np.tile(np.array([[.6,.1],[.1,.6]],dtype=np.float32),(4,1,1))
            target=np.tile(np.array([[1,0],[0,1]],dtype=np.float32),(4,1,1))
            for name in ['champion','u0']:
                cache=root/name;cache.mkdir()
                idx={'split':'val','dtype':'float32','coordinates':'original_image',
                     'channel_order':['EX','HE','SE','MA'],
                     'samples':[{'sample_id':sid} for sid in ids], 'checkpoint_sha256':'u0-test'}
                (cache/'index.json').write_text(json.dumps(idx))
                for sid in ids:save_probability(cache/sid,base)
            lock={'deploy':[{'family':'convnext_unet','alpha':.25}]*4,
                  'source_fusion':{'locked_val_miou':1.},
                  'candidates':{'convnext_unet':{'members':[{'checkpoint_sha256':'u0-test'}]}}}
            p=root/'lock.json';p.write_text(json.dumps(lock))
            probe=FixedFusionProbe({'champion_cache':str(root/'champion'),
                'baseline_u0_cache':str(root/'u0'),'baseline_lock':str(p),
                'baseline_lock_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                'baseline_counts_cache':str(root/'counts.json')})
            for sid in ids:probe.update(sid,np.zeros_like(base),target,np.ones_like(base))
            result=probe.compute()
            self.assertEqual(result['fixed_fusion_baseline_miou'],1.)
            self.assertEqual(result['fixed_fusion_miou'],0.)
            self.assertEqual(result['fixed_fusion_counts'],[[0,0,152]]*4)

    def test_threshold_and_ignored_pixels(self):
        prob=np.tile(np.array([[.5,.8],[.9,.1]],dtype=np.float32),(4,1,1))
        target=np.tile(np.array([[1,1],[0,1]],dtype=np.float32),(4,1,1))
        valid=np.tile(np.array([[1,1],[0,1]],dtype=np.float32),(4,1,1))
        self.assertEqual(pixel_counts(prob,target,valid).tolist(),[[1,0,2]]*4)

    def test_keep_complementary_candidate_even_when_solo_falls(self):
        rows=[{'update':1000,'mIoU_lesion4':.39,'fixed_fusion_miou':.435},
              {'update':2000,'mIoU_lesion4':.38,'fixed_fusion_miou':.437}]
        policy={'enabled':True,'mode':'paired_continuation','reference_fusion_miou':.434877}
        self.assertFalse(decide_stop(rows,4000,policy)['stop'])

    def test_stop_stalled_unhelpful_pair(self):
        rows=[{'update':1000,'mIoU_lesion4':.38,'fixed_fusion_miou':.431},
              {'update':2000,'mIoU_lesion4':.379,'fixed_fusion_miou':.430}]
        policy={'enabled':True,'mode':'paired_continuation','reference_fusion_miou':.434877}
        self.assertTrue(decide_stop(rows,4000,policy)['stop'])


if __name__=='__main__':unittest.main()
