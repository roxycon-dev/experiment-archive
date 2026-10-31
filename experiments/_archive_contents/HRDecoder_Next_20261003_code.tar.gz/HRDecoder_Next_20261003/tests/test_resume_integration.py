"""Exercise the real new trainer and resume path on tiny synthetic images, without mmcv."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from nextlib.common import ROOT,read_json,trusted_load,strict_load,checked_asset,write_json
from nextlib.data import make_stream
from nextlib.model import SemanticDetailUNet
from nextlib.train import train_to_node
from trainer.models.convnext_unet import ConvNeXtUNet
from preflight import SyntheticGlobal
from nextlib.evaluate import evaluate


class TinyDDR:
    stems=['a','b']
    split='val'
    def __len__(self):return 2
    def load(self,sid):
        rgb=np.random.default_rng(ord(sid)).integers(0,256,size=(32,32,3),dtype=np.uint8)
        label=np.zeros((32,32),np.uint8)
        for c in range(4):label[3+c*5:5+c*5,2+c*5:5+c*5]=c+1
        return rgb,label


@unittest.skipUnless(torch.cuda.is_available(),'Small actual CUDA trainer acceptance requires CUDA')
class ResumeIntegration(unittest.TestCase):
    def test_checkpoint_resume_matches_uninterrupted_and_completed_eval_is_reused(self):
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        settings=read_json(ROOT/'configs/settings.json')
        protocol=dict(updates=4,warmup_updates=2,micro_batch=2,grad_accum=4,workers=0,
                      center=32,halo=16,global_long_side=64,gradient_checkpointing=False)
        lock=dict(start_asset='U0_highres',canvas='native',data_sha256='synthetic-data',
                  source_sha256='synthetic-test-source',baseline_val_miou=0.)
        def build(s,start,arm,seed):
            torch.manual_seed(seed)
            local=ConvNeXtUNet()
            strict_load(local,checked_asset(s,start))
            global_branch=SyntheticGlobal()
            torch.manual_seed(seed+1)
            return SemanticDetailUNet(local,global_branch,True),{'scope':'synthetic global acceptance only'}
        with tempfile.TemporaryDirectory(prefix='resume_acceptance_',dir=ROOT/'tests') as temp:
            root=Path(temp).resolve()
            assert (ROOT/'tests').resolve() in root.parents
            ds=TinyDDR()
            stream=make_stream(ds,root/'stream.json',42,'native',4,8,32,16)
            baseline=root/'base'
            def base_prediction(rgb):
                return np.full((4,32,32),.1,np.float32)
            evaluate(ds,base_prediction,baseline,{'scope':'synthetic'})
            with patch('nextlib.train.build_fusion',build):
                for node in (0,1,2):
                    train_to_node(settings,protocol,stream,ds,ds,lock,'M1',42,node,root/'resumed',baseline)
                train_to_node(settings,protocol,stream,ds,ds,lock,'M1',42,2,root/'direct',baseline)
            resumed=trusted_load(root/'resumed/last.pth');direct=trusted_load(root/'direct/last.pth')
            self.assertEqual(resumed['stream_cursor'],direct['stream_cursor'])
            self.assertEqual(resumed['stream_cursor'],16)
            a,b=resumed['model_state'],direct['model_state']
            maximum=max(float((a[k]-b[k]).abs().max()) for k in a)
            self.assertEqual(maximum,0.)
            def must_not_run(rgb):raise AssertionError('Complete exact cache must not rerun inference')
            evaluate(ds,must_not_run,baseline,{'scope':'synthetic'})
            write_json(ROOT/'reports/resume_acceptance.json',dict(status='PASS',scope='tiny synthetic images and global branch; real U0 and real new trainer',
                       resumed_and_direct_model_max_error=maximum,stream_cursor=16,complete_cache_reused=True,
                       cuda=torch.cuda.get_device_name(0)))


if __name__=='__main__':unittest.main()
