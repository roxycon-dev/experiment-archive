import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from nextlib.geometry import tiles,patch_tensor,supervision,global_view,align_global,augment_full
from nextlib.decision import best_candidate,paired_stop,seed_gate
from nextlib.data import make_stream,replay
from nextlib.evaluate import counts,metrics,Diagnostics,evaluate
from nextlib.common import digest


class FakeDDR:
    stems=['a','b']
    def __len__(self):return 2
    def load(self,sid):
        rgb=np.zeros((25,43,3),np.uint8)
        label=np.zeros((25,43),np.uint8)
        label[1,2]=1;label[24,42]=4
        return rgb,label


def history(values,arm='M0'):
    return [dict(arm=arm,update=u,solo={'mIoU_lesion4':v},fixed_half={'mIoU_lesion4':v-.001})
            for u,v in values]


class ProtocolTests(unittest.TestCase):
    def test_partial_evaluation_resumes_without_repeating_finished_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempts=[]
            def interrupted(rgb):
                attempts.append(1)
                if len(attempts)==2:raise RuntimeError('synthetic interruption')
                return np.full((4,25,43),.1,np.float32)
            with self.assertRaises(RuntimeError):
                evaluate(FakeDDR(),interrupted,tmp,{'test':'partial'})
            calls=[]
            def resumed(rgb):
                calls.append(1);return np.full((4,25,43),.1,np.float32)
            report=evaluate(FakeDDR(),resumed,tmp,{'test':'partial'})
            self.assertEqual(len(calls),1)
            self.assertEqual(report['n_images'],2)
            self.assertEqual(report['counts'][3],[0,0,2])
    def test_irregular_coverage_label_and_padding(self):
        rgb=np.random.default_rng(4).integers(0,256,size=(25,43,3),dtype=np.uint8)
        lab=np.zeros((25,43),np.uint8);lab[0,0]=4;lab[-1,-1]=1
        hit=np.zeros(lab.shape,int)
        restored=np.zeros((4,*lab.shape),np.uint8)
        for tile in tiles(*lab.shape,16,8):
            patch=patch_tensor(rgb,tile)[0]
            target,valid=supervision(lab,tile)
            hit[tile.y:tile.y+tile.height,tile.x:tile.x+tile.width]+=1
            restored[:,tile.y:tile.y+tile.height,tile.x:tile.x+tile.width]=target.numpy()[0,:,:tile.height,:tile.width]
            self.assertEqual(int(valid.sum()),tile.height*tile.width*4)
            if tile.y==0:self.assertTrue(torch.equal(patch[:,:8],torch.zeros_like(patch[:,:8])))
        self.assertTrue((hit==1).all())
        for c in range(4):self.assertTrue(np.array_equal(restored[c],lab==c+1))

    def test_coordinate_alignment_non_square_noninteger_scale(self):
        rgb=np.zeros((70,100,3),np.uint8)
        _,ledger=global_view(rgb,64)
        ph,pw=ledger['padded_hw']
        y=(torch.arange(ph//4)+.5)*4/ledger['scale_y']
        x=(torch.arange(pw//4)+.5)*4/ledger['scale_x']
        feat=(y[:,None]+2*x[None,:])[None,None]
        tile=next(tiles(70,100,32,16))
        aligned=align_global(feat,ledger,tile)[0,0]
        py=tile.y-tile.halo+(torch.arange(tile.window//4)+.5)*4
        px=tile.x-tile.halo+(torch.arange(tile.window//4)+.5)*4
        expected=py[:,None]+2*px[None,:]
        # Interior bins avoid the deliberately zero-padded feature border.
        self.assertTrue(torch.allclose(aligned[6:10,6:10],expected[6:10,6:10],atol=2e-5))
        self.assertTrue(torch.equal(aligned[:4],torch.zeros_like(aligned[:4])))

    def test_stream_is_exact_replay_and_foreground_is_in_center(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream=make_stream(FakeDDR(),Path(tmp)/'one.json',42,'native',64,8,16,8)
            torch.randn(731);np.random.rand(123)
            other=make_stream(FakeDDR(),Path(tmp)/'two.json',42,'native',64,8,16,8)
            self.assertEqual(stream['rows'],other['rows'])
            for row in stream['rows']:
                rgb,lab,tile=replay(FakeDDR(),row,'native',16,8)
                self.assertEqual(list(lab.shape),row['hw'])
            self.assertTrue(any(r['anchor_class']==4 for r in stream['rows']))

    def test_no_unilateral_stop_at_two_k_and_no_trend_is_paired(self):
        first=history([(0,.40),(1000,.401),(2000,.399)])
        self.assertFalse(paired_stop({'M0':first,'M1':first},.434877)[0])
        flat=first+history([(4000,.398)])
        self.assertTrue(paired_stop({'M0':flat,'M1':flat},.434877)[0])
        rising=first+history([(4000,.411)])
        self.assertFalse(paired_stop({'M0':flat,'M1':rising},.434877)[0])
        gained=history([(0,.43),(2000,.444),(4000,.441)])
        self.assertFalse(paired_stop({'M0':flat,'M1':gained},.434877)[0])

    def test_tie_order_and_seed_gate(self):
        h=[dict(update=1000,solo={'mIoU_lesion4':.44},fixed_half={'mIoU_lesion4':.44}),
           dict(update=2000,solo={'mIoU_lesion4':.44},fixed_half={'mIoU_lesion4':.44})]
        b=best_candidate(h)
        self.assertEqual((b['update'],b['output']),(1000,'solo'))
        self.assertTrue(seed_gate({'M0':h,'M1':h},.434877)[0])

    def test_completed_pair_reuses_selection_with_identity_guard(self):
        from nextlib.train import run_pair
        from nextlib.common import write_json,digest
        protocol={'updates':8000};stream={'rows_sha256':'same-data'};lock={'canvas':'native'}
        identity=dict(protocol_sha256=digest(protocol),stream_sha256=stream['rows_sha256'],
                      geometry_sha256=digest(lock),seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            expected={'arm':'M1','score':.44,'update':1000,'output':'solo'}
            write_json(Path(tmp)/'selection.json',dict(identity=identity,second_seed_gate=True,winner=expected))
            gate,winner=run_pair({},protocol,stream,None,None,lock,42,tmp,None)
            self.assertTrue(gate);self.assertEqual(winner,expected)
            with self.assertRaises(ValueError):
                run_pair({},dict(updates=4000),stream,None,None,lock,42,tmp,None)

    def test_metrics_strict_threshold_global_counts_and_lesion_miss(self):
        p=np.full((4,4,5),.5,np.float32)
        label=np.zeros((4,5),np.uint8);label[1,1]=4
        p[0,0,0]=np.nextafter(np.float32(.5),np.float32(1))
        ct=counts(p,label)
        self.assertEqual(ct[0].tolist(),[0,1,0]);self.assertEqual(ct[3].tolist(),[0,0,1])
        d=Diagnostics();d.add('a',p,label)
        r=d.report()
        self.assertEqual(r['lesion_buckets'][3][0][:4],[1,1,1,1])
        self.assertEqual(r['counts'],ct.tolist())

    def test_fixed25_capacity_bound(self):
        champion=torch.tensor([1/3,1/3-.001,2/3+.001])
        expert=torch.tensor([1.,1.,0.])
        pred=.75*champion+.25*expert>.5
        self.assertEqual(pred.tolist(),[False,False,True])


if __name__=='__main__':unittest.main()
