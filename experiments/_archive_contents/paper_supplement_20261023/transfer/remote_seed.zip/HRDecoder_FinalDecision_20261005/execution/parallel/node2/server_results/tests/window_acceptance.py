"""Packed cache has EXACT inherited tensors for rotations, flips and padding."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from inputs import Windows
from qlib.data import WindowDataset
from qlib.common import write_json,utc_now
ROOT=Path(__file__).resolve().parents[1]

def main():
    rng=np.random.default_rng(51005);rgb=rng.integers(0,256,(803,901,3),dtype=np.uint8)
    label=rng.integers(0,5,(803,901),dtype=np.uint8);label[:33,:27]=255
    target=np.stack([label==c for c in range(1,5)],-1).astype(np.uint8)
    valid=np.broadcast_to((label!=255)[...,None],target.shape).astype(np.uint8)
    class Mock:
        def load(self,sid):return rgb,target,valid
    rows=[]
    for k in range(4):
        for fh in (False,True):
            for fv in (False,True):
                h,w=rgb.shape[:2] if k%2==0 else rgb.shape[1::-1]
                for y,x in ((0,0),(h-40,w-51)):
                    rows.append({'source':'DDR','sid':'synthetic','rotation':k,'flip_h':fh,'flip_v':fv,
                        'hw':[h,w],'y':y,'x':x,'foreground_class':None,'brightness':.94,'contrast':1.07})
    compact=Windows({'DDR':{'synthetic':(rgb,label)}},rows,0,len(rows));original=WindowDataset({'DDR':Mock()},rows)
    for i in range(len(rows)):
        new,old=compact[i],original[i]
        for a,b in zip(new[:3],old):assert torch.equal(a,b),(i,float((a-b).abs().max()))
    write_json(ROOT/'reports/window_acceptance.json',{'status':'PASS','cases':len(rows),
        'RGB_target_valid_bitwise_equal':True,'covers':['four rotations','both flips','photometric augmentation','padding/ignore'], 'time':utc_now()})
    print('PACKED_WINDOWS_EXACT_PASS',len(rows))

if __name__=='__main__':main()
