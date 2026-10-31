"""Speed changes must preserve binary geometry and refresh persistent epochs."""
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trainer.data.ddr_dataset import DDRSegDataset
from trainer.data.augment import TrainAugment
from trainer.seed import worker_init_fn

def main():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        for d in ('images/train','labels/train'):(root/d).mkdir(parents=True)
        for i in range(2):
            Image.fromarray(np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3)).save(root/f'images/train/{i}.jpg')
            Image.fromarray(np.ones((32,32),dtype=np.uint8)).save(root/f'labels/train/{i}.png')
        aug=TrainAugment(crop_size=32,resize_long_side=32,resize_ratio_range=(1,1))
        ds=DDRSegDataset(str(root),'train',mean=[0]*3,std=[1]*3,augment=aug)
        x=ds._load_arrays(0);y=ds._load_arrays(0)
        assert x[1].dtype==np.uint8 and x is y
        u=aug(*x[:3],np.random.RandomState(3))
        f=aug(x[0],x[1].astype('float32'),x[2].astype('float32'),np.random.RandomState(3))
        assert all(np.array_equal(a,b) for a,b in zip(u,f))
        loader=torch.utils.data.DataLoader(ds,batch_size=1,num_workers=1,
            persistent_workers=True,worker_init_fn=worker_init_fn)
        first=[b['image'] for b in loader];ds.set_epoch(1);second=[b['image'] for b in loader]
        assert any(not torch.equal(a,b) for a,b in zip(first,second))
        loader._iterator._shutdown_workers()
    print(json.dumps(dict(uint8_geometry_equivalent=True,cache_reuses_raw_arrays=True,
                          persistent_workers_receive_new_epoch=True)))

if __name__=='__main__':main()
