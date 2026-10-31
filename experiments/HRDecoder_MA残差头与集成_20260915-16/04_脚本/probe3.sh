#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
cd /root/autodl-tmp/hrplus/repo

echo '=== A) align_corners 来源 ==='
$PY - <<'EOF'
import sys
sys.path.insert(0,'/root/autodl-tmp/hrplus/repo')
from mmcv import Config
c=Config.fromfile('/root/autodl-tmp/hrplus/repo/configs/lesion/exp/hrplus_patch_lovasz.py')
print('model.align_corners =', c.model.get('align_corners'))
print('decode_head:', {k:v for k,v in c.model.decode_head.items() if k in ('type','align_corners','num_classes','out_channels','loss_decode')} if c.model.get('decode_head') else None)
print('base hr100_a model keys:', list(c.model.keys()))
EOF
echo '--- 类里 align_corners 默认值 ---'
grep -n 'align_corners' hr100_refinement.py | head -20

echo
echo '=== B) seg_fields 与 pipeline ==='
grep -n 'seg_fields' hrplus_patch.py
echo '--- lesion_dataset seg_fields ---'
grep -n 'seg_fields\|CLASSES\|ignore_index' mmseg/datasets/lesion_dataset.py | head -20

echo
echo '=== C) Dice smooth 实际取值 ==='
grep -rn 'smooth' configs/lesion/exp/hr100_a.py configs/lesion/exp/hrplus_patch_dice.py 2>/dev/null | head
echo '--- binary_loss 默认 ---'
grep -n 'def __init__\|smooth' mmseg/models/losses/binary_loss.py | head -20

echo
echo '=== D) DDR 标签里是否存在 255（抽样统计）==='
$PY - <<'EOF'
import numpy as np, glob, os
try:
    from PIL import Image
except Exception:
    import cv2; Image=None
d='/root/autodl-tmp/hrplus/repo/data/DDR_split_613/labels'
subs=sorted(os.listdir(d))[:2]
n255=0; tot=0; files=0
for s in subs:
    p=os.path.join(d,s)
    if not os.path.isdir(p): continue
    fs=sorted(os.listdir(p))[:8]
    for f in fs:
        a=np.array(Image.open(os.path.join(p,f))) if Image else None
        if a is None: continue
        files+=1; tot+=a.size; n255+=int((a==255).sum())
print('dirs:',subs,'files:',files,'total px:',tot,'px==255:',n255)
print('unique values sample:', sorted(set(np.unique(a).tolist())) if a is not None else 'n/a')
EOF
