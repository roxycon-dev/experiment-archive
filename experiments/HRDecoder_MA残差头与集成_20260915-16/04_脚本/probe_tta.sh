#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /root/autodl-tmp/hrplus/repo

echo '=== 1) mmseg base.py 是否实现 aug_test ==='
grep -n 'def aug_test\|def simple_test\|def forward_test\|num_augs\|NotImplementedError' mmseg/models/segmentors/base.py | head -20
echo
echo '=== 2) EncoderDecoder 是否实现 aug_test ==='
grep -n 'def aug_test' mmseg/models/segmentors/encoder_decoder.py mmseg/models/segmentors/*.py | head
echo
echo '=== 3) tools/test.py 是否有 --aug-test ==='
grep -n 'aug.test\|aug_test\|img_ratios' tools/test.py | head -20
echo
echo '=== 4) 当前 test pipeline 长什么样 ==='
/root/miniconda3/bin/python - <<'PYEOF'
import sys
sys.path.insert(0,'/root/autodl-tmp/hrplus/repo')
from mmcv import Config
c=Config.fromfile('/root/autodl-tmp/hrplus/repo/configs/lesion/exp/hrplus_patch_lovasz.py')
print('test pipeline:')
for t in c.data.test.pipeline:
    print('   ', t.get('type'), {k:v for k,v in t.items() if k!='type'})
print()
print('test data keys:', list(c.data.test.keys()))
print('test img_dir/split:', getattr(c.data.test,'img_dir',None), getattr(c.data.test,'split',None))
PYEOF
echo
echo '=== 5) D 测试日志的指标段落（确认输出格式与可比性）==='
tail -25 /root/autodl-tmp/hrplus/d_test.log 2>/dev/null
