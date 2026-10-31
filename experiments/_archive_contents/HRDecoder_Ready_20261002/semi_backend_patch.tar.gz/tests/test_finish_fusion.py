"""End-to-end nomination, cross-fit, locked vote and paired bootstrap fixture."""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import campaign.finish_campaign as finish
import tools.crossfit_fusion as cross
from tools.probability_cache import save_probability

def main():
    with tempfile.TemporaryDirectory(prefix='_tmp_finish_',dir=str(ROOT/'tests')) as tmp:
        root=Path(tmp).resolve();assert ROOT.resolve() in root.parents
        ids=[f'im_{i:02d}' for i in range(76)]
        (root/'reports').mkdir();(root/'labels').mkdir()
        folds=dict(split='val',folds={s:i%3 for i,s in enumerate(ids)},limitation='synthetic fixture')
        (root/'reports/fusion_folds.json').write_text(json.dumps(folds))
        gt=np.zeros((8,8),np.uint8)
        for c in range(4):gt[c,:3]=c+1
        good=np.stack([np.where(gt==c+1,.9,.1) for c in range(4)]).astype('float32')
        base=good.copy()
        for c in range(4):base[c,c,0]=.3;base[c,7,c]=.7
        samples=[dict(sample_id=s,shape=[4,8,8]) for s in ids]
        for name,p in [('champion',base),('candidate',good)]:
            d=root/name;d.mkdir()
            for sid in ids:save_probability(d/sid,p);Image.fromarray(gt).save(root/'labels'/f'{sid}.png')
            (d/'index.json').write_text(json.dumps(dict(split='val',dtype='float32',coordinates='original_image',
                channel_order=['EX','HE','SE','MA'],samples=samples,checkpoint_sha256='synthetic',config_sha256='synthetic')))
        spec=dict(id='candidate',family='toy',cache=str(root/'candidate'),members=[])
        finish.ROOT=root
        a,b=finish.assess([spec],root/'champion',root/'labels',ids)
        assert abs(a['base_miou']-.5)<1e-9 and a['nominated'][0]['alone_miou']==1.
        def command(args,tag):
            sys.argv=[str(args[0])]+list(map(str,args[1:]))
            with contextlib.redirect_stdout(io.StringIO()):cross.main()
        finish.command=command
        result=finish.fusion(a,root/'champion',root/'labels','test')
        assert result['selected_oof_miou']==1. and result['locked_val_miou']==1.
        assert finish.useful(result) and all(p['alpha']==.5 for p in result['deploy'])
        assert np.allclose(result['descriptive_paired_bootstrap_delta_ci95'],[.5,.5])
    print('Nomination, cross-fit, locked vote, lossless cache and paired bootstrap fixture passed.')

if __name__=='__main__':main()
