"""CPU-only smoke of the unchanged legacy image pipeline on fixed holdout IDs."""
import copy
import json
from pathlib import Path
import sys
import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
READY = EXP/'HRDecoder_Ready_20261002'


def main():
    cv2.setNumThreads(1); torch.set_num_threads(2)
    sys.path.insert(0,str(READY/'legacy'))
    sys.path.append(str(EXP/'HRDecoder_ArchitectureDesign_20261008/training_execution/python_deps'))
    import runtime
    repo = READY/'vendor/champion_repo'; runtime.inject_repo(str(repo))
    cfg = runtime.load_config(str(repo),'configs/lesion/exp/hrplus_patch_dice.py')
    from mmseg.datasets import build_dataset
    d = copy.deepcopy(cfg.data['val']); d.data_root = 'D:/dataset/best_segmentation'
    d.img_dir = 'images/testing';d.ann_dir = 'labels/testing'
    d.pipeline = copy.deepcopy(cfg.data['test']['pipeline']); d.test_mode = True
    if 'split' in d:d.split=None
    dataset = build_dataset(d)
    by_id = {Path(r['filename']).stem:i for i,r in enumerate(dataset.img_infos)}
    cohort = json.loads((ROOT/'evidence/external_holdout_manifest.json').read_text(encoding='utf-8'))
    assert set(by_id) == {r['sid'] for r in cohort['samples']}
    rows = {r['sid']:r for r in cohort['samples']}
    selected = ['IDRiD_55','IDRiD_66','IDRiD_81']  # first, maximum stored size, last; not outcome-selected
    results = []
    for sid in selected:
        item = dataset[by_id[sid]]
        x = runtime._unwrap_img(item['img']); meta = runtime._unwrap_meta(item['img_metas'])
        assert list(meta[0]['ori_shape'][:2]) == rows[sid]['hw']
        assert x.dtype == torch.float32 and not x.is_cuda and torch.isfinite(x).all()
        results.append({'sid':sid,'tensor_shape':list(x.shape),'original_shape':list(meta[0]['ori_shape']),
                        'image_shape':list(meta[0]['img_shape']),'flip':meta[0].get('flip'),
                        'normalization':{k:v.tolist() if hasattr(v,'tolist') else v for k,v in meta[0]['img_norm_cfg'].items()}})
    result = {'status':'PASS','cohort_images':len(by_id),'checked':results,'models_loaded':False,
              'CUDA_initialized':torch.cuda.is_initialized(),'ground_truth_in_test_pipeline':False,
              'pipeline':str(d.pipeline)}
    (ROOT/'evidence/external_preprocessing_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','checked':selected}),flush=True)


if __name__=='__main__':main()
