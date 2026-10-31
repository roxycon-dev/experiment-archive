"""Bounded inference diagnosis on three fixed TRAIN crops; never selects a model."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import make_augment_factory
from trainer.checkpoint import load_checkpoint
from trainer.config import load_yaml, ddr_data_config
from trainer.data.ddr_dataset import DDRSegDataset
from trainer.models import build_model
from trainer.eval import sliding_window_logits
from trainer.data.common import normalize_image


def main():
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(0.12)
    cfg = load_yaml(root/'configs/resolved/A0_hacdr_adapted.yaml')
    dc = ddr_data_config(cfg)
    ckpt = load_checkpoint(str(root/'work_dirs/A0_ddr_finetune/ckpt_last.pth'))
    update = ckpt['update']
    model, _ = build_model(cfg, stage='ddr_finetune', gradient_checkpointing=False)
    initial = model.state_dict()
    changes = {}
    for name in ('backbone.patch_embed1.proj.0.weight', 'decoder.cls.weight'):
        if name in initial:
            changes[name] = float((initial[name] - ckpt['model_state'][name]).abs().mean())
    model.load_state_dict(ckpt['model_state'], strict=True)
    del ckpt, initial
    bn = [(name, m) for name, m in model.named_modules()
          if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    bn_stats = [{'name': name, 'batches': int(m.num_batches_tracked),
                 'min_running_var': float(m.running_var.min()),
                 'max_running_var': float(m.running_var.max()),
                 'max_abs_running_mean': float(m.running_mean.abs().max())}
                for name, m in bn]
    model.to('cuda').eval()
    aug = make_augment_factory(cfg['ddr_finetune_stage'], train_crop=dc['train_crop'])(dc['train_crop'])
    ds = DDRSegDataset(dc['ddr_root'], 'train', mean=dc['mean'], std=dc['std'], augment=aug)
    records = []
    started = time.monotonic()
    for idx in (0, 1, 2):
        sample = ds[idx]
        x = sample['image'][None].to('cuda')
        target = sample['target'].to('cuda').bool()
        valid = sample['valid'].to('cuda').bool()
        raw = x[0] * x.new_tensor(dc['std'])[:, None, None] + x.new_tensor(dc['mean'])[:, None, None]
        dark = raw.mean(0) < 20
        runs = {}
        base = None
        for mode in ('eval_fp32', 'eval_amp', 'batch_stats_fp32'):
            model.eval()
            if mode == 'batch_stats_fp32':
                # A separate in-memory diagnostic, with dropout still disabled.
                # track_running_stats=False prevents any buffer mutation.
                for _, m in bn:
                    m.train()
                    m.track_running_stats = False
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=mode == 'eval_amp'):
                logits = model(x)[0].float()
            for _, m in bn:
                m.track_running_stats = True
            pred = logits > 0
            classes = []
            for c in range(4):
                p, t, v = pred[c], target[c], valid[c]
                tp = int((p & t & v).sum())
                fp = int((p & ~t & v).sum())
                fn = int((~p & t & v).sum())
                classes.append(dict(tp=tp, fp=fp, fn=fn, iou=tp/max(1, tp+fp+fn),
                                    positives=int((p & v).sum()),
                                    fp_in_dark=int((p & ~t & v & dark).sum())))
            runs[mode] = dict(finite=bool(torch.isfinite(logits).all()), classes=classes,
                             min_logit=float(logits.min()), max_logit=float(logits.max()))
            if base is None:
                base = logits.clone()
            else:
                runs[mode]['mean_abs_delta_from_eval_fp32'] = float((logits-base).abs().mean())
                runs[mode]['prediction_disagreement_from_eval_fp32'] = float(((base>0) != pred).float().mean())
            del logits, pred
        records.append(dict(sample_id=sample['meta']['sample_id'], split='train',
                            target_pixels=[int(t.sum()) for t in target], runs=runs))
        del x, target, valid, raw, dark, base
    # The first three sorted images have almost no lesions. Check two additional
    # images nominated by TRAIN labels alone, using the actual sliding geometry.
    full_fov = []
    used = set()
    for required_class in (1, 2):
        chosen = None
        for idx in range(min(80, len(ds))):
            if idx in used:
                continue
            rgb, target_np, _, sid = ds.load_raw(idx)
            if int(target_np[required_class].sum()) >= 100:
                chosen = (idx, rgb, target_np, sid)
                break
        if chosen is None:
            continue
        idx, rgb, target_np, sid = chosen
        used.add(idx)
        rgb = cv2.resize(rgb, (2048, 2048), interpolation=cv2.INTER_LINEAR)
        target = torch.from_numpy(np.stack([cv2.resize(t, (2048,2048), interpolation=cv2.INTER_NEAREST)
                                           for t in target_np])).to('cuda').bool()
        x = torch.from_numpy(normalize_image(rgb, dc['mean'], dc['std']).transpose(2,0,1).copy())[None].to('cuda')
        dark = torch.from_numpy(rgb.mean(2) < 20).to('cuda')
        runs = {}
        for mode in ('eval_fp32', 'batch_stats_fp32'):
            model.eval()
            if mode == 'batch_stats_fp32':
                for _, m in bn:
                    m.train(); m.track_running_stats = False
            with torch.no_grad():
                logits = sliding_window_logits(model, x, 1024, 768)[0]
            for _, m in bn:
                m.track_running_stats = True
            pred = logits > 0
            classes = []
            for c in range(4):
                p, t = pred[c], target[c]
                tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
                classes.append(dict(tp=tp, fp=fp, fn=fn, iou=tp/max(1,tp+fp+fn),
                                    fp_in_dark=int((p & ~t & dark).sum())))
            runs[mode] = dict(classes=classes, finite=bool(torch.isfinite(logits).all()),
                             max_logit=float(logits.max()), min_logit=float(logits.min()))
            del logits, pred
        full_fov.append(dict(sample_id=sid, train_index=idx, nomination_class=required_class,
                             target_pixels=[int(t.sum()) for t in target], runs=runs))
        del x, target, dark
    result = dict(purpose='engineering_diagnosis_only_not_validation_or_selection',
                  checkpoint_update=update, checkpoint='ckpt_last.pth',
                  sampled_indices=[0,1,2], parameter_changes_from_seed42=changes,
                  bn_stats=bn_stats, samples=records, full_fov_train_probes=full_fov,
                  seconds=time.monotonic()-started,
                  peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2)
    path = root/'reports/A0_train_diagnosis.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'bn_stats'}), flush=True)


if __name__ == '__main__':
    main()
