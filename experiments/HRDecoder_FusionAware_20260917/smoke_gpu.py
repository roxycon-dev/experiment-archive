# -*- coding: utf-8 -*-
"""Server-side GPU smoke test.  Run with CWD = glm dir:

    /root/miniconda3/bin/python smoke_gpu.py

Five gates (all must pass before the guarded launch):
  1. student build: strict D@2000 student + B reference, authorized
     trainable set only (decode_head + backbone.stage4)
  2. one REAL B-arm training step on one real train image with a fake
     2048 reference, plus first_update_evidence (frozen params / BN / grads)
  3. champion ensemble build: 4 members + MA head, strict loads
  4. one champion predict on one real val image through the exact test
     pipeline (flip TTA + head), shape/finite/range contract
  5. D_full embedded reference == B (proves student_only+B == full for the
     champion D member)
Exits nonzero on any failure."""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

import runtime as rt                                          # noqa: E402
import run_protocol as rp                                     # noqa: E402

D_FULL = '/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth'


def main():
    with open('settings.json', encoding='utf-8') as f:
        S = json.load(f)
    repo = S['repo']

    def a_(rel):
        return rel if os.path.isabs(rel) else os.path.join(repo, rel)

    settings = dict(S)
    settings['_abs'] = {
        'reference_checkpoint': a_(S['reference_checkpoint']),
        'student_checkpoint': a_(S['student_checkpoint']),
    }
    settings['_hashes'] = {'reference_checkpoint': 'smoke'}

    # same environment as run_protocol.main(): repo injected (CWD + sys.path)
    # so config custom_imports ('hrplus_patch' etc.) resolve
    rt.inject_repo(repo)

    t0 = time.time()
    assert torch.cuda.is_available(), 'CUDA unavailable'
    print('[smoke] cuda ok: %s (%.1fs)' %
          (torch.cuda.get_device_name(0), time.time() - t0), flush=True)

    # ---- gate 1: student build + authorized trainable set ----
    model, cfg = rp.build_student(settings)
    auth = [n for n, p in model.named_parameters() if p.requires_grad]
    assert auth, 'no trainable parameters'
    bad = [n for n in auth
           if not n.startswith(('decode_head.', 'backbone.stage4.'))]
    assert not bad, 'unauthorized trainable params: %s' % bad[:5]
    assert not any(n.startswith('reference.') for n in auth)
    print('[smoke] gate1 student ok: %d trainable tensors, %.1fM params' %
          (len(auth), sum(p.numel() for p in model.parameters()
                          if p.requires_grad) / 1e6), flush=True)

    # ---- gate 2: one real B-arm step + first-update evidence ----
    snap, bn = rt.snapshot_state(model), rt.snapshot_buffers(model)
    model.train()
    rt.freeze_bn(model)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=5e-5, momentum=0.9, weight_decay=5e-4)
    data_root = S['data_root']
    itr = os.scandir(os.path.join(data_root, 'images', 'train'))
    stems = sorted(e.name for e in itr
                   if e.name.lower().endswith(('.jpg', '.jpeg')))
    img, lab = rt.load_image_label(data_root, 'train',
                                   os.path.splitext(stems[0])[0])
    rng = np.random.RandomState(0)
    ref = rng.rand(2048, 2048, 4).astype(np.float32)
    im2, la2, rf2, tr = rt.paired_geometry(img, lab, ref, rng)
    assert im2.shape == (3, 1024, 1024) and la2.shape == (1, 1024, 1024) \
        and rf2.shape == (4, 1024, 1024), \
        'geometry shapes %r %r %r' % (im2.shape, la2.shape, rf2.shape)
    logits = model.encode_decode(torch.from_numpy(im2).unsqueeze(0).cuda(),
                                 [{}])
    loss = rt.loss_pair(logits,
                        torch.from_numpy(la2).unsqueeze(0).cuda(),
                        torch.from_numpy(rf2).unsqueeze(0).cuda(), 'B')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 5.0)
    opt.step()
    rt.freeze_bn(model)
    ev = rt.first_update_evidence(model, snap, bn)
    assert all(ev.values()), 'first update evidence failed: %s' % ev
    print('[smoke] gate2 B-arm step ok: loss %.4f, evidence pass (%.1fs)' %
          (float(loss), time.time() - t0), flush=True)
    del model, opt, snap, bn
    torch.cuda.empty_cache()

    # ---- gate 3: champion build (4 members + MA head) ----
    champ = rt.ChampionEnsemble(settings, settings['champion'])
    names = [spec['name'] for _, _, spec in champ.members]
    assert sorted(names) == ['B', 'D', 'E', 'E4'], names
    print('[smoke] gate3 champion built: members %s + head %s' %
          (names, sorted(champ.heads)), flush=True)

    # ---- gate 4: one champion predict on one real val image ----
    import copy
    from mmseg.datasets import build_dataset
    d = copy.deepcopy(cfg.data['val'])
    d.test_mode = True
    d.img_dir = os.path.join(data_root, 'images', 'val')
    d.ann_dir = os.path.join(data_root, 'labels', 'val')
    ds = build_dataset(d)
    assert len(ds) == 76, 'val split size %d' % len(ds)
    s = ds[0]
    im = rt._unwrap_img(s['img']).unsqueeze(0).cuda()
    meta = rt._unwrap_meta(s['img_metas'])
    t1 = time.time()
    prob = champ.predict(im, meta)
    ori = tuple(meta[0]['ori_shape'][:2])
    assert tuple(prob.shape) == (4, ori[0], ori[1]), \
        'prob shape %s vs ori %s' % (tuple(prob.shape), ori)
    assert torch.isfinite(prob).all() and float(prob.min()) >= 0.0 \
        and float(prob.max()) <= 1.0
    print('[smoke] gate4 champion predict ok: shape %s, %.1fs/img' %
          (tuple(prob.shape), time.time() - t1), flush=True)
    del champ, ds
    torch.cuda.empty_cache()

    # ---- gate 5: D_full embedded reference == B ----
    bck = torch.load(a_(S['reference_checkpoint']), map_location='cpu')
    b_sd = bck.get('state_dict', bck)
    dck = torch.load(D_FULL, map_location='cpu')
    d_sd = dck.get('state_dict', dck)
    emb = {k[len('reference.'):]: v for k, v in d_sd.items()
           if k.startswith('reference.')}
    keys_ok = set(emb) == set(b_sd)
    vals_ok = keys_ok and all(torch.equal(emb[k], b_sd[k]) for k in emb)
    print('[smoke] gate5 D_full embedded reference == B: keys %s values %s '
          '(%d tensors)' % (keys_ok, vals_ok, len(emb)), flush=True)
    assert keys_ok and vals_ok, \
        'D embedded reference differs from B; switch D member to full mode'

    print('[smoke] ALL GATES PASS (%.1fs total)' % (time.time() - t0),
          flush=True)


if __name__ == '__main__':
    main()
