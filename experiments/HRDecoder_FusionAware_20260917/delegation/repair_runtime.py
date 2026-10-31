# delegation/repair_runtime.py
# Bounded one-shot patch script. Run from D:/dataset/experiment/HRDecoder_FusionAware_20260917
# Patches ONLY: runtime.py, run_protocol.py. Exact replacements, counts asserted.
from pathlib import Path


def patch(path, old, new, count=1):
    txt = path.read_text(encoding='utf-8')
    n = txt.count(old)
    assert n == count, '%s: anchor found %d times (expected %d): %r' % (
        path.name, n, count, old[:80])
    path.write_text(txt.replace(old, new), encoding='utf-8')


def replace_between(path, start, end, new_block):
    txt = path.read_text(encoding='utf-8')
    i = txt.index(start)
    j = txt.index(end, i)
    path.write_text(txt[:i] + new_block + txt[j:], encoding='utf-8')


rt = Path('runtime.py')
rp = Path('run_protocol.py')

# =============== runtime.py ===============

# R1: ChampionEnsemble -- only set source_checkpoint when the member config
# already has the field OR the model type is HRPlusRefinement; B/E4 configs
# get nothing injected; E-style unknown paths must be explicit, never guessed.
patch(rt,
"""            c.model['source_checkpoint'] = spec.get(
                'source_checkpoint', settings['reference_checkpoint'])
""",
"""            mtype = str(c.model.get('type', ''))
            if 'source_checkpoint' in c.model or mtype == 'HRPlusRefinement':
                c.model['source_checkpoint'] = spec.get(
                    'source_checkpoint', settings['reference_checkpoint'])
            elif 'source_checkpoint' in spec:
                raise RuntimeError(
                    'champion member %s: source_checkpoint given but config '
                    'model has no such field (not HRPlusRefinement)' % name)
""")

# R2: ReferenceCache mapping protocol used by the driver
# (cache[stem] and `stem in cache`).
patch(rt,
"""    # -- build --------------------------------------------------------------
    def build(self, model, image_ids, deadline, log):""",
"""    def __getitem__(self, stem):
        return self.get(stem)

    def __contains__(self, stem):
        if self._manifest is None:
            if not os.path.isfile(self.manifest_path):
                return False
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                self._manifest = json.load(f)
        return stem in self._manifest.get('entries', {})

    # -- build --------------------------------------------------------------
    def build(self, model, image_ids, deadline, log):""")

# R3: rt.load_image_label -- direct-dir RGB jpg read + label png
# shape/value validation. Inserted just above ReferenceCache.
patch(rt,
"class ReferenceCache(object):",
'''def load_image_label(data_root, split, stem):
    """Read RGB jpg from images/<split>/<stem>.jpg|jpeg directly (no glob
    guessing) and the label png; validate shape match and values 0..4/255.
    Returns (img_rgb_uint8_HxWx3, label_int64_HxW)."""
    import cv2
    img_path = None
    for ext in ('.jpg', '.jpeg'):
        p = os.path.join(data_root, 'images', split, stem + ext)
        if os.path.isfile(p):
            img_path = p
            break
    if img_path is None:
        raise RuntimeError('no jpg image for stem %r in %s/train' %
                           (stem, split))
    img = cv2.imread(img_path, 1)
    if img is None:
        raise RuntimeError('unreadable image: %s' % img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab_path = os.path.join(data_root, 'labels', split, stem + '.png')
    lab = cv2.imread(lab_path, 0)
    if lab is None:
        raise RuntimeError('unreadable label: %s' % lab_path)
    if lab.shape[:2] != img.shape[:2]:
        raise RuntimeError('label/image shape mismatch: %s' % stem)
    if not np.all(np.isin(np.unique(lab), (0, 1, 2, 3, 4, 255))):
        raise RuntimeError('label values outside 0..4/255: %s' % lab_path)
    return img, lab.astype(np.int64)


class ReferenceCache(object):''')

# R4: audit_split -- keyed 3 splits + _ids_sha256; intersection check.
replace_between(rt,
    'def audit_split(data_root, expected=(454, 76, 227)):',
    '\n\n# ---------------------------------------------------------------------------\n# deadline',
'''def audit_split(data_root, expected=(454, 76, 227)):
    """Verify per-split counts, txt/dir agreement, label png agreement, no
    duplicates, no cross-split intersection.  Returns dict keyed
    'train'/'val'/'test' -> sorted stem lists plus '_ids_sha256' digest over
    the full ordered (split, stem) ID set."""
    ids = {}
    for si, split in enumerate(('train', 'val', 'test')):
        img_dir = os.path.join(data_root, 'images', split)
        lab_dir = os.path.join(data_root, 'labels', split)
        list_file = os.path.join(data_root, split + '.txt')
        imgs = sorted(f for f in os.listdir(img_dir)
                      if f.lower().endswith(('.jpg', '.jpeg')))
        if len(imgs) != expected[si]:
            raise RuntimeError('split %s has %d images, expected %d' %
                               (split, len(imgs), expected[si]))
        if not os.path.isfile(list_file):
            raise RuntimeError('missing split list %s' % list_file)
        listed = [l.strip() for l in open(list_file, encoding='utf-8')
                  if l.strip()]
        if sorted(os.path.splitext(x)[0] for x in listed) != \\
                sorted(os.path.splitext(x)[0] for x in imgs):
            raise RuntimeError('%s does not match %s images' %
                               (list_file, split))
        stems = [os.path.splitext(f)[0] for f in imgs]
        if len(set(stems)) != len(stems):
            raise RuntimeError('duplicate stem in %s' % split)
        labs = set(os.path.splitext(x)[0] for x in os.listdir(lab_dir)
                   if x.endswith('.png'))
        if split == 'test':
            if labs:
                raise RuntimeError('test labels present -- forbidden')
        else:
            if labs != set(stems):
                raise RuntimeError('labels/%s mismatch' % split)
        ids[split] = stems
    all_ids = ids['train'] + ids['val'] + ids['test']
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError('intersection across splits detected')
    h = hashlib.sha256()
    for s in ('train', 'val', 'test'):
        for x in ids[s]:
            h.update(('%s|%s\\n' % (s, x)).encode('utf-8'))
    ids['_ids_sha256'] = h.hexdigest()
    return ids

''')

# =============== run_protocol.py ===============

# P1: build_student -- constructor source is the REFERENCE checkpoint (B),
# then strict_initialize loads the student weights (D2000).  Also publish
# settings['_cfg'] (evaluate_split / evaluate_candidates rely on it).
patch(rp,
"    cfg.model['source_checkpoint'] = settings['_abs']['student_checkpoint']",
"    cfg.model['source_checkpoint'] = settings['_abs']['reference_checkpoint']")
patch(rp,
"    rt.configure_trainable(model)\n    return model.cuda(), cfg",
"    rt.configure_trainable(model)\n    settings['_cfg'] = cfg\n    return model.cuda(), cfg")

# P2: fingerprint must cover reference+student checkpoint hashes AND the
# actual split IDs (source .py files were already covered).
patch(rp,
"""    blob['source_files'] = {}
    for p in sorted(srcs):
        blob['source_files'][p] = sha256(p)
    return blob""",
"""    blob['source_files'] = {}
    for p in sorted(srcs):
        blob['source_files'][p] = sha256(p)
    blob['checkpoint_hashes'] = {
        k: v for k, v in settings['_hashes'].items() if 'checkpoint' in k}
    blob['split_ids_sha256'] = rt.audit_split(
        settings['data_root'])['_ids_sha256']
    return blob""")

# P3: evaluate_candidates -- remove the dead `if False` branch, enforce
# val=76 / test=227, save per-image AND global counts for all three names
# under dump_prefix.
replace_between(rp,
    'def evaluate_candidates(settings, champion, new_model, split, deadline, log,',
    '\n\n# ---------------------------------------------------------------------------\n# timing pilot',
'''def evaluate_candidates(settings, champion, new_model, split, deadline, log,
                        dump_prefix=None):
    """Shared dataset loop: champion/new/mix probs computed once per image;
    true per-image and global counts kept for all three names; enforced
    expected counts; per-name count JSONs written under dump_prefix."""
    import runtime as rt
    import copy as _c
    import cv2
    from mmseg.datasets import build_dataset
    cfg = settings['_cfg']
    d = _c.deepcopy(cfg.data[split])
    d.test_mode = True
    d.img_dir = os.path.join(settings['data_root'], 'images', split)
    d.ann_dir = os.path.join(settings['data_root'], 'labels', split)
    ds = build_dataset(d)
    expected = {'val': 76, 'test': 227}[split]
    if len(ds) != expected:
        raise RuntimeError('split %s has %d items, expected %d' %
                           (split, len(ds), expected))
    gt_maps = ds.get_gt_seg_maps()
    totals = {'champion': [[0, 0, 0] for _ in range(4)],
              'new': [[0, 0, 0] for _ in range(4)],
              'mix': [[0, 0, 0] for _ in range(4)]}
    per_image = {k: {} for k in totals}
    for i in range(len(ds)):
        if time.time() > deadline:
            raise ShutdownGraceful('deadline during %s candidate eval' % split)
        s = ds[i]
        img = rt._unwrap_img(s['img']).unsqueeze(0).cuda()
        meta = rt._unwrap_meta(s['img_metas'])
        with torch.no_grad():
            pc = champion.predict(img, meta)
            pn = d_inf(new_model, img, meta).detach()
            pm = rt.mix_probs(pc, pn)
        gt = np.asarray(gt_maps[i])
        ori = tuple(meta[0]['ori_shape'][:2])
        if gt.shape != ori:
            gt = cv2.resize(gt.astype(np.uint8), (ori[1], ori[0]),
                            interpolation=cv2.INTER_NEAREST)
        stem = os.path.splitext(os.path.basename(meta[0]['ori_filename']))[0]
        for name, p in (('champion', pc), ('new', pn), ('mix', pm)):
            cnt = rt.per_image_counts(
                p.detach().cpu().numpy().astype(np.float32),
                gt.astype(np.int64))
            per_image[name][stem] = {rt.CLASSES[c]: cnt[c]
                                     for c in range(4)}
            for c in range(4):
                for j in range(3):
                    totals[name][c][j] += cnt[c][j]
    out = {}
    for name in totals:
        miou, per = rt.global_miou([tuple(x) for x in totals[name]])
        out[name] = {'miou': miou, 'per_class': per,
                     'global_counts': totals[name],
                     'per_image': per_image[name]}
        log('final-candidates %s %s mIoU %.4f' % (split, name, miou))
        if dump_prefix:
            rt.atomic_json('%s_%s_counts.json' % (dump_prefix, name),
                           out[name])
    return out

''')

# P4: every quarter -- AFTER val (and RNG restore) also refresh last.pth
# with the new history so the persisted last state is never pre-val/unsafe.
patch(rp,
"""            save_checkpoint(os.path.join(ckpt_dir, 'iter_%d.pth' % (it + 1)),
                            model, opt, it + 1, seed, T, arm, settings,
                            split_hashes, geom_rng, order_rng, hist)  # AFTER val
""",
"""            save_checkpoint(os.path.join(ckpt_dir, 'iter_%d.pth' % (it + 1)),
                            model, opt, it + 1, seed, T, arm, settings,
                            split_hashes, geom_rng, order_rng, hist)  # AFTER val
            save_checkpoint(last_path, model, opt, it + 1, seed, T, arm,
                            settings, split_hashes, geom_rng, order_rng,
                            hist)  # last state AFTER val, with history/RNG
""")

# P5: pilot -- explicit cuda placement check.
patch(rp,
"""        model, _ = build_student(settings)
        model.train()
        rt.freeze_bn(model)""",
"""        model, _ = build_student(settings)
        assert next(model.parameters()).is_cuda, 'pilot model not on cuda'
        model.train()
        rt.freeze_bn(model)""")
patch(rp,
"""        model, _ = build_student(settings)
        model.eval()
        t0 = time.time()""",
"""        model, _ = build_student(settings)
        assert next(model.parameters()).is_cuda, 'pilot eval model not on cuda'
        model.eval()
        t0 = time.time()""")

# P6: main consumes the new audit_split shape (splits + _ids_sha256).
patch(rp,
"""    assignment = rt.audit_split(data_root)
    split_hashes = {'config': settings['_hashes']['config'],
                    'split': rt._arr_sha(
                        np.array(sorted(assignment.keys())))}""",
"""    assignment = rt.audit_split(data_root)
    split_hashes = {'config': settings['_hashes']['config'],
                    'split': assignment['_ids_sha256']}""")
patch(rp,
"""    audit_labels(data_root, 'train', [f for f, v in assignment.items()
                                      if v['split'] == 'train'], log)""",
"    audit_labels(data_root, 'train', assignment['train'], log)")

print('patch OK: runtime.py + run_protocol.py updated (verify with --dry-run)')
