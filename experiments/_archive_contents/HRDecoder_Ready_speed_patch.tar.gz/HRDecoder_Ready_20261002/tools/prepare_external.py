"""Prepare the user's curated IDRiD or original-resolution MAPLES release.

IDRiD polygon masks are explicitly labelled as derived annotations. They are
usable exploratory supervision, never asserted to be the untouched official GT.
Only the supplied training split is exported; 27 testing images remain unused.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_external_manifest import record_for_sample
from contract_common import compute_perceptual_hashes, sha256_file, write_jsonl_atomic

ORDER = ('EX','HE','SE','MA')
POLYGON_MAP = {0:'MA', 1:'HE', 2:'EX', 3:'SE'}  # D:/dataset/AGENTS.md
MAPLES_MAP = {'EX':'Exudates', 'HE':'Hemorrhages',
              'SE':'CottonWoolSpots', 'MA':'Microaneurysms'}

def entry(path, base, positive):
    return dict(annotated=True, valid_mode='full', mask_path=path.relative_to(base).as_posix(),
                mask_sha256=sha256_file(str(path)), valid_mask_path=None, positive_pixels=positive)

def curated_idrid(root, output):
    records = []
    images = sorted((root/'images/training').glob('*.jpg'))
    if len(images) != 54:
        raise ValueError(f'Expected 54 curated training images, got {len(images)}')
    for image in images:
        w,h = Image.open(image).size
        label = root/'labels/training'/f'{image.stem}.txt'
        polygons = {c:[] for c in ORDER}
        for line in label.read_text().splitlines():
            row = line.split()
            if not row:
                continue
            values = np.asarray(list(map(float,row[1:])))
            if len(values)<6 or len(values)%2 or np.any(values<0) or np.any(values>1):
                raise ValueError(f'Invalid polygon in {label}')
            cls = POLYGON_MAP[int(row[0])]
            xy = np.rint(values.reshape(-1,2)*[w,h]).astype(np.int32)
            xy[:,0] = xy[:,0].clip(0,w-1)
            xy[:,1] = xy[:,1].clip(0,h-1)
            polygons[cls].append(xy)
        target_image = output/'IDRiD/images/train'/image.name
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target_image)
        lesions = {}
        for cls in ORDER:
            mask = np.zeros((h,w), dtype=np.uint8)
            if polygons[cls]:
                cv2.fillPoly(mask, polygons[cls], 255)
            path = output/'IDRiD/labels/train'/cls/f'{image.stem}_{cls}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(path)
            lesions[cls] = entry(path, output, int(np.count_nonzero(mask)))
        records.append(record_for_sample(
            source='IDRiD', source_version='user_curated_best_segmentation_20261002',
            official_split='train', image_id=image.stem, patient_id=None, patient_id_source=None,
            image_path=target_image.relative_to(output).as_posix(), image_sha=sha256_file(str(target_image)),
            width=w,height=h,lesions=lesions,phashes=compute_perceptual_hashes(str(target_image)),
            notes=['annotation_quality=polygon_rasterized_not_original_official_pixel_masks',
                   'mapping=0:MA,1:HE,2:EX,3:SE; source=D:/dataset/AGENTS.md',
                   'all_four_classes_covered_by_user_curated_label_format',
                   'original_dimensions_means_user_supplied_processed_image_dimensions',
                   'round(normalized_xy*[width,height]); independent masks retain overlaps',
                   f'polygon_label_sha256={sha256_file(str(label))}']))
        print(f'IDRiD {len(records)}/54', flush=True)
    return records

def maples(root, images_root, output):
    records = []
    # Native Figshare MAPLES-DR.zip is train/<biomarker>/<stem>.png.
    image_files = {p.stem:p for p in images_root.rglob('*')
                   if p.suffix.lower() in ['.tif','.tiff','.jpg','.jpeg','.png']}
    ids = sorted(p.stem for p in (root/'train/Exudates').glob('*.png'))
    if len(ids) != 138:
        raise ValueError(f'MAPLES official train must contain 138, got {len(ids)}')
    for sid in ids:
        if sid not in image_files:
            raise FileNotFoundError(f'MESSIDOR fundus missing: {sid}')
        image = image_files[sid]
        rgb = Image.open(image).convert('RGB')
        w,h = rgb.size
        target = output/'MAPLES-DR/images/train'/f'{sid}.png'
        target.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(target)
        lesions = {}
        for cls,folder in MAPLES_MAP.items():
            source = root/'train'/folder/f'{sid}.png'
            mask = np.asarray(Image.open(source))
            if mask.shape != (h,w) or not set(np.unique(mask)).issubset({0,1,255}):
                raise ValueError(f'Native MAPLES mask/image geometry or encoding mismatch: {source}')
            path = output/'MAPLES-DR/labels/train'/cls/f'{sid}_{cls}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source,path)
            lesions[cls] = entry(path,output,int(np.count_nonzero(mask)))
            uncertain = root/'train'/('BrightUncertains' if cls in ('EX','SE') else 'RedUncertains')/f'{sid}.png'
            if uncertain.exists():
                unknown = np.asarray(Image.open(uncertain))
                if unknown.shape != (h,w):
                    raise ValueError(f'Uncertain-region geometry mismatch: {uncertain}')
                validity = output/'MAPLES-DR/valid/train'/f'{sid}_{cls}.png'
                validity.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(np.where(unknown>0,0,255).astype(np.uint8)).save(validity)
                lesions[cls]['valid_mode'] = 'file'
                lesions[cls]['valid_mask_path'] = validity.relative_to(output).as_posix()
        records.append(record_for_sample(
            source='MAPLES-DR',source_version='figshare_24328660_v3_file45795381',official_split='train',
            image_id=sid,patient_id=None,patient_id_source=None,image_path=target.relative_to(output).as_posix(),
            image_sha=sha256_file(str(target)),width=w,height=h,lesions=lesions,
            phashes=compute_perceptual_hashes(str(target)),notes=['Native original-resolution Figshare release',
                  f'MESSIDOR_original_sha256={sha256_file(str(image))}']))
    return records

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idrid-curated', type=Path)
    ap.add_argument('--maples-labels', type=Path)
    ap.add_argument('--messidor-images', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    output = args.output.resolve()
    records = []
    if args.idrid_curated:
        records += curated_idrid(args.idrid_curated, output)
    if args.maples_labels:
        if not args.messidor_images:
            raise ValueError('MAPLES labels require real MESSIDOR fundus images')
        records += maples(args.maples_labels, args.messidor_images,output)
    if not records:
        raise ValueError('No dataset supplied')
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'external_train.jsonl'
    for rec in records:
        if isinstance(rec.get('notes'),list):
            rec['notes']='; '.join(rec['notes'])
    write_jsonl_atomic(str(manifest),records)
    (output/'PROVENANCE.json').write_text(json.dumps(dict(
        n_train=len(records), n_test_used=0, channel_order=ORDER,
        curated_idrid=args.idrid_curated is not None,
        maples_available=args.maples_labels is not None),indent=2),encoding='utf-8')
    print(manifest)

if __name__ == '__main__':
    main()
