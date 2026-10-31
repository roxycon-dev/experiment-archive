"""Conditional E asset gate. Independent pixel masks and explicit unknown regions only."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from nextlib.common import CLASSES,read_json,write_json,sha,digest


def audit(manifest,out):
    path=Path(manifest).resolve();rows=read_json(path)
    if not isinstance(rows,list) or not rows:raise ValueError('Official E manifest must be a nonempty list')
    accepted=[]
    for row in rows:
        if row.get('source_kind') not in ('official_IDRiD_pixels','official_MAPLES_pixels'):
            raise ValueError('E gate rejects curated polygons and unofficial pixel-label substitutes')
        if row.get('split')!='train' or not row.get('provenance_url'):
            raise ValueError('Official training split and provenance are required')
        def resolve(p):
            p=Path(p);return p if p.is_absolute() else path.parent/p
        image=resolve(row['image'])
        if image.suffix.lower() not in ('.jpg','.jpeg','.png','.tif','.tiff'):
            raise ValueError('Unexpected image asset')
        with Image.open(image) as im:hw=(im.height,im.width)
        entry=dict(sid=row['sid'],source_kind=row['source_kind'],image_sha256=sha(image),hw=list(hw),classes={})
        if not row.get('original_image_geometry_verified'):
            raise ValueError('Curated square images must not be declared official untransformed originals')
        for c in CLASSES:
            spec=row['classes'][c]
            if spec.get('status')=='unknown':
                entry['classes'][c]=dict(valid=0,note='Entire class ignored, never a negative mask')
                continue
            if spec.get('status') not in ('pixel_mask','official_verified_absent'):
                raise ValueError('Missing mask is not automatically a negative label')
            if spec['status']=='official_verified_absent':
                if not spec.get('evidence'):raise ValueError('Official absence declaration needs evidence')
                entry['classes'][c]=dict(valid=1,positive_pixels=0,evidence=spec['evidence'])
                continue
            mask=resolve(spec['mask'])
            if mask.suffix.lower() not in ('.png','.tif','.tiff'):raise ValueError('Pixel-mask format required')
            with Image.open(mask) as im:arr=np.array(im)
            if arr.ndim!=2 or arr.shape!=hw:raise ValueError('Official image/mask coordinate mismatch')
            valid=np.ones(hw,bool)
            if spec.get('ignore_mask'):
                ignore=resolve(spec['ignore_mask'])
                with Image.open(ignore) as im:unknown=np.array(im)
                if unknown.shape!=hw:raise ValueError('Unknown-region coordinate mismatch')
                valid=unknown==0
            entry['classes'][c]=dict(valid=1,mask_sha256=sha(mask),positive_pixels=int(((arr>0)&valid).sum()),
                                    ignored_pixels=int((~valid).sum()))
        accepted.append(entry)
    result=dict(status='ASSET_AUDIT_PASS',manifest_sha256=sha(path),samples=accepted,
                content_sha256=digest(accepted),note='Source declarations still require provenance review and DDR duplicate exclusion before E training')
    write_json(out,result);return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();print(audit(a.manifest,a.out)['status'])
