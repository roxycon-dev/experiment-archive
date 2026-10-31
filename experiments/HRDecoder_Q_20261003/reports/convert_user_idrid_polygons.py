"""Export user-provided polygons as independent pixel masks, without claiming official GT."""
from pathlib import Path
import json
import sys
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qlib.common import sha,write_json,utc_now

SOURCE=Path('D:/dataset/best_segmentation')
OUTPUT=ROOT/'assets/external_derived/IDRiD_best_segmentation_pixel_masks'
CLASSES=('EX','HE','SE','MA')
MAPPING={0:'MA',1:'HE',2:'EX',3:'SE'}
cv2.setNumThreads(1)
rows=[]
for split,count in [('training',54),('testing',27)]:
    images=sorted((SOURCE/'images'/split).glob('IDRiD_*.jpg'))
    assert len(images)==count
    for image in images:
        with Image.open(image) as im: w,h=im.size
        text=SOURCE/'labels'/split/(image.stem+'.txt')
        before=sha(text)
        polygons={c:[] for c in CLASSES}
        for line_number,line in enumerate(text.read_text(encoding='utf-8').splitlines(),1):
            values=line.split()
            if not values: continue
            raw_class=float(values[0]);class_id=int(raw_class)
            if raw_class!=class_id or class_id not in MAPPING:
                raise ValueError(f'Invalid class in {text}:{line_number}')
            xy=np.asarray([float(v) for v in values[1:]],np.float64)
            if len(xy)<6 or len(xy)%2 or not np.isfinite(xy).all() or (xy<0).any() or (xy>1).any():
                raise ValueError(f'Invalid normalized polygon in {text}:{line_number}')
            points=np.rint(xy.reshape(-1,2)*[w,h]).astype(np.int32)
            points[:,0]=points[:,0].clip(0,w-1)
            points[:,1]=points[:,1].clip(0,h-1)
            polygons[MAPPING[class_id]].append(points)
        masks={}
        for c in CLASSES:
            a=np.zeros((h,w),np.uint8)
            if polygons[c]: cv2.fillPoly(a,polygons[c],255)
            path=OUTPUT/split/c/(image.stem+'_'+c+'.png')
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                with Image.open(path) as old:
                    if not np.array_equal(np.asarray(old),a):
                        raise ValueError(f'Existing converted file differs; no overwrite: {path}')
            else: Image.fromarray(a).save(path)
            with Image.open(path) as check:
                assert np.array_equal(np.asarray(check),a),str(path)
            masks[c]={'path':path.relative_to(OUTPUT).as_posix(),'sha256':sha(path),
                'polygons':len(polygons[c]),'positive_pixels':int(np.count_nonzero(a)),
                'official_completeness_verified':False,
                'empty_note':'No user polygon for this class; not proof of an officially reviewed negative' if not polygons[c] else None}
        assert sha(text)==before
        rows.append({'sid':image.stem,'source_split':split,'image_path':str(image),
            'source_image_sha256':sha(image),'source_polygon_sha256':before,'size_wh':[w,h],
            'masks':masks,'training_allowed_by_original_Q_MIX_protocol':False})
    print(f'{split}: converted {count} images / {count*4} independent masks',flush=True)
manifest={'status':'DERIVED_PIXEL_MASK_EXPORT_COMPLETE_NOT_OFFICIAL_GT',
    'created_at':utc_now(),'source_root':str(SOURCE),'output_root':str(OUTPUT),
    'images_copied_or_modified':False,'source_labels_modified':False,'training_images':54,'testing_images':27,
    'testing_images_excluded_from_training':True,'mask_files':len(rows)*4,
    'class_order':list(CLASSES),'polygon_mapping':MAPPING,
    'rasterization':'numpy.rint(normalized_xy * [width,height]), clipped to image; cv2.fillPoly independently for each class; values 0/255; overlaps retained',
    'original_geometry_restored':False,'official_pixel_GT_restored':False,
    'missing_information':'Original native image geometry/transform and original pixel annotations. Cropped/resampled image information and simplified contour detail cannot be recreated by format conversion.',
    'samples':rows}
write_json(OUTPUT/'manifest.json',manifest)
(OUTPUT/'README.md').write_text(
    '# IDRiD 用户多边形转逐类像素掩膜\n\n'
    '已转换54张training和27张testing，共324份PNG，0为背景、255为对应类多边形覆盖区域。四类分别保存，保留类别重叠。图像与源TXT未修改、未复制；原图路径和每份SHA在manifest.json。testing的27张不进训练。\n\n'
    '这里的原坐标指best_segmentation当前方图坐标，未还原官方4288×2848几何。这是用户多边形的栅格化结果，无法证明与官方逐像素GT一致。无多边形时导出全零文件，但不据此确认官方阴性。此目录不满足原Q_MIX官方资产验收，未标READY。\n',encoding='utf-8')
write_json(ROOT/'reports/idrid_polygon_conversion_receipt.json',{
    'completed_at':utc_now(),'output_root':str(OUTPUT),'manifest_sha256':sha(OUTPUT/'manifest.json'),
    'converted_images':len(rows),'mask_files':len(rows)*4,'mask_encode_decode_exact':True,
    'source_txt_sha_unchanged':True,'official_restoration_claimed':False,'Q_MIX_ready':False})
print({'converted_images':len(rows),'mask_files':len(rows)*4,'official_GT_restored':False})
