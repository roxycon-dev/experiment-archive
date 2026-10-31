"""Representative failure crops, selected by error counts, with a manual categorization sheet."""
import sys
sys.path.insert(0,str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import argparse
import csv
import html
from pathlib import Path
import cv2
cv2.setNumThreads(1)
import numpy as np
from PIL import Image,ImageDraw

from nextlib.common import ROOT,read_json,CLASSES
from nextlib.data import DDR
from tools.probability_cache import load_probability


def largest_box(mask,shape,size=512):
    n,cc,stat,centroids=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    if n<=1:return 0,0,min(shape[1],size),min(shape[0],size)
    index=1+int(np.argmax(stat[1:,4]))
    cx,cy=centroids[index]
    x=max(0,min(shape[1]-size,int(cx)-size//2));y=max(0,min(shape[0]-size,int(cy)-size//2))
    return x,y,min(shape[1],x+size),min(shape[0],y+size)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--baseline',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--top',type=int,default=2);a=ap.parse_args()
    assert 1 <= a.top <= 2
    s=read_json(ROOT/'configs/settings.json');dataset=DDR(s['ddr_root'],'val',read_json(s['val_contract'])['stems'])
    report=read_json(Path(a.cache)/'metrics.json')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    exact_errors={}
    for sid in dataset.stems:
        _,label=dataset.load(sid)
        p=load_probability(a.cache,sid)>.5
        q=load_probability(a.baseline,sid)>.5
        exact_errors[sid]={}
        for c in range(4):
            gt=label==c+1
            exact_errors[sid][c]={'new_FP':int((p[c]&~q[c]&~gt).sum()),
                'remaining_FN':int((gt&~p[c]).sum()),'rescued_FN':int((gt&p[c]&~q[c]).sum())}
    from nextlib.common import write_json
    write_json(out/'exact_error_selection.json',dict(baseline=a.baseline,candidate=a.cache,
        per_image=exact_errors,rule='actual pixel sets, not net changes in FP/FN'))
    selected=[]
    for c,name in enumerate(CLASSES):
        for kind,column,direction in [('new_FP',1,1),('remaining_FN',2,1),('rescued_FN',2,-1)]:
            def rank(row):
                return exact_errors[row['sid']][c][kind]
            for row in sorted(report['per_image'],key=rank,reverse=True)[:a.top]:
                selected.append((row['sid'],c,kind,rank(row)))
    cards=[];manual=[]
    for sid,c,kind,rank in selected:
        rgb,label=dataset.load(sid)
        p=load_probability(a.cache,sid)[c]>.5;q=load_probability(a.baseline,sid)[c]>.5;gt=label==c+1
        if kind=='new_FP':error=p&~q&~gt
        elif kind=='rescued_FN':error=p&~q&gt
        else:error=gt&~p
        box=largest_box(error,label.shape)
        x,y,x2,y2=box
        base=rgb[y:y2,x:x2].copy()
        annotated=base.copy();truth=gt[y:y2,x:x2];pred=p[y:y2,x:x2]
        annotated[truth&pred]=(0.5*annotated[truth&pred]+np.array([0,160,0])*.5).astype(np.uint8)
        annotated[~truth&pred]=(0.4*annotated[~truth&pred]+np.array([255,0,0])*.6).astype(np.uint8)
        annotated[truth&~pred]=(0.4*annotated[truth&~pred]+np.array([255,220,0])*.6).astype(np.uint8)
        strip=Image.new('RGB',(base.shape[1]*2,base.shape[0]+28),'white')
        strip.paste(Image.fromarray(base),(0,28));strip.paste(Image.fromarray(annotated),(base.shape[1],28))
        title=f'{sid} {CLASSES[c]} {kind} rank={rank}'
        ImageDraw.Draw(strip).text((5,5),title,fill='black')
        filename=f'{sid}_{CLASSES[c]}_{kind}.png';strip.save(out/filename)
        cards.append(f'<figure><img src="{html.escape(filename)}"><figcaption>{html.escape(title)} '
                     f'crop={box}</figcaption></figure>')
        manual.append(dict(sid=sid,lesion_class=CLASSES[c],selection=kind,rank=rank,
                           crop=str(box),error_type='unclassified',notes=''))
    with (out/'manual_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(manual[0]));writer.writeheader();writer.writerows(manual)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>DDR error review</title>'
        '<style>body{font:16px sans-serif;margin:24px;background:#fafafa}figure{margin:24px 0}img{max-width:100%}</style>'
        '<h1>按误差计数选择的代表样本</h1><p>绿：TP；红：FP；黄：FN。原坐标裁块，无阈值调节。'
        '请在 manual_review.csv 标注血管、视盘、黑边、普通组织、整颗漏检或边缘。'
        '没有血管/视盘真值，程序不冒称已自动识别这些错误类型。</p>'+''.join(cards),encoding='utf-8')
    print(out/'index.html')


if __name__=='__main__':main()
