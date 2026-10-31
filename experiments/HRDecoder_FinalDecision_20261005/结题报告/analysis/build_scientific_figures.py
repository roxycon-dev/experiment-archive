"""Publication figures from frozen saved metrics and FP32 validation crops only."""
from pathlib import Path
import json, csv, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

OUT=Path(__file__).resolve().parents[1]; EX=OUT.parent/'execution'; SR=EX/'server_results'
FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
CL=['EX','HE','SE','MA']; COLORS={'C':'#5b6770','S':'#0072b2','D':'#d55e00','SD':'#009e73'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                    'svg.fonttype':'none','pdf.fonttype':42,'axes.labelsize':10,'legend.fontsize':8})
def read(p):return json.loads(Path(p).read_text('utf8'))
ev=read(OUT/'evidence/four_arm_saved_validation.json'); test=read(OUT/'evidence/final_locked_test_evidence.json')
def row(a,u):return next(x for x in ev['records'] if x['arm']==a and x['update']==u)
def score(a,u,f='fixed_fusion'):return row(a,u)['outputs'][f]['miou_pct']
def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',facecolor='white'); plt.close(fig)
def box(ax,x,y,w,h,text,color='#eef2f4'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.018',fc=color,ec='#66737c',lw=.8))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
def arrow(ax,p,q,color='#65727b',dashed=False):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=12,lw=1.2,color=color,linestyle='--' if dashed else '-'))

# F01 deliberately separates trained parameter inheritance from prediction fusion.
fig,ax=plt.subplots(figsize=(10.8,6));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.02,.96,'TRAINED PARAMETER ANCESTRY',weight='bold')
for x,y,w,h,t,col in [(.02,.71,.16,.15,'HRNet-W48\nImageNet start','#eef2f4'),(.23,.71,.13,.15,'E4 detail\nRGB stride4','#eef2f4'),(.41,.71,.13,.15,'B continuation\nlow LR + BN freeze','#eef2f4'),(.59,.80,.13,.12,'D local student\n1024 crop','#eef2f4'),(.59,.63,.13,.12,'E MA residual\nB frozen','#eef2f4'),(.79,.68,.18,.18,'D / E / B / E4\n+ learned H4\nHistorical C_old','#f5ecd9'),(.02,.28,.16,.16,'DINOv3 ViT-B/16\nLVD pretraining','#e5f1f7'),(.23,.28,.14,.16,'Q_DDR16k\nDDR head + last4','#e5f1f7'),(.42,.28,.13,.16,'F12 +12k\nall 12 blocks','#e5f1f7'),(.62,.28,.15,.16,'C / S / D / SD\npaired fresh optimizer','#e5f1f7'),(.83,.28,.14,.16,'S20k EMA\nselected new branch','#d9edf6')]:box(ax,x,y,w,h,t,col)
for p,q in [((.18,.78),(.23,.78)),((.36,.78),(.41,.78)),((.54,.8),(.59,.86)),((.54,.72),(.59,.69)),((.18,.36),(.23,.36)),((.37,.36),(.42,.36)),((.55,.36),(.62,.36)),((.77,.36),(.83,.36))]:arrow(ax,p,q)
for p,q in [((.72,.86),(.79,.79)),((.72,.69),(.79,.72))]:arrow(ax,p,q,dashed=True)
ax.text(.03,.15,'ConvNeXt-S ImageNet -> independent U0 DDR training (no S parameter inheritance)',fontsize=9)
ax.text(.03,.07,'Dashed = prediction composition. TJDR-trained D/SD and historical MIX do not parent S.',fontsize=9,color='#444')
save(fig,'F01_training_genealogy')

fig,ax=plt.subplots(figsize=(10.6,5.9));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,.02,.75,.25,.17,'Historical C_old\n3D + 2E + B + E4, /7\nflip TTA + H4(MA)','#f5ecd9')
box(ax,.02,.47,.25,.15,'Historical U0\nnative whole-image probability','#eef2f4')
box(ax,.36,.58,.22,.20,'P0 = .75 C_old + .25 U0\nimmutable FP32 probabilities','#eef2f4')
box(ax,.02,.12,.25,.17,'S20k EMA\nDINOv3 + spatial interaction\n1024 window / center768','#d9edf6')
box(ax,.69,.34,.27,.27,'LOCKED FINAL RECIPE\n.75 P0 + .25 S20k_EMA\nstrict probability > .5\nval 45.3218 / test 45.8604','#d9edf6')
for p,q in [((.27,.83),(.36,.72)),((.27,.54),(.36,.64)),((.58,.67),(.69,.53)),((.27,.21),(.69,.40))]:arrow(ax,p,q)
ax.text(.33,.17,'Expanded outer weights: .5625 C_old + .1875 U0 + .25 S_EMA\nH4 is nonlinear; C_old cannot be reduced to four fixed raw-model weights.',fontsize=9)
ax.text(.03,.02,'Reference R uses the identical .75 P0 partner with Q_DDR16k instead of S.',fontsize=9)
save(fig,'F02_locked_inference_system')

fig,ax=plt.subplots(figsize=(9.5,4.5)); names=['B0','E4+B','D old','4-branch\nflip','+H4','P0\n+U0','R\n+Q16k','Final\n+S20k']
vals=[42.2,42.35,42.576,43.228924245323747,44.2115035614162,44.78089145110081,45.72155710642771,test['locked_test_miou_pct']]
ax.plot(range(len(vals)),vals,'o-',color='#0072b2',lw=2)
for i,v in enumerate(vals):ax.annotate(f'{v:.3f}',(i,v),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
ax.set_xticks(range(len(vals)),names);ax.set_ylabel('Historical recorded test mIoU (%)');ax.set_ylim(41.8,47.5)
ax.axhline(47.2115,color='#d55e00',ls='--',lw=1);ax.text(.05,47.25,'Registered target 47.2115',fontsize=8)
ax.grid(axis='y',alpha=.2);ax.set_title('Recorded system milestones; descriptive history, not isolated causal increments')
save(fig,'F03_historical_system_milestones')

fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
for ax,f,title in zip(axes,['single','fixed_fusion'],['New branch alone','Fixed .75 P0 + .25 branch']):
    for a,col in COLORS.items():
        rows=sorted([x for x in ev['records'] if x['arm']==a],key=lambda x:x['update'])
        ax.plot([x['update']/1000 for x in rows],[x['outputs'][f]['miou_pct'] for x in rows],'-o',ms=3,color=col,label=a)
    for u in [20,32]:ax.axvline(u,color='#888',ls=':',lw=.9)
    ax.set(xlabel='Completed updates (thousands)',ylabel='Validation mIoU (%)',title=title);ax.grid(alpha=.18);ax.legend(ncol=4)
axes[1].axhline(44.355692338538375,color='#555',ls='--',lw=1)
axes[1].scatter([16],[score('S',16000)],s=65,marker='s',facecolors='none',edgecolors='#0072b2',zorder=8)
axes[1].annotate('16k diagnostic',(16,score('S',16000)),xytext=(21,45.46),arrowprops={'arrowstyle':'-','lw':.6},ha='left',fontsize=8)
fig.tight_layout();save(fig,'F04_complete_validation_curves')

fig,axes=plt.subplots(1,2,figsize=(10,4.1));contr=ev['paired_registered_endpoint_contrasts']
for ax,f,title in zip(axes,['single','fixed_fusion'],['Standalone matched changes','Fixed25 matched changes']):
    labels=['S-C','D-C','SD-C','Interaction']; xx=np.arange(4)
    for j,u in enumerate([20000,32000]):
        d=next(x for x in contr if x['formula']==f and x['update']==u)['contrasts']
        vv=[d['S_minus_C_pp'],d['D_minus_C_pp'],d['SD_minus_C_pp'],d['SD_minus_S_minus_D_plus_C_pp']]
        ax.bar(xx+(j-.5)*.33,vv,.31,label=f'{u//1000}k',color=['#0072b2','#56b4e9'][j])
    ax.axhline(0,color='#333',lw=.7);ax.set_xticks(xx,labels);ax.set(ylabel='Validation difference (percentage points)',title=title);ax.legend();ax.grid(axis='y',alpha=.15)
fig.tight_layout();save(fig,'F05_matched_effects')

fig,axes=plt.subplots(1,2,figsize=(10,4.2)); rr=next(x for x in ev['records'] if x['label']=='Q_DDR16k_R42')
for ax,groups,title in [(axes[0],[('R',rr['outputs']['fixed_fusion']['per_class']),('S20 fixed25',row('S',20000)['outputs']['fixed_fusion']['per_class'])],'Validation class IoU'),(axes[1],[('B0 reported', {c:{'iou_pct':v} for c,v in zip(CL,[51.56,43.13,49.98,24.11])}),('R',test['R_test_per_class']),('S20 fixed25',test['test_per_class'])],'Locked test class IoU')]:
    xx=np.arange(4); ww=.8/len(groups)
    for j,(name,d) in enumerate(groups):ax.bar(xx+(j-(len(groups)-1)/2)*ww,[d[c]['iou_pct'] for c in CL],ww*.92,label=name,color=['#9ca7ad','#56b4e9','#0072b2'][-len(groups)+j])
    ax.set_xticks(xx,CL);ax.set(ylabel='IoU (%)',title=title,ylim=(0,70));ax.legend(fontsize=8);ax.grid(axis='y',alpha=.17)
fig.tight_layout();save(fig,'F06_per_class_results')

fig,axes=plt.subplots(1,2,figsize=(10.3,4.2))
vp=read(OUT/'evidence/validation_paired_images.json')
for ax,d,title in [(axes[0],vp['per_image'],'76 validation images'),(axes[1],test['per_image'],'227 locked test images')]:
    vv=np.array([x['delta_image_iou_pp'] for x in d]);ax.hist(vv,bins=22,color='#0072b2',alpha=.8,edgecolor='white')
    ax.axvline(0,color='#333',lw=1);ax.set(title=title,xlabel='Diagnostic per-image four-class mIoU difference vs R (pp)',ylabel='Images')
    ax.text(.96,.94,f'Positive {sum(vv>0)} / Negative {sum(vv<0)}',transform=ax.transAxes,ha='right',va='top',fontsize=8)
fig.tight_layout();save(fig,'F07_image_heterogeneity')

fig,axes=plt.subplots(1,2,figsize=(10,4.1)); mech=[]
for a in COLORS:
    for u in [20000,32000]:
        x=read(SR/f'runs/{a}_seed42/val_u{u}/metrics.json')
        for c in ['HE','MA']:
            gt=sum(z['mechanism'][c]['gt_lesions'] for z in x['per_image']);miss=sum(z['mechanism'][c]['missed_lesions'] for z in x['per_image']);fp=sum(z['mechanism'][c]['new_false_positive_regions'] for z in x['per_image'])
            mech.append({'arm':a,'update':u,'class':c,'gt_components':gt,'missed_components':miss,'miss_pct':100*miss/gt,'outside_GT_predicted_components':fp})
for ax,c in zip(axes,['HE','MA']):
    for j,u in enumerate([20000,32000]):ax.bar(np.arange(4)+(j-.5)*.34,[next(x for x in mech if x['arm']==a and x['update']==u and x['class']==c)['miss_pct'] for a in COLORS],.32,label=f'{u//1000}k',color=['#0072b2','#56b4e9'][j])
    ax.set_xticks(np.arange(4),list(COLORS));ax.set(title=f'{c}: standalone branch only',ylabel='Zero-overlap GT component rate (%)');ax.legend();ax.grid(axis='y',alpha=.15)
fig.tight_layout();save(fig,'F08_saved_component_misses')
(OUT/'evidence/component_diagnostics.json').write_text(json.dumps(mech,indent=2),'utf8')

bench=read(SR/'reports/gpu_readiness.json')['benchmarks'];fig,axes=plt.subplots(1,2,figsize=(10,3.8))
arms=list(COLORS);xx=np.arange(4)
axes[0].bar(xx,[bench[a]['seconds_per_update'] for a in arms],color=list(COLORS.values()));axes[0].set_xticks(xx,arms);axes[0].set(ylabel='Seconds per complete update',title='Accepted 100-update timing')
axes[1].bar(xx,[bench[a]['peak_gpu_bytes']/1024**3 for a in arms],color=list(COLORS.values()));axes[1].set_xticks(xx,arms);axes[1].set(ylabel='Peak allocated GPU memory (GiB)',title='Same acceptance measurement')
for ax in axes:ax.grid(axis='y',alpha=.15)
fig.tight_layout();save(fig,'F09_accepted_efficiency')

# Fixed threshold, no calibration search; all qualitative cases chosen in saved plan.
plan=read(EX/'reports/publication_server_evidence_plan.json');der=read(SR/'reports/publication_server_evidence/derivation.json')
case_summary=[]
def overlay(rgb,mask,gt=None):
    a=rgb.astype(np.float32).copy(); palette=np.array([[230,70,40],[40,200,70],[235,205,25],[35,125,240]])
    for c in range(4):a[mask[c]]=.42*a[mask[c]]+.58*palette[c]
    return np.clip(a,0,255).astype(np.uint8)
for group, cases in [(10,plan['cases'][:3]),(11,plan['cases'][3:])]:
    fig,axes=plt.subplots(len(cases),5,figsize=(11,2.3*len(cases)))
    for i,case in enumerate(cases):
        sid=case['sid'];x0,y0,x1,y1=case['crop_xyxy']
        rgb=np.asarray(Image.open(Path('D:/HRDecoder/data/DDR_split_613/images/val')/(sid+'.jpg')).convert('RGB'))[y0:y1,x0:x1]
        g=np.asarray(Image.open(Path('D:/HRDecoder/data/DDR_split_613/labels/val')/(sid+'.png')))[y0:y1,x0:x1]
        pp=SR/'reports/publication_server_evidence/val_p0_crops'/f'{sid}.npz';qq=SR/'runs/S_seed42/val_u20000'/f'{sid}.npz'
        assert hashlib.sha256(pp.read_bytes()).hexdigest()==next(x['sha256'] for x in der['files'] if x['sid']==sid)
        with np.load(pp) as z:p=z['prob']
        with np.load(qq) as z:q=z['prob'][:,y0:y1,x0:x1]
        assert p.dtype==q.dtype==np.float32 and p.shape==q.shape==(4,768,768)
        gt=np.stack([g==j for j in range(1,5)]); ff=np.float32(.75)*p+np.float32(.25)*q
        arrays=[rgb,overlay(rgb,gt),overlay(rgb,p>.5),overlay(rgb,q>.5),overlay(rgb,ff>.5)]
        for j,(ax,im) in enumerate(zip(axes[i],arrays)):
            ax.imshow(im);ax.axis('off')
            if i==0:ax.set_title(['Original RGB','Ground truth','Frozen P0','S20k alone','Locked fixed25'][j],fontsize=9)
        axes[i,0].text(-.02,-.03,sid,transform=axes[i,0].transAxes,fontsize=8)
        axes[i,4].text(.02,-.03,case['crop_lesion_class']+' / saved-count selection',transform=axes[i,4].transAxes,fontsize=7)
        ct={}
        for name,a in [('P0',p),('S',q),('locked',ff)]:
            pred=a>.5;ct[name]=[[int((pred[c]&gt[c]).sum()),int((pred[c]&~gt[c]).sum()),int((~pred[c]&gt[c]).sum())] for c in range(4)]
        case_summary.append({'sid':sid,'crop_xyxy':case['crop_xyxy'],'class':case['crop_lesion_class'],'selection_rule':case['rule'],'crop_counts':ct,'uses_validation_only':True,'no_inference':True})
    fig.tight_layout(pad=.7,rect=(0,.035,1,1))
    fig.text(.5,.006,'Overlay colours: EX orange | HE green | SE yellow | MA blue. Identical 768 x 768 coordinates per row.',ha='center',fontsize=8)
    save(fig,f'F{group:02d}_predeclared_validation_cases')
(OUT/'evidence/qualitative_crop_counts.json').write_text(json.dumps(case_summary,indent=2),'utf8')
# One tidy table backs all endpoint/curve plots.
with (OUT/'evidence/validation_curve_source.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['arm','update','formula','miou_pct','source_path','source_sha256'])
    for x in ev['records']:
        for formula,v in x['outputs'].items():w.writerow([x['label'],x['update'],formula,v['miou_pct'],x['source_path'],x['source_sha256']])
print('Created 11 scientific figures in PNG/SVG/PDF; original counts and FP32 validation evidence only.')
