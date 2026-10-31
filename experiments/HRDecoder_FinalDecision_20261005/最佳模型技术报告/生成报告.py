import json,re,hashlib
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image
from docx import Document
from docx.shared import Cm,Pt,Inches,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'结题报告'/'evidence'
EX=ROOT.parent/'execution'
TITLE='HRDecoder最佳模型训练与改进报告'
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=read(OLD/'four_arm_saved_validation.json');t=read(OLD/'final_locked_test_evidence.json')
def rec(a,u):return next(x for x in v['records'] if x.get('arm')==a and x['update']==u)
def score(a,u,f='fixed_fusion'):return rec(a,u)['outputs'][f]['miou_pct']
def f(x):return f'{x:.4f}'
tables={}
def tab(key,caption,cols,rows,weights):tables[key]={'caption':caption,'columns':cols,'rows':rows,'weights':weights}
testfinal=t['locked_test_miou_pct'];rtest=t['R_test_miou_pct'];p0=44.78089145110081;cold=44.21100725782683
milestone=[('原HRDecoder','HRNet-W48与高分辨率解码',42.2,None),('四路局部与细节融合','局部学生、MA残差、B、E4和翻转平均',43.2289242453,43.2289242453-42.2),('MA修正后的旧系统','H4修正融合结果的MA通道',cold,cold-43.2289242453),('P0','加入25%的ConvNeXt预测',p0,p0-cold),('R','加入25%的DINO Q16预测',rtest,rtest-p0),('最终最佳系统','改用空间交互S20的EMA预测',testfinal,testfinal-rtest)]
tab('milestones','表1 主要成果形成过程',['阶段','实际改动','测试mIoU %','提高百分点'],[[a,b,('42.20' if i==0 else f(z)),('—' if d is None else '+'+f(d))] for i,(a,b,z,d) in enumerate(milestone)],[1.45,3.1,1.15,1.2])
tab('detail_results','表2 原图细节与低学习率训练的结果',['方案','实际改动','验证mIoU %','测试mIoU %'],[['原HRDecoder','原始对照','39.3900','42.20'],['E4','增加原图stride4细节分支','38.7660','41.6260'],['B','从E4低学习率继续训练','39.2050','41.9180'],['E4与B各50%','融合两路病灶概率','39.5400','42.3500']],[1.2,3.1,1.3,1.3])
tab('dino_results','表3 DINO训练阶段与微调范围',['模型','训练方式','单分支验证 %','与P0融合验证 %'],[['Q16','先训头2,000步，再训最后4块至16,000步','37.8937','44.3557'],['F4','从Q16追加12,000步，只训最后4块','38.7411','44.3069'],['F12','从Q16追加12,000步，微调全部12块','40.2418','44.5888']],[.8,3.2,1.5,1.6])
tab('arms','表4 最后一轮四个方案',['方案','空间交互','TJDR监督','DDR训练设置'],[['C继续训练','无','无','同一父模型，相同32k日程'],['S空间交互','有','无','同一父模型，相同32k日程'],['D外部监督','无','前20k','同一父模型，相同32k日程'],['SD联合','有','前20k','同一父模型，相同32k日程']],[1.6,1.15,1.15,2.8])
tab('ablation','表5 同条件对照的完整保留节点',['方案','训练步数','单分支验证 %','75%P0加25%分支 %'],[[a,f'{u:,}',f(score(a,u,'single')),f(score(a,u))] for u in [20000,32000] for a in ['C','S','D','SD']],[.8,1.35,1.8,2.65])
baseline={'EX':51.56,'HE':43.13,'SE':49.98,'MA':24.11}
names={'EX':'硬性渗出EX','HE':'出血HE','SE':'软性渗出SE','MA':'微动脉瘤MA'}
tab('final_classes','表6 最佳模型相对原基线的类别提升',['病灶类别','原基线IoU %','最佳模型IoU %','提高百分点'],[[names[c],f'{baseline[c]:.2f}',f(t['test_per_class'][c]['iou_pct']),f(t['test_per_class'][c]['iou_pct']-baseline[c])] for c in ['EX','HE','SE','MA']]+[['四类平均mIoU','42.20',f(testfinal),f(testfinal-42.2)]],[1.8,1.55,1.65,1.3])
tab('not_kept','表7 主要未采纳方向及实际结果',['改动','结果','后续处理'],[['加入Lovasz损失','测试41.7770%，低于保留成员','不进入最终融合'],['U0高分辨率继续训练G','融合测试44.6345%，低于P0的44.7809%','保留原U0'],['IDRiD和MAPLES混合监督','24,000步单分支比DDR方案低1.3530个百分点','DINO父链采用DDR方案'],['负类语义修订E_NEG','融合验证44.1643%，低于R的44.3557%','未替换R分支'],['本轮TJDR监督D','20,000步融合45.1010%，低于C的45.1762%','不进入最佳模型'],['空间与TJDR联合SD','20,000步融合44.8832%；32,000步45.1389%','最终选择S']],[1.75,3.3,1.25])

content=(ROOT/'报告正文.md').read_text('utf8')
equations=read(ROOT/'公式源.json')
expanded=content
def mdtable(key):
    a=tables[key]
    rows=['**'+a['caption']+'**','| '+' | '.join(a['columns'])+' |','| '+' | '.join(['---']*len(a['columns']))+' |']
    rows+=['| '+' | '.join(map(str,r))+' |' for r in a['rows']]
    return '\n\n'+ '\n'.join(rows)+'\n\n'
expanded=re.sub(r'\[\[TABLE:(\w+)\]\]',lambda m:mdtable(m[1]),expanded)
expanded=re.sub(r'\[\[FIG:([^|]+)\|([^\]]+)\]\]',lambda m:'!['+m[2]+'](图示/'+m[1]+'.png)\n\n'+m[2],expanded)
expanded=re.sub(r'\[\[EQ:(\w+)\]\]',lambda m:'$$\n'+equations[m[1]]+'\n$$',expanded)
(ROOT/(TITLE+'.md')).write_text(expanded,'utf8')

doc=Document();sec=doc.sections[0]
sec.page_width=Cm(21);sec.page_height=Cm(29.7);sec.top_margin=Cm(1.9);sec.bottom_margin=Cm(1.9);sec.left_margin=Cm(2.1);sec.right_margin=Cm(2.1);sec.footer_distance=Cm(.85)
for name in ['Normal','Title','Heading 1','Heading 2','Caption']:
    style=doc.styles[name];style.font.name='Calibri';style.font.color.rgb=RGBColor(0,0,0)
    style._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'宋体' if name in ['Normal','Caption'] else '黑体')
normal=doc.styles['Normal'];normal.font.size=Pt(11);normal.paragraph_format.line_spacing=1.24;normal.paragraph_format.space_after=Pt(6);normal.paragraph_format.widow_control=True
for name,size in [('Title',21),('Heading 1',15),('Heading 2',12)]:
    s=doc.styles[name];s.font.size=Pt(size);s.font.bold=True;s.paragraph_format.keep_with_next=True;s.paragraph_format.space_before=Pt(10);s.paragraph_format.space_after=Pt(6)
doc.styles['Caption'].font.size=Pt(9.2);doc.styles['Caption'].paragraph_format.space_after=Pt(7)
for e in doc.styles['Title']._element.xpath('.//w:pBdr'):e.getparent().remove(e)
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
footer.add_run('第 ');field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field);footer.add_run(' 页')
for r in footer.runs:r.font.size=Pt(9)

def inline(p,text):
    for i,s in enumerate(re.split(r'\*\*(.*?)\*\*',text)):
        r=p.add_run(s);r.bold=i%2==1
def addtable(key):
    t=tables[key];p=doc.add_paragraph(t['caption'],style='Caption');p.paragraph_format.keep_with_next=True
    table=doc.add_table(rows=1,cols=len(t['columns']));table.alignment=WD_TABLE_ALIGNMENT.CENTER;table.autofit=False
    widths=[16.8*w/sum(t['weights']) for w in t['weights']]
    for i,col in enumerate(table.columns):col.width=Cm(widths[i])
    for i,c in enumerate(table.rows[0].cells):c.text=t['columns'][i]
    rep=OxmlElement('w:tblHeader');table.rows[0]._tr.get_or_add_trPr().append(rep)
    for row in t['rows']:
        cells=table.add_row().cells
        for i,(c,tx) in enumerate(zip(cells,row)):c.text=str(tx);c.width=Cm(widths[i])
    for ri,row in enumerate(table.rows):
        ns=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(ns)
        for i,c in enumerate(row.cells):
            c.width=Cm(widths[i]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=c._tc.get_or_add_tcPr();m=OxmlElement('w:tcMar')
            for side in ['top','bottom','left','right']:
                e=OxmlElement('w:'+side);e.set(qn('w:w'),'80');e.set(qn('w:type'),'dxa');m.append(e)
            pr.append(m)
            fill='E8EEF3' if ri==0 else ('F5F7F9' if ri%2==0 else 'FFFFFF')
            shade=OxmlElement('w:shd');shade.set(qn('w:fill'),fill);pr.append(shade)
            for p in c.paragraphs:
                p.paragraph_format.line_spacing=1.13;p.paragraph_format.space_before=Pt(1);p.paragraph_format.space_after=Pt(1)
                p.paragraph_format.keep_with_next=(ri==0 or (key=='final_classes' and ri<len(table.rows)-1))
                p.alignment=WD_ALIGN_PARAGRAPH.CENTER if (i>0 and key not in ['not_kept','arms']) else WD_ALIGN_PARAGRAPH.LEFT
                if key in ['milestones','detail_results','dino_results'] and i==1:p.alignment=WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:r.font.size=Pt(9.5);r.bold=ri==0
    borders=OxmlElement('w:tblBorders')
    for side in ['top','left','bottom','right','insideH','insideV']:
        e=OxmlElement('w:'+side);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
    table._tbl.tblPr.append(borders)
    doc.add_paragraph().paragraph_format.space_after=Pt(0)

eqcount=0;figcount=0
for block in re.split(r'\n\s*\n',content):
    block=block.strip()
    if not block:continue
    mt=re.fullmatch(r'\[\[TABLE:(\w+)\]\]',block);mf=re.fullmatch(r'\[\[FIG:([^|]+)\|([^\]]+)\]\]',block);me=re.fullmatch(r'\[\[EQ:(\w+)\]\]',block)
    if mt:addtable(mt[1]);continue
    if mf:
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
        p.add_run().add_picture(str(ROOT/'图示'/(mf[1]+'.png')),width=Cm(16.8))
        p=doc.add_paragraph(mf[2],style='Caption');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;figcount+=1;continue
    if me:
        pic=ROOT/'图示'/('公式_'+me[1]+'.png');w,h=Image.open(pic).size
        width=min(6.45,w/300*11.5/17)
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_before=Pt(3);p.paragraph_format.space_after=Pt(7)
        p.add_run().add_picture(str(pic),width=Inches(width));eqcount+=1;continue
    if block.startswith('# '):doc.add_paragraph(block[2:],style='Title');continue
    if block.startswith('## '):
        p=doc.add_paragraph(block[3:],style='Heading 1')
        # Natural pagination keeps the report compact; no separate cover or abstract.
        continue
    if block.startswith('### '):doc.add_paragraph(block[4:],style='Heading 2');continue
    p=doc.add_paragraph();inline(p,block.replace('\n',' '));p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
doc.core_properties.title=TITLE;doc.core_properties.subject='原始架构 模块改进 训练过程与结果';doc.core_properties.author='项目组'
doc.save(ROOT/(TITLE+'.docx'))

sourcepaths=[OLD/'original_HRDecoder_baseline_evidence.json',OLD/'four_arm_saved_validation.json',OLD/'final_locked_test_evidence.json',OLD/'historical_experiment_ledger.json',EX/'spatial.py',EX/'final_model.py',EX/'qlib'/'model.py',EX/'qlib'/'loss.py',ROOT.parent.parent/'hrdecoder_plus3'/'repo'/'mmseg'/'models'/'necks'/'detail_branch.py',ROOT.parent.parent/'HRDecoder_MA残差头与集成_20260915-16'/'03_代码'/'ma_residual.py',ROOT.parent.parent/'HRDecoder_审查轮_20260916'/'tools'/'h4_post_fusion_corrector.py',ROOT.parent.parent/'HRDecoder_FullCompetition_20260920'/'trainer'/'models'/'convnext_unet.py',Path('D:/HRDecoder/mmseg/models/segmentors/HRDecoder.py'),Path('D:/HRDecoder/mmseg/models/decode_heads/fcn_head.py')]
ledger={'created_at':datetime.now(timezone.utc).isoformat(),'tables':tables,'final_test':testfinal,'gain_vs_baseline_approx_pp':testfinal-42.2,'gain_vs_R_pp':testfinal-rtest,'space_effect_20k_single_pp':score('S',20000,'single')-score('C',20000,'single'),'space_effect_32k_single_pp':score('S',32000,'single')-score('C',32000,'single'),'sources':[{'path':str(p),'sha256':sha(p)} for p in sourcepaths],'no_new_model_training_or_inference':True,'prior_report_untouched':True}
(ROOT/'指标与来源.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2),'utf8')
assert '用户' not in expanded
assert not re.search(r'\[\[(?:FIG|TABLE|EQ):',expanded)
print(json.dumps({'docx':str(ROOT/(TITLE+'.docx')),'tables':len(tables),'figures':figcount,'equations':eqcount,'chinese_characters':len(re.findall('[\u4e00-\u9fff]',expanded))},ensure_ascii=False))
