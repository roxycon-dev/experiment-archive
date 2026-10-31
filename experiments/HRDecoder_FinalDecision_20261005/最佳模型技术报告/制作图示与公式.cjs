const fs=require('fs'),path=require('path');
const packages='C:/Users/GuanXingRen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const sharp=require(path.join(packages,'sharp'));
const root=__dirname, out=path.join(root,'图示');fs.mkdirSync(out,{recursive:true});
const equations={
 detail:'z_{E4}=z_{HRDecoder}+\\alpha\\,\\operatorname{Up}(D(x)),\\qquad\\alpha_0=0',
 local:'P_{local}=0.5P_{student}+0.5P_B',
 early_ma:'z_{MA}^{new}=z_{MA}^{base}+h(F_{detail},F_{semantic},z_{MA}^{base})',
 old_mix:'P_{mix}=\\frac{3P_{local}+2P_{MAres}+P_B+P_{E4}}{7}',
 ma_correct:'P_{MA}^{new}=\\sigma\\!\\left(\\operatorname{logit}(P_{mix,MA})+6\\tanh(h(X))\\right)',
 p0:'P_0=0.75P_{old}+0.25P_{U0}',
 msda:'A(q,F)=W_o\\operatorname{Concat}_{m=1}^{12}\\!\\left[\\sum_{\\ell=1}^{L}\\sum_{k=1}^{4}a_{m\\ell k}(q)\\,W_v^mF_{\\ell}(r+\\Delta_{m\\ell k}(q))\\right]',
 inject:'T\\leftarrow T+\\operatorname{MSDA}(\\operatorname{LN}(T),\\operatorname{LN}(C))',
 extract:'C\\leftarrow C+\\operatorname{MSDA}(\\operatorname{LN}(C),\\operatorname{LN}(T))',
 decoder_residual:'F_i^{new}=F_i^{DINO}+\\operatorname{Conv}_{1\\times1}(C_i)',
 bce:'L_{BCE}=-\\frac{\\sum_j v_j[y_j\\log p_j+(1-y_j)\\log(1-p_j)]}{\\sum_jv_j}',
 dice:'L_{Dice}=1-\\frac{2\\sum_jv_jp_jy_j+1}{\\sum_jv_jp_j+\\sum_jv_jy_j+1}',
 loss:'L_{micro}=\\frac{1}{4}\\sum_{c=1}^{4}\\left(0.5L_{BCE,c}+0.5L_{Dice,c}\\right)',
 source_loss:'L_{update}=\\frac{1}{4}\\sum_{t=1}^{4}L_{DDR,t}+0.25\\left(\\frac{1}{2}\\sum_{t=1}^{2}L_{TJDR,t}\\right)',
 ema:'\\bar\\theta_t=0.999\\bar\\theta_{t-1}+0.001\\theta_t',
 final:'P_{final}=0.75P_0+0.25P_{S20,EMA}=0.5625P_{old}+0.1875P_{U0}+0.25P_{S20,EMA}',
 miou:'\\operatorname{IoU}_c=\\frac{TP_c}{TP_c+FP_c+FN_c},\\qquad\\operatorname{mIoU}=\\frac{1}{4}\\sum_{c=1}^4\\operatorname{IoU}_c'
};
function escape(x){return x.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');}
const blue='#E8F0F8',green='#E9F2EC',gray='#F1F3F5';
function label(x,y,lines,size=26,color='#182530'){return lines.map((l,i)=>`<text x="${x}" y="${y+i*size*1.42}" text-anchor="middle" fill="${color}" font-size="${size}">${escape(l)}</text>`).join('');}
function box(x,y,w,h,lines,fill=blue,size=26){return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="${fill}" stroke="#667789" stroke-width="2"/>`+label(x+w/2,y+h/2-(lines.length-1)*size*.71+size*.35,lines,size);}
function arrow(x1,y1,x2,y2){return `<path d="M${x1} ${y1} L${x2} ${y2}" fill="none" stroke="#53677B" stroke-width="3" marker-end="url(#arr)"/>`;}
function route(points){return `<polyline points="${points}" fill="none" stroke="#53677B" stroke-width="3" marker-end="url(#arr)"/>`;}
function frame(w,h,body){return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><defs><marker id="arr" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto"><path d="M0,0 L0,8 L8,4 z" fill="#53677B"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="Microsoft YaHei,SimSun,Arial">${body}</g></svg>`;}
const figures={};
figures['01_original']=frame(1400,520,
box(20,190,175,120,['RGB输入','2048×2048'])+arrow(195,250,230,250)+box(230,190,185,120,['缩放输入','1024×1024'])+arrow(415,250,450,250)+
box(450,150,235,200,['HRNet四级特征','48 / 96通道','192 / 384通道'])+arrow(685,250,725,250)+box(725,190,190,120,['对齐并拼接','720通道'])+
route('915,250 945,250 945,100 970,100')+box(970,40,230,120,['整图解码','1×1 → 7×7','输出四通道'],gray,25)+
route('915,250 945,250 945,370 970,370')+box(970,300,230,140,['裁特征并放大','共享FCN解码','恢复局部结果'],green,25)+
route('1200,100 1240,100 1240,250 1270,250')+route('1200,370 1240,370 1240,250 1270,250')+box(1270,175,110,150,['两路','分数','平均'],blue,26)+
label(700,485,['高分辨率路径读取同一份主干特征  原图细节分支在下一阶段增加'],25));
figures['02_ma_head']=frame(1400,490,
box(20,30,330,175,['7通道输入','RGB + 三路MA分数','学生与参考概率差'],gray,27)+arrow(350,118,390,118)+
box(390,55,235,125,['3×3卷积','7 → 16 + ReLU'])+arrow(625,118,665,118)+box(665,55,235,125,['3×3卷积','16 → 16 + ReLU'])+
arrow(900,118,940,118)+box(940,55,185,125,['1×1卷积','16 → 1'])+arrow(1125,118,1170,118)+box(1170,55,205,125,['6 × tanh','MA修正量'],green)+
box(85,295,260,120,['原融合MA概率','转为logit'],gray)+arrow(345,355,630,355)+box(630,305,140,100,['相加'],green)+
route('1270,180 1270,250 700,250 700,305')+arrow(770,355,890,355)+box(890,295,300,120,['sigmoid','新的MA概率'])+
label(700,470,['EX HE SE三个通道沿用原融合结果'],26));
figures['03_dino']=frame(1400,625,
box(20,225,185,120,['原生窗口','1024×1024'],gray)+arrow(205,285,245,285)+box(245,195,285,180,['DINOv3 ViT-B/16','64×64位置网格','12层  768通道'])+
route('530,250 565,250 565,95 615,95')+box(615,30,275,130,['取第3 / 6 / 9 / 12层','768 → 128通道','重建四个尺度'],blue,25)+
arrow(890,95,940,95)+box(940,30,435,130,['步幅32 → 16 → 8 → 4','逐级放大 + 跳接 + 残差卷积'],green,25)+
route('205,285 225,285 225,475 615,475')+box(615,410,275,130,['RGB浅层卷积','步幅4  64通道'],gray)+
route('1150,160 1150,265')+route('890,475 1150,475 1150,385')+box(965,265,370,120,['拼接128 + 64通道','卷积到128 → 四类输出'],blue,25)+
label(700,605,['主干学习图像关系  解码器和RGB路径恢复位置与细节'],26));
figures['04_spatial']=frame(1400,600,
label(700,45,['上路为DINO位置特征   下路为空间特征'],29)+
box(20,135,180,120,['原图窗口','1024×1024'],gray,25)+arrow(200,195,240,195)+
box(20,410,180,140,['卷积','金字塔'],green,26)+route('110,255 110,410')+arrow(200,480,240,480)+
[0,1,2,3].map((i)=>{let x=240+i*290;return box(x,125,255,140,[`DINO第${3*i+1}至${3*i+3}块`,'注入后更新语义'],blue,24)+
box(x,415,255,130,['空间特征','提取后更新细节'],green,24)+
arrow(x+70,415,x+70,265)+arrow(x+185,265,x+185,415)+
label(x+60,333,['注入'],23)+label(x+198,372,['提取'],23)+
(i<3?arrow(x+255,195,x+290,195)+arrow(x+255,480,x+290,480):'');}).join('')+
label(700,587,['四轮交互后  四级空间特征通过残差加入原解码器'],26));
figures['05_final']=frame(1400,560,
box(25,35,425,135,['改进后的HRDecoder系统','局部预测 + MA修正','旧系统测试44.2110%'],gray,26)+
box(25,255,425,110,['ConvNeXt分支U0','独立卷积网络'],green)+
route('450,103 520,103 520,220 575,220')+route('450,310 520,310 520,220 575,220')+
box(575,140,340,160,['75%旧系统 + 25%U0','形成P0','测试44.7809%'],blue,27)+
box(575,385,340,125,['DINO空间交互分支S','20,000步EMA'],green,25)+
route('915,220 960,220 960,320 1010,320')+route('915,448 960,448 960,320 1010,320')+
box(1010,220,365,200,['75%P0 + 25%S','最终最佳输出','测试45.8604%'],blue,28)+
label(700,550,['每个像素先融合概率  再用0.5阈值输出病灶区域'],26));
(async()=>{
 for(const[id,svg]of Object.entries(figures)){
  fs.writeFileSync(path.join(out,id+'.svg'),svg);await sharp(Buffer.from(svg)).resize({width:2800}).png().toFile(path.join(out,id+'.png'));
 }
 fs.writeFileSync(path.join(root,'公式源.json'),JSON.stringify(equations,null,2));
 process.stdout.write(JSON.stringify({equations:Object.keys(equations).length,figures:Object.keys(figures).length})+'\n');
})();
