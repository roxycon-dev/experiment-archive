"""Save the already verified20k report and diagnostic provenance locally."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
here=ROOT/'sacd_execution'
series=json.loads((here/'validation_series.json').read_text(encoding='utf-8'))
lines=['# 验证指标与趋势','', '完整76图val；各指标以原始TP/FP/FN核对。单位为百分比，变化为百分点。','',
       '| Step | mIoU | ΔmIoU | EX IoU/P/R | HE IoU/P/R | SE IoU/P/R | MA IoU/P/R |',
       '|---|---:|---:|---|---|---|---|']
for r in series['records']:
    delta='—' if r['delta_mIoU_pp'] is None else '%+.3f'%r['delta_mIoU_pp']
    values=['%.3f/%.3f/%.3f'%(c['IoU'],c['precision'],c['recall']) for c in r['classes']]
    lines.append('| %s | %.3f | %s | %s |'%(r['step'],r['mIoU'],delta,' | '.join(values)))
lines += ['', '当前best-val：%s，%.6f%%。固定40k端点仍为主结果，尚未完成。'%(series['best_val']['step'],series['best_val']['mIoU'])]
(here/'验证指标与趋势.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
analysis='''

## 2026-10-09 20k及配对诊断启动

20k mIoU37.184798%，较16k+5.712220pp，较12k+0.330975pp，当前best为20k。原HRDecoder同节点+4.661798pp；CET同节点-0.814873pp；v3同节点+22.133885pp。标签GT阳性总数和原始counts均一致。

HE FP1881983→417270（约下降77.83%），precision26.610919%→61.781777%，recall56.587977%→55.935449%；主要恢复来自误检压低，仍约44%阳性漏检。EX召回下降至57.372%，MA召回下降至34.061%；不能把总体回升解释为所有类别共同改善。纠错相对基础头净mIoU收益1.424537pp，HE略低于基础头41.718443%。

16k单次回落之后已恢复，不能再称持续退化或确认无希望。梯度耦合、动态目标跳变、GT目标与纯原版差异均有源码依据，哪一个主导波动仍未证明。下一步仅验证已登记的feature-detach假设。

20k持久last.pt不可变硬链接已保存。原SACD控制器/训练按身份匹配暂停，不改其核心代码。两个500步诊断臂的3步实际2048/batch2 smoke必须丢弃后重载同父模型/SGD/RNG。

首次诊断control在第一次smoke及初始完整76图val之后，第二次fork DataLoader时停滞。观察到自身12个工作进程与主进程futex等待，无首25步进度回执或有效诊断checkpoint。保留失败尝试全部文本，并只终止身份匹配的自身控制器、探针和它的直接工作进程，未触碰外部AI诊断。修复为同一persistent DataLoader复用，重置sampler起止；CPU exposure/seed复用核对及每组新代码GPU gate必须再次通过。未知未报告更新保守上限24，不能声称绝对零消费。

修复控制器PID31202，仅允许一次显式修复，预登记指标门槛不改变。原20k父模型及自身RAM保留，服务器持续开机；暖诊断及fresh8k未通过之前，不启动新的完整40k。修复后的正式GPU吞吐/探针结果尚待实际回执。
'''
with (here/'趋势分析记录.md').open('a',encoding='utf-8') as file:file.write(analysis)
with (ROOT/'sacd_micro_execution/短测启动记录.md').open('a',encoding='utf-8') as file:
    file.write('\n原SACD实际20k已完成，mIoU37.184798%，当前最高。诊断初次加载器fork停滞的失败尝试已归档，按身份只结束自身进程后采用复用persistent loader显式修复，修复控制器31202。GPU门槛重做，500步及指标通过条件不改变。\n')
print('VERIFIED20K_ANALYSIS_SAVED')
