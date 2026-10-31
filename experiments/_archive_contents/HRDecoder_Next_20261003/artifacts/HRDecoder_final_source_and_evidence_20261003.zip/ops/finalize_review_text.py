"""Turn the reviewed saved results into the final Chinese report; no ready marker."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports'
test=json.loads((ROOT/'server_results/work/final_test.json').read_text(encoding='utf-8-sig'))
assert test['n_images']==227 and not test['target_met']
assert abs(test['old_complete_pipeline']['mIoU_lesion4']-.44780891451100807)<1e-12
assert abs(test['selected_complete_pipeline']['mIoU_lesion4']-.446344519463188)<1e-12
text=(R/'最终实验报告_草稿.md').read_text(encoding='utf-8')
text=text.replace('# HRDecoder严格方案最终报告（汇总草稿，等待主代理审查）',
 '# HRDecoder严格方案最终报告（2026-10-03）\n\n本轮没有产生新的test最佳管线，继续保留旧完整部署 **44.7809%**。val选中的高分辨率G固定25%候选在一次冻结227图test得到44.6345%，相对旧部署−0.1464个百分点。主目标47.2115%未达到；以保留的最佳成绩计算，还差 **2.4306个百分点**。未依据test重新调参或搜索比例。')
start=text.index('验证集选择的最高完整管线：')
end=text.index('当前旧完整部署val',start)
text=text[:start]+'验证集最高合法完整管线为原冠军75%＋U0高分辨率@1000专家25%，原生画布、1024局部窗口、步幅768滑窗，val43.8679%。选择依据为完整76张val，严格大于0.5的固定阈值。新候选未取代原有已部署赢家。\n'+text[end:]
text=text.replace('目标未达到，距离目标还差2.5770点。','此新候选未达到目标，候选本身距离目标2.5770点；保留旧赢家的距离为2.4306点。')
text=text.replace('最终test的按图配对bootstrap增量95%区间：[-0.004093892388482364, 0.0009480918002468813]。该区间使用IoU的小数单位，不替代训练种子差异。',
 '最终test按图配对bootstrap增量95%区间为 **[−0.4094, +0.0948]个百分点**，包含0。它不替代训练种子差异。本次HE IoU降低0.7581点，贡献了主要退步；四类TP都减少，FP也都减少，因此压误报没有转化为总体test收益。')
text=text.replace('seed42终点：2000步，止损理由：Both complete arms remain over 1 percentage point below old deployment; no recent rise; paired-image 95% upper bounds below old deployment。最佳完整候选为{\'update\': 1000, \'output\': \'fixed_half\', \'score\': 0.42355241147194167, \'arm\': \'M0\'}，第二种子门槛False。',
 'seed42在完整匹配2000步终止。两臂各自最佳均落后旧部署超过1点、最近区间回落，2000步相对旧部署的按图95%区间上界均低于0。最佳M完整候选为M0@1000固定半融合42.3552%；第二种子门槛未触发。')
text=text.replace('完整val入围增量的按图95%区间：[-0.0007707208240791922, 0.008521455897028659]（IoU小数单位）。',
 '最佳G的完整val增量为+0.3802点，按图95%区间为[−0.0771, +0.8521]个百分点，包含0。')
text=text.replace('本报告是已完成数值的自动汇总草稿，仍需主代理核查解释、必要的M误差图库、全部必需本机备份及关机回执。此脚本不会写completion_ready或触发关机。',
 '已审阅完整G/M节点、冻结test计数、按图不确定性、G的24张与M的12张人工代表误差图。源代码与原始/执行配置分开归档；精确序列化权重的本机映射见frozen_asset_local_map.json。\n\n必要best/last、冻结协议、全部节点与连续概率的验收状态见[备份验收](backup_verification.json)，授权关机请求的状态见[关机回执](shutdown_state.json)。只有这些验收通过才生成REVIEWED_COMPLETE，再由守卫检查服务器无剩余实验/GPU进程并执行AutoDL官方关机。SSH断开不等于平台已显示OFF。\n\n为减少付费等待，U1原生整图、原生滑窗、长边2048整图三组已分析非赢家G概率逐文件SHA核验后复制到持久数据盘，关机后仍保留，未将它们冒称全部在本机。其余必需组件及稳定M最佳概率须完成本机SHA备份。原赢家和源权重不删除。\n\n下一组新实验按用户要求先提高有效GPU吞吐，实施方向与数值/数据流验收见[下一轮GPU吞吐要求](下一轮GPU吞吐要求.md)。本轮没有为利用率曲线中途改变micro4×2、Dice或两臂数据。每小时一次模型巡检，本地脚本继续监控、备份与关机守卫。')
path=R/'HRDecoder_严格实验最终报告_20261003.md';path.write_text(text,encoding='utf-8')
print(str(path))
