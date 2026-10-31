"""Create paper-ready result tables and a candid evidence handoff after all evaluations."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import zipfile
import numpy as np
import scipy
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'full':'完整系统','no_h4':'去掉 H4 MA 修正','no_convnext':'去掉 ConvNeXt','matched_c20':'S20 换为匹配 C20'}
ENGLISH = {'full':'Full system','no_h4':'Without H4','no_convnext':'Without ConvNeXt','matched_c20':'S20 replaced by C20'}


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, obj): p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def f(v): return f'{v:.4f}'
def interval(v): return '['+', '.join(f(x) for x in v)+']'


def pixel_md(table):
    text = '| 系统 | EX IoU | HE IoU | SE IoU | MA IoU | mIoU | MA像素P | MA像素R |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in table:
        text += '| '+LABELS[r['method']]+' | '+' | '.join(f(r[k]) for k in ['EX_IoU_pct','HE_IoU_pct','SE_IoU_pct','MA_IoU_pct','miou_pct','MA_precision_pct','MA_recall_pct'])+' |\n'
    return text


def latex_pixel(table, caption, label):
    rows = [r'\begin{table}[t]',r'\centering\small',r'\caption{'+caption+'}',r'\label{'+label+'}',
            r'\begin{tabular}{lrrrrrrr}',r'\toprule',r'System & EX & HE & SE & MA & mIoU & MA P & MA R \\',r'\midrule']
    for r in table:
        rows.append(ENGLISH[r['method']]+' & '+' & '.join(f'{r[k]:.2f}' for k in ['EX_IoU_pct','HE_IoU_pct','SE_IoU_pct','MA_IoU_pct','miou_pct','MA_precision_pct','MA_recall_pct'])+r' \\')
    rows += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    return '\n'.join(rows)


def main():
    ddr_run = ROOT/'results/ddr_test_local_v1'
    ext_run = ROOT/'results/idrid_holdout_v1'
    profile_run = ROOT/'results/full_system_profile_v1'
    for run in [ddr_run, ext_run, profile_run]:
        assert (run/'completion.json').exists() and not (run/'failure.json').exists()
    d = read(ddr_run/'analysis/summary.json'); e = read(ext_run/'analysis/summary.json'); p = read(profile_run/'analysis/summary.json')
    assert d['status'] == e['status'] == p['status'] == 'COMPLETE'
    assert d['n_images'] == 227 and e['n_images'] == 27
    dependencies=read(ROOT/'evidence/dependency_inventory.json')
    assert all(sha(Path(q))==digest for q,digest in dependencies['sources'].items()), 'Model source changed during evaluation'
    historical_path = ROOT.parent/'paper_writing_20261023/evidence/locked_test_source.json'
    old = read(historical_path)
    old_rows = {r['sid']: np.asarray(r['new_counts'],np.int64) for r in old['per_image']}
    records = [read(ddr_run/'per_image'/(sid+'.json')) for sid in read(ddr_run/'manifest.json')['sample_ids']]
    assert set(old_rows) == {r['sid'] for r in records}
    changes = [{'sid':r['sid'],'new_minus_historical_counts':(np.asarray(r['counts']['full'])-old_rows[r['sid']]).tolist()}
               for r in records if not np.array_equal(np.asarray(r['counts']['full']),old_rows[r['sid']])]
    total_delta = np.sum([np.asarray(r['counts']['full'])-old_rows[r['sid']] for r in records],axis=0)
    replay = {'new_miou_pct':d['pixel_table'][0]['miou_pct'],'historical_miou_pct':old['locked_test_miou_pct'],
              'new_minus_historical_pp':d['pixel_table'][0]['miou_pct']-old['locked_test_miou_pct'],
              'images_same_counts':len(records)-len(changes),'images_changed_counts':len(changes),
              'EX_HE_SE_MA_TP_FP_FN_total_delta':total_delta.tolist(),'changed_images':changes,
              'historical_source_sha256':sha(historical_path),
              'interpretation':'Same fixed recipe/weights and dataset; new runtime reevaluation, not bitwise historical reproduction. No tuning to reduce drift.'}
    write(ROOT/'results/historical_runtime_comparison.json',replay)
    full = next(r for r in d['lesion_table'] if r['method']=='full')
    no = next(r for r in d['lesion_table'] if r['method']=='no_h4')
    report = '# 论文补充实验结果（2026-10-23）\n\n'
    report += '本轮零新增训练，未改检查点、融合权重或0.5阈值。完成DDR 227图四行系统对照、MA病灶分析、27图IDRiD保留子集冻结评估及完整系统实际前向计时。原split613 HRDecoder基线未找回，按用户指示跳过；不能称三模型基线比较已补齐。\n\n'
    report += '## DDR固定系统组件贡献\n\n单位均为百分比；原图逐类TP/FP/FN全局累加后计算。\n\n'+pixel_md(d['pixel_table'])+'\n'
    report += '| 对照 | 完整系统减对照，mIoU百分点 | 95%配对图像bootstrap区间 |\n|---|---:|---|\n'
    for k,r in d['pixel_comparisons'].items():report += '| '+LABELS[k]+' | '+f(r['full_minus_row_pp'])+' | '+interval(r['paired_image_ci95_pp'])+' |\n'
    report += '\n这是固定系统的移除/替换效果，不等同于各模块独立训练的因果效果。C/S同父模型、seed42、原32k日程的20k节点；390个S20导出EMA张量与原检查点逐位一致。未新增种子。\n\n'
    report += '## MA病灶结果\n\n8连通、保留单像素，一对一IoU>=0.10，先最大匹配数量、再最大IoU。合并预测最多命中一个真值，分裂的多余预测计FP。大小边界只从454张训练图的6793个GT组件确定：<=32、33–74、>74像素；这些是像素面积，不能当物理病灶大小。\n\n'
    report += '| 系统 | 小MA检出% | 中MA检出% | 大MA检出% | 全部检出% | FP/图 | 病灶P% | TP/FP/FN |\n|---|---:|---:|---:|---:|---:|---:|---|\n'
    for r in d['lesion_table']:
        values = [r['size_groups'][g]['detection_pct'] for g in ['small','medium','large']]+[r['detection_pct'],r['FP_per_image'],r['lesion_precision_pct']]
        report += '| '+LABELS[r['method']]+' | '+' | '.join(f(v) if v is not None else 'NA' for v in values)+' | '+f"{r['detected_lesions']}/{r['false_lesions']}/{r['missed_lesions']}"+' |\n'
    report += f"\nMA真值组件{full['gt_lesions']}个，MA阳性图{full['MA_positive_images']}/227；FP分母包含所有227图。完整系统相对去H4：全部病灶检出率变化{full['detection_pct']-no['detection_pct']:+.4f}个百分点，小MA检出率变化{full['size_groups']['small']['detection_pct']-no['size_groups']['small']['detection_pct']:+.4f}个百分点，FP/图变化{full['FP_per_image']-no['FP_per_image']:+.4f}。像素recall与病灶检出率分别报告，不互相替代。\n\n"
    report += '各组区间与配对差异区间见DDR analysis/summary.json。例图按预先固定规则，每种结局按sid/组件ID取前三例，包括恢复、仍漏检和新增误检；全部结果保留，没有挑选漂亮案例。彩色边缘仅用于显示，未用于指标滤波。\n\n'
    report += '## IDRiD保留测试子集\n\n用户明确确认IDRiD_55–81从未参与训练、选型、调参或比较方案；本轮冻结27图，没有微调或改阈值。选中S/C路线为DDR-only Q_DDR→F12；项目其他未选中探索用过IDRiD training54，不能声称整个研究从未接触IDRiD。\n\n'+pixel_md(e['pixel_table'])+'\n'
    c = e['pixel_comparisons']['matched_c20']
    report += f"完整系统减C20替换系统：{c['full_minus_row_pp']:+.4f}个百分点，95%配对图像区间{interval(c['paired_image_ci95_pp'])}。所有预定移除/替换行一并保留。MA病灶结果见external_MA_lesion_table.csv。\n\n"
    report += '身份/标签核查：27原图/TXT/108份mask SHA及重新栅格化逐像素恒等；历史重复审计全部1036图的当前RGB指纹与全图/去黑边近似哈希重新验证一致，审计未发现重复。无患者ID，不能证明患者级隔离。标注完整性依据历史用户确认官方GT无损转换；本机无官方原始像素GT可独立对照。使用当前3415–4033方形JPEG的坐标，不声称官方原始几何或官方基准可比性。应标为IDRiD-derived保留子集评估。\n\n'
    report += '## 运行环境与效率\n\n'
    report += f"历史系统分数{old['locked_test_miou_pct']:.4f}%，本轮统一重评{replay['new_miou_pct']:.4f}%，变化{replay['new_minus_historical_pp']:+.4f}个百分点；{len(changes)}/227图计数不同。新旧分数必须分开。原CUDA MSDA历史环境未逐位复现，本轮使用现有reference实现及本地BF16。完整差异清单见historical_runtime_comparison.json；不能把差异全归因一个算子。\n\n"
    params = p['registered_parameters']; timing = p['pipeline_to_fused_mask_seconds']
    report += f"在{p['hardware']}、batch1上固定前三张validation图，另一次warmup不计。完整实际前向到融合掩膜平均{timing['mean']:.3f}秒/图，范围{timing['min']:.3f}–{timing['max']:.3f}秒；含GT计数和压缩落盘平均{p['through_counts_and_persistence_seconds']['mean']:.3f}秒/图。包含所有专家/TTA、H4、U、S、对齐融合、CPU/GPU搬移及跨环境文件交换，不包含模型初始化。不是缓存伙伴的假计时，也没有混入C20额外前向。\n\n"
    report += f"完整系统已实例化注册参数合计{params['full_system_total_registered_parameters']/1e6:.3f}M，部署权重文件{params['weight_artifact_bytes_full']/1024**3:.3f}GiB。参数含各模型复制的reference及辅助模块，不能等同于去重参数或FLOPs。按顺序执行的单进程allocated峰值最大{p['max_single_process_allocated_peak_bytes']/1024**3:.3f}GiB，不含驱动/CUDA上下文或其他应用，不是整机显存峰值。仅3张固定图，未测4卡吞吐。\n\n"
    report += f"另用nvidia-smi每200ms采样整张显卡占用，观察最大{p['sampled_GPU_total_memory']['maximum_MiB']/1024:.3f}GiB（含两个推理进程、上下文及其他应用）；这是采样最大值，可能遗漏短于200ms的瞬时峰值。原始采样CSV一并保留。\n\n"
    stage=p['DDR227_recorded_A_U_S_stage_sum_seconds']
    report += f"另汇总227张不同原尺寸图在四行运行中记录的A+U+S阶段时间和：平均{stage['mean']:.3f}秒、中位{stage['median']:.3f}秒、P95={stage['p95']:.3f}秒。它不包含C前向，但当时C参数常驻，且不计调度/融合/最终落盘，因此仅作阶段时间描述，不能冒充完整部署端到端延迟。\n\n"
    storage={run.name:{'binary_prediction_files':len(list((run/'predictions').glob('*.npz'))),
                       'binary_prediction_bytes':sum(q.stat().st_size for q in (run/'predictions').glob('*.npz')),
                       'per_image_record_bytes':sum(q.stat().st_size for q in (run/'per_image').glob('*.json'))}
             for run in [ddr_run,ext_run,profile_run]}
    write(ROOT/'results/storage_receipt.json',storage)
    report += f"DDR四行二值掩膜持久缓存{storage[ddr_run.name]['binary_prediction_bytes']/1024**2:.3f}MiB；外部四行二值掩膜缓存{storage[ext_run.name]['binary_prediction_bytes']/1024**2:.3f}MiB。完整FP32 S/C概率仅按图临时交换、核验后清理，不是全队列概率缓存。\n\n"
    report += '服务器已验证为4×RTX2080Ti 11GB；本轮未在服务器启动训练。本地长验证保留已完成进度；新外部批次先按固定首图和面积加20%余量估算，未超过一小时才继续。\n\n'
    report += '## 仍然保留的缺口\n\n- 原split613 baseline32000权重缺失；42.20%仍是不可逐图重算的历史报告，不能作为本轮完全同协议基线。\n- 单种子、历史内部test重复开发使用、无患者身份；图像bootstrap只描述固定模型条件下的图像波动，不表示训练重复性。\n- IDRiD当前存储几何与用户提供标注的溯源限制；独立使用历史有用户明确确认，但没有外部机器全部历史运行的独立审计。\n- 本轮不自动证明重大模型创新、临床诊断敏感度或每个历史路径的独立因果贡献。两新种子四次C/S配对训练尚未启动。\n\n'
    report += '## 文件入口\n\n- results/ddr_test_local_v1/analysis：四行CSV、MA分层CSV、完整JSON、真实例图及PDF/SVG/PNG科学图。\n- results/idrid_holdout_v1/analysis：27图外部四行表、MA病灶表、区间和完整JSON。\n- results/full_system_profile_v1/analysis：实际完整系统效率与限制。\n- evidence：资产、数据、血缘、IDRiD隔离和原基线搜索记录。\n- 每个正式run保留逐图JSON和可重算的逐类二值掩膜NPZ；不含模型训练。\n'
    (ROOT/'论文补充实验结果.md').write_text(report,encoding='utf-8')
    tex = '% Requires booktabs. Percent values. New local runtime reevaluation, not exact historical replay.\n'
    tex += latex_pixel(d['pixel_table'],'Fixed-system component substitutions on the 227-image DDR test set. All values are percentages; P/R denote pixel precision/recall.','tab:supp-ddr')+'\n\n'
    tex += latex_pixel(e['pixel_table'],'Frozen evaluation on the 27-image IDRiD-derived held-out subset in stored image coordinates.','tab:supp-idrid')+'\n'
    (ROOT/'results/paper_tables.tex').write_text(tex,encoding='utf-8')
    selection = read(ddr_run/'analysis/example_selection.json')
    for category in ['recovered','still_missed','new_fp']:
        images = [Image.open(r['figure']).convert('RGB') for r in selection['examples'] if r['category']==category]
        if images:
            canvas = Image.new('RGB',(images[0].width,sum(im.height for im in images)),'white')
            y=0
            for im in images:canvas.paste(im,(0,y));y+=im.height
            canvas.save(ddr_run/'analysis/figures'/('ma_'+category+'_all_selected.png'),dpi=(300,300))
    write(ROOT/'results/evaluation_receipt.json',{'status':'COMPLETE','new_training_runs':0,'ddr_images':227,'external_images':27,
          'baseline_available':False,'paired_training_seeds':1,'all_predeclared_rows_reported':True,
          'python_analysis':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
          'ddr_analysis_sha256':sha(ddr_run/'analysis/summary.json'),'external_analysis_sha256':sha(ext_run/'analysis/summary.json'),
          'profile_analysis_sha256':sha(profile_run/'analysis/summary.json'),
          'report_sha256':sha(ROOT/'论文补充实验结果.md'),'scripts':{q.name:sha(q) for q in sorted((ROOT/'scripts').glob('*.py'))}})
    print(json.dumps({'status':'COMPLETE','report':str(ROOT/'论文补充实验结果.md')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
