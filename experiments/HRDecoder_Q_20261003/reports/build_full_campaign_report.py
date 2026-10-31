"""Report recovered paired evidence; no model inference or selection changes."""
from datetime import datetime
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qlib.common import CLASSES, read_json, sha, utc_now, write_json
from qlib.metrics import summarize

RP = ROOT / 'reports'
FINAL = ROOT / 'server_results/continuation_evidence/reports'
ARMS = ('Q_DDR', 'Q_MIX')
SEEDS = (42, 2024)
UPDATES = (2000, 4000, 8000, 12000, 16000, 20000, 24000)


def score(counts):
    a = np.asarray(counts)
    return (a[..., 0] / np.maximum(a.sum(-1), 1)).mean(-1) * 100


def table(lines, header, rows):
    lines += ['', '|'+ '|'.join(header)+'|', '|'+ '|'.join('---' for _ in header)+'|']
    lines += ['|'+'|'.join(map(str,row))+'|' for row in rows]
    lines.append('')


def main():
    evidence = read_json(RP / 'continuation_final_evidence_backup_receipt.json')
    assert evidence['stage'] == 'COMPLETE_SHA_VERIFIED'
    lock = read_json(FINAL / 'continuation_final_selection.json')
    assert sha(FINAL / 'continuation_final_selection.json') == evidence['lock_sha256']
    summary = read_json(FINAL / 'continuation_final_test/summary.json')
    assert summary['status'] == 'COMPLETE' and summary['lock_sha256'] == evidence['lock_sha256']
    nodes = {}; backups = {}; invocation_seconds = {}
    for arm in ARMS:
        for seed in SEEDS:
            name = '%s_seed%s' % (arm, seed)
            backup = read_json(RP / ('run_backup_%s_24k.json' % name))
            assert backup['complete'] and len(backup['files']) == 327 and backup['endpoint'] == 24000
            inventory = {row['path']:row for row in backup['files']}
            run = ROOT / 'server_results/runs' / name
            status = read_json(run / 'status.json')
            assert status['stage'] == 'COMPLETE' and status['completed'] == 24000
            seconds = status['total_seconds_this_invocation']
            prior_status = ROOT / 'server_results/stage_backups' / (name+'_16000') / 'runs' / name / 'status.json'
            if prior_status.exists():
                seconds += read_json(prior_status)['total_seconds_this_invocation']
            invocation_seconds[name] = seconds
            nodes[name] = {}
            for update in UPDATES:
                relative = 'runs/%s/val_u%s/metrics.json' % (name, update)
                path = ROOT / 'server_results' / relative
                assert sha(path) == inventory[relative]['sha256']
                m = read_json(path)
                assert m['update'] == update and m['n_images'] == 76
                for formula in ('single', 'fixed_fusion', 'p0'):
                    assert summarize(np.asarray(m[formula]['counts'])) == m[formula]
                nodes[name][str(update)] = {f:m[f] for f in ('single','fixed_fusion','p0')}
            backups[name] = dict(files=len(backup['files']),bytes=backup['total_bytes'],
                                 verified_at=backup['verified_at'],receipt_sha256=sha(RP/('run_backup_%s_24k.json'%name)))
    validation = dict(created_at=utc_now(), common_endpoint=24000, classes=CLASSES,
                      nodes=nodes, source='Complete recovered formal metrics; exact TP/FP/FN aggregation',
                      probabilities_threshold='strictly greater than 0.5', new_inference=False)
    write_json(RP / 'paired_validation_all_nodes_24k.json', validation)
    paired = []
    for seed in SEEDS:
        a = nodes['Q_DDR_seed%s'%seed]['24000']['single']
        b = nodes['Q_MIX_seed%s'%seed]['24000']['single']
        paired.append(dict(seed=seed,DDR_single_pct=a['miou_pct'],MIX_single_pct=b['miou_pct'],
                           MIX_minus_DDR_pp=b['miou_pct']-a['miou_pct'],
                           per_class_delta_pp={c:b['per_class'][c]['iou_pct']-a['per_class'][c]['iou_pct'] for c in CLASSES}))
    delta = [x['MIX_minus_DDR_pp'] for x in paired]
    comparison = dict(created_at=utc_now(),update=24000,formula='single',seeds=paired,
        mean_MIX_minus_DDR_pp=float(np.mean(delta)),range_MIX_minus_DDR_pp=[min(delta),max(delta)],
        exposure_windows={'Q_DDR':{'DDR':192000},'Q_MIX':{'DDR':144000,'IDRiD':24000,'MAPLES':24000}},
        interpretation='The fixed MIX protocol did not improve this paired DDR validation comparison. Exposure and annotation source differ; not a universal conclusion about external supervision.',
        IDRiD_source_exception='User-authorized stored uncropped assertion and polygon-derived masks; official native pixel GT not restored')
    write_json(RP / 'supervision_comparison_common24k.json', comparison)

    tests = {s:read_json(FINAL / ('continuation_final_test/seed%s.json'%s)) for s in SEEDS}
    ids = sorted(tests[42]['images'])
    assert len(ids) == 227 and set(ids) == set(tests[2024]['images'])
    p0_counts = np.asarray([tests[42]['images'][i]['p0'] for i in ids],dtype=np.int64)
    p0 = summarize(p0_counts.sum(0))
    assert np.array_equal(p0_counts,np.asarray([tests[2024]['images'][i]['p0'] for i in ids]))
    indices = np.random.default_rng(100303).integers(0,227,(2000,227))
    p0_boot = score(p0_counts[indices].sum(1))
    per_seed = {}; bootstrap = []; error_rows = []
    for seed, result in tests.items():
        assert result['status'] == 'COMPLETE' and result['lock_sha256'] == evidence['lock_sha256']
        counts = np.asarray([result['images'][i]['selected'] for i in ids],dtype=np.int64)
        assert summarize(counts.sum(0)) == result['selected']
        assert result['p0'] == p0
        values = score(counts[indices].sum(1)); bootstrap.append(values)
        timings = [result['images'][i]['q_inference_seconds'] for i in ids]
        per_seed[str(seed)] = dict(score_pct=result['selected']['miou_pct'],
            delta_P0_pp=result['selected']['miou_pct']-p0['miou_pct'],
            image_percentile95_pct=np.quantile(values,[.025,.975]).tolist(),
            paired_image_percentile95_pp=np.quantile(values-p0_boot,[.025,.975]).tolist(),
            Q_per_image_seconds_mean=float(np.mean(timings)),Q_per_image_seconds_median=float(np.median(timings)),
            Q_per_image_seconds_p90=float(np.quantile(timings,.9)))
        before=np.asarray(result['p0_lesion_buckets']); after=np.asarray(result['selected_lesion_buckets'])
        assert np.array_equal(before[:,:,3].sum(1),p0_counts.sum(0)[:,2])
        assert np.array_equal(after[:,:,3].sum(1),counts.sum(0)[:,2])
        for ci,c in enumerate(CLASSES):
            for bi,label in enumerate(('≤16','17–64','65–256','257–1024','>1024')):
                o,n=before[ci,bi],after[ci,bi]
                error_rows.append([seed,c,label,int(o[0]),int(o[1]),int(n[1]),int(o[3]-o[4]),int(n[3]-n[4]),int(o[4]),int(n[4])])
    mean_boot = np.mean(bootstrap,axis=0)
    uncertainty = dict(image_resamples=2000,seed=100303,unit='paired image',includes_training_seed_variance=False,
        mean_delta_P0_image_percentile95_pp=np.quantile(mean_boot-p0_boot,[.025,.975]).tolist())
    write_json(RP / 'final_test_uncertainty_and_error_buckets.json',dict(per_seed=per_seed,uncertainty=uncertainty,
        error_bucket_columns=['seed','class','area_pixels','lesions','P0_whole_missed','selected_whole_missed','P0_body_FN','selected_body_FN','P0_boundary_FN','selected_boundary_FN'],
        error_bucket_rows=error_rows,boundary_definition=tests[42]['boundary_definition']))
    preflight=read_json(ROOT/'server_results/reports/gpu_preflight.json')
    performance={k:v for k,v in preflight['performance'].items() if k!='timing_rows'}
    result=dict(status='EXPERIMENTS_AND_FINAL_EVALUATION_COMPLETE',created_at=utc_now(),
        selection=lock['selection'],lock_sha256=evidence['lock_sha256'],test_action=lock['test_action'],
        final_test=summary,supervision_comparison=comparison,per_seed_test=per_seed,uncertainty=uncertainty,
        formal_training_invocation_seconds=invocation_seconds,
        training_plus_val_checkpoint_wall_hours=sum(invocation_seconds.values())/3600,
        accepted_training_preflight_performance=performance,backups=backups,
        actual_charged_CNY=None,actual_hourly_rate_CNY=None,full_system_peak_memory='Not separately instrumented',
        automatic_extra_experiments=False,shutdown_or_platform_OFF_verified=False)
    write_json(RP/'original_plan_experiment_result.json',result)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    colors={'Q_DDR':'#1565c0','Q_MIX':'#c45121'}
    for axis,formula,title in zip(axes,('single','fixed_fusion'),('Q single model','Fixed 0.75 P0 + 0.25 Q')):
        for arm in ARMS:
            for seed in SEEDS:
                ys=[nodes['%s_seed%s'%(arm,seed)][str(u)][formula]['miou_pct'] for u in UPDATES]
                axis.plot(np.asarray(UPDATES)/1000,ys,marker='o',markersize=4,color=colors[arm],
                          linestyle='-' if seed==42 else '--',label='%s seed%s'%(arm,seed))
        axis.axhline(43.48770322103657,color='#666',linewidth=1,linestyle=':',label='P0 validation')
        axis.set(title=title,xlabel='Optimizer updates (thousands)',ylabel='DDR validation mIoU (%)',xticks=[2,4,8,12,16,20,24])
        axis.grid(alpha=.2);axis.legend(fontsize=8)
    for suffix in ('png','svg'):fig.savefig(RP/('HRDecoder_paired_validation_20261004.'+suffix),dpi=160)
    plt.close(fig)

    selected=lock['selection']['selected']; mean=summary['mean_test_pct']; improvement=mean-p0['miou_pct']
    lines=['# HRDecoder 原方案完整实验报告（2026-10-04）','',
        f"本轮没有达到 **test47.2115%** 目标。最终固定配方两种子平均 **{mean:.6f}%**，相对P0 **{improvement:+.6f}点**，距目标 {47.2115-mean:.6f}点。",
        f"24k同终点单模配对，外部混合监督MIX−DDR平均 **{comparison['mean_MIX_minus_DDR_pp']:+.6f}点**，两个种子都下降；本冻结混合方案未显示监督收益。",
        '四条正式训练均完成24000更新。完成规定的一次配对复核后停止本轮追加；不换损失、阈值、TTA、架构或更多种子。报告完成不等于平台OFF，关机/清理状态另见original_plan_completion.json。','',
        '## 同终点监督比较',
        '只比较相同24000更新数、相同种子的单模，未混入P0，未各自挑最优节点。']
    table(lines,['种子','DDR单模，%','MIX单模，%','MIX−DDR，点'],
          [[x['seed'],f"{x['DDR_single_pct']:.6f}",f"{x['MIX_single_pct']:.6f}",f"{x['MIX_minus_DDR_pp']:+.6f}"] for x in paired])
    lines += ['DDR臂每种子暴露192000个DDR窗口；MIX臂144000个DDR+24000个IDRiD+24000个MAPLES窗口。MIX前12k每批DDR4/IDRiD2/MAPLES2，后12k全DDR。比较同时改变监督来源和DDR曝光，不能据此推成所有外部监督无效。',
        'IDRiD54训练图使用用户确认未裁剪的存储方图与独立多边形栅格掩膜；未证完整性的空类unknown逐类ignore，27测试图隔离，未恢复或冒称官方像素GT。MAPLES使用官方138训练图/掩膜与配套MESSIDOR原图，60测试图隔离，保留逐类uncertain ignore与确认阴性。192外部训练图与DDR全部757图的图像去重及20张叠图验收通过；没有患者ID独立性保证。']
    table(lines,['类别','seed42 MIX−DDR IoU，点','seed2024 MIX−DDR IoU，点'],
          [[c,f"{paired[0]['per_class_delta_pp'][c]:+.6f}",f"{paired[1]['per_class_delta_pp'][c]:+.6f}"] for c in CLASSES])
    lines += ['## 最终选择与测试',
        f"根据两种子共同晚期val，一次锁定 {selected['arm']}@{selected['update']} 的 {selected['formula']}，平均val {selected['mean_val_pct']:.6f}%。候选总数 {len(lock['selection']['all_candidates'])}。",
        f"冻结时间 {lock['created_at']}；新锁SHA256 `{evidence['lock_sha256']}`。原DDR16k锁SHA256 `{lock['prior_final_selection_sha256']}` 保留。",
        f"测试动作 `{lock['test_action']}`。相同已测身份仅复用原test227逐图计数及其SHA；不进行重复推理，不根据test改选。预定部署seed42，两种子均披露。",
        '固定融合完整公式为0.75P0+0.25Q=0.5625C+0.1875U0+0.25Q；C保留D/E/B/E4及H4残差。该成绩属于7份权重的系统收益，不能写成Q单模型或某个模块提升3点。']
    table(lines,['模型','test，%','相对P0，点'],[['P0',f"{p0['miou_pct']:.6f}",'—']]+[[f'seed{s}',f"{per_seed[str(s)]['score_pct']:.6f}",f"{per_seed[str(s)]['delta_P0_pp']:+.6f}"] for s in SEEDS]+[['两种子平均',f'{mean:.6f}',f'{improvement:+.6f}']])
    lines += [f"两种子范围 {summary['range_test_pct']}%，均未达目标。两种子均值相对P0的配对图像95%区间 {uncertainty['mean_delta_P0_image_percentile95_pp']}点；图像ID重采样2000次后分别累计计数计算mIoU再平均，不包含训练种子方差。n=2不是未来训练成功保证。",
        'DDR val76/test227已有历史使用；此次承接发生在已测DDR16k之后，结果属于探索性延续。官方大规模DINOv3预训练逐图清单不公开，不能证明与DDR绝无重合。外部独立测试和论文消融的先决条件未建立，本轮不自动追加。']
    rows=[]
    for label,m in [('P0',p0)]+[(f'seed{s}',tests[s]['selected']) for s in SEEDS]:
        for c in CLASSES:
            a=m['per_class'][c];rows.append([label,c,a['tp'],a['fp'],a['fn'],f"{a['iou_pct']:.6f}",f"{100*a['precision']:.4f}",f"{100*a['recall']:.4f}"])
    table(lines,['身份','类别','TP','FP','FN','IoU，%','precision，%','recall，%'],rows)
    lines += ['## 全部正式验证节点',
        '阈值严格>.5；EX/HE/SE/MA独立通道；原坐标全图全数据集累计TP/FP/FN后四类宏平均，非逐图mIoU平均。28份正式节点的全部四类计数、IoU、precision/recall见paired_validation_all_nodes_24k.json。',
        '![完整验证曲线]('+str((RP/'HRDecoder_paired_validation_20261004.png').as_posix())+')']
    table(lines,['训练臂','种子','更新数','单模，%','固定融合，%'],
          [[arm,seed,u,f"{nodes['%s_seed%s'%(arm,seed)][str(u)]['single']['miou_pct']:.6f}",f"{nodes['%s_seed%s'%(arm,seed)][str(u)]['fixed_fusion']['miou_pct']:.6f}"] for arm in ARMS for seed in SEEDS for u in UPDATES])
    lines += ['## 原坐标误差分桶',
        'GT按8连通域分组；面积桶为像素面积，不是体积。边界为3×3零边界腐蚀形成的一像素内圈，主体是其余GT像素。各类FN与测试聚合计数精确一致。']
    table(lines,['种子','类别','面积桶','病灶数','P0完全漏检','选中完全漏检','P0主体FN','选中主体FN','P0边界FN','选中边界FN'],error_rows)
    lines += ['## 固定实现、成本与复现交付',
        'DINOv3官方ViT-B/16 LVD，层2/5/8/11，128通道GN解码与RGB支路；1024窗口/128halo/中央768；micro2×accum4 BF16，masked BCE与微批Dice各0.5，AdamW头3e-4/末四层5e-6；原24k调度从起点预定义，恢复不重置。所有核心源码、配置及模型身份保持锁定。',
        f"真实训练预检峰值allocated {performance['peak_allocated_bytes']/1024**3:.4f}GiB、reserved {performance['peak_reserved_bytes']/1024**3:.4f}GiB；末四层100次更新平均 {performance['unfrozen_seconds_per_update_mean']:.6f}秒/更新，数据等待 {performance['data_wait_seconds_mean']:.6f}秒。GPU利用率曲线不替代吞吐；完整系统推理峰值未另采集。",
        f"正式训练调用墙钟合计 {result['training_plus_val_checkpoint_wall_hours']:.6f}小时，含val/保存，包含保存的16k原调用与24k恢复调用，不能当成精确GPU核函数时长或账单。实际费率和收费回执未提供，记录为unknown，不虚构价格。"]
    table(lines,['身份','Q每图均值，秒','中位，秒','P90，秒'],[[s,f"{per_seed[str(s)]['Q_per_image_seconds_mean']:.6f}",f"{per_seed[str(s)]['Q_per_image_seconds_median']:.6f}",f"{per_seed[str(s)]['Q_per_image_seconds_p90']:.6f}"] for s in SEEDS])
    lines += ['推理时间仅含Q前向与CPU拼接，不含P0、SHA或病灶诊断。Q参数88,263,748；选中完整系统需D/E/B/E4/H4/U0/Q seed42共7份权重。']
    table(lines,['训练身份','完整备份文件数','字节','本机SHA核验时间'],[[name,b['files'],b['bytes'],b['verified_at']] for name,b in backups.items()])
    lines += ['25项冻结配置/采样流/身份共160,622,777字节及251项源文件本机SHA通过；四run各327文件完整回收，保留全部模型、last、恢复状态、日志、正式逐图计数和必要概率。原16k可变证据另保存在stage_backups，原测试和旧锁未改写。',
        '新最终输出逐文件本机SHA回执：continuation_final_evidence_backup_receipt.json。完整最终选择、test227两个种子逐图计数、旧证据复用回执位于server_results/continuation_evidence/reports。',
        '清理只按约定依赖审计和明确文件清单执行，冠军/P0/最终选中/必要基准/源码配置/日志计数/未知用途资产保留。平台OFF核验及实际收费状态另见original_plan_completion.json，不能用SSH断开证明停费。']
    report=RP/'HRDecoder_原方案完整实验报告_20261004.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    outputs=[report,RP/'original_plan_experiment_result.json',RP/'paired_validation_all_nodes_24k.json',RP/'supervision_comparison_common24k.json',RP/'final_test_uncertainty_and_error_buckets.json',RP/'HRDecoder_paired_validation_20261004.png',RP/'HRDecoder_paired_validation_20261004.svg']
    write_json(RP/'original_plan_report_receipt.json',dict(stage='COMPLETE_LOCAL_SHA_VERIFIED',created_at=utc_now(),
        lock_sha256=evidence['lock_sha256'],new_inference=False,selection_changed=False,
        files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in outputs]))
    print('COMPLETE_REPORT',comparison['mean_MIX_minus_DDR_pp'],mean,report)


if __name__ == '__main__':
    main()
