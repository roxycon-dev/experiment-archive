"""Evidence-based report; no new inference, scans, or test-derived decisions."""
from decision import ENDPOINT,NODES,FORMULAS
from train import ROOT
from qlib.common import read_json,utc_now,write_json


def main():
    lines=['# HRDecoder 2026-10-04 实验结果','',f'报告生成：{utc_now()}','',
           '固定口径：EX/HE/SE/MA 原图像素TP/FP/FN汇总后的四类宏平均IoU，阈值严格>0.5。',
           '本轮相对对应种子的当前R计算新增收益；部署身份固定seed42。','',
           '|实验|种子|更新|单模%|25%新模型%|50%新模型%|状态|','|---|---:|---:|---:|---:|---:|---|']
    outcomes={}
    for arm in ('E_NEG','F4','F12'):
        for seed in (42,2024):
            run=ROOT/f'runs/{arm}_seed{seed}'
            if not (run/'status.json').exists():continue
            status=read_json(run/'status.json');outcomes[f'{arm}_seed{seed}']=status
            for node in NODES[arm]:
                file=run/f'val_u{node}/metrics.json'
                if file.exists():
                    m=read_json(file)
                    lines.append(f'|{arm}|{seed}|{node}|'+ '|'.join(f'{m[f]["miou_pct"]:.6f}' for f in FORMULAS)+f'|{status["stage"]}|')
            if arm=='E_NEG' and status['stage']=='COMPLETE':
                old=read_json(ROOT/f'historical/Q_MIX_seed{seed}/metrics.json')
                new=read_json(run/'val_u24000/metrics.json')
                write_json(run/'paired_negative_diagnostic.json',{'old_MIX_single':old['single'],
                    'new_negative_fixed_single':new['single'],'delta_single_miou_pp':new['single']['miou_pct']-old['single']['miou_pct'],
                    'SE_false_positive_delta':new['single']['per_class']['SE']['fp']-old['single']['per_class']['SE']['fp'],
                    'valid_semantics_only_treatment':True,'initialization_evidence':'Deterministic verified-source reconstruction; no archived u0 tensor digest'})
    lines.extend(['','科学比较为E_NEG24k对旧MIX24k、F12新12k对F4新12k的同种子单模结果。',
                  '若某F臂4k止损，共同终点仅为4k，不声称完整12k配对；另一健康臂仍可完成。',''])
    paired=[]
    for seed in (42,2024):
        statuses=[outcomes.get(f'{arm}_seed{seed}',{}) for arm in ('F4','F12')]
        endpoint=12000 if all(s.get('stage')=='COMPLETE' for s in statuses) else 4000
        paths=[ROOT/f'runs/{arm}_seed{seed}/val_u{endpoint}/metrics.json' for arm in ('F4','F12')]
        if all(p.exists() for p in paths):
            m4,m12=map(read_json,paths)
            paired.append({'seed':seed,'common_endpoint':endpoint,'F12_minus_F4_single_pp':m12['single']['miou_pct']-m4['single']['miou_pct']})
    write_json(ROOT/'reports/scientific_comparisons.json',paired)
    for p in paired:
        lines.append(f'种子{p["seed"]}，共同{p["common_endpoint"]}更新：F12−F4单模 {p["F12_minus_F4_single_pp"]:+.6f}点。')
    lock=read_json(ROOT/'reports/final_selection.json')
    lines.extend(['',f'最终候选数量：{len(lock["candidates"])}，上限9；未完成两个种子的配方不参与最终选型。'])
    if lock['selected']:
        lines.append('验证锁定：'+str(lock['selected']))
    else:
        lines.append('没有具备两个完整种子的终点候选，保留当前R部署。')
    if not lock['test_authorized']:
        lines.append('未达到test晋级条件，不运行新test227。')
    test_path=ROOT/'reports/final_test/summary.json'
    if test_path.exists():
        test=read_json(test_path)
        lines.extend(['', '历史冠军test44.2115%，P0 test44.780891%，当前R均值45.699312%。',
                      f'新配方test均值{test["test_mean"]:.6f}%；seed42仍为预定部署。',
                      '两种子均达到目标：'+str(test['target47_2115_both_seeds']),
                      '只有均值达到目标：'+str(test['target47_2115_mean_only'])])
        for seed,r in test['seed_results'].items():
            lines.append(f'seed{seed}: {r["miou_pct"]:.6f}%，相对对应R {r["gain_vs_R_pp"]:+.6f}点。')
    lines.extend(['','图像bootstrap只刻画图像重采样波动；两次训练差异另外报告。历史val/test重复使用带来选择偏差，不能称为独立盲态确认。',
                  '正确的IDRiD阴性语义不因性能不涨而回滚。没有自动追加ViT-Adapter或其他后备实验。',
                  '关机状态以本地回收收据和平台OFF证据为准；本报告生成不代表服务器已经关闭。'])
    if (ROOT/'reports/performance_revision.json').exists():
        lines.extend(['','训练执行优化：原图/掩膜一次解码后放入只读共享RAM缓存，8个worker预取，监督计数移至worker。',
            'micro2×accum4、BF16、模型/损失/优化器/LR、固定窗口流及val/test评估不变。',
            'E_NEG在u1600完整断点切换；保留原始断点及日志，只迁移源代码和执行配置身份。',
            '确定性CUDA仅用于优化前后的一致性对照：权重、梯度、优化器、RNG、loss和来源审计一致；正式训练沿用历史CUDA设置。',
            '执行配置与加速证据见performance_acceptance.json和performance_revision.json。'])
    (ROOT/'reports/最终实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPORT_READY')


if __name__=='__main__':main()
