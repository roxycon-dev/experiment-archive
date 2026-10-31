"""Verify archived validation counts, maintain per-point tables and reporting ledger."""
import argparse
from datetime import datetime,timezone,timedelta
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RUN=ROOT/'cache_execution/server_receipts/training_runs/v3'
CLASSES=('EX','HE','SE','MA')

def ratio(numerator,denominator): return numerator/denominator*100 if denominator else 0.

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mark-reported',type=int,nargs='*',default=[])
    args=parser.parse_args()
    ledger_path=HERE/'validation_reporting.json'
    ledger=json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else {
        'reported_steps':[],'report_every_validation':True,'validation_steps':list(range(4000,40001,4000)),
        'validation_images':76,'class_order':list(CLASSES),'primary_result':'40000 fixed endpoint',
        'report_metrics':['mIoU','per_class_IoU','per_class_precision','per_class_recall','delta_percentage_points','best_val_step'],
        'last_reported_at':None}
    historical_path=HERE/'historical_checkpoint_comparison.json'
    historical=json.loads(historical_path.read_text(encoding='utf-8')) if historical_path.exists() else {}
    baseline_by_step={r['step']:r['mIoU'] for r in historical.get('baseline_history',[])}
    records=[]
    for path in sorted(RUN.glob('val_*.json')):
        # Older download helpers created zero-byte placeholders for future points.
        # They are not completed validation receipts.
        if path.stat().st_size==0: continue
        step=int(path.stem.split('_')[1])
        assert step in ledger['validation_steps']
        raw=json.loads(path.read_text(encoding='utf-8'))
        per_class=[]
        for index,name in enumerate(CLASSES):
            tp,fp,fn=raw['counts'][index]
            assert all(isinstance(n,int) and n>=0 for n in (tp,fp,fn))
            computed={'IoU_percent':ratio(tp,tp+fp+fn),'precision_percent':ratio(tp,tp+fp),
                'recall_percent':ratio(tp,tp+fn),'Dice_F1_percent':ratio(2*tp,2*tp+fp+fn)}
            for key in ('IoU_percent','precision_percent','recall_percent'):
                assert math.isfinite(raw[key][index]) and abs(raw[key][index]-computed[key])<1e-8,(path,key,name)
            per_class.append(dict(name=name,counts=[tp,fp,fn],**computed))
        assert abs(sum(c['IoU_percent'] for c in per_class)/4-raw['mIoU_percent'])<1e-8
        previous=records[-1] if records else None
        record={'step':step,'mIoU_percent':raw['mIoU_percent'],'classes':per_class,'counts_verified':True,
            'mIoU_delta_pp':raw['mIoU_percent']-previous['mIoU_percent'] if previous else None,
            'class_IoU_delta_pp':[c['IoU_percent']-p['IoU_percent'] for c,p in zip(per_class,previous['classes'])] if previous else None,
            'source':str(path)}
        baseline=baseline_by_step.get(step)
        record['historical_baseline_mIoU_percent']=baseline
        record['v3_minus_historical_baseline_pp']=raw['mIoU_percent']-baseline if baseline is not None else None
        records.append(record)
    assert set(args.mark_reported).issubset({r['step'] for r in records})
    if args.mark_reported:
        ledger['reported_steps']=sorted(set(ledger['reported_steps'])|set(args.mark_reported))
        ledger['last_reported_at']=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    ledger['available_steps']=[r['step'] for r in records]
    ledger['check_interval_minutes']=45
    ledger['last_checked_at']=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    ledger_path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    best=max(records,key=lambda r:r['mIoU_percent']) if records else None
    summary={'validation_images':76,'records':records,'best_val_step':best['step'] if best else None,
        'best_val_mIoU':best['mIoU_percent'] if best else None,
        'unreported_steps':[r['step'] for r in records if r['step'] not in ledger['reported_steps']]}
    (HERE/'validation_series.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    text=['# v3 验证指标与趋势','',
        '固定全量76张val，阈值严格>0.5；类别顺序EX/HE/SE/MA。数值单位为%，变化单位为百分点。',
        '原始TP/FP/FN已与IoU、precision和recall逐项核对；Dice/F1由2TP/(2TP+FP+FN)额外计算，不是训练损失。',
        '每4k汇报一次，40k固定端点为主结果，best-val为次。单次运行不作为机制因果证明。','',
        '| 步骤 | mIoU | ΔmIoU | EX IoU | HE IoU | SE IoU | MA IoU |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for record in records:
        delta='—' if record['mIoU_delta_pp'] is None else f"{record['mIoU_delta_pp']:+.2f}"
        values=' | '.join(f"{c['IoU_percent']:.2f}" for c in record['classes'])
        text.append(f"| {record['step']} | {record['mIoU_percent']:.2f} | {delta} | {values} |")
    if not records: text+=['','尚无已完成的验证点，不计算趋势。']
    if best: text+=['',f"截至当前best-val：{best['step']}步，mIoU {best['mIoU_percent']:.2f}%。"]
    if baseline_by_step:
        text+=['','## 历史同节点参照','',
            '| 步骤 | v3 mIoU | 原HRDecoder mIoU | v3变化（百分点） |',
            '|---:|---:|---:|---:|']
        for record in records:
            baseline=record['historical_baseline_mIoU_percent']
            if baseline is not None:
                text.append(f"| {record['step']} | {record['mIoU_percent']:.2f} | {baseline:.2f} | {record['v3_minus_historical_baseline_pp']:+.2f} |")
        text+=['','历史结果来自已核对的76张相同val标签记录，未重新运行对照；不同随机流、有效域与精度实现仍限制因果解释。没有对应历史节点时不插值。原baseline固定40k为38.835%，best-val为39.390%@32k。验证与测试成绩不混比。']
    for record in records:
        text+=['',f"## {record['step']}步",'',
            '| 类别 | IoU | ΔIoU | precision | recall | Dice/F1（计算） |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for index,c in enumerate(record['classes']):
            delta='—' if record['class_IoU_delta_pp'] is None else f"{record['class_IoU_delta_pp'][index]:+.2f}"
            text.append(f"| {c['name']} | {c['IoU_percent']:.2f} | {delta} | {c['precision_percent']:.2f} | {c['recall_percent']:.2f} | {c['Dice_F1_percent']:.2f} |")
        if record['mIoU_delta_pp'] is None:
            text+=['','首个点建立验证基线，尚不能判断提升趋势、收敛或过拟合。']
        else:
            text+=['',f"相对上一点，mIoU变化{record['mIoU_delta_pp']:+.2f}个百分点。趋势解释须结合各类precision/recall与多个验证点，不凭单个点宣布收敛或过拟合。"]
        analysis=HERE/'validation_analysis'/('val_%05d.md'%record['step'])
        if analysis.exists(): text+=['',analysis.read_text(encoding='utf-8').strip()]
    (HERE/'验证指标与趋势.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    state_path=ROOT/'cache_execution/handover_state.json'
    state=json.loads(state_path.read_text(encoding='utf-8'))
    state.update(validation_reporting_required=True,validation_reporting_ledger=str(ledger_path),
        validation_report=str(HERE/'验证指标与趋势.md'),reported_validation_steps=ledger['reported_steps'],
        available_validation_steps=ledger['available_steps'],heartbeat_check_interval_minutes=45)
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
