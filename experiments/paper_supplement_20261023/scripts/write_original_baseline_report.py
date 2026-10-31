"""Report the recovered original baseline from verified data; overall campaign stays incomplete."""
import json
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()

def main():
    identity=read('evidence/recovered_original_baseline_identity.json')
    pixels=read('results/baseline_original_local/analysis/verified_summary.json')
    ma=read('results/baseline_original_threeway/summary.json')
    review=read('evidence/baseline_original_threeway_visual_review.json')
    cost=read('results/baseline_original_profile_local/summary.json')
    fullcost=read('results/full_system_profile_v1/analysis/summary.json')
    history=read('evidence/recovered_original_baseline/history_validation_nodes.json')
    assert identity['status']=='PASS_RECOVERED_HISTORICAL_BASELINE_IDENTITY_AND_LOCAL_BACKUP'
    assert pixels['status']=='PASS_LOCAL_MASK_AND_GT_SHA_RECOUNT'
    assert ma['status']=='COMPLETE_THREEWAY_BASELINE_MA_COMPARISON'
    assert review['status']=='PASS' and len(review['figures'])==9
    for row in review['figures']:assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
    assert cost['status']=='COMPLETE' and cost['hardware']==fullcost['hardware']
    assert cost['sample_ids']==fullcost['profile_samples'] and cost['warmup_images_excluded']==1
    assert {row['training_node'] for row in history['validation_nodes']}==set(range(4000,40001,4000))
    d=pixels['cohorts']['ddr']['baseline'];v=pixels['cohorts']['val']['baseline'];e=pixels['cohorts']['idrid']['baseline']
    assert round(d['mIoU_pct'],2)==42.20
    text='# 阶段3：原HRDecoder基线找回与统一复核\n\n'
    text+='已从用户提供的原工程找回原DDR_split613、seed42、best32k权重。原文件、配置、训练日志与本地备份均验SHA，1844个模型状态及923个优化器槽有限；纯推理导出逐张量一致，严格加载无缺失或额外键。主比较使用原权重，不使用v1/v2重训替代。此阶段完成不表示3407配对或整体完成。\n\n'
    text+=f"原checkpoint SHA：`{identity['source_checkpoint_sha256']}`。历史原hook验证mIoU为{identity['historical_validation_mIoU']*100:.8f}%；当前统一原图计数为{v['mIoU_pct']:.8f}%，差{v['mIoU_pct']-identity['historical_validation_mIoU']*100:+.8f}个百分点。DDR测试mIoU为**{d['mIoU_pct']:.8f}%**，四舍五入复核原报告42.20%。不同Torch/CUDA运行时不声称逐像素重放，原报告SE49.98与当前49.9876约差0.01个百分点，不篡改当前计数去凑历史小数。\n\n"
    text+='| 数据与模式 | EX IoU% | HE IoU% | SE IoU% | MA IoU% | mIoU% | MA像素P% | MA像素R% |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for label,row in [('DDR val76',v),('DDR test227',d),('DDR训练→IDRiD-derived保留27图',e)]:
        text+='| '+label+' | '+' | '.join(f"{row[k]:.4f}" for k in ('EX_IoU_pct','HE_IoU_pct','SE_IoU_pct','MA_IoU_pct','mIoU_pct','MA_pixel_precision_pct','MA_pixel_recall_pct'))+' |\n'
    text+='\n全部330图（76验证、227测试、27外部）的原图掩膜、逐图TP/FP/FN、图像/GT/掩膜SHA已保存并独立从GT重算。没有新选权重、改阈值、TTA或融合搜索。其他找回的增强/损失变体不参与基线选择。\n\n'
    text+='| 数据 | 模型 | 小MA检出% | 中MA检出% | 大MA检出% | 全部MA检出% | FP/图 | 病灶P% |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for cohort in ('ddr','idrid'):
        for method,label in [('baseline','原HRDecoder'),('no_h4','去H4'),('full','完整系统')]:
            rows={r['size']:r for r in ma['MA_lesion_rows'] if r['cohort']==cohort and r['method']==method}
            all_=rows['all'];text+=f"| {cohort} | {label} | "+' | '.join(f"{rows[size]['detection_pct']:.4f}" for size in ('small','medium','large','all'))+f" | {all_['false_positives_per_image']:.4f} | {all_['lesion_precision_pct']:.4f} |\n"
    text+='\n8连通、单像素保留、IoU≥0.10一对一；大小边界32/74像素来自DDR训练集。完整分母、图像bootstrap区间与逐图匹配见baseline_original_threeway/summary.json及baseline_original_local/analysis。H4增加检出与误检、降低病灶precision的负结果保留。9例沿用阶段1固定选例并实际审阅；类别相对full/noH4定义，部分still_missed案例反而被原基线检出，部分new_fp也被原基线误检，均保留。\n\n'
    text+=f"同RTX4060、相同三张固定val、另一次warmup排除，原基线实际平均{cost['pipeline_to_native_binary_mask_seconds']['mean']:.6f}秒/图，完整系统{fullcost['pipeline_to_fused_mask_seconds']['mean']:.6f}秒/图。原基线参数{cost['registered_parameters']:,}，真实模型/传输/原图对齐等成本口径与之前冻结协议一致，三图仅作本机描述。\n\n"
    text+='v1曾40k完成但裁块尺度错误，其40.0124%不作为原基线；v2恢复裁块尺度但使用双卡SyncBN，与原单卡BN随机过程不同。找回原模型后v2在2980步停止，2k checkpoint与原日志保留，不称完成40k，不把错误尝试隐去。核心代码在相同BN/合成几何下损失与923组梯度数值检查差值0；该诊断不能证明分布式完整重训必达42.20。\n\n'
    text+='IDRiD为DDR训练→外部冻结测试，未在IDRiD train54重训，不能与原文域内51.94%直接比较。保持方形存储/转换GT溯源缺官方原始独立像素比对、用户确认IDRiD55–81历史隔离、无患者身份、DDR历史test开发限制。原基线找回不自动建立42.20权重→E4的历史加载边，仍按实际原日志区分。\n'
    report=ROOT/'阶段3_原基线找回与复核.md';report.write_text(text,encoding='utf-8')
    evidence_paths=['evidence/recovered_original_baseline_identity.json','evidence/recovered_original_baseline_export.json',
        'evidence/recovered_original_baseline_strict_load.json','evidence/baseline_original_threeway_visual_review.json',
        'evidence/recovered_original_baseline/history_validation_nodes.json','results/baseline_original_local/analysis/verified_summary.json',
        'results/baseline_original_profile_local/summary.json','results/baseline_original_threeway/summary.json',
        'plan/original_baseline_recovery_amendment.json','evidence/baseline_core_equivalence_audit.json']
    receipt={'status':'ORIGINAL_BASELINE_PHASE_COMPLETE','primary_baseline':'Recovered original historical best32k',
        'DDR_test_mIoU_pct':d['mIoU_pct'],'IDRiD_external_mIoU_pct':e['mIoU_pct'],
        'matches_reported_42_20_when_rounded':True,'report_sha256':sha(report.name),
        'input_evidence_sha256':{p:sha(p) for p in evidence_paths},'overall_campaign_complete':False}
    (ROOT/'results/original_baseline_phase_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))

if __name__=='__main__':main()
