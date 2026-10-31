"""Render the mirrored evidence as a live Chinese result report; no experiments."""
import argparse,json,re,time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'server_results'

def read(p,default=None):
    try:return json.loads(Path(p).read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError):return {} if default is None else default

def pct(x):return '—' if x is None else f'{x*100:.4f}%'

def recorded_updates(directory,name,health):
    texts=[]
    if directory:
        path=directory/'train.log'
        if path.exists():texts.append(path.read_text(encoding='utf-8',errors='replace'))
    if health.get('campaign',{}).get('active_job')==name:
        texts.append('\n'.join(health.get('active_log_tail',[])))
    matches=[re.findall(r'\[engine\] update (\d+)/(\d+)',text) for text in texts]
    latest=[(int(m[-1][0]),int(m[-1][1])) for m in matches if m]
    if not latest:return '—'
    done,total=max(latest)
    return f'{done}/{total}'

def render():
    main=read(DATA/'work_dirs/campaign_state.json');finish=read(DATA/'work_dirs/finish_state.json')
    health=read(ROOT/'reports/monitor_health.json')
    mirror=read(DATA/'mirror_status.json')
    if health.get('checked_at_epoch',0)>mirror.get('updated_at_epoch',0):
        main=health.get('campaign') or main;finish=health.get('finish') or finish
    status=finish.get('status') if finish.get('status')!='waiting_for_main_training' else main.get('status')
    lines=['# 实验运行与结果报告','',f'更新时间：{datetime.now().isoformat(timespec="seconds")}（本机时间）。状态：`{status or "等待同步"}`。',
        '', '本文件来自服务器报告镜像，自动更新；运行中指标不是最终实验结论。',
        '', '## 固定基线及执行调整','',
        '历史冠军：val **42.6694%**，历史 test **44.2115%**。历史 U0 已完整训练12k，val 38.4464%；历史 H0 在6k为36.9226%，仍在学习，缺少恢复权重。复用这些事实，避免重新做控制臂。',
        '', '当前队列：U1外部→U1 DDR→H1外部→H1 DDR→HACDR DDR；单GPU串行。原计划外部8k，在看新验证前按实际48 train/6 dev统一减到2.4k；DDR保留12k及3k/6k/9k/12k完整验证，停滞且无类别优势时提前终止。',
        '', '外部监督来自本机IDRiD训练54图的多边形栅格化，不是原始官方逐类像素掩膜；27张test未用于训练。MAPLES只有标注、没有MESSIDOR原图。H使用源HRDecoder全局/局部推理几何，但不复现完整论文训练；HACDR是官方骨干加项目解码/损失的适配。实现、快速kernel及微批分组变化，使旧控制只能作描述性比较。',
        '', '## 实际训练与验证','', '|阶段|状态|已记录更新|最佳节点|指标|耗时|','|---|---|---:|---:|---:|---:|']
    directories={p.name:p for p in (DATA/'work_dirs').glob('*') if p.is_dir()}
    jobs=main.get('jobs',{})
    names=list(jobs)+sorted(set(directories)-set(jobs))
    for name in names:
        p=directories.get(name);summary=read(p/'completion_summary.json') if p else {}
        j=jobs.get(name,{})
        if not summary:summary=j.get('summary',{})
        if not summary and not j:continue
        best=summary.get('best',{});value=best.get('mIoU_lesion4');node=best.get('update')
        if not summary and p:
            evaluations=[read(f) for f in p.glob('eval_u*.json')]
            if evaluations:
                e=max(evaluations,key=lambda d:d.get('mIoU_lesion4',-1));value=e.get('mIoU_lesion4');node=e.get('update')
        label='外部dev' if 'external_pretrain' in name else 'DDR val'
        elapsed=summary.get('elapsed_seconds');elapsed='—' if elapsed is None else f'{elapsed/3600:.2f}h'
        updates=summary.get('updates_done')
        if updates is None:updates=recorded_updates(p,name,health)
        lines.append(f'|{name}|{summary.get("status",j.get("status","—"))}|{updates}|{node or "—"}|{label} {pct(value)}|{elapsed}|')
    lines+=['','运行中的更新数取已写入日志的节点，通常最多滞后100次更新。']
    diagnosis=read(ROOT/'reports/A0_6000_decision.json') or read(ROOT/'reports/A0_3000_decision.json')
    if diagnosis:
        lines+=['',diagnosis['summary_zh'],'',diagnosis['decision_zh']]
    incident=read(ROOT/'reports/incident_gpu_idle_repair.json')
    if incident:
        reproduction=incident.get('full_U0_reproduction')
        proof=f'全部76图已通过原有一致性检查，复现mIoU {pct(reproduction["metrics"]["mIoU_lesion4"])}。' if reproduction else '全部76图还须通过原有指标一致性检查。'
        lines+=['','导出故障记录：历史U0的原始验证为原尺寸整图，新导出误套2048滑窗导致一致性检查退出；完整保留检查，恢复历史推理方式。首张val图与历史四类二值预测逐像素一致。'+proof+'冠军已复现42.6691%。监控状态与大权重备份改为独立SSH连接，并由后台每分钟采集服务器健康；模型巡检仍每小时一次。']
    if mirror.get('updated_at_epoch') and time.time()-mirror['updated_at_epoch']>120:
        lines+=['',f'状态镜像已滞后 {int(time.time()-mirror["updated_at_epoch"])} 秒，请结合最近服务器健康检查判断；文件生成时间不代表服务器证据新鲜。']
    if health.get('checked_at_epoch'):
        stamp=datetime.fromtimestamp(health['checked_at_epoch']).isoformat(timespec='seconds')
        gpu=[x.strip() for x in health.get('gpu','').split(',')]
        if len(gpu)==4:
            mem=health.get('memory_current_bytes',0)/2**30
            cap=health.get('memory_max_bytes',0)/2**30
            disk=health.get('disk_free_bytes',0)/2**30
            lines+=['',f'最近服务器检查：{stamp}（本机时区）；GPU利用率 {gpu[0]}，显存 {gpu[1]} / {gpu[2]}，温度 {gpu[3]}°C；内存 {mem:.2f} / {cap:.0f} GiB；磁盘剩余 {disk:.2f} GiB。']
    speed=read(DATA/'reports/speed/H_DDR_selection.json')
    if speed:
        chosen=speed['chosen'];lines+=['',f'H DDR实际校准选择微批次 {chosen["micro_batch"]}，每更新平均 {chosen["mean_seconds"]:.4f} 秒；有效batch8。']
    lines+=['', 'U DDR校准均值0.4649秒/更新；HACDR DDR均值1.3940秒/更新。校准均值不含完整训练的验证、保存等额外耗时。',
             '', '## 融合、扩展与最终部署','']
    fusion=finish.get('final_fusion') or finish.get('fusion')
    if fusion:
        lines += [f'融合交叉拟合：原冠军 {pct(fusion.get("oof_base_miou"))}；所选模式 `{fusion.get("selected_mode")}` 的OOF {pct(fusion.get("selected_oof_miou"))}；固定投票部署规则的描述性val {pct(fusion.get("locked_val_miou"))}。',
                  '', '每类只补一个家族，权重仅0/0.125/0.25/0.5；三折投票后不在完整val重新搜索。checkpoint已用val选过，OOF不是完全独立外层验证；bootstrap同样只作描述。']
    else:lines+=['等待主队列完成后缓存原图坐标fp32概率，再提名最多两种家族做三折受约束融合。']
    for key in ['highres_skipped','second_seed_skipped','semisupervision']:
        if key in finish:lines+=['',f'`{key}`：{json.dumps(finish[key],ensure_ascii=False)}']
    test=finish.get('final_test')
    if isinstance(test,dict):
        lines+=['',f'最终一次锁定test（227图）：冠军 {pct(test["champion_miou"])}；新管线 {pct(test["selected_pipeline_miou"])}；差值 {test["delta"]*100:+.4f} 个百分点；部署决定 `{test["deployment"]}`。',
                '', 'test在历史实验中已多次查看，不能称为独立外部验证。若新管线未超过冠军，保留冠军；不重新挑选候选看test。']
    elif test:lines+=['',str(test)]
    else:lines+=['', '尚未执行本轮最终test，不能宣称超过44.2115%。']
    if finish.get('error'):lines+=['',f'工程问题：{finish["error"]}。工程故障不记为算法负结果。']
    lines+=['', '原始证据及权重位于本目录旁的 `server_results`；完成阶段的最佳checkpoint自动同步。当前报告只是展示已有证据，不会启动额外实验。']
    p=ROOT/'reports/运行结果.md';p.parent.mkdir(exist_ok=True);p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return finish.get('status') in ['complete','engineering_attention_required']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');a=ap.parse_args()
    while True:
        done=render()
        if done or not a.watch:break
        time.sleep(60)
    print('REPORT_READY',flush=True)

if __name__=='__main__':main()
