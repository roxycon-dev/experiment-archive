"""Explicit user-amended seed42 exploration; original candidate formulas/ties."""
import sys,json,traceback,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'reports'))
from qlib.common import read_json,write_json,sha,digest,utc_now
import decision

def choose_single(candidates,references):
    eligible=[]
    for c in candidates:
        score=c['scores']['42'];gain=score-references['42']
        c.update(gains_pp={'42':gain},mean_gain_pp=gain,mean_score=score,eligible=gain>=.5,seed_count=1)
        if c['eligible']:eligible.append(c)
    return min(eligible,key=lambda c:(-c['mean_score'],0 if c['formula']=='single' else 1,c['update'],decision.ARMS.index(c['arm']),decision.FORMULAS.index(c['formula']))) if eligible else None

def main():
    amendment=ROOT/'reports/single_seed_parallel_amendment.json';assert read_json(amendment)['seeds']==[42]
    assert not (ROOT/'reports/final_selection_lock.json').exists(), 'Do not refreeze an existing selection'
    for arm in decision.ARMS:assert read_json(ROOT/f'runs/{arm}_seed42/status.json')['stage'] in ('COMPLETE','SCIENTIFIC_STOP_INCOMPLETE')
    try:
        write_json(ROOT/'reports/driver_state.json',{'stage':'SINGLE_SEED_FINAL_SELECTION','pid':os.getpid(),'time':utc_now()})
        decision.SEEDS=(42,);decision.choose=choose_single
        result=decision.finalize();result.update(seed_scope=[42],amendment_sha256=sha(amendment),single_seed_exploratory=True)
        test=None
        if result['test_authorized']:
            path=ROOT/'reports/final_selection_lock.json';lock=read_json(path)
            lock.update(seed_scope=[42],single_seed_exploratory=True,amendment_sha256=sha(amendment),inference_command='python -u reports/single_seed_test.py --lock reports/final_selection_lock.json; seed42 only, once')
            lock['lock_sha256']=digest({k:v for k,v in lock.items() if k!='lock_sha256'});write_json(path,lock)
            result['selection_lock_sha256']=lock['lock_sha256'];write_json(ROOT/'reports/decision.json',result)
            write_json(ROOT/'reports/campaign.json',{'stage':'WAITING_FOR_SELECTED_CHECKPOINT','queue':[[a,42] for a in decision.ARMS],'time':utc_now()})
            from storage import restore_selection
            restore_selection(lock)
            from single_seed_test import run
            test=run(path,ROOT/'configs/runtime.server.json')
        else:write_json(ROOT/'reports/decision.json',result)
        lines=['# 最后一轮单种子探索实验报告','','用户明确取消seed2024；只完成seed42的C/S/D/SD。本轮不代表原双种子协议全部完成，也不作双种子复现或均值达标声明。','',f"候选数量：{result['candidate_count']}；固定20k/32k及single/25%/50%公式。单种子晋级要求相对R42至少+0.5pp。",'']
        for label,values in result['scientific_comparisons'].items():
            lines.extend([f'## {label}',''])
            for formula,row in values.items():lines.append(f"- {formula}：{row['scores']}；同公式差值pp：{row['deltas_pp']}")
        if result['final_selection']:
            c=result['final_selection'];lines.append(f"锁定配方：{c['arm']} u{c['update']} {c['formula']}；seed42 val={c['mean_score']:.6f}，对R42增益={c['mean_gain_pp']:.6f}pp。")
        else:lines.append('没有符合单种子晋级门槛的候选，保留R，不执行新test。')
        if test:
            score=test['seeds']['42']['miou_pct'];lines.append(f'seed42锁定test={score:.6f}；是否达到47.2115目标：{score>=47.2115}。')
        lines.extend(['','历史test已被使用；本轮是单种子探索。图像bootstrap条件于已选模型，不覆盖训练种子、选型或病人相关性不确定性。','若达标，后续严谨复验等待新的用户任务范围，不自动启动seed2024。','提前科学停止的分支如实记为不完整，不补造终点。'])
        (ROOT/'reports/最终实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        report_sha=sha(ROOT/'reports/最终实验报告.md')
        waiting={'stage':'GLOBAL_FINAL_REPORT_READY_WAITING_NODE2','seed_scope':[42],'report_sha256':report_sha,'amendment_sha256':sha(amendment),'time':utc_now()}
        write_json(ROOT/'reports/campaign.json',waiting);write_json(ROOT/'reports/driver_state.json',waiting)
        ack_path=ROOT/'reports/parallel_global_final_report_mirrored.json'
        while not ack_path.exists():time.sleep(5)
        ack=read_json(ack_path)
        assert ack['stage']=='BOTH_NODES_GLOBAL_REPORT_SHA_VERIFIED' and ack['report_sha256']==report_sha and ack['amendment_sha256']==sha(amendment)
        state={'stage':'READY_FOR_LOCAL_BACKUP_AND_SHUTDOWN','queue':[[a,42] for a in decision.ARMS],'seed_scope':[42],'amendment_sha256':sha(amendment),'final_selection':result['final_selection'],'test_authorized':result['test_authorized'],'test_complete':test is not None,'time':utc_now()}
        write_json(ROOT/'reports/campaign.json',state);write_json(ROOT/'reports/driver_state.json',state)
    except BaseException as e:
        write_json(ROOT/'reports/driver_state.json',{'stage':'TECHNICAL_PAUSE','reason':str(e),'automatic_restart':False,'time':utc_now()});traceback.print_exc();raise

if __name__=='__main__':main()
