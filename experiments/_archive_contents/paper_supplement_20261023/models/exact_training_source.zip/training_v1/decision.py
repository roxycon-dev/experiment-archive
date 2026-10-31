"""Locked24 two-seed recipes, eligibility first, exact unrounded ties."""
from pathlib import Path
import numpy as np
from qlib.common import read_json,write_json,digest,sha,utc_now
from qlib.metrics import paired_bootstrap
from storage import checkpoint_sha
ROOT=Path(__file__).resolve().parent
ARMS=('C','S','D','SD');FORMULAS=('single','fixed_fusion','half_fusion');SEEDS=(42,2024)

def choose(candidates,references):
    eligible=[]
    for candidate in candidates:
        scores=candidate['scores'];gains={str(s):scores[str(s)]-references[str(s)] for s in SEEDS}
        mean_gain=sum(gains.values())/2;mean_score=sum(scores.values())/2
        candidate.update(gains_pp=gains,mean_gain_pp=mean_gain,mean_score=mean_score,
            eligible=all(g>0 for g in gains.values()) and mean_gain>=.5)
        if candidate['eligible']:eligible.append(candidate)
    return sorted(eligible,key=lambda r:(-r['mean_score'],0 if r['formula']=='single' else 1,r['update'],ARMS.index(r['arm']),FORMULAS.index(r['formula'])))[0] if eligible else None

def finalize():
    protocol=read_json(ROOT/'configs/protocol.json');references=protocol['evaluation']['R_val_by_seed']
    outcomes={};candidates=[]
    for arm in ARMS:
        outcomes[arm]={}
        for seed in SEEDS:
            run=ROOT/f'runs/{arm}_seed{seed}';status=read_json(run/'status.json')
            assert status['stage'] in ('COMPLETE','SCIENTIFIC_STOP_INCOMPLETE')
            outcomes[arm][str(seed)]={'status':status,'nodes':{str(u):read_json(run/f'val_u{u}/metrics.json') for u in (20000,32000) if (run/f'val_u{u}/metrics.json').exists()}}
        for u in (20000,32000):
            # Both registered endpoints are independent product candidates.
            # A later collapse leaves a completed paired20k endpoint eligible;
            # it never creates a missing32k result in the scientific table.
            if all(str(u) in outcomes[arm][str(s)]['nodes'] and
                   outcomes[arm][str(s)]['status']['completed']>=u for s in SEEDS):
                for formula in FORMULAS:
                    paths={str(s):ROOT/f'runs/{arm}_seed{s}/checkpoint_u{u}.pth' for s in SEEDS}
                    candidates.append({'arm':arm,'update':u,'formula':formula,
                        'scores':{str(s):outcomes[arm][str(s)]['nodes'][str(u)][formula]['miou_pct'] for s in SEEDS},
                        'checkpoints':{s:{'path':p.relative_to(ROOT).as_posix(),'sha256':checkpoint_sha(p.relative_to(ROOT).as_posix())} for s,p in paths.items()}})
    selected=choose(candidates,references)
    science={}
    for seed in SEEDS:
        for u in (20000,32000):
            values={arm:outcomes[arm][str(seed)]['nodes'].get(str(u)) for arm in ARMS}
            comparisons={}
            for formula in ('single','fixed_fusion'):
                scores={arm:node[formula]['miou_pct'] for arm,node in values.items() if node is not None}
                deltas={}
                for name,a,b in (('S-C','S','C'),('D-C','D','C'),('SD-D','SD','D'),('SD-S','SD','S')):
                    if a in scores and b in scores:deltas[name]=scores[a]-scores[b]
                if len(scores)==4:deltas['SD-S-D+C']=scores['SD']-scores['S']-scores['D']+scores['C']
                comparisons[formula]={'scores':scores,'deltas_pp':deltas}
            science[f'seed{seed}_u{u}']=comparisons
    result={'stage':'FINAL_SELECTION_FROZEN','time':utc_now(),'candidate_count':len(candidates),'candidates':candidates,
        'scientific_comparisons':science,'outcomes':outcomes,'final_selection':selected,'test_authorized':selected is not None,
        'deployment_seed':42,'historical_test_reuse_exploratory':True}
    if selected:
        lock={'campaign':'20261005','arm':selected['arm'],'update':selected['update'],'formula':selected['formula'],
            'checkpoints':selected['checkpoints'],'p0_identity_sha256':sha(ROOT/'configs/p0_identity.json'),
            'data_identity_sha256':sha(ROOT/'reports/data_identity.json'),'source_sha256':read_json(ROOT/'reports/source_identity.json')['sha256'],
            'runtime_sha256':sha(ROOT/'configs/runtime.server.json'),
            'validation_files_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'runs').glob('*/val_u*/metrics.json')},
            'inference':'native1024/center768/halo128; RGB ImageNet normalization; FP32 sigmoid; strict>.5; original-pixel counts',
            'inference_command':'python -u final_test.py --lock reports/final_selection_lock.json; both seeds once; deployment seed42'}
        lock['lock_sha256']=digest(lock);write_json(ROOT/'reports/final_selection_lock.json',lock)
        result['selection_lock_sha256']=lock['lock_sha256']
    write_json(ROOT/'reports/decision.json',result)
    return result

def write_report(result,test=None):
    lines=['# 最后一轮决定性实验报告','',f"候选数量：{result['candidate_count']}；所有指标按原图valid像素汇总四类IoU。",'']
    for label,values in result['scientific_comparisons'].items():
        lines.append(f"## {label}");lines.append('')
        for formula,row in values.items():lines.append(f"- {formula}：{row['scores']}；同公式差值（pp）：{row['deltas_pp']}")
        lines.append('')
    selected=result['final_selection']
    lines.append('最终部署：继续使用R；没有符合双种子验证门槛的候选，不执行新test。' if selected is None else f"最终锁定：{selected['arm']} u{selected['update']} {selected['formula']}；val均分{selected['mean_score']:.6f}，对R平均增益{selected['mean_gain_pp']:.6f}pp。部署种子固定42。")
    if test:
        scores=[test['seeds'][str(s)]['miou_pct'] for s in SEEDS];mean=sum(scores)/2
        lines.extend(['',f'锁定test：seed42={scores[0]:.6f}，seed2024={scores[1]:.6f}，均值={mean:.6f}；均值目标47.2115是否达到：{mean>=47.2115}。',
            '历史test已经被使用，本轮结论具有探索性；配对图像bootstrap仅描述锁定模型，不能覆盖训练种子、选型和病人相关性的不确定性。'])
    lines.extend(['','D/SD与C/S的计算量不同。本轮比较是在相同DDR暴露量下比较工程方案，不是纯新增数据的等计算因果实验。',
        '图像去重不能保证未知病人独立性。若分支停止，记为不完整，不补造32k结果。'])
    (ROOT/'reports/最终实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
