"""Finalize a completed validation-only queue, then request authorized AutoDL shutdown.

Runs locally without model calls. No credentials are persisted. Running or failed
training and incomplete backups never reach the shutdown command.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import time

import paramiko

LOCK_SHA = '644974f4a888835f838472199dcb623e879be2d62e73dab502394759551ceac4'
TAGS = ['U0_continuation2048', 'U0_highres']
TZ = datetime.timezone(datetime.timedelta(hours=8))


class Waiting(Exception):
    pass


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.shutdown-part')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def prepare(root):
    """Validate local evidence and backups; return an independently checkable plan."""
    root = Path(root).resolve()
    policy = read(root/'shutdown_policy.json')
    if policy.get('enabled') is not True or policy.get('job_order') != TAGS:
        raise Waiting('Shutdown authorization is disabled or scope differs')
    state = read(root/'server_results/work_dirs/campaign_state.json')
    finish = read(root/'server_results/work_dirs/finish_state.json')
    if state.get('status') != 'complete' or finish.get('status') != 'complete':
        raise Waiting('Queue is running or needs engineering attention')
    protocol = read(root/'protocol.json')
    if protocol['job_order'] != TAGS or state.get('active_job') is not None:
        raise ValueError('Unexpected queue scope or active job')
    result = read(root/'server_results/reports/round_result.json')
    if result != finish:
        raise Waiting('Final result metadata has not finished mirroring')
    journal = read(root/'reports/verified_backups.json')
    plan = {'root': str(root), 'remote_root': policy['remote_root'], 'jobs': {},
            'finish_sha256': sha(root/'server_results/work_dirs/finish_state.json'),
            'queue_wall_seconds': state['finished_at_epoch']-state['started_at_epoch'],
            'completed_at_cst': datetime.datetime.fromtimestamp(state['finished_at_epoch'], TZ).isoformat(),
            'test_policy': protocol['test_policy']}
    for tag in TAGS:
        job = state['jobs'][tag]
        summary = read(root/'server_results/work_dirs'/tag/'completion_summary.json')
        if summary != job['summary'] or summary['status'] not in ('complete', 'stopped'):
            raise Waiting('Job summary is missing or not final: '+tag)
        if summary['status'] == 'stopped' and summary['stop_reason'] != 'paired_no_gain_or_learning':
            raise ValueError('Unrecognized stop reason: '+tag)
        if not 2000 <= summary['updates_done'] <= protocol['max_updates_per_arm']:
            raise ValueError('Unexpected update count: '+tag)
        expected = protocol['config_sha256'][tag]
        for path in [root/'configs'/f'{tag}.yaml', root/'server_results/configs'/f'{tag}.yaml']:
            if sha(path) != expected:
                raise ValueError('Configuration changed: '+tag)
        if summary['config_file_sha256'] != expected:
            raise ValueError('Summary configuration mismatch: '+tag)
        rows = [read(p) for p in (root/'server_results/work_dirs'/tag).glob('eval_u*.json')]
        by_update = {r['update']: r for r in rows}
        required = [u for u in (1000, 2000, 4000) if u <= summary['updates_done']]
        if not all(u in by_update for u in required):
            raise Waiting('Full validation evidence has not finished mirroring: '+tag)
        for row in rows:
            if (row['n_images'] != 76 or row['fixed_fusion_images'] != 76 or
                    row['eval_inference'] != 'whole_image' or row['split'] != 'val' or
                    abs(row['fixed_fusion_baseline_miou']-protocol['baseline_val_miou']) > 1e-8):
                raise ValueError('Validation protocol mismatch: '+tag)
            counts = row['fixed_fusion_counts']
            pooled = sum(tp/(tp+fp+fn) for tp, fp, fn in counts)/4
            if len(counts) != 4 or abs(pooled-row['fixed_fusion_miou']) > 1e-10:
                raise ValueError('Validation counts mismatch: '+tag)
        best = by_update[summary['best']['update']]
        if (summary['best']['selection_metric'] != 'fixed_fusion_miou' or
                abs(best['fixed_fusion_miou']-max(r['fixed_fusion_miou'] for r in rows)) > 1e-10 or
                abs(best['fixed_fusion_miou']-summary['best']['selection_score']) > 1e-10 or
                result['results'][tag]['best_update'] != best['update']):
            raise ValueError('Best checkpoint selection mismatch: '+tag)
        key = tag+'/ckpt_best.pth'
        backup = journal.get(key)
        path = root/'server_results/work_dirs'/key
        if backup is None or not path.exists():
            raise Waiting('Verified best backup is not complete: '+tag)
        if (Path(backup['local']).resolve() != path.resolve() or
                path.stat().st_size != backup['bytes'] or sha(path) != backup['sha256']):
            raise ValueError('Local best backup hash/path mismatch: '+tag)
        plan['jobs'][tag] = {'best': best, 'updates_done': summary['updates_done'],
                             'stop_reason': summary.get('stop_reason'), 'backup': backup,
                             'config_sha256': expected,
                             'wall_seconds': job['finished_at_epoch']-job['started_at_epoch']}
    main = root.parent
    manifest = read(main/'best_pipeline/local_weight_manifest.json')
    if (manifest['original_lock_sha256'] != LOCK_SHA or
            sha(main/'best_pipeline/original_deployment_lock.json') != LOCK_SHA):
        raise ValueError('Original evaluated deployment lock changed')
    for item in manifest['weights'].values():
        path = Path(item['path'])
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Previous best pipeline backup is invalid')
    plan['historical_pipeline_verified'] = True
    return plan


def finalize(root, plan):
    root = Path(root)
    body = ['# 高分辨率配对续训结果', '',
            '两臂已完成或按预设验证规则早停。本轮仅探索 train/val；上一轮 test 44.7809% 保持原记录。', '',
            '|候选|实际更新|最佳节点|单模型 val mIoU|固定融合 val mIoU|较既有融合增益（百分点）|本机 best|',
            '|---|---:|---:|---:|---:|---:|---|']
    for tag, item in plan['jobs'].items():
        b = item['best']
        body.append('|%s|%d|%d|%.4f%%|%.4f%%|%+.4f|%s|' % (
            tag, item['updates_done'], b['update'], b['mIoU_lesion4']*100,
            b['fixed_fusion_miou']*100, b['fixed_fusion_gain']*100, item['backup']['local']))
    gain = max(item['best']['fixed_fusion_gain'] for item in plan['jobs'].values())
    body += ['', '发现：'+('新候选在验证集出现收益，保留已校验 best 供后续研究；这不是新的 test 结论。' if gain > 0 else
                              '本轮未找到超过既有融合的候选，继续保留原最佳管线。'),
             '固定融合为旧冠军 75%＋候选 25%，没有再搜索融合权重。控制臂与高分辨率臂只在共有完整验证节点作配对比较；不同早停预算分别披露。',
             '每臂均最多 4000 步。基准长边分别为 2048 与随机 2048/2560，均保留 0.5–2.0 缩放扰动。',
             '队列墙钟耗时：%.1f 分钟（包含验证和控制臂接口修复/恢复，不含等待本机备份）；实际训练更新合计 %d，较上限节省 %d。' % (
                 plan['queue_wall_seconds']/60, sum(x['updates_done'] for x in plan['jobs'].values()),
                 8000-sum(x['updates_done'] for x in plan['jobs'].values())),
             '控制臂从 1000 步严格兼容 checkpoint 恢复，没有重训已完成更新；DataLoader 随机流恢复不是逐位相同轨迹。纯 GPU 训练时长未单独记录，不能把恢复后的 elapsed_seconds 当作整臂耗时。',
             '完成时间（北京时间）：'+plan['completed_at_cst'],
             '两臂 best 和历史部署所需 6 个权重均已在本机重新核对 SHA-256。服务器关机是独立的后续动作，实际状态见 reports/shutdown_state.json。', '']
    text = '\n'.join(body)
    target = root/'reports/最终结果.md'
    target.write_text(text, encoding='utf-8')
    archive = root.parent.parent/'_全实验档案_20260920'
    archive.mkdir(exist_ok=True)
    record = archive/'05_高分辨率配对续训_20261003.md'
    record.write_text(text, encoding='utf-8')
    marker = '<!-- HRD_ROUND2_COMPLETE_20261003 -->'
    notice = ('\n\n'+marker+'\n\n高分辨率配对续训已完成，原 test 保持锁定。实际更新：'+
              '；'.join('%s %d 步'%(tag, item['updates_done']) for tag,item in plan['jobs'].items())+
              '。结果与权重备份校验见 ['+record.name+']('+str(record).replace('\\','/')+')。\n')
    for path in [archive/'01_全目录实验清单.md', archive/'02_HRDecoder逐实验事实档案.md',
                 archive/'04_本轮实验补充_20261002.md', root.parent.parent/'实验全景解读报告.md',
                 root.parent/'reports/最终结果.md', root.parent/'reports/运行结果.md']:
        original = path.read_text(encoding='utf-8-sig') if path.exists() else ''
        if marker not in original:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(original+notice, encoding='utf-8')
    finalized = dict(plan, final_report_sha256=sha(target), workspace_archive=str(record))
    write(root/'reports/completion_ready.json', finalized)
    write(record.with_suffix('.json'), finalized)
    return {'final_report_sha256': finalized['final_report_sha256'], 'workspace_archive': str(record)}


REMOTE_GUARD = r'''
import hashlib,json,os,pathlib,subprocess,sys,time
request=json.loads(sys.argv[1]);root=pathlib.Path(request['remote_root'])
finish=root/'work_dirs/finish_state.json';state=json.loads((root/'work_dirs/campaign_state.json').read_text())
assert state['status']=='complete' and state['active_job'] is None
assert json.loads(finish.read_text())['status']=='complete'
assert hashlib.sha256(finish.read_bytes()).hexdigest()==request['finish_sha256']
marker=root/'reports/shutdown_request.json'
assert not marker.exists(), 'A shutdown request already exists; inspect it instead of repeating'
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid(): continue
 try: args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
 except (OSError,ProcessLookupError): continue
 blocked=['train.py','run_round2.py','run_campaign.py','finish_campaign.py','export_predictions.py','export_champion.py','final_eval_locked.py','crossfit_fusion.py']
 assert not any(pathlib.PurePosixPath(a).name in blocked for a in args), 'Experiment process still running'
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
assert gpu.returncode==0 and not gpu.stdout.strip(), 'GPU compute process still running'
import torch,yaml
for tag,item in request['jobs'].items():
 config=root/'configs'/(tag+'.yaml')
 assert hashlib.sha256(config.read_bytes()).hexdigest()==item['config_sha256']
 effective_sha=hashlib.sha256(json.dumps(yaml.safe_load(config.read_text()),sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
 path=root/'work_dirs'/tag/'ckpt_best.pth';h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4194304),b''):h.update(block)
 assert h.hexdigest()==item['backup']['sha256'] and path.stat().st_size==item['backup']['bytes']
 checkpoint=torch.load(path,map_location='cpu')
 assert checkpoint['update']==item['best']['update'] and checkpoint['config_sha256']==effective_sha
 assert abs(checkpoint['metrics_at_save']['fixed_fusion_miou']-item['best']['fixed_fusion_miou'])<1e-10
 del checkpoint
assert os.access('/usr/bin/shutdown',os.X_OK), 'AutoDL shutdown utility missing'
if request.get('dry_run'):
 print(json.dumps({'status':'dry_run_ready','shutdown_executed':False}));sys.exit(0)
request['requested_at_epoch']=time.time();marker.write_text(json.dumps(request,indent=2))
os.sync()
result=subprocess.run(['/bin/bash','/usr/bin/shutdown'],capture_output=True,text=True,timeout=25)
record={'status':'shutdown_request_accepted' if result.returncode==0 else 'shutdown_command_failed',
        'returncode':result.returncode,'requested_at_epoch':request['requested_at_epoch']}
try:
 response=json.loads(result.stdout)
 if isinstance(response,dict):
  record['utility_response_keys']=list(response)
  record['utility_success']=response.get('success') if isinstance(response.get('success'),bool) else None
  code=response.get('code')
  record['utility_code']=code if isinstance(code,(int,float)) or isinstance(code,str) and len(code)<32 else None
  if record['utility_success'] is False:
   record['status']='shutdown_command_failed'
except (ValueError,TypeError): pass
(root/'reports/shutdown_command_result.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-root', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(args.local_root).resolve()
    password = os.environ.pop('HRD_SHUTDOWN_PASSWORD', None)
    if password is None:
        raise ValueError('Missing in-memory SSH credential')
    status_path = root/'reports/shutdown_state.json'
    if status_path.exists() and read(status_path).get('status') in ('dispatching', 'shutdown_request_accepted', 'shutdown_dispatched_unconfirmed'):
        raise ValueError('A request already exists; inspect before any retry')
    write(root/'reports/shutdown_controller_pid.json', {'pid': os.getpid(), 'started_at_epoch': time.time()})
    while True:
        try:
            plan = prepare(root)
        except (Waiting, FileNotFoundError) as error:
            write(status_path, {'status': 'waiting', 'reason': str(error), 'checked_at_epoch': time.time()})
            time.sleep(30)
            continue
        except Exception as error:
            write(status_path, {'status': 'needs_attention', 'error': repr(error), 'checked_at_epoch': time.time()})
            raise
        if not args.dry_run:
            plan.update(finalize(root, plan))
        plan['dry_run'] = args.dry_run
        policy = read(root/'shutdown_policy.json')
        if policy.get('enabled') is not True:
            raise Waiting('Authorization disabled before dispatch')
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        dispatched = False
        try:
            client.connect(policy['host'], port=policy['port'], username='root', password=password,
                           allow_agent=False, look_for_keys=False, timeout=20, auth_timeout=20)
            if not args.dry_run:
                sftp = client.open_sftp()
                try:
                    sftp.put(str(root/'reports/最终结果.md'), policy['remote_root']+'/reports/最终结果.md')
                    sftp.put(str(root/'reports/completion_ready.json'), policy['remote_root']+'/reports/local_completion_ready.json')
                finally:
                    sftp.close()
                write(status_path, {'status': 'dispatching', 'at_epoch': time.time(), 'final_report': str(root/'reports/最终结果.md')})
            _, out, err = client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(REMOTE_GUARD)+' '+shlex.quote(json.dumps(plan)), timeout=60)
            dispatched = True
            stdout = out.read().decode('utf-8', 'replace')
            stderr = err.read().decode('utf-8', 'replace')
            rc = out.channel.recv_exit_status()
            if rc != 0:
                if 'AssertionError' in stderr or rc > 0:
                    raise ValueError('Remote guard rejected shutdown: '+stderr[-1800:])
                response = {'status': 'shutdown_dispatched_unconfirmed', 'reason': 'SSH closed before acknowledgement'}
            else:
                response = json.loads(stdout.strip().splitlines()[-1])
            response.update(local_final_report=str(root/'reports/最终结果.md'), recorded_at_epoch=time.time(), platform_state_independently_confirmed=False)
            write(status_path, response)
            return
        except Exception as error:
            write(status_path, {'status': 'shutdown_dispatched_unconfirmed' if dispatched and not isinstance(error, ValueError) else 'needs_attention',
                                'error': repr(error), 'checked_at_epoch': time.time(), 'automatic_retry': False})
            raise
        finally:
            client.close()


if __name__ == '__main__':
    main()
