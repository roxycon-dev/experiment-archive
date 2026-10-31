"""Save small local terminal evidence; never controls the remote instance."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tomllib

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'observed_server_attempt03'
RUN = CACHE / 'run_seed42_launch02'
OUT = HERE / 'completion_20261017'
SOURCE = 'd3032fc8b20f30a058adbcf3e6bc5a3485b83adafe9b7cdce0682a2d55d7f627'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    progress = read(RUN / 'progress.json')
    lock = read(RUN / 'endpoint_lock.json')
    result = read(RUN / 'result.json')
    assert progress['run_identity']['source_sha256'] == lock['source_sha256'] == SOURCE
    assert progress['status'] == 'COMPLETE'
    assert progress['completed'] == progress['executed_updates'] == progress['checkpoint_persisted_updates'] == 40000
    assert progress['in_flight_update'] is None
    assert lock['completed'] == lock['executed_updates'] == lock['checkpoint_persisted_updates'] == 40000
    assert lock['checkpoint_sha256'] == progress['checkpoint_sha256']
    assert result['completed'] == 40000
    assert result['status'] == lock['status'] == 'VAL_TARGET_NOT_REACHED'
    assert lock['test_permitted'] is False and lock['test_accessed'] is False
    assert not (RUN / 'test_result.json').exists()
    OUT.mkdir(exist_ok=True)
    relative_files = [
        'run_seed42_launch02/progress.json', 'run_seed42_launch02/result.json',
        'run_seed42_launch02/endpoint_lock.json', 'preflight_attempt03/preflight.json',
    ]
    relative_files += [str(p.relative_to(CACHE)) for p in sorted(RUN.glob('val_*.json'))]
    # These controller files may be stale. Retain them as observations only;
    # never treat a previous source's result/receipt as this run's shutdown proof.
    relative_files += [p for p in ['job_status.json', 'job_result.json', 'receipts/shutdown_request.json']
                       if (CACHE / p).is_file()]
    manifest = []
    for rel in relative_files:
        src = CACHE / rel
        dst = OUT / 'observations' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest.append({'file': str(dst.relative_to(OUT)), 'bytes': dst.stat().st_size,
                         'sha256': hashlib.sha256(dst.read_bytes()).hexdigest()})
    val = lock['val']
    (OUT / 'final_validation.json').write_text(json.dumps(val, indent=2) + '\n', encoding='utf-8')
    report = {
        'recorded_at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256': SOURCE, 'training_status': 'COMPLETE',
        'executed_updates': 40000, 'checkpoint_persisted_updates': 40000,
        'in_flight_update': None, 'primary': lock['primary'],
        'checkpoint_remote_path': lock['checkpoint'],
        'checkpoint_sha256': lock['checkpoint_sha256'], 'checkpoint_downloaded_locally': False,
        'final_validation': {k: val[k] for k in ['update', 'mIoU_percent', 'IoU_percent', 'recall_percent', 'images']},
        'validation_target_percent': 45.321831227388,
        'validation_gap_percentage_points': 45.321831227388 - val['mIoU_percent'],
        'best_validation_secondary': lock['best_val_secondary'],
        'test_permitted': False, 'test_accessed': False,
        'current_run_shutdown_request_confirmed': False,
        'ssh_checks_after_completion_refused': 2,
        'ssh_observation': 'Connection to connect.nmb2.seetacloud.com:27443 refused after completion; this alone does not prove platform stop or billing stop.',
        'platform_stopped_confirmed': False, 'billing_stopped_confirmed': False,
        'shutdown_owner': 'Existing controller v3; this inspection issued no shutdown, training, resume or test command.',
        'user_decision': '继续当前轮，不重开',
        'observations_manifest': manifest,
    }
    confirmation_path = OUT / 'platform_stop_user_confirmation.json'
    if confirmation_path.is_file():
        confirmation = read(confirmation_path)
        assert confirmation['source_sha256'] == SOURCE
        assert confirmation['run'] == 'run_seed42_launch02'
        assert confirmation['host'] == 'connect.nmb2.seetacloud.com' and confirmation['port'] == 27443
        assert confirmation['platform_state'] == 'OFF'
        report['platform_stopped_confirmed'] = True
        report['platform_confirmation'] = confirmation
    platform_note = ('用户在本对话明确确认AutoDL控制台显示“已关机”。平台停机状态依据用户观察确认；未直接检查计费明细。'
                     if report['platform_stopped_confirmed'] else
                     '当前无可访问的已登录AutoDL控制台，已请用户核对实例状态；平台停止和停止计费均未确认。')
    automation_path = Path.home() / '.codex/automations/nativepyramid-sp-v3/automation.toml'
    report['heartbeat_status'] = tomllib.loads(automation_path.read_text(encoding='utf-8'))['status']
    (OUT / 'completion_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    rows = '\n'.join(f'| {c} | {iou:.2f}% | {recall:.2f}% |'
                     for c, iou, recall in zip(val['classes'], val['IoU_percent'], val['recall_percent']))
    md = f'''# NativePyramid-SP v3 本轮终态记录

正式训练已完成固定40000步，已持久化40000步，in_flight为空。源代码身份为 `{SOURCE}`。本轮继续原配方，未切换performance04或重开训练。

主结果为固定40000步原始权重，最终验证76张图，mIoU **{val['mIoU_percent']:.6f}%**。验证目标45.321831%，相差 **{report['validation_gap_percentage_points']:.6f}个百分点**，未达标。按锁定规则，test未获许可且未访问。

| 类别 | 最终IoU | 最终召回率 |
|---|---:|---:|
{rows}

历史最高验证仅作次要记录：36000步，mIoU **{lock['best_val_secondary']['mIoU_percent']:.6f}%**；不替代固定终点主结果。

最终检查点远端路径：`{lock['checkpoint']}`。

SHA256：`{lock['checkpoint_sha256']}`。本地保存了小型结果及哈希，未下载模型权重。

## 关机与计费证据的边界

训练结束后SSH连接被拒绝。本地未取得本轮控制器最终job_result或本轮关机请求回执。保留的旧job_result属于source ddb4，旧shutdown_request属于前一次预检失败，二者均不能作为本轮关机证据。现有控制器是唯一关机责任方；本次检查没有另发关机命令。

{platform_note} 原45分钟巡检状态：{report['heartbeat_status']}。详细终态锁、最终验证、历史验证与观测文件哈希见同目录JSON文件。
'''
    (OUT / '训练完成记录.md').write_text(md, encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['training_status', 'executed_updates', 'checkpoint_persisted_updates',
                                           'final_validation', 'best_validation_secondary', 'test_accessed',
                                           'current_run_shutdown_request_confirmed', 'platform_stopped_confirmed']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
