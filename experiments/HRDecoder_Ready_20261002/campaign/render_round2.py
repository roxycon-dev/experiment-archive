"""Small local report refreshed by the metadata mirror, without model calls."""
import datetime
import json
from pathlib import Path


def render(root):
    root=Path(root)
    state=json.loads((root/'server_results/work_dirs/campaign_state.json').read_text(encoding='utf-8'))
    health=json.loads((root/'reports/monitor_health.json').read_text(encoding='utf-8'))
    tz=datetime.timezone(datetime.timedelta(hours=8))
    at=datetime.datetime.fromtimestamp(health['checked_at_epoch'],tz)
    text=['# 第二轮：U0高分辨率配对续训','',
          '状态：`'+state['status']+'`；当前阶段：`'+str(state.get('active_job'))+'`。',
          '', '上一轮已完成test44.7809%，本轮只使用train/val，不再重选或重新评价该test。',
          '', '两臂均从历史U0 best开始，新优化器，LR乘0.2，crop1024、有效batch8。',
          '控制臂基准长边2048；高分辨率臂随机基准长边2048/2560。均保留0.5–2.0缩放扰动。各最多4000更新，1000/2000/4000完整val。',
          '固定融合为旧冠军75%＋候选25%；以该融合val保存best，兼顾学习曲线与互补性决定2k是否停止。',
          '', '|阶段|更新|单模型val|固定融合val|相对既有43.4877%的增益（百分点）|',
          '|---|---:|---:|---:|---:|']
    for tag in ('U0_continuation2048','U0_highres'):
        for p in sorted((root/'server_results/work_dirs'/tag).glob('eval_u*.json'),key=lambda p:int(p.stem[6:])):
            row=json.loads(p.read_text(encoding='utf-8'))
            text.append('|%s|%d|%.4f%%|%.4f%%|%+.4f|'%(tag,row['update'],row['mIoU_lesion4']*100,row['fixed_fusion_miou']*100,row['fixed_fusion_gain']*100))
    text+=['',f'服务器证据时间：{at:%Y-%m-%d %H:%M:%S}（北京时间）；GPU：{health["gpu"]}；磁盘剩余：{health["disk_free_bytes"]/1024**3:.2f}GiB。',
           '', 'checkpoint已使用val选点，以上是探索结果；下一轮独立测试需要另行定义评估数据。',
           '', '完整配置及协议在configs/和protocol.json；服务器镜像及best备份在server_results/。']
    p=root/'reports/进度.md'
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text('\n'.join(text)+'\n',encoding='utf-8');tmp.replace(p)
