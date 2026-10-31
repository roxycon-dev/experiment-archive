#!/bin/bash
# 收尾：汇总 -> 确认无残留 -> 关机
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python

echo "======================= 最终 test 结果汇总 $(date '+%F %T') ======================="
$PY - <<'PYEOF'
import json, glob, os
rows = []
for f in glob.glob('/root/autodl-tmp/hrplus/results/*/eval_*.json'):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if d.get('split') != 'test':
        continue
    rows.append((os.path.basename(f).replace('eval_', '').replace('.json', ''),
                 d['mIoU'] * 100, d.get('scales'), d.get('flip_tta'),
                 len(d['models']), d.get('per_class_iou') or {}))
rows.sort(key=lambda r: -r[1])
print('%-30s %8s %13s %5s %2s' % ('name', 'mIoU%', 'scales', 'flip', 'n'))
for n, m, sc, fl, k, pc in rows:
    print('%-30s %8.3f %13s %5s %2d  EX=%s HE=%s SE=%s MA=%s'
          % (n, m, sc, fl, k, pc.get('EX'), pc.get('HE'),
             pc.get('SE'), pc.get('MA')))
print()
print('历史参照：baseline 42.20 / E4 41.63 / B 41.92 / D@2000 42.575 / 目标 45.35')
PYEOF

echo
echo "======================= 检查点清单 ======================="
ls -la /root/autodl-tmp/hrplus/results/E_0916/*.pth /root/autodl-tmp/hrplus/results/E2_0916/*.pth /root/autodl-tmp/hrplus/results/E3_0916/*.pth 2>/dev/null
echo "E 的 sha256:"; sha256sum /root/autodl-tmp/hrplus/results/E_0916/best_mIoU_iter_1500.pth 2>/dev/null

echo
echo "======================= 残留进程清理 ======================="
LEFT=$(ps -ef | grep -E '[h]rplus/ops/eval|[t]ools/train.py|[r]un_ens_de|[r]un_scale_tta|[l]aunch_E' | awk '{print $2}')
if [ -n "$LEFT" ]; then
  echo "清理: $LEFT"
  for p in $LEFT; do kill -TERM $p 2>/dev/null; done
  sleep 10
  for p in $LEFT; do kill -9 $p 2>/dev/null; done
  sleep 5
fi
ps -ef | grep -E '[h]rplus/ops/eval|[t]ools/train.py' | awk '{print $2}' || echo "无残留"
echo "GPU:"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

echo
echo "======================= 停机保护 ======================="
ps -ef | grep '[d]eadline_guard' | awk '{print $2, $9, $10, $11}' || echo "guard 不在"

echo
echo "======================= 执行关机 ======================="
sync
sleep 3
echo "调用 bash /usr/bin/shutdown ..."
/bin/bash /usr/bin/shutdown
