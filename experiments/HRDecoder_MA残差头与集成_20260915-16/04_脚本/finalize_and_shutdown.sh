#!/bin/bash
# 收尾：汇总结果 -> 确认无残留进程 -> 关机
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
OUT=/root/autodl-tmp/hrplus/results/L_0915

echo "======================= 最终结果汇总 $(date '+%F %T') ======================="
for f in $OUT/eval_*.json; do
  [ -f "$f" ] || continue
  $PY -c "
import json,sys
d=json.load(open(sys.argv[1]))
print('%-24s mIoU=%7.3f%%  flip=%s  n=%d  mAUPR=%.3f' % (
  sys.argv[1].split('eval_')[1].replace('.json',''),
  d['mIoU']*100, d['flip_tta'], len(d['models']), d.get('mAUPR',0)*100))
" "$f"
done

echo
echo "======================= 权重加载报告 ======================="
for f in $OUT/eval_*.json; do
  [ -f "$f" ] || continue
  $PY -c "
import json,sys
d=json.load(open(sys.argv[1]))
for r in d.get('load_report',[]):
    print('%-24s %-28s missing=%d unexpected=%d' % (sys.argv[1].split('eval_')[1][:22], r['ckpt'][:28], len(r['missing']), len(r['unexpected'])))
" "$f"
done

echo
echo "======================= 检查点清单 ======================="
ls -la $OUT/*.pth
echo
echo "sha256 最佳学生权重:"
sha256sum $OUT/best_mIoU_iter_6000.pth 2>/dev/null

echo
echo "======================= L 训练最终 val 曲线 ======================="
grep -aE 'Best mIoU|Iter\(val\) \[76\]' $OUT/l_train.log | tail -10

echo
echo "======================= 残留进程检查 ======================="
LEFT=$(ps -ef | grep -E 'hrplus/ops/eval|tools/train.py|run_evals' | grep -v grep)
if [ -n "$LEFT" ]; then
  echo "!! 仍有进程，先清理:"
  echo "$LEFT"
  for p in $(echo "$LEFT" | awk '{print $2}'); do kill -TERM "$p" 2>/dev/null; done
  sleep 10
  for p in $(echo "$LEFT" | awk '{print $2}'); do kill -9 "$p" 2>/dev/null; done
  sleep 5
  ps -ef | grep -E 'hrplus/ops/eval|tools/train.py|run_evals' | grep -v grep || echo "已清理"
else
  echo "无残留进程"
fi
echo "GPU:"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

echo
echo "======================= 停机保护状态 ======================="
ps -ef | grep ops_deadline_guard | grep -v grep || echo "guard 已不在"

echo
echo "======================= 执行关机 ======================="
sync
sleep 3
echo "调用 bash /usr/bin/shutdown ..."
/bin/bash /usr/bin/shutdown
