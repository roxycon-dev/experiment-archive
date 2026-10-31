set -u
R=/root/autodl-tmp/HRDecoder_FullCompetition_20260920
echo "=== per-dir byte totals (int8packed npz) ==="
for d in $R/expert_outputs/hrplus/*/ $R/expert_outputs/fullmodel/*/ $R/expert_outputs/_shared/gt/; do
  echo -n "$(du -sb $d | cut -f1) bytes  $(ls $d*.npz | wc -l) files  $d"
  echo
done
echo "=== sample ls -l ==="
ls -l $R/expert_outputs/hrplus/champion/ | head -4
ls -l $R/expert_outputs/hrplus/C2000/ | head -4
echo "=== total expert_outputs ==="; du -sh $R/expert_outputs
echo "=== md5 of champion vs BASE_0 for one stem ==="
md5sum $R/expert_outputs/hrplus/champion/007-1774-100.npz $R/expert_outputs/hrplus/BASE_0/007-1774-100.npz $R/expert_outputs/hrplus/C2000/007-1774-100.npz
echo "=== server logs for the 3 runs ==="
find /root -maxdepth 3 -name "*.log" -newermt "2026-09-20 21:00" 2>/dev/null | head -30
echo "=== logs_expert_comp ==="; find $R/logs_expert_comp -type f | head -40
