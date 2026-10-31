set -u
R=/root/autodl-tmp/HRDecoder_FullCompetition_20260920
echo "=== HOSTNAME/DATE ==="; hostname; date
echo "=== ROOT ==="; ls -la $R
echo "=== expert_outputs ==="; ls -la $R/expert_outputs
echo "=== hrplus ==="; ls -la $R/expert_outputs/hrplus
echo "=== fullmodel ==="; ls -la $R/expert_outputs/fullmodel
echo "=== analysis ==="; ls -la $R/expert_outputs/analysis 2>/dev/null
echo "=== npz counts ==="
for d in $R/expert_outputs/hrplus/*/; do echo -n "$d "; ls $d*.npz 2>/dev/null | wc -l; done
for d in $R/expert_outputs/fullmodel/*/; do echo -n "$d "; ls $d*.npz 2>/dev/null | wc -l; done
echo -n "_shared/gt "; ls $R/expert_outputs/_shared/gt/*.npz 2>/dev/null | wc -l
echo "=== exit code files ==="; find $R -maxdepth 3 -name "*exit*" -exec ls -la {} \;
echo "=== logs dir ==="; find $R -maxdepth 2 -name "logs" -exec ls -la {} \;
echo "=== GPU ==="; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo "=== running python ==="; ps -eo pid,etime,cmd | grep -E "python|phase_cs" | grep -v grep
