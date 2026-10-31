set -eu
cd /root/autodl-tmp/HRDecoder_Q_20261003
/root/miniconda3/envs/hrdecoder_q_20261003/bin/python advance_campaign.py
/root/miniconda3/envs/hrdecoder_q_20261003/bin/python reports/final_evaluation.py --phase freeze
