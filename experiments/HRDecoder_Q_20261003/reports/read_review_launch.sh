set -eu
cd /root/autodl-tmp/HRDecoder_Q_20261003
tail -n 8 reports/review_launcher.log
tail -n 4 reports/review_stream_seed2024.log
if [ -f reports/formal_Q_DDR_seed2024.log ]; then tail -n 3 reports/formal_Q_DDR_seed2024.log; fi
if [ -f runs/Q_DDR_seed2024/status.json ]; then cat runs/Q_DDR_seed2024/status.json; fi
