set -eu
ROOT=/root/autodl-tmp/HRDecoder_Q_20261003
ENV=/root/miniconda3/envs/hrdecoder_q_20261003
mkdir -p "$ROOT/reports" "$ROOT/assets"
if [ ! -x "$ENV/bin/python" ]; then
  /root/miniconda3/bin/conda create --yes --override-channels -c conda-forge --prefix "$ENV" python=3.11 pip=25.1.1
fi
"$ENV/bin/python" -m pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
"$ENV/bin/python" -m pip install --no-cache-dir numpy==2.2.6 pillow==11.2.1 opencv-python-headless==4.11.0.86 termcolor==3.1.0
"$ENV/bin/python" -m pip freeze > "$ROOT/reports/environment_freeze.txt"
printf 'ISOLATED_ENVIRONMENT_READY\n'
