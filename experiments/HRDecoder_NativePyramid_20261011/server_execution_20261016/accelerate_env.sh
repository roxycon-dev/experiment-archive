set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
PY="$BASE/venv/bin/python"
export TMPDIR="$BASE/tmp" PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# Only the owned, still-waiting bootstrap is paused; no training has started.
/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'
from pathlib import Path
import os,signal,zipfile,json
base=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
assert not (base/'controller_started.json').exists(),'Training controller already started'
pid=1631
assert Path(f'/proc/{pid}/cwd').resolve()==base
assert b'bootstrap.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
os.kill(pid,signal.SIGSTOP)
dest=base/'packages/wheelhouse';dest.mkdir(exist_ok=True)
saved=[]
for p in (base/'tmp').rglob('*.whl'):
    try:
        with zipfile.ZipFile(p) as z:z.infolist()
    except (zipfile.BadZipFile,OSError):continue
    target=dest/p.name
    if not target.exists():os.link(p,target)
    saved.append({'name':p.name,'bytes':p.stat().st_size})
pip=1175
command=Path(f'/proc/{pip}/cmdline').read_bytes()
assert str(base/'venv/bin/python').encode() in command and b'pip' in command
os.kill(pip,signal.SIGINT)
(base/'receipts/environment_download_salvage.json').write_text(json.dumps(saved,indent=2)+'\n')
print(json.dumps({'preserved_complete_wheels':len(saved),'preserved_bytes':sum(x['bytes'] for x in saved)}))
PY
trap 'kill -CONT 1631 2>/dev/null || true' EXIT
sleep 2
"$PY" -m pip install --no-index --no-deps "$BASE"/packages/wheelhouse/*.whl
"$PY" -m pip install numpy==2.4.1 Pillow==12.1.0 timm==1.0.27 opencv-python==4.13.0.90 --index-url https://pypi.tuna.tsinghua.edu.cn/simple
"$PY" -m pip check
"$PY" -m pip freeze > "$BASE/receipts/pip_freeze.txt"
"$PY" - <<'PY'
import torch,timm,cv2,numpy,PIL,sys,json
print(json.dumps({'python':sys.version,'torch':torch.__version__,'timm':timm.__version__,
                  'cv2':cv2.__version__,'numpy':numpy.__version__,'Pillow':PIL.__version__,
                  'bf16':torch.cuda.is_bf16_supported(),'gpu':torch.cuda.get_device_name()},indent=2))
PY
touch "$BASE/receipts/environment_ready"
