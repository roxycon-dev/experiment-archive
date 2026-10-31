"""Copy reviewed modules and reuse the server's pure-Python MMCV dependency."""
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "local_preparation"))
from remote import connect

source = (HERE.parent / "verify_v3_modules.py").read_text(encoding="utf-8")
core = source[source.index("class ChannelLayerNorm2d"):source.index("class RandomSharedFeatureStub")]
header = '"""Private modules copied unchanged from the reviewed v3 prototype."""\nimport math\nfrom typing import Dict, Sequence, Tuple\nimport torch\nfrom torch import Tensor, nn\nimport torch.nn.functional as F\n\n'
(HERE / "private_modules.py").write_text(header+core, encoding="utf-8")
client = connect()
try:
    command = "tar -czf /root/autodl-tmp/HRDecoder_v3_20261008/reports/local_mmengine_deps.tar.gz -C /root/miniconda3/lib/python3.8/site-packages mmengine termcolor prettytable wcwidth"
    _, stdout, stderr = client.exec_command(command)
    error = stderr.read().decode(); assert stdout.channel.recv_exit_status() == 0, error
    with client.open_sftp() as ftp:
        ftp.get("/root/autodl-tmp/HRDecoder_v3_20261008/reports/local_mmengine_deps.tar.gz", str(HERE / "local_mmengine_deps.tar.gz"))
finally:
    client.close()
destination = (HERE / "python_deps").resolve()
destination.mkdir(exist_ok=True)
with tarfile.open(HERE / "local_mmengine_deps.tar.gz") as archive:
    members = []
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        assert target.is_relative_to(destination)
        assert not member.issym() and not member.islnk()
        if "__pycache__" not in Path(member.name).parts:
            members.append(member)
    archive.extractall(destination, members=members, filter="data")
print(json.dumps({"reviewed_modules_copied": True, "pure_python_deps_copied": True}))
