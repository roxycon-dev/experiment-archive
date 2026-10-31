"""Capacity for actual experiment output. Only the known unrelated G cold cache."""
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import utc_now,write_json


def main():
    disk=Path('/root/autodl-tmp').resolve()
    assert str(disk)=='/root/autodl-tmp'
    before=shutil.disk_usage(disk).free
    removed=[]
    candidate=disk/'HRDecoder_Next_20261003/cold_storage/G'
    if before<20*1024**3 and candidate.exists():
        resolved=candidate.resolve()
        assert resolved==candidate and resolved.is_relative_to(disk)
        assert not candidate.is_symlink() and candidate.is_dir()
        # G was a closed, failed 2026-10-03 branch. Neither current P0 nor Q16k
        # nor DDR/IDRiD/MAPLES assets refer to this separate cold_storage/G directory.
        # User explicitly requires all experiment scripts/logs to remain intact.
        # Remove only retired data/weights, never the entire directory tree.
        for path in candidate.rglob('*'):
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in ('.npz','.npy','.pth','.pt','.ckpt'):
                actual=path.resolve()
                assert actual.is_relative_to(resolved)
                path.unlink()
                removed.append(str(path))
    after=shutil.disk_usage(disk).free
    receipt={'before_free_bytes':before,'after_free_bytes':after,'removed':removed,
             'authorization':'2026-10-04 user permits deletion of experiment-unrelated server data',
             'all_scripts_logs_and_structured_evidence_preserved':True,
             'preserved':'P0/DDR/IDRiD/MAPLES/DINO/Q16k/current experiment inputs and results','time':utc_now()}
    write_json(ROOT/'reports/space_preparation.json',receipt)
    if after<8*1024**3:
        raise OSError('Less than8GiB available: review other unrelated data before launch')
    print(receipt)


if __name__=='__main__':main()
