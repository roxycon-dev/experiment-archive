"""Import only author TRAIN archive, validating the official 896-file inventory."""
import sys,time,zipfile,shutil,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qlib.common import sha,write_json,read_json,utc_now
ROOT=Path(__file__).resolve().parents[1]

def main():
    downloads=Path.home()/'Downloads';destination=ROOT/'inputs/TJDR'
    expected={f'train/{folder}/TJDR_train_{i:03d}.png' for folder in ('annotation','image') for i in range(1,449)}
    deadline=time.monotonic()+3600;archive=None
    while time.monotonic()<deadline:
        candidates=[p for p in downloads.iterdir() if p.is_file() and p.suffix.lower() in ('.zip','.crdownload') and p.stat().st_mtime>time.time()-7200]
        for path in candidates:
            if not zipfile.is_zipfile(path):continue
            with zipfile.ZipFile(path) as z:
                names={r.filename for r in z.infolist() if not r.is_dir()}
                if names==expected:archive=path;break
        if archive:break
        write_json(ROOT/'reports/TJDR_zip_import.json',{'stage':'WAITING_AUTHOR_TRAIN_ARCHIVE','time':utc_now(),'eligible_downloads':[{'name':p.name,'bytes':p.stat().st_size} for p in candidates]})
        time.sleep(15)
    assert archive,'Official TRAIN-only ZIP not complete within bounded wait'
    assert shutil.disk_usage(ROOT).free>10*1024**3,'Insufficient local extraction reserve'
    recovered=[]
    with zipfile.ZipFile(archive) as z:
        for name in sorted(expected):
            path=(destination/name).resolve();assert path.is_relative_to(destination.resolve())
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix('.backup_import.tmp')
            with z.open(name) as inp,temporary.open('wb') as out:shutil.copyfileobj(inp,out,8*1024**2)
            from PIL import Image
            with Image.open(temporary) as im:im.verify()
            if path.exists():
                assert sha(path)==sha(temporary),'Google/OneDrive official mirror disagreement: '+name
                temporary.unlink()
            else:temporary.replace(path)
            recovered.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size})
    old=ROOT/'reports/TJDR_download_receipt.json'
    if old.exists():shutil.copy2(old,ROOT/'reports/TJDR_Google_interrupted_receipt.json')
    receipt={'stage':'DOWNLOAD_COMPLETE','source':'Author public OneDrive TRAIN backup','archive_sha256':sha(archive),
        'expected_files':896,'completed_files':recovered,'failures':[],'official_test_downloaded':False,'time':utc_now(),
        'overlapping_Google_files_exact':True}
    write_json(old,receipt);write_json(ROOT/'reports/TJDR_zip_import.json',receipt)
    print('OFFICIAL_TRAIN_IMPORTED',flush=True)
    subprocess.run([sys.executable,'-u','ops/accept_tjdr.py'],cwd=ROOT,check=True)

if __name__=='__main__':main()
