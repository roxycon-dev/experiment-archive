"""Record current model source dependencies and installed package identities."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT.parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    folders=[EXP/'HRDecoder_Ready_20261002/legacy',EXP/'HRDecoder_Ready_20261002/vendor/champion_repo',
             EXP/'HRDecoder_Ready_20261002/trainer',EXP/'HRDecoder_FinalDecision_20261005/execution',
             EXP/'HRDecoder_Q_20261003/vendor/dinov3',
             EXP/'HRDecoder_ArchitectureDesign_20261008/training_execution/python_deps']
    sources={str(p):sha(p) for folder in folders for p in sorted(folder.rglob('*.py'))}
    # Use one version lookup per package; absent optional packages are explicitly null.
    program="import importlib.metadata as m,json,platform;names=['torch','mmcv-full','mmengine','numpy','Pillow','opencv-python','scipy'];r={'python':platform.python_version(),'platform':platform.platform(),'packages':{}}\nfor n in names:\n try:r['packages'][n]=m.version(n)\n except m.PackageNotFoundError:r['packages'][n]=None\nprint(json.dumps(r))"
    envs={}
    for name,path in {'legacy':'C:/Users/GuanXingRen/Miniconda3/envs/M2MRF-Lesion-Segmentation/python.exe',
                      'modern':'C:/Users/GuanXingRen/Miniconda3/python.exe'}.items():
        r=subprocess.run([path,'-c',program],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=0x08000000,check=True)
        envs[name]=json.loads(r.stdout)
    ddr=json.loads((ROOT/'results/ddr_test_local_v1/manifest.json').read_text(encoding='utf-8'))
    for name,digest in ddr['scripts'].items():assert sha(ROOT/'scripts'/name)==digest
    proof={'status':'RECORDED','sources':sources,'environments':envs,
           'timing':'Collected while DDR evaluation was running; model/vendor source files were not modified by this supplementary work. New driver/worker hashes were recorded before inference in each run manifest.',
           'compiled_extension_note':'Reference MSDA used; legacy mmcv-full native binaries identified by installed package version, not all DLL bytes archived',
           'source_count':len(sources),'DDR_preforward_driver_hashes_still_match':True}
    (ROOT/'evidence/dependency_inventory.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print(json.dumps({'status':'RECORDED','source_files':len(sources),'environments':envs}),flush=True)


if __name__=='__main__':main()
