"""Create an audited ATen API compatibility copy; retain author source intact."""
import sys,shutil,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qlib.common import sha,write_json,utc_now
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'vendor/vit_adapter_reference/detection/ops/src';target=ROOT/'vendor/msda_compat/src'
    target.mkdir(parents=True,exist_ok=True);receipt=[]
    for path in sorted(source.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(source);out=target/rel;out.parent.mkdir(parents=True,exist_ok=True)
        if path.suffix in ('.h','.cu','.cpp'):
            text=path.read_text(encoding='utf-8');updated=re.sub(r'(\w+)\.type\(\)\.is_cuda\(\)',r'\1.is_cuda()',text)
            updated=updated.replace('AT_DISPATCH_FLOATING_TYPES(value.type(),','AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),')
            out.write_text(updated,encoding='utf-8')
        else:shutil.copy2(path,out)
        receipt.append({'file':rel.as_posix(),'official_sha256':sha(path),'compat_sha256':sha(out),'changed':sha(path)!=sha(out)})
    write_json(ROOT/'reports/kernel_source_identity.json',{'status':'SOURCE_PREPARED_GPU_COMPILE_PENDING','time':utc_now(),
        'author_commit':'fd64a5190f9f4530e8e5961cc87b2a3ad591190b',
        'change':'Only deprecated ATen type().is_cuda()/dispatch type() replaced by tensor is_cuda()/scalar_type(); no algorithm change',
        'files':receipt})

if __name__=='__main__':main()
