"""Compile only the pinned, inspected CUDA dependency required by S/SD."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def load_kernel():
    if 'MultiScaleDeformableAttention' in sys.modules:return sys.modules['MultiScaleDeformableAttention']
    if os.name!='nt':
        os.environ.setdefault('CUDA_HOME','/root/miniconda3/envs/hrdecoder_q_20261003')
        binpath=str(Path(os.environ['CUDA_HOME'])/'bin')
        if binpath not in os.environ.get('PATH','').split(os.pathsep):os.environ['PATH']=binpath+os.pathsep+os.environ.get('PATH','')
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.9') # Assigned RTX4090; avoid unrelated architecture compilation.
    from torch.utils.cpp_extension import load
    from qlib.common import read_json,sha
    source=ROOT/'vendor/msda_compat/src';receipt=read_json(ROOT/'reports/kernel_source_identity.json')
    for row in receipt['files']:assert sha(source/row['file'])==row['compat_sha256']
    build=ROOT/'vendor/msda_build';build.mkdir(exist_ok=True)
    # PyTorch's CUDA runtime wheels already contain the matching BLAS/sparse
    # headers. nvcc's standalone compiler prefix does not include those paths.
    import torch
    nvidia=Path(torch.__file__).resolve().parents[1]/'nvidia'
    target_headers=sorted(Path(os.environ.get('CUDA_HOME','/nonexistent')).glob('targets/*/include'))
    includes=[str(source),*[str(p) for p in target_headers],*[str(p) for p in sorted(nvidia.glob('*/include')) if p.is_dir()]]
    os.environ.setdefault('MAX_JOBS','4')
    module=load(name='MultiScaleDeformableAttention',sources=[str(source/'vision.cpp'),
        *map(str,(source/'cpu').glob('*.cpp')),*map(str,(source/'cuda').glob('*.cu'))],
        extra_include_paths=includes,extra_cflags=['-O3','-DWITH_CUDA'],
        extra_cuda_cflags=['-O3','-DWITH_CUDA','-D__CUDA_NO_HALF_OPERATORS__','-D__CUDA_NO_HALF_CONVERSIONS__',
                          '-D__CUDA_NO_HALF2_OPERATORS__'],build_directory=str(build),verbose=True)
    sys.modules['MultiScaleDeformableAttention']=module
    return module
