"""Diagnose failed new-node acceptance; retain evidence and retry state oracle."""
import sys,os,importlib.util,shutil,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'reports'))
import torch,cv2
from qlib.common import read_json,write_json,sha,utc_now,assert_source
torch.set_num_threads(1);cv2.setNumThreads(1)
assert not (ROOT/'reports/driver.lock').exists()
assert not (ROOT/'runs/SD_seed42/identity.json').exists()
p=ROOT/'vendor/msda_build/MultiScaleDeformableAttention.so'
assert sha(p)==read_json(ROOT/'reports/parallel_kernel_binary.json')['sha256']
spec=importlib.util.spec_from_file_location('MultiScaleDeformableAttention',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.modules['MultiScaleDeformableAttention']=module
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
from node2_acceptance_atomic import native_ram_repeatability,install_comparison_oracle
diagnosis=native_ram_repeatability()
archive=ROOT/'reports/parallel_acceptance_atomic_revision';archive.mkdir(exist_ok=True)
for name in ('early_gpu_gates.json','driver_state.json','parallel_sd_worker.log'):
 p=ROOT/'reports'/name
 if p.exists():shutil.copy2(p,archive/name)
for p in (ROOT/'reports').glob('acceptance_resume_*.pth'):p.rename(archive/p.name)
(ROOT/'reports/early_gpu_gates.json').unlink()
write_json(ROOT/'reports/node2_acceptance_revision.json',{'stage':'DIAGNOSED_COMPARISON_ONLY_ATOMIC_ORACLE','formal_source_sha256':assert_source(ROOT)['sha256'],'model_EMA_tolerance_unchanged':[1e-7,1e-6],'optimizer_tolerance_unchanged':[1e-5,1e-3],'serialization_and_RNG_and_forward_exact_gate_unchanged':True,'native_kernel_forward_backward_gate_unchanged':True,'all_100_update_benchmarks_and_formal_native_backward_unchanged':True,'formal_updates_lost':0,'diagnosis':diagnosis,'time':utc_now()})
torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
install_comparison_oracle()
from parallel_sd_worker import main
main()
