"""Offline design arithmetic only: no torch, model loading, inference or training."""
from pathlib import Path
import hashlib
import json
import math
import random

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
spec = json.loads((ROOT / '02_架构规格.json').read_text(encoding='utf-8'))
w = spec['student_fusion']['weights']
assert sum(w) == 1.0 and all(v > 0 for v in w)

def conv(ci, co, k=1, bias=False, groups=1):
    return ci * co * k * k // groups + (co if bias else 0)

def ln(c):
    return 2 * c

def block(c, k):
    return conv(c, c, k, groups=c) + ln(c) + conv(c, c)

l_count = sum(conv(a, b, k) + ln(b) for a, b, k, _ in spec['detail_L']['rgb_stem'])
l_count += sum(conv(a, b) + ln(b) for a, b in spec['detail_L']['semantic_projections'])
l_count += conv(96, 64) + ln(64) + 2 * block(64, 3) + conv(64, 4, bias=True)
c_count = sum(conv(a, b) + ln(b) for a, b in spec['context_C']['semantic_projections'])
c_count += 2 * block(64, 7) + conv(64, 4, bias=True)
h_count = sum(conv(a, b, k, bias=True) for a, b, k in spec['H4']['convolutions'])
assert [l_count, c_count, h_count] == [26004, 58372, 3361]

# Multiplications-accumulations count convolution weights, not biases/norm/interpolation.
l_mac = 1024**2*conv(3,16,3) + 512**2*conv(16,32,3)
l_mac += 256**2*conv(48,32) + 128**2*conv(96,32) + 512**2*conv(96,64)
l_mac += 2*512**2*(conv(64,64,3,groups=64)+conv(64,64)) + 512**2*conv(64,4)
c_mac = sum(hw**2*conv(ci,64) for hw,ci in [(128,96),(64,192),(32,384)])
c_mac += sum(hw**2*(conv(64,64,7,groups=64)+conv(64,64)) for hw in [64,128])
c_mac += 128**2*conv(64,4)
h_mac_per_pixel = sum(conv(a,b,k) for a,b,k in spec['H4']['convolutions'])

def logsigmoid(x):
    return -max(-x, 0.0) - math.log1p(math.exp(-abs(x)))

def lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x-m) for x in xs))

def mix_logit(zs):
    return lse([math.log(a)+logsigmoid(z) for a,z in zip(w,zs)]) - lse([math.log(a)+logsigmoid(-z) for a,z in zip(w,zs)])

identity_error = max(abs(mix_logit([z,z,z])-z) for z in [-1000,-100,-20,-2,0,2,20,100,1000])
assert identity_error < 1e-10
rng = random.Random(42)
reconstruct_error = 0.0
for _ in range(10000):
    a,u,s = [rng.random() for _ in range(3)]
    mean = sum(x*y for x,y in zip(w,[a,u,s]))
    d_l, d_c = u-a, s-a
    ra = mean-w[1]*d_l-w[2]*d_c
    reconstruct_error = max(reconstruct_error, abs(ra-a), abs(ra+d_l-u), abs(ra+d_c-s))
assert reconstruct_error < 1e-12

# Explicit counterexample: binary OR degrades IoU while a soft mixture improves it.
gt = [1,1,0,0]
pa = [.9,.4,.4,.1]
pb = [.8,.9,.51,.8]
def iou(p):
    pred = [x > .5 for x in p]
    tp = sum(a and b for a,b in zip(pred,gt))
    union = sum(a or b for a,b in zip(pred,gt))
    return tp/union
or_prediction = [float(a>.5 or b>.5) for a,b in zip(pa,pb)]
soft_prediction = [.75*a+.25*b for a,b in zip(pa,pb)]
or_example = {'A':iou(pa), 'OR':iou(or_prediction), 'probability_blend':iou(soft_prediction)}
# Here OR equals baseline; append one further candidate false positive to make it worse.
gt.append(0); pa.append(.1); pb.append(.8)
or_prediction = [float(a>.5 or b>.5) for a,b in zip(pa,pb)]
soft_prediction = [.75*a+.25*b for a,b in zip(pa,pb)]
or_example = {'A':iou(pa), 'OR':iou(or_prediction), 'probability_blend':iou(soft_prediction)}
assert or_example['OR'] < or_example['A'] < or_example['probability_blend']

split_result = None
identity_path = WORKSPACE / 'HRDecoder_FinalDecision_20261005/execution/server_results/reports/data_identity.json'
official_root = WORKSPACE.parent / 'lesion_segmentation'
if identity_path.is_file() and (official_root / 'train/image').is_dir():
    identities = json.loads(identity_path.read_text(encoding='utf-8'))
    official = {s:{p.stem for p in (official_root / s / 'image').iterdir() if p.is_file()} for s in ['train','valid','test']}
    split_result = {}
    for split in ['train','val','test']:
        own = {s['sid'] for s in identities['splits'][split]['samples']}
        split_result[split] = {s:len(own & ids) for s,ids in official.items()}
    assert split_result['train'] == {'train':232,'valid':80,'test':142}

selected_sources = [
    'HRDecoder_FinalDecision_20261005/最佳模型技术报告/HRDecoder最佳模型训练与改进报告.md',
    'HRDecoder_FinalDecision_20261005/结题报告/HRDecoder项目结题报告.md',
    'HRDecoder_FinalDecision_20261005/结题报告/evidence/original_HRDecoder_baseline_evidence.json',
    'HRDecoder_FinalDecision_20261005/结题报告/evidence/final_locked_test_evidence.json',
    '_全实验档案_20260920/02_HRDecoder逐实验事实档案.md',
    'HRDecoder_审查轮_20260916/tools/h4_post_fusion_corrector.py',
    'autodl100/work/HRDecoder/mmseg/models/segmentors/HRDecoder.py',
    'HRDecoder_重新诊断与三点提升方案_20260926/04_审查修正与证据链.md',
]
hashes = {}
for relative in selected_sources:
    path = WORKSPACE / relative
    hashes[relative] = {'bytes':path.stat().st_size, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
result = {
    'scope':'design arithmetic and existing ID-list readback only; no model execution',
    'new_parameters_analytical': {'L':l_count,'C':c_count,'H4':h_count,'total':l_count+c_count+h_count},
    'new_convolution_MAC_per_image':{'L_2048':l_mac,'C_2048':c_mac,'H4_per_output_pixel':h_mac_per_pixel,'total_if_output_2048':l_mac+c_mac+h_mac_per_pixel*2048**2},
    'three_teacher_train_caches_454_2048_FP32_GiB':454*3*4*2048**2*4/1024**3,
    'scalar_float64_identity_max_abs_error':identity_error,
    'mean_difference_inverse_max_abs_error_10000_synthetic_samples':reconstruct_error,
    'synthetic_OR_counterexample_IoU':or_example,
    'custom_official_image_ID_intersections':split_result,
    'main_updates':40000,'H4_updates_within_main_updates':40000-38400,
    'training_or_inference_performed':False,
    'GPU_checks_performed':False,
    'actual_model_parameter_count_measured':False,
    'source_hashes':hashes
}
(ROOT/'evidence/design_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k != 'source_hashes'},ensure_ascii=False,indent=2))
