"""Executable decoder specification, not a training entry point.

No backbone, dataset, optimizer, checkpoint loading or GPU requirement.
The audit uses small synthetic feature maps and never performs an SGD update.
"""
from pathlib import Path
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F


class ExplicitConv(nn.Conv2d):
    def reset_parameters(self):
        # Every parameter is written explicitly in Decoder.__init__.
        # Avoid temporary initialization consuming the prescribed RNG sequence.
        pass


class BankBN(nn.Module):
    def __init__(self, banks):
        super().__init__()
        self.banks = tuple(banks)
        self.weight = nn.Parameter(torch.ones(64))
        self.bias = nn.Parameter(torch.zeros(64))
        for bank in banks:
            self.register_buffer('mean_' + bank, torch.zeros(64))
            self.register_buffer('var_' + bank, torch.ones(64))
            self.register_buffer('count_' + bank, torch.tensor(0, dtype=torch.long))

    def forward(self, x, bank):
        if bank not in self.banks:
            raise ValueError(bank)
        if self.training:
            getattr(self, 'count_' + bank).add_(1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = F.batch_norm(x.float(), getattr(self, 'mean_' + bank),
                             getattr(self, 'var_' + bank), self.weight, self.bias,
                             self.training, momentum=0.1, eps=1e-5)
        return y.to(x.dtype)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.p_g = ExplicitConv(720, 64, 1, bias=False)
        self.p_n = ExplicitConv(720, 64, 1, bias=False)
        self.k_g = ExplicitConv(64, 64, 7, padding=3, bias=False)
        self.k_n = ExplicitConv(64, 64, 7, padding=3, bias=False)
        self.pbn_g, self.pbn_n = BankBN(['G', 'aux']), BankBN(['N'])
        self.sbn_g, self.sbn_n = BankBN(['G', 'aux']), BankBN(['N'])
        self.mixer = ExplicitConv(128, 64, 1, bias=False)
        self.classifier = ExplicitConv(64, 4, 1, bias=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            nn.init.kaiming_normal_(self.p_g.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.k_g.weight, mode='fan_out', nonlinearity='relu')
            nn.init.normal_(self.classifier.weight, mean=0, std=0.01)
        with torch.no_grad():
            self.classifier.bias.zero_()
            self.p_n.weight.copy_(self.p_g.weight)
            self.k_n.weight.copy_(self.k_g.weight)
            self.pbn_n.weight.copy_(self.pbn_g.weight)
            self.pbn_n.bias.copy_(self.pbn_g.bias)
            self.sbn_n.weight.copy_(self.sbn_g.weight)
            self.sbn_n.bias.copy_(self.sbn_g.bias)
            self.mixer.weight.zero_()
            indices = torch.arange(64)
            self.mixer.weight[indices, indices, 0, 0] = 0.5
            self.mixer.weight[indices, indices + 64, 0, 0] = 0.5

    def global_branch(self, feature, bank='G'):
        x = F.relu(self.pbn_g(self.p_g(feature), bank))
        return F.relu(self.sbn_g(self.k_g(x), bank))

    def native_branch(self, feature):
        x = F.relu(self.pbn_n(self.p_n(feature), 'N'))
        return F.relu(self.sbn_n(self.k_n(x), 'N'))

    def forward(self, fg, fn, aux_features=None):
        g_low = self.global_branch(fg)
        g = F.interpolate(g_low.float(), size=fn.shape[-2:],
                          mode='bilinear', align_corners=False)
        n = self.native_branch(fn)
        h = F.relu(self.mixer(torch.cat([n.float(), g], dim=1)))
        z = self.classifier(h)
        auxiliary = None
        if aux_features is not None:
            if aux_features.shape[0] == 0:
                raise ValueError('All-ignored auxiliary instances must be skipped before calling the head')
            auxiliary = self.classifier(self.global_branch(aux_features, 'aux'))
        return {'logits_grid': z, 'aux_logits_grid': auxiliary,
                'N': n, 'G': g, 'G_before_upsample': g_low}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent
    spec_path = root / 'jointcontext_spec_r4.json'
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    before = torch.get_rng_state().clone()
    model = Decoder()
    checks = {'initialization_preserves_caller_rng': torch.equal(before, torch.get_rng_state())}
    count = sum(p.numel() for p in model.parameters())
    checks['actual_decoder_parameter_count'] = count == spec['decoder']['expected_parameters'] == 502532
    copy_pairs = [(model.p_g.weight, model.p_n.weight), (model.k_g.weight, model.k_n.weight),
                  (model.pbn_g.weight, model.pbn_n.weight), (model.pbn_g.bias, model.pbn_n.bias),
                  (model.sbn_g.weight, model.sbn_n.weight), (model.sbn_g.bias, model.sbn_n.bias)]
    checks['copies_equal_but_independent'] = all(torch.equal(a, b) and a.data_ptr() != b.data_ptr()
                                                for a, b in copy_pairs)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        fg = torch.randn(2, 720, 8, 8)
        fn = torch.randn(2, 720, 16, 16)
        aux = torch.randn(3, 720, 8, 8)
    model.eval()
    with torch.no_grad():
        outputs = model(fg, fn)
        direct_average = 0.5 * (model.classifier(outputs['N']) + model.classifier(outputs['G']))
        identity_error = float((outputs['logits_grid'] - direct_average).abs().max())
        checks['initial_average_identity'] = torch.allclose(outputs['logits_grid'], direct_average, atol=2e-6, rtol=2e-6)
        # Include nonzero common bias in this separate algebraic identity case.
        model.classifier.bias.copy_(torch.tensor([0.13, -0.17, 0.23, -0.29]))
        biased = model(fg, fn)
        target = 0.5 * (model.classifier(biased['N']) + model.classifier(biased['G']))
        checks['common_bias_not_doubled'] = torch.allclose(biased['logits_grid'], target, atol=2e-6, rtol=2e-6)

    model = Decoder().train()
    outputs = model(fg, fn, aux)
    named = list(model.named_parameters())
    aux_loss = outputs['aux_logits_grid'].square().mean()
    aux_grads = torch.autograd.grad(aux_loss, [p for _, p in named], allow_unused=True, retain_graph=True)
    aux_support = {name: grad is not None for (name, _), grad in zip(named, aux_grads)}
    checks['aux_only_G_decoder_and_classifier'] = all(
        supported == name.startswith(('p_g.', 'pbn_g.', 'k_g.', 'sbn_g.', 'classifier.'))
        for name, supported in aux_support.items())
    (outputs['logits_grid'].square().mean() + 0.1 * aux_loss).backward()
    gradient_norms = {name: float(p.grad.norm()) if p.grad is not None else None for name, p in named}
    checks['synthetic_main_aux_gradients_finite_and_present'] = all(
        p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm()) > 0
        for _, p in named)
    counters_before = {name: int(value) for name, value in model.named_buffers() if 'count_' in name}
    with torch.no_grad():
        model(fg, fn, None)
    counters_after = {name: int(value) for name, value in model.named_buffers() if 'count_' in name}
    checks['BN_domain_updates_and_skip_aux'] = (all(x == 1 for x in counters_before.values()) and all(
        value == (1 if name.endswith('_aux') else 2) for name, value in counters_after.items()))

    # Fixed-state 1-D spatial operator counterexample, away from boundaries.
    impulse = torch.zeros(1, 1, 32, dtype=torch.float64)
    impulse[0, 0, 13] = 1
    k = torch.ones(1, 1, 7, dtype=torch.float64) / 7
    up = lambda x, factor: F.interpolate(x, scale_factor=factor, mode='linear', align_corners=False)
    resize_first = F.conv1d(up(impulse, 2), k, padding=3)
    decode_first = up(F.conv1d(impulse, k, padding=3), 2)
    a, b = float(resize_first[0, 0, 32]), float(decode_first[0, 0, 32])
    checks['spatial_order_counterexample'] = a == 0 and math.isclose(b, 1/7, abs_tol=1e-12)
    # Two resizes are not a direct resize: no B0-function identity claim.
    cascade = up(up(impulse, 2), 4)
    direct = up(impulse, 8)
    cascade_error = float((cascade-direct).abs().max())
    checks['nested_resize_not_B0_identity'] = cascade_error > 0

    proj_macs = 720 * 64 * (256**2 + 512**2)
    kernel_params = 64 * 64 * 7 * 7
    old_main = proj_macs + 2 * kernel_params * 512**2 + 64 * 4 * 512**2
    new_main = proj_macs + kernel_params * (256**2 + 512**2) + 128 * 64 * 512**2 + 64 * 4 * 512**2
    result = {'date':'2026-10-18','reviewer':'root','torch_version':torch.__version__,
              'device':'CPU','dtype':'FP32 decoder; FP64 spatial counterexample',
              'scope':'decoder reference and architecture identities only; no W48 model or metric validation',
              'checks':checks,'passed':all(checks.values()),
              'spec_sha256':sha(spec_path),'script_sha256':sha(Path(__file__)),
              'decoder_parameters':count,'total_parameters_with_archived_backbone':count+65325120,
              'initial_average_max_abs_error':identity_error,'synthetic_gradient_norms':gradient_norms,
              'aux_parameter_gradient_support':aux_support,'BN_counters_before_skip':counters_before,
              'BN_counters_after_skip':counters_after,
              'spatial_order_example':{'coarse_input_length':32,'coarse_impulse_index':13,'fine_output_index':32,
                                       'resize_then_7tap':a,'7tap_then_resize':b,
                                       'scope':'one spatial axis, fixed linear decoder operator; not observed accuracy'},
              'two_resize_vs_one_max_abs_error':cascade_error,
              'convolution_MACs_per_original_main_decoder':{'r3':old_main,'r4':new_main,
                  'reduction_fraction':1-new_main/old_main,'scope':'excludes W48, M4, interpolation, BN, activations, memory and backward'},
              'new_optimizer_steps':0,'full_model_executed':False,'data_pixels_read':False,
              'new_val_test_metric':None,'performance_guaranteed':False}
    (root/'r4_decoder_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['passed','checks','decoder_parameters','total_parameters_with_archived_backbone',
        'initial_average_max_abs_error','spatial_order_example','two_resize_vs_one_max_abs_error',
        'convolution_MACs_per_original_main_decoder','new_optimizer_steps']},ensure_ascii=False,indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    audit()
