"""SACD FP32 probability transport, complementary targets and routed supervision."""
import torch
import torch.nn.functional as F
from fast_objectives import gt_loss, trusted_kl

def lambda_k(update):
    assert 0 <= update <= 40000
    if update <= 1500: return 0.
    if update <= 4000: return .5 * (update - 1500) / 2500
    if update <= 32000: return .5
    return .5 - .4 * (update - 32000) / 8000

def learning_rate(index):
    assert 0 <= index < 40000
    regular = .005 * (1 - index / 40000)
    if index < 1500: regular *= 1 - (1 - index / 1500) * (1 - 1e-6)
    return regular

def transport_logits(z0, action_logits):
    """Actions [N,4,3,H,W], stable unclamped log-probability composition."""
    dtype = torch.float64 if z0.dtype == torch.float64 else torch.float32
    z = z0.to(dtype)
    assert tuple(action_logits.shape) == (z.shape[0], 4, 3, *z.shape[2:])
    logs = action_logits.to(dtype).log_softmax(dim=2)
    keep, on, off = logs.unbind(dim=2)
    logp = torch.logaddexp(keep + F.logsigmoid(z), on)
    logq = torch.logaddexp(keep + F.logsigmoid(-z), off)
    return logp - logq, logs

@torch.no_grad()
def complementary_targets(a, u, s, label, valid, base_logits):
    a, u, s = (x.detach().float() for x in (a, u, s))
    assert a.shape == u.shape == s.shape and a.shape[1] == 4
    teacher = .5625*a + .1875*u + .25*s
    qmax = torch.maximum(torch.maximum(a, u), s)
    qmin = torch.minimum(torch.minimum(a, u), s)
    labels = label[:, 0] if label.ndim == 4 else label
    y = torch.stack([labels == c for c in range(1, 5)], dim=1)
    if valid.ndim == 3: valid = valid[:, None]
    mask = valid.bool().expand_as(y) & (labels[:, None] != 255)
    # Keep CET's final soft target; change only what the action head is correcting.
    teacher_plus = mask & y & (teacher <= .5) & (qmax > .5)
    teacher_minus = mask & ~y & (teacher > .5) & (qmin <= .5)
    corrected_teacher = torch.where(teacher_plus, qmax, torch.where(teacher_minus, qmin, teacher))
    student = base_logits.detach().float().sigmoid()
    assert student.shape == teacher.shape
    plus = mask & y & (student <= .5) & (qmax > .5)
    minus = mask & ~y & (student > .5) & (qmin <= .5)
    zero = mask & ((student > .5) == y)
    # Each active-region denominator already meets this bound. The bound only
    # makes excluded pixels finite, so dense kernels equal indexed targets and
    # avoid CUDA nonzero/index synchronization. No active probability is clipped.
    on = torch.where(plus,(qmax-student)/(1-student).clamp_min(.5),0.)
    off = torch.where(minus,(student-qmin)/student.clamp_min(.5),0.)
    keep = 1 - on - off
    return {'teacher': teacher, 'corrected_teacher': corrected_teacher,
            'action_target': torch.stack((keep, on, off), dim=2),
            'regions': (plus, minus, zero), 'valid': mask,
            'unknown': mask & ~(plus | minus | zero)}

def action_kl(log_probs, targets):
    q = targets['action_target'].detach().to(log_probs.dtype)
    # xlogy handles exact zero target entries; no probability clamp.
    per_pixel = (torch.xlogy(q, q) - q*log_probs).sum(dim=2)
    pair = torch.zeros_like(per_pixel[:, :, 0, 0])
    for region in targets['regions']:
        count = region.sum((-2, -1))
        pair = pair + (per_pixel * region).sum((-2, -1)) / count.clamp_min(1)
    active = targets['valid'].sum((-2, -1)) > 0
    return (pair * active).sum() / (3 * active.sum().clamp_min(1))

def losses(output, label, valid, a, u, s, update):
    final_gt = gt_loss(output['logits'], label, valid)
    base_gt = gt_loss(output['base_logits'], label, valid)
    gt = .75*final_gt + .25*base_gt + .1*output['hr_loss']
    targets = complementary_targets(a, u, s, label, valid, output['base_logits'])
    lk = lambda_k(update)
    final_kd = trusted_kl(output['logits'], targets['corrected_teacher'], label, valid)
    base_kd = trusted_kl(output['base_logits'], targets['teacher'], label, valid)
    actions = action_kl(output['action_log_probs'], targets)
    task = gt + lk*(final_kd + .25*base_kd)
    auxiliary = .25*lk*actions
    return task, auxiliary, {'gt_final': final_gt, 'gt_base': base_gt,
        'gt_hr_crop': output['hr_loss'], 'kl_final': final_kd, 'kl_base': base_kd,
        'action_kl': actions, 'lambda_k': lk, 'lambda_action': .25*lk,
        'task_loss': task, 'weighted_action_loss': auxiliary}

def routed_backward(task, auxiliary, parameters, *, action_active=True):
    parameters = tuple(parameters)
    if not action_active:
        task.backward()
        return
    gradients = torch.autograd.grad(auxiliary, parameters, retain_graph=True, allow_unused=True)
    task.backward()
    for parameter, gradient in zip(parameters, gradients):
        if gradient is not None:
            if parameter.grad is None: parameter.grad = gradient.detach()
            else: parameter.grad.add_(gradient.detach())
