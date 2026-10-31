"""FP32 masked v3 losses and the sole SGD/teacher schedules."""
import torch
import torch.nn.functional as F

def macro(value, valid):
    active = valid.sum((-2, -1)) > 0
    return torch.where(active, value, torch.zeros_like(value)).sum()/active.sum().clamp_min(1)

def one_hot(label, reference):
    if label.ndim == 4: label = label[:, 0]
    return torch.stack([label == c for c in range(1, 5)], dim=1).to(reference.dtype)

def masked_dice(logits, label, valid):
    logits = logits.float()
    if logits.shape[-2:] != label.shape[-2:]:
        logits = F.interpolate(logits, size=label.shape[-2:], mode="bilinear", align_corners=False)
    probability = logits.sigmoid()
    target = one_hot(label, probability)
    mask = valid.float().expand_as(probability)
    numerator = 2*(probability*target*mask).sum((-2, -1))+1e-5
    denominator = ((probability.square()+target.square())*mask).sum((-2, -1))+1e-5
    return macro(1-numerator/denominator, mask)

def masked_kl(logits, teacher, valid):
    logits = logits.float(); teacher = teacher.detach().float()
    mask = valid.float().expand_as(logits)
    entropy = -(torch.xlogy(teacher, teacher)+torch.xlogy(1-teacher, 1-teacher))
    loss = F.binary_cross_entropy_with_logits(logits, teacher, reduction="none")-entropy
    return macro((loss*mask).sum((-2, -1))/mask.sum((-2, -1)).clamp_min(1), mask)

def signed_aux(predictions, differences, valid, *, coordinate="signed"):
    values = []
    for prediction, difference in zip(predictions, differences):
        prediction = prediction.float(); difference = difference.detach().float()
        mask = valid.float().expand_as(prediction)
        weight = mask*(1+4*difference.abs())
        target = difference if coordinate == "signed" else difference.abs()
        error = F.smooth_l1_loss(prediction, target, beta=.1, reduction="none")
        values.append(macro((weight*error).sum((-2, -1))/weight.sum((-2, -1)).clamp_min(1), mask))
    return (values[0]+values[1])*.5

def teacher_coordinates(a, u, s):
    a, u, s = a.detach().float(), u.detach().float(), s.detach().float()
    p0 = .75*a+.25*u
    return .75*p0+.25*s, u-a, s-p0

def lambda_k(update):
    assert 0 <= update <= 40000
    if update <= 1500: return 0.
    if update <= 4000: return .5*(update-1500)/2500
    if update <= 32000: return .5
    return .5-.4*(update-32000)/8000

def learning_rate(index):
    # Original MMCV PolyLrUpdaterHook: warmup multiplies the current poly LR.
    assert 0 <= index < 40000
    regular = .005*(1-index/40000)
    if index < 1500:
        regular *= 1-(1-index/1500)*(1-1e-6)
    return regular

def losses(output, label, valid, a, u, s, update, *, mode="v3"):
    dice = masked_dice(output["logits"], label, valid)
    kd = aux = output["logits"].float().sum()*0
    if mode != "B0":
        teacher, du, ds = teacher_coordinates(a, u, s)
        kd = masked_kl(output["logits"], teacher, valid)
        if mode != "B1":
            aux = signed_aux((output["d_hat_u"], output["d_hat_s"]), (du, ds), valid)
    lk = 0. if mode == "B0" else lambda_k(update)
    task = dice+.1*output["hr_loss"]+lk*kd
    ld = .2*lk if mode not in ("B0", "B1", "capacity_kd") else 0.
    return task, ld*aux, {"dice": dice, "hr_dice": output["hr_loss"], "kl": kd,
        "signed_aux": aux, "lambda_k": lk, "lambda_d": ld}

def routed_backward(task_loss, auxiliary_loss, private_parameters, *, scale=1., route="private"):
    if route == "ordinary":
        ((task_loss+auxiliary_loss)*scale).backward()
        return
    assert route == "private"
    parameters = tuple(private_parameters)
    if not parameters:
        (task_loss*scale).backward()
        return
    auxiliary_gradients = torch.autograd.grad(auxiliary_loss*scale, parameters,
        retain_graph=True, allow_unused=True)
    (task_loss*scale).backward()
    for parameter, gradient in zip(parameters, auxiliary_gradients):
        if gradient is not None:
            if parameter.grad is None: parameter.grad = gradient.detach()
            else: parameter.grad.add_(gradient.detach())
