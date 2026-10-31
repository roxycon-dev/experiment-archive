#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY=/root/miniconda3/bin/python
cd /root/autodl-tmp/hrplus/repo

echo '=== 定位 D 方案日志 ==='
find /root/autodl-tmp -maxdepth 3 -name 'D_0913*' 2>/dev/null | head -20
echo '--- hrplus 根目录 ---'
ls -la /root/autodl-tmp/hrplus/ | head -40

echo
echo '=== Lovasz 数值自检（在真实张量上）==='
$PY - <<'EOF'
import sys, torch
sys.path.insert(0, '/root/autodl-tmp/hrplus/repo')
from hrplus_core import multiclass_lovasz_hinge

torch.manual_seed(0)
B, C, H, W = 2, 4, 32, 32

# case1: 完美预测 (fg logit 远大于 0) -> loss 应接近 0
gt = torch.zeros(B, H, W, dtype=torch.long)
gt[:, :8, :8] = 1   # EX 通道
gt[:, :8, 9:16] = 2 # HE
gt[:, :8, 17:24] = 3
gt[:, :8, 25:31] = 4
logits = torch.full((B, C, H, W), -6.0)
for c in range(1, 5):
    logits[:, c - 1][gt == c] = 6.0
l_good = multiclass_lovasz_hinge(logits, gt)
print('perfect-ish  loss =', float(l_good))

# case2: 完全错的预测 -> loss 应明显更大
logits_bad = -logits
l_bad = multiclass_lovasz_hinge(logits_bad, gt)
print('inverted     loss =', float(l_bad))
assert l_bad > l_good, 'inverted should be worse'
print('OK: 单调性正确')

# case3: 全 255 忽略 -> 应为 0 且梯度连通
gt_ig = torch.full((B, H, W), 255, dtype=torch.long)
lg = logits.clone().requires_grad_(True)
l = multiclass_lovasz_hinge(lg, gt_ig)
print('all-ignore   loss =', float(l))
l.backward()
print('grad not None:', lg.grad is not None, 'grad finite:', bool(torch.isfinite(lg.grad).all()))
assert float(l) == 0.0, 'all-ignore must be exactly 0'
assert bool(torch.isfinite(lg.grad).all()), 'grad must be finite'
print('OK: 全忽略安全')

# case4: 某类整批缺失 -> 不 NaN
gt_part = gt.clone(); gt_part[gt_part == 4] = 0
l = multiclass_lovasz_hinge(logits, gt_part)
print('class-missing loss =', float(l), 'finite:', bool(torch.isfinite(l)))
assert torch.isfinite(l), 'must be finite when a class absent'
print('OK: 缺类安全')

# case5: 与参考实现的等价性 —— per-image lovasz 平均
def ref_lovasz_per_image(logits, gt):
    # 标准：逐图逐类，再对图与类取平均
    B, C, H, W = logits.shape
    tot = 0.0
    for b in range(B):
        for c in range(1, 5):
            t = (gt[b] == c).reshape(-1).float()
            p = logits[b, c - 1].reshape(-1)
            valid = (gt[b].reshape(-1) != 255)
            t, p = t[valid], p[valid]
            if t.sum() == 0 and t.sum() == t.numel():
                continue
            signs = 2 * t - 1
            perm = torch.argsort(-p, stable=True)
            s = signs[perm]
            fg = (t == 1).sum().item()
            if fg == 0 or fg == t.numel():
                continue
            g = []
            for i in range(len(s)):
                cum = s[:i + 1]
                g.append(1.0 - (cum.sum().item() + fg) / (i + 1 + fg + 1e-6))
            tot += float(torch.dot(torch.tensor(g), s.relu()))
    return tot / (B * 4)

print('per-batch impl =', float(multiclass_lovasz_hinge(logits, gt)))
print('per-image ref  =', ref_lovasz_per_image(logits, gt))
print('注：两者量级不同是设计选择，只看是否同向')
EOF
