"""Design arithmetic checks only: no torch, models, pixels, GPU, or SGD."""
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EPS = 1e-5
CHECKS = []


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def softplus(x):
    return max(x, 0) + math.log1p(math.exp(-abs(x)))


def main_class(z, y, valid):
    indices = [i for i, keep in enumerate(valid) if keep]
    if not indices:
        raise ValueError("all-ignore main supervision")
    if any(not math.isfinite(z[i]) for i in indices):
        raise ValueError("nonfinite supervised logits")
    grad = [0.0] * len(z)
    p = {i: sigmoid(z[i]) for i in indices}
    gt = sum(y[i] for i in indices)
    if gt == 0:
        value = sum(softplus(z[i]) for i in indices) / len(indices)
        for i in indices:
            grad[i] = p[i] / len(indices)
        return value, grad
    numerator = 2 * sum(p[i] * y[i] for i in indices) + EPS
    denominator = sum(p[i] ** 2 for i in indices) + gt + EPS
    for i in indices:
        grad[i] = (2 * numerator * p[i] / denominator ** 2 - 2 * y[i] / denominator) * p[i] * (1 - p[i])
    return 1 - numerator / denominator, grad


def aux_instance(z, y, valid):
    indices = [i for i, keep in enumerate(valid) if keep]
    if not indices:
        return None
    p = [sigmoid(z[i]) for i in indices]
    yy = [y[i] for i in indices]
    return 1 - (2 * sum(a * b for a, b in zip(p, yy)) + EPS) / (sum(a * a for a in p) + sum(yy) + EPS)


def close(a, b, atol=1e-9):
    assert abs(a - b) <= atol, (a, b)


def checked(name, detail):
    CHECKS.append({"name": name, "status": "PASS", "detail": detail})


def main():
    spec_path = ROOT / "jointcontext_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    rng = random.Random(42)
    max_error = 0.0
    for trial in range(120):
        n = 2 * rng.randrange(2, 15)
        z = [rng.uniform(-8, 8) for _ in range(n)]
        y = [int(rng.random() < .25) for _ in range(n)]
        valid = [rng.random() > .2 for _ in range(n)]
        valid[0] = True
        if trial % 4 == 0:
            y = [0] * n
        value, grad = main_class(z, y, valid)
        for i in range(n):
            plus, minus = list(z), list(z)
            plus[i] += 1e-5
            minus[i] -= 1e-5
            numeric = (main_class(plus, y, valid)[0] - main_class(minus, y, valid)[0]) / 2e-5
            max_error = max(max_error, abs(numeric - grad[i]))
            assert abs(numeric - grad[i]) < 2e-8
            if valid[i]:
                assert grad[i] <= 1e-12 if y[i] else grad[i] >= -1e-12
            else:
                assert grad[i] == 0
        assert math.isfinite(value)
    checked("conditional main loss finite-difference gradients", {"cases": 120, "max_absolute_error": max_error})
    checked("positive/negative gradient direction and ignore zeros", "Checked each logit in the same 120 cases; statement is about logits, not shared network updates")

    val, grad = main_class([1000, -1000, float("nan")], [0, 0, 0], [True, True, False])
    close(val, 500)
    close(grad[0], .5)
    assert grad[1] == grad[2] == 0
    checked("empty-class BCE numerical extremes and masked NaN", "finite softplus at +/-1000; supervised false-positive gradient remains nonzero")
    try:
        main_class([0], [0], [False])
        raise AssertionError("missing rejection")
    except ValueError:
        pass
    checked("all-ignore main batch", "rejected before a model forward in the execution contract")
    values = [aux_instance([float("nan")], [0], [False]), aux_instance([0], [0], [True])]
    active = [x for x in values if x is not None]
    assert len(active) == 1
    close(sum(active) / len(active), values[1])
    assert aux_instance([0], [0], [False]) is None
    checked("aux valid-background and all-ignore are distinct", "all-background kept; all-ignore excluded; empty aux set maps to zero, no aux BN call")

    z, y, valid = [-.7, .2, 1.3, -.1, .9, -.3], [1, 0, 0, 0, 0, 0], [True] * 6
    pooled, pooled_grad = main_class(z, y, valid)
    permutation = [5, 2, 0, 4, 1, 3]
    pval, _ = main_class([z[i] for i in permutation], [y[i] for i in permutation], valid)
    close(pooled, pval)
    assert pooled_grad[4] > 0
    checked("pooled reduction permutation and negative-image supervision", "negative pixels share the positive batch numerator; no per-image ratio averaging")

    for _ in range(100):
        n, g, kc, kd = [rng.uniform(-3, 3) for _ in range(4)]
        common, difference = (n + g) / 2, (n - g) / 2
        close(common + difference, n)
        close(common - difference, g)
        close(kc * common + kd * difference, (kc + kd) / 2 * n + (kc - kd) / 2 * g)
    checked("common/difference basis and concatenated-kernel equivalence", "100 scalar instances; applies elementwise and by convolution linearity")
    kc, kd, n, g, upstream = 2.0, 0.0, 3.0, 1.0, .7
    close((kc + kd) / 2 * upstream, .7)
    close((kc - kd) / 2 * upstream, .7)
    assert ((n - g) / 2) * upstream != 0
    checked("zero difference kernel does not sever either input route", "both inputs receive the common-kernel derivative; difference weights have a first-step derivative in this nondegenerate example")

    theta, xn, xg, kc, kd, label = 1.2, .7, 1.5, 1.1, -.2, .4
    def objective(t):
        n, g = t * xn, t * xg
        return .5 * (kc * (n + g) / 2 + kd * (n - g) / 2 - label) ** 2
    n, g = theta * xn, theta * xg
    error = kc * (n + g) / 2 + kd * (n - g) / 2 - label
    gn, gg = (kc + kd) / 2 * error, (kc - kd) / 2 * error
    staged = gn * xn + gg * xg
    numeric = (objective(theta + 1e-5) - objective(theta - 1e-5)) / 2e-5
    close(staged, numeric, 1e-9)
    checked("two-view staged VJP chain rule", {"staged": staged, "direct_finite_difference": numeric, "scope": "scalar algebra only, not the CUDA model"})

    bank = {"mean": 0.0, "count": 0}
    def bn_example(values, update):
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        if update:
            bank["mean"] = .9 * bank["mean"] + .1 * mean
            bank["count"] += 1
        return [(x - mean) / math.sqrt(variance + EPS) for x in values]
    original = bn_example([1, 2, 4, 5], True)
    state = dict(bank)
    recomputed = bn_example([1, 2, 4, 5], False)
    assert original == recomputed and bank == state and bank["count"] == 1
    checked("recompute BN state contract", "training batch statistics reproduce normalization without a second EMA/counter write; formula-level check only")

    decoder = 720 * 64 + 2 * 64 + 2 * (64 * 64 * 7 * 7) + 2 * 64 + 64 * 4 + 4
    assert decoder == spec["decoder"]["expected_parameters"] == 448004
    assert decoder + spec["backbone"]["expected_parameters"] == spec["decoder"]["expected_total_parameters"]
    assert spec["constraints"]["updates"] * spec["constraints"]["global_batch"] == 80000
    checked("parameter and update budget arithmetic", {"decoder": decoder, "total": 65773124, "original_images": 80000})
    rates = {str(t): .005 * (1 - t / 40000) * (1e-6 + (1 - 1e-6) * t / 1500 if t < 1500 else 1) for t in [0, 1499, 1500, 39999]}
    close(rates["0"], 5e-9)
    assert all(x > 0 and math.isfinite(x) for x in rates.values())
    checked("schedule boundary arithmetic", rates)
    report = {"reviewer": "root", "date": "2026-10-17", "scope": "design algebra and scalar numerical checks only", "model_executed": False,
              "training_updates": 0, "new_pixel_evaluation": False, "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
              "checks": CHECKS, "passed": len(CHECKS), "failed": 0}
    (ROOT / "jointcontext_algebra_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
