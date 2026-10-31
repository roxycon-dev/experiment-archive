"""Architecture algebra and archived-count analysis; no torch or training."""
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
TIMM = Path(r"C:\Users\GuanXingRen\Miniconda3\Lib\site-packages\timm\models\hrnet.py")


def literal(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        assert not node.args and all(k.arg is not None for k in node.keywords)
        return {k.arg: literal(k.value) for k in node.keywords}
    return ast.literal_eval(node)


def row_rank(rows):
    a = [[float(x) for x in row] for row in rows]
    pivot = 0
    for col in range(len(a[0])):
        found = next((r for r in range(pivot, len(a)) if abs(a[r][col]) > 1e-12), None)
        if found is None:
            continue
        a[pivot], a[found] = a[found], a[pivot]
        scale = a[pivot][col]
        a[pivot] = [x / scale for x in a[pivot]]
        for r in range(len(a)):
            if r != pivot and abs(a[r][col]) > 1e-12:
                coeff = a[r][col]
                a[r] = [x - coeff * y for x, y in zip(a[r], a[pivot])]
        pivot += 1
        if pivot == len(a):
            break
    return pivot


def main():
    spec_path = ROOT / "jointcontext_spec_r3.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    cfg_node = next(node.value for node in ast.parse(TIMM.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cfg_cls" for t in node.targets))
    cfg = literal(cfg_node)["hrnet_w48"]
    assert cfg["stage1"]["block_type"] == "BOTTLENECK"
    # Track one convolution path through the newly created lowest branch.
    # Ignoring upsample support and other paths is deliberate; not effective RF.
    rf, jump = 1, 1
    path = []
    for name in ["stem1", "stem2"]:
        rf += 2 * jump
        jump *= 2
        path.append({"node": name, "rf_side": rf, "jump": jump})
    rf += cfg["stage1"]["num_blocks"][0] * 2 * jump
    path.append({"node": "stage1", "rf_side": rf, "jump": jump})
    for stage in [2, 3, 4]:
        rf += 2 * jump
        jump *= 2
        c = cfg["stage" + str(stage)]
        assert c["block_type"] == "BASIC"
        rf += c["num_modules"] * c["num_blocks"][-1] * 2 * 2 * jump
        path.append({"node": "stage" + str(stage) + "_lowest_branch", "rf_side": rf, "jump": jump})
    assert rf == 2783 and jump == 32

    pn = [[int(j == i) for j in range(720)] for i in range(64)]
    pg = [[int(j == i + 64) for j in range(720)] for i in range(64)]
    tied_rank, independent_rank = row_rank(pn + pn), row_rank(pn + pg)
    assert tied_rank == 64 and independent_rank == 128

    projection = 720 * 64 + 2 * 64
    decoder = 2 * projection + 2 * 64 * 64 * 7 * 7 + 2 * 64 + 64 * 4 + 4
    assert decoder == spec["decoder"]["expected_parameters"] == 494212
    assert decoder + spec["backbone"]["expected_parameters"] == 65819332

    prior = json.loads((PROJECT / "review_20261017/root_independent_audit.json").read_text(encoding="utf-8"))
    budgets = {}
    current_ious, target_ious = [], []
    for name, data in prior["per_class"].items():
        tp, fp, fn = data["v3_counts"]
        ttp, tfp, tfn = data["target_counts"]
        assert tp + fn == ttp + tfn
        target = ttp / (ttp + tfp + tfn)
        rhs = target * (tp + fn + fp) - tp
        current_ious.append(tp / (tp + fp + fn))
        target_ious.append(target)
        budgets[name] = {"target_iou": target,
                         "strict_requirement_if_matching_this_class": "delta_TP - target_iou * delta_FP > rhs",
                         "rhs": rhs,
                         "scope": "sufficient per-class route, not necessary for beating mean IoU"}
    gap = 100 * (sum(target_ious) - sum(current_ious)) / 4
    assert math.isclose(gap, 5.896280222360275, abs_tol=1e-10)
    result = {
        "date": "2026-10-18", "reviewer": "root", "scope": "architecture-only algebra; no learned-performance conclusion",
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {str(TIMM): hashlib.sha256(TIMM.read_bytes()).hexdigest()},
        "receptive_field": {"path": path, "side": rf, "scope": "nominal convolution support of one deepest path at inference; not effective or full-network RF"},
        "channel_row_space_example": {"shared_basis_union_rank": tied_rank, "independent_basis_union_rank": independent_rank,
                                     "scope": "rank in the SAME 720-channel coordinate space across two view roles; both joint block-diagonal projections still have at most rank 128 in 1440-dimensional input space",
                                     "not_established": "actual lesion information lies in these extra directions"},
        "containment": {"construction": "P_N=P_G with matching affine and corresponding domain buffers reproduces v2; split parameters are separate objects",
                        "initial_forward_identity": "algebraic", "same_SGD_trajectory": False,
                        "full_model_runtime_identity_tested": False,
                        "applies_to_prediction_family_not_regularized_finite_step_optimization": True},
        "parameters": {"per_projection": projection, "decoder": decoder, "model": decoder + 65325120, "extra_vs_v2": projection},
        "target_gap_percentage_points": gap, "error_budgets": budgets,
        "optimizer_steps": 0, "full_model_executed": False, "new_miou": None,
        "guaranteed_target_exceedance": False}
    (ROOT / "architecture_capacity_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
