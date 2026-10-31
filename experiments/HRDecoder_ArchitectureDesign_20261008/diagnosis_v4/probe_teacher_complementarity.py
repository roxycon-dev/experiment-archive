"""Read-only, one-pass audit of the remote A/U/S train probability caches.

The local driver streams the embedded worker over the existing SSH key-auth
connection. The remote worker reads one image's three ZIP members in bounded
blocks and returns aggregate/per-image scalar statistics only; no image or
probability arrays leave the server.
"""
from __future__ import annotations

import importlib.util
import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
REMOTE_HELPER = ROOT / "local_preparation" / "remote.py"
REPORT_JSON = HERE / "evidence" / "teacher_complementarity.json"
REPORT_MD = HERE / "evidence" / "teacher_complementarity.md"

REMOTE_ROOT = "/root/autodl-tmp/HRDecoder_v3_20261008"
DATA_ROOT = "/root/HRDecoder/data/DDR_split_613"
BLOCK = 262144
THRESHOLD = 0.5


REMOTE_WORKER = r'''import hashlib, json, os, sys, time, zipfile
from contextlib import ExitStack
import numpy as np
from PIL import Image

ROOT = "/root/autodl-tmp/HRDecoder_v3_20261008"
DATA = "/root/HRDecoder/data/DDR_split_613"
CLASSES = ["EX", "HE", "SE", "MA"]
MEMBERS = ["A", "U", "S"]
WEIGHTS = {"A": .5625, "U": .1875, "S": .25}
THRESHOLD = .5
BLOCK = 262144

def fresh_counts():
    return {k: 0 for k in ("tp", "fp", "fn", "tn", "ties")}

def update_metrics(acc, pos, neg, pred, ties):
    acc["tp"] += int(np.count_nonzero(pos & pred))
    acc["fp"] += int(np.count_nonzero(neg & pred))
    acc["fn"] += int(np.count_nonzero(pos & ~pred))
    acc["tn"] += int(np.count_nonzero(neg & ~pred))
    acc["ties"] += int(np.count_nonzero(ties))

def metric_view(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    den_dice = 2 * tp + fp + fn
    den_iou = tp + fp + fn
    return {**counts,
        "dice_pixel_pooled": (2 * tp / den_dice if den_dice else None),
        "iou_pixel_pooled": (tp / den_iou if den_iou else None)}

def fresh_direction():
    return {k: 0.0 for k in ("correct_mass", "wrong_mass", "absolute_mass",
                              "correct_pixels", "wrong_pixels", "zero_pixels")}

def update_direction(acc, delta, is_positive):
    # delta = member probability minus weighted mean of the other two.
    # Positive GT wants delta>0; negative GT wants delta<0.
    toward = delta if is_positive else -delta
    correct = toward > 0
    wrong = toward < 0
    acc["correct_mass"] += float(toward[correct].sum(dtype=np.float64))
    acc["wrong_mass"] += float((-toward[wrong]).sum(dtype=np.float64))
    acc["absolute_mass"] += float(np.abs(toward).sum(dtype=np.float64))
    acc["correct_pixels"] += int(np.count_nonzero(correct))
    acc["wrong_pixels"] += int(np.count_nonzero(wrong))
    acc["zero_pixels"] += int(np.count_nonzero(toward == 0))

def direction_view(d, pixels):
    mass = d["absolute_mass"]
    return {**d,
        "correct_fraction_of_abs_disagreement": d["correct_mass"] / mass if mass else None,
        "wrong_fraction_of_abs_disagreement": d["wrong_mass"] / mass if mass else None,
        "correct_mass_per_gt_pixel": d["correct_mass"] / pixels if pixels else None,
        "wrong_mass_per_gt_pixel": d["wrong_mass"] / pixels if pixels else None}

def open_prob(stack, path, shape):
    zf = stack.enter_context(zipfile.ZipFile(path, "r"))
    names = zf.namelist()
    if names != ["prob.npy"]:
        raise RuntimeError(f"unexpected NPZ members {path}: {names}")
    stream = stack.enter_context(zf.open("prob.npy", "r"))
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        got_shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        got_shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise RuntimeError(f"unsupported NPY version {version}: {path}")
    if list(got_shape) != shape or fortran or dtype != np.dtype("float32"):
        raise RuntimeError(f"cache header mismatch {path}: {got_shape}, {fortran}, {dtype}")
    return stream

def process_image(row, indexes, aggregate):
    sid = row["sid"]
    h, w = row["hw"]
    n = h * w
    label_path = os.path.join(DATA, "labels", "train", sid + ".png")
    with Image.open(label_path) as im:
        label = np.asarray(im, dtype=np.uint8).reshape(-1)
    if label.size != n:
        raise RuntimeError(f"GT shape mismatch {sid}: {label.shape} vs {row['hw']}")
    unknown = set(np.unique(label).tolist()) - {0, 1, 2, 3, 4, 255}
    if unknown:
        raise RuntimeError(f"unexpected GT values {sid}: {sorted(unknown)}")

    entries, paths = [], []
    for member in MEMBERS:
        entry = indexes[member]["entries"][sid]
        if entry["shape"] != [4, h, w] or entry["dtype"] != "float32":
            raise RuntimeError(f"index shape/dtype mismatch {sid}/{member}")
        if entry["sid"] != sid or entry["image_sha256"] != row["image_sha256"]:
            raise RuntimeError(f"index image identity mismatch {sid}/{member}")
        if entry["label_sha256"] != row["label_sha256"]:
            raise RuntimeError(f"index label identity mismatch {sid}/{member}")
        entries.append(entry)
        paths.append(os.path.join(ROOT, "cache", "train", member, entry["file"]))

    classes = []
    hashes = [hashlib.sha256() for _ in paths]
    valid_total = int(np.count_nonzero(label != 255))
    with ExitStack() as stack:
        streams = [open_prob(stack, path, [4, h, w]) for path in paths]
        for c, cname in enumerate(CLASSES):
            metrics = {k: fresh_counts() for k in ("A", "U", "S", "A_plus_U", "A_plus_U_plus_S")}
            patterns_pos = [0] * 8
            patterns_neg = [0] * 8
            rescue = {k: 0 for k in ("T_fn_any_member_positive", "T_fn_A_positive", "T_fn_U_positive",
                                     "T_fn_S_positive", "A_fn_U_correct", "A_fn_S_correct",
                                     "A_fn_both_U_S_correct", "A_fp_U_correct", "A_fp_S_correct",
                                     "A_fp_both_U_S_correct", "T_fp_any_member_negative",
                                     "T_fp_A_negative", "T_fp_U_negative", "T_fp_S_negative",
                                     "T_fp_any_member_correct_negative")}
            directions = {m: {"gt_positive": fresh_direction(), "gt_negative": fresh_direction()} for m in MEMBERS}
            pos_count = neg_count = 0
            for start in range(0, n, BLOCK):
                stop = min(start + BLOCK, n)
                size = stop - start
                prob = []
                for stream, digest in zip(streams, hashes):
                    raw = stream.read(size * 4)
                    if len(raw) != size * 4:
                        raise RuntimeError(f"truncated cache {sid}/{cname}")
                    digest.update(raw)
                    v = np.frombuffer(raw, dtype=np.float32)
                    if not np.isfinite(v).all() or v.min(initial=0) < 0 or v.max(initial=0) > 1:
                        raise RuntimeError(f"invalid cache probability {sid}/{cname}")
                    prob.append(v)
                a, u, s = prob
                gt = label[start:stop]
                valid = gt != 255
                pos = gt == c + 1
                neg = valid & ~pos
                if not valid.any():
                    continue

                # Fixed probability ladder whose final weights are the locked A/U/S target.
                au = .75 * a + .25 * u
                t = .75 * au + .25 * s
                # Explicit weighted expression cross-checks the algebra/weights.
                t_direct = WEIGHTS["A"] * a + WEIGHTS["U"] * u + WEIGHTS["S"] * s
                if np.max(np.abs(t - t_direct), initial=0) > 2e-7:
                    raise RuntimeError(f"teacher mixture mismatch {sid}/{cname}")
                pa, pu, ps = a > THRESHOLD, u > THRESHOLD, s > THRESHOLD
                pau, pt = au > THRESHOLD, t > THRESHOLD
                for key, pred, ties in (
                    ("A", pa, a == THRESHOLD), ("U", pu, u == THRESHOLD), ("S", ps, s == THRESHOLD),
                    ("A_plus_U", pau, au == THRESHOLD), ("A_plus_U_plus_S", pt, t == THRESHOLD)):
                    update_metrics(metrics[key], pos, neg, pred, ties & valid)

                pos_count += int(np.count_nonzero(pos))
                neg_count += int(np.count_nonzero(neg))
                # Most-significant to least-significant bits are A/U/S so that
                # the printed three-character pattern reads in that same order.
                pattern = (pa.astype(np.uint8) << 2) | (pu.astype(np.uint8) << 1) | ps.astype(np.uint8)
                if pos.any():
                    patterns_pos += np.bincount(pattern[pos], minlength=8).tolist()
                if neg.any():
                    patterns_neg += np.bincount(pattern[neg], minlength=8).tolist()

                t_fn = pos & ~pt
                t_fp = neg & pt
                rescue["T_fn_any_member_positive"] += int(np.count_nonzero(t_fn & (pa | pu | ps)))
                rescue["T_fn_A_positive"] += int(np.count_nonzero(t_fn & pa))
                rescue["T_fn_U_positive"] += int(np.count_nonzero(t_fn & pu))
                rescue["T_fn_S_positive"] += int(np.count_nonzero(t_fn & ps))
                rescue["A_fn_U_correct"] += int(np.count_nonzero(pos & ~pa & pu))
                rescue["A_fn_S_correct"] += int(np.count_nonzero(pos & ~pa & ps))
                rescue["A_fn_both_U_S_correct"] += int(np.count_nonzero(pos & ~pa & pu & ps))
                rescue["A_fp_U_correct"] += int(np.count_nonzero(neg & pa & ~pu))
                rescue["A_fp_S_correct"] += int(np.count_nonzero(neg & pa & ~ps))
                rescue["A_fp_both_U_S_correct"] += int(np.count_nonzero(neg & pa & ~pu & ~ps))
                rescue["T_fp_any_member_negative"] += int(np.count_nonzero(t_fp & (~pa | ~pu | ~ps)))
                rescue["T_fp_A_negative"] += int(np.count_nonzero(t_fp & ~pa))
                rescue["T_fp_U_negative"] += int(np.count_nonzero(t_fp & ~pu))
                rescue["T_fp_S_negative"] += int(np.count_nonzero(t_fp & ~ps))
                rescue["T_fp_any_member_correct_negative"] += int(np.count_nonzero(t_fp & (~pa | ~pu | ~ps)))

                # Member probability versus the weighted mean of the other two.
                rest = {
                    "A": (WEIGHTS["U"] * u + WEIGHTS["S"] * s) / (WEIGHTS["U"] + WEIGHTS["S"]),
                    "U": (WEIGHTS["A"] * a + WEIGHTS["S"] * s) / (WEIGHTS["A"] + WEIGHTS["S"]),
                    "S": (WEIGHTS["A"] * a + WEIGHTS["U"] * u) / (WEIGHTS["A"] + WEIGHTS["U"]),
                }
                for name, value in (("A", a), ("U", u), ("S", s)):
                    delta = value - rest[name]
                    update_direction(directions[name]["gt_positive"], delta[pos], True)
                    update_direction(directions[name]["gt_negative"], delta[neg], False)

            if pos_count == 0 and neg_count == 0:
                raise RuntimeError(f"no valid GT pixels {sid}/{cname}")
            classes.append({
                "class": cname, "gt_positive_pixels": pos_count, "gt_negative_pixels": neg_count,
                "metrics": {k: metric_view(v) for k, v in metrics.items()},
                "joint_prediction_patterns": {
                    "bit_order": {"A": 4, "U": 2, "S": 1},
                    "gt_positive": {format(i, "03b"): patterns_pos[i] for i in range(8)},
                    "gt_negative": {format(i, "03b"): patterns_neg[i] for i in range(8)},
                },
                "rescue_and_member_correct_counts": rescue,
                "directional_probability_disagreement_vs_weighted_other_two": {
                    m: {stratum: direction_view(value, pos_count if stratum == "gt_positive" else neg_count)
                        for stratum, value in d.items()} for m, d in directions.items()
                },
            })
        for stream, digest, entry, path in zip(streams, hashes, entries, paths):
            extra = stream.read(1)
            if extra != b"":
                raise RuntimeError(f"excess probability bytes {sid}: {path}; tell={stream.tell()}, "
                                   f"shape={entry['shape']}, uncompressed={entry['uncompressed_bytes']}, extra={extra.hex()}")
            if digest.hexdigest() != entry["probability_sha256"]:
                raise RuntimeError(f"tensor SHA mismatch {sid}: {path}")

    return {"sid": sid, "hw": [h, w], "valid_pixels": valid_total, "class_statistics": classes}

def add_tree(dst, src):
    if isinstance(dst, dict):
        for k, v in src.items():
            if isinstance(v, dict):
                add_tree(dst.setdefault(k, {}), v)
            elif isinstance(v, (int, float)):
                dst[k] = dst.get(k, 0) + v
    # lists and metadata deliberately not aggregated here.

def main():
    started = time.time()
    with open(os.path.join(ROOT, "metadata", "data_identity.json"), "r", encoding="utf-8") as f:
        identity = json.load(f)
    rows = identity["splits"]["train"]["samples"]
    if len(rows) != 454:
        raise RuntimeError(f"expected 454 train images, got {len(rows)}")
    indexes = {}
    for member in MEMBERS:
        path = os.path.join(ROOT, "cache", "train", member, "index.json")
        with open(path, "r", encoding="utf-8") as f:
            ix = json.load(f)
        if ix["status"] != "COMPLETE" or ix["completed_images"] != 454 or ix["expected_images"] != 454:
            raise RuntimeError(f"cache index incomplete: {member}")
        if ix["identity"]["channels"] != CLASSES or ix["identity"]["coordinates"] != "original_image":
            raise RuntimeError(f"cache contract mismatch: {member}")
        if ix["identity"]["in_sample_teacher_predictions"] is not True or ix["identity"]["OOF"] is not False:
            raise RuntimeError(f"teacher OOF identity unexpected: {member}")
        indexes[member] = ix

    aggregate = {}
    by_image = []
    for i, row in enumerate(rows, 1):
        result = process_image(row, indexes, aggregate)
        by_image.append(result)
        # Compact progress; no raw predictions are emitted.
        print(json.dumps({"progress": i, "total": len(rows), "sid": row["sid"]}), flush=True)

    # Build aggregate summaries directly from per-image scalar rows.
    summary = {c: {"metrics": {}, "joint_prediction_patterns": {"gt_positive": {format(i, "03b"): 0 for i in range(8)},
                                                                 "gt_negative": {format(i, "03b"): 0 for i in range(8)}},
                   "rescue_and_member_correct_counts": {},
                   "directional_probability_disagreement_vs_weighted_other_two": {m: {"gt_positive": fresh_direction(), "gt_negative": fresh_direction()} for m in MEMBERS},
                   "images_with_gt_positive": 0, "images_with_gt_negative": 0}
               for c in CLASSES}
    for img in by_image:
        for row in img["class_statistics"]:
            c = row["class"]; dst = summary[c]
            if row["gt_positive_pixels"]: dst["images_with_gt_positive"] += 1
            if row["gt_negative_pixels"]: dst["images_with_gt_negative"] += 1
            for model, m in row["metrics"].items():
                add_tree(dst["metrics"].setdefault(model, fresh_counts()), {k: m[k] for k in fresh_counts()})
            for stratum in ("gt_positive", "gt_negative"):
                for pat, n in row["joint_prediction_patterns"][stratum].items():
                    dst["joint_prediction_patterns"][stratum][pat] += n
            add_tree(dst["rescue_and_member_correct_counts"], row["rescue_and_member_correct_counts"])
            for m in MEMBERS:
                for stratum in ("gt_positive", "gt_negative"):
                    src = row["directional_probability_disagreement_vs_weighted_other_two"][m][stratum]
                    raw = {k: src[k] for k in fresh_direction()}
                    add_tree(dst["directional_probability_disagreement_vs_weighted_other_two"][m][stratum], raw)
    for c in CLASSES:
        s = summary[c]
        s["metrics"] = {k: metric_view(v) for k, v in s["metrics"].items()}
        for m in MEMBERS:
            for stratum in ("gt_positive", "gt_negative"):
                d = s["directional_probability_disagreement_vs_weighted_other_two"][m][stratum]
                pixels = s["metrics"]["A"]["tp"] + s["metrics"]["A"]["fn"] if stratum == "gt_positive" else s["metrics"]["A"]["tn"] + s["metrics"]["A"]["fp"]
                s["directional_probability_disagreement_vs_weighted_other_two"][m][stratum] = direction_view(d, pixels)

    report = {
        "status": "PASS", "purpose": "read-only cached teacher complementarity statistics",
        "remote_root": ROOT, "data_root": DATA, "split": "train", "n_images": len(rows),
        "cache_members": MEMBERS, "channel_order": CLASSES, "coordinates": "original_image",
        "gt_encoding": {"0": "background", "1": "EX", "2": "HE", "3": "SE", "4": "MA", "255": "ignore"},
        "threshold_rule": f"probability > {THRESHOLD}; equality is counted as tie and predicted negative",
        "probability_ladder": {"A": "A", "A_plus_U": ".75*A + .25*U", "A_plus_U_plus_S": ".75*(.75*A+.25*U)+.25*S = .5625*A+.1875*U+.25*S"},
        "joint_pattern_bit_order": {"A": 4, "U": 2, "S": 1},
        "per_image": by_image, "aggregate_by_class": summary,
        "teacher_cache_is_in_sample": True, "OOF": False,
        "raw_images_or_probability_arrays_transferred": False,
        "block_elements": BLOCK, "arrays_loaded_whole": False,
        "seconds": time.time() - started,
        "interpretation_limit": "454 train-set predictions are in-sample; complementarity is descriptive and not independent/generalization evidence."
    }
    print("RESULT_JSON_BEGIN", flush=True)
    print(json.dumps(report, separators=(",", ":")), flush=True)
    print("RESULT_JSON_END", flush=True)

if __name__ == "__main__":
    main()
'''


def render_markdown(report: dict) -> str:
    lines = [
        "# A/U/S 教师缓存互补性只读审计",
        "",
        f"- 数据：训练 split {report['n_images']} 张；缓存 `[4,H,W]`，原图坐标，通道 EX/HE/SE/MA。",
        "- GT：`0=background, 1=EX, 2=HE, 3=SE, 4=MA, 255=ignore`。按 `p>0.5` 判正，等于 0.5 记 tie 且判负。",
        "- 融合阶梯：A → `.75A+.25U` → `.5625A+.1875U+.25S`。这是固定概率权重的缓存统计，不等于独立测试结果。",
        "- 每个像素模式字符串顺序为 A/U/S；例如 `010` 表示仅 U 判阳性。正负 GT 区分别统计。",
        "- 所有教师训练预测均为 in-sample（OOF=false）；下面的互补是训练集描述统计，不能当独立泛化证据。",
        "- 远端逐图逐块处理三份缓存，未传出原图或大数组；逐份验证缓存 tensor SHA。",
        "",
        "## 固定融合阶梯的像素池化指标",
        "",
        "| 类别 | 指标 | A | A+U | A+U+S | 阳性GT图数 | GT+像素 | GT−像素 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cname in ("EX", "HE", "SE", "MA"):
        row = report["aggregate_by_class"][cname]
        for met in ("dice_pixel_pooled", "iou_pixel_pooled"):
            vals = [row["metrics"][k][met] for k in ("A", "A_plus_U", "A_plus_U_plus_S")]
            fmt = lambda x: "n/a" if x is None else f"{x:.6f}"
            lines.append(f"| {cname} | {met} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} | {row['images_with_gt_positive']} | {row['metrics']['A']['tp'] + row['metrics']['A']['fn']} | {row['metrics']['A']['tn'] + row['metrics']['A']['fp']} |")
    lines += ["", "## 各教师与固定混合的逐类 TP/FP/FN", ""]
    lines.append("| 类别 | 输出 | TP | FP | FN | Dice | IoU | ties |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for cname in ("EX", "HE", "SE", "MA"):
        metrics = report["aggregate_by_class"][cname]["metrics"]
        for key in ("A", "U", "S", "A_plus_U", "A_plus_U_plus_S"):
            m = metrics[key]
            lines.append(f"| {cname} | {key} | {m['tp']} | {m['fp']} | {m['fn']} | {m['dice_pixel_pooled']:.6f} | {m['iou_pixel_pooled']:.6f} | {m['ties']} |")
    lines += ["", "## 互补像素计数", "", "每类下列 A/FN→U/S/二者修复统计发生在 GT+ 像素；T FN/FP 条目针对固定最终混合。成员二值模式的完整 8 格分布及逐图详情均在相邻 JSON。", ""]
    lines.append("| 类别 | A漏检且U正确 | A漏检且S正确 | A漏检且U/S都正确 | T漏检但任一成员判阳 | T误报但任一成员判阴 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cname in ("EX", "HE", "SE", "MA"):
        r = report["aggregate_by_class"][cname]["rescue_and_member_correct_counts"]
        lines.append(f"| {cname} | {r.get('A_fn_U_correct',0)} | {r.get('A_fn_S_correct',0)} | {r.get('A_fn_both_U_S_correct',0)} | {r.get('T_fn_any_member_positive',0)} | {r.get('T_fp_any_member_correct_negative',0)} |")
    lines += ["", "## 相对其余两教师的概率分歧方向", "", "`correct_fraction_of_abs_disagreement` 是成员相对其余两教师加权均值的绝对概率差中，朝 GT 正确方向的比例；wrong fraction 同理。GT+ 应向上，GT− 应向下。它描述概率互补方向，不代表性能因果。", "", "| 类别 | 成员 | GT+正确质量占比 | GT+错误质量占比 | GT−正确质量占比 | GT−错误质量占比 |", "|---|---|---:|---:|---:|---:|"]
    for cname in ("EX", "HE", "SE", "MA"):
        dirs = report["aggregate_by_class"][cname]["directional_probability_disagreement_vs_weighted_other_two"]
        for member in ("A", "U", "S"):
            p, n = dirs[member]["gt_positive"], dirs[member]["gt_negative"]
            def fmt(x): return "n/a" if x is None else f"{x:.4f}"
            lines.append(f"| {cname} | {member} | {fmt(p['correct_fraction_of_abs_disagreement'])} | {fmt(p['wrong_fraction_of_abs_disagreement'])} | {fmt(n['correct_fraction_of_abs_disagreement'])} | {fmt(n['wrong_fraction_of_abs_disagreement'])} |")
    lines += ["", "## 限制", "", "这些结果来自训练缓存，且 A/U/S 均在该训练集上产生预测（OOF=false）；因此它们能证明缓存输出之间的像素级互补/冲突样式，不足以证明验证/测试增益或统计独立性。全量缓存对应 454 张图，缓存完成记录所报合计压缩 84.24 GB、未压缩 112.64 GB；当前分析按单图 block 流处理。", ""]
    return "\n".join(lines)


def main() -> int:
    spec = importlib.util.spec_from_file_location("hrdecoder_remote", REMOTE_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load existing SSH helper: {REMOTE_HELPER}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    print(f"Connecting with existing campaign key; remote worker reads one image at a time. Script={__file__}", flush=True)
    client = helper.connect()
    try:
        payload = base64.b64encode(REMOTE_WORKER.encode("utf-8")).decode("ascii")
        command = "/root/miniconda3/bin/python -u -c \"import base64; exec(compile(base64.b64decode('" + payload + "'), '<teacher-complementarity>', 'exec'))\""
        stdin, stdout, stderr = client.exec_command(command, get_pty=False)
        channel = stdout.channel
        stdin.close()
        stdout_text, stderr_text = [], []
        result_text = None
        in_result = False
        pending = ""

        def handle_line(line: str) -> None:
            nonlocal in_result, result_text
            if line == "RESULT_JSON_BEGIN":
                in_result = True
                result_text = ""
            elif line == "RESULT_JSON_END":
                in_result = False
            elif in_result:
                result_text = line
            else:
                stdout_text.append(line)
                if line.startswith('{"progress":'):
                    event = json.loads(line)
                    if event["progress"] % 25 == 0 or event["progress"] == 1 or event["progress"] == event["total"]:
                        print(f"Processed {event['progress']}/{event['total']}: {event['sid']}", flush=True)

        while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
            if channel.recv_ready():
                chunk = channel.recv(65536).decode("utf-8", errors="replace")
                pending += chunk
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    handle_line(line.rstrip("\r"))
            if channel.recv_stderr_ready():
                stderr_text.append(channel.recv_stderr(65536).decode("utf-8", errors="replace"))
        rc = channel.recv_exit_status()
        if pending:
            handle_line(pending.rstrip("\r"))
        if rc != 0:
            raise RuntimeError("remote worker failed: " + "".join(stderr_text)[-6000:] + "\n" + "\n".join(stdout_text[-40:]))
        if not result_text:
            raise RuntimeError("remote worker finished without RESULT_JSON")
        report = json.loads(result_text)
    finally:
        client.close()
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {REPORT_JSON} and {REPORT_MD}; {report['n_images']} images, {report['seconds']:.1f}s remote analysis.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
