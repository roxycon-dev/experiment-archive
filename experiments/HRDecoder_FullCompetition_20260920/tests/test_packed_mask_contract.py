# -*- coding: utf-8 -*-
"""test_packed_mask_contract — 专家掩膜契约的 CPU 单测（不需要 GPU/不需要真实权重）。

覆盖：
1) packbits 往返（含不同 H/W、非 8 倍数、1×1）；
2) 填充位/字节数不符必须拒绝；
3) 严格阈值（p==0.5 判阴）、fp16 拒绝、非有限/越界/通道数拒绝；
4) 逐类 TP/FP/FN 语义（含 ignore 像素永不计 FP）；
5) OR / AND 与"共同/独有"分解的独立参照实现一致 + 恒等式；
6) union==0 fail closed；
7) 原子写 + 契约完整性（篡改/缺失/多出概率字段都必须被 read_model_dir 拒绝）；
8) val-only 闸门（split=test、路径含 test 分量、图像数 != 76）；
9) 共享 GT 幂等与篡改检测；
10) 端到端：76 张合成小图 + 3 个模型 → 分析器 exit 0 并给出正确互补数字；篡改掩膜后 exit 2。

运行：`python tests/test_packed_mask_contract.py`（也可用 pytest 收集 test_* 函数）。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ANALYSIS = os.path.join(_REPO, "analysis")
for _p in (_REPO, _ANALYSIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import expert_mask_contract as C  # noqa: E402

TMP = os.path.join(_REPO, "tests", "_tmp_expert_contract")
N_IMAGES = C.EXPECTED_VAL_IMAGES


# ---------------------------------------------------------------------------
# 1) pack / unpack
# ---------------------------------------------------------------------------

def test_pack_unpack_roundtrip_varying_hw():
    rs = np.random.RandomState(1234)
    for (h, w) in [(1, 1), (1, 9), (8, 8), (3, 5), (13, 7), (17, 23), (32, 24), (7, 64)]:
        bits = rs.rand(4, h, w) > 0.5
        packed = C.pack_bits(bits)
        assert packed.dtype == np.uint8
        assert packed.shape == (4, C.nbytes_of(h * w)), (h, w, packed.shape)
        back = C.unpack_bits(packed, h * w).reshape(4, h, w)
        assert np.array_equal(back, bits), (h, w)
    # 非 8 倍数的有效位必须被准确记录（否则索引错位会串图）
    for n in (1, 7, 8, 9, 15, 16, 17, 100):
        b = (rs.rand(4, n) > 0.5)
        assert np.array_equal(C.unpack_bits(C.pack_bits(b), n), b), n


def test_unpack_rejects_padding_and_length():
    b = np.zeros((4, 13), dtype=bool)
    b[0, 12] = True
    p = C.pack_bits(b)
    # 正常路径
    assert np.array_equal(C.unpack_bits(p, 13), b)
    # 字节数不符
    bad = np.zeros((4, 3), np.uint8)
    try:
        C.unpack_bits(bad, 13)
        raise AssertionError("字节数不符必须拒绝")
    except C.ContractError:
        pass
    # 填充位非零（把最后一个字节的低位点亮）
    p2 = p.copy()
    p2[:, -1] |= np.uint8(0x01)
    try:
        C.unpack_bits(p2, 13)
        raise AssertionError("填充位非零必须拒绝")
    except C.ContractError:
        pass
    # 通道数错（默认期望 4）
    try:
        C.unpack_bits(np.zeros((3, 2), np.uint8), 13)
        raise AssertionError("通道数必须为 4")
    except C.ContractError:
        pass


# ---------------------------------------------------------------------------
# 2) 严格阈值 / 精度
# ---------------------------------------------------------------------------

def test_strict_threshold_and_precision():
    prob = np.array([[[0.5]], [[0.5000001]], [[0.4999999]], [[1.0]]], dtype=np.float32)
    out = C.binarize_strict(prob).reshape(-1).tolist()
    assert out == [False, True, False, True], out          # p == 0.5 → 阴性
    # fp16 必须在 .float() 之前被拦住
    for bad in (prob.astype(np.float16), prob.astype(np.float64)):
        try:
            C.binarize_strict(bad)
            raise AssertionError("非 float32 概率必须拒绝")
        except C.ContractError:
            pass
    for broken in (
        prob.copy().astype(np.float32) * np.nan,
        np.array([[[-0.1]], [[0.5]], [[0.5]], [[0.5]]], dtype=np.float32),
        np.array([[[1.1]], [[0.5]], [[0.5]], [[0.5]]], dtype=np.float32),
        np.zeros((2, 4, 4), dtype=np.float32),
        np.zeros((4, 4), dtype=np.float32),
    ):
        try:
            C.binarize_strict(broken)
            raise AssertionError("非法概率必须拒绝: %r" % (broken.shape,))
        except C.ContractError:
            pass
    # torch 侧 fp16 也要被抓到（check_float32_like 不依赖 torch）
    class _Fake:
        dtype = "torch.float16"
    try:
        C.check_float32_like(_Fake(), "t")
        raise AssertionError("fp16 必须拒绝")
    except C.ContractError:
        pass


# ---------------------------------------------------------------------------
# 3) 计数语义
# ---------------------------------------------------------------------------

def test_counts_hand_computed_with_ignore():
    # 单类单像素级别的精确断言
    gt = np.zeros((4, 6), dtype=bool)
    pred = np.zeros((4, 6), dtype=bool)
    ignore = np.zeros(6, dtype=bool)
    gt[0, 0] = True          # gt 正
    pred[0, 0] = True        # -> TP
    pred[0, 1] = True        # -> FP
    gt[0, 2] = True          # 未预测 -> FN
    pred[0, 3] = True        # 预测在 ignore 像素上 -> 不计 FP
    ignore[3] = True
    cnt = C.counts_from_bits(pred, gt, ignore)
    assert cnt[0] == [1, 1, 1], cnt[0]
    for c in range(1, 4):
        assert cnt[c] == [0, 0, 0], cnt[c]
    # ignore 以 [4,N] 形式给出也支持
    cnt2 = C.counts_from_bits(pred, gt, np.broadcast_to(ignore[None, :], gt.shape))
    assert cnt2 == cnt
    # 形状不符必须拒绝
    try:
        C.counts_from_bits(pred, gt[:, :5])
        raise AssertionError("形状不符必须拒绝")
    except C.ContractError:
        pass


def test_or_and_and_decomposition_identity():
    """OR/AND 的计数必须等于"共同/独有"分解的计数（独立参照实现）。"""
    rs = np.random.RandomState(99)
    for _ in range(20):
        n = int(rs.randint(1, 400))
        gt = rs.rand(4, n) > 0.7
        p = rs.rand(4, n) > 0.5
        q = rs.rand(4, n) > 0.5
        ign = rs.rand(n) > 0.95
        champ = C.counts_from_bits(q, gt, ign)
        cand = C.counts_from_bits(p, gt, ign)
        or_cnt = C.counts_from_bits(C.or_bits(p, q), gt, ign)
        and_cnt = C.counts_from_bits(C.and_bits(p, q), gt, ign)
        for c in range(4):
            g, pp, qq = gt[c], p[c], q[c]
            neg = (~g) & (~ign)
            common = int(np.count_nonzero(pp & qq & g))
            q_only_tp = int(np.count_nonzero(qq & ~pp & g))
            p_only_tp = int(np.count_nonzero(pp & ~qq & g))
            both_fn = int(np.count_nonzero(~pp & ~qq & g))
            q_only_fp = int(np.count_nonzero(qq & ~pp & neg))
            p_only_fp = int(np.count_nonzero(pp & ~qq & neg))
            both_fp = int(np.count_nonzero(pp & qq & neg))
            assert [common + q_only_tp, q_only_fp + both_fp,
                    both_fn + p_only_tp] == champ[c], (c, "champ 恒等式")
            assert [common + p_only_tp, p_only_fp + both_fp,
                    both_fn + q_only_tp] == cand[c], (c, "cand 恒等式")
            assert or_cnt[c] == [common + q_only_tp + p_only_tp,
                                 q_only_fp + p_only_fp + both_fp, both_fn]
            assert and_cnt[c] == [common, both_fp, both_fn + q_only_tp + p_only_tp]
        # IoU 一致性
        miou_a, per_a, _ = C.iou_from_counts(or_cnt, fail_on_zero_union=False)
        assert miou_a is not None
    # 互补构造：一半漏、一半补 → OR 恰好等于 GT（IoU=1）
    gt = np.zeros((4, 8), dtype=bool)
    gt[0, :4] = True
    a = gt.copy()
    a[0, :2] = False
    b = gt.copy()
    b[0, 2:4] = False
    orc = C.counts_from_bits(C.or_bits(a, b), gt, None)
    assert orc[0] == [4, 0, 0], orc[0]
    assert C.iou_from_counts(orc, fail_on_zero_union=False)[1]["EX"] == 1.0


def test_iou_zero_union_fails_closed():
    zeros = [[0, 0, 0] for _ in range(4)]
    try:
        C.iou_from_counts(zeros)
        raise AssertionError("union==0 必须 fail closed")
    except C.ContractError:
        pass
    miou, per, _ = C.iou_from_counts(zeros, fail_on_zero_union=False)
    assert per["EX"] is None and np.isnan(miou)


# ---------------------------------------------------------------------------
# 4) val-only 闸门
# ---------------------------------------------------------------------------

def test_val_only_guard():
    C.assert_val_only("val", paths=("/a/val/b", "/c/fullmodel"), n_images=76)
    for split in ("test", "TEST", "train", ""):
        try:
            C.assert_val_only(split)
            raise AssertionError("split=%r 必须拒绝" % split)
        except C.ContractError:
            pass
    for bad_path in ("/a/test/b", "C:/x/test", "/root/test"):
        try:
            C.assert_val_only("val", paths=(bad_path,))
            raise AssertionError("路径含 test 必须拒绝: %s" % bad_path)
        except C.ContractError:
            pass
    for n in (1, 75, 77):
        try:
            C.assert_val_only("val", n_images=n)
            raise AssertionError("图像数 %d 必须拒绝" % n)
        except C.ContractError:
            pass


# ---------------------------------------------------------------------------
# 5) 原子写 + 契约完整性
# ---------------------------------------------------------------------------

def _tiny_model_dir(root, name, family, stems, gt_dir, *, mutate=None, h=8, w=8):
    d = os.path.join(root, family, name)
    C.ensure_dir(d)
    per_image, counts = {}, [[0, 0, 0] for _ in range(4)]
    for i, s in enumerate(stems):
        g = C.load_gt_npz(os.path.join(gt_dir, C.MASK_FILENAME % s))
        bits = g["bits"].copy()
        if mutate is not None:
            bits = mutate(i, bits)
        rec = C.save_mask_npz(os.path.join(d, C.MASK_FILENAME % s), bits, (h, w))
        cnt = C.counts_from_bits(bits.reshape(4, -1), g["bits"].reshape(4, -1),
                                 g["ignore"].reshape(-1))
        rec["counts"] = {C.CLASSES[c]: list(cnt[c]) for c in range(4)}
        per_image[s] = rec
        for c in range(4):
            for j in range(3):
                counts[c][j] += cnt[c][j]
    miou, per_class, detail = C.iou_from_counts(counts)
    gt_man = C.load_gt_contract(gt_dir)["contract"]
    contract = {
        "model_name": name, "pipeline_family": family, "split": "val",
        "n_images": len(stems), "stems": list(stems),
        "eval_contract": C.build_eval_contract(
            pipeline_family=family, eval_loader="synthetic", inference="whole_image",
            tta="none"),
        "pipeline_definition": {"name": name, "synthetic": True},
        "identity": {"checkpoints": [{"role": name, "sha256": "0" * 64, "path": "/syn"}],
                     "configs": [{"path": "/syn.yaml", "sha256": "1" * 64}]},
        "gt": {"gt_contract_sha256": gt_man["gt_contract_sha256"],
               "gt_semantic_fingerprint": gt_man["gt_semantic_fingerprint"],
               "data_root": gt_man["data_root"], "split": "val"},
        "per_image": per_image,
        "global_counts": {C.CLASSES[c]: list(counts[c]) for c in range(4)},
        "metrics": {"mIoU_lesion4": miou, "per_class_iou": per_class,
                    "per_class_detail": detail},
        "self_checks": {"synthetic": True},
        "blockers": [], "created_at": "synthetic",
    }
    C.finalize_model_dir(d, contract)
    return d


def test_contract_integrity_and_atomic_write():
    root = os.path.join(TMP, "integrity")
    if os.path.isdir(root):
        shutil.rmtree(root)
    data_root, gt_dir, stems = _make_gt_fixture(root, h=8, w=8)
    d = _tiny_model_dir(root, "m0", "hrplus", stems, gt_dir)
    m = C.read_model_dir(d, expected_family="hrplus")
    assert m["name"] == "m0" and len(m["stems"]) == N_IMAGES
    got = C.load_verified_mask(m, stems[0])
    assert got["bits"].shape == (4, 8, 8)
    # family 不符
    try:
        C.read_model_dir(d, expected_family="fullmodel")
        raise AssertionError("family 不符必须拒绝")
    except C.ContractError:
        pass
    # 篡改掩膜文件 → 拒绝
    p = os.path.join(d, C.MASK_FILENAME % stems[0])
    raw = bytearray(open(p, "rb").read())
    raw[-1] ^= 0xFF
    open(p, "wb").write(bytes(raw))
    try:
        C.load_verified_mask(m, stems[0])
        raise AssertionError("篡改掩膜必须拒绝")
    except C.ContractError:
        pass
    # 恢复，再删文件 → 拒绝
    shutil.rmtree(d)
    d = _tiny_model_dir(root, "m0", "hrplus", stems, gt_dir)
    m = C.read_model_dir(d, expected_family="hrplus")
    os.remove(os.path.join(d, C.MASK_FILENAME % stems[1]))
    try:
        C.load_verified_mask(m, stems[1])
        raise AssertionError("掩膜缺失必须拒绝")
    except C.ContractError:
        pass
    # contract 里塞概率字段 → 拒绝
    cpath = os.path.join(d, C.CONTRACT_FILENAME)
    with open(cpath, encoding="utf-8") as f:
        man = json.load(f)
    man["prob_fp32"] = "forbidden"
    C.atomic_write_json(cpath, man)
    try:
        C.read_model_dir(d, expected_family="hrplus")
        raise AssertionError("契约含概率字段必须拒绝")
    except C.ContractError:
        pass
    # 删 contract → 拒绝（半截产物不可用）
    os.remove(cpath)
    try:
        C.read_model_dir(d, expected_family="hrplus")
        raise AssertionError("缺 contract 必须拒绝")
    except C.ContractError:
        pass
    # 原子写：临时文件不会残留
    tgt = os.path.join(root, "atomic.json")
    C.atomic_write_json(tgt, {"a": 1})
    assert os.path.isfile(tgt) and not os.path.isfile(tgt + ".tmp")
    assert json.load(open(tgt, encoding="utf-8")) == {"a": 1}


# ---------------------------------------------------------------------------
# 5b) 概率载荷禁令：保留键名硬黑名单 + 语义级浮点数组禁令（回归：C2000 contract）
# ---------------------------------------------------------------------------

def _expect_rejected(obj, why):
    try:
        C.assert_no_probability_payload(obj, "ut")
    except C.ContractError:
        return
    raise AssertionError("必须拒绝：%s" % why)


def test_probability_key_blacklist_and_float_payload_ban():
    """回归：元数据字段名撞上保留键 ⇒ fail closed；浮点数组禁令不可绕过。

    真实事故：pipeline_definition(C2000/S4000) 曾用保留键 `probability` 存**文字说明**，
    写 contract 时被判为概率载荷，hrplus_exit_code=2，三条管线拿不到 contract。
    修法是给元数据字段改名（probability_source），保留键黑名单**不放松**。
    """
    # (1) 保留键：无论值是什么都拒绝（含字符串说明、None、标量）
    for bad_val in ("_student_prob（说明）", None, 0.5, 1, [0, 0, 0], {"a": 1}):
        _expect_rejected({"probability": bad_val}, "保留键 probability 值=%r" % (bad_val,))
    for forbidden in C.FORBIDDEN_CONTRACT_KEYS:
        _expect_rejected({"outer": {"inner": {forbidden: "s"}}},
                         "嵌套保留键 %s" % forbidden)

    # (2) 语义级：任意键名下的浮点数组 / 纯浮点列表都必须拒绝
    for shape in ((4, 8), (2,), (4, 8, 8)):
        for dt in (np.float16, np.float32, np.float64):
            _expect_rejected({"benign": np.zeros(shape, dtype=dt)},
                             "任意键浮点数组 %s/%s" % (shape, dt.__name__))
    _expect_rejected({"data": [[0.1, 0.2], [0.3, 0.4]]}, "纯浮点嵌套列表")
    _expect_rejected({"x": np.logspace(-3, 0, 8, dtype=np.float32).tolist()},
                     "float 列表（list of python float）")

    # (3) 合法契约形状必须**放行**：指标标量、int 计数列表、字符串/字典列表
    C.assert_no_probability_payload({
        "model_name": "C2000",
        "metrics": {"mIoU_lesion4": 0.39797, "per_class_iou": {"EX": 0.45, "HE": None}},
        "global_counts": {"EX": [1, 2, 3]},
        "stems": ["007-1774-100"],
        "identity": {"checkpoints": [{"role": "C", "sha256": "0" * 64}]},
        "pipeline_definition": {"probability_source": "_student_prob（说明）"},
        "self_checks": {"champion_shared_bitwise_equal": True,
                        "champion_shared_max_abs_diff": 0.0},
        "packing": {"bitorder": "big", "n_pixels_recorded": True},
        "orig_hw": [1728, 2592],
    }, "ut")

    # (4) 真实 pipeline_definition：修复后放行，旧键名版本必须拒绝
    import export_hrplus_expert_masks as E
    for name in ("C2000", "S4000", "MIX_C2000", "champion", "BASE_0", "D_blend"):
        d = E.pipeline_definition(name)
        C.assert_no_probability_payload(d, "pipeline_definition[%s]" % name)
        assert "probability" not in d, "%s 不得再用保留键 probability" % name
        if name in ("C2000", "S4000"):
            assert "probability_source" in d
    legacy = E.pipeline_definition("C2000")
    legacy["probability"] = legacy.pop("probability_source")
    _expect_rejected(legacy, "旧 C2000 pipeline_definition（保留键 probability）")


def test_export_scripts_do_not_use_reserved_keys():
    """静态回归：两个导出脚本不得把保留键名字面量当**字典键**用。

    真实事故（连撞两次）：
      1) export_hrplus_expert_masks.pipeline_definition("C2000") 用 `"probability"`；
      2) export_fullmodel_masks 的 pipeline_definition 用 `"probabilities"`。
    两者都只是**文字说明**，却在写 contract 时被载荷黑名单 fail closed。
    保留键名黑名单不放松，所以这里把"再用一次"变成单测失败。
    """
    import re
    editors = [os.path.join(_ANALYSIS, "export_hrplus_expert_masks.py"),
               os.path.join(_ANALYSIS, "export_fullmodel_masks.py")]
    reserved = set(C.FORBIDDEN_CONTRACT_KEYS)
    # 只看"用作字典键"的字面量：  "prob": / 'prob':
    pat = re.compile(r"""["']([A-Za-z0-9_]+)["']\s*:""")
    offenders = []
    for p in editors:
        with open(p, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for m in pat.finditer(line):
                    key = m.group(1).lower()
                    if key in reserved:
                        offenders.append("%s:%d: %r 用作字典键"
                                         % (os.path.basename(p), ln, m.group(1)))
    assert not offenders, "导出脚本使用了保留键名: %s" % offenders


# ---------------------------------------------------------------------------
# 6) 共享 GT
# ---------------------------------------------------------------------------

def _make_gt_fixture(root, *, h=8, w=8, n=N_IMAGES, seed=5,
                     mutate_label=None):
    """造 <root>/ddr/labels/val/*.png（外加 images/val 以便路径检查）+ 共享 GT。"""
    import cv2

    data_root = os.path.join(root, "ddr")
    for sub in ("images/val", "labels/val"):
        os.makedirs(os.path.join(data_root, sub), exist_ok=True)
    rs = np.random.RandomState(seed)
    stems = []
    for i in range(n):
        stem = "SYN%03d" % i
        stems.append(stem)
        lab = np.zeros((h, w), np.uint8)
        # 每类都制造 GT 正样本（分析端 union==0 是 fail closed）
        lab[0, 0] = 1
        lab[1, 1] = 2
        lab[2, 2] = 3
        lab[3, 3] = 4
        if mutate_label is not None:
            lab = mutate_label(i, lab)
        cv2.imwrite(os.path.join(data_root, "labels/val", stem + ".png"), lab)
        cv2.imwrite(os.path.join(data_root, "images/val", stem + ".jpg"),
                    (rs.rand(h, w, 3) * 255).astype(np.uint8))
    gt_dir = os.path.join(root, "out", "_shared", "gt")

    def reader(p):
        return cv2.imread(p, cv2.IMREAD_UNCHANGED)

    man = C.write_or_verify_gt(gt_dir, data_root=data_root, split="val", stems=stems,
                               label_reader=reader, image_hw={s: (h, w) for s in stems})
    return data_root, gt_dir, sorted(man["stems"])


def test_gt_idempotent_and_tamper_detection():
    import cv2

    root = os.path.join(TMP, "gt")
    if os.path.isdir(root):
        shutil.rmtree(root)
    data_root, gt_dir, stems = _make_gt_fixture(root)
    man = C.load_gt_contract(gt_dir)["contract"]
    assert man["n_images"] == N_IMAGES and man["stems"] == stems
    # 幂等：再调一次不会改哈希
    again = C.write_or_verify_gt(
        gt_dir, data_root=data_root, split="val", stems=stems,
        label_reader=lambda p: cv2.imread(p, cv2.IMREAD_UNCHANGED),
        image_hw={s: (8, 8) for s in stems})
    assert again["gt_contract_sha256"] == man["gt_contract_sha256"]
    # 篡改某个 GT npz → load_gt_contract 自校验通过但取图时哈希不符
    p = os.path.join(gt_dir, C.MASK_FILENAME % stems[3])
    raw = bytearray(open(p, "rb").read())
    raw[-1] ^= 0xFF
    open(p, "wb").write(bytes(raw))
    assert C.sha256_file(p) != man["per_image"][stems[3]]["file_sha256"]
    # 篡改标签后重建必须被拒绝（已存在但不一致 → 不覆盖也不复用）
    try:
        C.write_or_verify_gt(
            gt_dir, data_root=data_root, split="val", stems=stems,
            label_reader=lambda q: cv2.imread(q, cv2.IMREAD_UNCHANGED),
            image_hw={s: (8, 8) for s in stems})
        raise AssertionError("GT 不一致时必须拒绝而不是重写")
    except C.ContractError:
        pass
    # 图像数与尺寸不一致要拒绝
    try:
        C.write_or_verify_gt(os.path.join(root, "out2", "_shared", "gt"),
                             data_root=data_root, split="val", stems=stems[:10],
                             label_reader=lambda q: cv2.imread(q, cv2.IMREAD_UNCHANGED),
                             image_hw={s: (8, 8) for s in stems[:10]})
        raise AssertionError("图像数 != 76 必须拒绝")
    except C.ContractError:
        pass
    # 标签尺寸与图像尺寸不符要拒绝
    try:
        C.write_or_verify_gt(os.path.join(root, "out3", "_shared", "gt"),
                             data_root=data_root, split="val", stems=stems,
                             label_reader=lambda q: cv2.imread(q, cv2.IMREAD_UNCHANGED),
                             image_hw={s: (9, 9) for s in stems})
        raise AssertionError("标签尺寸 != 图像尺寸必须拒绝")
    except C.ContractError:
        pass


# ---------------------------------------------------------------------------
# 7) 端到端：合成 76 图 + 3 模型 → 分析器
# ---------------------------------------------------------------------------

def test_end_to_end_complementarity_synthetic():
    root = os.path.join(TMP, "e2e")
    if os.path.isdir(root):
        shutil.rmtree(root)
    h, w = 16, 24
    data_root, gt_dir, stems = _make_gt_fixture(root, h=h, w=w)
    out_root = os.path.join(root, "out")

    # 通道顺序固定为 ("EX","HE","SE","MA") → 0/1/2/3。
    # 造出一个"每类各有所长"的可判定局面：
    #   champion: EX/HE 全漏，MA 漏 1 像素，SE 正确 → 只有 SE 强
    #   BASE_0  : EX/HE/SE 正确（补上 champion 漏的 EX/HE），MA 也漏 1 像素
    #   U0      : MA 正确（该类最优），SE 漏 → 只在 MA 上超过 champion
    def champ_mutate(i, bits):
        b = bits.copy()
        b[0, :, :] = False          # champion 全漏 EX
        b[1, :, :] = False          # champion 全漏 HE
        b[3, 3, 3] = False          # champion 漏掉唯一的 MA 像素
        return b

    def expert_mutate(i, bits):
        b = bits.copy()
        b[2, :, :] = False          # BASE_0 漏 SE
        b[3, 3, 3] = False          # BASE_0 同样漏 MA（保证 U0 在 MA 胜出）
        return b

    def u0_mutate(i, bits):
        b = bits.copy()
        b[2, :, :] = False          # U0 相对 champion 只在 MA 更好
        return b

    _tiny_model_dir(out_root, "champion", "hrplus", stems, gt_dir, mutate=champ_mutate, h=h, w=w)
    _tiny_model_dir(out_root, "BASE_0", "hrplus", stems, gt_dir, mutate=expert_mutate, h=h, w=w)
    _tiny_model_dir(out_root, "U0", "fullmodel", stems, gt_dir, mutate=u0_mutate, h=h, w=w)

    out_dir = os.path.join(out_root, "analysis")
    cmd = [sys.executable, os.path.join(_ANALYSIS, "analyze_expert_complementarity.py"),
           "--out-root", out_root, "--out-dir", out_dir,
           "--models", "champion", "BASE_0", "U0", "--champion", "champion"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, "分析器必须 exit 0\nSTDOUT:%s\nSTDERR:%s" % (r.stdout[-2000:],
                                                                          r.stderr[-2000:])
    with open(os.path.join(out_dir, "expert_complementarity.json"), encoding="utf-8") as f:
        pay = json.load(f)
    # champion: EX 漏 1 像素；BASE_0: EX 全预测 → OR 的 EX 必须恰好等于 GT（IoU=1）
    ex = pay["pairwise"]["BASE_0"]["per_class"]["EX"]
    assert ex["or_iou"] == 1.0, ex
    assert ex["champion_only_tp"] == 0 and ex["candidate_only_tp"] > 0, ex
    # 恒等式：候选 own tp = common + cand_only
    d = pay["model_metrics"]["BASE_0"]["per_class_detail"]["EX"]
    assert d["tp"] == ex["common_tp"] + ex["candidate_only_tp"]
    # HE：champion 全漏 → 候选(也是 GT) 独有 TP = GT 正样本数
    he = pay["pairwise"]["BASE_0"]["per_class"]["HE"]
    assert he["candidate_only_tp"] == pay["gt_global_positive_pixels"]["HE"], he
    assert he["or_iou"] == 1.0
    # U0 只改 MA
    for cname in ("EX", "HE", "SE"):
        u = pay["pairwise"]["U0"]["per_class"][cname]
        assert u["common_tp"] + u["champion_only_tp"] + u["candidate_only_tp"] == \
            pay["gt_global_positive_pixels"][cname], (cname, u)
    assert pay["direct_four_class_combination"]["picks"]["MA"]["model"] in ("U0",)
    assert pay["gt"]["gt_semantic_fingerprint"] == \
        C.load_gt_contract(gt_dir)["contract"]["gt_semantic_fingerprint"]
    assert os.path.isfile(os.path.join(out_dir, "expert_complementarity_per_class.csv"))
    assert os.path.isfile(os.path.join(out_dir, "expert_complementarity_per_image.csv"))
    assert os.path.isfile(os.path.join(out_dir, "expert_complementarity_area_buckets.csv"))
    assert os.path.isfile(os.path.join(out_dir, "EXPERT_COMPLEMENTARITY.md"))

    # 篡改一个掩膜 → 分析必须拒绝（exit 2）
    p = os.path.join(out_root, "hrplus", "champion", C.MASK_FILENAME % stems[0])
    raw = bytearray(open(p, "rb").read())
    raw[-1] ^= 0xFF
    open(p, "wb").write(bytes(raw))
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    assert r2.returncode == 2, "篡改掩膜后必须 exit 2，实际 %s\n%s" % (
        r2.returncode, r2.stdout[-500:])

    # test 路径 / test split 必须拒绝
    r3 = subprocess.run([sys.executable,
                         os.path.join(_ANALYSIS, "analyze_expert_complementarity.py"),
                         "--out-root", os.path.join(out_root, "test")],
                        capture_output=True, text=True)
    assert r3.returncode == 2, "路径含 test 必须 exit 2，实际 %s" % r3.returncode


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _all_tests():
    return [(k, v) for k, v in sorted(globals().items())
            if k.startswith("test_") and callable(v)]


def main():
    if os.path.isdir(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP, exist_ok=True)
    fails = []
    for name, fn in _all_tests():
        try:
            fn()
            print("[PASS] %s" % name)
        except Exception as e:                       # noqa: BLE001
            import traceback
            print("[FAIL] %s: %s" % (name, e))
            traceback.print_exc()
            fails.append(name)
    print("\n%d/%d passed" % (len(_all_tests()) - len(fails), len(_all_tests())))
    if fails:
        print("FAILED: %s" % fails)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
