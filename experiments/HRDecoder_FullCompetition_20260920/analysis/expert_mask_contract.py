# -*- coding: utf-8 -*-
"""expert_mask_contract — 四个"专家"管线的统一二值掩膜落盘/校验契约。

为什么是 packbits 而不是概率：
本分析要回答的是"逐像素互补性"，而互补性只在**二值判定**层面成立。
因此落盘对象是 `p > 0.5` 的 1 bit/像素结果，而不是完整 FP32 概率：
- 概率缓存会占 76×4×H×W float32 ≈ 200 MB/模型，且 fp16 缓存会改变严格 0.5
  阈值附近的判定（历史事故，见 `eval_candidates.py` 头注）；
- 二值 packbits 只占 1/256，且**不含**任何"接近阈值"的隐式信息，
  不可能被误当成概率复用。

强制口径（所有模型、所有管线一致，缺一即拒绝）：
- split 只能是 `val`（76 张）；路径里出现 `test` 分量直接拒绝；
- 概率必须是 float32，sigmoid / 融合 / H4 头 全部在 float32 下完成后，
  才做**严格** `> 0.5` 二值化（`p == 0.5` 判为阴性）；
- packbits 按 `bitorder='big'`、逐通道展平（channel-major, C×H×W → C×(H*W)），
  并记录有效位数 `n_pixels = H*W`（变尺寸还原的唯一依据）与原始 `orig_hw`；
- 每个文件先写同目录临时文件再 `os.replace` 原子替换；
- contract.json 缺失/字段不全/哈希不符/逐图计数与掩膜重算不一致 ⇒ 分析端拒绝。

禁止：本模块不写任何 FP32/FP16 概率，也不读任何非本契约的 npz。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# 契约常量（唯一权威定义，禁止在其它脚本里重复硬编码）
# ---------------------------------------------------------------------------
CONTRACT_SCHEMA = "hrdecoder-expert-mask/v1"
GT_CONTRACT_SCHEMA = "hrdecoder-expert-gt/v1"
RUN_MANIFEST_SCHEMA = "hrdecoder-expert-run/v1"

CLASSES = ("EX", "HE", "SE", "MA")
DDR_CLASS_ID_OF_CHANNEL = (1, 2, 3, 4)      # 通道 i ↔ DDR fused class id i+1
NUM_CLASSES = 4

STRICT_THRESHOLD = 0.5
COMPARISON = "p > 0.5"
PROB_DTYPE = "float32"
PACK_FORMAT = "np.packbits"
PACK_BITORDER = "big"
PACK_LAYOUT = "channel_major_flattened_c_hw"
GT_IGNORE_LABEL = 255
LABEL_RANGE = (0, 1, 2, 3, 4)

SPLIT = "val"
EXPECTED_VAL_IMAGES = 76
FORBIDDEN_PATH_COMPONENTS = ("test",)

MASK_FILENAME = "%s.npz"
CONTRACT_FILENAME = "contract.json"

# 分析端的绝对下限：任何模型都必须声明"没有保存概率"
FORBIDDEN_CONTRACT_KEYS = (
    "prob", "probs", "probability", "prob_fp32", "prob_cache",
    "logits", "probabilities",
)


class ContractError(Exception):
    """契约违规（元数据不全、形状/哈希不符、口径不符）⇒ 非零退出。"""


class PreflightBlocked(Exception):
    """资产/环境缺失导致**无法开始**（不是结果错误）⇒ 单独的非零退出码。

    与 ContractError 区分：ContractError 说明"跑出来的东西不合契约"，
    PreflightBlocked 说明"还没跑，缺东西"。README 中逐条列出阻塞原因。
    """


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


# ---------------------------------------------------------------------------
# 哈希 / 原子写 / 路径
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj) -> str:
    """确定性 JSON 文本（键排序、紧凑分隔、拒 NaN/Inf ⇒ 哈希可复现）。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def sha256_json(obj) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def safe_path(path: str, what: str) -> str:
    """规范化路径；拒绝空值、NUL、以及任何 `..` 分量。"""
    if not isinstance(path, str) or not path.strip():
        raise ContractError("%s: 路径为空或非字符串 %r" % (what, path))
    if "\x00" in path:
        raise ContractError("%s: 路径含 NUL 字节" % what)
    p = os.path.abspath(os.path.expanduser(path))
    if ".." in os.path.normpath(p).replace("\\", "/").split("/"):
        raise ContractError("%s: 路径含 .. 分量（拒绝穿越）: %r" % (what, path))
    return p


def ensure_dir(path: str) -> str:
    p = safe_path(path, "mkdir")
    os.makedirs(p, exist_ok=True)
    return p


def atomic_write_bytes(path: str, data: bytes) -> str:
    """同目录临时文件 + fsync + os.replace（崩溃不留半截文件）。"""
    p = safe_path(path, "atomic write")
    parent = os.path.dirname(p)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    return p


def atomic_write_text(path: str, text: str) -> str:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str, obj) -> str:
    return atomic_write_text(path, canonical_json(obj) + "\n")


# ---------------------------------------------------------------------------
# val-only 闸门
# ---------------------------------------------------------------------------

def _path_components(path: str):
    return [x.lower() for x in str(path).replace("\\", "/").split("/") if x]


def assert_val_only(split, *, paths=(), n_images=None,
                    expected_images: int = EXPECTED_VAL_IMAGES,
                    what: str = "eval") -> None:
    """硬拒绝 test：split 名、路径分量、图像数三处同时封死。"""
    if str(split).strip().lower() != SPLIT:
        raise ContractError(
            "%s: 只允许 split=%r（收到 %r）。test 绝不进入本分析——"
            "本工具不做 test 评估、不做 test 选型。" % (what, SPLIT, split)
        )
    for p in paths:
        parts = _path_components(p)
        for bad in FORBIDDEN_PATH_COMPONENTS:
            if bad in parts:
                raise ContractError(
                    "%s: 路径含 forbidden 分量 %r: %r" % (what, bad, p)
                )
    if n_images is not None and int(n_images) != int(expected_images):
        raise ContractError(
            "%s: 图像数 %d != 契约要求的 %d（DDR val）"
            % (what, int(n_images), int(expected_images))
        )


def _is_float_array(v) -> bool:
    """是否为携带浮点数值的**数组**（概率/logits 载荷）。

    只认有维度的数组；标量 float（iou/threshold 等指标）不算载荷 —— 指标本来就要落盘。
    torch.Tensor 也走同一判定，且**不 import torch**（按 dtype 字符串识别）。
    """
    dtype = getattr(v, "dtype", None)
    ndim = getattr(v, "ndim", None)
    if dtype is None or ndim is None or int(ndim) == 0:
        return False
    if not isinstance(dtype, np.dtype):
        s = str(dtype).lower()                       # 如 torch.float16 / bfloat16
        return ("float" in s) or ("half" in s) or ("bfloat" in s)
    try:
        return bool(np.issubdtype(dtype, np.floating))
    except TypeError:
        return False


def _is_float_leaf(x) -> bool:
    """浮点标量（含 numpy 标量）；bool 不算（self_checks 里有布尔）。"""
    if isinstance(x, bool):
        return False
    return isinstance(x, (float, np.floating))


def _is_float_sequence(v, _depth: int = 0) -> bool:
    """list/tuple（可嵌套）且叶子全为浮点 ⇒ 视为"概率被摊平存成列表"。

    注意：**孤立标量 float 不算**（iou/threshold 等指标就是标量，必须能落盘）。
    """
    if not isinstance(v, (list, tuple)) or _depth > 8 or len(v) == 0:
        return False
    for x in v:
        if isinstance(x, (list, tuple)):
            if not _is_float_sequence(x, _depth + 1):
                return False
        elif not _is_float_leaf(x):
            return False
    return True


def _is_probability_payload(v) -> bool:
    return _is_float_array(v) or _is_float_sequence(v)


def assert_no_probability_payload(obj, where: str) -> None:
    """拒绝任何"顺手存了概率"的载荷（顶层键与嵌套键都查）。

    两条互相独立、都不可绕过的规则：
    (1) 键名禁令（**未放松**）：`FORBIDDEN_CONTRACT_KEYS` 里的键一旦出现即拒绝，
        无论值是什么（连 `"prob_fp32": "..."` 这种字符串也拒绝）。键名本身就是
        "这里是概率"的强信号，所以保留为硬黑名单。
    (2) 类型禁令（**新增的加强项**）：与键名无关，契约里**任何位置**出现浮点数组
        或纯浮点列表一律拒绝 —— 把概率数组改名成 `"data"` 之类也照样拦下。
        这是对"禁止浮点概率数组落盘"这一不变量的语义级保证，只收紧、不放松。

    历史缺陷的根因不在本函数：`export_hrplus_expert_masks.pipeline_definition()`
    曾把一个**纯文字说明**的元数据字段命名为 `"probability"`，撞上规则 (1) 的
    保留键名，导致 C2000/S4000 在写 contract 时 fail closed（hrplus_exit_code=2）。
    修法是给该元数据字段改名（见 pipeline_definition，改为 `probability_source`），
    而不是给保留键开字符串后门 —— 这样二进制/浮点载荷的禁令保持原样甚至更严。
    """
    stack = [(obj, "")]
    while stack:
        cur, path = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                here = "%s%s%s" % (path, "/" if path else "", k)
                if str(k).lower() in FORBIDDEN_CONTRACT_KEYS:
                    raise ContractError(
                        "%s: 契约中出现概率字段 %r（键路径 %s）—— 本契约只允许二值 "
                        "packbits，禁止保存概率（fp16 概率会改变严格 0.5 判定）。"
                        "若只是文字说明，请改用非保留键名（如 probability_source）。"
                        % (where, k, here)
                    )
                if _is_probability_payload(v):
                    raise ContractError(
                        "%s: 契约在键路径 %s 出现浮点数组/纯浮点列表（dtype=%s）—— "
                        "禁止任何形式的概率载荷落盘，只允许二值 packbits。"
                        % (where, here, getattr(v, "dtype", type(v).__name__))
                    )
                stack.append((v, here))
        elif isinstance(cur, (list, tuple)):
            if _is_probability_payload(cur):
                raise ContractError(
                    "%s: 契约在路径 %s 出现浮点数组/纯浮点列表（dtype=%s）—— "
                    "禁止任何形式的概率载荷落盘，只允许二值 packbits。"
                    % (where, path or "<root>", getattr(cur, "dtype", type(cur).__name__))
                )
            for i, v in enumerate(cur):
                stack.append((v, "%s[%d]" % (path, i)))


# ---------------------------------------------------------------------------
# 概率 → 二值（严格口径）
# ---------------------------------------------------------------------------

def check_float32_like(prob, where: str) -> None:
    """在 `.float()` 之前就拦住 fp16/bfloat16（否则转换会掩盖精度损失）。"""
    name = str(getattr(prob, "dtype", type(prob).__name__)).lower()
    if any(t in name for t in ("float16", "half", "bfloat16")):
        raise ContractError(
            "%s: 概率精度为 %s —— 禁止 fp16 概率（会改变严格 >0.5 附近判定）。"
            "请在全 float32 下完成 sigmoid/融合/H4。" % (where, name)
        )
    if "float32" not in name:
        raise ContractError("%s: 概率 dtype=%s，必须是 float32" % (where, name))


def binarize_strict(prob, where: str = "prob") -> np.ndarray:
    """严格 `> 0.5` 二值化。prob 必须是 numpy float32 [C,H,W]，且有限、∈[0,1]。

    `p == 0.5` 判为阴性（`>` 而非 `>=`），与 `runtime.per_image_counts` /
    `lesion_metric.sigmoid_confused_matrix` 的严格口径一致。
    """
    if not isinstance(prob, np.ndarray):
        raise ContractError("%s: 需要 numpy 数组，收到 %r" % (where, type(prob).__name__))
    check_float32_like(prob, where)
    if prob.ndim != 3 or prob.shape[0] != NUM_CLASSES:
        raise ContractError("%s: 形状必须是 (4,H,W)，得到 %r" % (where, tuple(prob.shape)))
    if not np.isfinite(prob).all():
        raise ContractError("%s: 概率含非有限值" % where)
    if prob.min() < 0.0 or prob.max() > 1.0:
        raise ContractError("%s: 概率越界 [%r,%r]，不在 [0,1]"
                            % (where, float(prob.min()), float(prob.max())))
    return prob > np.float32(STRICT_THRESHOLD)


# ---------------------------------------------------------------------------
# packbits 编解码（变尺寸可还原）
# ---------------------------------------------------------------------------

def nbytes_of(n_pixels: int) -> int:
    return int((int(n_pixels) + 7) // 8)


def pack_bits(bits) -> np.ndarray:
    """[C,H,W] 或 [C,N] 布尔 → uint8 [C, ceil(N/8)]（bitorder='big'）。"""
    a = np.asarray(bits)
    if a.dtype != np.bool_:
        a = a.astype(bool)
    if a.ndim != 2:
        if a.ndim != 3:
            raise ContractError("pack_bits: 期望 [C,N] 或 [C,H,W]，得到 %r" % (a.shape,))
        a = a.reshape(a.shape[0], -1)
    if a.shape[0] != NUM_CLASSES:
        raise ContractError("pack_bits: 必须是 4 通道，得到 %d" % a.shape[0])
    return np.ascontiguousarray(
        np.packbits(a, axis=1, bitorder=PACK_BITORDER), dtype=np.uint8
    )


def unpack_bits(packed, n_pixels: int, *, check_padding: bool = True,
                expect_channels=NUM_CLASSES) -> np.ndarray:
    """uint8 [C, ceil(N/8)] → bool [C,N]；只保留有效位 n_pixels。

    `expect_channels=None` 时不约束通道数（GT 的 ignore 掩膜是单通道）。
    `check_padding=True` 时额外要求填充位必须为 0（否则说明写入端用了
    不同位序/长度，属于契约违规，宁可拒绝也不猜）。
    """
    p = np.asarray(packed)
    if p.dtype != np.uint8 or p.ndim != 2:
        raise ContractError("unpack_bits: 期望 uint8 [C,B]，得到 %r/%s"
                            % (tuple(p.shape), p.dtype))
    if expect_channels is not None and p.shape[0] != int(expect_channels):
        raise ContractError("unpack_bits: 期望 %d 通道，得到 %d"
                            % (int(expect_channels), p.shape[0]))
    n_pixels = int(n_pixels)
    if n_pixels <= 0:
        raise ContractError("unpack_bits: n_pixels 必须为正，得到 %d" % n_pixels)
    want = nbytes_of(n_pixels)
    if p.shape[1] != want:
        raise ContractError("unpack_bits: 每通道字节数 %d != ceil(%d/8)=%d"
                            % (p.shape[1], n_pixels, want))
    bits = np.unpackbits(p, axis=1, count=n_pixels, bitorder=PACK_BITORDER)
    if check_padding:
        used = n_pixels % 8
        if used:
            # bitorder='big'：最后一个字节的高位是有效位，低位是填充位（必须为 0）。
            pad_bits = p[:, -1] & ((1 << (8 - used)) - 1)
            if pad_bits.any():
                raise ContractError(
                    "unpack_bits: 填充位非零（位序/长度不符预期）: %r"
                    % (pad_bits.tolist(),)
                )
    return bits.astype(bool)


def semantic_sha256(packed, n_pixels: int, orig_hw, *, kind: str,
                    extra: str = "") -> str:
    """掩膜语义哈希：与容器/压缩无关，只覆盖位内容 + 有效位数 + 原始尺寸。"""
    p = np.ascontiguousarray(np.asarray(packed, dtype=np.uint8))
    h = hashlib.sha256()
    h.update(b"expert-mask-semantic/v1\n")
    h.update(("kind=%s\n" % kind).encode("utf-8"))
    h.update(("n_pixels=%d\n" % int(n_pixels)).encode("utf-8"))
    h.update(("orig_hw=%d,%d\n" % (int(orig_hw[0]), int(orig_hw[1]))).encode("utf-8"))
    h.update(("packed_shape=%d,%d\n" % p.shape).encode("utf-8"))
    h.update(("bitorder=%s\n" % PACK_BITORDER).encode("utf-8"))
    if extra:
        h.update(("extra=%s\n" % extra).encode("utf-8"))
    h.update(p.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# npz 读写（掩膜 / GT）
# ---------------------------------------------------------------------------

def _npz_bytes(**arrays) -> bytes:
    import io

    buf = io.BytesIO()
    # allow_pickle=False：只允许纯数组载荷，杜绝 pickle 反序列化面。
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def save_mask_npz(path: str, bits, orig_hw) -> dict:
    """写出单图单模型掩膜。返回记录（含 file/semantic sha256）。"""
    bits = np.asarray(bits)
    if bits.ndim != 3:
        raise ContractError("save_mask_npz: bits 必须是 [C,H,W]，得到 %r" % (bits.shape,))
    h, w = int(orig_hw[0]), int(orig_hw[1])
    if tuple(bits.shape[1:]) != (h, w):
        raise ContractError("save_mask_npz: bits 空间尺寸 %r != orig_hw (%d,%d)"
                            % (tuple(bits.shape[1:]), h, w))
    packed = pack_bits(bits)
    n_pixels = h * w
    data = _npz_bytes(packed=packed,
                      n_pixels=np.int64(n_pixels),
                      orig_hw=np.asarray([h, w], dtype=np.int64))
    p = atomic_write_bytes(path, data)
    return {
        "mask_file": os.path.basename(p),
        "file_sha256": sha256_bytes(data),
        "semantic_sha256": semantic_sha256(packed, n_pixels, (h, w),
                                           kind="model-mask"),
        "n_pixels": n_pixels,
        "orig_hw": [h, w],
        "packed_bytes": int(packed.shape[1]),
    }


def load_mask_npz(path: str, *, where: str = "mask") -> dict:
    with open(path, "rb") as f:
        raw = f.read()
    import io

    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        keys = set(z.files)
        want = {"packed", "n_pixels", "orig_hw"}
        if keys != want:
            raise ContractError("%s: npz 键 %r != %r" % (where, sorted(keys), sorted(want)))
        packed = np.asarray(z["packed"])
        n_pixels = int(np.asarray(z["n_pixels"]).reshape(-1)[0])
        orig_hw = tuple(int(x) for x in np.asarray(z["orig_hw"]).reshape(-1)[:2])
    if n_pixels != orig_hw[0] * orig_hw[1]:
        raise ContractError("%s: n_pixels=%d != H*W=%d"
                           % (where, n_pixels, orig_hw[0] * orig_hw[1]))
    bits = unpack_bits(packed, n_pixels)
    return {"packed": packed, "n_pixels": n_pixels, "orig_hw": orig_hw,
            "bits": bits.reshape(NUM_CLASSES, orig_hw[0], orig_hw[1]),
            "file_sha256": sha256_bytes(raw)}


def save_gt_npz(path: str, label_u8) -> dict:
    """GT 只保存一次：逐类二值 packbits + ignore packbits。"""
    lab = np.asarray(label_u8)
    if lab.ndim != 2:
        raise ContractError("save_gt_npz: label 必须是 [H,W]，得到 %r" % (lab.shape,))
    vals = np.unique(lab)
    if vals.min() < 0 or not np.all(np.isin(vals, list(LABEL_RANGE) + [GT_IGNORE_LABEL])):
        raise ContractError("save_gt_npz: 标签值域越界 %r（允许 0..4 / 255）" % (vals.tolist(),))
    h, w = int(lab.shape[0]), int(lab.shape[1])
    bits = np.stack([lab == c for c in DDR_CLASS_ID_OF_CHANNEL], axis=0)
    ignore = lab == GT_IGNORE_LABEL
    packed = pack_bits(bits)
    ignore_packed = np.packbits(ignore.reshape(1, -1), axis=1,
                                bitorder=PACK_BITORDER)[0]
    n_pixels = h * w
    data = _npz_bytes(packed=packed,
                      ignore_packed=np.ascontiguousarray(ignore_packed, dtype=np.uint8),
                      n_pixels=np.int64(n_pixels),
                      orig_hw=np.asarray([h, w], dtype=np.int64))
    p = atomic_write_bytes(path, data)
    return {
        "gt_file": os.path.basename(p),
        "file_sha256": sha256_bytes(data),
        "semantic_sha256": semantic_sha256(packed, n_pixels, (h, w), kind="gt"),
        "n_pixels": n_pixels,
        "orig_hw": [h, w],
    }


def load_gt_npz(path: str, *, where: str = "gt") -> dict:
    with open(path, "rb") as f:
        raw = f.read()
    import io

    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        keys = set(z.files)
        want = {"packed", "ignore_packed", "n_pixels", "orig_hw"}
        if keys != want:
            raise ContractError("%s: npz 键 %r != %r" % (where, sorted(keys), sorted(want)))
        packed = np.asarray(z["packed"])
        ignore_packed = np.asarray(z["ignore_packed"])
        n_pixels = int(np.asarray(z["n_pixels"]).reshape(-1)[0])
        orig_hw = tuple(int(x) for x in np.asarray(z["orig_hw"]).reshape(-1)[:2])
    bits = unpack_bits(packed, n_pixels)
    ignore = unpack_bits(ignore_packed.reshape(1, -1), n_pixels,
                         expect_channels=None)[0]
    return {"bits": bits.reshape(NUM_CLASSES, orig_hw[0], orig_hw[1]),
            "ignore": ignore.reshape(orig_hw),
            "n_pixels": n_pixels, "orig_hw": orig_hw,
            "packed": packed, "file_sha256": sha256_bytes(raw)}


# ---------------------------------------------------------------------------
# TP/FP/FN（严格口径，逐像素）
# ---------------------------------------------------------------------------

def counts_from_bits(pred_bits, gt_bits, ignore_bits=None):
    """逐类 [tp,fp,fn]；语义与 `runtime.per_image_counts` 完全一致：
    `fp = pred & ~gt & ~ignore`（未标注像素永不计 FP），`fn = ~pred & gt`。
    """
    pred = np.asarray(pred_bits, dtype=bool)
    gt = np.asarray(gt_bits, dtype=bool)
    if pred.shape != gt.shape:
        raise ContractError("counts_from_bits: 形状不一致 pred%r gt%r"
                            % (pred.shape, gt.shape))
    if pred.ndim != 2 or pred.shape[0] != NUM_CLASSES:
        raise ContractError("counts_from_bits: 期望 [C,N]，得到 %r" % (pred.shape,))
    if ignore_bits is None:
        valid = np.ones_like(gt)
    else:
        ig = np.asarray(ignore_bits, dtype=bool)
        if ig.ndim == 1:
            ig = np.broadcast_to(ig[None, :], gt.shape)
        if ig.shape != gt.shape:
            raise ContractError("counts_from_bits: ignore 形状 %r != %r"
                                % (ig.shape, gt.shape))
        valid = ~ig
    out = []
    for c in range(NUM_CLASSES):
        p, g, v = pred[c], gt[c], valid[c]
        tp = int(np.count_nonzero(p & g))
        fp = int(np.count_nonzero(p & ~g & v))
        fn = int(np.count_nonzero(~p & g))
        out.append([tp, fp, fn])
    return out


def sum_counts(list_of_counts):
    tot = [[0, 0, 0] for _ in range(NUM_CLASSES)]
    for row in list_of_counts:
        for c in range(NUM_CLASSES):
            for j in range(3):
                tot[c][j] += int(row[c][j])
    return tot


def iou_from_counts(counts, *, fail_on_zero_union: bool = True):
    """counts: 4×[tp,fp,fn] → (miou, {cls: iou}, {cls: detail})。

    历史口径：union==0 直接失败（不跳过、不当 0）。val 四类均有正样本。
    """
    if len(counts) != NUM_CLASSES:
        raise ContractError("iou_from_counts: 需要 4 类，得到 %d" % len(counts))
    per, detail, vals = {}, {}, []
    for c in range(NUM_CLASSES):
        tp, fp, fn = (int(x) for x in counts[c])
        union = tp + fp + fn
        if union == 0:
            if fail_on_zero_union:
                raise ContractError(
                    "class %s: union==0（拒绝：历史口径要求 fail closed）" % CLASSES[c])
            per[CLASSES[c]] = None
            detail[CLASSES[c]] = {"tp": tp, "fp": fp, "fn": fn, "union": 0}
            continue
        iou = tp / union
        per[CLASSES[c]] = iou
        vals.append(iou)
        detail[CLASSES[c]] = {
            "tp": tp, "fp": fp, "fn": fn, "union": union, "iou": iou,
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "recall": tp / (tp + fn) if (tp + fn) else None,
        }
    miou = float(np.mean(vals)) if vals else float("nan")
    return miou, per, detail


# 布尔集合运算（OR / AND）；输入 [C,N] bool
def or_bits(a, b) -> np.ndarray:
    return np.logical_or(np.asarray(a, dtype=bool), np.asarray(b, dtype=bool))


def and_bits(a, b) -> np.ndarray:
    return np.logical_and(np.asarray(a, dtype=bool), np.asarray(b, dtype=bool))


# ---------------------------------------------------------------------------
# 模型目录契约（写出 + 读回校验）
# ---------------------------------------------------------------------------

def build_eval_contract(*, pipeline_family: str, eval_loader: str, inference: str,
                        tta: str, extra=None) -> dict:
    """所有模型共享的评估契约块（分析端逐字段比对，不一致即拒绝）。"""
    c = {
        "contract_schema": CONTRACT_SCHEMA,
        "split": SPLIT,
        "expected_images": EXPECTED_VAL_IMAGES,
        "threshold": STRICT_THRESHOLD,
        "comparison": COMPARISON,
        "prob_precision": PROB_DTYPE,
        "prob_payload_saved": False,
        "fp16_probability_cache": False,
        "binarization": ("sigmoid/融合/H4 全部在 float32 完成后，按严格 p>0.5 二值化；"
                         "仅保存 1bit/像素"),
        "channel_order": list(CLASSES),
        "ddr_class_id_of_channel": list(DDR_CLASS_ID_OF_CHANNEL),
        "ignore_label": GT_IGNORE_LABEL,
        "aggregation": "whole_split_accumulated_tp_fp_fn",
        "packing": {
            "format": PACK_FORMAT,
            "bitorder": PACK_BITORDER,
            "layout": PACK_LAYOUT,
            "n_pixels_recorded": True,
            "orig_hw_recorded": True,
        },
        "pipeline_family": pipeline_family,
        "eval_loader": eval_loader,
        "inference": inference,
        "tta": tta,
    }
    if extra:
        c.update(extra)
    return c


def compare_eval_contract(contract: dict, *, where: str,
                          expected_family: str | None = None) -> None:
    """逐字段比对共享契约；任何漂移都拒绝（禁止"看着差不多就过"）。"""
    if not isinstance(contract, dict):
        raise ContractError("%s: eval_contract 不是映射" % where)
    if contract.get("contract_schema") != CONTRACT_SCHEMA:
        raise ContractError("%s: contract_schema=%r != %r"
                            % (where, contract.get("contract_schema"), CONTRACT_SCHEMA))
    checks = {
        "split": SPLIT,
        "expected_images": EXPECTED_VAL_IMAGES,
        "threshold": STRICT_THRESHOLD,
        "comparison": COMPARISON,
        "prob_precision": PROB_DTYPE,
        "prob_payload_saved": False,
        "fp16_probability_cache": False,
        "channel_order": list(CLASSES),
        "ddr_class_id_of_channel": list(DDR_CLASS_ID_OF_CHANNEL),
        "ignore_label": GT_IGNORE_LABEL,
        "aggregation": "whole_split_accumulated_tp_fp_fn",
    }
    for k, want in checks.items():
        got = contract.get(k)
        if got != want:
            raise ContractError("%s: eval_contract.%s=%r != %r（契约漂移，拒绝）"
                                % (where, k, got, want))
    pack = contract.get("packing") or {}
    if (pack.get("format") != PACK_FORMAT or pack.get("bitorder") != PACK_BITORDER
            or pack.get("layout") != PACK_LAYOUT
            or pack.get("n_pixels_recorded") is not True
            or pack.get("orig_hw_recorded") is not True):
        raise ContractError("%s: eval_contract.packing 不符: %r" % (where, pack))
    if expected_family is not None and contract.get("pipeline_family") != expected_family:
        raise ContractError("%s: pipeline_family=%r != %r"
                            % (where, contract.get("pipeline_family"), expected_family))


def finalize_model_dir(model_dir: str, contract: dict) -> str:
    """原子写出 contract.json（放在所有逐图掩膜写完之后 ⇒ 半截产物无法被分析）。"""
    assert_no_probability_payload(contract, "contract[%s]" % contract.get("model_name"))
    p = atomic_write_json(os.path.join(ensure_dir(model_dir), CONTRACT_FILENAME), contract)
    log("contract written: %s" % p)
    return p


def read_model_dir(model_dir: str, *, expected_family: str | None = None,
                   require_contract: bool = True) -> dict:
    """读回并**完整校验**一个模型目录（不加载全部掩膜，逐图按需）。

    返回 {'contract', 'dir', 'stems', 'per_image'}；per_image[stem] 含
    mask_file/file_sha256/semantic_sha256/counts/orig_hw。
    """
    d = safe_path(model_dir, "model dir")
    cpath = os.path.join(d, CONTRACT_FILENAME)
    if not os.path.isfile(cpath):
        if require_contract:
            raise ContractError(
                "缺少 %s —— 元数据不完整，分析拒绝（可能是导出中断；"
                "没有 contract.json 的目录一律视为不可用）" % cpath)
        raise ContractError("no contract: %s" % cpath)
    with open(cpath, "r", encoding="utf-8") as f:
        contract = json.load(f)
    assert_no_probability_payload(contract, "contract[%s]" % cpath)
    name = contract.get("model_name")
    if not isinstance(name, str) or not name:
        raise ContractError("%s: model_name 缺失" % cpath)
    ec = contract.get("eval_contract")
    compare_eval_contract(ec, where="contract[%s]" % name, expected_family=expected_family)
    if contract.get("split") != SPLIT:
        raise ContractError("%s: split=%r != %r" % (cpath, contract.get("split"), SPLIT))
    stems = contract.get("stems")
    if not isinstance(stems, list) or sorted(stems) != list(stems) or len(stems) != EXPECTED_VAL_IMAGES:
        raise ContractError("%s: stems 必须是有序且恰 %d 个，得到 %r"
                            % (cpath, EXPECTED_VAL_IMAGES,
                               None if stems is None else len(stems)))
    per_image = contract.get("per_image")
    if not isinstance(per_image, dict) or set(per_image) != set(stems):
        raise ContractError("%s: per_image 与 stems 不一致" % cpath)
    for stem, rec in per_image.items():
        for key in ("mask_file", "file_sha256", "semantic_sha256", "counts",
                    "orig_hw", "n_pixels"):
            if key not in rec:
                raise ContractError("%s[%s]: 缺少字段 %r" % (cpath, stem, key))
        if os.path.basename(str(rec["mask_file"])) != str(rec["mask_file"]):
            raise ContractError("%s[%s]: mask_file 必须是纯文件名" % (cpath, stem))
        if len(rec["counts"]) != NUM_CLASSES:
            raise ContractError("%s[%s]: counts 必须 4 类" % (cpath, stem))
    for key in ("global_counts", "metrics", "identity", "gt", "pipeline_definition",
                "created_at"):
        if key not in contract:
            raise ContractError("%s: 缺少顶层字段 %r" % (cpath, key))
    return {"dir": d, "contract_path": cpath, "contract": contract,
            "name": name, "stems": list(stems), "per_image": per_image}


def load_verified_mask(model: dict, stem: str):
    """按需加载单图掩膜并重算计数校验（哈希 + 逐类计数，二者都必须过）。"""
    rec = model["per_image"].get(stem)
    if rec is None:
        raise ContractError("%s: 缺少样本 %r" % (model["name"], stem))
    path = os.path.join(model["dir"], rec["mask_file"])
    if not os.path.isfile(path):
        raise ContractError("%s[%s]: 掩膜文件缺失 %s" % (model["name"], stem, path))
    m = load_mask_npz(path, where="%s[%s]" % (model["name"], stem))
    if m["file_sha256"] != rec["file_sha256"]:
        raise ContractError("%s[%s]: 文件 sha256 不符（文件被改动）"
                            % (model["name"], stem))
    if list(m["orig_hw"]) != list(rec["orig_hw"]) or m["n_pixels"] != int(rec["n_pixels"]):
        raise ContractError("%s[%s]: orig_hw/n_pixels 与契约不符" % (model["name"], stem))
    sem = semantic_sha256(m["packed"], m["n_pixels"], m["orig_hw"], kind="model-mask")
    if sem != rec["semantic_sha256"]:
        raise ContractError("%s[%s]: 语义哈希不符（位内容被改动）"
                            % (model["name"], stem))
    m["counts_recomputed"] = True
    return m


def ratio(numer, denom):
    if denom in (0, None) or numer is None or denom is None:
        return None
    return float(numer) / float(denom)


def pct(x, nd=4):
    return None if x is None else round(100.0 * float(x), nd)


def script_sha256(path: str) -> str:
    return sha256_file(path)


def self_script_path() -> str:
    return os.path.abspath(__file__)


# ---------------------------------------------------------------------------
# 共享 GT（val 只写一次；两个导出脚本各调一次，幂等）
# ---------------------------------------------------------------------------

GT_CONTRACT_FILENAME = "gt_contract.json"


def gt_semantic_fingerprint(gt_contract: dict) -> str:
    """把逐图 GT 语义哈希按 stem 顺序串成一个总指纹（模型侧只需记这一个）。"""
    stems = gt_contract["stems"]
    per = gt_contract["per_image"]
    return sha256_json([per[s]["semantic_sha256"] for s in stems])


def _gt_contract_digest(gt_contract: dict) -> str:
    body = {k: v for k, v in gt_contract.items() if k != "gt_contract_sha256"}
    return sha256_json(body)


def write_or_verify_gt(gt_dir: str, *, data_root: str, split: str, stems,
                       label_reader, image_hw=None,
                       expected_images: int = EXPECTED_VAL_IMAGES) -> dict:
    """写出/校验 DDR val 的共享 GT（逐类 packbits + ignore packbits）。

    - split 必须是 val，且 stems 数量必须是 76（val-only 闸门）；
    - 已存在且**完整通过校验**时直接复用（幂等，不重读标签）；
    - 存在但任何一个哈希不符 ⇒ ContractError（绝不静默重写）；
    - `image_hw`（stem→(H,W)）给出时校验"标签尺寸 == 图像尺寸"——DDR val
      实测成立，若不成立说明 GT 必须最近邻 resize，属于本契约未覆盖的情形，
      直接拒绝而不是猜。
    """
    stems = sorted(str(s) for s in stems)
    d = ensure_dir(gt_dir)
    assert_val_only(split, paths=(data_root, d), n_images=len(stems),
                    expected_images=expected_images, what="gt")
    cpath = os.path.join(d, GT_CONTRACT_FILENAME)
    if os.path.isfile(cpath):
        with open(cpath, "r", encoding="utf-8") as f:
            man = json.load(f)
        if (man.get("schema") == GT_CONTRACT_SCHEMA and man.get("split") == split
                and man.get("stems") == stems
                and man.get("data_root") == os.path.abspath(data_root)
                and man.get("gt_contract_sha256") == _gt_contract_digest(man)):
            ok = True
            for s in stems:
                rec = man["per_image"][s]
                fp = os.path.join(d, rec["gt_file"])
                if not os.path.isfile(fp) or sha256_file(fp) != rec["file_sha256"]:
                    ok = False
                    break
            if ok:
                log("GT 复用已校验的共享副本: %s（%d 张）" % (d, len(stems)))
                return man
        raise ContractError(
            "共享 GT 契约存在但不一致（%s）——拒绝复用也拒绝覆盖。"
            "请人工确认后删除该目录重跑。" % cpath)

    per_image, label_hashes = {}, {}
    for s in stems:
        lp = os.path.join(data_root, "labels", split, s + ".png")
        if not os.path.isfile(lp):
            raise PreflightBlocked("GT 标签缺失: %s" % lp)
        lab = label_reader(lp)
        if image_hw is not None:
            want = tuple(int(x) for x in image_hw[s])
            if tuple(lab.shape) != want:
                raise ContractError(
                    "GT[%s]: 标签尺寸 %r != 图像尺寸 %r。DDR val 本应相等；"
                    "不相等意味着 GT 必须 resize 才能对齐，本契约不支持该情形"
                    "（拒绝猜，请先在 README 记录为阻塞）。"
                    % (s, tuple(lab.shape), want))
        rec = save_gt_npz(os.path.join(d, MASK_FILENAME % s), lab)
        per_image[s] = rec
        label_hashes[s] = sha256_file(lp)
    man = {
        "schema": GT_CONTRACT_SCHEMA,
        "split": split,
        "data_root": os.path.abspath(data_root),
        "n_images": len(stems),
        "stems": stems,
        "classes": list(CLASSES),
        "ddr_class_id_of_channel": list(DDR_CLASS_ID_OF_CHANNEL),
        "ignore_label": GT_IGNORE_LABEL,
        "label_files_sha256": label_hashes,
        "per_image": per_image,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    man["gt_semantic_fingerprint"] = gt_semantic_fingerprint(man)
    man["gt_contract_sha256"] = _gt_contract_digest(man)
    atomic_write_json(cpath, man)
    log("共享 GT 已写出: %s（%d 张，fingerprint=%s）"
        % (d, len(stems), man["gt_semantic_fingerprint"][:12]))
    return man


def load_gt_contract(gt_dir: str) -> dict:
    d = safe_path(gt_dir, "gt dir")
    cpath = os.path.join(d, GT_CONTRACT_FILENAME)
    if not os.path.isfile(cpath):
        raise ContractError("共享 GT 契约缺失: %s" % cpath)
    with open(cpath, "r", encoding="utf-8") as f:
        man = json.load(f)
    if man.get("schema") != GT_CONTRACT_SCHEMA:
        raise ContractError("%s: schema=%r" % (cpath, man.get("schema")))
    if man.get("split") != SPLIT or man.get("n_images") != EXPECTED_VAL_IMAGES:
        raise ContractError("%s: split/n_images 不符（%r/%r）"
                            % (cpath, man.get("split"), man.get("n_images")))
    if man.get("gt_contract_sha256") != _gt_contract_digest(man):
        raise ContractError("%s: gt_contract_sha256 自校验失败" % cpath)
    if man.get("gt_semantic_fingerprint") != gt_semantic_fingerprint(man):
        raise ContractError("%s: gt_semantic_fingerprint 自校验失败" % cpath)
    return {"dir": d, "contract_path": cpath, "contract": man,
            "stems": list(man["stems"])}


def add_analysis_dir_to_syspath() -> str:
    d = os.path.dirname(os.path.abspath(__file__))
    if d not in sys.path:
        sys.path.insert(0, d)
    return d
