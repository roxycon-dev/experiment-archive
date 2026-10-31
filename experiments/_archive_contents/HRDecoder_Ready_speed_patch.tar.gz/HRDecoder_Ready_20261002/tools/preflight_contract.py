# -*- coding: utf-8 -*-
"""preflight_contract.py — 开 GPU 前的一条命令契约闸门（fail closed）。

检查四份臂配置（H0/H1/U0/U1）及其引用的外部预训练 manifest。检查项：

A. 配置 schema（轻量内置）：顶层键、臂角色与 pretrain_stage 有无、
   schema_version/status、structure 最小键集、loss.type/metrics.primary；
B. 占位符：字符串值等于或包含 REQUIRED/TBD 即为占位符。默认（正式模式）
   遇到占位符直接 FAIL；--allow-placeholders 仅用于结构/合成测试，跳过项
   在报告中显式列出（绝不静默）；
C. 配对公平性：H0↔H1、U0↔U1 的 equivalence_lock 互指、fields 集合一致、
   必含六个强制字段，且逐字段深度相等（first-difference 定位到子路径）；
D. 预训练规则：H0/U0 的 initialization.external_pretrain_checkpoint 必须
   显式为 null；H1/U1 必须提供 {file, sha256} 且文件存在、哈希一致；
   H1/U1 的 pretrain_stage.external_manifest 必须是同一份文件、仅含
   official_split=train、source 在白名单、lesions 恰为 EX/HE/SE/MA、
   forbidden_splits 必须含 test；
E. 通道序：配置中所有 channel_order 必须严格等于
   contract_common.LESION_CHANNEL_ORDER = [EX, HE, SE, MA]；
F. DDR split：ddr_root 存在；train/val/test 图像-标签 stem 对齐且非空；
   三 split 两两互斥（只读文件名，绝不读 test 像素）；data 与
   ddr_finetune_stage 内禁止出现 test 路径引用（data.ddr_test_split 除外）；
G. 训练/选型不含 test：checkpoint_selection.allowed_splits 恰为 [val] 且
   test_forbidden=true；test_usage 两项必须为 forbidden；
H. 批量守恒：effective_batch == micro_batch * grad_accum（ddr_finetune_stage
   与 pretrain_stage；占位符模式下跳过并记录）；
I. 权重文件：imagenet/encoder/外部预训练 checkpoint/init_checkpoint_output
   的 file 必须存在且 sha256 一致（占位符时跳过并记录）；
J. 跨臂一致性：四臂共享 ddr_root/channel_order/seed/updates/effective_batch，
   且 loss/metrics/checkpoint_selection/test_usage 四节深度相等。

退出码：0=通过；1=契约失败；2=用法/IO 致命错误；3=PyYAML 缺失（显式报错，
绝不静默跳过）。

`--layer readiness`（本机 readiness / GPU calibration 分层）：
只把 **readiness 层** 占位符判 FAIL（结构/契约/数据完整性等本机即可核验的字段），
calibration 层占位符（micro_batch / grad_accum / gradient_checkpointing / 权重文件 /
外部 manifest / 部署路径）记为 CALIBRATION_REQUIRED，无阻塞但必须在正式训练前清零。
该分层**只影响"能否先开一次短时 GPU 校准"**，绝不放宽正式训练闸门：
正式训练走 `train.py --mode formal`，仍要求配置零占位符。

真实资产未到位时正式 preflight 保持 FAIL 是设计行为，不是 bug。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contract_common import (  # noqa: E402
    ALLOWED_OFFICIAL_SPLITS,
    ALLOWED_SOURCES,
    LESION_CHANNEL_ORDER,
    MANIFEST_SCHEMA_VERSION,
    ContractError,
    load_manifest,
    sha256_file,
    write_text_atomic,
)

try:
    import yaml
except Exception as e:  # 显式失败：绝不在缺 PyYAML 时静默跳过
    print(
        f"[preflight_contract] FATAL: PyYAML 未安装（import yaml 失败: {e}）。"
        "preflight 必须解析四份 YAML 配置，缺失时失败而不是跳过。"
        "请先 `python -m pip install pyyaml` 后重试。",
        file=sys.stderr,
    )
    sys.exit(3)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

PLACEHOLDER_TOKENS = ("REQUIRED", "TBD")
MISSING = object()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:  # 层面分类的单一权威定义来自 trainer.gate（本工具不另维护一份表）
    from trainer.gate import placeholder_layer as _placeholder_layer
except Exception:  # pragma: no cover — 缺 trainer 包时退化为"全部按 readiness 处理"
    _placeholder_layer = None

ARM_ORDER = ("H0", "H1", "U0", "U1")
CONTROL_ARMS = ("H0", "U0")
PRETRAIN_ARMS = ("H1", "U1")
PAIRS = (("H0", "H1"), ("U0", "U1"))

CONFIG_SCHEMA_VERSION = "1.0"
ALLOWED_STATUS = ("locked-intent",)

COMMON_TOP_KEYS = (
    "schema_version", "arm_id", "role", "status", "structure", "initialization",
    "data", "ddr_finetune_stage", "loss", "metrics", "checkpoint_selection",
    "test_usage", "equivalence_lock",
)
STRUCTURE_MIN_KEYS = ("family", "paired_group", "definition_ref")
U_STRUCTURE_MIN_KEYS = ("encoder", "decoder", "output_heads", "detail_branch")

# 配对锁必须覆盖的公平性字段（缺任何一个都无法保证配对公平）
MANDATORY_LOCK_FIELDS = (
    "data", "ddr_finetune_stage", "loss", "metrics", "checkpoint_selection", "test_usage",
)
# 四臂必须一致的实验级字段/节（跨结构族公平）
CROSS_ARM_FIELDS = (
    "data.ddr_root", "data.channel_order", "ddr_finetune_stage.seed",
    "ddr_finetune_stage.updates", "ddr_finetune_stage.effective_batch",
)
CROSS_ARM_SECTIONS = ("loss", "metrics", "checkpoint_selection", "test_usage")

TEST_PATH_RE = re.compile(r"(?:^|[/\\])test(?:[/\\]|$)", re.IGNORECASE)
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
IMG_EXTS = (".jpg", ".png")
LABEL_EXTS = (".png",)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def is_placeholder(v) -> bool:
    return isinstance(v, str) and (
        v.strip() in PLACEHOLDER_TOKENS or any(t in v for t in PLACEHOLDER_TOKENS)
    )


def scan_placeholders(obj, path: str = "$"):
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            hits += scan_placeholders(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += scan_placeholders(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        if is_placeholder(obj):
            hits.append((path, obj.strip()))
    return hits


def get_path(obj, dotted: str, default=MISSING):
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def first_difference(a, b, path: str = "$"):
    """返回第一个深度差异的描述；完全相等时返回 None。"""
    if type(a) is not type(b):
        return f"{path}: 类型 {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a.keys()) | set(b.keys())):
            if k not in a:
                return f"{path}.{k}: 仅右侧存在"
            if k not in b:
                return f"{path}.{k}: 仅左侧存在"
            d = first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: 列表长度 {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


def walk_key(obj, key: str, path: str = "$"):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append((f"{path}.{k}", v))
            out += walk_key(v, key, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += walk_key(v, key, f"{path}[{i}]")
    return out


def walk_strings(obj, path: str = "$"):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += walk_strings(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += walk_strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        out.append((path, obj))
    return out


def stems_of(d: str, exts):
    out = set()
    for f in os.listdir(d):
        if os.path.isfile(os.path.join(d, f)) and f.lower().endswith(exts):
            out.add(os.path.splitext(f)[0])
    return out


# ---------------------------------------------------------------------------
# 各项检查
# ---------------------------------------------------------------------------

def check_arm_schema(arm, cfg, errors):
    """A. 轻量配置 schema。"""
    if not isinstance(cfg, dict):
        errors.append(f"{arm}: 配置根必须是映射，当前 {type(cfg).__name__}")
        return False
    for k in COMMON_TOP_KEYS:
        if k not in cfg:
            errors.append(f"{arm}: 缺少顶层键 {k!r}")
    if cfg.get("arm_id") != arm:
        errors.append(f"{arm}: arm_id={cfg.get('arm_id')!r} 与文件预期 {arm!r} 不符")
    if cfg.get("schema_version") != CONFIG_SCHEMA_VERSION:
        errors.append(f"{arm}: schema_version 必须为 {CONFIG_SCHEMA_VERSION!r}，当前 {cfg.get('schema_version')!r}")
    if cfg.get("status") not in ALLOWED_STATUS:
        errors.append(f"{arm}: status={cfg.get('status')!r} 不在允许值 {ALLOWED_STATUS}")
    st = cfg.get("structure")
    if isinstance(st, dict):
        for k in STRUCTURE_MIN_KEYS:
            if k not in st:
                errors.append(f"{arm}: structure 缺少键 {k!r}")
        if arm.startswith("U"):
            for k in U_STRUCTURE_MIN_KEYS:
                if k not in st:
                    errors.append(f"{arm}: structure 缺少键 {k!r}")
    elif st is not MISSING and "structure" in cfg:
        errors.append(f"{arm}: structure 必须是映射")
    if arm in CONTROL_ARMS and "pretrain_stage" in cfg:
        errors.append(f"{arm}: 控制臂不允许存在 pretrain_stage 节")
    if arm in PRETRAIN_ARMS and not isinstance(cfg.get("pretrain_stage"), dict):
        errors.append(f"{arm}: 预训练臂必须提供 pretrain_stage 节")
    loss = cfg.get("loss")
    if isinstance(loss, dict):
        if loss.get("type") != "binary_per_class":
            errors.append(f"{arm}: loss.type 必须为 binary_per_class，当前 {loss.get('type')!r}")
    metrics = cfg.get("metrics")
    if isinstance(metrics, dict):
        if metrics.get("primary") != "mIoU_lesion4":
            errors.append(f"{arm}: metrics.primary 必须为 mIoU_lesion4，当前 {metrics.get('primary')!r}")
    return True


def check_train_test_exclusion(arm, cfg, errors):
    """F(part)/G. 训练数据与 checkpoint 选择不得指向 test。"""
    data = cfg.get("data")
    if isinstance(data, dict):
        for p, v in walk_strings(data, "$.data"):
            leaf = p.rsplit(".", 1)[-1]
            if leaf == "ddr_test_split":
                continue  # ddr_test_split=test 是合法的官方划分声明（仅最终一次评估）
            if TEST_PATH_RE.search(v):
                errors.append(
                    f"{arm}: {p}={v!r} 疑似引用 test 路径/划分——训练与验证数据禁止指向 test"
                )
        for role in ("train", "val"):
            v = data.get(f"ddr_{role}_split")
            if not is_placeholder(v) and v == "test":
                errors.append(f"{arm}: data.ddr_{role}_split 不得为 test")
    cs = cfg.get("checkpoint_selection")
    if isinstance(cs, dict):
        allowed = cs.get("allowed_splits")
        if allowed != ["val"]:
            errors.append(
                f"{arm}: checkpoint_selection.allowed_splits 必须恰为 ['val']（test 禁止参与选型），当前 {allowed!r}"
            )
        if cs.get("test_forbidden") is not True:
            errors.append(f"{arm}: checkpoint_selection.test_forbidden 必须为 true")
    else:
        errors.append(f"{arm}: checkpoint_selection 必须是映射")
    tu = cfg.get("test_usage")
    if isinstance(tu, dict):
        if tu.get("train_on_test") != "forbidden":
            errors.append(f"{arm}: test_usage.train_on_test 必须为 forbidden，当前 {tu.get('train_on_test')!r}")
        if tu.get("select_checkpoint_on_test") != "forbidden":
            errors.append(
                f"{arm}: test_usage.select_checkpoint_on_test 必须为 forbidden，当前 {tu.get('select_checkpoint_on_test')!r}"
            )
    else:
        errors.append(f"{arm}: test_usage 必须是映射")
    # ddr_finetune_stage 内的 test 路径引用
    dfs = cfg.get("ddr_finetune_stage")
    if isinstance(dfs, dict):
        for p, v in walk_strings(dfs, "$.ddr_finetune_stage"):
            if TEST_PATH_RE.search(v):
                errors.append(f"{arm}: {p}={v!r} 疑似引用 test 路径——训练日程禁止指向 test")


def check_channel_order(arm, cfg, errors, skipped, allow_placeholders):
    """E. 所有 channel_order 必须严格等于 LESION_CHANNEL_ORDER。"""
    expected = list(LESION_CHANNEL_ORDER)
    found = walk_key(cfg, "channel_order")
    if not found:
        errors.append(f"{arm}: 配置中未声明任何 channel_order")
    for p, v in found:
        if is_placeholder(v):
            if allow_placeholders:
                skipped.append(f"{arm}: {p} 为占位符，通道序检查跳过")
            else:
                errors.append(f"{arm}: {p} 为占位符（正式模式禁止）")
            continue
        if v != expected:
            errors.append(f"{arm}: {p}={v!r} 必须严格等于 {expected}")


def check_batch_conservation(arm, stage_name, st, errors, skipped, allow_placeholders):
    """H. effective_batch == micro_batch * grad_accum。"""
    eb, mb, ga = st.get("effective_batch"), st.get("micro_batch"), st.get("grad_accum")
    vals = (eb, mb, ga)
    if any(is_placeholder(v) for v in vals):
        skipped.append(f"{arm}/{stage_name}: effective_batch/micro_batch/grad_accum 含占位符，乘积守恒校验跳过")
        return
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
        errors.append(f"{arm}/{stage_name}: effective_batch/micro_batch/grad_accum 必须为整数，当前 {vals!r}")
        return
    if eb != mb * ga:
        errors.append(f"{arm}/{stage_name}: effective_batch={eb} != micro_batch={mb} * grad_accum={ga}")


def verify_file_sha(arm, label, d, errors, skipped, allow_placeholders):
    """I. {file, sha256} 映射的文件存在性与哈希一致性（占位符跳过并记录）。"""
    if not isinstance(d, dict):
        errors.append(f"{arm}: {label} 必须是 {{file, sha256}} 映射，当前 {type(d).__name__}")
        return
    f, s = d.get("file"), d.get("sha256")
    if is_placeholder(f) or is_placeholder(s):
        skipped.append(f"{arm}: {label} 含占位符，文件存在性/SHA256 校验跳过")
        return
    if not isinstance(f, str) or not f:
        errors.append(f"{arm}: {label}.file 缺失或非字符串")
        return
    if not isinstance(s, str) or not HEX64_RE.match(s):
        errors.append(f"{arm}: {label}.sha256 必须为 64 位十六进制，当前 {s!r}")
        return
    if not os.path.isfile(f):
        errors.append(f"{arm}: {label}.file 不存在: {f}")
        return
    try:
        actual = sha256_file(f)
    except ContractError as e:
        errors.append(f"{arm}: {label}: {e}")
        return
    if actual != s:
        errors.append(f"{arm}: {label}.sha256 与实际文件不符（记录 {s[:12]}… != 实际 {actual[:12]}…）")


def check_initialization(arm, cfg, errors, skipped, allow_placeholders):
    """D(part)/I. 控制臂无外部预训练；预训练臂 checkpoint 文件+哈希。"""
    init = cfg.get("initialization")
    if not isinstance(init, dict):
        errors.append(f"{arm}: initialization 必须是映射")
        return
    ext = init.get("external_pretrain_checkpoint", MISSING)
    if arm in CONTROL_ARMS:
        if ext is MISSING:
            errors.append(f"{arm}: 控制臂必须显式声明 initialization.external_pretrain_checkpoint: null")
        elif ext is not None:
            errors.append(
                f"{arm}: 控制臂禁止外部预训练权重：external_pretrain_checkpoint 必须为 null，当前 {ext!r}"
            )
    else:  # H1/U1
        if ext is MISSING:
            errors.append(f"{arm}: 预训练臂必须提供 initialization.external_pretrain_checkpoint")
        elif is_placeholder(ext):
            if allow_placeholders:
                skipped.append(f"{arm}: initialization.external_pretrain_checkpoint 为占位符，文件/哈希校验跳过")
            else:
                errors.append(f"{arm}: initialization.external_pretrain_checkpoint 为占位符（正式模式禁止）")
        else:
            verify_file_sha(arm, "initialization.external_pretrain_checkpoint", ext,
                            errors, skipped, allow_placeholders)
    # ImageNet 初始化权重（四臂都要求）
    iw = init.get("imagenet_weights")
    if iw is MISSING or iw is None:
        errors.append(f"{arm}: initialization.imagenet_weights 必须存在")
    elif is_placeholder(iw):
        skipped.append(f"{arm}: initialization.imagenet_weights 为占位符，校验跳过")
    else:
        verify_file_sha(arm, "initialization.imagenet_weights", iw, errors, skipped, allow_placeholders)
    # U 臂编码器权重（与 imagenet_weights 同义，防止两处漂移）
    st = cfg.get("structure")
    if isinstance(st, dict) and arm.startswith("U"):
        for key in ("encoder_weights_file", "encoder_weights_sha256"):
            if key in st and is_placeholder(st[key]) and allow_placeholders:
                skipped.append(f"{arm}: structure.{key} 为占位符，校验跳过")
        fw, sw = st.get("encoder_weights_file"), st.get("encoder_weights_sha256")
        if is_placeholder(fw) or is_placeholder(sw):
            pass  # 占位符：上方已记录跳过；正式模式由占位符扫描拦截
        elif (fw is None) != (sw is None):
            # fail-closed：file/sha256 只提供其一时绝不允许静默跳过校验
            errors.append(
                f"{arm}: structure.encoder_weights_file 与 encoder_weights_sha256 必须成对提供，"
                f"当前 file={fw!r} sha256={sw!r}"
            )
        elif fw is not None and sw is not None:
            verify_file_sha(
                arm, "structure.encoder_weights",
                {"file": fw, "sha256": sw}, errors, skipped, allow_placeholders,
            )


def check_external_manifest(arm, path, errors):
    """D(part). 预训练 manifest 内容契约：仅 official_split=train、来源白名单、
    lesions 恰为四键、sample_id 唯一。返回记录数（失败返回 None）。"""
    if not os.path.isfile(path):
        errors.append(f"{arm}: pretrain_stage.external_manifest 不存在: {path}")
        return None
    try:
        records = load_manifest(path)
    except ContractError as e:
        errors.append(f"{arm}: pretrain_stage.external_manifest: {e}")
        return None
    if not records:
        errors.append(f"{arm}: pretrain_stage.external_manifest 为空: {path}")
        return None
    seen = set()
    bad_split = {}
    for i, r in enumerate(records):
        sid = r.get("sample_id", f"record#{i}")
        if sid in seen:
            errors.append(f"{arm}: manifest sample_id 重复: {sid}")
        seen.add(sid)
        if r.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            errors.append(
                f"{arm}: manifest {sid}: schema_version 必须为 {MANIFEST_SCHEMA_VERSION!r}，"
                f"当前 {r.get('schema_version')!r}"
            )
        sp = r.get("official_split")
        if sp != "train":
            bad_split[sp] = bad_split.get(sp, 0) + 1
        src = r.get("source")
        if src not in ALLOWED_SOURCES:
            errors.append(f"{arm}: manifest {sid}: source {src!r} 不在白名单 {list(ALLOWED_SOURCES)}")
        lesions = r.get("lesions")
        if not isinstance(lesions, dict) or list(lesions.keys()) != list(LESION_CHANNEL_ORDER):
            errors.append(
                f"{arm}: manifest {sid}: lesions 键必须恰为 {list(LESION_CHANNEL_ORDER)}（顺序不可变）"
            )
    if bad_split:
        errors.append(
            f"{arm}: 预训练 manifest 只允许 official_split=train，发现 {json.dumps(bad_split, ensure_ascii=False)}"
            "（official test 绝不能进入预训练池）"
        )
    return len(records)


def check_pretrain_stage(arm, cfg, errors, skipped, allow_placeholders, manifest_records):
    """D(part). pretrain_stage 结构与 manifest 引用。"""
    ps = cfg.get("pretrain_stage")
    if not isinstance(ps, dict):
        return  # schema 检查已报
    forbidden = ps.get("forbidden_splits")
    if "test" not in (forbidden or []):
        errors.append(f"{arm}: pretrain_stage.forbidden_splits 必须包含 'test'，当前 {forbidden!r}")
    em = ps.get("external_manifest", MISSING)
    if em is MISSING or em is None:
        errors.append(f"{arm}: pretrain_stage.external_manifest 必须存在")
    elif is_placeholder(em):
        if allow_placeholders:
            skipped.append(f"{arm}: pretrain_stage.external_manifest 为占位符，manifest 内容检查跳过")
        else:
            errors.append(f"{arm}: pretrain_stage.external_manifest 为占位符（正式模式禁止）")
    elif isinstance(em, str):
        n = check_external_manifest(arm, em, errors)
        if n is not None:
            manifest_records[f"{arm}@{os.path.abspath(em)}"] = n
    else:
        errors.append(f"{arm}: pretrain_stage.external_manifest 必须是路径字符串")
    ico = ps.get("init_checkpoint_output", MISSING)
    if ico is MISSING:
        errors.append(f"{arm}: pretrain_stage.init_checkpoint_output 必须存在（预训练产出路径+sha256）")
    elif is_placeholder(ico):
        if allow_placeholders:
            skipped.append(f"{arm}: pretrain_stage.init_checkpoint_output 为占位符，校验跳过")
        else:
            errors.append(f"{arm}: pretrain_stage.init_checkpoint_output 为占位符（正式模式禁止）")
    else:
        verify_file_sha(arm, "pretrain_stage.init_checkpoint_output", ico,
                        errors, skipped, allow_placeholders)


def check_ddr_splits(arm, cfg, errors, skipped, allow_placeholders):
    """F. DDR root 存在、stem 对齐、三 split 互斥（只读文件名，不读任何像素）。"""
    data = cfg.get("data")
    if not isinstance(data, dict):
        return
    root = data.get("ddr_root")
    if is_placeholder(root):
        skipped.append(f"{arm}: data.ddr_root 为占位符，DDR split 检查跳过")
        return
    if not isinstance(root, str) or not root:
        errors.append(f"{arm}: data.ddr_root 缺失或非字符串")
        return
    if not os.path.isdir(root):
        errors.append(f"{arm}: DDR 根目录不存在: {root}")
        return
    stem_sets = {}
    for role in ("train", "val", "test"):
        field = f"ddr_{role}_split"
        sv = data.get(field)
        if is_placeholder(sv):
            skipped.append(f"{arm}: data.{field} 为占位符，该 split 检查跳过")
            continue
        if sv != role:
            errors.append(f"{arm}: data.{field}={sv!r} 必须为 {role!r}（官方划分语义固定）")
            continue
        img_dir = os.path.join(root, "images", role)
        lbl_dir = os.path.join(root, "labels", role)
        if not os.path.isdir(img_dir):
            errors.append(f"{arm}: DDR images/{role} 目录缺失: {img_dir}")
            continue
        if not os.path.isdir(lbl_dir):
            errors.append(f"{arm}: DDR labels/{role} 目录缺失: {lbl_dir}")
            continue
        s_img = stems_of(img_dir, IMG_EXTS)
        s_lbl = stems_of(lbl_dir, LABEL_EXTS)
        if s_img != s_lbl:
            errors.append(
                f"{arm}: DDR {role}: image/label stem 不对齐（images {len(s_img)} vs labels {len(s_lbl)}）"
            )
        if not s_img:
            errors.append(f"{arm}: DDR {role}: images/{role} 为空")
        stem_sets[role] = s_img
    roles = [r for r in ("train", "val", "test") if r in stem_sets]
    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            ov = stem_sets[roles[i]] & stem_sets[roles[j]]
            if ov:
                errors.append(
                    f"{arm}: DDR split 泄漏：{roles[i]} ∩ {roles[j]} = {len(ov)} 张，例如 {sorted(ov)[:5]}"
                )
    return {r: len(s) for r, s in stem_sets.items()}


def check_pairing(cfgs, errors):
    """C. H0↔H1 / U0↔U1 配对锁互指、字段集合一致、逐字段深度相等。"""
    for a_id, b_id in PAIRS:
        a, b = cfgs[a_id], cfgs[b_id]
        la, lb = a.get("equivalence_lock"), b.get("equivalence_lock")
        if not isinstance(la, dict) or not isinstance(lb, dict):
            errors.append(f"配对 {a_id}↔{b_id}: equivalence_lock 必须是映射")
            continue
        if la.get("identical_with") != b_id:
            errors.append(f"配对 {a_id}↔{b_id}: {a_id}.equivalence_lock.identical_with="
                          f"{la.get('identical_with')!r} 必须为 {b_id!r}")
        if lb.get("identical_with") != a_id:
            errors.append(f"配对 {a_id}↔{b_id}: {b_id}.equivalence_lock.identical_with="
                          f"{lb.get('identical_with')!r} 必须为 {a_id!r}")
        fa, fb = list(la.get("fields") or []), list(lb.get("fields") or [])
        if set(fa) != set(fb):
            only_a = sorted(set(fa) - set(fb))
            only_b = sorted(set(fb) - set(fa))
            errors.append(
                f"配对 {a_id}↔{b_id}: equivalence_lock.fields 集合不一致"
                f"（仅 {a_id}: {only_a}；仅 {b_id}: {only_b}）"
            )
        for f in MANDATORY_LOCK_FIELDS:
            if f not in set(fa):
                errors.append(f"配对 {a_id}↔{b_id}: equivalence_lock.fields 缺少强制字段 {f!r}")
        for f in sorted(set(fa) & set(fb)):
            va, vb = get_path(a, f), get_path(b, f)
            if va is MISSING or vb is MISSING:
                which = a_id if va is MISSING else b_id
                errors.append(f"配对 {a_id}↔{b_id}: 锁定字段 {f} 在 {which} 中不存在（无法比对）")
                continue
            d = first_difference(va, vb, f"$.{f}")
            if d:
                errors.append(f"配对 {a_id}↔{b_id}: 锁定字段不一致: {d}")


def check_cross_arm(cfgs, errors):
    """J. 四臂共享实验级字段与四个公共节。"""
    for f in CROSS_ARM_FIELDS:
        vals = {arm: get_path(cfgs[arm], f) for arm in ARM_ORDER}
        ref_arm = next((a for a in ARM_ORDER if vals[a] is not MISSING), None)
        if ref_arm is None:
            continue
        ref = vals[ref_arm]
        for arm in ARM_ORDER:
            v = vals[arm]
            if v is MISSING:
                errors.append(f"跨臂一致性: {arm} 缺少 {f}")
                continue
            d = first_difference(ref, v, f"$.{f}")
            if d:
                errors.append(f"跨臂一致性: {f} 在 {ref_arm} 与 {arm} 不一致: {d}")
    for sec in CROSS_ARM_SECTIONS:
        ref_arm = next((a for a in ARM_ORDER if isinstance(cfgs[a].get(sec), dict)), None)
        if ref_arm is None:
            errors.append(f"跨臂一致性: 四臂均缺少 {sec} 节")
            continue
        ref = cfgs[ref_arm][sec]
        for arm in ARM_ORDER:
            v = cfgs[arm].get(sec)
            if not isinstance(v, dict):
                errors.append(f"跨臂一致性: {arm} 缺少 {sec} 节")
                continue
            d = first_difference(ref, v, f"$.{sec}")
            if d:
                errors.append(f"跨臂一致性: {sec} 节在 {ref_arm} 与 {arm} 不一致: {d}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="开 GPU 前的四配置契约闸门（fail closed；真实资产未到位时保持 FAIL）"
    )
    ap.add_argument("--configs", nargs=4, metavar="YAML", default=None,
                    help="四份臂配置 YAML 路径（按 arm_id 识别，须恰为 H0/H1/U0/U1）")
    ap.add_argument("--config-dir", default=None,
                    help="包含恰好四份臂配置 YAML 的目录（默认 <repo>/configs）")
    ap.add_argument("--allow-placeholders", action="store_true",
                    help="仅用于结构/合成测试：REQUIRED/TBD 占位符不阻塞（跳过项显式记录）。"
                         "正式放行绝不允许使用本旗标。")
    ap.add_argument("--layer", default="formal", choices=("formal", "readiness"),
                    help="formal=完整契约闸门（默认，任何占位符都 FAIL）；"
                         "readiness=只把 readiness 层占位符判 FAIL，calibration 层"
                         "（micro_batch/grad_accum/gradient_checkpointing/权重资产/部署路径）"
                         "记为 GPU_REQUIRED。任何模式下都不放宽正式训练闸门，"
                         "该分层只用于回答『能否先开一次短时 GPU 校准』。")
    ap.add_argument("--output-json", default=None, help="可选：把报告写入 JSON 文件")
    return ap.parse_args(argv)


def load_configs(args, errors):
    """加载配置；致命问题（文件缺失/YAML 损坏/臂集合不对）返回 None。"""
    if args.configs:
        paths = list(args.configs)
        for p in paths:
            if not os.path.isfile(p):
                print(f"[preflight_contract] FATAL: 配置文件不存在: {p}", file=sys.stderr)
                return None
    else:
        cfg_dir = args.config_dir or os.path.join(REPO_ROOT, "configs")
        if not os.path.isdir(cfg_dir):
            print(f"[preflight_contract] FATAL: 配置目录不存在: {cfg_dir}", file=sys.stderr)
            return None
        paths = sorted(
            os.path.join(cfg_dir, f) for f in os.listdir(cfg_dir)
            if f.lower().endswith((".yaml", ".yml"))
        )
    cfgs = {}
    arm_files = {}
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"[preflight_contract] FATAL: YAML 解析失败 {p}: {e}", file=sys.stderr)
            return None
        arm = cfg.get("arm_id") if isinstance(cfg, dict) else None
        if arm not in ARM_ORDER:
            print(f"[preflight_contract] FATAL: {p} 的 arm_id={arm!r} 不在 {ARM_ORDER}", file=sys.stderr)
            return None
        if arm in cfgs:
            print(f"[preflight_contract] FATAL: arm {arm} 有重复配置: {arm_files[arm]} 与 {p}", file=sys.stderr)
            return None
        cfgs[arm] = cfg
        arm_files[arm] = p
    missing = [a for a in ARM_ORDER if a not in cfgs]
    if missing:
        print(f"[preflight_contract] FATAL: 缺少臂配置: {missing}", file=sys.stderr)
        return None
    return cfgs, arm_files


def main(argv=None) -> int:
    args = parse_args(argv)
    mode = "placeholders-allowed" if args.allow_placeholders else "formal"
    layer = args.layer
    # readiness 层：只要求 readiness 层占位符清空；其余占位符按"待校准"处理，
    # 因此除顶层扫描外的各项检查一律按 allow-placeholders 的跳过语义运行。
    allow_ph = args.allow_placeholders or layer == "readiness"
    errors: list = []
    skipped: list = []
    calibration_pending: list = []
    manifest_records: dict = {}
    ddr_split_sizes: dict = {}

    loaded = load_configs(args, errors)
    if loaded is None:
        return 2
    cfgs, arm_files = loaded

    # ---- 每臂检查 ----
    for arm in ARM_ORDER:
        cfg = cfgs[arm]
        check_arm_schema(arm, cfg, errors)
        # B. 占位符（按层分类）
        ph = scan_placeholders(cfg)
        if ph:
            if args.allow_placeholders:
                for p, v in ph:
                    skipped.append(f"{arm}: 占位符 {v} @ {p}（允许，仅限结构/合成测试）")
            elif layer == "readiness":
                for p, v in ph:
                    lv = _placeholder_layer(p) if _placeholder_layer else "readiness"
                    if lv == "calibration":
                        calibration_pending.append(f"{arm}: {v} @ {p}（GPU_REQUIRED，待校准回填）")
                    else:
                        errors.append(f"{arm}: readiness 层占位符 {v} @ {p}")
            else:
                for p, v in ph:
                    errors.append(f"{arm}: 占位符 {v} @ {p}（正式模式遇到占位符必须失败）")
        check_train_test_exclusion(arm, cfg, errors)
        check_channel_order(arm, cfg, errors, skipped, allow_ph)
        init = cfg.get("initialization")
        if isinstance(init, dict):
            check_initialization(arm, cfg, errors, skipped, allow_ph)
        dfs = cfg.get("ddr_finetune_stage")
        if isinstance(dfs, dict):
            check_batch_conservation(arm, "ddr_finetune_stage", dfs, errors, skipped, allow_ph)
            norm = dfs.get("mixed_precision")
            if norm is not True:
                errors.append(f"{arm}: ddr_finetune_stage.mixed_precision 必须为 true，当前 {norm!r}")
        if arm in PRETRAIN_ARMS:
            ps = cfg.get("pretrain_stage")
            if isinstance(ps, dict):
                check_batch_conservation(arm, "pretrain_stage", ps, errors, skipped, allow_ph)
                check_pretrain_stage(arm, cfg, errors, skipped, allow_ph, manifest_records)
        sizes = check_ddr_splits(arm, cfg, errors, skipped, allow_ph)
        if sizes:
            ddr_split_sizes[arm] = sizes

    # ---- H1/U1 必须共用同一份 manifest ----
    em_paths = []
    for arm in PRETRAIN_ARMS:
        em = get_path(cfgs[arm], "pretrain_stage.external_manifest")
        if isinstance(em, str) and not is_placeholder(em):
            em_paths.append((arm, os.path.abspath(em)))
    # normcase：Windows 文件系统大小写不敏感，D:/x.jsonl 与 d:/x.jsonl 是同一份文件
    if len(em_paths) == 2 and os.path.normcase(em_paths[0][1]) != os.path.normcase(em_paths[1][1]):
        errors.append(
            f"配对 H1↔U1: pretrain_stage.external_manifest 必须是同一份文件"
            f"（H1={em_paths[0][1]}，U1={em_paths[1][1]}）"
        )

    # ---- 配对与跨臂 ----
    check_pairing(cfgs, errors)
    check_cross_arm(cfgs, errors)

    # ---- 报告 ----
    readiness_state = "LOCAL_READY" if not errors else "BLOCKED"
    calibration_state = "READY" if not calibration_pending else "GPU_REQUIRED"
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "layer": layer,
        "readiness": readiness_state,
        "calibration": calibration_state,
        "configs": {arm: os.path.abspath(p) for arm, p in arm_files.items()},
        "ok": not errors,
        "error_count": len(errors),
        "skipped_count": len(skipped),
        "calibration_pending_count": len(calibration_pending),
        "calibration_pending": calibration_pending,
        "manifest_records": manifest_records,
        "ddr_split_sizes": ddr_split_sizes,
        "errors": errors,
        "skipped": skipped,
    }
    if args.output_json:
        write_text_atomic(
            args.output_json, json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        print(f"[preflight_contract] report -> {os.path.abspath(args.output_json)}")

    if errors:
        print(f"[preflight_contract] FAIL（mode={mode} layer={layer}）: {len(errors)} 个契约错误：")
        for e in errors:
            print(f"  ERROR: {e}")
        if skipped:
            print(f"[preflight_contract] 跳过项（显式记录，不静默）: {len(skipped)}")
            for s in skipped:
                print(f"  SKIP: {s}")
        print(
            "[preflight_contract] 真实资产未到位时保持 FAIL 是设计行为；"
            "占位符全部补齐并复核后重跑。"
        )
        return 1

    print(f"[preflight_contract] PASS（mode={mode} layer={layer}）: 四臂配置、配对公平性、"
          f"预训练 manifest、通道序、DDR split、test 隔离、批量守恒全部通过")
    print(f"[preflight_contract] readiness={readiness_state} calibration={calibration_state}")
    if calibration_pending:
        print(f"[preflight_contract] 待 GPU 校准回填（{len(calibration_pending)} 项，"
              f"**不阻塞短时校准开机，但正式训练必须清零**）:")
        for s in calibration_pending:
            print(f"  CALIBRATION_REQUIRED: {s}")
    if skipped:
        print(f"[preflight_contract] 跳过项（显式记录，不静默）: {len(skipped)}")
        for s in skipped:
            print(f"  SKIP: {s}")
        if not allow_ph:
            print("[preflight_contract] 警告：正式模式出现跳过项属于异常，请人工复核")
    if mode != "formal" or layer != "formal":
        print("[preflight_contract] 注意：--allow-placeholders / --layer readiness 的 PASS "
              "不得作为正式训练放行依据（正式训练走 train.py --mode formal）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
