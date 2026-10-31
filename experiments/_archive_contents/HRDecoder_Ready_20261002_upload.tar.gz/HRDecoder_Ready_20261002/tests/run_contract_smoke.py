# -*- coding: utf-8 -*-
"""run_contract_smoke.py — 契约闸门合成冒烟测试（离线、无网络、不触碰真实 DDR 数据）。

在 tests/_tmp 下生成全套合成资产并验证两个闸门的正反例行为：

P0  py_compile：tools/*.py 与 tests/*.py 全部可编译（.pyc 落在 tests/_tmp/pyc）；
正例（必须 PASS，退出码 0）：
  P1  tools/preflight_contract.py  正式模式（无 --allow-placeholders），
      四份合成配置全部占位符已替换为可验证的合成资产 → 退出码 0，且零错误；
  P2  tools/audit_manifest.py      train-only 合成 manifest（图像/掩膜/SHA256/
      感知哈希全部真实对应合成文件）→ 退出码 0。
反例（每个必须 FAIL，退出码 1，且错误信息命中预期子串）：
  N1  H0/H1 equivalence_lock 锁定字段不一致（data.input_fov 被改动）；
  N2  U0/U1 equivalence_lock 锁定字段不一致（loss.smooth 被改动）；
  N3  预训练 manifest 混入 official_split=test 记录（泄漏）；
  N4  正式模式遇到 REQUIRED 占位符；
  N4b 正式模式遇到 TBD 占位符；
  N5  audit：manifest 引用的 image 文件缺失；
  N6  audit：manifest 引用的 mask 文件缺失；
  N7  audit：valid_mode=none 却携带 mask_path/mask_sha256。
真实闸门（收尾验收）：
  R1  真实四配置（configs/*.yaml，含 REQUIRED/TBD）正式模式 preflight：
      资产未齐必须以明确 blocker 非零退出（rc=1）。preflight 子进程只读
      配置与真实 DDR 的文件名清单（契约行为，绝不读任何像素）。

退出码：0=全部断言通过；1=存在失败用例；4=环境依赖缺失（PyYAML/Pillow，显式报告）。
测试脚本自身只读写 tests/_tmp：全部文件访问经由 _check 白名单守卫（规范化 +
前缀校验 + 拒绝 .. 分量），并用 pathlib 的 write_text/write_bytes/read_text/
read_bytes 完成；绝不读取任何图像像素（含真实 DDR test 像素），绝不联网，
不安装任何依赖。
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)

try:
    import yaml
except Exception as e:  # 显式报告，绝不静默
    print(f"[run_contract_smoke] FATAL: PyYAML 未安装（import yaml 失败: {e}）。"
          "本测试需要解析/生成 YAML 配置。请 `python -m pip install pyyaml` 后重试。",
          file=sys.stderr)
    sys.exit(4)

try:
    from PIL import Image
except Exception as e:  # 显式报告，绝不静默
    print(f"[run_contract_smoke] FATAL: Pillow 未安装（import PIL 失败: {e}）。"
          "本测试需要生成合成 RGB PNG 与二值掩膜。请 `python -m pip install pillow` 后重试。",
          file=sys.stderr)
    sys.exit(4)

from contract_common import (  # noqa: E402  合成资产哈希/感知哈希复用同一实现
    compute_perceptual_hashes,
    sha256_bytes,
    sha256_file,
)

TMP = os.path.normpath(os.path.join(REPO_ROOT, "tests", "_tmp"))
CONFIGS_OUT = os.path.join(TMP, "configs")
REPORTS = os.path.join(TMP, "reports")
DDR_ROOT = os.path.join(TMP, "ddr")
EXT_ROOT = os.path.join(TMP, "external_data")
WEIGHTS_DIR = os.path.join(TMP, "weights")
MANIFEST_DIR = os.path.join(TMP, "manifests")
MANIFEST_PATH = os.path.join(MANIFEST_DIR, "external_train_manifest.jsonl")
# 合成"旧仓库"桩：只为 reference_root 字段提供一个结构上像样的目标，
# preflight/契约层只看路径存在性；H 臂真正 import 时会因缺 mmcv 而 DEPENDENCY_BLOCKED。
REF_STUB = os.path.join(TMP, "refrepo")

ARM_ORDER = ("H0", "H1", "U0", "U1")
REAL_CONFIGS = {
    "H0": os.path.join(REPO_ROOT, "configs", "H0_hrdecoder_ddr_only.yaml"),
    "H1": os.path.join(REPO_ROOT, "configs", "H1_hrdecoder_ext_pretrain.yaml"),
    "U0": os.path.join(REPO_ROOT, "configs", "U0_convnext_unet_ddr_only.yaml"),
    "U1": os.path.join(REPO_ROOT, "configs", "U1_convnext_unet_ext_pretrain.yaml"),
}
IMG_W, IMG_H = 64, 48

_results = []

# ---------------------------------------------------------------------------
# 文件访问白名单守卫（本文件所有读写的前置校验；写入仅限 tests/_tmp）
# ---------------------------------------------------------------------------

_WRITE_ROOTS = (TMP,)
_READ_ROOTS = (TMP, REPO_ROOT)


def _check(path, roots):
    """规范化 + 前缀白名单 + 显式拒绝 .. 分量；通过后返回规范化绝对路径。"""
    p = os.path.normpath(os.path.abspath(path))
    if not any(p == r or p.startswith(r + os.sep) for r in roots):
        raise PermissionError(f"path outside allowed roots: {p}")
    if os.pardir in Path(p).parts:
        raise PermissionError(f"path must not contain '..' components: {p}")
    return p


def write_text(path, text):
    p = _check(path, _WRITE_ROOTS)
    Path(p).write_text(text, encoding="utf-8", newline="\n")


def write_bytes(path, blob):
    p = _check(path, _WRITE_ROOTS)
    Path(p).write_bytes(blob)


def read_text(path):
    return Path(_check(path, _READ_ROOTS)).read_text(encoding="utf-8")


def read_bytes(path):
    return Path(_check(path, _READ_ROOTS)).read_bytes()


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def run_py(script, *argv):
    """以 UTF-8 子进程运行 tools/ 下脚本，返回 (rc, stdout+stderr)。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [sys.executable, script, *argv],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=REPO_ROOT, env=env, timeout=120,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def write_rgb_png(path, w, h, seed):
    """确定性渐变 RGB PNG（内容随 seed 变化，保证图间哈希距离大）。"""
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = ((x * 3 + seed) % 256, (y * 5 + seed * 7) % 256, (x * y + seed * 11) % 256)
    img.save(_check(path, _WRITE_ROOTS), format="PNG")


def write_binary_mask(path, w, h, rects):
    """二值掩膜 PNG：背景 0，rects=(x0,y0,x1,y1) 内 255（非零值唯一）。"""
    img = Image.new("L", (w, h), 0)
    px = img.load()
    for x0, y0, x1, y1 in rects:
        for y in range(y0, min(y1, h)):
            for x in range(x0, min(x1, w)):
                px[x, y] = 255
    img.save(_check(path, _WRITE_ROOTS), format="PNG")


def sha_of(path):
    return sha256_file(_check(path, _READ_ROOTS))


def dump_yaml(cfg, path):
    write_text(path, yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))


def load_yaml(path):
    return yaml.safe_load(read_text(path))


# ---------------------------------------------------------------------------
# 合成资产构建
# ---------------------------------------------------------------------------

def build_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    out = {}
    blobs = (
        ("imagenet_dummy.pth", b"SYNTHETIC-IMAGENET-WEIGHTS-" + bytes(range(64))),
        ("external_pretrain_dummy.pth", b"SYNTHETIC-EXTERNAL-PRETRAIN-" + bytes(range(48))),
        ("encoder_dummy.pth", b"SYNTHETIC-CONVNEXT-ENCODER-" + bytes(range(56))),
        ("init_ckpt_output_dummy.pth", b"SYNTHETIC-INIT-CKPT-OUTPUT-" + bytes(range(40))),
    )
    for name, blob in blobs:
        path = os.path.join(WEIGHTS_DIR, name)
        write_bytes(path, blob)
        out[name] = (path, sha_of(path))
    return out


def build_ddr_tree():
    """合成 DDR 目录树：只含 64x48 PNG，train/val/test stem 互斥。"""
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(DDR_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DDR_ROOT, "labels", split), exist_ok=True)
    plan = {
        "train": (("DDR_tr_A", 1), ("DDR_tr_B", 2)),
        "val": (("DDR_va_C", 3),),
        "test": (("DDR_te_D", 4),),
    }
    for split, items in plan.items():
        for stem, seed in items:
            write_rgb_png(os.path.join(DDR_ROOT, "images", split, stem + ".png"),
                          IMG_W, IMG_H, seed)
            # 融合标签：0 背景 + 少量 255 区域（preflight 只看文件名，不读像素）
            write_binary_mask(os.path.join(DDR_ROOT, "labels", split, stem + ".png"),
                              IMG_W, IMG_H, ((4, 4, 20, 16), (30, 20, 44, 30)))


def build_reference_stub():
    """合成旧仓库桩目录（只含占位文件，不被 import）。"""
    d = os.path.join(REF_STUB, "mmseg", "models", "backbones")
    os.makedirs(d, exist_ok=True)
    write_text(os.path.join(d, "hrnet.py"), "# synthetic reference stub\n")


def build_external_assets():
    """合成外部数据树 + train-only manifest（含感知哈希，路径相对 TMP）。"""
    plan = (
        # (source, image_id, 每类掩膜目录名, {cls: valid_mode})
        ("IDRiD", "IDRiD_001", "labels",
         {"EX": "full", "HE": "full", "SE": "full", "MA": "full"}),
        ("MAPLES-DR", "20051019_38557_0100", "anomalies",
         {"EX": "full", "HE": "full", "SE": "none", "MA": "file"}),
    )
    os.makedirs(os.path.join(EXT_ROOT, "IDRiD", "images", "train"), exist_ok=True)
    for cls in ("EX", "HE", "SE", "MA"):
        os.makedirs(os.path.join(EXT_ROOT, "IDRiD", "labels", "train", cls), exist_ok=True)
    os.makedirs(os.path.join(EXT_ROOT, "MAPLES-DR", "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(EXT_ROOT, "MAPLES-DR", "anomalies", "train"), exist_ok=True)
    os.makedirs(os.path.join(EXT_ROOT, "MAPLES-DR", "quality", "train"), exist_ok=True)

    records = []
    for source, image_id, mask_rootname, modes in plan:
        img_abs = os.path.join(EXT_ROOT, source, "images", "train", image_id + ".jpg")
        write_rgb_png(img_abs, IMG_W, IMG_H, 17 if source == "IDRiD" else 99)
        phash = compute_perceptual_hashes(img_abs)

        lesions = {}
        for cls in ("EX", "HE", "SE", "MA"):
            mode = modes[cls]
            if mode == "none":
                lesions[cls] = {"annotated": False, "valid_mode": "none", "mask_path": None,
                                "mask_sha256": None, "valid_mask_path": None,
                                "positive_pixels": None}
                continue
            mask_abs = os.path.join(EXT_ROOT, source, mask_rootname, "train",
                                    f"{image_id}_{cls}.png")
            rects = ((2, 2, 18, 12), (28, 24, 40, 34)) if cls != "SE" else ()
            write_binary_mask(mask_abs, IMG_W, IMG_H, rects)
            valid_mask_rel = None
            if mode == "file":
                valid_abs = os.path.join(EXT_ROOT, source, "quality", "train",
                                         f"{image_id}_{cls}_valid.png")
                write_binary_mask(valid_abs, IMG_W, IMG_H, ((0, 0, IMG_W, IMG_H // 2),))
                valid_mask_rel = f"external_data/{source}/quality/train/{image_id}_{cls}_valid.png"
            lesions[cls] = {
                "annotated": True, "valid_mode": mode,
                "mask_path": f"external_data/{source}/{mask_rootname}/train/{image_id}_{cls}.png",
                "mask_sha256": sha_of(mask_abs),
                "valid_mask_path": valid_mask_rel,
                "positive_pixels": 260 if rects else 0,
            }
        records.append({
            "schema_version": "1.0",
            "sample_id": f"{source}_train_{image_id}",
            "source": source,
            "source_version": None,
            "official_split": "train",
            "patient_id": image_id if source == "MAPLES-DR" else None,
            "patient_id_source": "image_basename_fallback" if source == "MAPLES-DR" else None,
            "image_id": image_id,
            "original_width": IMG_W,
            "original_height": IMG_H,
            "image": {"path": f"external_data/{source}/images/train/{image_id}.jpg",
                      "sha256": sha_of(img_abs), "format": "jpg"},
            "lesions": lesions,
            "perceptual_hash": {"ahash": phash.get("ahash"), "dhash": phash.get("dhash"),
                                "phash": phash.get("phash")},
            "notes": "合成冒烟测试记录；非真实数据。",
        })

    os.makedirs(MANIFEST_DIR, exist_ok=True)
    write_manifest_records(MANIFEST_PATH, records)
    return records


def write_manifest_records(path, records):
    write_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


# ---------------------------------------------------------------------------
# 合成配置（真实四配置 → 占位符替换 → tests/_tmp/configs）
# ---------------------------------------------------------------------------

def fill_one(arm, cfg, weights):
    """单臂占位符填充（构建与恢复共用同一逻辑，防止漂移）。"""
    st = cfg["structure"]
    st["training_entry"] = "train.py"
    if cfg["structure"]["family"] == "hrdecoder":
        st["reference_root"] = REF_STUB.replace(os.sep, "/")
    cfg["initialization"]["imagenet_weights"] = {
        "source": "synthetic://imagenet_w48", "file": weights["imagenet_dummy.pth"][0],
        "sha256": weights["imagenet_dummy.pth"][1],
    }
    if arm in ("H1", "U1"):
        cfg["initialization"]["external_pretrain_checkpoint"] = {
            "file": weights["external_pretrain_dummy.pth"][0],
            "sha256": weights["external_pretrain_dummy.pth"][1],
        }
    if arm.startswith("U"):
        st["encoder_weights_source"] = "synthetic://convnext_small"
        st["encoder_weights_file"] = weights["encoder_dummy.pth"][0]
        st["encoder_weights_sha256"] = weights["encoder_dummy.pth"][1]
        st["library_version"] = "0.3.3-synthetic"
        cfg["data"]["ddr_normalization"] = {
            "mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375],
        }
    cfg["data"]["ddr_root"] = DDR_ROOT.replace(os.sep, "/")
    dfs = cfg["ddr_finetune_stage"]
    dfs["micro_batch"], dfs["grad_accum"], dfs["gradient_checkpointing"] = 2, 4, False
    cfg["loss"]["per_class_valid_masking"] = True
    if arm in ("H1", "U1"):
        ps = cfg["pretrain_stage"]
        ps["micro_batch"], ps["grad_accum"] = 2, 4
        ps["external_manifest"] = MANIFEST_PATH.replace(os.sep, "/")
        ps["overfit_monitor"] = "dev_loss_every_1000_updates"
        ps["init_checkpoint_output"] = {
            "file": weights["init_ckpt_output_dummy.pth"][0],
            "sha256": weights["init_ckpt_output_dummy.pth"][1],
        }
    return cfg


def write_synthetic_config(arm, weights):
    cfg = load_yaml(REAL_CONFIGS[arm])
    fill_one(arm, cfg, weights)
    dump_yaml(cfg, os.path.join(CONFIGS_OUT, f"{arm}_synthetic.yaml"))


# ---------------------------------------------------------------------------
# 闸门调用
# ---------------------------------------------------------------------------

def config_paths():
    return [os.path.join(CONFIGS_OUT, f"{arm}_synthetic.yaml") for arm in ARM_ORDER]


def run_preflight(report_name):
    report = os.path.join(REPORTS, report_name)
    rc, out = run_py(os.path.join(TOOLS_DIR, "preflight_contract.py"),
                     "--configs", *config_paths(), "--output-json", report)
    errors = []
    if os.path.isfile(report):
        errors = json.loads(read_text(report)).get("errors", [])
    return rc, out, errors


def run_audit(manifest, report_name):
    rj = os.path.join(REPORTS, report_name + ".json")
    rc, out = run_py(
        os.path.join(TOOLS_DIR, "audit_manifest.py"),
        "--manifest", manifest, "--base-dir", TMP,
        "--output-json", rj, "--output-md", os.path.join(REPORTS, report_name + ".md"),
    )
    errors = []
    if os.path.isfile(rj):
        errors = json.loads(read_text(rj)).get("errors", [])
    return rc, out, errors


def load_report_errors(report_name):
    return json.loads(read_text(os.path.join(REPORTS, report_name))).get("errors", [])


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------

_BASE_RECORDS = []
_WEIGHTS = None


def record(case, ok, detail=""):
    _results.append((case, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {case}" + (f" — {detail}" if detail and not ok else ""))


def case_py_compile():
    """P0 py_compile：tools/*.py 与 tests/*.py 全部可编译（.pyc 只落 tests/_tmp/pyc）。"""
    import py_compile
    pyc_dir = os.path.join(TMP, "pyc")
    os.makedirs(pyc_dir, exist_ok=True)
    targets = sorted(
        [os.path.join(TOOLS_DIR, f) for f in os.listdir(TOOLS_DIR) if f.endswith(".py")]
        + [os.path.join(REPO_ROOT, "tests", f)
           for f in os.listdir(os.path.join(REPO_ROOT, "tests")) if f.endswith(".py")]
    )
    failures = []
    for src in targets:
        cfile = os.path.join(pyc_dir, os.path.splitext(os.path.basename(src))[0] + ".pyc")
        try:
            py_compile.compile(src, cfile=cfile, doraise=True)
        except py_compile.PyCompileError as e:
            failures.append(f"{os.path.basename(src)}: {e}")
    ok = bool(targets) and not failures
    record(f"P0 py_compile {len(targets)} 个脚本（tools/*.py + tests/*.py）", ok,
           "; ".join(failures[:4]))


def case_positive_preflight():
    rc, out, errors = run_preflight("preflight_positive.json")
    ok = rc == 0 and not errors
    record("P1 preflight 正式模式合成配置 PASS（rc=0，零错误）", ok,
           f"rc={rc} errors={errors[:4]} out_tail={out[-400:]}")


def case_pairing_mismatch(pair, mutate):
    a, b = pair
    cfg = load_yaml(os.path.join(CONFIGS_OUT, f"{b}_synthetic.yaml"))
    mutate(cfg)
    dump_yaml(cfg, os.path.join(CONFIGS_OUT, f"{b}_synthetic.yaml"))
    try:
        rc, out, errors = run_preflight(f"preflight_negative_pair_{a}_{b}.json")
        joined = "\n".join(errors)
        ok = rc == 1 and f"配对 {a}↔{b}" in joined and "锁定字段不一致" in joined
        record(f"N配对 {a}↔{b} 锁定字段不一致 → preflight FAIL", ok,
               f"rc={rc} errors={errors[:4]} out_tail={out[-300:]}")
    finally:
        write_synthetic_config(b, _WEIGHTS)  # 恢复基线


def case_test_leakage():
    leaked = copy.deepcopy(_BASE_RECORDS)
    ghost = copy.deepcopy(leaked[0])
    ghost["sample_id"] = "IDRiD_test_GHOST_999"
    ghost["official_split"] = "test"
    ghost["image_id"] = "GHOST_999"
    leaked.append(ghost)
    write_manifest_records(MANIFEST_PATH, leaked)
    try:
        rc, out, errors = run_preflight("preflight_negative_test_leak.json")
        joined = "\n".join(errors)
        ok = rc == 1 and "official_split=train" in joined
        record("N3 manifest 混入 official_split=test → preflight FAIL", ok,
               f"rc={rc} errors={errors[:4]} out_tail={out[-300:]}")
    finally:
        write_manifest_records(MANIFEST_PATH, _BASE_RECORDS)


def case_required_formal_mode(token="REQUIRED", case_name="N4", report="preflight_negative_required"):
    """正式模式遇到占位符（REQUIRED/TBD）→ preflight FAIL。"""
    cfg = load_yaml(os.path.join(CONFIGS_OUT, "H1_synthetic.yaml"))
    cfg["initialization"]["imagenet_weights"]["file"] = token
    dump_yaml(cfg, os.path.join(CONFIGS_OUT, "H1_synthetic.yaml"))
    try:
        rc, out, errors = run_preflight(report + ".json")
        joined = "\n".join(errors)
        ok = rc == 1 and f"占位符 {token}" in joined and "正式模式遇到占位符必须失败" in joined
        record(f"{case_name} 正式模式遇到 {token} → preflight FAIL", ok,
               f"rc={rc} errors={errors[:4]} out_tail={out[-300:]}")
    finally:
        write_synthetic_config("H1", _WEIGHTS)


def case_real_formal_preflight():
    """R1 真实配置正式 preflight：资产未齐必须以明确 blocker 非零退出（rc=1）。

    preflight 子进程按契约只读配置文件与真实 DDR 的文件名清单（不读任何像素）；
    测试脚本自身对报告的读写仍经白名单守卫（报告落在 tests/_tmp/reports）。
    """
    rc, out = run_py(
        os.path.join(TOOLS_DIR, "preflight_contract.py"),
        "--config-dir", os.path.join(REPO_ROOT, "configs"),
        "--output-json", os.path.join(REPORTS, "preflight_real_formal.json"),
    )
    errors = []
    if os.path.isfile(os.path.join(REPORTS, "preflight_real_formal.json")):
        errors = load_report_errors("preflight_real_formal.json")
    joined = "\n".join(errors)
    blockers = [e for e in errors if "占位符" in e and ("REQUIRED" in e or "TBD" in e)]
    ok = rc == 1 and bool(errors) and bool(blockers)
    record("R1 真实配置正式 preflight 资产未齐 → 明确 blocker 非零退出（rc=1）", ok,
           f"rc={rc} blockers={len(blockers)}/{len(errors)} out_tail={out[-300:]}")


def case_audit_negative(name, report_name, mutate_records, needle):
    records = copy.deepcopy(_BASE_RECORDS)
    mutate_records(records)
    bad = os.path.join(MANIFEST_DIR, report_name + ".jsonl")
    write_manifest_records(bad, records)
    rc, out, errors = run_audit(bad, report_name)
    joined = "\n".join(errors)
    ok = rc == 1 and needle in joined
    record(f"N{name} {report_name} → audit FAIL", ok,
           f"rc={rc} needle={needle!r} errors={errors[:4]} out_tail={out[-300:]}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 72)
    print("[run_contract_smoke] 合成契约冒烟测试（离线；只写 tests/_tmp）")
    print("=" * 72)

    if os.path.isdir(TMP):
        shutil.rmtree(TMP)
    for d in (CONFIGS_OUT, REPORTS, DDR_ROOT, EXT_ROOT, WEIGHTS_DIR, MANIFEST_DIR):
        os.makedirs(d, exist_ok=True)

    # P0：py_compile（环境/语法先决条件）
    print("\n-- P0：py_compile tools/*.py + tests/*.py --")
    case_py_compile()

    global _BASE_RECORDS, _WEIGHTS
    _WEIGHTS = build_weights()
    build_ddr_tree()
    build_reference_stub()
    _BASE_RECORDS = build_external_assets()

    # P1：正例 preflight（正式模式）
    print("\n-- 构建合成四配置（占位符全部替换为可验证合成资产） --")
    for arm in ARM_ORDER:
        write_synthetic_config(arm, _WEIGHTS)
    case_positive_preflight()

    # N1/N2：配对字段不一致
    print("\n-- 反例：配对公平性 --")

    def mut_h1(cfg):
        cfg["data"]["input_fov"] = 1024  # 锁定字段 data 被改动

    def mut_u1(cfg):
        cfg["loss"]["smooth"] = 2.0e-5  # 锁定字段 loss 被改动

    case_pairing_mismatch(("H0", "H1"), mut_h1)
    case_pairing_mismatch(("U0", "U1"), mut_u1)

    # N3：manifest 混入 official_split=test
    print("\n-- 反例：预训练 manifest 泄漏 --")
    case_test_leakage()

    # N4/N4b：正式模式占位符（REQUIRED 与 TBD）
    print("\n-- 反例：正式模式占位符 --")
    case_required_formal_mode("REQUIRED", "N4", "preflight_negative_required")
    case_required_formal_mode("TBD", "N4b", "preflight_negative_tbd")

    # P2：正例 audit
    print("\n-- 正例：manifest 审计 --")
    rc, out, errors = run_audit(MANIFEST_PATH, "audit_positive")
    ok = rc == 0 and not errors
    record("P2 audit train-only 合成 manifest PASS（rc=0，零错误）", ok,
           f"rc={rc} errors={errors[:4]} out_tail={out[-400:]}")

    # N5/N6/N7：audit 反例
    print("\n-- 反例：manifest 审计 --")
    case_audit_negative("5", "audit_negative_missing_image",
                        lambda rs: rs[0]["image"].update(
                            path="external_data/IDRiD/images/train/GHOST.jpg"),
                        "image missing")
    case_audit_negative("6", "audit_negative_missing_mask",
                        lambda rs: rs[0]["lesions"]["EX"].update(
                            mask_path="external_data/IDRiD/labels/train/EX/GHOST_EX.png"),
                        "mask missing")
    case_audit_negative("7", "audit_negative_valid_none_with_mask",
                        lambda rs: rs[1]["lesions"]["SE"].update(
                            mask_path=rs[1]["lesions"]["MA"]["mask_path"],
                            mask_sha256=rs[1]["lesions"]["MA"]["mask_sha256"]),
                        "valid_mode=none 但 mask_path/sha256 非空")

    # R1：真实配置正式 preflight（收尾验收；只读文件名，不读像素）
    print("\n-- R1：真实配置正式 preflight（资产未齐必须明确 blocker 非零退出） --")
    case_real_formal_preflight()

    # 汇总
    print("\n" + "=" * 72)
    failed = [c for c, ok, _ in _results if not ok]
    for case, ok, detail in _results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {case}"
              + (f"\n         {detail}" if detail and not ok else ""))
    print("-" * 72)
    print(f"[run_contract_smoke] {len(_results) - len(failed)}/{len(_results)} 用例通过；"
          f"资产位置: {TMP}")
    if failed:
        print(f"[run_contract_smoke] FAIL: {len(failed)} 个用例失败")
        return 1
    print("[run_contract_smoke] PASS: 契约闸门正反例行为全部符合预期")
    return 0


if __name__ == "__main__":
    sys.exit(main())
