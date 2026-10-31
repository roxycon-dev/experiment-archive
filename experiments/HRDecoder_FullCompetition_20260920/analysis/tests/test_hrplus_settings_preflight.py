# -*- coding: utf-8 -*-
"""test_hrplus_settings_preflight — HRPlus 导出的 settings/preflight 对象结构 CPU 单测。

回归目标（服务器 smoke 在 main -> line261 抛 `KeyError: '_cfg'`）：
`run_protocol.validate_settings()` 的返回值**只**含原始 settings 键 + `_hashes` /
`_abs` / `_pending`，**不含 `_cfg`**；`_cfg` 只是 `run_protocol.build_student()` 的
副作用（历史交接版 run_protocol.py:250）。导出脚本在 build_student 之前就用
`cfg.data[val]` 建 val dataset，因此必须自己解析 `_cfg`，不能直接索引。

本测试用假 `runtime` 覆盖：
1) `resolve_cfg` 在没有 `_cfg` 的 validate_settings 形状上成功解析、写入补丁、幂等；
2) `build_identity` 在「build_student 从未被调用」的 settings 上不再 KeyError；
3) 反例：未解析前直接取 `settings['_cfg']` 会 KeyError（锁住根因）。

不需要 torch / mmseg / GPU / 真实权重。

运行：`python analysis/tests/test_hrplus_settings_preflight.py`
（也可用 pytest 收集 test_* 函数）。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ANALYSIS = os.path.dirname(_HERE)
_REPO = os.path.dirname(_ANALYSIS)
for _p in (_ANALYSIS, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import export_hrplus_expert_masks as M  # noqa: E402

REPO = "/repo"
CONFIG_REL = "configs/lesion/exp/hrplus_patch_dice.py"
REF_ABS = "/repo/weights/B_reference.pth"


class _FakeCfg:
    """最小 mmseg Config 替身：有 .model / .data / .dump()。"""

    def __init__(self, data):
        self.data = data
        self.model = {"source_checkpoint": None}

    def dump(self):
        return json.dumps({"data": list(self.data), "model": self.model},
                          sort_keys=True)


class _FakeRT:
    """最小 runtime 替身：只记录 load_config 调用并返回 _FakeCfg。"""

    def __init__(self):
        self.calls = []

    def load_config(self, repo, config_rel):
        self.calls.append((repo, config_rel))
        return _FakeCfg({M.C.SPLIT: {"type": "FakeDDRDataset", "test_mode": False}})


def _validated_settings():
    """精确模仿 run_protocol.validate_settings 的返回结构（注意：没有 _cfg）。"""
    return {
        "repo": REPO,
        "config": CONFIG_REL,
        "data_root": "/data/ddr",
        "student_checkpoint": "/repo/weights/D2000.pth",
        "reference_checkpoint": "weights/B_reference.pth",
        "cache_dir": "/scratch/cache",
        "champion": {"head": "weights/h4_best.pth", "members": []},
        "expected_sha256": {},
        # validate_settings 追加的三个键：
        "_hashes": {"config": "aa" * 10},
        "_abs": {"config": os.path.join(REPO, CONFIG_REL),
                 "reference_checkpoint": REF_ABS},
        "_pending": [],
    }


# ---------------------------------------------------------------------------
# 1) resolve_cfg：结构 + 补丁 + 幂等
# ---------------------------------------------------------------------------

def test_validate_settings_shape_has_no_cfg():
    """锁住根因：validate_settings 形状里没有 _cfg，直接索引必 KeyError。"""
    s = _validated_settings()
    assert "_cfg" not in s
    try:
        s["_cfg"]  # noqa: B018 (故意触发)
        raise AssertionError("缺少 _cfg 时必须 KeyError")
    except KeyError:
        pass


def test_resolve_cfg_populates_and_patches():
    s = _validated_settings()
    rt = _FakeRT()
    cfg = M.resolve_cfg(s, rt)
    # 结构：写回 settings['_cfg']，且返回值就是同一个对象
    assert s["_cfg"] is cfg
    assert cfg is not None and hasattr(cfg, "dump") and hasattr(cfg, "data")
    assert M.C.SPLIT in cfg.data
    # 与 build_student 同构造：source_checkpoint 被补成 _abs 的绝对路径
    assert cfg.model["source_checkpoint"] == REF_ABS
    # 用 (repo, config) 加载，且只加载一次
    assert rt.calls == [(REPO, CONFIG_REL)]


def test_resolve_cfg_idempotent_does_not_reload():
    s = _validated_settings()
    rt = _FakeRT()
    first = M.resolve_cfg(s, rt)
    second = M.resolve_cfg(s, rt)
    assert first is second is s["_cfg"]
    assert len(rt.calls) == 1, "已有 _cfg 时不得重复加载配置"


def test_resolve_cfg_reuses_cfg_set_by_build_student():
    """若 build_student 已写 _cfg，resolve_cfg 必须原样复用（不覆盖、不重载）。"""
    s = _validated_settings()
    sentinel = _FakeCfg({M.C.SPLIT: {"type": "FromBuildStudent"}})
    s["_cfg"] = sentinel
    rt = _FakeRT()
    assert M.resolve_cfg(s, rt) is sentinel
    assert rt.calls == []


# ---------------------------------------------------------------------------
# 2) build_identity：build_student 从未调用时也不得 KeyError
# ---------------------------------------------------------------------------

def test_build_identity_no_keyerror_without_build_student():
    s = _validated_settings()
    rt = _FakeRT()
    raw = {"fusionaware_src_dir": "/fw/src", "fusionaware_exec_dir": "/fw/exec"}

    orig = M.C.sha256_file
    M.C.sha256_file = lambda p: "0" * 64          # 桩掉文件哈希，聚焦对象结构
    sys.modules["runtime"] = rt                    # build_identity 内部 `import runtime`
    try:
        ident = M.build_identity(s, "/settings.json", [], raw)
    finally:
        M.C.sha256_file = orig
        sys.modules.pop("runtime", None)

    assert s["_cfg"] is not None
    want = hashlib.sha256(s["_cfg"].dump().encode("utf-8")).hexdigest()
    assert ident["config_text_sha256"] == want
    assert ident["weights_sha256"] == s["_hashes"]
    assert ident["repo"] == REPO and ident["data_root"] == s["data_root"]
    assert ident["settings_file_sha256"] == "0" * 64


def test_build_identity_hashes_scripts_independent_of_repo():
    """settings.repo 指向不含 analysis 的临时目录时，build_identity 仍须成功，
    且脚本哈希必须来自真实脚本文件（不得依赖 settings.repo 的相对路径）。"""
    import tempfile

    def _sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    s = _validated_settings()
    with tempfile.TemporaryDirectory() as tmp:
        repo_no_analysis = os.path.join(tmp, "repo_without_analysis")
        fw_src = os.path.join(tmp, "fw_src")
        fw_exec = os.path.join(tmp, "fw_exec")
        for d in (repo_no_analysis, fw_src, fw_exec):
            os.makedirs(d)
        for name in ("runtime.py", "run_protocol.py", "fusion_core.py"):
            with open(os.path.join(fw_src, name), "w", encoding="utf-8") as f:
                f.write("# %s stub\n" % name)
        with open(os.path.join(fw_exec, "eval_candidates.py"), "w",
                  encoding="utf-8") as f:
            f.write("# eval_candidates.py stub\n")
        s["repo"] = repo_no_analysis
        settings_path = os.path.join(tmp, "settings.json")
        with open(settings_path, "w", encoding="utf-8") as f:
            f.write("{}")
        raw = {"fusionaware_src_dir": fw_src, "fusionaware_exec_dir": fw_exec}
        rt = _FakeRT()
        sys.modules["runtime"] = rt
        try:
            ident = M.build_identity(s, settings_path, [], raw)
        finally:
            sys.modules.pop("runtime", None)

        assert ident["repo"] == repo_no_analysis
        code = ident["code_sha256"]
        export_abs = os.path.abspath(M.__file__)
        contract_abs = os.path.join(_ANALYSIS, "expert_mask_contract.py")
        assert os.path.isfile(export_abs) and os.path.isfile(contract_abs)
        assert not export_abs.startswith(repo_no_analysis)
        assert code["export_hrplus_expert_masks.py"] == _sha(export_abs)
        assert code["expert_mask_contract.py"] == _sha(contract_abs)


def test_build_identity_hashes_survive_runtime_chdir():
    """The legacy runtime changes cwd after import; script hashes must not move."""
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        elsewhere = os.path.join(tmp, "legacy_repo")
        os.makedirs(elsewhere)
        try:
            os.chdir(elsewhere)
            export_abs = os.path.join(M._HERE, "export_hrplus_expert_masks.py")
            contract_abs = os.path.join(M._HERE, "expert_mask_contract.py")
            assert os.path.isfile(export_abs)
            assert os.path.isfile(contract_abs)
            with open(export_abs, "rb") as f:
                export_expected = hashlib.sha256(f.read()).hexdigest()
            with open(contract_abs, "rb") as f:
                contract_expected = hashlib.sha256(f.read()).hexdigest()
            assert M.C.sha256_file(export_abs) == export_expected
            assert M.C.sha256_file(contract_abs) == contract_expected
        finally:
            os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# 3) 静态结构：main 不再在解析前裸索引 settings['_cfg']
# ---------------------------------------------------------------------------

def test_main_calls_resolve_cfg_before_cfg_index():
    """源码级护栏：main 里第一次出现 settings["_cfg"] 之前必须先 resolve_cfg。"""
    path = os.path.abspath(M.__file__)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    main_start = next(i for i, ln in enumerate(lines) if ln.startswith("def main("))
    idx_resolve = idx_cfg = None
    for i in range(main_start, len(lines)):
        if idx_resolve is None and "resolve_cfg(settings" in lines[i]:
            idx_resolve = i
        if idx_cfg is None and 'settings["_cfg"]' in lines[i]:
            idx_cfg = i
        if idx_resolve is not None and idx_cfg is not None:
            break
    assert idx_cfg is not None and idx_resolve is not None, (idx_resolve, idx_cfg)
    assert idx_resolve < idx_cfg, (
        "main 必须在首次索引 settings['_cfg'] 之前调用 resolve_cfg"
        "（否则就是 smoke 的 KeyError 根因）")


def test_each_pipeline_writes_to_its_own_model_directory():
    """Regression: six pipelines must never overwrite one shared stem.npz."""
    with open(M._HERE + os.sep + "export_hrplus_expert_masks.py",
              "r", encoding="utf-8") as f:
        source = f.read()
    assert "os.path.join(model_dirs[name], C.MASK_FILENAME % stem)" in source
    assert "os.path.join(out_dir, C.MASK_FILENAME % stem)" not in source


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _all_tests():
    return [(k, v) for k, v in sorted(globals().items())
            if k.startswith("test_") and callable(v)]


def main():
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
    total = len(_all_tests())
    print("\n%d/%d passed" % (total - len(fails), total))
    if fails:
        print("FAILED: %s" % fails)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
