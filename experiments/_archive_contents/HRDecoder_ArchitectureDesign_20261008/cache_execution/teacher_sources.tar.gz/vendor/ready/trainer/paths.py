# -*- coding: utf-8 -*-
"""trainer.paths — 落盘路径安全工具（所有写文件操作统一走这里）。

规则：
- 禁止路径中出现 `..` 分量（拒绝路径穿越）；
- 禁止空路径 / 纯空白；
- 输出路径一律先 normpath 规范化再使用；
- 相对路径以显式 base 目录解析（默认工作区根），不允许逃出 base。

本模块只做校验与规范化，不做任何 IO。
"""

from __future__ import annotations

import os

from .gate import GateError


def validate_output_path(path: str, base_dir: str | None = None) -> str:
    """校验并规范化一个用于写入的路径，返回绝对规范化路径。

    Args:
        path: 目标路径（绝对或相对）。
        base_dir: 相对路径的解析基准；None 时用工作区根目录
            （不用 os.getcwd()：入口脚本从任意 cwd 启动时落盘位置必须一致）。

    Raises:
        GateError: 路径为空、含 `..` 分量、或含非法控制字符。
    """
    if not isinstance(path, str) or not path.strip():
        raise GateError(f"非法输出路径（空或非字符串）: {path!r}")
    if "\x00" in path:
        raise GateError("输出路径包含 NUL 字节")
    p = os.path.expanduser(path)
    base = os.path.abspath(base_dir) if base_dir else repo_root()
    is_abs = os.path.isabs(p)
    if not is_abs:
        p = os.path.join(base, p)
    norm = os.path.normpath(p)
    if ".." in norm.replace("\\", "/").split("/"):
        raise GateError(f"输出路径包含 .. 分量（拒绝路径穿越）: {path!r}")
    if not is_abs and os.path.commonpath([norm, os.path.normpath(base)]) != os.path.normpath(base):
        raise GateError(f"相对输出路径逃出基准目录: {path!r}")
    return norm


def repo_root() -> str:
    """工作区根目录（trainer/ 的上一级）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_parent_dir(path: str, base_dir: str | None = None) -> str:
    """创建父目录（存在则跳过），返回规范化路径。"""
    p = validate_output_path(path, base_dir=base_dir)
    parent = os.path.dirname(p)
    if ".." in parent.replace("\\", "/").split("/"):
        raise GateError(f"父目录含 .. 分量: {parent}")
    os.makedirs(parent, exist_ok=True)
    return p


def ensure_dir(path: str, base_dir: str | None = None) -> str:
    """创建目录本身并返回规范化路径（同样走穿越校验）。"""
    p = validate_output_path(path, base_dir=base_dir)
    os.makedirs(p, exist_ok=True)
    return p
