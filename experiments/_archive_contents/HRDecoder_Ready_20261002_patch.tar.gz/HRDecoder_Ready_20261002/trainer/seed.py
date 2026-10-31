# -*- coding: utf-8 -*-
"""trainer.seed — 种子与确定性工具（同 seed 必须可重复；checkpoint 记录全部 RNG 态）。

torch 1.13（旧服务器栈 mmcv-full 1.7.2 依赖的 torch 版本）与本机 torch 2.x 都兼容：
只使用基础 API，不用 torch.compile / torch.func 等新接口。
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """固定 python/numpy/torch 种子；deterministic=True 时关闭 cudnn benchmark。"""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def capture_rng() -> dict:
    """捕获当前全部 RNG 状态，并归一化为 weights_only 兼容的纯原始类型。

    numpy/python 状态中的 ndarray/tuple 全部转为 list[int]（无 pickle 任意对象），
    使 checkpoint 可以用 torch.load(weights_only=True) 安全加载。
    """
    py_state = random.getstate()          # (version, tuple[int,...], gauss)
    np_state = np.random.get_state()      # ('MT19937', ndarray[624], pos, has_gauss, cached)
    state = {
        "python": [int(py_state[0]), [int(x) for x in py_state[1]],
                   py_state[2] if py_state[2] is None else float(py_state[2])],
        "numpy": [str(np_state[0]), [int(x) for x in np.asarray(np_state[1]).tolist()],
                  int(np_state[2]), int(np_state[3]), float(np_state[4])],
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    """恢复 RNG 状态（缺键时跳过，不伪造）。"""
    if "python" in state:
        version, ints, gauss = state["python"]
        random.setstate((int(version), tuple(int(x) for x in ints), gauss))
    if "numpy" in state:
        name, keys, pos, has_gauss, cached = state["numpy"]
        np.random.set_state((str(name), np.asarray(keys, dtype=np.uint32),
                             int(pos), int(has_gauss), float(cached)))
    if "torch" in state:
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in state["cuda"]])


def _as_byte_tensor(state):
    """兼容 CPU/GPU 序列化差异：还原为 uint8 ByteTensor。"""
    if torch.is_tensor(state):
        return state.cpu().to(torch.uint8)
    return torch.tensor(state, dtype=torch.uint8)


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker 种子：主进程已播种，torch 会自动派生；此处补 numpy。"""
    seed = torch.initial_seed() % (2**32 - 1)
    np.random.seed(seed)
    random.seed(seed)
