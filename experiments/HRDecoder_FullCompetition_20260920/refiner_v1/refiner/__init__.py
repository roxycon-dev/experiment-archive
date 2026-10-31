#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HRDecoder MA/HE candidate patch refiner (refiner_v1).

Package layout
--------------
  candidate.py  frozen candidate rule + fixed tile stitching (no dnn)
  haar.py       fixed, invertible Haar high-frequency features (no params)
  model.py      HaarHeteroRefiner (25,394 trainable params)
  losses.py     focal BCE + soft Dice, two terms only
  dataset.py    patch-cache reader + frozen sampling
  assemble.py   candidate-only write-back with fixed weighted averaging
  reference.py  frozen champion probability provider (never trained)
  guards.py     no-test / frozen-reference / grad-isolation guards

The package deliberately owns no training or evaluation loop: those live in
`train_refiner.py` / `eval_refiner.py` so that the library can be unit-tested
on CPU without a GPU or the champion checkpoint.
"""
from . import assemble, candidate, dataset, guards, haar, losses, model, reference  # noqa: F401

__all__ = ["assemble", "candidate", "dataset", "guards", "haar", "losses",
           "model", "reference"]
