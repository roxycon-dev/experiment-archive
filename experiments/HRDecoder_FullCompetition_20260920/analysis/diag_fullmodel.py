#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""diag_fullmodel — 打印 U0/H0 整模评测路径的真实 shape，并与 canonical
WholeImageEvaluator 对齐，定位 export_fullmodel_masks 的形状口径分歧。

只读、只打印，不写任何 outputs，不读 test。
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
for p in (_PROJ, os.path.join(_PROJ, "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import expert_mask_contract as C  # noqa: E402
import export_fullmodel_masks as E  # noqa: E402


def main():
    settings = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _PROJ, "analysis", "expert_eval_settings.json")
    models = sys.argv[2:] or ["U0"]
    raw = E.load_raw_settings(settings)
    specs = E.resolve_model_specs(raw, models)
    ddr_root = C.safe_path(raw["ddr_root"], "ddr_root")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from trainer.data.common import crop_logits_to_orig, normalize_image
    from trainer.eval import WholeImageEvaluator, pad_normalized_image

    for spec in specs:
        print("=== %s ===" % spec["name"])
        model, ds, ctx = E.build_one(spec, ddr_root=ddr_root, raw=raw,
                                     device=device, torch=torch)
        img_u8, target, valid, stem = ds.load_raw(0)
        print("  stem=%s img_u8=%s target=%s valid=%s" %
              (stem, img_u8.shape, target.shape, valid.shape))
        chw = normalize_image(img_u8, ctx["mean"], ctx["std"]).transpose(2, 0, 1)
        padded, (oh, ow) = pad_normalized_image(np.ascontiguousarray(chw))
        print("  padded=%s orig_hw=(%d,%d)" % (padded.shape, oh, ow))
        x = torch.from_numpy(padded[None]).to(device).float()
        with torch.no_grad():
            logits = (model.eval_forward(x) if ctx["eval_mode"] == "whole_image"
                      else None)
        print("  logits(raw)=" + (str(None) if logits is None else str(tuple(logits.shape))))
        if logits is not None:
            cropped = crop_logits_to_orig(logits, (oh, ow))
            print("  logits(crop)=" + str(tuple(cropped.shape)))
            probs4 = torch.sigmoid(cropped.float()).float().cpu().numpy()
            print("  probs(no-squeeze)=" + str(probs4.shape))
            print("  target[None]=" + str(target[None].shape)
                  + " valid[None]=" + str(valid[None].shape))
            from trainer.metrics import SegMetricsAccumulator
            for label, pr in (("no-squeeze", probs4),
                              ("squeezed[0]", probs4[0])):
                acc = SegMetricsAccumulator(threshold=0.5)
                try:
                    acc.update(pr, target[None], valid[None])
                    print("  metrics.update(%s) OK  (probs=%s)" % (label, pr.shape))
                except Exception as e:                          # noqa: BLE001
                    print("  metrics.update(%s) FAILED: %r  (probs=%s)"
                          % (label, e, pr.shape))
                try:
                    acc2 = SegMetricsAccumulator(threshold=0.5)
                    acc2.update(pr, target, valid)
                    print("  metrics.update(%s, target-as-is) OK" % label)
                except Exception as e:                          # noqa: BLE001
                    print("  metrics.update(%s, target-as-is) FAILED: %r" % (label, e))
        # canonical 评测器（trainer 自己的整图路径），max_images=1
        try:
            ev = WholeImageEvaluator(model, ds, device=device,
                                     mean=ctx["mean"], std=ctx["std"],
                                     mode="whole_image", threshold=0.5,
                                     max_images=1)
            rep = ev.run()
            print("  canonical WholeImageEvaluator OK: mIoU=%.12f n=%s"
                  % (rep["mIoU_lesion4"], rep.get("eval_images")))
        except Exception as e:                                  # noqa: BLE001
            print("  canonical WholeImageEvaluator FAILED: %r" % (e,))
        del model, ds
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
