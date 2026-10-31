"""Exact S20 EMA: native1024/center768/halo128, pinned MSDA CUDA, BF16 + FP32 sigmoid."""
import sys
import numpy as np
from PIL import Image
import torch

from cache_common import ROOT, DATA, Writer, arguments, torch_runtime, write_json

def main():
    args = arguments("S")
    torch.set_num_threads(4)
    sys.path.insert(0, str(ROOT / "vendor/msda_build"))
    sys.path.insert(0, str(ROOT / "vendor/final"))
    sys.path.insert(0, str(ROOT / "vendor/dinov3"))
    import MultiScaleDeformableAttention
    from dinov3.hub.backbones import dinov3_vitb16
    from qlib.model import Q
    from final_model import FinalModel
    from qlib.geometry import tiles, rgb_window, center_logits
    writer = Writer("S", {"checkpoint": "S seed42 update20000 EMA",
                          "window": 1024, "center": 768, "halo": 128,
                          "input": "native original RGB", "normalization": "ImageNet RGB",
                          "forward_dtype": "bfloat16", "sigmoid_and_output": "float32",
                          "attention": "pinned original MSDA CUDA kernel", "TTA": False}, torch_runtime(torch), args)
    if all(not writer.needed(row["sid"]) for row in writer.rows): writer.finish(); return
    # Every tensor, including the official backbone, is strictly replaced by the
    # locked complete S20 EMA state before any inference. No parent optimizer loaded.
    backbone = dinov3_vitb16(pretrained=False)
    model = FinalModel(Q(backbone), "S", 42, backend="cuda", checkpoint_blocks=False)
    saved = torch.load(ROOT / "weights/S20_ema_inference.pth", map_location="cpu", weights_only=False)
    assert saved["source_completed"] == saved["source_ema_completed"] == 20000
    model.load_state_dict(saved["ema"], strict=True)
    del saved
    model = model.cuda().eval().requires_grad_(False)
    with torch.no_grad():
        for row in writer.rows:
            if not writer.needed(row["sid"]): continue
            with Image.open(DATA / "images" / args.split / (row["sid"] + ".jpg")) as im:
                rgb = np.asarray(im.convert("RGB"))
            h, w = rgb.shape[:2]
            assert [h, w] == row["hw"]
            probability = np.empty((4, h, w), np.float32)
            for tile in tiles(h, w):
                image = torch.from_numpy(rgb_window(rgb, tile)).unsqueeze(0).cuda()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = center_logits(model(image))
                probability[:, tile.y:tile.y + tile.height, tile.x:tile.x + tile.width] = (
                    logits.float().sigmoid()[0, :, :tile.height, :tile.width].cpu().numpy())
            if args.split == "val" and row["sid"] == "007-1774-100":
                with np.load(ROOT / "references/S20_007-1774-100.npz", allow_pickle=False) as reference:
                    old = reference["prob"]
                delta = np.abs(probability - old)
                parity = {"sid": row["sid"], "bitwise_equal": bool(np.array_equal(probability, old)),
                          "max_abs_difference": float(delta.max()), "mean_abs_difference": float(delta.mean()),
                          "threshold_disagree_pixels": int(((probability > .5) != (old > .5)).sum()),
                          "reference_coordinates": "original image", "reference": "archived S20 EMA val cache"}
                write_json(ROOT / "reports/S20_canary_parity.json", parity)
                assert parity["max_abs_difference"] <= 1e-5, "S20 exporter differs from locked historical inference"
            writer.save(row, probability)
    writer.finish()

if __name__ == "__main__": main()
