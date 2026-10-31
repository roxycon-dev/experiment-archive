"""Exact U0: native whole-image normalized padding, FP32 logits/sigmoid."""
import sys
import cv2
import numpy as np
from PIL import Image
import torch

from cache_common import ROOT, DATA, Writer, arguments, torch_runtime

def main():
    args = arguments("U")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(ROOT / "vendor/ready"))
    from trainer.models.convnext_unet import ConvNeXtUNet
    from trainer.eval import pad_normalized_image
    from trainer.data.common import normalize_image
    writer = Writer("U", {"input": "native whole image", "padding": "normalized zero to divisor32",
                          "normalization_mean": [123.675, 116.28, 103.53],
                          "normalization_std": [58.395, 57.12, 57.375],
                          "TTA": False, "inference_math": "FP32", "checkpoint": "U0 historical best 12000"},
                    torch_runtime(torch), args)
    if all(not writer.needed(row["sid"]) for row in writer.rows): writer.finish(); return
    model = ConvNeXtUNet()
    ck = torch.load(ROOT / "weights/U0_inference.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"], strict=True)
    del ck
    model = model.cuda().eval().requires_grad_(False)
    with torch.no_grad():
        for row in writer.rows:
            if not writer.needed(row["sid"]): continue
            with Image.open(DATA / "images" / args.split / (row["sid"] + ".jpg")) as im:
                rgb = np.asarray(im.convert("RGB"))
            h, w = rgb.shape[:2]
            assert [h, w] == row["hw"]
            chw = normalize_image(rgb, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]).transpose(2, 0, 1)
            padded, _ = pad_normalized_image(np.ascontiguousarray(chw))
            image = torch.from_numpy(padded[None]).cuda().float()
            logits = model.eval_forward(image)[..., :h, :w]
            probability = logits.float().sigmoid()[0].cpu().numpy()
            writer.save(row, probability)
    writer.finish()

if __name__ == "__main__": main()
