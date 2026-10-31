"""Native-coordinate train ROIs for cache inspection; no accuracy claim."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE/"real_cache_probe"
row = json.loads((ROOT/"probe_identity.json").read_text())
data_root = Path("D:/HRDecoder/data/DDR_split_613")
rgb = np.asarray(Image.open(data_root/"images/train"/(row["sid"]+".jpg")).convert("RGB"))
label = np.asarray(Image.open(data_root/"labels/train"/(row["sid"]+".png")))
values = []
for member in ("A", "U", "S"):
    with np.load(ROOT/"cache/train"/member/(row["sid"]+".npz")) as archive: values.append(archive["prob"])
a, u, s = values
p0 = .75*a+.25*u
t = .75*p0+.25*s
du, ds = u-a, s-p0
fig, axes = plt.subplots(4, 7, figsize=(18, 10), constrained_layout=True)
titles = ["RGB + GT", "A", "U0", "S20 EMA", "T", "dU = U-A", "dS = S-P0"]
for c, name in enumerate(("EX", "HE", "SE", "MA")):
    mask = (label == c+1).astype(np.uint8)
    count, components, stats, centers = cv2.connectedComponentsWithStats(mask)
    if count > 1:
        selected = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y = (int(round(value)) for value in centers[selected])
    else: y, x = label.shape[0]//2, label.shape[1]//2
    radius = 128
    y1 = min(max(y-radius, 0), label.shape[0]-2*radius)
    x1 = min(max(x-radius, 0), label.shape[1]-2*radius)
    crop = np.s_[y1:y1+2*radius, x1:x1+2*radius]
    axes[c, 0].imshow(rgb[crop])
    roi_mask = mask[crop]
    if roi_mask.any(): axes[c, 0].contour(roi_mask, levels=[.5], colors=["cyan"], linewidths=.6)
    axes[c, 0].set_ylabel(name+("\nno GT" if count <= 1 else ""), fontsize=13)
    for j, value in enumerate((a, u, s, t, du, ds), 1):
        signed = j >= 5
        axes[c, j].imshow(value[c][crop], cmap="RdBu_r" if signed else "magma",
            vmin=-1 if signed else 0, vmax=1, interpolation="nearest")
    for j, title in enumerate(titles):
        axes[c, j].set_xticks([]); axes[c, j].set_yticks([])
        if c == 0: axes[c, j].set_title(title)
fig.suptitle("Train cache "+row["sid"]+": native ROIs at largest GT component, or image center if GT absent\n"
    "Probability scale [0,1]; signed scale [-1,1]. Inspection only; no performance result.", fontsize=13)
output = HERE/"train_cache_diagnostics.png"
fig.savefig(output, dpi=150); plt.close(fig)
print(str(output))
