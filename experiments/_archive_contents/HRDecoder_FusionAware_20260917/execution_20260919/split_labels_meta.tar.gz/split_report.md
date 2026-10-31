# DDR 6:1:3 re-split report (stratified by SE, seed=42)

Custom experimental split (not the original paper split). All 757 DDR images merged and re-split train:val:test = 454:76:227.

| split | n | tierA | tierB | tierC |
|---|---|---|---|---|
| train | 454 | 311 | 71 | 72 |
| val | 76 | 52 | 12 | 12 |
| test | 227 | 155 | 36 | 36 |
| all | 757 | 518 | 119 | 120 |

| split | class | pos_images | image_pct | pixel_pct |
|---|---|---|---|---|
| train | EX | 303 | 66.74 | 0.2466 |
| train | HE | 366 | 80.62 | 0.3781 |
| train | SE | 143 | 31.50 | 0.0566 |
| train | MA | 338 | 74.45 | 0.0225 |
| val | EX | 46 | 60.53 | 0.0728 |
| val | HE | 63 | 82.89 | 0.3011 |
| val | SE | 24 | 31.58 | 0.0502 |
| val | MA | 59 | 77.63 | 0.0173 |
| test | EX | 137 | 60.35 | 0.1337 |
| test | HE | 172 | 75.77 | 0.2580 |
| test | SE | 72 | 31.72 | 0.0482 |
| test | MA | 173 | 76.21 | 0.0160 |
| all | EX | 486 | 64.20 | 0.1949 |
| all | HE | 601 | 79.39 | 0.3341 |
| all | SE | 239 | 31.57 | 0.0534 |
| all | MA | 570 | 75.30 | 0.0200 |

Note: image_pct = share of images containing the lesion; pixel_pct = lesion pixels / total pixels over the split.
