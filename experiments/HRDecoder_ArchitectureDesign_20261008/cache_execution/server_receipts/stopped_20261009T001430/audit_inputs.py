"""Verify data and exact archived teacher source before GPU export."""
import numpy as np
from PIL import Image
from cache_common import ROOT, DATA, read_json, write_json, sha, samples

def main():
    sources = read_json(ROOT / "metadata/source_manifest.json")
    for relative, expected in sources.items():
        assert sha(ROOT / relative) == expected, "Archived source mismatch: " + relative
    result = {"status": "PASS", "data_root": str(DATA), "sources_verified": len(sources), "splits": {}}
    for split, expected in [("train", 454), ("val", 76), ("test", 227)]:
        rows = read_json(ROOT / "metadata/data_identity.json")["splits"][split]["samples"]
        assert len(rows) == expected
        assert {p.stem for p in (DATA / "images" / split).glob("*.jpg")} == {row["sid"] for row in rows}
        pixels = 0
        for row in rows:
            image = DATA / "images" / split / (row["sid"] + ".jpg")
            label = DATA / "labels" / split / (row["sid"] + ".png")
            assert sha(image) == row["image_sha256"]
            assert sha(label) == row["label_sha256"]
            with Image.open(image) as opened: width, height = opened.size
            with Image.open(label) as opened: mask = np.asarray(opened)
            assert [height, width] == row["hw"] and list(mask.shape) == row["hw"]
            assert np.isin(mask, [0, 1, 2, 3, 4, 255]).all()
            pixels += height * width
        result["splits"][split] = {"images": expected, "pixels": pixels, "image_and_label_sha_verified": True}
    write_json(ROOT / "reports/input_audit.json", result)
    print(result, flush=True)

if __name__ == "__main__": main()
