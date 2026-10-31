"""Exact archived A: D/E/B/E4 3:2:1:1, horizontal TTA, original H4 MA head."""
import copy
import os
import sys
import time

import torch

from cache_common import ROOT, DATA, Writer, arguments, torch_runtime

def main():
    args = arguments("A")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(ROOT / "vendor/legacy"))
    import runtime
    from cache_common import read_json
    original = read_json(ROOT / "metadata/original_champion_settings.json")
    settings = copy.deepcopy(original)
    settings["repo"] = str(ROOT / "vendor/champion_repo")
    settings["data_root"] = str(DATA)
    settings["reference_checkpoint"] = str(ROOT / "weights/B_weights.pth")
    settings["champion"]["head"] = str(ROOT / "weights/H4_best.pth")
    for member in settings["champion"]["members"]:
        name = member["checkpoint"].split("/")[-1]
        member["checkpoint"] = str(ROOT / "weights" / name)
    writer = Writer("A", {
        "member_weights": {"D": 3, "E": 2, "B": 1, "E4": 1},
        "horizontal_flip_TTA": True, "H4": "original MA correction",
        "inference_math": "FP32", "original_settings_sha256": __import__("cache_common").sha(ROOT / "metadata/original_champion_settings.json"),
    }, torch_runtime(torch), args)
    if all(not writer.needed(row["sid"]) for row in writer.rows): writer.finish(); return
    runtime.inject_repo(settings["repo"])
    cfg = runtime.load_config(settings["repo"], settings["config"])
    from mmseg.datasets import build_dataset
    data = copy.deepcopy(cfg.data["val"])
    data.data_root = str(DATA)
    data.img_dir = "images/" + args.split
    data.ann_dir = "labels/" + args.split
    data.pipeline = copy.deepcopy(cfg.data["test"]["pipeline"])
    data.test_mode = True
    if "split" in data: data.split = None
    dataset = build_dataset(data)
    lookup = {}
    for i, info in enumerate(dataset.img_infos):
        from pathlib import Path
        lookup[Path(info["filename"]).stem] = i
    assert set(lookup) == {row["sid"] for row in __import__("cache_common").samples(args.split)}
    champion = runtime.ChampionEnsemble(settings, settings["champion"], device="cuda")
    with torch.no_grad():
        for row in writer.rows:
            if not writer.needed(row["sid"]): continue
            item = dataset[lookup[row["sid"]]]
            image = runtime._unwrap_img(item["img"]).unsqueeze(0).cuda()
            meta = runtime._unwrap_meta(item["img_metas"])
            assert list(meta[0]["ori_shape"][:2]) == row["hw"]
            probability = champion.predict(image, meta).float().cpu().numpy()
            writer.save(row, probability)
    writer.finish()

if __name__ == "__main__": main()
