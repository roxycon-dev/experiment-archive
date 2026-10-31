"""Original image augmentation, bilinear soft-map replay, two nearest masks."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import mmcv
from mmseg.datasets.pipelines.transforms import Resize, RandomCrop, RandomFlip, RandomRotate, Normalize, Pad

MEAN = [81.205, 50.636, 21.216]
STD = [76.252, 48.798, 21.625]

def map_channels(value, operation):
    return np.stack([operation(channel) for channel in value], axis=0).astype(np.float32, copy=False)

class SoftResize(Resize):
    def __call__(self, results):
        results = super().__call__(results)
        h, w = results["img"].shape[:2]
        results["v_geo"] = mmcv.imresize(results["v_geo"], (w, h), interpolation="bilinear")
        for member, probability in results["probabilities"].items():
            results["probabilities"][member] = map_channels(probability,
                lambda channel: mmcv.imresize(channel, (w, h), interpolation="bilinear"))
        results["geometry"].append({"resize_hw": [h, w], "scale": list(results["scale"])})
        return results

class SoftCrop(RandomCrop):
    def get_crop_bbox(self, image):
        self.last_bbox = super().get_crop_bbox(image)
        return self.last_bbox

    def __call__(self, results):
        results = super().__call__(results)
        y1, y2, x1, x2 = self.last_bbox
        results["v_geo"] = results["v_geo"][y1:y2, x1:x2].copy()
        for member, probability in results["probabilities"].items():
            results["probabilities"][member] = probability[:, y1:y2, x1:x2].copy()
        results["geometry"].append({"crop_y1_y2_x1_x2": list(self.last_bbox)})
        return results

class SoftFlip(RandomFlip):
    def __call__(self, results):
        results = super().__call__(results)
        if results["flip"]:
            assert results["flip_direction"] == "horizontal"
            results["v_geo"] = results["v_geo"][:, ::-1].copy()
            for member, probability in results["probabilities"].items():
                results["probabilities"][member] = probability[:, :, ::-1].copy()
        results["geometry"].append({"horizontal_flip": bool(results["flip"])})
        return results

class SoftRotate(RandomRotate):
    def __call__(self, results):
        # Exactly the two random draws in the original RandomRotate.__call__.
        rotate = np.random.rand() < self.prob
        degree = np.random.uniform(min(*self.degree), max(*self.degree))
        if rotate:
            options = dict(angle=degree, center=self.center, auto_bound=self.auto_bound)
            results["img"] = mmcv.imrotate(results["img"], border_value=self.pal_val, **options)
            for key in results["seg_fields"]:
                results[key] = mmcv.imrotate(results[key], border_value=self.seg_pad_val,
                    interpolation="nearest", **options)
            results["v_geo"] = mmcv.imrotate(results["v_geo"], border_value=0,
                interpolation="bilinear", **options)
            for member, probability in results["probabilities"].items():
                results["probabilities"][member] = map_channels(probability,
                    lambda channel: mmcv.imrotate(channel, border_value=0,
                        interpolation="bilinear", **options))
        results["geometry"].append({"rotate": bool(rotate), "degrees": float(degree)})
        return results

class SoftPad(Pad):
    def __call__(self, results):
        results = super().__call__(results)
        h, w = results["img"].shape[:2]
        gh, gw = results["v_geo"].shape
        results["v_geo"] = np.pad(results["v_geo"], ((0, h-gh), (0, w-gw)), mode="constant")
        for member, probability in results["probabilities"].items():
            ph, pw = probability.shape[-2:]
            results["probabilities"][member] = np.pad(probability,
                ((0, 0), (0, h-ph), (0, w-pw)), mode="constant")
        return results

def training_pipeline(canvas=2048):
    return [SoftResize(img_scale=(canvas, canvas), ratio_range=(.5, 2.), keep_ratio=True),
        SoftCrop(crop_size=(canvas, canvas), cat_max_ratio=.75),
        SoftFlip(flip_ratio=.5, direction="horizontal"),
        SoftRotate(prob=1., degree=(-45, 45), pad_val=0, seg_pad_val=0, auto_bound=False),
        Normalize(mean=MEAN, std=STD, to_rgb=True),
        SoftPad(size=(canvas, canvas), pad_val=0, seg_pad_val=0)]

def augment(image_bgr, label, probabilities, pipeline):
    results = {"img": image_bgr, "img_shape": image_bgr.shape, "ori_shape": image_bgr.shape,
        "gt_semantic_seg": label, "v_label": (label != 255).astype(np.uint8),
        "v_geo": np.ones(label.shape, dtype=np.float32),
        "seg_fields": ["gt_semantic_seg", "v_label"],
        "probabilities": probabilities, "geometry": []}
    for transform in pipeline: results = transform(results)
    # Reject targets whose bilinear footprint contains any out-of-source fill.
    valid = ((results["v_label"] > 0) & (results["v_geo"] >= 1-1e-6)).astype(np.float32)
    tensors = {"image": torch.from_numpy(np.ascontiguousarray(results["img"].transpose(2, 0, 1))),
        "label": torch.from_numpy(results["gt_semantic_seg"].astype(np.int64)).unsqueeze(0),
        "valid": torch.from_numpy(valid).unsqueeze(0)}
    for member, probability in results["probabilities"].items():
        assert probability.dtype == np.float32 and np.isfinite(probability).all()
        assert probability.min() >= 0 and probability.max() <= 1
        tensors[member] = torch.from_numpy(np.ascontiguousarray(probability))
    return tensors, results

class TrainCacheDataset(Dataset):
    def __init__(self, root, data_root, *, canvas=2048, with_teachers=True):
        self.root = Path(root); self.data_root = Path(data_root)
        identity = json.loads((self.root/"metadata/data_identity.json").read_text())
        self.rows = identity["splits"]["train"]["samples"]
        assert len(self.rows) == 454
        self.images = {}
        for path in (self.data_root/"images/train").iterdir():
            if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                assert path.stem not in self.images
                self.images[path.stem] = path
        assert set(self.images) == {row["sid"] for row in self.rows}
        self.indexes = {}
        for member in ("A", "U", "S"):
            index = json.loads((self.root/"cache/train"/member/"index.json").read_text())
            assert index["status"] == "COMPLETE" and index["completed_images"] == 454
            self.indexes[member] = index
        self.pipeline = training_pipeline(canvas)
        self.with_teachers = with_teachers

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        sample_seed = None
        if isinstance(index, tuple): index, sample_seed = index
        row = self.rows[index]; sid = row["sid"]
        image = mmcv.imread(str(self.images[sid]), flag="color")
        label = mmcv.imread(str(self.data_root/"labels/train"/(sid+".png")), flag="unchanged")
        assert list(image.shape[:2]) == row["hw"] and label.shape == image.shape[:2]
        probabilities = {}
        for member, manifest in (self.indexes.items() if self.with_teachers else []):
            entry = manifest["entries"][sid]
            with np.load(self.root/"cache/train"/member/entry["file"], allow_pickle=False) as archive:
                probability = archive["prob"]
            assert probability.dtype == np.float32 and list(probability.shape) == [4]+row["hw"]
            probabilities[member] = probability
        state = np.random.get_state()
        try:
            if sample_seed is not None: np.random.seed(sample_seed)
            batch, _ = augment(image, label, probabilities, self.pipeline)
        finally:
            if sample_seed is not None: np.random.set_state(state)
        batch["sid"] = sid
        return batch

def evaluation_image(path, canvas=2048):
    image = mmcv.imread(str(path), flag="color")
    image = mmcv.imresize(image, (canvas, canvas), interpolation="bilinear")
    image = mmcv.imnormalize(image, np.array(MEAN, np.float32), np.array(STD, np.float32), to_rgb=True)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
