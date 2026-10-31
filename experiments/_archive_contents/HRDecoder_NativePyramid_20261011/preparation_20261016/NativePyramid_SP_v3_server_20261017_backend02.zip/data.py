"""The archived 454/76/227 split; deterministic per-sample augmentation.

Only train and val can be opened by the training entry point. The separate
locked evaluator is the sole entry point that requests test labels.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

MEAN = np.array([81.205, 50.636, 21.216], np.float32)
STD = np.array([76.252, 48.798, 21.625], np.float32)
ASSIGNMENT_SHA = "79181851e8026125bf900b6d1129ca7946b6cda56dfdb5708b27364b1173ea9f"
IDENTITY_SHA = "7c53b86bcb27ccd121042f667cc58e39b98716a8c477bfd1f1657fa126575719"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_identity(path):
    if sha256_file(path) != IDENTITY_SHA:
        raise ValueError("Data identity bytes differ from the accepted fused-label manifest")
    identity = json.loads(Path(path).read_text(encoding="utf-8"))
    if identity["assignment_sha256"] != ASSIGNMENT_SHA:
        raise ValueError("Different data split; this run cannot use the archived fusion target")
    if identity["classes"] != ["EX", "HE", "SE", "MA"]:
        raise ValueError("Unexpected class order")
    seen_ids, seen_hashes = set(), set()
    for split, expected in (("train", 454), ("val", 76), ("test", 227)):
        rows = identity["splits"][split]["samples"]
        sids = {r["sid"] for r in rows}
        hashes = {r["image_sha256"] for r in rows}
        if len(rows) != expected or len(sids) != expected or len(hashes) != expected:
            raise ValueError(f"Duplicate or missing identity in {split}")
        if seen_ids & sids or seen_hashes & hashes:
            raise ValueError("Cross-split image identity overlap")
        seen_ids.update(sids); seen_hashes.update(hashes)
    return identity


def image_index(directory):
    paths = [p for p in Path(directory).iterdir()
             if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    images = {p.stem: p for p in paths}
    if len(images) != len(paths):
        raise ValueError("Duplicate image stems with different extensions")
    return images


def verify_split_files(root, identity_path, splits=("train", "val"), *, allow_test=False):
    """Verify actual pair bytes. Test is permitted only by the locked evaluator."""
    if any(split not in ("train", "val", "test") for split in splits):
        raise ValueError("Unknown split")
    if "test" in splits and not allow_test:
        raise PermissionError("Test bytes require the locked evaluation entry point")
    identity = load_identity(identity_path)
    checks = {}
    for split in splits:
        directory = Path(root)
        images = image_index(directory / "images" / split)
        expected = {r["sid"] for r in identity["splits"][split]["samples"]}
        labels = {p.stem for p in (directory / "labels" / split).glob("*.png")}
        if set(images) != expected or labels != expected:
            raise ValueError(f"Image/label directory differs from {split} identity")
        for row in identity["splits"][split]["samples"]:
            for path, key in ((images[row["sid"]], "image_sha256"),
                              (directory / "labels" / split / (row["sid"] + ".png"), "label_sha256")):
                if sha256_file(path) != row[key]:
                    raise ValueError(f"Changed data content: {split}/{row['sid']}/{key}")
        checks[split] = {"images_and_labels_sha256_verified": len(expected)}
    return checks


def verify_training_files(root, identity_path):
    return verify_split_files(root, identity_path)


def read_image(path):
    array = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        raise OSError(f"Cannot read {path}")
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def read_label(path):
    array = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if array is None or array.ndim != 2:
        raise OSError(f"Cannot read indexed mask {path}")
    if not np.isin(array, [0, 1, 2, 3, 4, 255]).all():
        raise ValueError(f"Incorrect mask encoding: {path}")
    return array


def normalize(image):
    return torch.from_numpy(np.ascontiguousarray(((image.astype(np.float32) - MEAN) / STD)
                                                .transpose(2, 0, 1)))


def augment(image, label, canvas, seed):
    rng = np.random.RandomState(seed)
    h, w = label.shape
    factor = canvas * rng.uniform(0.5, 2.0) / max(h, w)
    nh, nw = max(1, int(h * factor + 0.5)), max(1, int(w * factor + 0.5))
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST)
    for _ in range(10):
        y = int(rng.randint(max(nh - canvas, 0) + 1))
        x = int(rng.randint(max(nw - canvas, 0) + 1))
        crop_label = label[y:y + canvas, x:x + canvas]
        _, counts = np.unique(crop_label[crop_label != 255], return_counts=True)
        if len(counts) > 1 and counts.max() / counts.sum() < 0.75:
            break
    image = image[y:y + canvas, x:x + canvas].copy()
    label = crop_label.copy()
    if rng.rand() < 0.5:
        image = image[:, ::-1].copy(); label = label[:, ::-1].copy()
    # One affine matrix for RGB and labels. Ignore synthetic border pixels.
    h, w = label.shape
    angle = float(rng.uniform(-45, 45))
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), -angle, 1.0)
    support = np.ones((h, w), np.float32)
    image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
    support = cv2.warpAffine(support, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    label[support < 1 - 1e-6] = 255
    tensor = normalize(image)
    padded = torch.zeros(3, canvas, canvas)
    padded[:, :h, :w] = tensor
    mask = torch.full((canvas, canvas), 255, dtype=torch.long)
    mask[:h, :w] = torch.from_numpy(label.astype(np.int64))
    return padded, mask


class SegmentationData(Dataset):
    def __init__(self, data_root, identity_path, split="train", canvas=2048, allow_test=False):
        if split == "test" and not allow_test:
            raise PermissionError("Test requires the locked evaluation entry point")
        self.root, self.split, self.canvas = Path(data_root), split, canvas
        self.identity = load_identity(identity_path)
        self.rows = self.identity["splits"][split]["samples"]
        self.images = image_index(self.root / "images" / split)
        if set(self.images) != {r["sid"] for r in self.rows}:
            raise ValueError(f"Image directory differs from {split} identity")

    def __len__(self):
        return len(self.rows)

    def native(self, index):
        row = self.rows[index]
        image = read_image(self.images[row["sid"]])
        label = read_label(self.root / "labels" / self.split / (row["sid"] + ".png"))
        if list(label.shape) != row["hw"] or image.shape[:2] != label.shape:
            raise ValueError("Image/label geometry differs from identity")
        return image, label, row

    def evaluation(self, index):
        image, label, row = self.native(index)
        image = cv2.resize(image, (self.canvas, self.canvas), interpolation=cv2.INTER_LINEAR)
        return normalize(image), label, row

    def __getitem__(self, item):
        index, seed = item
        image, label, row = self.native(index)
        image, label = augment(image, label, self.canvas, seed)
        return {"image": image, "label": label, "sid": row["sid"]}


class ResumableBatches(Sampler):
    def __init__(self, n, start, stop, seed=42):
        if n % 2:
            raise ValueError("Batch2 sampler requires an even train size")
        self.n, self.start, self.stop, self.seed = n, start, stop, seed

    def __len__(self):
        return self.stop - self.start

    def __iter__(self):
        previous_epoch = None
        for update in range(self.start, self.stop):
            epoch = update * 2 // self.n
            if epoch != previous_epoch:
                order = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
                previous_epoch = epoch
            offset = update * 2 % self.n
            yield [(order[offset + j], self.seed + update * 2 + j) for j in range(2)]


def worker_setup(_):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)

