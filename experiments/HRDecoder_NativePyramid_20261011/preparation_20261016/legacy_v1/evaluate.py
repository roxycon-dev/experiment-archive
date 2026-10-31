"""One evaluation of a hash-locked 40k checkpoint after passing the val target."""
import argparse
import json
from pathlib import Path
import torch

from data import SegmentationData
from model import NativePyramid, file_sha256
from train import evaluate, source_identity, write_json


def validate_lock(lock, digest):
    if lock.get("completed") != 40000 or lock.get("primary") != "fixed_40000_raw_weights":
        raise ValueError("Only the fixed 40k endpoint is eligible")
    if lock.get("source_sha256") != digest:
        raise ValueError("Source has changed after training")
    if lock.get("status") != "VAL_TARGET_PASSED" or lock.get("test_permitted") is not True:
        raise ValueError("Validation target has not been passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--identity", required=True); parser.add_argument("--pretrained", required=True)
    args = parser.parse_args()
    path = Path(args.lock); out = path.parent
    digest, _ = source_identity()
    lock = json.loads(path.read_text()); validate_lock(lock, digest)
    if (out / "test_result.json").exists() or (out / "test_started.json").exists():
        raise FileExistsError("Test was already started; inspect its existing record")
    if file_sha256(lock["checkpoint"]) != lock["checkpoint_sha256"]:
        raise ValueError("Locked checkpoint changed")
    state = torch.load(lock["checkpoint"], map_location="cpu", weights_only=False)
    if (state["completed"] != 40000 or state["status"] != "COMPLETE" or
            state["identity_sha256"] != file_sha256(args.identity)):
        raise ValueError("Checkpoint budget, status or dataset mismatch")
    torch.set_num_threads(4)
    model = NativePyramid(args.pretrained).cuda().eval()
    model.load_state_dict(state["model"], strict=True); del state
    write_json(out / "test_started.json", {"checkpoint_sha256": lock["checkpoint_sha256"],
                                           "source_sha256": digest})
    dataset = SegmentationData(args.data_root, args.identity, "test", 2048, allow_test=True)
    result = evaluate(model, dataset)
    protocol = json.loads((Path(__file__).parent / "protocol.json").read_text())
    result.update(checkpoint_sha256=lock["checkpoint_sha256"], completed=40000,
                  target_achieved=result["mIoU_percent"] > protocol["test_target_percent"],
                  val_target_passed=True)
    write_json(out / "test_result.json", result)
    write_json(out / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
