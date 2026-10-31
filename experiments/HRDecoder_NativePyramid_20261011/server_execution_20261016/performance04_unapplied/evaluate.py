"""One evaluation of a hash-locked 40k checkpoint after passing the val target."""
import argparse
import json
import math
import os
from pathlib import Path
import torch

from data import SegmentationData, verify_split_files
from model import NativePyramid, file_sha256
from train import evaluate, source_identity, write_json, runtime_identity, configure_runtime, execution_model


def validate_lock(lock, digest):
    if lock.get("completed") != 40000 or lock.get("primary") != "fixed_40000_raw_weights":
        raise ValueError("Only the fixed 40k endpoint is eligible")
    if lock.get("source_sha256") != digest:
        raise ValueError("Source has changed after training")
    if lock.get("status") != "VAL_TARGET_PASSED" or lock.get("test_permitted") is not True:
        raise ValueError("Validation target has not been passed")
    if lock.get("val", {}).get("update") != 40000:
        raise ValueError("Missing fixed-endpoint validation")
    value = lock.get("val", {}).get("mIoU_percent", -1)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 45.321831227388 <= value <= 100:
        raise ValueError("Validation value does not meet the locked target")


def validate_checkpoint_lock(lock, state, identity_sha256, source_sha256):
    validate_lock(lock, source_sha256)
    if any(state.get(key) != 40000 for key in ("completed", "executed_updates", "checkpoint_persisted_updates")):
        raise ValueError("Checkpoint budget counters must all be exactly 40000")
    if any(lock.get(key) != 40000 for key in ("executed_updates", "checkpoint_persisted_updates")):
        raise ValueError("Endpoint lock counters do not match the complete checkpoint")
    if (state.get("status") != "COMPLETE" or state.get("primary") != "fixed_40000_raw_weights"
            or state.get("identity_sha256") != identity_sha256 or state.get("source_sha256") != source_sha256):
        raise ValueError("Checkpoint status, primary identity, dataset or source mismatch")
    if not isinstance(state.get("run_identity"), dict) or state["run_identity"] != lock.get("run_identity"):
        raise ValueError("Endpoint lock belongs to another run")
    history = state.get("validation_history", [])
    if not history or history[-1] != lock.get("val"):
        raise ValueError("Endpoint validation is not the one persisted in this checkpoint")
    if state["run_identity"].get("identity_sha256") != identity_sha256:
        raise ValueError("Checkpoint run identity disagrees with its data metadata")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--identity", required=True); parser.add_argument("--pretrained", required=True)
    args = parser.parse_args()
    configure_runtime()
    path = Path(args.lock); out = path.parent
    digest, _ = source_identity()
    lock = json.loads(path.read_text(encoding="utf-8")); validate_lock(lock, digest)
    if (out / "failure.json").exists():
        raise RuntimeError("Terminal failure marker prevents test evaluation")
    if (out / "test_result.json").exists() or (out / "test_started.json").exists():
        raise FileExistsError("Test was already started; inspect its existing record")
    if file_sha256(lock["checkpoint"]) != lock["checkpoint_sha256"]:
        raise ValueError("Locked checkpoint changed")
    state = torch.load(lock["checkpoint"], map_location="cpu", weights_only=False)
    validate_checkpoint_lock(lock, state, file_sha256(args.identity), digest)
    progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    if (progress.get("status") != "COMPLETE" or progress.get("checkpoint_sha256") != lock["checkpoint_sha256"]
            or progress.get("executed_updates") != 40000 or progress.get("checkpoint_persisted_updates") != 40000
            or progress.get("in_flight_update") is not None or progress.get("run_identity") != state["run_identity"]):
        raise ValueError("Endpoint ledger is not a persisted COMPLETE boundary")
    protocol = json.loads((Path(__file__).parent / "protocol.json").read_text(encoding="utf-8"))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Locked evaluation requires the BF16 CUDA execution precision")
    if state.get("environment") != runtime_identity():
        raise ValueError("Evaluation runtime differs from the training/preflight identity")
    torch.set_num_threads(4)
    model = execution_model(args.pretrained).cuda().eval()
    model.load_state_dict(state["model"], strict=True); del state
    # Exclusive create consumes the single evaluation even if hashing/evaluation
    # later fails. Two concurrent invocations cannot both open the test split.
    with (out / "test_started.json").open("x", encoding="utf-8") as stream:
        json.dump({"checkpoint_sha256": lock["checkpoint_sha256"], "source_sha256": digest,
                   "identity_sha256": file_sha256(args.identity)}, stream)
        stream.flush(); os.fsync(stream.fileno())
    try:
        checks = verify_split_files(args.data_root, args.identity, ("test",), allow_test=True)
        write_json(out / "test_files_audit.json", checks)
        dataset = SegmentationData(args.data_root, args.identity, "test", 2048, allow_test=True)
        result = evaluate(model, dataset)
        result.update(checkpoint_sha256=lock["checkpoint_sha256"], completed=40000,
                      target_achieved=result["mIoU_percent"] >= protocol["test_target_percent"],
                      target_exceeded=result["mIoU_percent"] > protocol["test_target_percent"],
                      val_target_passed=True, test_files_audit=checks, status="COMPLETE")
        write_json(out / "test_result.json", result)
        write_json(out / "result.json", result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        write_json(out / "test_failure.json", {"status": "FAILED", "error_type": type(error).__name__,
                   "error": str(error), "retry_permitted": False})
        raise


if __name__ == "__main__":
    main()
