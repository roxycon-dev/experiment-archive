"""Extract exact U0 model tensors and S20 EMA tensors; preserve source identity."""
from pathlib import Path
import gc
import hashlib
import json
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]

def sha(path):
    d = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""): d.update(b)
    return d.hexdigest()

def main():
    output = HERE / "inference_weights"
    output.mkdir(exist_ok=True)
    jobs = [
        (EXP / "HRDecoder_Ready_20261002/best_pipeline/weights/U0_historical_best.pth",
         "98d98a5d02dc2615062c3c31cd15d061daf3d7ab41cb8f25cd90b062cd53bb34", "model_state", "U0_inference.pth"),
        (EXP / "HRDecoder_FinalDecision_20261005/execution/server_results/runs/S_seed42/checkpoint_u20000.pth",
         "c933256b07556689a7fd199dd80e50d4d0c27eb6b9ea782713d95ec51e6d8d62", "ema", "S20_ema_inference.pth"),
    ]
    rows = {}
    for source, expected, state_key, filename in jobs:
        actual = sha(source)
        assert actual == expected, str(source) + ": source SHA mismatch"
        original = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
        state = original[state_key]
        destination = output / filename
        saved = {state_key: state, "source_sha256": actual, "source_state_key": state_key,
                 "source_completed": original.get("completed", original.get("update")),
                 "source_ema_completed": original.get("ema_completed")}
        torch.save(saved, destination)
        reloaded = torch.load(destination, map_location="cpu", weights_only=False, mmap=True)
        assert set(reloaded[state_key]) == set(state)
        assert all(torch.equal(value, reloaded[state_key][name]) for name, value in state.items())
        rows[filename] = {"source_path": str(source), "source_sha256": actual,
                          "source_state_key": state_key, "inference_file_sha256": sha(destination),
                          "source_bytes": source.stat().st_size, "inference_bytes": destination.stat().st_size,
                          "tensor_count": len(state), "all_tensors_bitwise_equal": True,
                          "source_completed": saved["source_completed"],
                          "source_ema_completed": saved["source_ema_completed"]}
        print(json.dumps({"file": filename, **rows[filename]}), flush=True)
        del reloaded, saved, original, state
        gc.collect()
    (HERE / "inference_weight_receipt.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
