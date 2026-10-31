"""Read synced evidence and save one heartbeat snapshot; never controls training."""
import datetime
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "observed_server_attempt03"
RUN = CACHE / "run_seed42_launch02"
SOURCE = "d3032fc8b20f30a058adbcf3e6bc5a3485b83adafe9b7cdce0682a2d55d7f627"


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    progress = read(RUN / "progress.json")
    controller = read(CACHE / "job_status.json")
    require(progress is not None and controller is not None, "Cached evidence missing; remote inspection required")
    require(progress["run_identity"]["source_sha256"] == SOURCE
            and controller.get("source_sha256") == SOURCE, "Unexpected active source identity")
    completed = progress["executed_updates"]
    persisted = progress["checkpoint_persisted_updates"]
    require(progress["completed"] == completed and 0 <= persisted <= completed <= 40000,
            "Inconsistent update counters")
    require(progress["in_flight_update"] in (None, completed + 1), "Invalid in-flight boundary")
    age = now.timestamp() - (RUN / "progress.json").stat().st_mtime
    require(progress["status"] != "TRAINING" or age <= 120,
            "Active cache is stale; inspect remote process before drawing a conclusion")
    directory = HERE / "heartbeat_checks"
    directory.mkdir(exist_ok=True)
    previous = read(directory / "latest_check.json") or {}
    if previous.get("source_sha256") == SOURCE:
        require(completed >= previous.get("executed_updates", 0), "Executed count decreased")
        require(persisted >= previous.get("checkpoint_persisted_updates", 0), "Persisted count decreased")
    validations = []
    for path in sorted(RUN.glob("val_*.json")):
        item = read(path)
        validations.append({key: item.get(key) for key in
                            ("update", "mIoU_percent", "IoU_percent", "recall_percent")})
    old_nodes = {row["update"] for row in previous.get("validation", [])}
    record = {
        "checked_at_UTC": now.isoformat(), "automation_id": "nativepyramid-sp-v3",
        "check_interval_minutes": 45, "source_sha256": SOURCE,
        "status": progress["status"], "controller_status": controller["status"],
        "executed_updates": completed, "checkpoint_persisted_updates": persisted,
        "in_flight_update": progress["in_flight_update"],
        "checkpoint_sha256": progress["checkpoint_sha256"], "expected_updates": 40000,
        "validation": validations, "active_evidence_fresh": age <= 120,
        "cached_progress_age_seconds": age,
        "updates_since_previous_check": completed - previous.get("executed_updates", completed),
        "pause_reasons": progress.get("pause_reasons", []),
        "failure": read(RUN / "failure.json"), "test_failure": read(RUN / "test_failure.json"),
        "endpoint_lock": read(RUN / "endpoint_lock.json"), "test_result": read(RUN / "test_result.json"),
        "notes": [
            "Live progress is authoritative; result.json completed=0 is the initialization record until terminal state.",
            "Root job_result.json from controller v2/source ddb4 is unrelated to the active v3/source d303 outcome.",
            "No switch approval received. No restart, resume, runtime change, test or shutdown initiated by this check."
        ]
    }
    destination = directory / "latest_check.json"
    temporary = destination.with_suffix(".writing")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps({"status": record["status"], "executed_updates": completed,
                      "checkpoint_persisted_updates": persisted,
                      "new_validation": [row for row in validations if row["update"] not in old_nodes],
                      "pause_reasons": record["pause_reasons"], "failure": record["failure"],
                      "test_failure": record["test_failure"], "endpoint_lock": record["endpoint_lock"],
                      "test_result": record["test_result"]}))


if __name__ == "__main__":
    main()
