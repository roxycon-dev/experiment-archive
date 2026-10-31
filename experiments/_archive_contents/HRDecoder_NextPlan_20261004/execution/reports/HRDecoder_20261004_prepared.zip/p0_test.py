"""Reuse the historical P0 inference function verbatim behind the new frozen gate."""
import argparse
import importlib.util
from pathlib import Path
import sys
from qlib.common import read_json,sha

ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runtime',required=True);args=ap.parse_args()
    runtime=read_json(args.runtime);old=Path(runtime['old_root'])
    driver=old/'reports/final_evaluation.py'
    assert sha(driver)==runtime['old_p0_driver_sha256']
    assert sha(old/'configs/p0_original_lock.json')==runtime['old_p0_lock_sha256']
    lock_file=ROOT/'reports/final_selection.json';lock=read_json(lock_file)
    assert lock['test_authorized'] and lock['runtime_sha256']==sha(args.runtime)
    spec=importlib.util.spec_from_file_location('frozen_historical_p0',driver)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Only authorization/output location changes; model composition, transforms,
    # original-image inference, FP32 mixing, thresholds and historical count checks stay intact.
    module.OUT=ROOT/'reports/test_p0'
    module.OUT.mkdir(parents=True,exist_ok=True)
    module.gate=lambda:({'p0_cache':runtime['test_p0_cache']},sha(lock_file))
    module.p0_phase()


if __name__=='__main__':main()
