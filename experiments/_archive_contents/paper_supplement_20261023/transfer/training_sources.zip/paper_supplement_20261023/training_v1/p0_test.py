"""Exact historical P0 inference behind the new immutable selection gate."""
import argparse,importlib.util
from pathlib import Path
from qlib.common import read_json,sha,digest
ROOT=Path(__file__).resolve().parent

def main(runtime_path):
    runtime=read_json(runtime_path);old=Path(runtime['old_root']);driver=old/'reports/final_evaluation.py'
    assert sha(driver)==runtime['old_p0_driver_sha256']
    assert sha(old/'configs/p0_original_lock.json')==runtime['old_p0_lock_sha256']
    lock_file=ROOT/'reports/final_selection_lock.json';lock=read_json(lock_file)
    assert digest({k:v for k,v in lock.items() if k!='lock_sha256'})==lock['lock_sha256']
    assert lock['formula'] in ('fixed_fusion','half_fusion') and lock['runtime_sha256']==sha(runtime_path)
    assert read_json(ROOT/'reports/decision.json')['test_authorized']
    for rel,expected in lock['validation_files_sha256'].items():assert sha(ROOT/rel)==expected
    spec=importlib.util.spec_from_file_location('frozen_historical_p0',driver)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUT=ROOT/'reports/test_p0';module.OUT.mkdir(parents=True,exist_ok=True)
    module.gate=lambda:({'p0_cache':runtime['test_p0_cache']},sha(lock_file))
    module.p0_phase()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);a=p.parse_args();main(a.runtime)
