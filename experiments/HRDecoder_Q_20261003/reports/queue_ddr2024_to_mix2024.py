"""One registered DDR2024 extension -> fresh MIX2024 paired-review handoff."""
import pathlib
import sys

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reports'))
import queue_ddr24_to_mix24 as operation

operation.h.RUN='Q_DDR_seed2024'
operation.h.IDENTITY='c25584777fd2ded929bce248f6fc0e05d954ce9ba200f62d3066dbad41cdd252'
operation.PREFIX='ddr2024_24k'
operation.h.STATE=ROOT/'reports/ddr2024_24k_handoff_state.json'
operation.h.DELETED=ROOT/'reports/ddr2024_24k_cache_offload_receipt.json'
operation.h.PLAN_PREFIX='ddr2024_24k_cache_offload_plan'
operation.TARGET_ARM='Q_MIX'
operation.TARGET_SEED=2024
operation.TARGET_RESUME=False
operation.MIN_FREE_BYTES=9*1024**3
operation.TARGET_LOG='mix2024_24k_launcher.log'

if __name__=='__main__':
    try:operation.main()
    except BaseException as error:
        operation.h.state('TECHNICAL_PAUSE_HANDOFF',reason=f'{type(error).__name__}: {error}')
        raise
