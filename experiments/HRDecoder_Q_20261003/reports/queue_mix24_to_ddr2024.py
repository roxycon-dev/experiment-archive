"""One registered MIX42 discovery -> DDR2024 review extension handoff."""
import pathlib
import sys

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reports'))
import queue_ddr24_to_mix24 as operation

operation.h.RUN='Q_MIX_seed42'
operation.h.IDENTITY='9803ad79aecd9c3d7a426fda0053ae0e43f1afc663933e30d35364be38800a3e'
operation.PREFIX='mix42_24k'
operation.h.STATE=ROOT/'reports/mix42_24k_handoff_state.json'
operation.h.DELETED=ROOT/'reports/mix42_24k_cache_offload_receipt.json'
operation.h.PLAN_PREFIX='mix42_24k_cache_offload_plan'
operation.TARGET_ARM='Q_DDR'
operation.TARGET_SEED=2024
# This resume retains two additional checkpoints and exports each completed
# late NPZ node while training continues. One complete late node plus both new
# checkpoints and 2 GiB reserve fit within the original 8 GiB launcher gate.
operation.MIN_FREE_BYTES=8*1024**3
operation.TARGET_LOG='ddr2024_24k_launcher.log'

if __name__=='__main__':
    try:operation.main()
    except BaseException as error:
        operation.h.state('TECHNICAL_PAUSE_HANDOFF',reason=f'{type(error).__name__}: {error}')
        raise
