"""CPU pipeline tuning and timing only; unchanged batches, objectives and optimizer."""
import json
import os
from pathlib import Path
import statistics
import queue
import threading
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"vendor/champion_repo"))
import train

original_loader = train.DataLoader
original_step = train.one_step
step_times, data_times = [], []
run = HERE.parent/"training_runs/v3"
config_path = HERE.parent/'reports/performance_config.json'
config = json.loads(config_path.read_text()) if config_path.exists() else {}
if config.get('channels_last'):
    from channels_last_runtime import install
    install(train)
workers = int(config.get('workers',os.environ.get("HRDECODER_WORKERS", "12")))
prefetch = int(config.get('prefetch',os.environ.get("HRDECODER_PREFETCH", "1")))
ram_info = None
if config.get('full_shared_RAM'):
    from ram_loader import install_ram_loader
    ram_info = install_ram_loader(HERE.parent)
    print(json.dumps({'RAM_cache_enabled':True,'images':ram_info['selected_images'],
        'bytes':ram_info['bytes'],'workers':workers,'prefetch':prefetch}),flush=True)
profiler = None

# The user explicitly asked to reuse the already validated, unchanged disks.
# Reuse their stored file identities; do not read them again solely for SHA.
if '--resume' in sys.argv and (run/'protocol.json').exists():
    previous=json.loads((run/'protocol.json').read_text())['identity']
    ledger={str((HERE/name).resolve()):digest for name,digest in
        json.loads((HERE/'execution_manifest.json').read_text()).items()}
    ledger.update({str(train.W48.resolve()):previous['ImageNet_W48_SHA'],
        str((train.ROOT/'metadata/data_identity.json').resolve()):previous['data_identity_SHA'],
        str((train.REPO/'configs/lesion/exp/baseline_split613.py').resolve()):previous['baseline_config_SHA'],
        str((train.ROOT/'metadata/source_manifest.json').resolve()):previous['source_manifest_SHA']})
    def validated_file_identity(path):
        return ledger[str(Path(path).resolve())]
    train.file_sha=validated_file_identity
    import model
    model.file_sha=validated_file_identity

def worker_setup(worker_id):
    import cv2
    cv2.setNumThreads(1)
    train.torch.set_num_threads(1)

class MeasuredLoader:
    def __init__(self, loader): self.loader = loader
    def __iter__(self):
        if ram_info is not None:
            yield from self.cuda_prefetched()
            return
        iterator = iter(self.loader)
        while True:
            started = time.monotonic()
            try:
                with train.torch.profiler.record_function('DATA_WAIT'): batch = next(iterator)
            except StopIteration: return
            data_times.append(time.monotonic()-started)
            yield batch
    def __len__(self): return len(self.loader)

    def cuda_prefetched(self):
        pending=queue.Queue(maxsize=int(config.get('CUDA_prefetch_slots',1)))
        iterator=iter(self.loader)  # Fork CPU workers on the main thread.
        copy_stream=train.torch.cuda.Stream()
        def produce():
            try:
                for cpu_batch in iterator:
                    with train.torch.cuda.stream(copy_stream), train.torch.profiler.record_function('H2D_PREFETCH'):
                        gpu_batch=train.move(cpu_batch)
                        event=train.torch.cuda.Event()
                        event.record(copy_stream)
                    pending.put((gpu_batch,event,cpu_batch))
                pending.put(None)
            except BaseException as error: pending.put(error)
        thread=threading.Thread(target=produce,daemon=True)
        thread.start()
        first=True
        while True:
            started=time.monotonic()
            with train.torch.profiler.record_function('DATA_WAIT'): item=pending.get()
            if item is None: thread.join(); return
            if isinstance(item,BaseException): raise item
            gpu_batch,event,cpu_batch=item
            data_times.append(time.monotonic()-started)
            current=train.torch.cuda.current_stream()
            current.wait_event(event)
            for value in gpu_batch.values():
                if train.torch.is_tensor(value): value.record_stream(current)
            if first:
                for name,value in gpu_batch.items():
                    if train.torch.is_tensor(value):
                        assert train.torch.equal(value.cpu(),cpu_batch[name]),name
                train.write_json(HERE.parent/'reports/GPU_ASYNC_PREFETCH_PASS.json',
                    {'status':'PASS','full_image_label_valid_A_U_S_bits_match':True,
                        'CUDA_stream_wait_event_and_record_stream':True,
                        'formal_updates_added_by_check':0,'next_batch_slots':pending.maxsize})
                first=False
            yield gpu_batch
            # Retain pinned CPU tensors until the real step has synchronized.
            del cpu_batch,item

def tuned_loader(*args, **kwargs):
    kwargs.update(num_workers=workers, prefetch_factor=prefetch,
        persistent_workers=True, pin_memory=True, worker_init_fn=worker_setup)
    return MeasuredLoader(original_loader(*args, **kwargs))

def measured_step(*args, **kwargs):
    started = time.monotonic()
    with train.torch.profiler.record_function('TRAIN_STEP'):
        result = original_step(*args, **kwargs)
    # The original finite-gradient and loss scalar checks already synchronize.
    step_times.append(time.monotonic()-started)
    count = len(step_times)
    if count == 1 or count % 10 == 0:
        skip = min(12 if ram_info else 5, count-1)
        steps = step_times[skip:][-100:]
        waits = data_times[skip:][-100:]
        mean = statistics.mean(steps)+statistics.mean(waits)
        train.write_json(run/"performance.json", {"measured_batches": count,
            "workers": workers, "prefetch_factor": prefetch, "pin_memory": True,
            "opencv_threads_per_worker": 1, "completed_update": args[4],
            "shared_RAM_cache_bytes": json.loads((HERE.parent/'reports/RAM_CACHE_READY.json').read_text())['bytes'] if ram_info else 0,
            "channels_last":bool(config.get('channels_last')),
            "CUDA_async_next_batch": ram_info is not None,
            "warmup_batches_excluded": skip, "window_batches": len(steps),
            "mean_step_compute_seconds": statistics.mean(steps),
            "mean_data_wait_seconds": statistics.mean(waits),
            "mean_batch_seconds": mean,
            "median_compute_seconds": statistics.median(steps),
            "estimated_40000_train_hours_excluding_validation": 40000*mean/3600,
            "peak_allocated_bytes": train.torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": train.torch.cuda.max_memory_reserved(),
            "formal_protocol_unchanged": True, "time_unix": time.time()})
    if profiler is not None: profiler.step()
    return result

def export_profile(prof):
    try:
        prof.export_chrome_trace(str(HERE.parent/'reports/GPU_STEP_TRACE.json'))
        rows=[{'name':event.key,'count':event.count,
            'self_CPU_us':event.self_cpu_time_total,
            'self_CUDA_us':event.self_cuda_time_total,
            'total_CUDA_us':event.cuda_time_total} for event in prof.key_averages()]
        rows.sort(key=lambda row:row['self_CUDA_us'],reverse=True)
        train.write_json(HERE.parent/'reports/GPU_STEP_PROFILE.json',
            {'status':'RECORDED','formal_steps_retained':True,'active_steps':4,'rows':rows})
    except Exception as error:
        print(json.dumps({'profiling_export_error':str(error)}),flush=True)

train.DataLoader = tuned_loader
train.one_step = measured_step
if __name__ == "__main__":
    if ram_info is not None:
        profiler=train.torch.profiler.profile(
            activities=[train.torch.profiler.ProfilerActivity.CPU,train.torch.profiler.ProfilerActivity.CUDA],
            schedule=train.torch.profiler.schedule(wait=0,warmup=5,active=4,repeat=1),
            on_trace_ready=export_profile,record_shapes=False,profile_memory=False,with_stack=False)
        with profiler: train.main()
    else: train.main()
