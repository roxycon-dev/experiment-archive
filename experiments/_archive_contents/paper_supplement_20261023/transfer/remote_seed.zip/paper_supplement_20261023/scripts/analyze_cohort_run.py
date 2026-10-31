"""Verify holdout masks/counts, paired uncertainty and actual full-only timing."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from analyze_fixed_run import sha, write_json, summary, bootstrap_scores, ci, NAMES
from lesion_metrics import match_lesions

ROOT = Path(__file__).resolve().parents[1]


def binary_counts(b, gt):
    return np.stack([(b & gt).sum((1,2)), (b & ~gt).sum((1,2)), (~b & gt).sum((1,2))], axis=1).astype(np.int64)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('run', type=Path)
    args = parser.parse_args(); run = args.run.resolve()
    assert run.is_relative_to((ROOT/'results').resolve())
    assert (run/'completion.json').exists() and not (run/'failure.json').exists()
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    cohort = json.loads((run/'cohort.json').read_text(encoding='utf-8'))
    assert sha(run/'cohort.json') == manifest['cohort_sha256']
    out = run/'analysis'; out.mkdir(exist_ok=True)
    rows = []; lesion_rows = {k: [] for k in ['full', 'no_h4', 'matched_c20']}
    positive = 0
    for row in cohort['samples']:
        sid = row['sid']; r = json.loads((run/'per_image'/(sid+'.json')).read_text(encoding='utf-8'))
        assert r['sid'] == sid and not r['warmup']
        pred = Path(r['prediction_path'])
        assert pred.parent == run/'predictions' and sha(pred) == r['mask_sha256']
        assert sha(row['image_path']) == row['image_sha256'] == r['image_sha256']
        if manifest['mode'] == 'external':
            gt = []
            for c in ['EX','HE','SE','MA']:
                mr = row['masks'][c]; assert sha(mr['path']) == mr['sha256']
                gt.append(np.asarray(Image.open(mr['path'])) > 0)
            gt = np.stack(gt); positive += int(gt[3].any())
        else:
            label = np.asarray(Image.open(Path('D:/HRDecoder/data/DDR_split_613/labels/val')/(sid+'.png')))
            assert not (label == 255).any(); gt = np.stack([label == c for c in range(1,5)])
        with np.load(pred, allow_pickle=False) as z:
            if manifest['mode'] == 'external':
                assert set(z.files) == set(NAMES)
                assert np.array_equal(z['full'][:3], z['no_h4'][:3])
            for k in z.files:
                b = z[k]; assert b.dtype == bool and b.shape == gt.shape
                assert binary_counts(b, gt).tolist() == r['counts'][k], (sid,k)
                if manifest['mode'] == 'external' and k in lesion_rows:
                    lesion_rows[k].append({'sid': sid, **match_lesions(gt[3], b[3])})
        rows.append(r)
        print(json.dumps({'verified': len(rows), 'sid': sid}), flush=True)
    assert len(rows) == manifest['expected_images']
    if manifest['mode'] == 'profile':
        params = json.loads((run/'parameters.json').read_text(encoding='utf-8'))
        times = [r['timing']['pipeline_to_fused_mask_seconds'] for r in rows]
        persisted = [r['timing']['pipeline_through_persistence_seconds'] for r in rows]
        peaks = [max(r['GPU_memory']['A_parent_peak_allocated'], r['GPU_memory']['U_parent_peak_allocated'],
                     r['timing']['modern']['S']['peak_allocated_bytes']) for r in rows]
        report = {'status': 'COMPLETE', 'mode': 'full_system_efficiency', 'hardware': manifest['hardware'],
                  'registered_parameters': params, 'profile_samples': [r['sid'] for r in rows],
                  'image_shapes': [r['hw'] for r in rows], 'batch_size': 1, 'warmup_images_excluded': 1,
                  'pipeline_to_fused_mask_seconds': {'per_image': times, 'mean': float(np.mean(times)),
                                                   'median': float(np.median(times)), 'min': min(times), 'max': max(times)},
                  'through_counts_and_persistence_seconds': {'per_image': persisted, 'mean': float(np.mean(persisted))},
                  'max_single_process_allocated_peak_bytes': max(peaks),
                  'allocated_peak_method': 'Maximum across sequential A/U/S stages; no concurrent resident GPU parameters; driver contexts and other apps excluded',
                  'includes': 'image read/transform, all four experts and horizontal TTA, MA head, U whole-native FP32, S BF16 reference-MSDA tiles, CPU-GPU transfers, IPC, alignment and fusion',
                  'excludes': 'checkpoint/model startup, warmup, training, GT reading/scoring and compressed result persistence (shown separately)',
                  'timing_limitations': 'Three fixed validation images only, single laptop GPU, no multi-GPU throughput measurement; registered parameters include copied references and auxiliary modules'}
        observed=[]
        for line in (run/'GPU_memory_samples.csv').read_text(encoding='utf-8').splitlines():
            values=line.split(',')
            if len(values)==2:
                try: observed.append(float(values[1].strip()))
                except ValueError: pass
        assert observed, 'GPU total memory sampling did not produce numeric observations'
        report['sampled_GPU_total_memory']={'maximum_MiB':max(observed),'minimum_MiB':min(observed),
                                            'sampling_interval_ms':200,'samples':len(observed),
                                            'scope':'Whole GPU including both inference processes, contexts, driver and other apps; sampled maximum can miss sub-200ms peaks; starts with A already loaded'}
        ddr=ROOT/'results/ddr_test_local_v1'
        ddr_manifest=json.loads((ddr/'manifest.json').read_text(encoding='utf-8'))
        stage_times=[]
        for sid in ddr_manifest['sample_ids']:
            r=json.loads((ddr/'per_image'/(sid+'.json')).read_text(encoding='utf-8'))
            t=r['timing']; stage_times.append(t['A_inference_seconds']+t['U_inference_seconds']+t['S']['seconds'])
        report['DDR227_recorded_A_U_S_stage_sum_seconds']={'n_images':len(stage_times),'mean':float(np.mean(stage_times)),
             'median':float(np.median(stage_times)),'p95':float(np.quantile(stage_times,.95)),
             'scope':'Observed A+U+S stage times from the four-row run across heterogeneous native sizes; C forward omitted, C parameters still resident in that run. Excludes orchestration/fusion/persistence; not end-to-end latency.'}
    else:
        n = len(rows); indices = np.random.default_rng(20261023).integers(0,n,(2000,n))
        arrays = {k: np.asarray([r['counts'][k] for r in rows], np.int64) for k in NAMES}
        boots = {k: bootstrap_scores(a, indices) for k,a in arrays.items()}
        table = []
        for k in NAMES:
            s = summary(arrays[k]); s.pop('counts')
            lo,hi = ci(boots[k])
            table.append({'method': k, 'n_images': n, **s, 'miou_ci_low_pct':lo,'miou_ci_high_pct':hi})
        with (out/'external_component_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w = csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
        differences = {k: {'full_minus_row_pp': table[0]['miou_pct']-next(r['miou_pct'] for r in table if r['method']==k),
                           'paired_image_ci95_pp': ci(boots['full']-boots[k])} for k in NAMES if k != 'full'}
        lesions = []; lesion_boot = {}
        for k,items in lesion_rows.items():
            a = np.asarray([[r['tp'],r['fp'],r['fn']] for r in items],np.int64)
            t = a.sum(0); b = a[indices].sum(1)
            sensitivity = t[0]/max(t[0]+t[2],1)*100; fp = t[1]/n
            lesion_boot[k] = {'detection_pct': b[:,0]/np.maximum(b[:,0]+b[:,2],1)*100, 'FP_per_image': b[:,1]/n}
            lesions.append({'method': k, 'n_images': n, 'MA_positive_images':positive,
                            'tp':int(t[0]),'fp':int(t[1]),'fn':int(t[2]),'detection_pct':float(sensitivity),
                            'FP_per_image':float(fp),'lesion_precision_pct':float(t[0]/max(t[0]+t[1],1)*100),
                            'detection_ci95_pct':ci(lesion_boot[k]['detection_pct']),
                            'FP_per_image_ci95':ci(lesion_boot[k]['FP_per_image'])})
        with (out/'external_MA_lesion_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
            flat = [{k:v for k,v in r.items() if not isinstance(v,list)} for r in lesions]
            w = csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
        report = {'status':'COMPLETE','mode':'independent_heldout_subset','n_images':n,
                  'pixel_table':table,'pixel_comparisons':differences,'MA_lesion_table':lesions,
                  'MA_paired_intervals':{k:{metric:ci(lesion_boot['full'][metric]-lesion_boot[k][metric])
                                            for metric in lesion_boot['full']} for k in ['no_h4','matched_c20']},
                  'inference_geometry':'stored image native coordinates; U native whole-image, S/C 1024-center768; HRDecoder archived preprocessing',
                  'bootstrap':{'unit':'image','draws':2000,'seed':20261023,'training_seed_variance_included':False},
                  'history_authority':cohort['usage_history'],'dataset_training_exposure':cohort['dataset_training_exposure'],
                  'annotation_authority':cohort['annotation_authority'],
                  'limits':['Other project explorations used IDRiD training54, but selected S/C lineage is DDR-only; testing27 holdout history based on user confirmation',
                            'patient IDs unavailable; no patient-level independence or bootstrap',
                            'stored square geometry and user-provided converted annotation, no independent official-native GT comparison',
                            'single seed; no original HRDecoder baseline checkpoint; new local reference-MSDA runtime'],
                  'no_external_tuning':True}
        write_json(out/'lesion_metrics.json',lesion_rows)
    write_json(out/'summary.json',report)
    print(json.dumps({'status':'COMPLETE','mode':manifest['mode'],'output':str(out)},ensure_ascii=False),flush=True)


if __name__ == '__main__':main()
