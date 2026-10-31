def estimate_cost(frozen_seconds, unfrozen_seconds, full_val_seconds, hourly_rate=None, endpoints=(16000, 24000), backup_seconds=1800, test_seconds=0):
    rows = []
    for endpoint in endpoints:
        val_count = 5 if endpoint == 16000 else 7
        base = 2000*frozen_seconds + (endpoint-2000)*unfrozen_seconds + val_count*full_val_seconds
        # Two arms/seed pairing is only purchased after its evidence gate.
        for runs in (1, 2, 4):
            hours = (runs*base+backup_seconds+test_seconds)/3600*1.2
            rows.append({'endpoint': endpoint, 'formal_runs': runs, 'buffer_fraction': .2,
                         'gpu_hours_including_buffer': hours, 'estimated_cny': None if hourly_rate is None else hours*hourly_rate,
                         'val_nodes_per_run': val_count, 'includes_backup_seconds': backup_seconds, 'test_seconds': test_seconds})
    return rows
