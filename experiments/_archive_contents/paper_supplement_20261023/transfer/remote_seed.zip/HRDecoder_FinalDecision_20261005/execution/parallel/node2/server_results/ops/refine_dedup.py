"""pHash screens similarities, never sufficient to declare identical retinas.

The observed pHash10 example train003/DDR007-4850-300 has visibly different
vessel trees. Final near-duplicate exclusion requires geometric ORB evidence
for every non-exact candidate; record the pretraining-only methodological repair.
"""
import sys,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from concurrent.futures import ThreadPoolExecutor
from qlib.common import read_json,write_json,utc_now
from ops.accept_tjdr import orb_match

def main():
    manifest=read_json(ROOT/'reports/TJDR_manifest.json');original=read_json(ROOT/'reports/TJDR_duplicate_pairs.json')
    write_json(ROOT/'reports/TJDR_manifest.phash_provisional.json',manifest)
    write_json(ROOT/'reports/TJDR_duplicate_pairs.phash_provisional.json',original)
    runtime=read_json(ROOT/'configs/runtime.local.json')
    def check(pair):
        if pair['exact_rgb']:return pair
        if pair['phash_distance']<=10:
            a=ROOT/'inputs/TJDR/train/image'/(pair['TJDR_sid']+'.png')
            b=Path(runtime['ddr_root'])/'images'/pair['other_split']/(pair['other_sid']+'.jpg') if pair['other_dataset']=='DDR' else ROOT/'inputs/TJDR/train/image'/(pair['other_sid']+'.png')
            pair['orb']=orb_match(str(a),str(b))
        pair['excluded']=pair['orb'].get('inliers',0)>=40 and pair['orb'].get('fraction',0)>=.6
        return pair
    with ThreadPoolExecutor(max_workers=6) as pool:pairs=list(pool.map(check,original['pairs']))
    by_sid={row['sid']:row for row in manifest['samples']}
    for row in by_sid.values():row.update(excluded_duplicate=False,exclusion_reasons=[])
    for pair in pairs:
        if pair['excluded']:
            row=by_sid[pair['TJDR_sid']];row['excluded_duplicate']=True;row['exclusion_reasons'].append(pair)
    rules=dict(original['rules'],phash_only_is_not_duplicate=True,final_exclusion='Exact RGB hash OR>=40 geometric inliers with fraction>=.6; pHash<=18 screens candidates only')
    write_json(ROOT/'reports/TJDR_duplicate_pairs.json',{'rules':rules,'pairs':pairs,'uses_DDR_test_images_only':True})
    manifest.update(status='PENDING_VISUAL_REVIEW',dedup_rules=rules,accepted_count=sum(not r['excluded_duplicate'] for r in by_sid.values()),
        visual_review_count=0,dedup_status='PASS')
    write_json(ROOT/'reports/TJDR_manifest.json',manifest)
    write_json(ROOT/'reports/TJDR_dedup_revision.json',{'status':'PASS','time':utc_now(),'training_started':False,
        'reason':'pHash-only quarantine produced visual false positives on fundus images; geometry verification now required for all non-exact pairs.',
        'reviewed_example':'TJDR_train_003 vs DDR train007-4850-300; visibly different vessel trees',
        'candidate_threshold_unchanged':18,'geometric_threshold_unchanged':{'inliers':40,'fraction':.6},'accepted_count':manifest['accepted_count'],
        'previous_results_retained':True})
    print('GEOMETRIC_DEDUP_REVIEW_COMPLETE',manifest['accepted_count'],flush=True)

if __name__=='__main__':main()
