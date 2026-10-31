"""Verify reused old ensemble bytes and the recorded compact U export receipt."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    original = json.loads((EXP/'HRDecoder_Q_20261003/configs/p0_identity.json').read_text(encoding='utf-8'))
    receipt_path = EXP/'HRDecoder_ArchitectureDesign_20261008/cache_execution/inference_weight_receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    rows = []
    assets = EXP/'HRDecoder_Ready_20261002/best_pipeline/weights'
    paths = [original['champion']['head']] + [m['checkpoint'] for m in original['champion']['members']]
    for old in paths:
        old = Path(old); now = assets/old.name
        digest = sha(now)
        assert digest == sha(old), (str(old),str(now))
        rows.append({'name':old.name,'historical_path':str(old),'current_path':str(now),'sha256':digest,'bitwise_identity_via_SHA256':True})
    for name, r in receipt.items():
        now = receipt_path.parent/'inference_weights'/name
        assert sha(now) == r['inference_file_sha256']
        assert sha(Path(r['source_path'])) == r['source_sha256']
    proof = {'status':'PASS','A_assets':rows,'compact_export_receipt':str(receipt_path),'receipt_sha256':sha(receipt_path),
             'compact_exports':receipt,'threshold_and_fusion_searched':False,
             'limits':'Verifies weight identity and export receipt; does not establish numerical identity across CUDA/PyTorch runtimes'}
    (ROOT/'evidence/fixed_assets_identity.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','A_files':len(rows),'compact_files':len(receipt)}),flush=True)


if __name__=='__main__':main()
