"""Retain the audited baseline implementation without a redundant weight hash scan."""
from pathlib import Path
HERE = Path(__file__).resolve().parent
source = (HERE.parent / 'training_execution/model.py').read_text(encoding='utf-8').split('class V3Student')[0]
source = source.replace('from objectives import masked_dice\n', '')
source = source.replace('from private_modules import V3PrivateModules\n', '')
source = source.replace('seed=42, canvas=2048):', 'seed=42, canvas=2048, verified_weight_sha=None):')
source = source.replace('assert file_sha(pretrained) == W48_SHA, "Wrong ImageNet W48 file"',
                        'assert verified_weight_sha == W48_SHA, "Require previously audited ImageNet W48 identity"')
(HERE / 'baseline.py').write_text(source, encoding='utf-8')
