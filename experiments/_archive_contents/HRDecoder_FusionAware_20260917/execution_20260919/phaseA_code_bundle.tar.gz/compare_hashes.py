# -*- coding: utf-8 -*-
"""Compare local repo copies vs server hashes (from env_report.json)."""
import hashlib
import json
import os

LOCAL_REPO = r'D:\dataset\experiment\hrdecoder_plus3\repo'
SERVER = {  # from env_report/model_file_shas + inventory cfg_shas/base_cfg_shas
    'hrplus_patch.py': '6433a8fafc950c83b1a6fd432e8c901fb843b3fbaff032454e4289fe8bdd0f37',
    'hr100_refinement.py': '2d0db3056bc50bd68fd4da71f470f6e80fbf4deeb78264cbd4640dc2ab0c1b72',
    'hrplus_core.py': '4297d0fc3ffac8927e27a91531a9e9ea66280bfae6eb22a38a39fdaf780bf448',
    'ma_residual.py': '95438df2b26df31a0c29e4933afa482d44072e5be42524177586fb18c165c1ed',
    'mmseg/models/segmentors/HRDecoder.py': 'a97d00407c03947363df65a035dc70e5b458f440e38c21161a247b3282c1b933',
    'mmseg/models/segmentors/lesion_encoder_decoder.py': '3a11c417030f3066670df5d53e060746c16bec16f089dfeaaba4e77ac3cea5f8',
    'mmseg/datasets/lesion_dataset.py': '96e697e26f15c477d0d7707bc38f9d93515a646c508ac7c4dfc35e6e762f0c42',
    'mmseg/models/losses/binary_loss.py': 'a56634a14dccf92ce004c9a87684ef53c6d66d8705ff49b5ab7c184a4e5e903c',
    'configs/lesion/exp/hrplus_patch_dice.py': '2d687086e28533d4bb080f31c12c4abb802d5293fc4f2a463f9f0bd8f36452f9',
    'configs/lesion/exp/hr100_b.py': '556e7c2f1557c6658f6233f1fe1601b8d0b1c8382cc6bdaaaf108a712da3ed54',
    'configs/lesion/exp/e4_detail_split613.py': '65b2c5eb88db2790c1e00dbcb6e18deaf8fc6abc5bf3f148be9ae70d2f06fc1c',
    'configs/lesion/exp/hr100_a.py': 'f60a545f6120464275edd60f1e9e504291e676e3a5e6b0df7eba3071af83f45a',
    'configs/_base_/models/hrdecoder_fcn_hr18.py': '76e807d9c7f16bc4a76fd8d89230613350115f606dc287904c2c10d7c84463b1',
    'configs/_base_/datasets/hr_ddr_2048.py': '1ac4def51b66d95492c3631bba3024dd19db4baee3d0ad46319a028b1d046d91',
}

for rel, srv in sorted(SERVER.items()):
    p = os.path.join(LOCAL_REPO, rel.replace('/', os.sep))
    if not os.path.isfile(p):
        print('LOCAL-MISSING  %s' % rel)
        continue
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    print('%s  %s' % ('MATCH' if h == srv else 'DIFF!!!', rel))
    if h != srv:
        print('   local  %s' % h)
        print('   server %s' % srv)
