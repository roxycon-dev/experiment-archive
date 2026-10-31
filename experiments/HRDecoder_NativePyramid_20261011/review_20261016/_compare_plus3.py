import difflib,pathlib
pairs=[
(r'D:\dataset\experiment\hrdecoder_plus3\repo\mmseg\models\segmentors\HRDecoder.py',r'D:\dataset\experiment\HRDecoder_Ready_20261002\vendor\HRDecoder\mmseg\models\segmentors\HRDecoder.py'),
(r'D:\dataset\experiment\hrdecoder_plus3\repo\mmseg\models\losses\binary_loss.py',r'D:\dataset\experiment\HRDecoder_Ready_20261002\vendor\HRDecoder\mmseg\models\losses\binary_loss.py'),
(r'D:\dataset\experiment\hrdecoder_plus3\repo\mmseg\models\decode_heads\fcn_head.py',r'D:\dataset\experiment\HRDecoder_Ready_20261002\vendor\HRDecoder\mmseg\models\decode_heads\fcn_head.py')]
for a,b in pairs:
 A=pathlib.Path(a).read_text(encoding='utf-8',errors='replace').splitlines();B=pathlib.Path(b).read_text(encoding='utf-8',errors='replace').splitlines()
 print('\n===',a,'versus',b,'===')
 for line in difflib.unified_diff(B,A,fromfile='vendor',tofile='plus3',n=2):print(line)
