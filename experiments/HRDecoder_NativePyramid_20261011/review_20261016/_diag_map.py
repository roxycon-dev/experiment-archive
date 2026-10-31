import json,pathlib
root=pathlib.Path(r'D:\dataset\experiment\HRDecoder_NativePyramid_20261011/review_20261016')
inv=json.loads((root/'inventory_all.json').read_text(encoding='utf-8'))
want={'937a06538f57f5afb2e1cd408eb2935ddb332d4e357280e3e72ecdc9bffb459a','7f9884b235ada4d6a4cfe2ba395d03813268fe787eb8088ba340c4ea4f71d9e0','feaa589bd99138372d847e6106579d1920dc73727b744dd786c75effab9ef76d','a3798001fee9bc248e554b92e1a3addb75277533e517b91d4c1ee1ef66b0962c','da1cd6aebe8c663b72b3ab5b3f9f6d50d172978cb091da531e027cbdde140bc0','1059c248177b608b2dbba068de9bd2f2dca5b8cb9bb6e49443be46410c919738','0d16a687b59442564ccf6dabc72440fb5086e6fe4fc20cea21c12ac9d5351a83','f9097dd589460eb20558da9e73dae754826a73719ff19d69fc66e0e07fee6a31','60201eda40e6b3bdf63b9d583668f88e1c4eef2be111df89dbe5febdf20b347c','7065b21899fcb36c824578a5c3d49f915b295b844eeb7856970951837cd707f7','3e5abeb8360c49880130a933c41fd68f84dae777325346bd8af69a73d24d917'}
for x in inv['files']:
 if x.get('sha256') in want: print(x['sha256'],x['bytes'],x['path'])
