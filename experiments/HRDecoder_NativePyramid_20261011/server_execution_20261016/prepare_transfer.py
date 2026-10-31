"""Copy accepted image/label bytes into a transfer archive; no pixel decoding."""
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parents[1]
DATA = Path('D:/HRDecoder/data/DDR_split_613')
identity = json.loads((HERE / 'preparation_20261016/data_identity.json').read_text(encoding='utf-8'))
output = HERE / 'server_execution_20261016/accepted_fused_data.tar'
count = 0
with tarfile.open(output, 'w') as archive:
    for split in ('train', 'val', 'test'):
        images = {p.stem:p for p in (DATA/'images'/split).iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')}
        for row in identity['splits'][split]['samples']:
            image = images[row['sid']]
            label = DATA/'labels'/split/(row['sid']+'.png')
            archive.add(image, arcname=f'images/{split}/{image.name}', recursive=False)
            archive.add(label, arcname=f'labels/{split}/{label.name}', recursive=False)
            count += 2
print(json.dumps({'archive':str(output),'bytes':output.stat().st_size,'files':count,
                  'pixel_decoding':False,'test_evaluation':False}))
