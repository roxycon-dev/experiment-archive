import csv,gzip,tarfile,zipfile,hashlib,pathlib,sys
root=pathlib.Path(r'D:\dataset\experiment')
csvp=root/'HRDecoder_NativePyramid_20261011/review_20261016/archive_only_source_members.csv'
want=set(sys.argv[1:])
for row in csv.DictReader(csvp.open(encoding='utf-8-sig')):
    if row['member'] not in want: continue
    ap=pathlib.Path(row['archive']); member=row['member']
    if ap.suffix.lower()=='.zip':
        with zipfile.ZipFile(ap) as z: data=z.read(member)
    else:
        with tarfile.open(ap,'r:*') as t: data=t.extractfile(member).read()
    assert hashlib.sha256(data).hexdigest()==row['sha256']
    print(f'===== {ap.name} :: {member} :: {row["sha256"]} :: {len(data)} bytes =====')
    print(data.decode('utf-8-sig'))
