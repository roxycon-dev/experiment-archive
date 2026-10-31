"""Lossless tensor deduplication/XOR transport. Model values never quantized."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent


def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def encode(output):
    source=EXP/'HRDecoder_Ready_20261002/best_pipeline/weights'
    names=['B_weights.pth','D_student_2000.pth','E4_weights.pth','E_1500.pth']
    chunks={};bases={};files=[]
    torch.set_num_threads(2)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name in names:
            p=source/name;ck=torch.load(p,map_location='cpu',weights_only=False,mmap=True);state=ck.get('state_dict',ck)
            mapping=[]
            for key,tensor in state.items():
                assert isinstance(tensor,torch.Tensor)
                array=tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
                raw=array.tobytes();ident=digest(raw)
                canonical=re.sub(r'^(module\.|student\.|reference\.|student_model\.|reference_model\.)+','',key)
                basekey=(canonical,str(tensor.dtype),tuple(tensor.shape))
                if ident not in chunks:
                    previous=bases.get(basekey)
                    if previous is not None:
                        base_id,base_array=previous;payload=np.bitwise_xor(array,base_array).tobytes();kind='xor'
                    else:base_id=None;payload=raw;kind='raw'
                    z.writestr('chunks/'+ident,payload)
                    chunks[ident]={'kind':kind,'base_id':base_id,'nbytes':len(raw)}
                bases.setdefault(basekey,(ident,array))
                mapping.append({'key':key,'chunk':ident,'dtype':str(tensor.dtype).split('.')[-1],'shape':list(tensor.shape)})
            files.append({'name':name,'source_sha256':sha(p),'tensors':mapping,'meta':ck.get('meta',{})})
            print(json.dumps({'encoded':name,'unique_chunks':len(chunks),'archive_bytes':output.stat().st_size}),flush=True)
        manifest={'format':'exact-tensor-dedup-xor-v1','lossless':True,'chunks':chunks,'files':files,
                  'note':'Optimizer excluded; metadata serialized with str fallback. All inference state tensors preserve exact original bytes.'}
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,default=str))
    print(json.dumps({'status':'COMPLETE','bytes':output.stat().st_size,'source_file_bytes':sum((source/n).stat().st_size for n in names)}),flush=True)


def decode(archive,target):
    target.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('manifest.json'));decoded={}
        for ident,chunk in manifest['chunks'].items():
            payload=z.read('chunks/'+ident)
            raw=(np.bitwise_xor(np.frombuffer(payload,np.uint8),np.frombuffer(decoded[chunk['base_id']],np.uint8)).tobytes()
                 if chunk['kind']=='xor' else payload)
            assert len(raw)==chunk['nbytes'] and digest(raw)==ident
            decoded[ident]=raw
        receipts=[]
        for row in manifest['files']:
            state={}
            for r in row['tensors']:
                raw=decoded[r['chunk']]
                tensor=torch.from_numpy(np.frombuffer(raw,np.uint8).copy()).view(getattr(torch,r['dtype'])).reshape(r['shape'])
                assert digest(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())==r['chunk']
                state[r['key']]=tensor
            p=target/row['name'];assert not p.exists(),str(p)
            torch.save({'state_dict':state,'meta':row['meta'],'source_sha256':row['source_sha256']},p)
            receipts.append({'name':row['name'],'source_file_sha256':row['source_sha256'],'reconstructed_file_sha256':sha(p),
                             'state_tensors':len(state),'all_original_tensor_bytes_verified':True})
            print(json.dumps(receipts[-1]),flush=True)
        (target/'exact_transport_receipt.json').write_text(json.dumps({'status':'PASS','archive_sha256':sha(archive),'files':receipts},indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['encode','decode']);parser.add_argument('path',type=Path);parser.add_argument('--target',type=Path)
    a=parser.parse_args()
    if a.mode=='encode':encode(a.path)
    else:decode(a.path,a.target)
