import concurrent.futures, json, sys
from pathlib import Path
import requests
sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent.parent
cfg = json.loads((Path.home()/'.zcode/v2/config.json').read_text(encoding='utf-8-sig'))
provider = next(x for x in cfg['provider'].values() if x.get('name') == '智谱glm')['options']
def run(name):
    prompt = (ROOT/'delegation'/(name+'_prompt.txt')).read_text(encoding='utf-8')
    try:
        r = requests.post(provider['baseURL'].rstrip('/')+'/chat/completions', headers={'Authorization':'Bearer '+provider['apiKey']}, json={'model':'glm-5.3-flash','messages':[{'role':'user','content':prompt}],'max_tokens':20000,'reasoning_effort':'low','stream':False},timeout=(15,360))
        j=r.json()
        meta={'status':r.status_code,'usage':j.get('usage'),'error':j.get('error'),'finish':j.get('choices',[{}])[0].get('finish_reason')}
        (ROOT/'delegation'/(name+'_meta.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        s=j.get('choices',[{}])[0].get('message',{}).get('content','')
        (ROOT/'delegation'/(name+'_response.txt')).write_text(s,encoding='utf-8')
        s=s.strip()
        if s.startswith('```'): s=s.split('\n',1)[1].rsplit('```',1)[0]
        try:
            d=json.loads(s)
        except json.JSONDecodeError:
            import re
            original=(ROOT/'delegation'/(name+'_response.txt')).read_text(encoding='utf-8')
            blocks=re.findall(r'```(?:python|py)\s*\n(.*?)```',original,re.S)
            files={}
            for b in blocks:
                first=b.splitlines()[0].strip()
                if first.startswith('# ') and first[2:].strip().endswith('.py'):
                    files[first[2:].strip()]=b
            if not files: raise
            d={'files':files,'notes':'Recovered named Python blocks from GLM response.'}
        for rel,content in d['files'].items():
            dest=(ROOT/rel).resolve(); dest.relative_to(ROOT.resolve())
            if dest.suffix not in {'.py','.md','.json','.sh','.txt'}: raise ValueError('Unexpected file type')
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(content,encoding='utf-8',newline='\n')
        print(name,json.dumps({'files':list(d['files']),'notes':d.get('notes'),'usage':j.get('usage')},ensure_ascii=False),flush=True)
    except Exception as e: print(name,type(e).__name__,str(e)[:300],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run,sys.argv[1:]))
