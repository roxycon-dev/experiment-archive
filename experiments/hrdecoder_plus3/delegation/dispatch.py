import json,requests,concurrent.futures,sys
from pathlib import Path
root=Path(r'D:\dataset\experiment\hrdecoder_plus3\delegation')
c=json.loads(Path(r'C:\Users\GuanXingRen\.zcode\v2\config.json').read_text(encoding='utf-8-sig'))
p=next(x for x in c['provider'].values() if x.get('name')=='智谱glm')['options']
def run(name):
 try:
  r=requests.post(p['baseURL'].rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+p['apiKey']},json={'model':'glm-5.3-flash','messages':[{'role':'user','content':(root/(name+'_prompt.txt')).read_text(encoding='utf-8-sig')}],'max_tokens':14000 if name.startswith('implementation') else 9000,'reasoning_effort':'low','stream':False},timeout=(15,240))
  j=r.json(); content=j.get('choices',[{}])[0].get('message',{}).get('content')
  if content: (root/(name+'_response.txt')).write_text(content,encoding='utf-8')
  meta={'status':r.status_code,'usage':j.get('usage'),'finish':j.get('choices',[{}])[0].get('finish_reason'),'saved':bool(content),'error':j.get('error',{}).get('message')}
  (root/(name+'_usage.json')).write_text(json.dumps(meta),encoding='utf-8'); print(name,json.dumps(meta),flush=True)
 except Exception as e: print(name,type(e).__name__,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(run,sys.argv[1:] or ['implementation','operations']))
