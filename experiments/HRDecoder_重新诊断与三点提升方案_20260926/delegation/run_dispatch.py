import importlib.util, json, sys
from pathlib import Path
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('dispatch',r'C:\Users\GuanXingRen\.codex\skills\local-model-dispatch\scripts\dispatch.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
task=sys.argv[1]
prompt=(root/(task+'.txt')).read_text(encoding='utf-8')
try:
    r=m.workbuddy_run('deepseek-v4.1-flash',prompt,1800,cwd=str(root.parent))
    (root/(task+'_response.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/(task+'_response.md')).write_text(r['text'],encoding='utf-8')
    print(task,'DONE',r['text'][-2500:])
except Exception as e:
    (root/(task+'_error.txt')).write_text(str(e),encoding='utf-8');raise
