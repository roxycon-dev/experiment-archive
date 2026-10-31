import ast, csv, hashlib, json, math, os, re, sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(r"D:\dataset\experiment")
REVIEW = ROOT / "HRDecoder_NativePyramid_20261011" / "review_20261016"
INV_PATH = REVIEW / "runs_file_hash_inventory.json"

def rel(p):
    try: return str(p.relative_to(ROOT)).replace("\\", "/")
    except Exception: return str(p)

def safe_read(p):
    b=p.read_bytes()
    for enc in ("utf-8-sig","utf-8","gb18030","cp1252"):
        try: return b.decode(enc), enc
        except Exception: pass
    return b.decode("utf-8",errors="replace"), "utf-8-replace"

def redact_key(k):
    return any(x in str(k).lower() for x in ("password","passwd","secret","token","credential","access_key","private_key"))

def scalar(v):
    return v is None or isinstance(v,(str,int,float,bool))

def json_safe(v,k=None):
    if k is not None and redact_key(k): return "[REDACTED]"
    if scalar(v):
        if isinstance(v,float) and (math.isnan(v) or math.isinf(v)): return str(v)
        return v
    if isinstance(v,dict):
        return {str(kk):json_safe(vv,kk) for kk,vv in v.items() if not redact_key(kk)}
    if isinstance(v,(list,tuple)):
        return [json_safe(x) for x in v[:5000]]
    return str(type(v).__name__)

def write_jsonl(path, objs):
    with path.open("w",encoding="utf-8",newline="\n") as f:
        for obj in objs: f.write(json.dumps(obj,ensure_ascii=False,separators=(",",":"))+"\n")

inventory=json.loads(INV_PATH.read_text(encoding="utf-8"))
# Fill missing hashes for the remaining text-like extensions and persist a complete SHA index.
for r in inventory:
    if not r.get("sha256"):
        r["sha256"] = hashlib.sha256((ROOT/Path(r["path"])).read_bytes()).hexdigest()
INV_PATH.write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding="utf-8")
byext=defaultdict(list)
bysha=defaultdict(list)
for r in inventory:
    byext[r["ext"]].append(r)
    bysha[(r["ext"],r["sha256"])].append(r)

# Parse every unique JSON artifact fully. Emit scalar metric-bearing object nodes only.
metric_key_re=re.compile(r"(miou|iou|dice|tp|fp|fn|precision|recall|loss|step|update|iter|count|score|class|threshold|status|exit|stop|resume|complete|collapse|nan|inf)",re.I)
json_node_rows=[]
json_parse_counts=Counter()
json_aliases={}
json_unique=0
for (ext,sha), entries in sorted(bysha.items()):
    if ext!=".json": continue
    paths=sorted(e["path"] for e in entries)
    rep=paths[0]
    p=ROOT / Path(rep)
    json_unique+=1
    try:
        text,_=safe_read(p)
        data=json.loads(text)
        json_parse_counts["parsed"]+=1
    except Exception as e:
        json_parse_counts["failed"]+=1
        json_node_rows.append({"source":rep,"sha256":sha,"parse":"failed","error":type(e).__name__})
        continue
    json_aliases[rep]=paths
    stack=[("$",data)]
    nodes=0
    while stack:
        ptr,obj=stack.pop()
        if isinstance(obj,dict):
            selected={k:json_safe(v,k) for k,v in obj.items()
                      if metric_key_re.search(str(k)) and scalar(v) and not redact_key(k)}
            # Keep metric-bearing objects, or state objects with explicit lifecycle field.
            if selected:
                json_node_rows.append({"source":rep,"sha256":sha,"pointer":ptr,"fields":selected})
                nodes+=1
            for k,v in obj.items():
                if isinstance(v,(dict,list)):
                    stack.append((ptr+"/"+str(k).replace("~","~0").replace("/","~1"),v))
        elif isinstance(obj,list):
            for i,v in enumerate(obj):
                if isinstance(v,(dict,list)):
                    stack.append((ptr+"/"+str(i),v))
    if nodes==0: json_parse_counts["no_scalar_metric_nodes"]+=1
    json_parse_counts["metric_nodes"]+=nodes
write_jsonl(REVIEW/"runs_json_metric_nodes.jsonl",json_node_rows)

# Parse every unique CSV, keep every row structured, with secret-named columns redacted.
csv_rows=[]
csv_registry=[]
csv_unique=0
for (ext,sha), entries in sorted(bysha.items()):
    if ext!=".csv": continue
    paths=sorted(e["path"] for e in entries)
    rep=paths[0]
    p=ROOT/Path(rep)
    csv_unique+=1
    try:
        text,encoding=safe_read(p)
        reader=csv.reader(text.splitlines())
        all_rows=list(reader)
        if not all_rows:
            headers=[]; data_rows=[]; headerless=False
        else:
            # Detect conventional telemetry records by timestamp-like first cell / no alphabetic header.
            first=all_rows[0]
            has_header=any(re.search(r"[A-Za-z_]",cell or "") for cell in first) and not re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}", first[0] if first else "")
            headerless=not has_header
            headers=first if has_header else []
            data_rows=all_rows[1:] if has_header else all_rows
        csv_registry.append({"source":rep,"aliases":paths,"sha256":sha,"bytes":entries[0]["bytes"],"encoding":encoding,"headerless":headerless,"headers":headers,"column_count":max((len(r) for r in all_rows),default=0),"rows":len(data_rows)})
        for i,row in enumerate(data_rows,1):
            if headers:
                obj={h:(row[j] if j<len(row) else "") for j,h in enumerate(headers) if not redact_key(h)}
                # Redact common secret fields even when column names are nonstandard.
                clean=json_safe(obj)
            else: clean=[json_safe(x) for x in row]
            csv_rows.append({"source":rep,"sha256":sha,"row":i,"values":clean})
    except Exception as e:
        csv_registry.append({"source":rep,"aliases":paths,"sha256":sha,"parse_error":type(e).__name__})
write_jsonl(REVIEW/"runs_csv_rows.jsonl",csv_rows)
(REVIEW/"runs_csv_registry.json").write_text(json.dumps(csv_registry,ensure_ascii=False,indent=2),encoding="utf-8")

# Parse all unique log/out/err artifacts over every line, recording metric/lifecycle events without retaining raw lines.
event_re=re.compile(r"(miou|iou|dice|tp\s*[=:]|fp\s*[=:]|fn\s*[=:]|loss|nan|inf|oom|out of memory|traceback|exception|error|stopp?ed|abort|pause|resume|checkpoint|complete|failed|exit code|update|iter(?:ation)?|step)",re.I)
step_re=re.compile(r"\b(?:step|iter(?:ation)?|update|u)\s*[=: #]?\s*(\d{1,9})\b",re.I)
num_re=r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
metric_patterns={
 "mIoU":re.compile(r"\b(?:mIoU|miou)\b[^0-9+\-]{0,20}("+num_re+r")",re.I),
 "IoU":re.compile(r"\bIoU\b[^0-9+\-]{0,20}("+num_re+r")",re.I),
 "loss":re.compile(r"\bloss\b[^0-9+\-]{0,20}("+num_re+r")",re.I),
 "TP":re.compile(r"\bTP\b\s*[=:]\s*(\d+)",re.I),
 "FP":re.compile(r"\bFP\b\s*[=:]\s*(\d+)",re.I),
 "FN":re.compile(r"\bFN\b\s*[=:]\s*(\d+)",re.I),
}
log_rows=[]
log_counts=Counter()
log_aliases=[]
for ext in (".log",".out",".err"):
    for (e,sha), entries in sorted(bysha.items()):
        if e!=ext: continue
        paths=sorted(x["path"] for x in entries)
        rep=paths[0]
        p=ROOT/Path(rep)
        text,encoding=safe_read(p)
        line_count=0
        for lineno,line in enumerate(text.splitlines(),1):
            line_count+=1
            if not event_re.search(line): continue
            stepm=step_re.search(line)
            fields={k:(m.group(1) if (m:=pat.search(line)) else None) for k,pat in metric_patterns.items()}
            states=[word for word in ("OOM","out of memory","Traceback","Exception","Error","STOPPED","ABORT","PAUSED","resum","checkpoint","COMPLETE","FAILED","exit code","nan","inf") if re.search(re.escape(word),line,re.I)]
            log_rows.append({"source":rep,"aliases":paths,"sha256":sha,"line":lineno,"step":int(stepm.group(1)) if stepm else None,"metrics":fields,"states":states})
        log_counts["unique_files"]+=1
        log_counts["line_count"]+=line_count
        log_aliases.append({"source":rep,"sha256":sha,"aliases":paths,"line_count":line_count,"encoding":encoding})
write_jsonl(REVIEW/"runs_log_events.jsonl",log_rows)
(REVIEW/"runs_log_registry.json").write_text(json.dumps(log_aliases,ensure_ascii=False,indent=2),encoding="utf-8")

# Safe parse YAML and static-parse hand-maintained Python config-like files (never import/execute).
try:
    import yaml
    yaml_version=yaml.__version__
except Exception:
    yaml=None; yaml_version=None
yaml_records=[]
for ext in (".yaml",".yml"):
    for (e,sha),entries in sorted(bysha.items()):
        if e!=ext: continue
        rep=sorted(x["path"] for x in entries)[0]
        text,enc=safe_read(ROOT/Path(rep))
        rec={"source":rep,"sha256":sha,"aliases":sorted(x["path"] for x in entries),"bytes":entries[0]["bytes"],"encoding":enc}
        if yaml is None: rec["status"]="pyyaml_unavailable"
        else:
            try:
                obj=yaml.safe_load(text)
                rec["status"]="parsed"
                rec["top_level_type"]=type(obj).__name__
                rec["top_level_keys"]=list(obj.keys())[:150] if isinstance(obj,dict) else []
                # Capture a shallow, relevant config projection; never persist full config.
                relevant={}
                def walk(x,prefix=""):
                    if isinstance(x,dict):
                        for k,v in x.items():
                            key=str(k)
                            if redact_key(key): continue
                            path=(prefix+"."+key).strip(".")
                            if re.search(r"(max.?iter|iter|update|batch|crop|canvas|img.?scale|lr|optim|loss|pretrain|init|backbone|num.?class|threshold|seed|weight.?decay|momentum|bn|norm|scheduler)",key,re.I) and scalar(v):
                                relevant[path]=json_safe(v,key)
                            if isinstance(v,(dict,list)): walk(v,path)
                    elif isinstance(x,list):
                        for i,v in enumerate(x[:500]): walk(v,prefix+f"[{i}]")
                walk(obj)
                rec["relevant_scalars"]=relevant
            except Exception as ex:
                rec["status"]="parse_error"; rec["error"]=type(ex).__name__
        yaml_records.append(rec)
(REVIEW/"runs_yaml_inventory.json").write_text(json.dumps({"pyyaml_version":yaml_version,"unique_files":len(yaml_records),"records":yaml_records},ensure_ascii=False,indent=2),encoding="utf-8")

py_records=[]
for p in ROOT.rglob("*.py"):
    if "review_20261016" in p.parts: continue
    relp=rel(p).lower()
    if not ("config" in p.name.lower() or "configs" in relp.split("/")): continue
    if any(part in {"vendor","build","dist-packages","site-packages"} for part in p.parts): continue
    b=p.read_bytes(); sha=hashlib.sha256(b).hexdigest()
    text,enc=safe_read(p)
    rec={"source":rel(p),"sha256":sha,"bytes":len(b),"encoding":enc}
    try:
        tree=ast.parse(text,filename=rel(p))
        rec["status"]="ast_parsed"
        rec["module_assignments"]=[]
        for node in tree.body:
            if isinstance(node,(ast.Assign,ast.AnnAssign)):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                names=[t.id for t in targets if isinstance(t,ast.Name)]
                if not names: continue
                try: value=ast.literal_eval(node.value)
                except Exception: value=None
                if isinstance(value,dict):
                    rec["module_assignments"].append({"names":names,"keys":list(value.keys())[:100],"literal":json_safe({k:v for k,v in value.items() if not redact_key(k)}) if len(str(value))<20000 else "[large literal omitted]"})
                elif any(re.search(r"(cfg|config|lr|optim|batch|iter|crop|seed|weight|model)",n,re.I) for n in names):
                    rec["module_assignments"].append({"names":names,"literal_value":json_safe(value) if scalar(value) else "[nonliteral]"})
    except Exception as ex:
        rec["status"]="ast_parse_error"; rec["error"]=type(ex).__name__
    py_records.append(rec)
(REVIEW/"runs_python_config_inventory.json").write_text(json.dumps({"candidate_count":len(py_records),"records":py_records},ensure_ascii=False,indent=2),encoding="utf-8")

# Curated ArchitectureDesign validation table, retaining every available per-class node and count.
main_csv=ROOT/"HRDecoder_ArchitectureDesign_20261008"/"final_report"/"validation_metrics.csv"
rows=list(csv.DictReader(main_csv.open(encoding="utf-8-sig",newline="")))
groups=defaultdict(list)
for r in rows: groups[(r["run"],int(r["step"]))].append(r)
class_names=["EX","HE","SE","MA"]
field_suffixes=["IoU_percent","precision_percent","recall_percent","TP","FP","FN"]
node_table=[]
for (run,step),rs in sorted(groups.items(),key=lambda z:(z[0][0],z[0][1])):
    first=rs[0]
    out={"run":run,"step":step,"mIoU_percent":first.get("mIoU_percent"),"base_mIoU_percent":first.get("base_mIoU_percent"),"source":rel(main_csv)}
    byclass={r["class"]:r for r in rs}
    for cl in class_names:
        rr=byclass.get(cl,{})
        for f in field_suffixes:
            out[f"{cl}_{f}"]=rr.get(f)
    node_table.append(out)
with (REVIEW/"runs_validation_nodes.csv").open("w",encoding="utf-8-sig",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(node_table[0].keys()));w.writeheader();w.writerows(node_table)

protocol_base=ROOT/"HRDecoder_ArchitectureDesign_20261008"/"cache_execution"/"server_receipts"
run_protocols={}
for run in sorted({r["run"] for r in rows}):
    cand=[protocol_base/run/"training_runs"/run/"protocol.json",
          ROOT/"HRDecoder_ArchitectureDesign_20261008"/"training_runs"/run/"protocol.json"]
    for p in cand:
        if p.exists():
            try:
                d=json.loads(p.read_text(encoding="utf-8-sig"))
                pr=d.get("identity",{}).get("protocol",d.get("protocol",{}))
                run_protocols[run]={"source":rel(p),"identity_SHA":d.get("identity_SHA"),"baseline_initial_state_SHA":d.get("baseline_initial_state_SHA"),
                    "resume_from_own_update":d.get("resume_from_own_update"),
                    "name":pr.get("name"),"seed":pr.get("seed"),"private_seed":pr.get("private_seed"),
                    "canvas":pr.get("canvas"),"updates":pr.get("updates"),"global_batch":pr.get("global_batch"),
                    "optimizer":pr.get("optimizer"),"lr":pr.get("lr"),"lambda_k":pr.get("lambda_k"),"lambda_action":pr.get("lambda_action"),
                    "GT":pr.get("GT"),"KD":pr.get("KD"),"readout":pr.get("readout"),"gradient_route":pr.get("gradient_route"),
                    "mask":pr.get("mask"),"initialization":pr.get("initialization"),"screen_updates":pr.get("screen_updates")}
                break
            except Exception as ex:
                run_protocols[run]={"source":rel(p),"parse_error":type(ex).__name__}
# v3 direct path varies; locate exact unique matching path by recursive indexed filenames
for p in ROOT.rglob("protocol.json"):
    if "review_20261016" in p.parts: continue
    if re.search(r"training_runs[/\\]v3[/\\]protocol\.json$",str(p),re.I):
        try:
            d=json.loads(p.read_text(encoding="utf-8-sig")); pr=d.get("identity",{}).get("protocol",d.get("protocol",{}))
            run_protocols["v3"]={"source":rel(p),"identity_SHA":d.get("identity_SHA"),"name":pr.get("name"),"seed":pr.get("seed"),"private_seed":pr.get("private_seed"),"canvas":pr.get("canvas"),"updates":pr.get("updates"),"global_batch":pr.get("global_batch"),"optimizer":pr.get("optimizer"),"lr":pr.get("lr"),"lambda_k":pr.get("lambda_k"),"GT":pr.get("GT"),"KD":pr.get("KD"),"readout":pr.get("readout"),"gradient_route":pr.get("gradient_route"),"initialization":pr.get("initialization")}
        except: pass

# One experiment row per run with all validation nodes, best observed node and trajectory.
exp_rows=[]
for run in sorted({r["run"] for r in rows}):
    ns=[x for x in node_table if x["run"]==run]
    vals=[(int(x["step"]),float(x["mIoU_percent"])) for x in ns if x["mIoU_percent"] not in ("",None)]
    best=max(vals,key=lambda x:x[1]) if vals else (None,None)
    final=vals[-1] if vals else (None,None)
    trajectory=";".join(f"{s}:{v:.6f}" for s,v in vals)
    perclass={}
    for cl in class_names:
        perclass[cl]=";".join(f"{int(x['step'])}:{float(x[f'{cl}_IoU_percent']):.3f}" for x in ns if x.get(f"{cl}_IoU_percent") not in ("",None))
    proto=run_protocols.get(run,{})
    exp_rows.append({"run":run,"protocol_name":proto.get("name"),"source_protocol":proto.get("source"),"identity_SHA":proto.get("identity_SHA"),"seed":proto.get("seed"),"canvas":proto.get("canvas"),"planned_updates":proto.get("updates"),"global_batch":proto.get("global_batch"),"optimizer":json.dumps(proto.get("optimizer"),ensure_ascii=False) if proto.get("optimizer") else None,"initialization":proto.get("initialization"),"observed_steps":";".join(str(s) for s,v in vals),"mIoU_trajectory":trajectory,"best_step":best[0],"best_mIoU":best[1],"last_step":final[0],"last_mIoU":final[1],"class_IoU_trajectories":json.dumps(perclass,ensure_ascii=False),"source_metrics":rel(main_csv)})
with (REVIEW/"runs_experiment_table.csv").open("w",encoding="utf-8-sig",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(exp_rows[0].keys()));w.writeheader();w.writerows(exp_rows)
(REVIEW/"runs_protocol_catalog.json").write_text(json.dumps(run_protocols,ensure_ascii=False,indent=2),encoding="utf-8")

summary={
 "root":str(ROOT),"review_dir":str(REVIEW),"source_inventory":"runs_file_hash_inventory.json",
 "raw_textlike_inventory_count":len(inventory),
 "raw_logs":{ "physical_files":sum(len(bysha[(ext,sha)]) for ext in (".log",".out",".err") for e,sha in bysha if e==ext),
              "physical_log_artifacts_scanned_full_line_by_line":sum(1 for r in inventory if r["ext"] in (".log",".out",".err")),
              "unique_logs_by_sha":log_counts["unique_files"],"unique_log_lines_scanned":log_counts["line_count"],"structured_log_event_rows":len(log_rows),
              "event_output":"runs_log_events.jsonl","registry":"runs_log_registry.json"},
 "raw_json":{"physical_files":len(byext[".json"]),"unique_by_sha":json_unique,"valid_unique_parsed":json_parse_counts["parsed"],"invalid_unique":json_parse_counts["failed"],"metric_node_rows":json_parse_counts["metric_nodes"],"output":"runs_json_metric_nodes.jsonl"},
 "raw_jsonl":{"physical_files":len(byext[".jsonl"]),"scan_source":"runs_jsonl_audit.json","note":"all physical valid lines previously parsed; duplicates grouped by SHA in runs_file_hash_inventory.json"},
 "raw_csv":{"physical_files":len(byext[".csv"]),"unique_by_sha":csv_unique,"structured_rows":len(csv_rows),"output":"runs_csv_rows.jsonl","registry":"runs_csv_registry.json","headerless_files":sum(1 for r in csv_registry if r.get("headerless"))},
 "raw_yaml":{"physical_files":len(byext[".yaml"])+len(byext[".yml"]),"unique_by_sha":len(yaml_records),"safe_parsed":sum(r.get("status")=="parsed" for r in yaml_records),"output":"runs_yaml_inventory.json"},
 "python_config_candidates":len(py_records),"python_config_output":"runs_python_config_inventory.json",
 "curated_validation":{"source":rel(main_csv),"source_rows":len(rows),"nodes":len(node_table),"runs":len(exp_rows),"outputs":["runs_validation_nodes.csv","runs_experiment_table.csv","runs_protocol_catalog.json"]},
 "duplicate_reduction":{"unique_text_hash_groups":len(bysha),"physical_copies_excess":sum(len(v)-1 for v in bysha.values() if len(v)>1),
                         "duplicate_groups_by_ext":{ext:sum(1 for (e,sha),v in bysha.items() if e==ext and len(v)>1) for ext in sorted(byext)},
                         "excess_copies_by_ext":{ext:sum(len(v)-1 for (e,sha),v in bysha.items() if e==ext and len(v)>1) for ext in sorted(byext)}},
 "limits":["No model checkpoints executed or deserialized. No raw image/mask pixels were read.",
           "JSON and CSV files were parsed structurally after exact-content deduplication; JSONL physical copies were parsed by prior inventory scan and are SHA-indexed.",
           "Logs were fully scanned line by line after exact-content deduplication, with structured metric/lifecycle extraction; no assertion that every narrative line was manually interpreted.",
           "YAML used safe_load only. Python config candidates used AST parsing only, never import/execution.",
           "Some empty/truncated JSON records are listed as parse failures and must not be treated as absent runs or inferred outcomes."] }
(REVIEW/"runs_coverage.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"json":dict(json_parse_counts),"csv_unique":csv_unique,"csv_rows":len(csv_rows),"log_counts":dict(log_counts),"log_events":len(log_rows),"yaml_unique":len(yaml_records),"yaml_parsed":sum(r.get("status")=="parsed" for r in yaml_records),"py_configs":len(py_records),"validation_runs":len(exp_rows),"validation_nodes":len(node_table)},ensure_ascii=False))


