"""One user-started server; acceptance before finite campaign, no blind restart."""
import argparse,json,os,traceback
from qlib.common import read_json,write_json,sha,utc_now,assert_source
from experiment import ROOT

def main(resume_paused=False):
    # Fork workers inherit the parent thread pools. Initialize both pools
    # before model/validation work to prevent their locks blocking children.
    import torch,cv2
    torch.set_num_threads(1);cv2.setNumThreads(1)
    lock=ROOT/'reports/driver.lock';report=ROOT/'reports/driver_state.json'
    if report.exists():
        state=read_json(report)
        if state['stage']=='READY_FOR_LOCAL_BACKUP_AND_SHUTDOWN':return
        assert resume_paused,'Existing driver requires explicit diagnosed recovery'
    with lock.open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'time':utc_now()},f)
    try:
        source=assert_source(ROOT)
        gate=ROOT/'reports/gpu_readiness.json'
        if not gate.exists() or read_json(gate)['status']!='READY':
            assert not resume_paused,'Do not repeat an incomplete GPU acceptance blindly; diagnose its failed gate first'
            write_json(report,{'stage':'GPU_ACCEPTANCE','pid':os.getpid(),'time':utc_now()})
            from tests.gpu_acceptance import main as acceptance
            acceptance()
        else:
            receipt=read_json(gate);assert receipt['source_sha256']==source['sha256'] and receipt['runtime_sha256']==sha(ROOT/'configs/runtime.server.json')
        write_json(report,{'stage':'CAMPAIGN','pid':os.getpid(),'time':utc_now()})
        from campaign import main as campaign
        campaign(resume_paused=resume_paused)
        state=read_json(ROOT/'reports/campaign.json')
        write_json(report,{'stage':state['stage'],'campaign':state,'updated_at':utc_now()})
    except BaseException as e:
        write_json(report,{'stage':'TECHNICAL_PAUSE','reason':str(e),'automatic_restart':False,'updated_at':utc_now()});traceback.print_exc();raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume-paused',action='store_true');a=p.parse_args();main(a.resume_paused)
