"""Floor-only spawn executor: engineering scheduling, no scientific formulas.

Send a function NAME and its unchanged payload, never the dynamically aliased
M009 function. Spawned interpreters import this stable worker module and load
their own M009 implementation. No infrastructure-to-serial fallback is allowed.
"""
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time

_core=None
_timing=None
ALLOWED={'stage6_event_buffer_rows_task','stage6_wrong_region_rows_task','stage8_control_group_task','stage8_paired_wrong_region_task','stage9_cyclone_event_rows_task','stage10_ice_edge_event_task'}
def _scientific_fds():
    files=[]
    for p in Path('/proc/self/fd').iterdir():
        try:target=os.readlink(p)
        except OSError:continue
        if target.lower().endswith(('.nc','.nc4','.h5','.hdf','.hdf5')):files.append(target)
    return sorted(files)
def _append(directory,record):
    path=Path(directory)/('worker_'+str(os.getpid())+'.jsonl')
    with path.open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n')
def _initialize(root,timing):
    global _core,_timing
    _timing=timing
    inherited=_scientific_fds()
    if inherited:raise RuntimeError('HOLD_FLOOR015_INHERITED_SCIENTIFIC_HANDLES:'+repr(inherited))
    root=Path(root);os.environ['VRILE_PROJECT_ROOT']=str(root)
    sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('vrile_floor015_worker_stage2',root/'scripts/stage2_experiment_core.py')
    if spec is None or spec.loader is None:raise RuntimeError('HOLD_FLOOR015_WORKER_IMPORT')
    _core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=_core;spec.loader.exec_module(_core)
    _append(timing,{'event':'worker_initialized','pid':os.getpid(),'ppid':os.getppid(),'backend':'process_spawn','inherited_scientific_handles':inherited,'source':str(root/'scripts/stage2_experiment_core.py')})
def _execute(task):
    name,payload,index=task
    if name not in ALLOWED:raise RuntimeError('HOLD_FLOOR015_UNKNOWN_WORKER:'+name)
    start=time.monotonic_ns()
    try:result=getattr(_core,name)(payload)
    except BaseException as e:
        _append(_timing,{'event':'task_error','pid':os.getpid(),'task':index,'function':name,'error':repr(e),'started_ns':start,'ended_ns':time.monotonic_ns()});raise
    _append(_timing,{'event':'task_complete','pid':os.getpid(),'task':index,'function':name,'started_ns':start,'ended_ns':time.monotonic_ns(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'own_scientific_handles_after_task':_scientific_fds()})
    return result
@contextmanager
def installed(stage2,root,output_dir):
    """Route only this floor wrapper's map calls; restore the M009 module after."""
    original=stage2.run_with_fallback
    timing=Path(output_dir)/'floor015_execution_timing';timing.mkdir(exist_ok=False)
    batch=0
    def mapped(func,items,*,workers=None,parallel=False,backend='process',logger=None):
        nonlocal batch
        if func.__name__ not in ALLOWED:raise RuntimeError('HOLD_FLOOR015_UNREGISTERED_MAP')
        tasks=list(items);batch+=1;folder=timing/f'{batch:02d}_{func.__name__}';folder.mkdir()
        start=time.monotonic();effective=max(1,min(int(workers or 1),len(tasks)))
        use_process=bool(parallel and effective>1)
        record={'function':func.__name__,'requested_workers':workers,'task_count':len(tasks),'effective_workers':effective,'requested_backend':backend,'actual_backend':'process_spawn' if use_process else 'serial_reference','fallback_count':0,'parent_pid':os.getpid()}
        try:
            if use_process:
                out=[None]*len(tasks)
                with ProcessPoolExecutor(max_workers=effective,mp_context=multiprocessing.get_context('spawn'),initializer=_initialize,initargs=(str(root),str(folder))) as ex:
                    futures={ex.submit(_execute,(func.__name__,item,i)):i for i,item in enumerate(tasks)}
                    for future in as_completed(futures):out[futures[future]]=future.result()
            else:
                out=[]
                for i,item in enumerate(tasks):
                    begin=time.monotonic_ns();out.append(func(item))
                    _append(folder,{'event':'task_complete','pid':os.getpid(),'task':i,'function':func.__name__,'started_ns':begin,'ended_ns':time.monotonic_ns(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            record['status']='PASS'
            return out
        except BaseException as e:record.update(status='FAIL',error=repr(e));raise
        finally:
            record['wall_seconds']=time.monotonic()-start
            with (folder/'batch.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    stage2.run_with_fallback=mapped
    try:yield
    finally:stage2.run_with_fallback=original
