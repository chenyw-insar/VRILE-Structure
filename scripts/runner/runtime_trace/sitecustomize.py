"""Lightweight Python-level provenance trace; filesystem isolation is external.

Logs each distinct Python open/import/subprocess path once per process. Native
C-library opens are not all visible to Python's audit hook: do NOT describe this
as exhaustive syscall tracing. The namespace enforces access for those as well.
"""
import json
import os
import sys
import time
import threading

_root=os.environ.get('VRILE_TRACE_ROOT')
_pid=None
_fd=None
_seen=set()
_thread=threading.local()
_lock=threading.RLock()

def _after_fork():
    global _thread,_lock
    _thread=threading.local();_lock=threading.RLock()

os.register_at_fork(after_in_child=_after_fork)

def _emit(event,details):
    global _pid,_fd,_seen
    if getattr(_thread,'busy',False):return
    _thread.busy=True
    _lock.acquire()
    try:
        current=os.getpid()
        if _pid!=current:
            if _fd is not None:os.close(_fd)
            _pid=current;_seen=set()
            _fd=os.open(os.path.join(_root,f'{current}_{time.time_ns()}.jsonl'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
            header={'event':'trace_start','pid':current,'ppid':os.getppid(),'executable':sys.executable,'cwd':os.getcwd(),'pythonpath':os.environ.get('PYTHONPATH',''),'phase':os.environ.get('VRILE_EXECUTION_PHASE','UNSPECIFIED_SHORT_OR_PREFLIGHT'),'coverage':'PYTHON_AUDIT_EVENTS_NOT_EXHAUSTIVE_NATIVE_OPENS'}
            os.write(_fd,(json.dumps(header)+'\n').encode())
        record={'event':event,'details':details}
        token=json.dumps(record,sort_keys=True,default=str)
        if token not in _seen:
            _seen.add(token);os.write(_fd,(token+'\n').encode())
    except Exception:
        os._exit(97)
    finally:
        _lock.release()
        _thread.busy=False

def _audit(event,args):
    if event=='open':
        path=args[0]
        if isinstance(path,(str,bytes)):
            text=os.fsdecode(path)
            _emit(event,{'path':os.path.abspath(text),'mode':args[1],'flags':args[2]})
    elif event=='import':
        _emit(event,{'module':args[0],'resolved_file':args[1]})
    elif event=='subprocess.Popen':
        _emit(event,{'executable':args[0],'argv':args[1],'cwd':args[2]})

if _root:
    _emit('trace_enabled',{'time_ns':time.time_ns()})
    sys.addaudithook(_audit)
