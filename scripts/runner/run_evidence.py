"""Append-only run identity/snapshot/command evidence; no scientific logic."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid
from release_identity import manifest_name, read_identity

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write_new(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(data,stream,indent=2,sort_keys=True);stream.write('\n')

def inventory(root):
    """Never follow links (especially data/raw); retain exact link targets."""
    rows=[]
    for base,dirs,files in os.walk(root,followlinks=False):
        for name in sorted(dirs+files):
            path=Path(base)/name
            relative=path.relative_to(root).as_posix()
            if path.is_symlink():
                rows.append({'path':relative,'kind':'symlink','target':os.readlink(path),'realpath':str(path.resolve()),'exists':path.exists()})
            elif path.is_file():
                rows.append({'path':relative,'kind':'file','size_bytes':path.stat().st_size,'sha256':sha(path)})
            elif path.is_dir():rows.append({'path':relative,'kind':'directory'})
    return sorted(rows,key=lambda r:r['path'])

def init(run,project):
    identity={'run_id':run.name,'execution_id':str(uuid.uuid4()),'initialized_utc':now(),
              'run_root':str(run.resolve()),'project_root':str(project.resolve()),
              'release_identity':read_identity(project),
              'candidate_manifest_sha256':sha(project/manifest_name(project)),
              'candidate_sha256_ledger':sha(project/'sha256.txt')}
    write_new(run/'state/execution_identity.json',identity)
    write_new(run/'evidence/initial_snapshot.json',{
        'identity':identity,'captured_utc':now(),'scope':'ACTUAL_INITIALIZED_TREE_BEFORE_FIRST_SCIENCE',
        'self_reference_rule':'snapshot captures tree before its own creation; raw symlinks never followed',
        'entries':inventory(run),
    })

def identity(run):
    data=json.loads((run/'state/execution_identity.json').read_text())
    project=Path(data['project_root'])
    if data['run_root']!=str(run.resolve()) or data['candidate_manifest_sha256']!=sha(project/manifest_name(project)) or data['candidate_sha256_ledger']!=sha(project/'sha256.txt') or data['release_identity']!=read_identity(project):
        raise ValueError('run/candidate identity changed')
    return data

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['init','start','end'])
    parser.add_argument('--run_dir',type=Path,required=True, dest='run_root')
    parser.add_argument('--project-root',type=Path)
    parser.add_argument('--label')
    parser.add_argument('--command-exit',type=int)
    parser.add_argument('--tee-exit',type=int)
    args,command=parser.parse_known_args()
    args.command=command[1:] if command[:1]==['--'] else command
    if args.action!='start' and args.command:raise ValueError('unexpected arguments')
    if args.action=='init':init(args.run_root,args.project_root);return
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',args.label or ''):raise ValueError('unsafe command label')
    data=identity(args.run_root)
    location=args.run_root/'evidence/commands'
    if args.action=='start':
        if not args.command:raise ValueError('missing actual command')
        write_new(location/(args.label+'.start.json'),{'identity':data,'label':args.label,'started_utc':now(),'argv':args.command,'pid':os.getpid(),'parent_pid':os.getppid(),'cwd':str(Path.cwd()),'pythonpath':os.environ.get('PYTHONPATH','')})
    else:
        start=location/(args.label+'.start.json')
        before=json.loads(start.read_text())
        if before['identity']!=data:raise ValueError('command start/run candidate mismatch')
        if args.command_exit is None or args.tee_exit is None:raise ValueError('actual exits required')
        log=args.run_root/'logs'/(args.label+'.log')
        write_new(location/(args.label+'.end.json'),{'identity':data,'label':args.label,'start_sha256':sha(start),'ended_utc':now(),'command_exit':args.command_exit,'tee_exit':args.tee_exit,'log':str(log),'log_sha256':sha(log),'status':'PASS' if args.command_exit==args.tee_exit==0 else 'FAIL'})

if __name__=='__main__':main()
