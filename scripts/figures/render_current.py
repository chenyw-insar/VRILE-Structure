#!/usr/bin/env python3
"""Render current Figure 2–6/S1–S4 from explicitly supplied processed inputs.

No Methods flowchart, RAW reads, scientific producers or inference calls.
--dry-run checks file contracts and prints a plan without creating directories,
importing the vector/render modules, running GMT or starting subprocesses.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys

from prepare_inputs import KEYS, REQUIRED, check

HERE=Path(__file__).resolve().parent

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figure',choices=(*KEYS,'all'),required=True,help='all means the nine scientific figures, never Figure 1')
    p.add_argument('--input_dir',type=Path,required=True,help='Processed plotting tables grouped as Figure_2/, …, Figure_S4/')
    p.add_argument('--output_dir',type=Path,required=True,help='New isolated output directory; never overwrite existing outputs')
    p.add_argument('--dry-run',action='store_true',help='Read-only file-contract plan; no rendering/science/subprocess/file writes')
    p.add_argument('--timeout',type=int,default=120,help='Maximum seconds per whole figure and its clean panels (default 120)')
    p.add_argument('--gmt_bin_dir',type=Path,help='Optional GMT/Ghostscript executable directory; no hard-coded environment path')
    p.add_argument('--_render_one',action='store_true',help=argparse.SUPPRESS)
    return p

def one(key, inputs, output, runtime):
    # Imports occur only after main's read-only dry-run return.
    import subprocess
    from prepare_inputs import prepare
    from assemble_current import assemble
    import export_panels
    import remove_caption_titles
    import refine_layout
    output.mkdir(parents=True,exist_ok=False)
    job=output/'work'/f'Figure_{key}';job.mkdir(parents=True)
    gmt=job/'full_rendered';gmt.mkdir()
    env_values,annotations=prepare(inputs,key,job/'inputs')
    env=os.environ.copy();env['VRILE_PLOT_PYTHON']=sys.executable
    env['GMT_AUTO_DOWNLOAD']='off'
    command=['bash',str(HERE/'gmt'/f'figure_{key}.sh'),str(job/'inputs'),str(gmt)]
    with (output/'gmt.log').open('w') as log:
        log.write('COMMAND='+json.dumps(command)+'\n');log.flush()
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if key in ('3','4'):
        assemble(key,gmt,output)
    else:
        export_panels.B=job; remove_caption_titles.B=job
        refine_layout.B=job; refine_layout.OUT=output
        refine_layout.MOVE={(key,k):v for k,v in annotations.items()}
        if key=='S1': remove_caption_titles.TITLES[key]=['Northern Hemisphere annual context',env_values['EVENT_ID']+' detection example']
        if key=='S4': refine_layout.S4_NOTE=f"n = {env_values['N_CASES']}"
        panels=[]
        if key=='S3':
            rec=[dict(panel=k,source=gmt/f'Figure_S3{k}.pdf') for k in 'abc']
        else:
            export_panels.export(key)
            raw=json.loads((job/'panels'/f'Figure_{key}'/'panel_layout.json').read_text())['panels']
            rec=[dict(panel=r['panel'],source=job/(r['file']+'.pdf')) for r in raw]
        # Assembly's relative paths are rooted at output, not the temporary GMT directory.
        refine_layout.B=output
        for r in rec:
            name=f'Figure_{key}'+('' if r['panel']=='whole' else r['panel'])+'_clean'
            panels.append(refine_layout.panel(key,r['panel'],r['source'],output/'panels'/f'Figure_{key}'/name,True))
        refine_layout.assemble(key,panels,True)
    manifest=[]
    for p in sorted(output.rglob('*')):
        if p.is_file(): manifest.append(dict(path=str(p.relative_to(output)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size))
    (output/'render_manifest.json').write_text(json.dumps(dict(figure=key,inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in check(inputs,key)],
        files=manifest,png_dpi=600,methods_flowchart=False,scientific_run=False,raw_read=False,
        plotting_runtime=runtime),indent=2)+'\n')

def main():
    args=parser().parse_args(); keys=KEYS if args.figure=='all' else (args.figure,)
    inputs=args.input_dir.expanduser().resolve(); output=args.output_dir.expanduser().resolve()
    if args.timeout<=0: raise ValueError('timeout must be positive')
    if output==inputs or output.is_relative_to(inputs): raise ValueError('Output must not overwrite/enter the input tree')
    if output.is_relative_to(HERE.parents[1]): raise ValueError('Use an isolated output directory outside the source package')
    plan=[]
    for key in keys:
        files=check(inputs,key)
        plan.append(dict(figure=key,renderer=f'gmt/figure_{key}.sh',required_inputs=[str(p) for p in files],outputs=['PDF vector','SVG vector','PNG 600 dpi','clean panels'],timeout_seconds=args.timeout))
    if output.exists(): raise FileExistsError(f'OUTPUT_EXISTS:{output}; choose a new output directory')
    if args.dry_run:
        print(json.dumps(dict(mode='DRY_RUN_ONLY',figures=plan,scientific_execution=False,subprocesses_started=0,files_written=0),indent=2));return
    from plot_runtime import require_current_runtime
    runtime=require_current_runtime()
    if args.gmt_bin_dir: os.environ['PATH']=str(args.gmt_bin_dir.resolve())+os.pathsep+os.environ.get('PATH','')
    if not shutil.which('gmt') or not shutil.which('gs'): raise RuntimeError('GMT and Ghostscript must be available on PATH (or --gmt_bin_dir)')
    if args._render_one:
        if len(keys)!=1: raise ValueError('Internal worker accepts exactly one scientific figure')
        one(keys[0],inputs,output,runtime);return
    import subprocess
    output.mkdir(parents=True)
    for key in keys:
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--_render_one','--figure',key,'--input_dir',str(inputs),'--output_dir',str(output/f'Figure_{key}'),'--timeout',str(args.timeout)]
        print('PLOTTING_ONLY_COMMAND='+json.dumps(command),flush=True)
        # A process group ensures a timed-out GMT child is not left rendering in the background.
        with (output/f'Figure_{key}.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try: status=child.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                from process_control import terminate_process_group
                terminate_process_group(child)
                raise RuntimeError(f'PLOTTING_TIMEOUT:{key}; partial isolated outputs retained; no retry')
        if status: raise RuntimeError(f'PLOTTING_FAILED:{key}:exit={status}; see {output}/Figure_{key}.log')
    print('SELECTED_FIGURE_RENDERING=PASS\nFIGURE_RENDERER_RUNS_UPSTREAM_SCIENCE=NO')

if __name__=='__main__':
    try: main()
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError) as exc:
        print(str(exc),file=sys.stderr);raise SystemExit(2)
