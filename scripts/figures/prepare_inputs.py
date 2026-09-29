"""Display-only preparation of explicit, saved, processed figure inputs.

No science producer imports, RAW access, model fitting or missing-input recovery.
ECDF, descriptive quartiles, packing, pairing and units follow the adopted
September 25 display implementation. Input cardinalities and labels are data driven.
"""
from pathlib import Path
from decimal import Decimal
import csv
import json
import math
import shlex
import shutil
import statistics

KEYS = ('2', '3', '4', '5', '6', 'S1', 'S2', 'S3', 'S4')
REQUIRED = {
    '2': [f'f2_{kind}_{group}.tsv' for kind in ('map','scatter') for group in ('broad_only','severe_nonmajor','major_severe')] + ['f2_counts.tsv','f2_heatmap.tsv'],
    '3': ['f4_components.tsv','f4_largest_perimeter.gmt','representative.json','f4b_summary.tsv','stage3_component_sensitivity_matrix.csv','stage3_component_sensitivity_event_level.csv'] + [f'f4b_{k}_points.tsv' for k in ('neff','largest','top3','spread')],
    '4': ['stage3_component_sensitivity_event_level.csv','stage3_component_sensitivity_matrix.csv'],
    '5': ['f5a_matrix.tsv','plot_absolute.tsv','plot_normalized.tsv','rank_ticks.tsv','inset_ticks.tsv'] + [f'{k}_{s}.tsv' for k in ('absolute','normalized') for s in ('ordered','mean','median')],
    '6': [f'f3_region_{i}.tsv' for i in (1,2,5,6,7)] + ['f3_region_labels.tsv','f3_matrix_selected.tsv','plot_c1_primary_buffers.tsv','plot_c2_primary_buffers.tsv'],
    'S1': ['metadata.json','f1_annual.tsv','f1_year_ticks.txt','f1_date_ticks.txt','f1_window.tsv','f1_window_highlight.tsv','f1_window_anchor.tsv'],
    'S2': [],
    'S3': ['s1_summary.tsv']+[f's1_{k}_points.tsv' for k in ('neff','largest','top3')],
    'S4': ['plot_adjusted.tsv','adjusted_ticks.txt','metadata.json'],
}

def rows(path):
    with path.open(newline='') as f: return list(csv.DictReader(f))

def tsv(path):
    return [line.split() for line in path.read_text().splitlines() if line.strip() and not line.startswith('>')]

def write(path, values):
    path.write_text(''.join(' '.join(map(str,r))+'\n' for r in values))

def quantile(a, p):
    a=sorted(a); z=(len(a)-1)*p; i=int(z)
    return a[i] if i==len(a)-1 else a[i]+(a[i+1]-a[i])*(z-i)

def finite(values):
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('Missing/nonfinite display values')

def check(input_dir, key):
    root=input_dir.resolve()
    if any(x.lower()=='raw' for x in root.parts): raise ValueError('RAW is not a processed plotting input directory')
    paths=[]
    for name in REQUIRED[key]:
        p=root/f'Figure_{key}'/name
        if not p.is_file(): raise FileNotFoundError(f'MISSING_PROCESSED_FIGURE_INPUT:{p}; no science will be started')
        if not p.resolve().is_relative_to(root): raise ValueError(f'Input escapes processed-input directory: {p}')
        if p.stat().st_size>16*1024*1024: raise ValueError(f'Not a small plotting input: {p}')
        paths.append(p)
    return paths

def figure3(d):
    summary={r[0]:list(map(float,r[1:])) for r in tsv(d/'f4b_summary.tsv')}
    primary=[r for r in rows(d/'stage3_component_sensitivity_matrix.csv') if (r['sic_floor'],r['strict_min_cells'])==('none','4')]
    if len(primary)!=1: raise ValueError('Missing/duplicate primary None/>4 matrix row')
    stats={}; sizes=set()
    for name in ('neff','largest','top3','spread'):
        values=[float(r[0]) for r in tsv(d/f'f4b_{name}_points.tsv')]; finite(values)
        if values!=sorted(values): raise ValueError('Population values must retain approved stable ordering')
        n=len(values); sizes.add(n); median=summary[name][0]
        if not math.isclose(median,statistics.median(values),rel_tol=1e-12): raise ValueError(f'Summary/values mismatch: {name}')
        q1,q3=quantile(values,.25),quantile(values,.75)
        if name=='neff':
            a,b=float(primary[0]['Neff_q25']),float(primary[0]['Neff_q75'])
            if not (math.isclose(q1,a,rel_tol=1e-12) and math.isclose(q3,b,rel_tol=1e-12)): raise ValueError('Primary quartile mismatch')
            q1,q3=a,b
        stats[name]=dict(n=n,median=median,q1=q1,q3=q3)
        if name in ('largest','top3'):
            if not 0<=min(values)<=max(values)<=100: raise ValueError('Share outside percent domain')
            coords=[(0,0)]
            for i,v in enumerate(values): coords.extend([(v,i/n*100),(v,(i+1)/n*100)])
            coords.append((100,100)); write(d/f'{name}_ecdf.tsv',coords)
        else:
            xmin,xmax=(0,20) if name=='neff' else (900,2800)
            if min(values)<xmin or max(values)>xmax: raise ValueError(f'APPROVED_DISPLAY_RANGE_EXCEEDED:{name}; no clipping or science recomputation')
            scale=(6.65*72/2.54)/(xmax-xmin); placed=[]; diameter=2.6
            for i in sorted(range(n),key=lambda j:(abs(values[j]-median),j)):
                for lane in [0]+[s*k for k in range(1,40) for s in (1,-1)]:
                    yp=lane*diameter*.88
                    if all((scale*(values[i]-a))**2+(yp-b)**2>=diameter**2-.01 for a,b,_ in placed):
                        placed.append((values[i],yp,i)); break
            if len(placed)!=n or max(abs(v[1]) for v in placed)>=29: raise ValueError('Approved dot-packing envelope exceeded')
            yscale=(2.15*72/2.54)/1.65
            write(d/f'{name}_dots.tsv',[(a,1.05+b/yscale) for a,b,_ in sorted(placed,key=lambda t:t[2])])
            write(d/f'{name}_iqr.tsv',[(q1,.20),(q3,.20)])
            write(d/f'{name}_caps.tsv',[(q1,.11),(q1,.29),('>',),(q3,.11),(q3,.29)])
            write(d/f'{name}_median.tsv',[(median,.08),(median,.32)])
    if len(sizes)!=1: raise ValueError('Mismatched population universes')
    cells=tsv(d/'f4_components.tsv'); components={int(r[2]) for r in cells}
    meta=json.loads((d/'representative.json').read_text())
    # Read the saved, area-integrated result. Unweighted display-cell SIC sums
    # are NOT the scientific denominator and must never substitute for it.
    selected=[r for r in rows(d/'stage3_component_sensitivity_event_level.csv') if
              (r['pan_event_id'],r['event_date'],r['sic_floor'],r['strict_min_cells'])==
              (meta['event_id'],meta['event_date'],'none','4')]
    if len(selected)!=1 or int(selected[0]['n_components'])!=len(components): raise ValueError('Representative saved-result identity/components mismatch')
    saved_share=float(selected[0]['largest_component_share'])
    if not 0<saved_share<=1: raise ValueError('Invalid saved representative resolved share')
    share=round(100*saved_share,1)
    from datetime import date
    event_date=date.fromisoformat(meta['event_date'])
    # This is an explicitly selected saved example, not an inferred population result.
    label=f"{meta['event_id']} - {event_date.day} {event_date.strftime('%B %Y')}"
    env=dict(N_EVENTS=n,N_COMPONENTS=len(components),COMPONENT_CPT_END=max(components)+1,
             REP_SHARE=f'{share:.1f}',REP_OTHER=f'{100-share:.1f}',REP_SHARE_X=share/2,
             REP_OTHER_X=(100+share)/2,REP_LABEL=label,
             MED_LARGEST=stats['largest']['median'],MED_TOP3=stats['top3']['median'],
             MED_LARGEST_LABEL=f"{stats['largest']['median']:.1f}",MED_TOP3_LABEL=f"{stats['top3']['median']:.1f}",
             MED_NEFF_LABEL=f"{stats['neff']['median']:.2f}",MED_SPREAD_LABEL=f"{stats['spread']['median']:.0f}")
    (d/'display_summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    return env

def figure4(d):
    events=rows(d/'stage3_component_sensitivity_event_level.csv'); matrix=rows(d/'stage3_component_sensitivity_matrix.csv')
    configs=[(f,k) for f in ('none','0.15','0.3') for k in ('4','20','50')]
    by={(r['pan_event_id'],r['sic_floor'],r['strict_min_cells']):r for r in events}
    if len(by)!=len(events): raise ValueError('Duplicate event/configuration key')
    ordered=sorted({(r['event_date'],r['pan_event_id']) for r in events}); n=len(ordered)
    if n==0 or len({e for _,e in ordered})!=n or len(events)!=9*n: raise ValueError('Incomplete event/configuration universe')
    if set(by)!={(e,f,k) for _,e in ordered for f,k in configs}: raise ValueError('Missing paired None reference or setting')
    records=[]
    for name,field,factor in [('neff','Neff',Decimal(1)),('largest','largest_component_share',Decimal(100))]:
        none=[]; delta=[]
        for j,(floor,size) in enumerate(configs,1):
            for idx,(date,eid) in enumerate(ordered):
                r=by[eid,floor,size]; ref=by[eid,'none',size]
                if r['event_date']!=date or ref['event_date']!=date: raise ValueError('Paired event date differs')
                value=Decimal(r[field])*factor; base=Decimal(ref[field])*factor
                if not value.is_finite() or not base.is_finite(): raise ValueError('Nonfinite paired value')
                if floor=='none': none.append((j,n-idx,str(value)))
                else:
                    diff=value-base; delta.append((j,n-idx,str(diff)))
                    records.append(dict(metric=name,pan_event_id=eid,event_date=date,row_top_to_bottom=idx+1,sic_floor=floor,
                        strict_min_cells=size,source_value=str(value),reference_None_same_size=str(base),difference=str(diff),
                        unit='percentage points' if name=='largest' else 'effective components'))
        write(d/f'{name}_none.tsv',none); write(d/f'{name}_delta.tsv',delta)
        if any(not 0<=float(r[2])<=(20 if name=='neff' else 100) for r in none): raise ValueError('Approved absolute CPT envelope exceeded')
        step=5 if name=='largest' else 1
        bound=max(step,math.ceil(max(abs(float(r[2])) for r in delta)/step)*step)
        write(d/f'{name}_delta_limit.txt',[(bound,)])
    ticks=sorted({1,n,*[math.ceil(n*p) for p in (.25,.5,.75)]}-{0})
    write(d/'event_ticks.txt',[(i,'a',n+1-i) for i in ticks])
    write(d/'cell_ticks.txt',[(j+1,'a','>'+c[1]) for j,c in enumerate(configs)])
    m={(r['sic_floor'],r['strict_min_cells']):r for r in matrix}
    if len(m)!=9 or len(matrix)!=9 or set(m)!=set(configs): raise ValueError('Matrix settings differ')
    for j,config in enumerate(configs):
        r=m[config]; prefix='resolved_fraction_vs_primary_total_loss_'
        vals=[100*float(r[prefix+k]) for k in ('median','q25','q75')]
        if not 55<vals[1]<=vals[0]<=vals[2]<90: raise ValueError('Approved coverage display envelope exceeded')
        if int(r['n_events_total'])!=n: raise ValueError('Matrix/event universe mismatch')
        write(d/f'coverage_{j}.tsv',[(j+1,*vals)])
    with (d/'paired_change_source.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader();w.writerows(records)
    with (d/'event_row_map.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['row_top_to_bottom','pan_event_id','event_date']);w.writerows((i+1,e,date) for i,(date,e) in enumerate(ordered))
    # Round upward to six decimals as in the adopted 0.063637-cm 99-event grid;
    # this prevents hairline gaps without changing any heatmap data value.
    return dict(EVENT_TOP=n+.5,CELL_HEIGHT=f'{math.ceil(6.3/n*1e6)/1e6:.6f}',N_EVENTS=n)

def prepare(input_dir, key, dest):
    paths=check(input_dir,key); dest.mkdir(parents=True,exist_ok=False)
    for p in paths: shutil.copyfile(p,dest/p.name)
    env={}; annotations={}
    if key=='3': env=figure3(dest)
    elif key=='4': env=figure4(dest)
    elif key=='5':
        cells=tsv(dest/'f5a_matrix.tsv'); by={(float(r[0]),float(r[1])):int(r[2]) for r in cells}
        if set(by)!={(.5,1.5),(1.5,1.5),(.5,.5),(1.5,.5)}: raise ValueError('Expected four distinct X1/X2 cells')
        absolute=[float(r[1]) for r in tsv(dest/'absolute_ordered.tsv')]; norm=[float(r[1]) for r in tsv(dest/'normalized_ordered.tsv')]
        finite(absolute);finite(norm)
        if len(absolute)!=len(norm): raise ValueError('Different matched case universe')
        if len(absolute)>280: raise ValueError('Approved ranked-case display envelope exceeded')
        env=dict(N_MAJOR=sum(by.values()),N_CASES=len(absolute),N_X1=by[.5,1.5]+by[1.5,1.5],N_X2=by[.5,1.5]+by[.5,.5],
                 ABS_MEAN=f'{statistics.mean(absolute):.2f}',ABS_MEDIAN=f'{statistics.median(absolute):.2f}',
                 NORM_MEAN=f'{statistics.mean(norm):.5f}',NORM_MEDIAN=f'{statistics.median(norm):.5f}')
        annotations={'a':f"Major local events; n = {env['N_MAJOR']}",'b':f"X2-linked cases; n = {env['N_CASES']}",
                     'c':f"Mean: {env['ABS_MEAN']}  |  Median: {env['ABS_MEDIAN']}",
                     'd':f"Mean: {env['NORM_MEAN']}  |  Median: {env['NORM_MEDIAN']}"}
    elif key=='S1':
        meta=json.loads((dest/'metadata.json').read_text()); annual=tsv(dest/'f1_annual.tsv')
        selected=[r for r in annual if int(r[2])==1]
        if len(selected)!=1 or int(selected[0][0])!=int(meta['selected_year']): raise ValueError('Selected example/year mismatch')
        env=dict(SELECTED_YEAR=int(meta['selected_year']),EVENT_ID=meta['event_id'],ANCHOR_LABEL=meta['anchor_label'],
                 CHANGE_LABEL=meta['change_label'],DETECTOR_LABEL=meta['detector_label'])
        point=tsv(dest/'f1_window_anchor.tsv')
        if len(point)!=1: raise ValueError('Missing/ambiguous saved detector anchor')
        y=float(point[0][1]);env.update(ANCHOR_LEADER_Y=y+.10,ANCHOR_END_Y=y+.58,ANCHOR_TEXT_Y=y+.75)
    elif key=='S2':
        source=Path(__file__).resolve().parents[2]/'figure_specs/correspondence_schematic'
        for p in source.iterdir(): shutil.copyfile(p,dest/p.name)
    elif key=='S4':
        n=json.loads((dest/'metadata.json').read_text())['n_cases']
        if not isinstance(n,int) or n<=0: raise ValueError('Missing saved focal-strength sample size')
        for r in tsv(dest/'plot_adjusted.tsv'):
            if not 0<=float(r[2])<=float(r[0])<=float(r[3])<=.012: raise ValueError('Approved S4 interval display envelope exceeded')
        env=dict(N_CASES=n)
    for name,value in env.items():
        if '\n' in str(value) or '\r' in str(value): raise ValueError('Multiline display annotation rejected')
    (dest/'display.env').write_text(''.join(f'{k}={shlex.quote(str(v))}\n' for k,v in env.items()))
    return env,annotations
