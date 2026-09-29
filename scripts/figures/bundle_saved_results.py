#!/usr/bin/env python3
"""Adapt current-run saved display tables/results to the current nine-figure API.

Reads existing CSV/TSV only; never imports a scientific producer, fits a model,
recomputes detection or reads RAW. Missing results are explicit errors.
"""
from pathlib import Path
from datetime import date, timedelta
import argparse, csv, hashlib, json, math, shutil, statistics
from prepare_inputs import rows, tsv, write, KEYS, check

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared_dir',type=Path,required=True,help='Current-run shared GMT table export')
    p.add_argument('--sidecar_dir',type=Path,required=True,help='Current-run saved presentation sidecars')
    p.add_argument('--results_dir',type=Path,required=True,help='Current-run Stage 1/M021/M023 results')
    p.add_argument('--output_dir',type=Path,required=True)
    args=p.parse_args(); prepared=args.prepared_dir.resolve();source=args.sidecar_dir.resolve();results=args.results_dir.resolve();out=args.output_dir.resolve()
    if out.exists(): raise FileExistsError(f'OUTPUT_EXISTS:{out}')
    for root in [prepared,source,results]:
        if out==root or root.is_relative_to(out): raise ValueError('Output/input overlap')
    paths=set()
    def read(path):
        if not path.is_file(): raise FileNotFoundError(f'MISSING_SAVED_RESULT:{path}; no upstream computation is started')
        paths.add(path);return rows(path)
    def cp(path,dest):
        if not path.is_file(): raise FileNotFoundError(f'MISSING_SAVED_TABLE:{path}')
        paths.add(path);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    # Identify the already-selected example, never select a new scientific case here.
    cells=read(source/'F3_F4/F4A_REPRESENTATIVE_COMPONENT_CELLS.csv')
    examples={(r['pan_event_id'],r['event_date']) for r in cells}
    if len(examples)!=1: raise ValueError('Representative-cell source must identify exactly one saved example')
    eid,day=next(iter(examples));anchor=date.fromisoformat(day)
    event_matrix=results/'stage3_component_sensitivity/stage3_component_sensitivity_matrix.csv'
    event_values=results/'stage3_component_sensitivity/stage3_component_sensitivity_event_level.csv'
    # Resolve all extra result dependencies before writing a bundle.
    daily=read(results/'reproduce_sie/daily_sie_processed.csv')
    detected=read(results/'reproduce_sie/vrile_events_unique_both_jja_5p.csv')
    reverse=read(results/'m021_corrected_stage3/products/reverse_event_differences.csv')
    normdiff=read(results/'m023_cross_scale/REVERSE_NORMALIZED_EVENT_DIFFERENCES.csv')
    norm=read(source/'F4/F4D_NORMALIZED_SOURCE_DATA.csv')
    absolute=read(source/'F3_F4/F4S_ABSOLUTE_DOMAIN_CONTRAST_SOURCE_DATA.csv')
    read(event_matrix);read(event_values)
    out.mkdir()
    for key in KEYS: (out/f'Figure_{key}').mkdir()
    for name in ['f2_counts.tsv','f2_heatmap.tsv']+[f'f2_{k}_{g}.tsv' for k in ('map','scatter') for g in ('broad_only','severe_nonmajor','major_severe')]: cp(prepared/name,out/'Figure_2'/name)
    for name in ['f4_components.tsv','f4_largest_perimeter.gmt','f4b_summary.tsv']+[f'f4b_{k}_points.tsv' for k in ('neff','largest','top3','spread')]:cp(prepared/name,out/'Figure_3'/name)
    (out/'Figure_3/representative.json').write_text(json.dumps(dict(event_id=eid,event_date=day))+'\n')
    for key in ('3','4'):
        cp(event_matrix,out/f'Figure_{key}'/event_matrix.name);cp(event_values,out/f'Figure_{key}'/event_values.name)
    for name in [f'f3_region_{i}.tsv' for i in (1,2,5,6,7)]+['f3_region_labels.tsv']:cp(prepared/name,out/'Figure_6'/name)
    formal=read(source/'F3_F4/F3_revision_round3_formal_evidence_matrix.csv')
    # Select the four named diagnostics in the approved figure, not the first
    # four rows: meridional wind is row FIVE in the original eight-row matrix.
    order={'sic_state':3.5,'sic_change_5d':2.5,'wind_speed_10m':1.5,'meridional_wind_v10':.5}
    status={'NOT_SUPPORTED':(0,'N'),'SUPPORTED_PRIMARY_ONLY':(1,'P'),'SUPPORTED_WITH_ROBUSTNESS':(2,'R')}
    selected=[(int(r['region_order'])+.5,order[r['mechanism']],*status[r['manuscript_status']])
              for r in sorted(formal,key=lambda r:(int(r['mechanism_order']),int(r['region_order']))) if r['mechanism'] in order]
    if len(selected)!=20: raise ValueError('Four-diagnostic display selection is incomplete')
    write(out/'Figure_6/f3_matrix_selected.tsv',selected)
    for idx in (1,2):
        path=prepared/f'f3_c{idx}_primary.tsv';paths.add(path)
        vals=tsv(path)[:3]
        if [r[4:] for r in vals]!=[['100','km'],['300','km'],['500','km']]: raise ValueError('Primary wind buffer order changed')
        write(out/f'Figure_6/plot_c{idx}_primary_buffers.tsv',[(r[0],3-i,*r[2:]) for i,r in enumerate(vals)])
    for name in ['s1_summary.tsv']+[f's1_{k}_points.tsv' for k in ('neff','largest','top3')]:cp(prepared/name,out/'Figure_S3'/name)
    # The prescribed absolute/normalized estimands and their existing CIs are copied,
    # not estimated again. Only absolute loss is converted to 10^3 km2eq.
    ym={'focal':3,'residual':2,'other_region':1};am={'local_sic_loss_km2eq':'focal','residual_sic_loss_km2eq':'residual','other_region_sic_loss_km2eq':'other_region'}
    write(out/'Figure_5/plot_absolute.tsv',[(float(r['mean_difference'])/1000,ym[am[r['metric']]],float(r['ci_low'])/1000,float(r['ci_high'])/1000) for r in absolute])
    write(out/'Figure_5/plot_normalized.tsv',[(r['estimate'],ym[r['domain']],r['ci_low'],r['ci_high']) for r in norm if r['series']=='raw_normalized'])
    cp(prepared/'f5a_matrix.tsv',out/'Figure_5/f5a_matrix.tsv')
    aa=[(r['unique_local_event_id'],float(r['difference'])/1000) for r in reverse if r['metric']=='local_sic_loss_km2eq' and int(r['control_count'])==20 and r['difference'].strip() and math.isfinite(float(r['difference']))]
    bb=[(r['case_unique_local_event_id'],float(r['local_loss_fraction_diff'])) for r in normdiff if r['local_loss_fraction_diff'].strip() and math.isfinite(float(r['local_loss_fraction_diff']))]
    if len(dict(aa))!=len(aa) or len(dict(bb))!=len(bb) or set(dict(aa))!=set(dict(bb)):raise ValueError('Absolute/normalized focal case ID universes differ')
    for name,data in [('absolute',aa),('normalized',bb)]:
        data.sort(key=lambda v:(v[1],v[0]));values=[v for _,v in data]
        write(out/f'Figure_5/{name}_ordered.tsv',[(i+1,v,eid) for i,(eid,v) in enumerate(data)])
        for k,v in [('mean',statistics.mean(values)),('median',statistics.median(values))]:write(out/f'Figure_5/{name}_{k}.tsv',[(0,v),(280,v)])
    write(out/'Figure_5/rank_ticks.tsv',[(i,'a',i) for i in sorted({1,len(aa),*[i for i in (100,200) if i<len(aa)]})])
    write(out/'Figure_5/inset_ticks.tsv',[(i,'a',i) for i in sorted({1,math.ceil(len(aa)/20)*10,len(aa)})])
    adjusted=[r for r in norm if r['series']=='focal_strength_adjusted']
    counts={int(r['n_cases']) for r in adjusted}
    if len(adjusted)!=2 or len(counts)!=1:raise ValueError('Saved BASE focal-sensitivity source incomplete')
    write(out/'Figure_S4/plot_adjusted.tsv',[(r['estimate'],ym[r['domain']],r['ci_low'],r['ci_high']) for r in adjusted])
    write(out/'Figure_S4/adjusted_ticks.txt',[(2,'a','Residual'),(1,'a','Other-region')])
    (out/'Figure_S4/metadata.json').write_text(json.dumps(dict(n_cases=next(iter(counts))))+'\n')
    # Existing daily extent and detector membership only; do not recompute thresholds.
    if not any(r.get('date',r.get('event_date'))==day for r in detected): raise ValueError('Representative date absent from saved both-branch detector events')
    lookup={date.fromisoformat(r['date']):r for r in daily}
    if len(lookup)!=len(daily):raise ValueError('Duplicate daily dates')
    dates=[anchor+timedelta(days=i-15) for i in range(31)]
    if any(d not in lookup for d in dates):raise ValueError('Saved example window incomplete')
    annual={}
    for d,r in lookup.items(): annual.setdefault(d.year,[]).append(float(r['extent']))
    write(out/'Figure_S1/f1_annual.tsv',[(y,statistics.mean(v),int(y==anchor.year)) for y,v in sorted(annual.items())])
    years=sorted(annual)
    write(out/'Figure_S1/f1_year_ticks.txt',[(y,'af',y) for y in sorted({years[0],years[-1],*[y for y in years if y%10==0 and y!=1990]})])
    write(out/'Figure_S1/f1_window.tsv',[(i,lookup[d]['extent']) for i,d in enumerate(dates)])
    write(out/'Figure_S1/f1_window_highlight.tsv',[(i,lookup[dates[i]]['extent']) for i in (13,16)])
    write(out/'Figure_S1/f1_window_anchor.tsv',[(15,lookup[anchor]['extent'])])
    write(out/'Figure_S1/f1_date_ticks.txt',[(i,'af',dates[i].strftime('%-d %b')) for i in range(0,31,5)])
    start,end=dates[13],dates[16];delta=float(lookup[end]['extent'])-float(lookup[start]['extent'])
    if not math.isclose(delta,float(lookup[anchor]['delta_sie']),abs_tol=1e-12):raise ValueError('Saved delta/endpoint mismatch')
    meta=dict(event_id=eid,selected_year=anchor.year,anchor_label=anchor.strftime('%-d %b'),
        change_label=f"{start.day}-{end.day} {end.strftime('%b')}: 3-day change = {delta:.3f} x 10@+6@+ km@+2@+",
        detector_label='Both detector branches at or below q0.05')
    (out/'Figure_S1/metadata.json').write_text(json.dumps(meta)+'\n')
    for key in KEYS:check(out,key)
    (out/'INPUT_PROVENANCE.json').write_text(json.dumps([dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(paths)],indent=2)+'\n')
    print('CURRENT_NINE_FIGURE_INPUT_BUNDLE=PASS; SCIENTIFIC_COMPUTATION=NOT_RUN; RAW_READ=NO')

if __name__=='__main__':main()
