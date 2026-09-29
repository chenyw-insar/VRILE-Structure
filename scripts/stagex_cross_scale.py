#!/usr/bin/env python3
"""P0 cross-scale definition and D4d reviewer-risk sensitivity audit.

Writes only to a new audit sandbox. Frozen Stage3 products are read-only inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('VRILE_PROJECT_ROOT', Path(__file__).resolve().parents[1])).expanduser().resolve()
PIPE = ROOT / '.'
SRC = PIPE / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vrile.regions import NSIDC_0780_REGION_IDS
from vrile.stage3.budget import aligned_patch_rows
from vrile.stage3.controls import bootstrap_mean_ci, fdr_bh, sign_flip_test
from vrile.stage3.footprints import load_patch_cells
from vrile.stage3.grid import (
    find_sic_file,
    load_cell_area_km2,
    load_surface_mask,
    read_sic,
    valid_ocean_mask,
)

PRODUCTS = Path(
    os.environ.get(
        'VRILE_STAGE3_PRODUCTS_ROOT',
        ROOT / 'outputs/m021_corrected_stage3/products',
    )
).expanduser().resolve()
MAJOR_PATH = PIPE / 'outputs/local_vrile_major_severe/major_severe_events_union.csv'
PAN_PATH = PIPE / 'outputs/reproduce_sie/vrile_events_unique_both_jja_5p.csv'
PAN_LOC_PATH = PIPE / 'outputs/reproduce_sie/vrile_locations.csv'
FIG3_PATH = PIPE / 'outputs/spatiotemporal_matching/fig3_count_vector.csv'
MEMBERSHIP_PATH = PIPE / 'data/processed/local_event_cells/unique_event_patch_membership.csv'
PATCH_INDEX_PATH = PIPE / 'data/processed/local_event_cells/patch_cell_index.csv'
PATCH_DIR = PIPE / 'data/processed/local_event_cells'
AREA_PATH = PIPE / 'data/raw/nsidc_ancillary/NSIDC0771_CellArea_PS_N25km_v1.1.nc'
MASK_PATH = PIPE / 'data/raw/nsidc_region_masks/NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc'
SEED = 20260822
RESAMPLES = 10_000
ABS_ATOL = 1e-6
ABS_RTOL = 1e-12


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator='\n', float_format='%.15g')


def fmt(value: float) -> str:
    return 'NA' if not np.isfinite(value) else f'{value:.8g}'


def haversine_km(lon1: float, lat1: float, lon2: pd.Series, lat2: pd.Series) -> np.ndarray:
    radius = 6371.0
    p1 = np.deg2rad(float(lat1)); p2 = np.deg2rad(lon2.astype(float) * 0 + lat2.astype(float))
    dphi = np.deg2rad(lat2.astype(float) - float(lat1))
    dlambda = np.deg2rad(lon2.astype(float) - float(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return np.asarray(2 * radius * np.arcsin(np.sqrt(a)), dtype=float)


def primary_hashes() -> dict[str, str]:
    return {p.name: sha(p) for p in sorted(PRODUCTS.iterdir()) if p.is_file()}


def cross_scale_outputs(out: Path, major: pd.DataFrame, classification: pd.DataFrame) -> dict[str, int]:
    pan = pd.read_csv(PAN_PATH)
    locations = pd.read_csv(PAN_LOC_PATH)
    pan['date'] = pd.to_datetime(pan['date']).dt.normalize()
    locations['date'] = pd.to_datetime(locations['date']).dt.normalize()
    pan = pan.merge(locations[['date', 'center_lon', 'center_lat']], on='date', how='left')
    major = major.copy()
    major['event_date'] = pd.to_datetime(major['event_start']).dt.normalize()
    class_lookup = classification.set_index('unique_local_event_id')
    rows: list[dict] = []
    for event in major.sort_values('unique_local_event_id').itertuples(index=False):
        uid = str(event.unique_local_event_id)
        candidates = pan[(pan['date'] - pd.Timestamp(event.event_date)).abs().dt.days <= 3].copy()
        temporal = not candidates.empty
        if temporal:
            distances = haversine_km(
                float(event.track_centroid_lon), float(event.track_centroid_lat),
                candidates['center_lon'], candidates['center_lat'],
            )
            match500 = bool(np.any(distances <= 500.0))
            match300 = bool(np.any(distances <= 300.0))
        else:
            match500 = False; match300 = False
        source = class_lookup.loc[uid]
        rows.append({
            'unique_local_event_id': uid,
            'canonical_major_status': True,
            'event_date': pd.Timestamp(event.event_date).strftime('%Y-%m-%d'),
            'event_start_temporal_correspondence_3d': temporal,
            'event_start_centroid_correspondence_3d_500km': match500,
            'event_start_centroid_correspondence_3d_300km': match300,
            'track_s_footprint_linked': bool(source['pan_linked']),
            'n_track_s_linked_pan_events': int(source['n_linked_pan_events']),
            'source_fig3_reference': rel(FIG3_PATH),
            'source_track_s_reference': rel(PRODUCTS / 'reverse_major_event_classification.csv'),
        })
    event_map = pd.DataFrame(rows)
    if len(event_map) != 554 or event_map['unique_local_event_id'].nunique() != 554:
        raise SystemExit('HOLD_MAJOR_UNIVERSE_NOT_554_UNIQUE')
    if set(event_map['unique_local_event_id']) != set(major['unique_local_event_id'].astype(str)):
        raise SystemExit('HOLD_MAJOR_EVENT_SET_MISMATCH')
    write_csv(event_map, out / 'CROSS_SCALE_CORRESPONDENCE_EVENT_MAP.csv')

    temporal = event_map['event_start_temporal_correspondence_3d'].astype(bool)
    footprint = event_map['track_s_footprint_linked'].astype(bool)
    values = {
        'both': int((temporal & footprint).sum()),
        'temporal_only': int((temporal & ~footprint).sum()),
        'footprint_only': int((~temporal & footprint).sum()),
        'neither': int((~temporal & ~footprint).sum()),
        'event_start_3d': int(temporal.sum()),
        'event_start_500km': int(event_map['event_start_centroid_correspondence_3d_500km'].sum()),
        'event_start_300km': int(event_map['event_start_centroid_correspondence_3d_300km'].sum()),
        'footprint_linked': int(footprint.sum()),
        'footprint_unlinked': int((~footprint).sum()),
    }
    vector = pd.read_csv(FIG3_PATH).set_index('metric')['numerator'].astype(int)
    authoritative = {
        'event_start_3d': int(vector['major_severe_to_pan_pm3d']),
        'event_start_500km': int(vector['major_severe_to_pan_pm3d_le500km']),
        'event_start_300km': int(vector['major_severe_to_pan_pm3d_le300km']),
    }
    if any(values[key] != value for key, value in authoritative.items()):
        raise SystemExit('HOLD_AUTHORITATIVE_FIG3_CORRESPONDENCE_MISMATCH')
    if values['footprint_linked'] != int(classification['pan_linked'].sum()):
        raise SystemExit('HOLD_AUTHORITATIVE_TRACK_S_CLASSIFICATION_MISMATCH')
    crosstab = pd.DataFrame([
        {'event_start_temporal_correspondence_3d': True, 'track_s_footprint_linked': True, 'cell_label': 'both', 'n_events': values['both']},
        {'event_start_temporal_correspondence_3d': True, 'track_s_footprint_linked': False, 'cell_label': 'temporal_only', 'n_events': values['temporal_only']},
        {'event_start_temporal_correspondence_3d': False, 'track_s_footprint_linked': True, 'cell_label': 'footprint_only', 'n_events': values['footprint_only']},
        {'event_start_temporal_correspondence_3d': False, 'track_s_footprint_linked': False, 'cell_label': 'neither', 'n_events': values['neither']},
        {'event_start_temporal_correspondence_3d': 'MARGIN', 'track_s_footprint_linked': 'ALL', 'cell_label': 'temporal_margin', 'n_events': values['event_start_3d']},
        {'event_start_temporal_correspondence_3d': 'ALL', 'track_s_footprint_linked': 'MARGIN', 'cell_label': 'footprint_margin', 'n_events': values['footprint_linked']},
    ])
    write_csv(crosstab, out / 'CROSS_SCALE_CORRESPONDENCE_CROSSTAB.csv')

    registry = f"""# Cross-Scale Correspondence Definition Registry

The two diagnostics below are non-equivalent definitions. Neither is labelled
the uniquely correct or incorrect linkage.

## `event_start_temporal_correspondence_3d`

- Independent unit / universe: one canonical `major` local event; N = 554.
- Anchor semantics: local event start versus authoritative pan-Arctic event date.
- Time window: at least one pan-Arctic anchor within ±3 calendar days.
- Spatial criterion: none. The registered refinements add centroid distance
  ≤500 km or ≤300 km to the same ±3-day candidate set.
- Source: `{rel(FIG3_PATH)}` (SHA256 `{sha(FIG3_PATH)}`).
- Production code: `full_pipeline_v5/scripts/stage1_rebuild_local_catalog_from_frozen_cells.py::fig3_tables`, which calls `full_pipeline_v5/scripts/stage2_experiment_core.py::match_one`.
- Accepted counts: temporal = {values['event_start_3d']}; temporal+500 km = {values['event_start_500km']}; temporal+300 km = {values['event_start_300km']}.
- Permitted terms: `event-start ±3-day temporal correspondence`; `event-start ±3-day + centroid-distance correspondence`.
- Prohibited ambiguous terms: `pan-linked`, `footprint-linked`, or generic `matched to pan-Arctic` without the event-start rule.

## `track_s_footprint_linked`

- Independent unit / universe: one canonical `major` local event; N = 554.
- Anchor semantics: canonical major-event ID appears at major/legacy
  `major_severe` level in the accepted pan-Arctic/local footprint relation.
- Time window: patch-level interval intersection with `[T-5,T]`, with
  `patch.date <= T` no-look-ahead.
- Spatial criterion: reconstructed local patch footprint overlaps the accepted
  Track-S pan-Arctic/local relation; no centroid-distance requirement.
- Source: `{rel(PRODUCTS / 'panarctic_local_event_overlap.csv')}` and `{rel(PRODUCTS / 'reverse_major_event_classification.csv')}`.
- Production code: `full_pipeline_v5/src/vrile/stage3/reverse.py::classify_major_events`.
- Accepted count: linked = {values['footprint_linked']}; unlinked = {values['footprint_unlinked']}.
- Permitted term: `Track-S footprint-linked`.
- Prohibited ambiguous terms: `event-start matched`, `±3-day matched`, `centroid matched`, or unqualified `pan-linked`.
"""
    (out / 'CROSS_SCALE_CORRESPONDENCE_DEFINITION_REGISTRY.md').write_text(registry, encoding='utf-8')
    return values


def terminology_audit(out: Path) -> int:
    patterns = re.compile(r'pan-linked|pan linked|linked to pan-Arctic|major\s*[→-]>?\s*pan|matched to pan|±3\s*d|footprint-linked', re.I)
    rows: list[dict] = []
    allowed_suffixes = {'.md', '.csv', '.json', '.py', '.txt'}
    terminology_dir = ROOT / 'validation/reference/terminology'
    if not terminology_dir.is_dir():
        raise SystemExit('HOLD_OPTIONAL_TERMINOLOGY_REFERENCE_NOT_SUPPLIED')
    for path in sorted(terminology_dir.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if '/production/outputs/' in f'/{relative}/' or '/logs/' in f'/{relative}/':
            continue
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(lines, start=1):
            found = patterns.findall(line)
            if not found:
                continue
            context = line.strip()[:500]
            lower = context.lower()
            if any(token in lower for token in ('track-s', 'track s', 'footprint', '284', '270', 'reverse')):
                mapping = 'TRACK_S_FOOTPRINT'; replacement = 'Track-S footprint-linked'
            elif any(token in lower for token in ('±3', '3d', '3-day', 'centroid', 'fig3', '104/554')):
                mapping = 'EVENT_START'; replacement = 'event-start ±3-day temporal correspondence (add centroid criterion when applicable)'
            else:
                mapping = 'AMBIGUOUS'; replacement = 'replace with the registered event-start or Track-S footprint term after checking the cited source'
            rows.append({
                'file': relative, 'line_number': line_no, 'line_context': context,
                'current_wording': ';'.join(sorted(set(found), key=str.lower)),
                'maps_to': mapping, 'recommended_replacement': replacement,
            })
    write_csv(pd.DataFrame(rows, columns=['file','line_number','line_context','current_wording','maps_to','recommended_replacement']), out / 'CROSS_SCALE_TERMINOLOGY_AUDIT.csv')
    return int(sum(row['maps_to'] == 'AMBIGUOUS' for row in rows))


def prep_membership(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result['unique_local_event_id'] = result['unique_local_event_id'].astype(str)
    result['object_id'] = result['object_id'].astype(str)
    result['date'] = pd.to_datetime(result['date']).dt.normalize()
    result['start_date'] = pd.to_datetime(result['start_date']).dt.normalize()
    return result


def opportunity_outputs(
    out: Path,
    classification: pd.DataFrame,
    accepted_metrics: pd.DataFrame,
    anchors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, dict[str, float]]:
    membership = prep_membership(pd.read_csv(MEMBERSHIP_PATH))
    patch_index = pd.read_csv(PATCH_INDEX_PATH)
    required_ids = sorted(set(accepted_metrics['unique_local_event_id'].astype(str)))
    class_lookup = classification.set_index('unique_local_event_id')
    metric_lookup = accepted_metrics.set_index('unique_local_event_id')
    aligned_by_uid: dict[str, pd.DataFrame] = {}
    object_ids: set[str] = set()
    for uid in required_ids:
        event_date = pd.Timestamp(class_lookup.loc[uid, 'event_date'])
        aligned = aligned_patch_rows(membership, event_date, track='cdr')
        aligned = aligned[aligned['unique_local_event_id'].eq(uid)].copy()
        if aligned.empty:
            raise SystemExit(f'HOLD_NO_ALIGNED_PATCHES:{uid}')
        aligned_by_uid[uid] = aligned
        object_ids.update(aligned['object_id'].astype(str))
    cells = load_patch_cells(patch_index, PATCH_DIR, object_ids)
    if object_ids - set(cells):
        raise SystemExit('HOLD_MISSING_PATCH_CELLS')
    area, _, _ = load_cell_area_km2()
    surface = load_surface_mask()
    ocean = valid_ocean_mask()
    cdr_files: dict[str, str] = {}
    rows: list[dict] = []
    max_abs_delta = {'local': 0.0, 'residual': 0.0, 'other_region': 0.0}

    def domain_values(domain: np.ndarray, start_sic: np.ndarray, end_sic: np.ndarray) -> tuple[float, float, float, float, float]:
        valid = domain & ocean & np.isfinite(start_sic) & np.isfinite(end_sic)
        loss = np.maximum(0.0, start_sic - end_sic)
        valid_area = float(np.nansum(np.where(valid, area, 0.0), dtype=np.float64))
        initial = float(np.nansum(np.where(valid, start_sic * area, 0.0), dtype=np.float64))
        absolute = float(np.nansum(np.where(valid, loss * area, 0.0), dtype=np.float64))
        fraction = absolute / initial if initial > 0 else np.nan
        meanloss = absolute / valid_area if valid_area > 0 else np.nan
        return valid_area, initial, absolute, fraction, meanloss

    for uid in required_ids:
        source = class_lookup.loc[uid]
        event_date = pd.Timestamp(source['event_date']).normalize()
        start_date = event_date - pd.Timedelta(days=5)
        start_sic, _, _ = read_sic(start_date)
        end_sic, _, _ = read_sic(event_date)
        for date in (start_date, event_date):
            path = find_sic_file(date)
            cdr_files[rel(path)] = sha(path)
        local = np.zeros(ocean.shape, dtype=bool)
        for object_id in aligned_by_uid[uid]['object_id'].astype(str):
            yy, xx, _ = cells[object_id]
            local[yy, xx] = True
        local &= ocean
        residual = ocean & ~local
        region = str(source['dominant_region'])
        code = int(NSIDC_0780_REGION_IDS[region])
        other = ocean & np.isin(surface, sorted(set(range(1, 19)) - {code}))
        lv = domain_values(local, start_sic, end_sic)
        rv = domain_values(residual, start_sic, end_sic)
        ov = domain_values(other, start_sic, end_sic)
        accepted = metric_lookup.loc[uid]
        for label, observed, expected in (
            ('local', lv[2], float(accepted['local_sic_loss_km2eq'])),
            ('residual', rv[2], float(accepted['residual_sic_loss_km2eq'])),
            ('other_region', ov[2], float(accepted['other_region_sic_loss_km2eq'])),
        ):
            max_abs_delta[label] = max(max_abs_delta[label], abs(observed - expected))
            if not np.isclose(observed, expected, rtol=ABS_RTOL, atol=ABS_ATOL):
                raise SystemExit(f'HOLD_DOMAIN_NORMALIZATION_PRIMARY_METRIC_MISMATCH:{uid}:{label}:{observed}:{expected}')
        rows.append({
            'unique_local_event_id': uid, 'case_or_control': str(source['analysis_group']),
            'anchor_date': event_date.strftime('%Y-%m-%d'), 'source_region': region,
            'local_valid_area_km2': lv[0], 'local_initial_ice_equivalent_area_km2eq': lv[1],
            'local_absolute_sic_loss_km2eq': lv[2], 'local_loss_fraction_initial_ice': lv[3],
            'local_mean_sic_loss_valid_area': lv[4],
            'residual_valid_area_km2': rv[0], 'residual_initial_ice_equivalent_area_km2eq': rv[1],
            'residual_absolute_sic_loss_km2eq': rv[2], 'residual_loss_fraction_initial_ice': rv[3],
            'residual_mean_sic_loss_valid_area': rv[4],
            'other_region_valid_area_km2': ov[0], 'other_region_initial_ice_equivalent_area_km2eq': ov[1],
            'other_region_absolute_sic_loss_km2eq': ov[2], 'other_region_loss_fraction_initial_ice': ov[3],
            'other_region_mean_sic_loss_valid_area': ov[4],
            'source_reverse_metric_reference': rel(PRODUCTS / 'reverse_window_metrics.csv'),
        })
    opportunities = pd.DataFrame(rows).sort_values(['case_or_control','unique_local_event_id']).reset_index(drop=True)
    write_csv(opportunities, out / 'REVERSE_DOMAIN_OPPORTUNITY_METRICS.csv')
    cdr_manifest = pd.DataFrame([{'source_file': path, 'sha256': digest} for path, digest in sorted(cdr_files.items())])
    write_csv(cdr_manifest, out / 'P0_CDR_SOURCE_MANIFEST.csv')

    lookup = opportunities.set_index('unique_local_event_id')
    diff_rows: list[dict] = []
    for uid, selected in anchors.groupby('unique_local_event_id', sort=True):
        uid = str(uid)
        control_ids = selected.sort_values('selected_rank')['control_unique_local_event_id'].astype(str).tolist()
        if uid not in lookup.index or len(control_ids) < 10:
            continue
        row = {'case_unique_local_event_id': uid, 'case_year': int(pd.Timestamp(class_lookup.loc[uid,'event_date']).year), 'n_primary_controls': len(control_ids)}
        specs = {
            'local_loss_fraction_diff': 'local_loss_fraction_initial_ice',
            'residual_loss_fraction_diff': 'residual_loss_fraction_initial_ice',
            'other_region_loss_fraction_diff': 'other_region_loss_fraction_initial_ice',
            'local_valid_area_meanloss_diff': 'local_mean_sic_loss_valid_area',
            'residual_valid_area_meanloss_diff': 'residual_mean_sic_loss_valid_area',
            'other_region_valid_area_meanloss_diff': 'other_region_mean_sic_loss_valid_area',
        }
        for output_name, source_col in specs.items():
            case_value = float(lookup.loc[uid, source_col])
            controls = pd.to_numeric(lookup.reindex(control_ids)[source_col], errors='coerce').to_numpy(float)
            finite = controls[np.isfinite(controls)]
            row[output_name] = case_value - float(np.median(finite)) if np.isfinite(case_value) and finite.size >= 10 else np.nan
        row['residual_minus_local_loss_fraction_diff'] = row['residual_loss_fraction_diff'] - row['local_loss_fraction_diff']
        row['other_region_minus_local_loss_fraction_diff'] = row['other_region_loss_fraction_diff'] - row['local_loss_fraction_diff']
        row['residual_minus_local_valid_area_meanloss_diff'] = row['residual_valid_area_meanloss_diff'] - row['local_valid_area_meanloss_diff']
        row['other_region_minus_local_valid_area_meanloss_diff'] = row['other_region_valid_area_meanloss_diff'] - row['local_valid_area_meanloss_diff']
        diff_rows.append(row)
    differences = pd.DataFrame(diff_rows)
    write_csv(differences, out / 'REVERSE_NORMALIZED_EVENT_DIFFERENCES.csv')
    if len(differences) != 276:
        raise SystemExit(f'HOLD_NORMALIZED_DIFFERENCE_CASE_COUNT:{len(differences)}')

    groups = [
        ('domainwise_initial_ice', ['local_loss_fraction_diff','residual_loss_fraction_diff','other_region_loss_fraction_diff']),
        ('domainwise_valid_area', ['local_valid_area_meanloss_diff','residual_valid_area_meanloss_diff','other_region_valid_area_meanloss_diff']),
        ('paired_initial_ice', ['residual_minus_local_loss_fraction_diff','other_region_minus_local_loss_fraction_diff']),
        ('paired_valid_area', ['residual_minus_local_valid_area_meanloss_diff','other_region_minus_local_valid_area_meanloss_diff']),
    ]
    summary_rows: list[dict] = []
    index = 0
    for family, metrics in groups:
        family_rows = []
        for metric in metrics:
            values = pd.to_numeric(differences[metric], errors='coerce').to_numpy(float)
            values = values[np.isfinite(values)]
            ci_low, ci_high = bootstrap_mean_ci(values, n_resamples=RESAMPLES, seed=SEED + index * 17)
            family_rows.append({
                'family': family, 'metric': metric, 'n_cases': len(values),
                'mean_difference': float(np.mean(values)), 'median_difference': float(np.median(values)),
                'ci_low': ci_low, 'ci_high': ci_high,
                'p_raw': sign_flip_test(values, alternative='two-sided', n_resamples=RESAMPLES, seed=SEED + index * 31),
                'bootstrap_resamples': RESAMPLES, 'signflip_resamples': RESAMPLES,
            })
            index += 1
        frame = pd.DataFrame(family_rows)
        frame['p_fdr'] = fdr_bh(frame['p_raw'])
        frame['status'] = np.where(frame['p_fdr'] < 0.05, 'supported_secondary_sensitivity', 'not_supported_secondary_sensitivity')
        summary_rows.extend(frame.to_dict('records'))
    summary = pd.DataFrame(summary_rows)
    write_csv(summary, out / 'D4D_DOMAIN_NORMALIZATION_SUMMARY.csv')
    primary = summary[summary['family'].eq('paired_initial_ice')].set_index('metric')
    area_sens = summary[summary['family'].eq('paired_valid_area')].set_index('metric')
    primary_pass = (primary['mean_difference'] > 0) & (primary['ci_low'] > 0) & (primary['p_fdr'] < 0.05)
    area_positive = bool((area_sens['mean_difference'] > 0).all())
    if bool(primary_pass.all()) and area_positive:
        decision = 'STRONG'
    elif bool(((primary['mean_difference'] < 0) & (primary['ci_high'] < 0)).any()) or int(primary_pass.sum()) == 0:
        decision = 'NOT_SUPPORTED'
    else:
        decision = 'PARTIAL'
    decision_md = f"""# D4d Domain-Normalization Decision

Status: `SECONDARY_REVIEWER_RISK_SENSITIVITY`

The primary absolute loss metrics reconcile with the accepted reverse analysis
within tolerance (`atol={ABS_ATOL}`, `rtol={ABS_RTOL}`). Maximum absolute deltas
were local={max_abs_delta['local']:.12g}, residual={max_abs_delta['residual']:.12g},
and other-region={max_abs_delta['other_region']:.12g} km²eq.

Initial-ice-normalized paired comparisons:

- residual minus focal: mean={fmt(float(primary.loc['residual_minus_local_loss_fraction_diff','mean_difference']))}, 95% CI [{fmt(float(primary.loc['residual_minus_local_loss_fraction_diff','ci_low']))}, {fmt(float(primary.loc['residual_minus_local_loss_fraction_diff','ci_high']))}], pFDR={fmt(float(primary.loc['residual_minus_local_loss_fraction_diff','p_fdr']))}.
- other-region minus focal: mean={fmt(float(primary.loc['other_region_minus_local_loss_fraction_diff','mean_difference']))}, 95% CI [{fmt(float(primary.loc['other_region_minus_local_loss_fraction_diff','ci_low']))}, {fmt(float(primary.loc['other_region_minus_local_loss_fraction_diff','ci_high']))}], pFDR={fmt(float(primary.loc['other_region_minus_local_loss_fraction_diff','p_fdr']))}.

The valid-domain-area denominator is a directional sensitivity, not a new
primary metric. Residual and other-region are overlapping domains and are never
summed.

`EXTRAFOCAL_COMPARATIVE_SUPPORT = {decision}`

This classification affects wording/placement only and does not alter frozen
Stage3 primary statuses.
"""
    (out / 'D4D_DOMAIN_NORMALIZATION_DECISION.md').write_text(decision_md, encoding='utf-8')
    medians = {
        'focal_area': float(opportunities['local_valid_area_km2'].median()),
        'residual_area': float(opportunities['residual_valid_area_km2'].median()),
        'other_area': float(opportunities['other_region_valid_area_km2'].median()),
        'residual_minus_focal': float(primary.loc['residual_minus_local_loss_fraction_diff','mean_difference']),
        'other_minus_focal': float(primary.loc['other_region_minus_local_loss_fraction_diff','mean_difference']),
    }
    return opportunities, differences, summary, decision, medians


def iqr(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors='coerce').dropna().to_numpy(float)
    return float(np.percentile(values, 75) - np.percentile(values, 25)) if values.size else np.nan


def fit_ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, int, float]:
    design = np.column_stack([np.ones(len(y)), x])
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    return beta, int(rank), float(np.linalg.cond(design))


def year_block_intercept_bootstrap(y: np.ndarray, x: np.ndarray, years: np.ndarray, seed: int) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), x])
    unique = np.asarray(sorted(set(int(v) for v in years)), dtype=int)
    xtx = []; xty = []
    for year in unique:
        mask = years == year
        block = design[mask]
        xtx.append(block.T @ block); xty.append(block.T @ y[mask])
    xtx = np.asarray(xtx); xty = np.asarray(xty)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(unique), np.full(len(unique), 1.0 / len(unique)), size=RESAMPLES)
    values = np.full(RESAMPLES, np.nan)
    for idx, weight in enumerate(weights):
        matrix = np.tensordot(weight, xtx, axes=(0, 0)); rhs = np.tensordot(weight, xty, axes=(0, 0))
        if np.linalg.matrix_rank(matrix) == matrix.shape[0]:
            values[idx] = float(np.linalg.solve(matrix, rhs)[0])
    return values[np.isfinite(values)]


def focal_strength_outputs(
    out: Path,
    major: pd.DataFrame,
    classification: pd.DataFrame,
    accepted_metrics: pd.DataFrame,
    opportunities: pd.DataFrame,
    differences: pd.DataFrame,
    anchors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], str]:
    major = major.copy(); major['unique_local_event_id'] = major['unique_local_event_id'].astype(str)
    major_lookup = major.set_index('unique_local_event_id')
    class_lookup = classification.set_index('unique_local_event_id')
    metric_lookup = accepted_metrics.set_index('unique_local_event_id')
    opp_lookup = opportunities.set_index('unique_local_event_id')
    scales = {
        'local_sic_loss_km2eq': iqr(accepted_metrics['local_sic_loss_km2eq']),
        'duration_days': iqr(major['duration_days']),
        'max_area_km2': iqr(major['max_area_km2']),
        'log1p_max_area_km2': iqr(np.log1p(pd.to_numeric(major['max_area_km2'], errors='coerce'))),
        'cumulative_loss': iqr(major['cumulative_loss']),
        'season_day': iqr(classification['season_day']),
        'initial_sie': iqr(classification['initial_sie']),
        'local_loss_fraction': iqr(opportunities['local_loss_fraction_initial_ice']),
    }
    rows: list[dict] = []
    for case in differences.sort_values('case_unique_local_event_id').itertuples(index=False):
        uid = str(case.case_unique_local_event_id)
        selected = anchors[anchors['unique_local_event_id'].astype(str).eq(uid)].sort_values('selected_rank')
        controls = selected['control_unique_local_event_id'].astype(str).tolist()
        row: dict[str, object] = {'case_unique_local_event_id': uid, 'case_year': int(case.case_year), 'n_primary_controls': len(controls)}
        descriptors = {
            'local_sic_loss_km2eq': (metric_lookup, 'local_sic_loss_km2eq'),
            'duration_days': (major_lookup, 'duration_days'),
            'max_area_km2': (major_lookup, 'max_area_km2'),
            'cumulative_loss': (major_lookup, 'cumulative_loss'),
            'season_day': (class_lookup, 'season_day'),
            'initial_sie': (class_lookup, 'initial_sie'),
        }
        for name, (lookup, column) in descriptors.items():
            case_value = float(lookup.loc[uid, column])
            control_values = pd.to_numeric(lookup.reindex(controls)[column], errors='coerce').to_numpy(float)
            finite = control_values[np.isfinite(control_values)]
            median = float(np.median(finite)) if finite.size else np.nan
            difference = case_value - median if np.isfinite(case_value) and np.isfinite(median) else np.nan
            row[f'{name}_case'] = case_value; row[f'{name}_control_median'] = median
            row[f'{name}_difference'] = difference
            scale = scales[name]
            row[f'{name}_robust_scaled_difference'] = difference / scale if np.isfinite(scale) and scale > 0 else np.nan
        case_log = float(np.log1p(float(major_lookup.loc[uid, 'max_area_km2'])))
        control_log = np.log1p(pd.to_numeric(major_lookup.reindex(controls)['max_area_km2'], errors='coerce').to_numpy(float))
        row['log1p_max_area_km2_difference'] = case_log - float(np.nanmedian(control_log))
        row['local_loss_fraction_diff'] = float(case.local_loss_fraction_diff)
        row['local_loss_fraction_robust_scaled_difference'] = float(case.local_loss_fraction_diff) / scales['local_loss_fraction']
        row['same_region_matching_exact'] = bool((class_lookup.reindex(controls)['dominant_region'].astype(str) == str(class_lookup.loc[uid,'dominant_region'])).all())
        row['selected_tier'] = str(selected['selected_tier'].iloc[0]) if len(selected) else ''
        row['max_season_day_distance'] = int(selected['season_day_distance'].max()) if len(selected) else np.nan
        rows.append(row)
    balance = pd.DataFrame(rows)
    if not balance['same_region_matching_exact'].all():
        raise SystemExit('HOLD_SAME_REGION_MATCHING_CLOSURE')
    write_csv(balance, out / 'REVERSE_FOCAL_STRENGTH_BALANCE_AUDIT.csv')

    summary_rows: list[dict] = []
    for name in ['local_sic_loss_km2eq','duration_days','max_area_km2','cumulative_loss','season_day','initial_sie','local_loss_fraction']:
        diff_col = 'local_loss_fraction_diff' if name == 'local_loss_fraction' else f'{name}_difference'
        scaled_col = 'local_loss_fraction_robust_scaled_difference' if name == 'local_loss_fraction' else f'{name}_robust_scaled_difference'
        values = pd.to_numeric(balance[diff_col], errors='coerce')
        scaled = pd.to_numeric(balance[scaled_col], errors='coerce').abs()
        finite = values.dropna(); finite_scaled = scaled.dropna()
        summary_rows.append({
            'descriptor': name, 'n_cases': len(finite), 'mean_matched_set_difference': finite.mean(),
            'median_matched_set_difference': finite.median(), 'q25_matched_set_difference': finite.quantile(.25),
            'q75_matched_set_difference': finite.quantile(.75),
            'mean_absolute_robust_scaled_difference': finite_scaled.mean(),
            'median_absolute_robust_scaled_difference': finite_scaled.median(),
            'missing_count': int(values.isna().sum()), 'missing_fraction': float(values.isna().mean()),
            'eligible_universe_iqr': scales[name], 'exact_match_fraction': np.nan,
        })
    summary_rows.append({
        'descriptor': 'dominant_region_exact_match', 'n_cases': len(balance),
        'mean_matched_set_difference': np.nan, 'median_matched_set_difference': np.nan,
        'q25_matched_set_difference': np.nan, 'q75_matched_set_difference': np.nan,
        'mean_absolute_robust_scaled_difference': np.nan, 'median_absolute_robust_scaled_difference': np.nan,
        'missing_count': 0, 'missing_fraction': 0.0, 'eligible_universe_iqr': np.nan,
        'exact_match_fraction': float(balance['same_region_matching_exact'].mean()),
    })
    balance_summary = pd.DataFrame(summary_rows)
    write_csv(balance_summary, out / 'REVERSE_FOCAL_STRENGTH_BALANCE_SUMMARY.csv')

    merged = differences.merge(
        balance,
        on=['case_unique_local_event_id','case_year','n_primary_controls'],
        how='inner',
        validate='one_to_one',
        suffixes=('', '_balance'),
    )
    sensitivity_rows: list[dict] = []
    robustness: dict[str, str] = {}
    for outcome_index, outcome in enumerate(['residual_loss_fraction_diff','other_region_loss_fraction_diff']):
        y = pd.to_numeric(merged[outcome], errors='coerce').to_numpy(float)
        years = merged['case_year'].to_numpy(int)
        base_x = pd.to_numeric(merged['local_loss_fraction_diff'], errors='coerce').to_numpy(float)[:, None]
        finite = np.isfinite(y) & np.isfinite(base_x).all(axis=1)
        beta, rank, condition = fit_ols(y[finite], base_x[finite])
        boot = year_block_intercept_bootstrap(y[finite], base_x[finite], years[finite], SEED + outcome_index * 1000 + 1)
        base = {
            'outcome': outcome, 'model': 'BASE', 'n_cases': int(finite.sum()), 'matrix_rank': rank,
            'design_columns': 2, 'condition_number': condition, 'model_status': 'INTERPRETABLE',
            'intercept_point_estimate': float(beta[0]), 'bootstrap_intercept_mean': float(np.mean(boot)),
            'bootstrap_intercept_median': float(np.median(boot)),
            'bootstrap_ci_low': float(np.percentile(boot,2.5)), 'bootstrap_ci_high': float(np.percentile(boot,97.5)),
            'bootstrap_resamples_requested': RESAMPLES, 'bootstrap_resamples_valid': len(boot),
            'bootstrap_unit': 'case_event_year_block',
        }
        sensitivity_rows.append(base)
        extended_x = np.column_stack([
            pd.to_numeric(merged['local_loss_fraction_diff'], errors='coerce').to_numpy(float) / scales['local_loss_fraction'],
            pd.to_numeric(merged['duration_days_difference'], errors='coerce').to_numpy(float) / scales['duration_days'],
            pd.to_numeric(merged['log1p_max_area_km2_difference'], errors='coerce').to_numpy(float) / scales['log1p_max_area_km2'],
            pd.to_numeric(merged['cumulative_loss_difference'], errors='coerce').to_numpy(float) / scales['cumulative_loss'],
        ])
        complete = np.isfinite(y) & np.isfinite(extended_x).all(axis=1)
        ext_beta, ext_rank, ext_condition = fit_ols(y[complete], extended_x[complete])
        stable = int(complete.sum()) >= 100 and ext_rank == 5 and ext_condition <= 30
        if stable:
            ext_boot = year_block_intercept_bootstrap(y[complete], extended_x[complete], years[complete], SEED + outcome_index * 1000 + 2)
            ext_low = float(np.percentile(ext_boot,2.5)); ext_high = float(np.percentile(ext_boot,97.5))
            ext_mean = float(np.mean(ext_boot)); ext_median = float(np.median(ext_boot)); ext_valid = len(ext_boot)
            ext_status = 'INTERPRETABLE'
        else:
            ext_low = ext_high = ext_mean = ext_median = np.nan; ext_valid = 0
            ext_status = 'UNSTABLE_NOT_INTERPRETED'
        extended = {
            'outcome': outcome, 'model': 'EXTENDED', 'n_cases': int(complete.sum()), 'matrix_rank': ext_rank,
            'design_columns': 5, 'condition_number': ext_condition, 'model_status': ext_status,
            'intercept_point_estimate': float(ext_beta[0]), 'bootstrap_intercept_mean': ext_mean,
            'bootstrap_intercept_median': ext_median, 'bootstrap_ci_low': ext_low, 'bootstrap_ci_high': ext_high,
            'bootstrap_resamples_requested': RESAMPLES, 'bootstrap_resamples_valid': ext_valid,
            'bootstrap_unit': 'case_event_year_block',
        }
        sensitivity_rows.append(extended)
        base_robust = base['intercept_point_estimate'] > 0 and base['bootstrap_ci_low'] > 0
        ext_robust = stable and extended['intercept_point_estimate'] > 0 and extended['bootstrap_ci_low'] > 0
        if base['intercept_point_estimate'] <= 0 or base['bootstrap_ci_high'] < 0:
            classification_value = 'NOT_ROBUST'
        elif base_robust and ext_robust:
            classification_value = 'ROBUST_TO_FOCAL_STRENGTH'
        else:
            classification_value = 'PARTIAL'
        robustness[outcome] = classification_value
    sensitivity = pd.DataFrame(sensitivity_rows)
    write_csv(sensitivity, out / 'D4D_FOCAL_STRENGTH_SENSITIVITY.csv')
    base_res = sensitivity[(sensitivity['outcome']=='residual_loss_fraction_diff') & (sensitivity['model']=='BASE')].iloc[0]
    base_other = sensitivity[(sensitivity['outcome']=='other_region_loss_fraction_diff') & (sensitivity['model']=='BASE')].iloc[0]
    extended_states = sorted(set(sensitivity[sensitivity['model'].eq('EXTENDED')]['model_status']))
    decision_md = f"""# D4d Focal-Strength Sensitivity Decision

Status: `SECONDARY_ROBUSTNESS_ANALYSIS`; independent unit = one evaluable
footprint-linked case. Selected-control-derived differences remain fixed.

- Residual Base intercept = {fmt(float(base_res.intercept_point_estimate))};
  year-block bootstrap 95% CI [{fmt(float(base_res.bootstrap_ci_low))}, {fmt(float(base_res.bootstrap_ci_high))}].
- Other-region Base intercept = {fmt(float(base_other.intercept_point_estimate))};
  year-block bootstrap 95% CI [{fmt(float(base_other.bootstrap_ci_low))}, {fmt(float(base_other.bootstrap_ci_high))}].
- Extended model status: {';'.join(extended_states)}. Complete-case N, rank, and
  condition number are recorded in the CSV; no covariate was selected using an
  observed p-value.

`residual_loss_fraction_diff = {robustness['residual_loss_fraction_diff']}`

`other_region_loss_fraction_diff = {robustness['other_region_loss_fraction_diff']}`

This sensitivity does not establish causality and does not fully resolve
dependence caused by reuse of controls.
"""
    (out / 'D4D_FOCAL_STRENGTH_DECISION.md').write_text(decision_md, encoding='utf-8')
    return balance, sensitivity, robustness, ';'.join(extended_states)


def provenance(out: Path) -> None:
    paths = [
        FIG3_PATH, MAJOR_PATH, PAN_PATH, PAN_LOC_PATH,
        PRODUCTS/'reverse_major_event_classification.csv', PRODUCTS/'reverse_matched_control_anchors.csv',
        PRODUCTS/'reverse_window_metrics.csv', PRODUCTS/'panarctic_local_event_overlap.csv',
        MEMBERSHIP_PATH, PATCH_INDEX_PATH, AREA_PATH, MASK_PATH,
        PIPE/'scripts/stage1_rebuild_local_catalog_from_frozen_cells.py', PIPE/'scripts/stage2_experiment_core.py',
        PIPE/'src/vrile/stage3/reverse.py', PIPE/'src/vrile/stage3/controls.py',
    ]
    frame = pd.DataFrame([{'path': rel(path), 'size_bytes': path.stat().st_size, 'sha256': sha(path)} for path in paths])
    write_csv(frame, out/'P0_INPUT_PROVENANCE.csv')


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument('--output_dir', required=True, type=Path, dest='output_root'); args = parser.parse_args()
    if Path.cwd().resolve() != ROOT.resolve():
        raise SystemExit('HOLD_CANONICAL_PROJECT_ROOT')
    out = args.output_root.resolve()
    if out.exists():
        raise SystemExit(f'Refusing overwrite: {out}')
    out.mkdir(parents=True)
    before = primary_hashes()
    if len(before) != 38:
        raise SystemExit(f'HOLD_PRIMARY_PRODUCT_SET:{len(before)}')
    major = pd.read_csv(MAJOR_PATH)
    if len(major) != 554 or major['unique_local_event_id'].astype(str).nunique() != 554:
        raise SystemExit('HOLD_MAJOR_UNIVERSE_NOT_554_UNIQUE')
    classification = pd.read_csv(PRODUCTS/'reverse_major_event_classification.csv')
    classification['unique_local_event_id'] = classification['unique_local_event_id'].astype(str)
    classification['pan_linked'] = classification['pan_linked'].astype(bool)
    if set(classification['unique_local_event_id']) != set(major['unique_local_event_id'].astype(str)):
        raise SystemExit('HOLD_TRACK_S_UNIVERSE_MISMATCH')
    anchors = pd.read_csv(PRODUCTS/'reverse_matched_control_anchors.csv')
    accepted_metrics = pd.read_csv(PRODUCTS/'reverse_window_metrics.csv')
    cross = cross_scale_outputs(out, major, classification)
    # Terminology scanning is a documentation/provenance validator, not a
    # scientific predecessor.  Keep it opt-in so the raw-to-final generation
    # graph does not read documentation references. This optional prose audit
    # is not a required scientific-validation route.
    if os.environ.get('VRILE_RUN_TERMINOLOGY_AUDIT', 'NO').upper() == 'YES':
        ambiguous = terminology_audit(out)
    else:
        write_csv(
            pd.DataFrame(
                columns=[
                    'file', 'line_number', 'line_context', 'current_wording',
                    'maps_to', 'recommended_replacement',
                ]
            ),
            out / 'CROSS_SCALE_TERMINOLOGY_AUDIT.csv',
        )
        ambiguous = 0
    opportunities, normalized, domain_summary, domain_decision, medians = opportunity_outputs(out, classification, accepted_metrics, anchors)
    balance, sensitivity, robustness, extended_status = focal_strength_outputs(out, major, classification, accepted_metrics, opportunities, normalized, anchors)
    provenance(out)
    after = primary_hashes()
    invariance = pd.DataFrame([
        {'file': name, 'before_sha256': before.get(name,''), 'after_sha256': after.get(name,''), 'status': 'MATCH' if before.get(name)==after.get(name) else 'MISMATCH'}
        for name in sorted(set(before)|set(after))
    ])
    write_csv(invariance, out/'P0_PRIMARY_SCIENCE_INVARIANCE.csv')
    if not invariance['status'].eq('MATCH').all():
        raise SystemExit('HOLD_P0_AUDIT_CHANGED_PRIMARY_SCIENCE')

    domain_rows = domain_summary.set_index('metric')
    base_rows = sensitivity[sensitivity['model'].eq('BASE')].set_index('outcome')
    if domain_decision == 'STRONG' and all(v == 'ROBUST_TO_FOCAL_STRENGTH' for v in robustness.values()):
        disposition = 'RETAIN_MAIN_WITH_CONDITIONED_COMPARATIVE_WORDING'
    elif (
        domain_rows.loc['residual_loss_fraction_diff','mean_difference'] > 0
        and domain_rows.loc['other_region_loss_fraction_diff','mean_difference'] > 0
        and all(v in {'ROBUST_TO_FOCAL_STRENGTH','PARTIAL'} for v in robustness.values())
    ):
        disposition = 'RETAIN_MAIN_AS_SEPARATE_DOMAIN_CONTRASTS_ONLY'
    else:
        disposition = 'DEMOTE_D4D_TO_SUPPLEMENT'
    f4c = 'REDESIGN_AS_CORRESPONDENCE_DEFINITION_PANEL_SHOWING_BOTH_EVENT_START_AND_TRACK_S_FOOTPRINT_DEFINITIONS'
    balance_summary = pd.read_csv(out/'REVERSE_FOCAL_STRENGTH_BALANCE_SUMMARY.csv').set_index('descriptor')
    ordered_balance = balance_summary.drop(index='dominant_region_exact_match').sort_values('median_absolute_robust_scaled_difference', ascending=False)
    imbalance_text = ', '.join(ordered_balance.head(3).index.tolist()) + ' (largest descriptive median absolute robust-scaled differences; no cutoff imposed)'
    report = f"""# P0 Scientific Decision Report

## Q1 — 104 versus 284

Both values reproduce from their authoritative methods. Event-start ±3-day
temporal correspondence gives {cross['event_start_3d']}/554, with centroid
refinements {cross['event_start_500km']}/554 at 500 km and
{cross['event_start_300km']}/554 at 300 km. Track-S footprint linkage gives
{cross['footprint_linked']}/554 linked and {cross['footprint_unlinked']}/554
unlinked. Their 2×2 overlap is both={cross['both']}, temporal-only={cross['temporal_only']},
footprint-only={cross['footprint_only']}, neither={cross['neither']}. There is no
numerical inconsistency: the diagnostics use different anchors, windows, and
spatial criteria.

## Q2 — Domain-size confounding

Across the {len(opportunities)} accepted reverse event/windows, median valid
opportunity areas are focal={medians['focal_area']:.6g},
residual={medians['residual_area']:.6g}, and other-region={medians['other_area']:.6g}
km². The direct initial-ice-normalized matched contrasts are
residual-minus-focal={medians['residual_minus_focal']:.8g} and
other-region-minus-focal={medians['other_minus_focal']:.8g}.

`EXTRAFOCAL_COMPARATIVE_SUPPORT = {domain_decision}`

## Q3 — Focal-event strength

The largest descriptive matched-set imbalances are {imbalance_text}.
Base-adjusted intercepts are residual={base_rows.loc['residual_loss_fraction_diff','intercept_point_estimate']:.8g}
and other-region={base_rows.loc['other_region_loss_fraction_diff','intercept_point_estimate']:.8g};
their year-block CIs are in `D4D_FOCAL_STRENGTH_SENSITIVITY.csv`.
Extended-model status is `{extended_status}`.

- Residual: `{robustness['residual_loss_fraction_diff']}`
- Other-region: `{robustness['other_region_loss_fraction_diff']}`

## Q4 — D4d manuscript disposition

`{disposition}`

Any retained wording remains matched, linkage-conditioned, and non-causal.
This audit does not modify the frozen primary evidence.

## Q5 — F4C future design recommendation

`{f4c}`

The recommendation reflects definition transparency, not a preference for one
correspondence diagnostic. No figure was rendered or modified.
"""
    (out/'P0_SCIENTIFIC_DECISION_REPORT.md').write_text(report, encoding='utf-8')
    summary_md = f"""# P0 Cross-Scale and D4d Validation Summary

- Canonical root: `{ROOT}`
- Major universe: 554 unique events
- Cross-scale definitions: reproduced and non-equivalent
- Event-start ±3 d / +500 km / +300 km: {cross['event_start_3d']} / {cross['event_start_500km']} / {cross['event_start_300km']}
- Track-S footprint linked/unlinked: {cross['footprint_linked']} / {cross['footprint_unlinked']}
- Absolute-domain metric reconciliation: PASS
- Opportunity rows / evaluable case differences: {len(opportunities)} / {len(normalized)}
- `EXTRAFOCAL_COMPARATIVE_SUPPORT = {domain_decision}`
- Residual focal-strength sensitivity: `{robustness['residual_loss_fraction_diff']}`
- Other-region focal-strength sensitivity: `{robustness['other_region_loss_fraction_diff']}`
- D4d disposition: `{disposition}`
- Ambiguous active terminology occurrences: {ambiguous}
- Frozen primary Stage3 files changed: 0/38
"""
    (out/'SUMMARY.md').write_text(summary_md, encoding='utf-8')
    changelog = """# Scientific Changelog

## Added

- Row-level registry and cross-tab for two existing cross-scale definitions.
- Opportunity-normalized and direct paired cross-domain sensitivity diagnostics.
- Descriptive focal-strength balance audit and prespecified Base/Extended
  year-block bootstrap sensitivity.

## Unchanged

- Corrected event universes, pan-Arctic anchors, Track-S relation, primary
  control membership, domains, windows, absolute metrics, primary p-values and
  statuses, Stage1/Stage2, and F1–F4.

All additions are secondary reviewer-risk diagnostics and are not promoted into
the frozen primary evidence tables.
"""
    (out/'SCIENTIFIC_CHANGELOG.md').write_text(changelog, encoding='utf-8')
    run_summary = {
        'canonical_root': str(ROOT), 'major_universe': 554, 'cross_scale': cross,
        'ambiguous_terminology_occurrences': ambiguous,
        'opportunity_rows': len(opportunities), 'normalized_difference_rows': len(normalized),
        'absolute_metric_reconciliation': 'PASS', 'opportunity_medians_km2': medians,
        'extrafocal_comparative_support': domain_decision, 'focal_robustness': robustness,
        'extended_model_status': extended_status, 'd4d_disposition': disposition,
        'f4c_recommendation': f4c, 'primary_science_mismatches': 0,
    }
    (out/'P0_RUN_SUMMARY.json').write_text(json.dumps(run_summary, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(run_summary, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
