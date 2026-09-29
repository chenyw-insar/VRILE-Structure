"""Batch 4A Track S matched-control co-loss and compensation diagnostics."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from vrile.regions import NSIDC_0780_REGION_IDS
from vrile.stage3.budget import aligned_patch_rows, hierarchy_lookup
from vrile.stage3.grid import find_sic_file, load_cell_area_km2, load_pan_events, load_surface_mask, read_sic, valid_ocean_mask

CONTROL_TIERS = (30, 45, 60)
CONTROL_TARGET_N = 20
CONTROL_LIMITED_MIN_N = 10
METRICS = [
    "global_sic_loss_km2eq",
    "global_sic_gain_km2eq",
    "global_compensation_ratio",
    "residual_sic_loss_km2eq",
    "residual_sic_gain_km2eq",
    "residual_net_sic_loss_km2eq",
    "residual_compensation_ratio",
    "other_region_sic_loss_km2eq",
    "other_region_sic_gain_km2eq",
    "other_region_net_sic_loss_km2eq",
    "other_region_compensation_ratio",
]
COLOSS_METRICS = ["residual_sic_loss_km2eq", "other_region_sic_loss_km2eq"]
COMPENSATION_METRICS = ["global_compensation_ratio", "residual_compensation_ratio", "other_region_compensation_ratio"]
PRIMARY_OUTPUT_NAMES = [
    "panarctic_matched_control_anchors.csv",
    "panarctic_control_window_metrics.csv",
    "panarctic_coloss_event_differences.csv",
    "panarctic_regional_coloss_matrix.csv",
    "panarctic_synchronization_metrics.csv",
    "panarctic_control_test_summary.csv",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stage3_processed_root(root: Path) -> Path:
    value = os.environ.get("VRILE_STAGE3_DATA_ROOT", "").strip()
    return Path(value).resolve() if value else root / "data/processed"


def capture_batch3_hashes(root: Path, out_dir: Path, validation_root: Path) -> pd.DataFrame:
    files = [
        out_dir / "track_alignment_audit.csv",
        out_dir / "class_partition_overlap_audit.csv",
        out_dir / "panarctic_detector_contribution_budget.csv",
        out_dir / "panarctic_cdr_contribution_budget.csv",
        out_dir / "panarctic_detector_region_budget.csv",
        out_dir / "panarctic_cdr_region_budget.csv",
        out_dir / "region_budget_accounting_audit.csv",
        out_dir / "panarctic_spatial_composition_metrics.csv",
    ]
    rows = []
    for path in files:
        rows.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path), "kind": "batch3_csv"})
    part_dir = stage3_processed_root(root) / "panarctic_class_partition_fields"
    for path in sorted(part_dir.glob("PAN*.nc")):
        try:
            recorded_path = str(path.relative_to(root))
        except ValueError:
            recorded_path = str(path)
        rows.append({"path": recorded_path, "sha256": sha256_file(path), "kind": "class_partition_netcdf"})
    df = pd.DataFrame(rows)
    validation_root.mkdir(parents=True, exist_ok=True)
    baseline = validation_root / "batch3_frozen_hash_baseline.csv"
    if not baseline.exists():
        df.to_csv(baseline, index=False)
    return df


def check_batch3_hashes(root: Path, validation_root: Path) -> pd.DataFrame:
    baseline = pd.read_csv(validation_root / "batch3_frozen_hash_baseline.csv")
    rows = []
    for row in baseline.itertuples(index=False):
        path = root / row.path
        rows.append(
            {
                "path": row.path,
                "kind": row.kind,
                "baseline_sha256": row.sha256,
                "current_sha256": sha256_file(path),
                "status": "PASS" if sha256_file(path) == row.sha256 else "FAIL",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(validation_root / "batch3_frozen_hash_regression.csv", index=False)
    return out


def capture_primary_output_hashes(root: Path, out_dir: Path, validation_root: Path) -> pd.DataFrame:
    rows = []
    for name in PRIMARY_OUTPUT_NAMES:
        path = out_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen primary Batch 4A output: {path}")
        rows.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path), "kind": "batch4a_primary_csv"})
    df = pd.DataFrame(rows)
    validation_root.mkdir(parents=True, exist_ok=True)
    baseline = validation_root / "batch4a_primary_output_hash_baseline.csv"
    if not baseline.exists():
        df.to_csv(baseline, index=False)
    return df


def check_primary_output_hashes(
    root: Path,
    validation_root: Path,
    *,
    baseline_path: Path | None = None,
) -> pd.DataFrame:
    baseline_path = baseline_path or validation_root / "batch4a_primary_output_hash_baseline.csv"
    baseline = pd.read_csv(baseline_path)
    if len(baseline) != len(PRIMARY_OUTPUT_NAMES) or set(Path(v).name for v in baseline["path"]) != set(PRIMARY_OUTPUT_NAMES):
        raise ValueError(f"Invalid frozen Batch 4A primary baseline: {baseline_path}")
    rows = []
    for row in baseline.itertuples(index=False):
        path = root / row.path
        current = sha256_file(path)
        rows.append(
            {
                "path": row.path,
                "kind": row.kind,
                "baseline_sha256": row.sha256,
                "current_sha256": current,
                "status": "PASS" if current == row.sha256 else "FAIL",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(validation_root / "batch4a_primary_output_hash_regression.csv", index=False)
    return out


def primary_windows(events: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return [
        (pd.Timestamp(d).normalize() - pd.Timedelta(days=5), pd.Timestamp(d).normalize())
        for d in pd.to_datetime(events["date"]).dt.normalize()
    ]


def intervals_overlap(a: tuple[pd.Timestamp, pd.Timestamp], b: tuple[pd.Timestamp, pd.Timestamp]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def control_window_overlaps_primary(anchor: pd.Timestamp, windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> bool:
    interval = (anchor - pd.Timedelta(days=5), anchor)
    return any(intervals_overlap(interval, win) for win in windows)


def readable_track_s_window(anchor: pd.Timestamp) -> bool:
    try:
        find_sic_file(anchor - pd.Timedelta(days=5))
        find_sic_file(anchor)
        return True
    except FileNotFoundError:
        return False


def selected_control_anchors(events: pd.DataFrame, daily_sie: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sie = daily_sie.copy()
    sie["date"] = pd.to_datetime(sie["date"]).dt.normalize()
    sie = sie.set_index("date").sort_index()
    windows = primary_windows(events)
    summary_rows = []
    selected_rows = []
    jja = sie[sie.index.month.isin([6, 7, 8])].copy()
    for event in events.itertuples(index=False):
        pan_id = str(event.pan_event_id)
        event_date = pd.Timestamp(event.date).normalize()
        init_date = event_date - pd.Timedelta(days=5)
        if init_date not in sie.index:
            raise ValueError(f"Missing event initial SIE date {init_date:%Y-%m-%d} for {pan_id}")
        event_initial = float(sie.loc[init_date, "extent"])
        all_candidates = []
        year_jja = jja[jja.index.year == event_date.year]
        for anchor, row in year_jja.iterrows():
            start = anchor - pd.Timedelta(days=5)
            if start not in sie.index:
                continue
            if control_window_overlaps_primary(anchor, windows):
                continue
            if not readable_track_s_window(anchor):
                continue
            all_candidates.append(
                {
                    "control_anchor_date": anchor,
                    "control_window_start": start,
                    "control_window_end": anchor,
                    "control_initial_sie": float(sie.loc[start, "extent"]),
                    "abs_initial_sie_difference": abs(float(sie.loc[start, "extent"]) - event_initial),
                    "abs_day_difference": abs((anchor - event_date).days),
                }
            )
        candidates = pd.DataFrame(all_candidates)
        tier_counts = {
            tier: int(len(candidates[candidates["abs_day_difference"] <= tier])) if not candidates.empty else 0
            for tier in CONTROL_TIERS
        }
        chosen_tier = np.nan
        selected = pd.DataFrame()
        status = "insufficient_background_controls"
        for tier in CONTROL_TIERS:
            tier_df = candidates[candidates["abs_day_difference"] <= tier].copy() if not candidates.empty else pd.DataFrame()
            if len(tier_df) >= CONTROL_TARGET_N:
                chosen_tier = tier
                selected = tier_df.sort_values(["abs_initial_sie_difference", "control_anchor_date"]).head(CONTROL_TARGET_N).copy()
                status = "matched_controls"
                break
        if selected.empty:
            tier = CONTROL_TIERS[-1]
            chosen_tier = tier
            tier_df = candidates[candidates["abs_day_difference"] <= tier].copy() if not candidates.empty else pd.DataFrame()
            if len(tier_df) >= CONTROL_LIMITED_MIN_N:
                selected = tier_df.sort_values(["abs_initial_sie_difference", "control_anchor_date"]).copy()
                status = "limited_background_controls"
            else:
                selected = tier_df.sort_values(["abs_initial_sie_difference", "control_anchor_date"]).copy()
        summary_rows.append(
            {
                "pan_event_id": pan_id,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "event_window_start": init_date.strftime("%Y-%m-%d"),
                "event_window_end": event_date.strftime("%Y-%m-%d"),
                "event_initial_sie": event_initial,
                "candidate_count_tier_30": tier_counts.get(30, 0),
                "candidate_count_tier_45": tier_counts.get(45, 0),
                "candidate_count_tier_60": tier_counts.get(60, int(len(candidates))),
                "selected_tier_days": int(chosen_tier) if np.isfinite(chosen_tier) else np.nan,
                "selected_control_count": int(len(selected)),
                "matching_status": status,
            }
        )
        for rank, control in enumerate(selected.itertuples(index=False), start=1):
            selected_rows.append(
                {
                    "pan_event_id": pan_id,
                    "event_date": event_date.strftime("%Y-%m-%d"),
                    "event_initial_sie": event_initial,
                    "matching_status": status,
                    "selected_tier_days": int(chosen_tier),
                    "selected_rank": rank,
                    "control_anchor_date": control.control_anchor_date.strftime("%Y-%m-%d"),
                    "control_window_start": control.control_window_start.strftime("%Y-%m-%d"),
                    "control_window_end": control.control_window_end.strftime("%Y-%m-%d"),
                    "control_initial_sie": control.control_initial_sie,
                    "abs_initial_sie_difference": control.abs_initial_sie_difference,
                    "abs_day_difference": int(control.abs_day_difference),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(selected_rows)


def reconstruct_matching_independently(events: pd.DataFrame, daily_sie: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sie = daily_sie.copy()
    sie["date"] = pd.to_datetime(sie["date"]).dt.normalize()
    sie = sie.set_index("date").sort_index()
    event_dates = pd.to_datetime(events["date"]).dt.normalize()
    primary = [(date - pd.Timedelta(days=5), date) for date in event_dates]
    jja_dates = [date for date in sie.index if date.month in (6, 7, 8)]
    summary_rows: list[dict] = []
    selected_rows: list[dict] = []
    for event in events.itertuples(index=False):
        pan_id = str(event.pan_event_id)
        event_date = pd.Timestamp(event.date).normalize()
        event_start = event_date - pd.Timedelta(days=5)
        if event_start not in sie.index:
            raise ValueError(f"Missing SIE initial state for {pan_id}: {event_start:%Y-%m-%d}")
        event_initial = float(sie.loc[event_start, "extent"])
        candidates: list[dict] = []
        for anchor in jja_dates:
            anchor = pd.Timestamp(anchor).normalize()
            if anchor.year != event_date.year:
                continue
            start = anchor - pd.Timedelta(days=5)
            if start not in sie.index:
                continue
            window = (start, anchor)
            if any(window[0] <= end and begin <= window[1] for begin, end in primary):
                continue
            try:
                find_sic_file(start)
                find_sic_file(anchor)
            except FileNotFoundError:
                continue
            control_initial = float(sie.loc[start, "extent"])
            candidates.append(
                {
                    "control_anchor_date": anchor,
                    "control_window_start": start,
                    "control_window_end": anchor,
                    "control_initial_sie": control_initial,
                    "abs_initial_sie_difference": abs(control_initial - event_initial),
                    "abs_day_difference": abs((anchor - event_date).days),
                }
            )
        cand = pd.DataFrame(candidates)
        tier_counts = {
            tier: int(len(cand[cand["abs_day_difference"] <= tier])) if not cand.empty else 0
            for tier in CONTROL_TIERS
        }
        selected = pd.DataFrame()
        selected_tier = CONTROL_TIERS[-1]
        status = "insufficient_background_controls"
        for tier in CONTROL_TIERS:
            tier_df = cand[cand["abs_day_difference"] <= tier].copy() if not cand.empty else pd.DataFrame()
            if len(tier_df) >= CONTROL_TARGET_N:
                selected_tier = tier
                selected = tier_df.sort_values(["abs_initial_sie_difference", "control_anchor_date"]).head(CONTROL_TARGET_N).copy()
                status = "matched_controls"
                break
        if selected.empty:
            tier_df = cand[cand["abs_day_difference"] <= CONTROL_TIERS[-1]].copy() if not cand.empty else pd.DataFrame()
            selected = tier_df.sort_values(["abs_initial_sie_difference", "control_anchor_date"]).copy()
            status = "limited_background_controls" if len(selected) >= CONTROL_LIMITED_MIN_N else "insufficient_background_controls"
        summary_rows.append(
            {
                "pan_event_id": pan_id,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "event_window_start": event_start.strftime("%Y-%m-%d"),
                "event_window_end": event_date.strftime("%Y-%m-%d"),
                "event_initial_sie": event_initial,
                "candidate_count_tier_30": tier_counts[30],
                "candidate_count_tier_45": tier_counts[45],
                "candidate_count_tier_60": tier_counts[60],
                "selected_tier_days": int(selected_tier),
                "selected_control_count": int(len(selected)),
                "matching_status": status,
            }
        )
        for rank, row in enumerate(selected.itertuples(index=False), start=1):
            selected_rows.append(
                {
                    "pan_event_id": pan_id,
                    "event_date": event_date.strftime("%Y-%m-%d"),
                    "event_initial_sie": event_initial,
                    "matching_status": status,
                    "selected_tier_days": int(selected_tier),
                    "selected_rank": rank,
                    "control_anchor_date": row.control_anchor_date.strftime("%Y-%m-%d"),
                    "control_window_start": row.control_window_start.strftime("%Y-%m-%d"),
                    "control_window_end": row.control_window_end.strftime("%Y-%m-%d"),
                    "control_initial_sie": row.control_initial_sie,
                    "abs_initial_sie_difference": row.abs_initial_sie_difference,
                    "abs_day_difference": int(row.abs_day_difference),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(selected_rows)


def matching_reconstruction_audit(
    current_summary: pd.DataFrame,
    current_anchors: pd.DataFrame,
    reconstructed_summary: pd.DataFrame,
    reconstructed_anchors: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    summary_cols = [
        "event_initial_sie",
        "candidate_count_tier_30",
        "candidate_count_tier_45",
        "candidate_count_tier_60",
        "selected_tier_days",
        "selected_control_count",
        "matching_status",
    ]
    cur_s = current_summary.set_index("pan_event_id")
    rec_s = reconstructed_summary.set_index("pan_event_id")
    for pan_id in sorted(set(cur_s.index) | set(rec_s.index)):
        mismatches = []
        for col in summary_cols:
            cur = cur_s.loc[pan_id, col] if pan_id in cur_s.index else np.nan
            rec = rec_s.loc[pan_id, col] if pan_id in rec_s.index else np.nan
            if isinstance(cur, (float, np.floating)) or isinstance(rec, (float, np.floating)):
                ok = bool(np.isclose(float(cur), float(rec), rtol=0.0, atol=1e-12, equal_nan=True))
            else:
                ok = str(cur) == str(rec)
            if not ok:
                mismatches.append(col)
        rows.append({"record_type": "event_summary", "pan_event_id": pan_id, "selected_rank": 0, "status": "PASS" if not mismatches else "FAIL", "mismatch_fields": ";".join(mismatches)})
    anchor_cols = [
        "event_initial_sie",
        "matching_status",
        "selected_tier_days",
        "control_anchor_date",
        "control_window_start",
        "control_window_end",
        "control_initial_sie",
        "abs_initial_sie_difference",
        "abs_day_difference",
    ]
    cur_a = current_anchors.set_index(["pan_event_id", "selected_rank"]).sort_index()
    rec_a = reconstructed_anchors.set_index(["pan_event_id", "selected_rank"]).sort_index()
    for key in sorted(set(cur_a.index) | set(rec_a.index)):
        mismatches = []
        for col in anchor_cols:
            cur = cur_a.loc[key, col] if key in cur_a.index else np.nan
            rec = rec_a.loc[key, col] if key in rec_a.index else np.nan
            if isinstance(cur, (float, np.floating, int, np.integer)) or isinstance(rec, (float, np.floating, int, np.integer)):
                ok = bool(np.isclose(float(cur), float(rec), rtol=0.0, atol=1e-12, equal_nan=True))
            else:
                ok = str(cur) == str(rec)
            if not ok:
                mismatches.append(col)
        rows.append({"record_type": "selected_control", "pan_event_id": key[0], "selected_rank": int(key[1]), "status": "PASS" if not mismatches else "FAIL", "mismatch_fields": ";".join(mismatches)})
    return pd.DataFrame(rows)


def matching_quality_audit(control_summary: pd.DataFrame, control_anchors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for pan_id, group in control_anchors.groupby("pan_event_id", sort=False):
        event_initial = float(group["event_initial_sie"].iloc[0])
        if not np.isfinite(event_initial) or event_initial == 0:
            raise ValueError(f"Invalid event_initial_sie for matching-quality audit: {pan_id}={event_initial}")
        absdiff = pd.to_numeric(group["abs_initial_sie_difference"], errors="coerce").to_numpy(float)
        signed = pd.to_numeric(group["control_initial_sie"], errors="coerce").to_numpy(float) - event_initial
        rel = absdiff / abs(event_initial)
        summary_row = control_summary.set_index("pan_event_id").loc[pan_id]
        rows.append(
            {
                "pan_event_id": pan_id,
                "event_initial_sie": event_initial,
                "selected_tier_days": int(summary_row["selected_tier_days"]),
                "n_selected_controls": int(len(group)),
                "median_abs_initial_sie_difference": float(np.nanmedian(absdiff)),
                "p90_abs_initial_sie_difference": float(np.nanpercentile(absdiff, 90)),
                "max_abs_initial_sie_difference": float(np.nanmax(absdiff)),
                "median_relative_initial_sie_difference": float(np.nanmedian(rel)),
                "p90_relative_initial_sie_difference": float(np.nanpercentile(rel, 90)),
                "max_relative_initial_sie_difference": float(np.nanmax(rel)),
                "median_signed_control_minus_event_sie": float(np.nanmedian(signed)),
            }
        )
    return pd.DataFrame(rows)


def differences_from_metric_windows(metrics: pd.DataFrame, control_count: int) -> pd.DataFrame:
    rows: list[dict] = []
    for pan_id, group in metrics.groupby("pan_event_id", sort=False):
        event_rows = group[group["window_role"].eq("event")]
        control_rows = group[(group["window_role"].eq("control")) & (pd.to_numeric(group["selected_rank"], errors="coerce") <= control_count)]
        event_date = str(event_rows["event_date"].iloc[0])
        for metric in METRICS:
            event_value = float(event_rows[metric].iloc[0])
            values = pd.to_numeric(control_rows[metric], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            if np.isfinite(event_value) and finite.size >= CONTROL_LIMITED_MIN_N:
                control_median = float(np.median(finite))
                difference = event_value - control_median
                status = "ok"
            else:
                control_median = np.nan
                difference = np.nan
                status = "insufficient_controls"
            rows.append(
                {
                    "pan_event_id": pan_id,
                    "event_date": event_date,
                    "control_count": int(control_count),
                    "metric": metric,
                    "event_value": event_value,
                    "control_median": control_median,
                    "difference": difference,
                    "n_valid_controls": int(finite.size),
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def control_count_sensitivity(
    metrics: pd.DataFrame,
    primary_tests: pd.DataFrame,
    *,
    counts: tuple[int, ...],
    seed: int,
    n_bootstrap: int,
    n_signflip: int,
    primary_differences: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_diffs = []
    summaries = []
    primary = primary_tests.set_index("metric")
    for count in counts:
        if count == CONTROL_TARGET_N and primary_differences is not None:
            diffs = primary_differences.copy()
            diffs["control_count"] = int(count)
        else:
            diffs = differences_from_metric_windows(metrics, count)
        tests = infer_control_tests(diffs, seed=seed, n_bootstrap=n_bootstrap, n_signflip=n_signflip)
        tests["control_count"] = int(count)
        tests["analysis_role"] = "primary_regression" if count == CONTROL_TARGET_N else "control_count_sensitivity"
        tests["direction_consistent_with_primary"] = tests.apply(
            lambda row: bool(
                row["metric"] in primary.index
                and np.sign(float(row["mean_difference"])) == np.sign(float(primary.loc[row["metric"], "mean_difference"]))
            ),
            axis=1,
        )
        tests["status_consistent_with_primary"] = tests.apply(
            lambda row: bool(row["metric"] in primary.index and str(row["status"]) == str(primary.loc[row["metric"], "status"])),
            axis=1,
        )
        event_diffs.append(diffs)
        summaries.append(tests)
    return pd.concat(event_diffs, ignore_index=True), pd.concat(summaries, ignore_index=True)


def classify_year_block_confirmation(
    *,
    primary_status: str,
    primary_mean: float,
    ci_low: float,
    ci_high: float,
    n_years: int,
    n_valid_bootstrap: int,
) -> dict[str, object]:
    """Classify a confirmation-only year-block interval for a supported metric."""
    if str(primary_status) != "supported":
        raise ValueError("year-block confirmation is only defined for primary supported metrics")
    if not np.isfinite(primary_mean):
        primary_direction = "nonfinite"
    elif primary_mean > 0:
        primary_direction = "positive"
    elif primary_mean < 0:
        primary_direction = "negative"
    else:
        primary_direction = "zero"

    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        ci_direction = "nonfinite"
    elif ci_low > ci_high:
        ci_direction = "invalid"
    elif ci_low > 0:
        ci_direction = "positive"
    elif ci_high < 0:
        ci_direction = "negative"
    else:
        ci_direction = "crosses_zero"
    direction_consistent = bool(
        primary_direction in {"positive", "negative"} and ci_direction == primary_direction
    )
    ci_excludes_zero = ci_direction in {"positive", "negative"}

    if n_years < 2:
        confirmed = False
        status = "supported_but_year_block_confirmation_insufficient"
        reason = "fewer_than_two_year_blocks"
    elif n_valid_bootstrap <= 0:
        confirmed = False
        status = "supported_but_year_block_confirmation_insufficient"
        reason = "no_valid_year_block_bootstrap_resamples"
    elif primary_direction == "nonfinite":
        confirmed = False
        status = "supported_but_year_block_confirmation_insufficient"
        reason = "primary_direction_nonfinite"
    elif primary_direction == "zero":
        confirmed = False
        status = "supported_but_year_block_confirmation_insufficient"
        reason = "primary_direction_zero"
    elif ci_direction == "nonfinite":
        confirmed = False
        status = "supported_but_year_block_confirmation_insufficient"
        reason = "year_block_ci_nonfinite"
    elif ci_direction == "invalid":
        confirmed = False
        status = "supported_but_year_block_confirmation_insufficient"
        reason = "year_block_ci_invalid"
    elif ci_direction == "crosses_zero":
        confirmed = False
        status = "supported_but_not_year_block_confirmed"
        reason = "year_block_ci_crosses_zero"
    elif ci_direction != primary_direction:
        confirmed = False
        status = "supported_but_year_block_direction_conflict"
        reason = "year_block_ci_opposes_primary_direction"
    else:
        confirmed = True
        status = "supported_year_block_confirmed"
        reason = "ci_excludes_zero_and_matches_primary_direction"
    return {
        "primary_direction": primary_direction,
        "ci_direction": ci_direction,
        "direction_consistent_with_primary": direction_consistent,
        "ci_excludes_zero": ci_excludes_zero,
        "year_block_confirmed": confirmed,
        "interpretation_downgraded": not confirmed,
        "final_interpretation_status": status,
        "confirmation_reason": reason,
    }


def validate_year_block_confirmation_gates(
    year_block: pd.DataFrame,
    primary_tests: pd.DataFrame,
    *,
    count_column: str,
) -> pd.DataFrame:
    """Independently recalculate the shared year-block output contract."""
    supported = set(
        primary_tests.loc[primary_tests["status"].eq("supported"), "metric"].astype(str)
    )
    required = {
        "metric",
        count_column,
        "n_years",
        "primary_mean_difference",
        "block_bootstrap_ci_low",
        "block_bootstrap_ci_high",
        "ci_excludes_zero",
        "primary_status",
        "primary_direction",
        "ci_direction",
        "direction_consistent_with_primary",
        "n_bootstrap_requested",
        "n_bootstrap_valid",
        "year_block_confirmed",
        "interpretation_downgraded",
        "final_interpretation_status",
        "confirmation_reason",
        "analysis_role",
    }
    missing = required - set(year_block.columns)
    actual = set(year_block["metric"].astype(str)) if "metric" in year_block else set()
    duplicate_count = (
        int(year_block.duplicated("metric").sum()) if "metric" in year_block else len(year_block)
    )
    numeric_valid = not missing
    semantic_valid = not missing
    if not missing:
        for row in year_block.itertuples(index=False):
            count = int(getattr(row, count_column))
            n_years = int(row.n_years)
            n_requested = int(row.n_bootstrap_requested)
            n_valid = int(row.n_bootstrap_valid)
            low = float(row.block_bootstrap_ci_low)
            high = float(row.block_bootstrap_ci_high)
            primary_mean = float(row.primary_mean_difference)
            numeric_valid = bool(
                numeric_valid
                and count >= 0
                and n_years >= 0
                and n_requested >= 0
                and 0 <= n_valid <= n_requested
                and (
                    (n_years < 2 or n_valid == 0)
                    or (np.isfinite(primary_mean) and np.isfinite(low) and np.isfinite(high) and low <= high)
                )
            )
            expected = classify_year_block_confirmation(
                primary_status=str(row.primary_status),
                primary_mean=primary_mean,
                ci_low=low,
                ci_high=high,
                n_years=n_years,
                n_valid_bootstrap=n_valid,
            )
            semantic_valid = bool(
                semantic_valid
                and str(row.primary_status) == "supported"
                and str(row.primary_direction) == expected["primary_direction"]
                and str(row.ci_direction) == expected["ci_direction"]
                and bool(row.direction_consistent_with_primary)
                == expected["direction_consistent_with_primary"]
                and bool(row.ci_excludes_zero) == expected["ci_excludes_zero"]
                and bool(row.year_block_confirmed) == expected["year_block_confirmed"]
                and bool(row.interpretation_downgraded) == expected["interpretation_downgraded"]
                and str(row.final_interpretation_status) == expected["final_interpretation_status"]
                and str(row.confirmation_reason) == expected["confirmation_reason"]
            )
    confirmed = int(year_block.get("year_block_confirmed", pd.Series(dtype=bool)).astype(bool).sum())
    downgraded = int(year_block.get("interpretation_downgraded", pd.Series(dtype=bool)).astype(bool).sum())
    insufficient = int(
        year_block.get("final_interpretation_status", pd.Series(dtype=str))
        .astype(str)
        .eq("supported_but_year_block_confirmation_insufficient")
        .sum()
    )
    gates = [
        (
            "year_block_metric_set",
            actual == supported,
            f"actual={sorted(actual)} supported={sorted(supported)}",
        ),
        (
            "year_block_one_row_per_supported_metric",
            duplicate_count == 0 and len(year_block) == len(supported),
            f"rows={len(year_block)} supported={len(supported)} duplicates={duplicate_count}",
        ),
        (
            "year_block_numeric_contract",
            numeric_valid,
            f"missing_fields={sorted(missing)}",
        ),
        (
            "year_block_semantic_contract",
            semantic_valid,
            f"confirmed={confirmed}/{len(year_block)} downgraded={downgraded}/{len(year_block)} insufficient={insufficient}/{len(year_block)}",
        ),
        (
            "year_block_analysis_role",
            bool(not missing and year_block["analysis_role"].eq("confirmation_only").all()),
            f"rows={len(year_block)}",
        ),
    ]
    return pd.DataFrame(
        [
            {"gate": name, "status": "PASS" if passed else "FAIL", "detail": detail}
            for name, passed, detail in gates
        ]
    )


def year_block_bootstrap_confirmation(
    differences: pd.DataFrame,
    primary_tests: pd.DataFrame,
    *,
    seed: int,
    n_bootstrap: int,
) -> pd.DataFrame:
    supported = primary_tests[primary_tests["status"].eq("supported")]["metric"].astype(str).tolist()
    rows: list[dict] = []
    for i, metric in enumerate(supported):
        sub = differences[(differences["metric"].eq(metric)) & differences["difference"].notna()].copy()
        sub["year"] = pd.to_datetime(sub["event_date"]).dt.year
        years = sorted(sub["year"].dropna().unique())
        blocks = {year: pd.to_numeric(sub.loc[sub["year"].eq(year), "difference"], errors="coerce").dropna().to_numpy(float) for year in years}
        primary_values = pd.to_numeric(sub["difference"], errors="coerce").dropna().to_numpy(float)
        if not years or primary_values.size == 0:
            ci_low = np.nan
            ci_high = np.nan
        else:
            rng = np.random.default_rng(seed + i * 101)
            means = []
            for _ in range(n_bootstrap):
                sampled = rng.choice(np.asarray(years), size=len(years), replace=True)
                vals = np.concatenate([blocks[int(year)] for year in sampled if blocks[int(year)].size])
                means.append(np.mean(vals) if vals.size else np.nan)
            arr = np.asarray(means, dtype=float)
            arr = arr[np.isfinite(arr)]
            ci_low = float(np.percentile(arr, 2.5)) if arr.size else np.nan
            ci_high = float(np.percentile(arr, 97.5)) if arr.size else np.nan
        primary_mean = float(np.mean(primary_values)) if primary_values.size else np.nan
        semantic = classify_year_block_confirmation(
            primary_status="supported",
            primary_mean=primary_mean,
            ci_low=ci_low,
            ci_high=ci_high,
            n_years=len(years),
            n_valid_bootstrap=int(arr.size) if years and primary_values.size else 0,
        )
        rows.append(
            {
                "metric": metric,
                "n_events": int(primary_values.size),
                "n_years": int(len(years)),
                "primary_mean_difference": primary_mean,
                "block_bootstrap_ci_low": ci_low,
                "block_bootstrap_ci_high": ci_high,
                "primary_status": "supported",
                "n_bootstrap_requested": int(n_bootstrap),
                "n_bootstrap_valid": int(arr.size) if years and primary_values.size else 0,
                **semantic,
                "analysis_role": "confirmation_only",
            }
        )
    return pd.DataFrame(rows)


def source_major_region_codes(
    event_date: pd.Timestamp,
    membership: pd.DataFrame,
    major_events: pd.DataFrame,
    region_map: dict[str, int] | None = None,
) -> tuple[list[int], list[str]]:
    region_map = region_map or NSIDC_0780_REGION_IDS
    membership = membership.copy()
    membership["unique_local_event_id"] = membership["unique_local_event_id"].astype(str)
    membership["object_id"] = membership["object_id"].astype(str)
    membership["date"] = pd.to_datetime(membership["date"]).dt.normalize()
    membership["start_date"] = pd.to_datetime(membership["start_date"]).dt.normalize()
    major_lookup = major_events.set_index("unique_local_event_id")["dominant_region"].astype(str).to_dict()
    levels = set(major_events["unique_local_event_id"].astype(str))
    aligned = aligned_patch_rows(membership, event_date, "cdr")
    ids = sorted(set(aligned["unique_local_event_id"].astype(str)) & levels)
    regions = sorted({major_lookup[uid] for uid in ids})
    unknown = [r for r in regions if r not in region_map or region_map[r] == 0]
    if unknown:
        raise ValueError(f"Unknown or non-named dominant_region in aligned major_severe events: {unknown}")
    return sorted(region_map[r] for r in regions), ids


def cdr_window_fields(start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start, _, _ = read_sic(start_date)
    end, _, _ = read_sic(end_date)
    signed = end - start
    signed = np.where(np.isfinite(signed), signed, np.nan)
    loss = np.where(np.isfinite(signed), np.maximum(0.0, -signed), np.nan)
    gain = np.where(np.isfinite(signed), np.maximum(0.0, signed), np.nan)
    return signed, loss, gain


def compensation_ratio(gain: float, loss: float) -> float:
    return gain / loss if np.isfinite(loss) and loss > 0 else np.nan


def window_metrics(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    residual_domain: np.ndarray,
    other_region_domain: np.ndarray,
    region_codes: np.ndarray,
    area_km2: np.ndarray,
    valid_ocean: np.ndarray,
    local_domain: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[int, float]]:
    _, loss, gain = cdr_window_fields(start_date, end_date)
    loss_w = np.where(valid_ocean & np.isfinite(loss), loss * area_km2, 0.0)
    gain_w = np.where(valid_ocean & np.isfinite(gain), gain * area_km2, 0.0)
    global_loss = float(np.nansum(loss_w, dtype=np.float64))
    global_gain = float(np.nansum(gain_w, dtype=np.float64))
    residual_loss = float(np.nansum(np.where(residual_domain, loss_w, 0.0), dtype=np.float64))
    residual_gain = float(np.nansum(np.where(residual_domain, gain_w, 0.0), dtype=np.float64))
    other_loss = float(np.nansum(np.where(other_region_domain, loss_w, 0.0), dtype=np.float64))
    other_gain = float(np.nansum(np.where(other_region_domain, gain_w, 0.0), dtype=np.float64))
    metrics = {
        "global_sic_loss_km2eq": global_loss,
        "global_sic_gain_km2eq": global_gain,
        "global_compensation_ratio": compensation_ratio(global_gain, global_loss),
        "residual_sic_loss_km2eq": residual_loss,
        "residual_sic_gain_km2eq": residual_gain,
        "residual_net_sic_loss_km2eq": residual_loss - residual_gain,
        "residual_compensation_ratio": compensation_ratio(residual_gain, residual_loss),
        "other_region_sic_loss_km2eq": other_loss,
        "other_region_sic_gain_km2eq": other_gain,
        "other_region_net_sic_loss_km2eq": other_loss - other_gain,
        "other_region_compensation_ratio": compensation_ratio(other_gain, other_loss),
    }
    if local_domain is not None:
        if local_domain.shape != valid_ocean.shape:
            raise ValueError(f"local_domain shape mismatch: {local_domain.shape} != {valid_ocean.shape}")
        local = valid_ocean & np.asarray(local_domain, dtype=bool)
        local_loss = float(np.nansum(np.where(local, loss_w, 0.0), dtype=np.float64))
        local_gain = float(np.nansum(np.where(local, gain_w, 0.0), dtype=np.float64))
        metrics.update(
            {
                "local_sic_loss_km2eq": local_loss,
                "local_sic_gain_km2eq": local_gain,
                "local_net_sic_loss_km2eq": local_loss - local_gain,
                "local_compensation_ratio": compensation_ratio(local_gain, local_loss),
            }
        )
    regional_loss = {
        code: float(np.nansum(np.where((region_codes == code) & valid_ocean, loss_w, 0.0), dtype=np.float64))
        for code in range(1, 19)
    }
    return metrics, regional_loss


def fdr_bh(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return out
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy(float)
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[order] = np.minimum(adjusted, 1.0)
    return out


def sign_flip_test(diffs: np.ndarray, *, alternative: str, n_resamples: int, seed: int) -> float:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return np.nan
    obs = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, diffs.size))
    null = np.mean(signs * diffs, axis=1)
    if alternative == "greater":
        return float((np.sum(null >= obs) + 1) / (n_resamples + 1))
    if alternative == "less":
        return float((np.sum(null <= obs) + 1) / (n_resamples + 1))
    if alternative == "two-sided":
        return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_resamples + 1))
    raise ValueError(f"Unknown sign-flip alternative: {alternative}")


def bootstrap_mean_ci(diffs: np.ndarray, *, n_resamples: int, seed: int) -> tuple[float, float]:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_resamples, diffs.size))
    means = np.mean(diffs[idx], axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def infer_control_tests(differences: pd.DataFrame, *, seed: int, n_bootstrap: int, n_signflip: int) -> pd.DataFrame:
    rows = []
    for family, metrics, alternative in [
        ("coloss", COLOSS_METRICS, "greater"),
        ("compensation", COMPENSATION_METRICS, "two-sided"),
    ]:
        for metric in metrics:
            sub = differences[(differences["metric"] == metric) & (differences["difference"].notna())]
            diffs = pd.to_numeric(sub["difference"], errors="coerce").to_numpy(float)
            diffs = diffs[np.isfinite(diffs)]
            ci_low, ci_high = bootstrap_mean_ci(diffs, n_resamples=n_bootstrap, seed=seed + len(rows) * 17)
            rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "alternative": alternative,
                    "n_events": int(diffs.size),
                    "mean_difference": float(np.mean(diffs)) if diffs.size else np.nan,
                    "median_difference": float(np.median(diffs)) if diffs.size else np.nan,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_raw": sign_flip_test(diffs, alternative=alternative, n_resamples=n_signflip, seed=seed + len(rows) * 31),
                    "bootstrap_resamples": n_bootstrap,
                    "signflip_resamples": n_signflip,
                }
            )
    out = pd.DataFrame(rows)
    out["p_fdr"] = np.nan
    for family, idx in out.groupby("family").groups.items():
        out.loc[idx, "p_fdr"] = fdr_bh(out.loc[idx, "p_raw"])
    out["status"] = np.where(
        out["n_events"] < 10,
        "insufficient_samples",
        np.where(out["p_fdr"] < 0.05, "supported", "not_supported"),
    )
    return out


def load_cdr_major_mask(partition_dir: Path, pan_event_id: str) -> np.ndarray:
    path = partition_dir / f"{pan_event_id}.nc"
    if not path.exists():
        raise FileNotFoundError(f"Missing class-partition NetCDF for {pan_event_id}: {path}")
    with xr.open_dataset(path) as ds:
        return np.asarray(ds["cdr_local_class_code"].values, dtype=np.int8) == 3


def event_domains(
    pan_event_id: str,
    event_date: pd.Timestamp,
    partition_dir: Path,
    membership: pd.DataFrame,
    major_events: pd.DataFrame,
    valid_ocean: np.ndarray,
    surface_codes: np.ndarray,
) -> dict[str, object]:
    major_mask = load_cdr_major_mask(partition_dir, pan_event_id)
    if major_mask.shape != valid_ocean.shape:
        raise ValueError(f"Shape mismatch for {pan_event_id}: major_mask={major_mask.shape} valid_ocean={valid_ocean.shape}")
    source_codes, source_ids = source_major_region_codes(event_date, membership, major_events)
    named_codes = set(range(1, 19))
    other_codes = sorted(named_codes - set(source_codes))
    residual_domain = valid_ocean & ~major_mask
    other_region_domain = valid_ocean & np.isin(surface_codes, other_codes)
    return {
        "major_mask": major_mask,
        "residual_domain": residual_domain,
        "source_major_region_codes": source_codes,
        "source_major_event_ids": source_ids,
        "other_region_codes": other_codes,
        "other_region_domain": other_region_domain,
    }


def _finite_median(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else np.nan


def _background_percentile(control_values: np.ndarray, event_value: float) -> float:
    control_values = np.asarray(control_values, dtype=float)
    control_values = control_values[np.isfinite(control_values)]
    if control_values.size == 0 or not np.isfinite(event_value):
        return np.nan
    return float(np.mean(control_values <= event_value))


def compute_control_outputs(
    pan_events: pd.DataFrame,
    control_summary: pd.DataFrame,
    control_anchors: pd.DataFrame,
    membership: pd.DataFrame,
    major_events: pd.DataFrame,
    *,
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    area_km2, _, _ = load_cell_area_km2()
    surface = load_surface_mask()
    valid_ocean = valid_ocean_mask()
    if area_km2.shape != surface.shape or surface.shape != valid_ocean.shape:
        raise ValueError(f"Stage 3 grid shape mismatch: area={area_km2.shape} surface={surface.shape} ocean={valid_ocean.shape}")
    partition_dir = stage3_processed_root(root) / "panarctic_class_partition_fields"
    metrics_rows: list[dict] = []
    regional_rows: list[dict] = []
    difference_rows: list[dict] = []
    sync_rows: list[dict] = []
    controls_by_event = {k: v.copy() for k, v in control_anchors.groupby("pan_event_id", sort=False)} if not control_anchors.empty else {}
    summary_by_event = control_summary.set_index("pan_event_id").to_dict("index") if not control_summary.empty else {}

    for pan in pan_events.itertuples(index=False):
        pan_id = str(pan.pan_event_id)
        event_date = pd.Timestamp(pan.date).normalize()
        domains = event_domains(pan_id, event_date, partition_dir, membership, major_events, valid_ocean, surface)
        event_start = event_date - pd.Timedelta(days=5)
        event_metrics, event_region_loss = window_metrics(
            event_start,
            event_date,
            domains["residual_domain"],
            domains["other_region_domain"],
            surface,
            area_km2,
            valid_ocean,
        )
        source_codes = ";".join(str(v) for v in domains["source_major_region_codes"])
        source_ids = ";".join(str(v) for v in domains["source_major_event_ids"])
        base = {
            "pan_event_id": pan_id,
            "event_date": event_date.strftime("%Y-%m-%d"),
            "source_major_region_codes": source_codes,
            "source_major_event_ids": source_ids,
            "window_role": "event",
            "selected_rank": 0,
            "anchor_date": event_date.strftime("%Y-%m-%d"),
            "window_start": event_start.strftime("%Y-%m-%d"),
            "window_end": event_date.strftime("%Y-%m-%d"),
        }
        metrics_rows.append({**base, **event_metrics})
        control_metric_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
        control_region_values: dict[int, list[float]] = {code: [] for code in range(1, 19)}
        controls = controls_by_event.get(pan_id, pd.DataFrame())
        for ctrl in controls.itertuples(index=False):
            start = pd.Timestamp(ctrl.control_window_start).normalize()
            end = pd.Timestamp(ctrl.control_window_end).normalize()
            ctrl_metrics, ctrl_region_loss = window_metrics(
                start,
                end,
                domains["residual_domain"],
                domains["other_region_domain"],
                surface,
                area_km2,
                valid_ocean,
            )
            row = {
                "pan_event_id": pan_id,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "source_major_region_codes": source_codes,
                "source_major_event_ids": source_ids,
                "window_role": "control",
                "selected_rank": int(ctrl.selected_rank),
                "anchor_date": pd.Timestamp(ctrl.control_anchor_date).strftime("%Y-%m-%d"),
                "window_start": start.strftime("%Y-%m-%d"),
                "window_end": end.strftime("%Y-%m-%d"),
            }
            metrics_rows.append({**row, **ctrl_metrics})
            for metric in METRICS:
                control_metric_values[metric].append(ctrl_metrics[metric])
            for code in range(1, 19):
                control_region_values[code].append(ctrl_region_loss[code])

        for metric in METRICS:
            values = np.asarray(control_metric_values[metric], dtype=float)
            finite = values[np.isfinite(values)]
            event_value = float(event_metrics[metric])
            if np.isfinite(event_value) and finite.size >= CONTROL_LIMITED_MIN_N:
                control_median = float(np.median(finite))
                diff = event_value - control_median
                status = "ok"
            else:
                control_median = np.nan
                diff = np.nan
                status = "insufficient_controls"
            difference_rows.append(
                {
                    "pan_event_id": pan_id,
                    "event_date": event_date.strftime("%Y-%m-%d"),
                    "metric": metric,
                    "event_value": event_value,
                    "control_median": control_median,
                    "difference": diff,
                    "n_valid_controls": int(finite.size),
                    "matching_status": summary_by_event.get(pan_id, {}).get("matching_status", ""),
                    "status": status,
                }
            )

        regional_percentiles = []
        regions_above = 0
        for code in range(1, 19):
            controls_for_region = np.asarray(control_region_values[code], dtype=float)
            finite = controls_for_region[np.isfinite(controls_for_region)]
            event_loss = float(event_region_loss[code])
            if finite.size >= CONTROL_LIMITED_MIN_N and np.isfinite(event_loss):
                median_loss = float(np.median(finite))
                anomaly = event_loss - median_loss
                pct = _background_percentile(finite, event_loss)
                status = "ok"
                if event_loss > median_loss:
                    regions_above += 1
                regional_percentiles.append(pct)
            else:
                median_loss = np.nan
                anomaly = np.nan
                pct = np.nan
                status = "insufficient_controls"
            regional_rows.append(
                {
                    "pan_event_id": pan_id,
                    "event_date": event_date.strftime("%Y-%m-%d"),
                    "region_code": code,
                    "region_name": NSIDC_0780_REGION_IDS and {v: k for k, v in NSIDC_0780_REGION_IDS.items()}[code],
                    "event_sic_loss_km2eq": event_loss,
                    "control_median_sic_loss_km2eq": median_loss,
                    "loss_anomaly_km2eq": anomaly,
                    "background_percentile": pct,
                    "n_valid_controls": int(finite.size),
                    "status": status,
                }
            )
        finite_percentiles = np.asarray(regional_percentiles, dtype=float)
        finite_percentiles = finite_percentiles[np.isfinite(finite_percentiles)]
        sync_rows.append(
            {
                "pan_event_id": pan_id,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "n_regions_above_control_median_sic_loss": int(regions_above),
                "fraction_regions_above_control_median_sic_loss": float(regions_above / 18.0),
                "median_region_background_percentile": float(np.median(finite_percentiles)) if finite_percentiles.size else np.nan,
                "n_regions_evaluable": int(finite_percentiles.size),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    differences_df = pd.DataFrame(difference_rows)
    regional_df = pd.DataFrame(regional_rows)
    sync_df = pd.DataFrame(sync_rows)
    return metrics_df, differences_df, regional_df, sync_df


def validate_control_gates(
    pan_events: pd.DataFrame,
    control_summary: pd.DataFrame,
    control_anchors: pd.DataFrame,
    regional_matrix: pd.DataFrame,
    differences: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    windows = primary_windows(pan_events)
    rows.append(
        {
            "gate": "primary_pan_event_count",
            "status": "PASS" if len(pan_events) == 99 and pan_events["pan_event_id"].nunique() == 99 else "FAIL",
            "detail": f"rows={len(pan_events)} unique={pan_events['pan_event_id'].nunique()}",
        }
    )
    overlap_fail = 0
    same_year_fail = 0
    jja_fail = 0
    readable_fail = 0
    for ctrl in control_anchors.itertuples(index=False):
        anchor = pd.Timestamp(ctrl.control_anchor_date).normalize()
        event_date = pd.Timestamp(ctrl.event_date).normalize()
        if control_window_overlaps_primary(anchor, windows):
            overlap_fail += 1
        if anchor.year != event_date.year:
            same_year_fail += 1
        if anchor.month not in (6, 7, 8):
            jja_fail += 1
        if not readable_track_s_window(anchor):
            readable_fail += 1
    rows.extend(
        [
            {"gate": "control_overlap_exclusion", "status": "PASS" if overlap_fail == 0 else "FAIL", "detail": str(overlap_fail)},
            {"gate": "control_same_year", "status": "PASS" if same_year_fail == 0 else "FAIL", "detail": str(same_year_fail)},
            {"gate": "control_jja_anchor", "status": "PASS" if jja_fail == 0 else "FAIL", "detail": str(jja_fail)},
            {"gate": "control_readable_endpoints", "status": "PASS" if readable_fail == 0 else "FAIL", "detail": str(readable_fail)},
            {
                "gate": "regional_matrix_99x18",
                "status": "PASS" if len(regional_matrix) == len(pan_events) * 18 else "FAIL",
                "detail": f"rows={len(regional_matrix)} expected={len(pan_events) * 18}",
            },
            {
                "gate": "one_difference_per_event_metric",
                "status": "PASS" if len(differences) == len(pan_events) * len(METRICS) else "FAIL",
                "detail": f"rows={len(differences)} expected={len(pan_events) * len(METRICS)}",
            },
            {
                "gate": "matching_summary_one_row_per_event",
                "status": "PASS" if len(control_summary) == len(pan_events) and control_summary["pan_event_id"].nunique() == len(pan_events) else "FAIL",
                "detail": f"rows={len(control_summary)} unique={control_summary['pan_event_id'].nunique() if not control_summary.empty else 0}",
            },
        ]
    )
    return pd.DataFrame(rows)
