"""Batch 4B reverse analysis for canonical major_severe local events."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vrile.regions import NSIDC_0780_REGION_IDS
from vrile.stage3.budget import aligned_patch_rows
from vrile.stage3.controls import (
    CONTROL_LIMITED_MIN_N,
    CONTROL_TARGET_N,
    bootstrap_mean_ci,
    classify_year_block_confirmation,
    fdr_bh,
    intervals_overlap,
    primary_windows,
    readable_track_s_window,
    sign_flip_test,
    validate_year_block_confirmation_gates,
    window_metrics,
)
from vrile.stage3.footprints import load_patch_cells
from vrile.stage3.grid import load_cell_area_km2, load_surface_mask, valid_ocean_mask

CONTROL_TIERS = (30, 45, 60)
INFERENCE_SPECS = {
    "local_strength": {"local_sic_loss_km2eq": "greater"},
    "coloss": {
        "residual_sic_loss_km2eq": "greater",
        "other_region_sic_loss_km2eq": "greater",
    },
    "compensation": {
        "global_compensation_ratio": "less",
        "residual_compensation_ratio": "less",
        "other_region_compensation_ratio": "less",
    },
}
INFERENCE_METRICS = [metric for specs in INFERENCE_SPECS.values() for metric in specs]


def _prep_membership(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["unique_local_event_id"] = out["unique_local_event_id"].astype(str)
    out["object_id"] = out["object_id"].astype(str)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["start_date"] = pd.to_datetime(out["start_date"]).dt.normalize()
    return out


def _season_day(date: pd.Timestamp) -> int:
    reference = pd.Timestamp(year=2001, month=int(date.month), day=int(date.day))
    return int(reference.dayofyear)


def classify_major_events(
    major_events: pd.DataFrame,
    relation: pd.DataFrame,
    pan_events: pd.DataFrame,
    daily_sie: pd.DataFrame,
) -> pd.DataFrame:
    major = major_events.copy()
    major["unique_local_event_id"] = major["unique_local_event_id"].astype(str)
    major["event_date"] = pd.to_datetime(major["event_date"]).dt.normalize()
    if major["unique_local_event_id"].duplicated().any():
        raise ValueError("major_severe ULE universe contains duplicate unique_local_event_id")
    linked = relation[relation["event_level"].eq("major_severe")].copy()
    linked["unique_local_event_id"] = linked["unique_local_event_id"].astype(str)
    linked_counts = linked.groupby("unique_local_event_id")["pan_event_id"].nunique().to_dict()
    unknown = sorted(set(linked_counts) - set(major["unique_local_event_id"]))
    if unknown:
        raise ValueError(f"Frozen Track S relation contains major_severe IDs outside canonical universe: {unknown[:5]}")

    sie = daily_sie.copy()
    sie["date"] = pd.to_datetime(sie["date"]).dt.normalize()
    sie_lookup = sie.set_index("date")["extent"]
    pan_windows = primary_windows(pan_events)
    rows: list[dict] = []
    for event in major.itertuples(index=False):
        uid = str(event.unique_local_event_id)
        date = pd.Timestamp(event.event_date).normalize()
        start = date - pd.Timedelta(days=5)
        reasons: list[str] = []
        if date.month not in (6, 7, 8):
            reasons.append("non_jja_event_date")
        if not readable_track_s_window(date):
            reasons.append("cdr_endpoints_unreadable")
        initial_sie = float(sie_lookup.get(start, np.nan))
        if not np.isfinite(initial_sie):
            reasons.append("initial_sie_unavailable")
        pan_linked = uid in linked_counts
        control_window_excluded = bool(
            not pan_linked and any(intervals_overlap((start, date), window) for window in pan_windows)
        )
        control_eligible = bool(not pan_linked and not reasons and not control_window_excluded)
        rows.append(
            {
                "unique_local_event_id": uid,
                "event_date": date.strftime("%Y-%m-%d"),
                "window_start": start.strftime("%Y-%m-%d"),
                "window_end": date.strftime("%Y-%m-%d"),
                "dominant_region": str(event.dominant_region),
                "pan_linked": pan_linked,
                "analysis_group": "case" if pan_linked else "control",
                "n_linked_pan_events": int(linked_counts.get(uid, 0)),
                "initial_sie": initial_sie,
                "season_day": _season_day(date),
                "event_eligible": not reasons,
                "control_window_intersects_primary_pan_window": control_window_excluded,
                "control_eligible": control_eligible,
                "eligibility_status": "eligible" if not reasons else ";".join(reasons),
            }
        )
    out = pd.DataFrame(rows)
    if len(out) != len(major) or out["unique_local_event_id"].nunique() != len(major):
        raise ValueError("Reverse group definition is not unique/exhaustive over major_severe ULE universe")
    return out


def select_reverse_controls(classification: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    controls = classification[classification["control_eligible"]].copy()
    cases = classification[(classification["pan_linked"]) & (classification["event_eligible"])].copy()
    anchor_rows: list[dict] = []
    summary_rows: list[dict] = []
    quality_rows: list[dict] = []
    for case in cases.itertuples(index=False):
        case_date = pd.Timestamp(case.event_date)
        candidates = controls[controls["dominant_region"].eq(case.dominant_region)].copy()
        candidates["season_day_distance"] = (candidates["season_day"] - int(case.season_day)).abs()
        candidates["abs_initial_sie_difference"] = (candidates["initial_sie"] - float(case.initial_sie)).abs()
        candidates["abs_year_difference"] = (pd.to_datetime(candidates["event_date"]).dt.year - case_date.year).abs()
        tier_counts = {tier: int((candidates["season_day_distance"] <= tier).sum()) for tier in CONTROL_TIERS}
        selected_tier: int | str = "all"
        pool = candidates
        for tier in CONTROL_TIERS:
            tier_pool = candidates[candidates["season_day_distance"] <= tier].copy()
            if len(tier_pool) >= CONTROL_TARGET_N:
                selected_tier = tier
                pool = tier_pool
                break
        selected = pool.sort_values(
            [
                "abs_initial_sie_difference",
                "abs_year_difference",
                "season_day_distance",
                "event_date",
                "unique_local_event_id",
            ]
        ).head(CONTROL_TARGET_N)
        n_selected = int(len(selected))
        status = (
            "matched_controls"
            if n_selected >= CONTROL_TARGET_N
            else "limited_background_controls"
            if n_selected >= CONTROL_LIMITED_MIN_N
            else "insufficient_background_controls"
        )
        summary_rows.append(
            {
                "unique_local_event_id": case.unique_local_event_id,
                "event_date": case.event_date,
                "dominant_region": case.dominant_region,
                "event_initial_sie": float(case.initial_sie),
                "candidate_count_tier_30": tier_counts[30],
                "candidate_count_tier_45": tier_counts[45],
                "candidate_count_tier_60": tier_counts[60],
                "candidate_count_all": int(len(candidates)),
                "selected_tier": str(selected_tier),
                "n_selected_controls": n_selected,
                "matching_status": status,
            }
        )
        for rank, control in enumerate(selected.itertuples(index=False), start=1):
            anchor_rows.append(
                {
                    "unique_local_event_id": case.unique_local_event_id,
                    "event_date": case.event_date,
                    "dominant_region": case.dominant_region,
                    "event_initial_sie": float(case.initial_sie),
                    "matching_status": status,
                    "selected_tier": str(selected_tier),
                    "selected_rank": rank,
                    "control_unique_local_event_id": control.unique_local_event_id,
                    "control_event_date": control.event_date,
                    "control_initial_sie": float(control.initial_sie),
                    "abs_initial_sie_difference": float(control.abs_initial_sie_difference),
                    "abs_year_difference": int(control.abs_year_difference),
                    "season_day_distance": int(control.season_day_distance),
                }
            )
        absdiff = pd.to_numeric(selected.get("abs_initial_sie_difference", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        quality_rows.append(
            {
                "unique_local_event_id": case.unique_local_event_id,
                "dominant_region": case.dominant_region,
                "matching_status": status,
                "selected_tier": str(selected_tier),
                "n_selected_controls": n_selected,
                "median_abs_initial_sie_difference": float(np.median(absdiff)) if absdiff.size else np.nan,
                "p90_abs_initial_sie_difference": float(np.percentile(absdiff, 90)) if absdiff.size else np.nan,
                "max_abs_initial_sie_difference": float(np.max(absdiff)) if absdiff.size else np.nan,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(anchor_rows), pd.DataFrame(quality_rows)


def compute_reverse_window_metrics(
    classification: pd.DataFrame,
    anchors: pd.DataFrame,
    membership: pd.DataFrame,
    patch_index: pd.DataFrame,
    patch_base_dir: Path,
) -> pd.DataFrame:
    membership = _prep_membership(membership)
    required_ids = set(classification.loc[classification["pan_linked"] & classification["event_eligible"], "unique_local_event_id"].astype(str))
    required_ids.update(anchors["control_unique_local_event_id"].astype(str))
    class_lookup = classification.set_index("unique_local_event_id")
    aligned_by_uid: dict[str, pd.DataFrame] = {}
    all_object_ids: set[str] = set()
    for uid in sorted(required_ids):
        event_date = pd.Timestamp(class_lookup.loc[uid, "event_date"])
        aligned = aligned_patch_rows(membership, event_date, track="cdr")
        aligned = aligned[aligned["unique_local_event_id"].eq(uid)].copy()
        if aligned.empty:
            raise ValueError(f"No Track S aligned member patch for canonical ULE {uid}")
        aligned_by_uid[uid] = aligned
        all_object_ids.update(aligned["object_id"].astype(str))
    cells = load_patch_cells(patch_index, patch_base_dir, all_object_ids)
    missing = sorted(all_object_ids - set(cells))
    if missing:
        raise ValueError(f"Missing sparse cells for reverse analysis patches: {missing[:5]}")

    area_km2, _, _ = load_cell_area_km2()
    surface = load_surface_mask()
    valid_ocean = valid_ocean_mask()
    shape = valid_ocean.shape
    rows: list[dict] = []
    for uid in sorted(required_ids):
        source = class_lookup.loc[uid]
        region = str(source["dominant_region"])
        if region not in NSIDC_0780_REGION_IDS:
            raise ValueError(f"Unknown dominant_region for reverse analysis: {uid}={region!r}")
        region_code = int(NSIDC_0780_REGION_IDS[region])
        local = np.zeros(shape, dtype=bool)
        for object_id in aligned_by_uid[uid]["object_id"].astype(str):
            y_idx, x_idx, _ = cells[object_id]
            local[y_idx, x_idx] = True
        local &= valid_ocean
        residual = valid_ocean & ~local
        other_region = valid_ocean & np.isin(surface, sorted(set(range(1, 19)) - {region_code}))
        end = pd.Timestamp(source["event_date"])
        start = end - pd.Timedelta(days=5)
        metrics, _ = window_metrics(
            start,
            end,
            residual,
            other_region,
            surface,
            area_km2,
            valid_ocean,
            local_domain=local,
        )
        rows.append(
            {
                "unique_local_event_id": uid,
                "analysis_group": str(source["analysis_group"]),
                "pan_linked": bool(source["pan_linked"]),
                "event_date": end.strftime("%Y-%m-%d"),
                "window_start": start.strftime("%Y-%m-%d"),
                "window_end": end.strftime("%Y-%m-%d"),
                "dominant_region": region,
                "dominant_region_code": region_code,
                "aligned_patch_count": int(len(aligned_by_uid[uid])),
                "local_footprint_cells": int(local.sum()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def reverse_differences(
    metrics: pd.DataFrame,
    anchors: pd.DataFrame,
    matching_summary: pd.DataFrame,
    *,
    control_count: int = CONTROL_TARGET_N,
    common_case_ids: set[str] | None = None,
) -> pd.DataFrame:
    lookup = metrics.set_index("unique_local_event_id")
    rows: list[dict] = []
    for summary in matching_summary.itertuples(index=False):
        uid = str(summary.unique_local_event_id)
        if common_case_ids is not None and uid not in common_case_ids:
            continue
        selected = anchors[(anchors["unique_local_event_id"].eq(uid)) & (anchors["selected_rank"] <= control_count)]
        control_ids = selected["control_unique_local_event_id"].astype(str).tolist()
        for metric in INFERENCE_METRICS:
            case_value = float(lookup.loc[uid, metric]) if uid in lookup.index else np.nan
            values = pd.to_numeric(lookup.loc[control_ids, metric], errors="coerce").to_numpy(float) if control_ids else np.asarray([], dtype=float)
            finite = values[np.isfinite(values)]
            valid = np.isfinite(case_value) and finite.size >= CONTROL_LIMITED_MIN_N
            median = float(np.median(finite)) if valid else np.nan
            rows.append(
                {
                    "unique_local_event_id": uid,
                    "event_date": summary.event_date,
                    "dominant_region": summary.dominant_region,
                    "metric": metric,
                    "control_count": int(control_count),
                    "case_value": case_value,
                    "control_median": median,
                    "difference": case_value - median if valid else np.nan,
                    "n_valid_controls": int(finite.size),
                    "matching_status": summary.matching_status,
                    "status": "ok" if valid else "insufficient_controls",
                }
            )
    return pd.DataFrame(rows)


def infer_reverse_tests(
    differences: pd.DataFrame,
    *,
    seed: int,
    n_bootstrap: int,
    n_signflip: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for family, specs in INFERENCE_SPECS.items():
        family_rows = []
        for metric, alternative in specs.items():
            values = pd.to_numeric(
                differences.loc[differences["metric"].eq(metric), "difference"], errors="coerce"
            ).dropna().to_numpy(float)
            index = len(rows) + len(family_rows)
            ci_low, ci_high = bootstrap_mean_ci(values, n_resamples=n_bootstrap, seed=seed + index * 17)
            family_rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "alternative": alternative,
                    "n_cases": int(values.size),
                    "mean_difference": float(np.mean(values)) if values.size else np.nan,
                    "median_difference": float(np.median(values)) if values.size else np.nan,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_raw": sign_flip_test(values, alternative=alternative, n_resamples=n_signflip, seed=seed + index * 31),
                    "bootstrap_resamples": n_bootstrap,
                    "signflip_resamples": n_signflip,
                }
            )
        frame = pd.DataFrame(family_rows)
        frame["p_fdr"] = fdr_bh(frame["p_raw"])
        frame["status"] = np.where(
            frame["n_cases"] < 10,
            "insufficient_samples",
            np.where(frame["p_fdr"] < 0.05, "supported", "not_supported"),
        )
        rows.extend(frame.to_dict("records"))
    return pd.DataFrame(rows)


def control_count_robustness(
    metrics: pd.DataFrame,
    anchors: pd.DataFrame,
    matching_summary: pd.DataFrame,
    primary_tests: pd.DataFrame,
    *,
    seed: int,
    n_bootstrap: int,
    n_signflip: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = set(
        matching_summary.loc[matching_summary["n_selected_controls"] >= CONTROL_TARGET_N, "unique_local_event_id"].astype(str)
    )
    diffs_all: list[pd.DataFrame] = []
    tests_all: list[pd.DataFrame] = []
    primary = primary_tests.set_index("metric")
    for count in (10, 15, 20):
        diffs = reverse_differences(metrics, anchors, matching_summary, control_count=count, common_case_ids=common)
        tests = infer_reverse_tests(diffs, seed=seed, n_bootstrap=n_bootstrap, n_signflip=n_signflip)
        tests["control_count"] = count
        tests["analysis_role"] = "primary_common_case_regression" if count == 20 else "control_count_sensitivity"
        tests["direction_consistent_with_primary"] = tests.apply(
            lambda row: bool(np.sign(row["mean_difference"]) == np.sign(primary.loc[row["metric"], "mean_difference"])), axis=1
        )
        tests["status_consistent_with_primary"] = tests.apply(
            lambda row: bool(str(row["status"]) == str(primary.loc[row["metric"], "status"])), axis=1
        )
        diffs_all.append(diffs)
        tests_all.append(tests)
    return pd.concat(diffs_all, ignore_index=True), pd.concat(tests_all, ignore_index=True)


def year_block_confirmation(
    differences: pd.DataFrame,
    primary_tests: pd.DataFrame,
    *,
    seed: int,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    supported = primary_tests.loc[primary_tests["status"].eq("supported"), "metric"].astype(str).tolist()
    for i, metric in enumerate(supported):
        sub = differences[differences["metric"].eq(metric)].copy()
        sub["year"] = pd.to_datetime(sub["event_date"]).dt.year
        sub["difference"] = pd.to_numeric(sub["difference"], errors="coerce")
        sub = sub[sub["difference"].notna()]
        years = sorted(sub["year"].unique())
        blocks = {year: sub.loc[sub["year"].eq(year), "difference"].to_numpy(float) for year in years}
        rng = np.random.default_rng(seed + i * 101)
        means = []
        for _ in range(n_bootstrap):
            sampled = rng.choice(np.asarray(years), size=len(years), replace=True)
            values = np.concatenate([blocks[int(year)] for year in sampled])
            means.append(float(np.mean(values)))
        valid_means = np.asarray(means, dtype=float)
        valid_means = valid_means[np.isfinite(valid_means)]
        ci_low, ci_high = np.percentile(valid_means, [2.5, 97.5]) if valid_means.size else (np.nan, np.nan)
        primary_mean = float(sub["difference"].mean())
        semantic = classify_year_block_confirmation(
            primary_status="supported",
            primary_mean=primary_mean,
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            n_years=len(years),
            n_valid_bootstrap=int(valid_means.size),
        )
        rows.append(
            {
                "metric": metric,
                "n_cases": int(len(sub)),
                "n_years": int(len(years)),
                "primary_mean_difference": primary_mean,
                "block_bootstrap_ci_low": float(ci_low),
                "block_bootstrap_ci_high": float(ci_high),
                "primary_status": "supported",
                "n_bootstrap_requested": int(n_bootstrap),
                "n_bootstrap_valid": int(valid_means.size),
                **semantic,
                "analysis_role": "confirmation_only",
            }
        )
    return pd.DataFrame(rows)


def validate_reverse_gates(
    classification: pd.DataFrame,
    relation: pd.DataFrame,
    anchors: pd.DataFrame,
    differences: pd.DataFrame,
    tests: pd.DataFrame,
    year_block: pd.DataFrame,
    batch3_regression: pd.DataFrame,
    batch4a_regression: pd.DataFrame,
) -> pd.DataFrame:
    linked_relation = set(
        relation.loc[relation["event_level"].eq("major_severe"), "unique_local_event_id"].astype(str)
    )
    linked_class = set(classification.loc[classification["pan_linked"], "unique_local_event_id"].astype(str))
    control_lookup = classification.set_index("unique_local_event_id")
    selected_controls = anchors["control_unique_local_event_id"].astype(str)
    same_region = anchors.apply(
        lambda row: str(row["dominant_region"]) == str(control_lookup.loc[str(row["control_unique_local_event_id"]), "dominant_region"]), axis=1
    )
    nonlinked = selected_controls.map(lambda uid: not bool(control_lookup.loc[uid, "pan_linked"]))
    excluded = selected_controls.map(lambda uid: not bool(control_lookup.loc[uid, "control_window_intersects_primary_pan_window"]))
    expected_families = {family: set(specs) for family, specs in INFERENCE_SPECS.items()}
    actual_families = {family: set(group["metric"].astype(str)) for family, group in tests.groupby("family")}
    rows = [
        ("major_universe_unique_exhaustive", len(classification) == classification["unique_local_event_id"].nunique(), f"rows={len(classification)}"),
        ("pan_linked_traceable_to_frozen_relation", linked_relation == linked_class, f"linked={len(linked_class)}"),
        ("selected_controls_non_pan_linked", bool(nonlinked.all()), f"controls={len(anchors)}"),
        ("selected_control_windows_exclude_primary", bool(excluded.all()), f"controls={len(anchors)}"),
        ("same_region_matching_exact", bool(same_region.all()), f"controls={len(anchors)}"),
        ("one_difference_per_case_metric", not differences.duplicated(["unique_local_event_id", "metric"]).any(), f"rows={len(differences)}"),
        ("fdr_families_exact", actual_families == expected_families, str(actual_families)),
        ("batch3_frozen_hashes_unchanged", bool(batch3_regression["status"].eq("PASS").all()), f"pass={batch3_regression['status'].eq('PASS').sum()}/{len(batch3_regression)}"),
        ("batch4a_primary_hashes_unchanged", bool(batch4a_regression["status"].eq("PASS").all()), f"pass={batch4a_regression['status'].eq('PASS').sum()}/{len(batch4a_regression)}"),
    ]
    base = pd.DataFrame(
        [{"gate": gate, "status": "PASS" if passed else "FAIL", "detail": detail} for gate, passed, detail in rows]
    )
    year_block_gates = validate_year_block_confirmation_gates(
        year_block,
        tests,
        count_column="n_cases",
    )
    confirmed = int(year_block["year_block_confirmed"].astype(bool).sum())
    downgraded = int(year_block["interpretation_downgraded"].astype(bool).sum())
    insufficient = int(
        year_block["final_interpretation_status"]
        .astype(str)
        .eq("supported_but_year_block_confirmation_insufficient")
        .sum()
    )
    result_gate = pd.DataFrame(
        [
            {
                "gate": "year_block_confirmation_result",
                "status": "PASS",
                "detail": (
                    f"confirmed={confirmed}/{len(year_block)} "
                    f"downgraded={downgraded}/{len(year_block)} "
                    f"insufficient={insufficient}/{len(year_block)}"
                ),
            }
        ]
    )
    return pd.concat([base, year_block_gates, result_gate], ignore_index=True)
