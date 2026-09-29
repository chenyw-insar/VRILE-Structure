"""Batch 3 dual-track local class partitions and contribution budgets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from vrile.stage3.footprints import load_patch_cells
from vrile.stage3.grid import REGION_BY_CODE, load_cell_area_km2, load_surface_mask

CLASS_CODEBOOK = {
    0: "residual",
    1: "broad_nonsevere",
    2: "severe_nonmajor",
    3: "major_severe",
}
LEVEL_TO_CODE = {"broad": 1, "severe": 2, "major_severe": 3}


def hierarchy_lookup(unique: pd.DataFrame, severe: pd.DataFrame, major: pd.DataFrame) -> dict[str, str]:
    severe_ids = set(severe["unique_local_event_id"].astype(str)) if not severe.empty else set()
    major_ids = set(major["unique_local_event_id"].astype(str)) if not major.empty else set()
    out = {}
    for uid in unique["unique_local_event_id"].astype(str):
        if uid in major_ids:
            out[uid] = "major_severe"
        elif uid in severe_ids:
            out[uid] = "severe"
        else:
            out[uid] = "broad"
    return out


def _prep_membership(membership: pd.DataFrame) -> pd.DataFrame:
    out = membership.copy()
    out["unique_local_event_id"] = out["unique_local_event_id"].astype(str)
    out["object_id"] = out["object_id"].astype(str)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["start_date"] = pd.to_datetime(out["start_date"]).dt.normalize()
    return out


def aligned_patch_rows(membership: pd.DataFrame, event_date: pd.Timestamp, track: str) -> pd.DataFrame:
    event_date = pd.Timestamp(event_date).normalize()
    if track == "detector":
        start = event_date - pd.Timedelta(days=2)
        end = event_date + pd.Timedelta(days=1)
    elif track == "cdr":
        start = event_date - pd.Timedelta(days=5)
        end = event_date
    else:
        raise ValueError(f"Unknown track: {track}")
    return membership[(membership["start_date"] <= end) & (membership["date"] >= start) & (membership["date"] <= end)].copy()


def build_class_code(
    member_rows: pd.DataFrame,
    patch_index: pd.DataFrame,
    local_cells_dir: Path,
    shape: tuple[int, int],
    levels: dict[str, str],
) -> np.ndarray:
    code = np.zeros(shape, dtype=np.int8)
    cells = load_patch_cells(patch_index, local_cells_dir, set(member_rows["object_id"].astype(str)))
    missing = sorted(set(member_rows["object_id"].astype(str)) - set(cells))
    if missing:
        raise ValueError(f"Missing sparse patch cells for {len(missing)} aligned object_id values; first={missing[:5]}")
    for uid, group in member_rows.groupby("unique_local_event_id", sort=False):
        level_code = LEVEL_TO_CODE.get(levels.get(str(uid), "broad"), 1)
        for object_id in group["object_id"].astype(str):
            y_idx, x_idx, _ = cells[object_id]
            code[y_idx, x_idx] = np.maximum(code[y_idx, x_idx], level_code)
    return code


def _raw_priority_masks(
    member_rows: pd.DataFrame,
    patch_index: pd.DataFrame,
    local_cells_dir: Path,
    shape: tuple[int, int],
    levels: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    broad = np.zeros(shape, dtype=bool)
    severe = np.zeros(shape, dtype=bool)
    major = np.zeros(shape, dtype=bool)
    cells = load_patch_cells(patch_index, local_cells_dir, set(member_rows["object_id"].astype(str)))
    missing = sorted(set(member_rows["object_id"].astype(str)) - set(cells))
    if missing:
        raise ValueError(f"Missing sparse patch cells for raw class audit; first={missing[:5]}")
    for uid, group in member_rows.groupby("unique_local_event_id", sort=False):
        level = levels.get(str(uid), "broad")
        for object_id in group["object_id"].astype(str):
            y_idx, x_idx, _ = cells[object_id]
            if level == "broad":
                broad[y_idx, x_idx] = True
            elif level == "severe":
                severe[y_idx, x_idx] = True
            elif level == "major_severe":
                major[y_idx, x_idx] = True
            else:
                raise ValueError(f"Unknown local event level {level!r} for {uid}")
    expected = np.zeros(shape, dtype=np.int8)
    expected[broad] = 1
    expected[severe] = 2
    expected[major] = 3
    return broad, severe, major, expected


def build_class_partitions(
    pan_events: pd.DataFrame,
    membership: pd.DataFrame,
    patch_index: pd.DataFrame,
    unique: pd.DataFrame,
    severe: pd.DataFrame,
    major: pd.DataFrame,
    local_cells_dir: Path,
    detector_dir: Path,
    partition_dir: Path,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = _prep_membership(membership)
    levels = hierarchy_lookup(unique, severe, major)
    partition_dir.mkdir(parents=True, exist_ok=True)
    _, x, y = load_cell_area_km2()
    shape = (len(y), len(x))
    alignment_rows: list[dict] = []
    audit_rows: list[dict] = []

    for pan in pan_events.itertuples(index=False):
        pan_id = str(pan.pan_event_id)
        event_date = pd.Timestamp(pan.date).normalize()
        det_rows = aligned_patch_rows(membership, event_date, "detector")
        cdr_rows = aligned_patch_rows(membership, event_date, "cdr")
        det_ids = set(det_rows["unique_local_event_id"].astype(str))
        cdr_ids = set(cdr_rows["unique_local_event_id"].astype(str))
        alignment_rows.append(
            {
                "pan_event_id": pan_id,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "detector_window_start": (event_date - pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                "detector_window_end": (event_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "cdr_window_start": (event_date - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                "cdr_window_end": event_date.strftime("%Y-%m-%d"),
                "detector_aligned_patch_count": int(len(det_rows)),
                "detector_aligned_ule_count": int(len(det_ids)),
                "cdr_aligned_patch_count": int(len(cdr_rows)),
                "cdr_aligned_ule_count": int(len(cdr_ids)),
                "shared_ule_count": int(len(det_ids & cdr_ids)),
                "detector_only_ule_count": int(len(det_ids - cdr_ids)),
                "cdr_only_ule_count": int(len(cdr_ids - det_ids)),
            }
        )
        det_code = build_class_code(det_rows, patch_index, local_cells_dir, shape, levels)
        cdr_code = build_class_code(cdr_rows, patch_index, local_cells_dir, shape, levels)
        raw_by_track = {}
        for track, rows_for_track in (("detector", det_rows), ("cdr", cdr_rows)):
            raw_broad, raw_severe, raw_major, expected = _raw_priority_masks(rows_for_track, patch_index, local_cells_dir, shape, levels)
            raw_by_track[track] = (raw_broad, raw_severe, raw_major, expected)
        ds = xr.Dataset(
            {
                "detector_local_class_code": (("y", "x"), det_code),
                "cdr_local_class_code": (("y", "x"), cdr_code),
            },
            coords={"x": x, "y": y},
            attrs={
                "pan_event_id": pan_id,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "class_codebook": "; ".join(f"{k}={v}" for k, v in CLASS_CODEBOOK.items()),
                "detector_alignment_semantics": "patch.start_date <= T+1 and patch.date >= T-2 and patch.date <= T+1",
                "cdr_alignment_semantics": "patch.start_date <= T and patch.date >= T-5 and patch.date <= T",
                "hierarchy": "major_severe > severe > broad",
            },
        )
        out_path = partition_dir / f"{pan_id}.nc"
        ds.to_netcdf(out_path)
        with xr.open_dataset(out_path) as persisted:
            persisted_by_track = {
                "detector": np.asarray(persisted["detector_local_class_code"].values, dtype=np.int8),
                "cdr": np.asarray(persisted["cdr_local_class_code"].values, dtype=np.int8),
            }
        for track, arr in persisted_by_track.items():
            raw_broad, raw_severe, raw_major, expected = raw_by_track[track]
            values = set(int(v) for v in np.unique(arr))
            valid = values.issubset(CLASS_CODEBOOK)
            expected_match = bool(np.array_equal(arr, expected))
            major_severe_overlap = raw_major & raw_severe
            major_broad_overlap = raw_major & raw_broad
            severe_broad_overlap = raw_severe & raw_broad
            triple = raw_major & raw_severe & raw_broad
            cross = (raw_major.astype(int) + raw_severe.astype(int) + raw_broad.astype(int)) > 1
            raw_union = raw_major | raw_severe | raw_broad
            audit_rows.append(
                {
                    "pan_event_id": pan_id,
                    "track": track,
                    "valid_class_codes": bool(valid),
                    "exclusive_exhaustive_partition": bool(valid and arr.shape == shape),
                    "class_codes_present": ";".join(str(v) for v in sorted(values)),
                    "raw_major_cells": int(raw_major.sum()),
                    "raw_severe_cells": int(raw_severe.sum()),
                    "raw_broad_cells": int(raw_broad.sum()),
                    "major_severe_overlap_cells": int(major_severe_overlap.sum()),
                    "major_broad_overlap_cells": int(major_broad_overlap.sum()),
                    "severe_broad_overlap_cells": int(severe_broad_overlap.sum()),
                    "triple_overlap_cells": int(triple.sum()),
                    "raw_cross_class_overlap_cells": int(cross.sum()),
                    "raw_union_cells": int(raw_union.sum()),
                    "raw_cross_class_overlap_fraction": float(cross.sum() / raw_union.sum()) if raw_union.any() else 0.0,
                    "exclusive_major_severe_cells": int((expected == 3).sum()),
                    "exclusive_severe_nonmajor_cells": int((expected == 2).sum()),
                    "exclusive_broad_nonsevere_cells": int((expected == 1).sum()),
                    "residual_cells": int((arr == 0).sum()),
                    "broad_nonsevere_cells": int((arr == 1).sum()),
                    "severe_nonmajor_cells": int((arr == 2).sum()),
                    "major_severe_cells": int((arr == 3).sum()),
                    "expected_partition_exact_match": expected_match,
                    "status": "PASS" if valid and arr.shape == shape and expected_match else "FAIL",
                }
            )

    align = pd.DataFrame(alignment_rows)
    audit = pd.DataFrame(audit_rows)
    align.to_csv(out_dir / "track_alignment_audit.csv", index=False)
    audit.to_csv(out_dir / "class_partition_overlap_audit.csv", index=False)
    if not audit["status"].eq("PASS").all():
        raise ValueError("Class partition overlap audit contains FAIL rows")
    return align, audit


def _class_budget(mask_loss: np.ndarray, mask_gain: np.ndarray, class_code: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    rows: dict[str, float] = {}
    loss_vals = []
    gain_vals = []
    for code, name in CLASS_CODEBOOK.items():
        cls = class_code == code
        loss = float(np.nansum(np.where(cls & mask_loss, weight, 0.0)))
        gain = float(np.nansum(np.where(cls & mask_gain, weight, 0.0)))
        rows[f"{name}_gross_loss"] = loss
        rows[f"{name}_gross_gain"] = gain
        loss_vals.append(loss)
        gain_vals.append(gain)
    total_loss = float(np.nansum(np.where(mask_loss, weight, 0.0)))
    total_gain = float(np.nansum(np.where(mask_gain, weight, 0.0)))
    for code, name in CLASS_CODEBOOK.items():
        rows[f"{name}_gross_loss_fraction"] = rows[f"{name}_gross_loss"] / total_loss if total_loss > 0 else np.nan
        rows[f"{name}_gross_gain_fraction"] = rows[f"{name}_gross_gain"] / total_gain if total_gain > 0 else np.nan
    rows["gross_loss"] = total_loss
    rows["gross_gain"] = total_gain
    rows["class_loss_sum"] = float(sum(loss_vals))
    rows["class_gain_sum"] = float(sum(gain_vals))
    rows["partition_loss_closure_error"] = rows["class_loss_sum"] - total_loss
    rows["partition_gain_closure_error"] = rows["class_gain_sum"] - total_gain
    rows["partition_closure_status"] = (
        "PASS"
        if np.isclose(rows["partition_loss_closure_error"], 0.0, rtol=1e-12, atol=1e-6)
        and np.isclose(rows["partition_gain_closure_error"], 0.0, rtol=1e-12, atol=1e-6)
        else "FAIL"
    )
    rows["compensation_ratio"] = total_gain / total_loss if total_loss > 0 else np.nan
    rows["net_loss_efficiency"] = (total_loss - total_gain) / total_loss if total_loss > 0 else np.nan
    return rows


def _amounts(mask_loss: np.ndarray, mask_gain: np.ndarray, class_code: np.ndarray, loss_weight: np.ndarray, gain_weight: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for code, name in CLASS_CODEBOOK.items():
        cls = class_code == code
        out[name] = {
            "loss": float(np.nansum(np.where(cls & mask_loss, loss_weight, 0.0), dtype=np.float64)),
            "gain": float(np.nansum(np.where(cls & mask_gain, gain_weight, 0.0), dtype=np.float64)),
        }
    return out


DETECTOR_COLUMNS = [
    "pan_event_id",
    "event_date",
    "detector_gross_extent_loss_km2",
    "detector_gross_extent_gain_km2",
    "detector_physical_net_extent_change_km2",
    "detector_extent_loss_major_severe_km2",
    "detector_extent_loss_severe_nonmajor_km2",
    "detector_extent_loss_broad_nonsevere_km2",
    "detector_extent_loss_residual_km2",
    "detector_extent_gain_major_severe_km2",
    "detector_extent_gain_severe_nonmajor_km2",
    "detector_extent_gain_broad_nonsevere_km2",
    "detector_extent_gain_residual_km2",
    "major_detector_extent_loss_fraction",
    "severe_nonmajor_detector_extent_loss_fraction",
    "broad_nonsevere_detector_extent_loss_fraction",
    "residual_detector_extent_loss_fraction",
    "detector_loss_partition_closure_error_km2",
    "detector_gain_partition_closure_error_km2",
    "detector_partition_closure_status",
    "detector_compensation_ratio",
    "detector_net_loss_efficiency",
]

CDR_COLUMNS = [
    "pan_event_id",
    "event_date",
    "cdr_gross_sic_loss_km2eq",
    "cdr_gross_sic_gain_km2eq",
    "cdr_signed_sic_change_km2eq",
    "cdr_sic_loss_major_severe_km2eq",
    "cdr_sic_loss_severe_nonmajor_km2eq",
    "cdr_sic_loss_broad_nonsevere_km2eq",
    "cdr_sic_loss_residual_km2eq",
    "cdr_sic_gain_major_severe_km2eq",
    "cdr_sic_gain_severe_nonmajor_km2eq",
    "cdr_sic_gain_broad_nonsevere_km2eq",
    "cdr_sic_gain_residual_km2eq",
    "major_cdr_sic_loss_fraction",
    "severe_nonmajor_cdr_sic_loss_fraction",
    "broad_nonsevere_cdr_sic_loss_fraction",
    "residual_cdr_sic_loss_fraction",
    "cdr_loss_partition_closure_error_km2eq",
    "cdr_gain_partition_closure_error_km2eq",
    "cdr_partition_closure_status",
    "cdr_compensation_ratio",
    "cdr_net_loss_efficiency",
]


def detector_contribution_budget(pan_events: pd.DataFrame, detector_dir: Path, partition_dir: Path, out_dir: Path) -> pd.DataFrame:
    area, _, _ = load_cell_area_km2()
    rows = []
    for pan in pan_events.itertuples(index=False):
        pan_id = str(pan.pan_event_id)
        with xr.open_dataset(detector_dir / f"{pan_id}.nc") as det, xr.open_dataset(partition_dir / f"{pan_id}.nc") as part:
            loss_mask = np.asarray(det["detector_extent_loss_mask"].values, dtype=bool)
            gain_mask = np.asarray(det["detector_extent_gain_mask"].values, dtype=bool)
            code = np.asarray(part["detector_local_class_code"].values, dtype=np.int8)
        vals = _amounts(loss_mask, gain_mask, code, area, area)
        total_loss = float(np.nansum(np.where(loss_mask, area, 0.0), dtype=np.float64))
        total_gain = float(np.nansum(np.where(gain_mask, area, 0.0), dtype=np.float64))
        loss_sum = sum(v["loss"] for v in vals.values())
        gain_sum = sum(v["gain"] for v in vals.values())
        loss_err = loss_sum - total_loss
        gain_err = gain_sum - total_gain
        row = {
            "pan_event_id": pan_id,
            "event_date": pd.Timestamp(pan.date).strftime("%Y-%m-%d"),
            "detector_gross_extent_loss_km2": total_loss,
            "detector_gross_extent_gain_km2": total_gain,
            "detector_physical_net_extent_change_km2": total_gain - total_loss,
            "detector_extent_loss_major_severe_km2": vals["major_severe"]["loss"],
            "detector_extent_loss_severe_nonmajor_km2": vals["severe_nonmajor"]["loss"],
            "detector_extent_loss_broad_nonsevere_km2": vals["broad_nonsevere"]["loss"],
            "detector_extent_loss_residual_km2": vals["residual"]["loss"],
            "detector_extent_gain_major_severe_km2": vals["major_severe"]["gain"],
            "detector_extent_gain_severe_nonmajor_km2": vals["severe_nonmajor"]["gain"],
            "detector_extent_gain_broad_nonsevere_km2": vals["broad_nonsevere"]["gain"],
            "detector_extent_gain_residual_km2": vals["residual"]["gain"],
            "major_detector_extent_loss_fraction": vals["major_severe"]["loss"] / total_loss if total_loss > 0 else np.nan,
            "severe_nonmajor_detector_extent_loss_fraction": vals["severe_nonmajor"]["loss"] / total_loss if total_loss > 0 else np.nan,
            "broad_nonsevere_detector_extent_loss_fraction": vals["broad_nonsevere"]["loss"] / total_loss if total_loss > 0 else np.nan,
            "residual_detector_extent_loss_fraction": vals["residual"]["loss"] / total_loss if total_loss > 0 else np.nan,
            "detector_loss_partition_closure_error_km2": loss_err,
            "detector_gain_partition_closure_error_km2": gain_err,
            "detector_partition_closure_status": "PASS" if np.isclose(loss_err, 0.0, rtol=1e-12, atol=1e-6) and np.isclose(gain_err, 0.0, rtol=1e-12, atol=1e-6) else "FAIL",
            "detector_compensation_ratio": total_gain / total_loss if total_loss > 0 else np.nan,
            "detector_net_loss_efficiency": (total_loss - total_gain) / total_loss if total_loss > 0 else np.nan,
        }
        rows.append(row)
    df = pd.DataFrame(rows)[DETECTOR_COLUMNS]
    df.to_csv(out_dir / "panarctic_detector_contribution_budget.csv", index=False)
    _validate_budget(df, "detector")
    return df


def cdr_contribution_budget(pan_events: pd.DataFrame, cdr_dir: Path, partition_dir: Path, out_dir: Path) -> pd.DataFrame:
    area, _, _ = load_cell_area_km2()
    rows = []
    for pan in pan_events.itertuples(index=False):
        pan_id = str(pan.pan_event_id)
        with xr.open_dataset(cdr_dir / f"{pan_id}.nc") as cdr, xr.open_dataset(partition_dir / f"{pan_id}.nc") as part:
            loss = np.asarray(cdr["cdr_sic_loss"].values, dtype=float)
            gain = np.asarray(cdr["cdr_sic_gain"].values, dtype=float)
            code = np.asarray(part["cdr_local_class_code"].values, dtype=np.int8)
        loss_mask = np.isfinite(loss)
        gain_mask = np.isfinite(gain)
        loss_weight = np.where(loss_mask, loss * area, 0.0)
        gain_weight = np.where(gain_mask, gain * area, 0.0)
        vals = _amounts(loss_mask, gain_mask, code, loss_weight, gain_weight)
        total_loss = float(np.nansum(loss_weight, dtype=np.float64))
        total_gain = float(np.nansum(gain_weight, dtype=np.float64))
        loss_sum = sum(v["loss"] for v in vals.values())
        gain_sum = sum(v["gain"] for v in vals.values())
        loss_err = loss_sum - total_loss
        gain_err = gain_sum - total_gain
        row = {
            "pan_event_id": pan_id,
            "event_date": pd.Timestamp(pan.date).strftime("%Y-%m-%d"),
            "cdr_gross_sic_loss_km2eq": total_loss,
            "cdr_gross_sic_gain_km2eq": total_gain,
            "cdr_signed_sic_change_km2eq": total_gain - total_loss,
            "cdr_sic_loss_major_severe_km2eq": vals["major_severe"]["loss"],
            "cdr_sic_loss_severe_nonmajor_km2eq": vals["severe_nonmajor"]["loss"],
            "cdr_sic_loss_broad_nonsevere_km2eq": vals["broad_nonsevere"]["loss"],
            "cdr_sic_loss_residual_km2eq": vals["residual"]["loss"],
            "cdr_sic_gain_major_severe_km2eq": vals["major_severe"]["gain"],
            "cdr_sic_gain_severe_nonmajor_km2eq": vals["severe_nonmajor"]["gain"],
            "cdr_sic_gain_broad_nonsevere_km2eq": vals["broad_nonsevere"]["gain"],
            "cdr_sic_gain_residual_km2eq": vals["residual"]["gain"],
            "major_cdr_sic_loss_fraction": vals["major_severe"]["loss"] / total_loss if total_loss > 0 else np.nan,
            "severe_nonmajor_cdr_sic_loss_fraction": vals["severe_nonmajor"]["loss"] / total_loss if total_loss > 0 else np.nan,
            "broad_nonsevere_cdr_sic_loss_fraction": vals["broad_nonsevere"]["loss"] / total_loss if total_loss > 0 else np.nan,
            "residual_cdr_sic_loss_fraction": vals["residual"]["loss"] / total_loss if total_loss > 0 else np.nan,
            "cdr_loss_partition_closure_error_km2eq": loss_err,
            "cdr_gain_partition_closure_error_km2eq": gain_err,
            "cdr_partition_closure_status": "PASS" if np.isclose(loss_err, 0.0, rtol=1e-12, atol=1e-6) and np.isclose(gain_err, 0.0, rtol=1e-12, atol=1e-6) else "FAIL",
            "cdr_compensation_ratio": total_gain / total_loss if total_loss > 0 else np.nan,
            "cdr_net_loss_efficiency": (total_loss - total_gain) / total_loss if total_loss > 0 else np.nan,
        }
        rows.append(row)
    df = pd.DataFrame(rows)[CDR_COLUMNS]
    df.to_csv(out_dir / "panarctic_cdr_contribution_budget.csv", index=False)
    _validate_budget(df, "cdr")
    return df


def _validate_budget(df: pd.DataFrame, label: str) -> None:
    required = DETECTOR_COLUMNS if label == "detector" else CDR_COLUMNS
    if list(df.columns) != required:
        raise ValueError(f"{label} contribution budget schema is not canonical")
    forbidden = {
        "gross_loss",
        "gross_gain",
        "residual_gross_loss",
        "broad_nonsevere_gross_loss",
        "severe_nonmajor_gross_loss",
        "major_severe_gross_loss",
        "partition_loss_closure_error",
        "partition_gain_closure_error",
        "partition_closure_status",
        "compensation_ratio",
        "net_loss_efficiency",
    }
    if forbidden & set(df.columns):
        raise ValueError(f"{label} contribution budget contains forbidden generic aliases")
    amount_cols = [
        c
        for c in df.columns
        if ("gross" in c or "_loss_" in c or "_gain_" in c) and (c.endswith("_km2") or c.endswith("_km2eq"))
    ]
    amount_cols = [c for c in amount_cols if "closure_error" not in c]
    if (df[amount_cols] < -1e-9).any().any():
        raise ValueError(f"{label} contribution budget has negative class amount")
    status_col = "detector_partition_closure_status" if label == "detector" else "cdr_partition_closure_status"
    if not df[status_col].eq("PASS").all():
        raise ValueError(f"{label} contribution budget closure failed")
    frac_cols = [c for c in df.columns if c.endswith("_loss_fraction")]
    sums = df[frac_cols].sum(axis=1, skipna=True)
    total_col = "detector_gross_extent_loss_km2" if label == "detector" else "cdr_gross_sic_loss_km2eq"
    valid = df[total_col] > 0
    if valid.any() and not np.allclose(sums[valid], 1.0, rtol=1e-10, atol=1e-8):
        raise ValueError(f"{label} loss fractions do not sum to 1")


def region_budgets(pan_events: pd.DataFrame, detector_dir: Path, cdr_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    area, _, _ = load_cell_area_km2()
    surface = load_surface_mask()
    detector_rows = []
    cdr_rows = []
    for pan in pan_events.itertuples(index=False):
        pan_id = str(pan.pan_event_id)
        event_date = pd.Timestamp(pan.date).strftime("%Y-%m-%d")
        with xr.open_dataset(detector_dir / f"{pan_id}.nc") as det:
            det_loss = np.asarray(det["detector_extent_loss_mask"].values, dtype=bool)
            det_gain = np.asarray(det["detector_extent_gain_mask"].values, dtype=bool)
        with xr.open_dataset(cdr_dir / f"{pan_id}.nc") as cdr:
            cdr_loss = np.asarray(cdr["cdr_sic_loss"].values, dtype=float)
            cdr_gain = np.asarray(cdr["cdr_sic_gain"].values, dtype=float)
        for code in range(0, 19):
            region_mask = surface == code
            detector_rows.append(
                {
                    "pan_event_id": pan_id,
                    "event_date": event_date,
                    "region_code": code,
                    "region_name": REGION_BY_CODE.get(code, f"unknown_{code}"),
                    "detector_gross_extent_loss_km2": float(np.nansum(np.where(region_mask & det_loss, area, 0.0))),
                    "detector_gross_extent_gain_km2": float(np.nansum(np.where(region_mask & det_gain, area, 0.0))),
                }
            )
            cdr_rows.append(
                {
                    "pan_event_id": pan_id,
                    "event_date": event_date,
                    "region_code": code,
                    "region_name": REGION_BY_CODE.get(code, f"unknown_{code}"),
                    "cdr_gross_sic_loss_km2eq": float(np.nansum(np.where(region_mask & np.isfinite(cdr_loss), cdr_loss * area, 0.0))),
                    "cdr_gross_sic_gain_km2eq": float(np.nansum(np.where(region_mask & np.isfinite(cdr_gain), cdr_gain * area, 0.0))),
                }
            )
    detector_df = pd.DataFrame(detector_rows)
    cdr_df = pd.DataFrame(cdr_rows)
    detector_df.to_csv(out_dir / "panarctic_detector_region_budget.csv", index=False)
    cdr_df.to_csv(out_dir / "panarctic_cdr_region_budget.csv", index=False)
    return detector_df, cdr_df


def region_budget_accounting_audit(
    detector_budget: pd.DataFrame,
    cdr_budget: pd.DataFrame,
    detector_region: pd.DataFrame,
    cdr_region: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    det_global = detector_budget[["pan_event_id", "detector_gross_extent_loss_km2"]].copy()
    cdr_global = cdr_budget[["pan_event_id", "cdr_gross_sic_loss_km2eq"]].copy()
    det_codes = (
        detector_region[detector_region["region_code"].between(0, 18)]
        .groupby("pan_event_id", as_index=False)["detector_gross_extent_loss_km2"]
        .sum()
        .rename(columns={"detector_gross_extent_loss_km2": "detector_codes_0_18_loss_km2"})
    )
    cdr_codes = (
        cdr_region[cdr_region["region_code"].between(0, 18)]
        .groupby("pan_event_id", as_index=False)["cdr_gross_sic_loss_km2eq"]
        .sum()
        .rename(columns={"cdr_gross_sic_loss_km2eq": "cdr_codes_0_18_loss_km2eq"})
    )
    out = (
        det_global.rename(columns={"detector_gross_extent_loss_km2": "detector_global_loss_km2"})
        .merge(det_codes, on="pan_event_id", how="left")
        .merge(cdr_global.rename(columns={"cdr_gross_sic_loss_km2eq": "cdr_global_loss_km2eq"}), on="pan_event_id", how="left")
        .merge(cdr_codes, on="pan_event_id", how="left")
    )
    out["detector_unassigned_loss_km2"] = out["detector_global_loss_km2"] - out["detector_codes_0_18_loss_km2"]
    out["detector_unassigned_loss_fraction"] = np.where(
        out["detector_global_loss_km2"] > 0,
        out["detector_unassigned_loss_km2"] / out["detector_global_loss_km2"],
        np.nan,
    )
    out["cdr_unassigned_loss_km2eq"] = out["cdr_global_loss_km2eq"] - out["cdr_codes_0_18_loss_km2eq"]
    out["cdr_unassigned_loss_fraction"] = np.where(
        out["cdr_global_loss_km2eq"] > 0,
        out["cdr_unassigned_loss_km2eq"] / out["cdr_global_loss_km2eq"],
        np.nan,
    )
    out.to_csv(out_dir / "region_budget_accounting_audit.csv", index=False)
    return out
