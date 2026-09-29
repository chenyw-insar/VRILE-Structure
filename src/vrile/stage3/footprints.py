"""Sparse local footprint reconstruction for Stage 3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vrile.local_objects import detect_connected_components, merge_local_patches, patch_key
from vrile.stage3.grid import load_cell_area_km2, pan_arctic_stage1_mask, read_sic_with_lonlat


def filtered_primary_objects(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date", "start_date"])
    return df.sort_values(["date", "window_days", "threshold", "object_id"]).reset_index(drop=True)


def reconstruct_patch_cells(filtered: pd.DataFrame, out_dir: Path, reconciliation_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    area_km2, _, _ = load_cell_area_km2()
    pan_mask = pan_arctic_stage1_mask()
    index_rows: list[dict] = []
    recon_rows: list[dict] = []
    year_arrays: dict[int, dict[str, list[np.ndarray]]] = {}

    grouped = list(filtered.groupby(["date", "start_date", "window_days", "threshold"], sort=True))
    for group_idx, ((date, start_date, window, threshold), report) in enumerate(grouped, start=1):
        date = pd.Timestamp(date).normalize()
        start_date = pd.Timestamp(start_date).normalize()
        end_arr, _, _, lon2, lat2 = read_sic_with_lonlat(date)
        start_arr, _, _, _, _ = read_sic_with_lonlat(start_date)
        diff = end_arr - start_arr
        components = detect_connected_components(
            diff,
            np.isfinite(diff) & pan_mask,
            float(threshold),
            50,
            lon2=lon2,
            lat2=lat2,
            binary_closing=False,
            strict_min_cells=False,
        )
        report_sorted = report.sort_values("object_id").reset_index(drop=True)
        comp_sorted = components
        if len(comp_sorted) != len(report_sorted):
            raise ValueError(f"Patch reconstruction count mismatch for {date:%Y-%m-%d} window={window} threshold={threshold}: recomputed={len(comp_sorted)} reported={len(report_sorted)}")

        year = int(date.year)
        store = year_arrays.setdefault(year, {"object_index": [], "y_idx": [], "x_idx": [], "delta_sic": []})
        for rank0, (row, comp) in enumerate(zip(report_sorted.to_dict("records"), comp_sorted)):
            object_index = len(index_rows)
            y_idx = comp.y_idx.astype(np.int32)
            x_idx = comp.x_idx.astype(np.int32)
            delta_sic = diff[y_idx, x_idx].astype(np.float32)
            start_offset = sum(len(a) for a in store["y_idx"])
            store["object_index"].append(np.full(len(y_idx), object_index, dtype=np.int32))
            store["y_idx"].append(y_idx)
            store["x_idx"].append(x_idx)
            store["delta_sic"].append(delta_sic)
            area_sum = float(np.nansum(area_km2[y_idx, x_idx]))
            reported_patch_key = patch_key(row)
            group_key = f"{date:%Y-%m-%d}|{start_date:%Y-%m-%d}|{int(window)}|{float(threshold):.6g}"
            reported_cell_count = int(row["object_area_cells"])
            recomputed_legacy_area = float(comp.cell_count) * 625.0
            metric_pass = (
                reported_cell_count == comp.cell_count
                and np.isclose(float(row["mean_sic_change"]), comp.mean_sic_change, rtol=1e-8, atol=5e-6)
                and np.isclose(float(row["min_sic_change"]), comp.min_sic_change, rtol=1e-8, atol=5e-6)
                and np.isclose(float(row["cumulative_sic_loss"]), comp.cumulative_sic_loss, rtol=1e-6, atol=1e-3)
                and np.isclose(float(row["centroid_lon"]), comp.weighted_centroid_lon, rtol=1e-8, atol=5e-6)
                and np.isclose(float(row["centroid_lat"]), comp.weighted_centroid_lat, rtol=1e-8, atol=5e-6)
                and np.isclose(float(row["max_loss_lon"]), comp.max_loss_lon, rtol=1e-8, atol=5e-6)
                and np.isclose(float(row["max_loss_lat"]), comp.max_loss_lat, rtol=1e-8, atol=5e-6)
                and np.isclose(float(row["object_area_km2"]), recomputed_legacy_area, rtol=0.0, atol=1e-9)
            )
            recon_rows.append(
                {
                    "group_key": group_key,
                    "reported_object_id": row["object_id"],
                    "reported_patch_key": reported_patch_key,
                    "reported_component_rank": rank0 + 1,
                    "recomputed_label_id": comp.label_id,
                    "recomputed_component_rank": rank0 + 1,
                    "reported_cell_count": reported_cell_count,
                    "recomputed_cell_count": comp.cell_count,
                    "reported_mean_sic_change": float(row["mean_sic_change"]),
                    "recomputed_mean_sic_change": comp.mean_sic_change,
                    "reported_min_sic_change": float(row["min_sic_change"]),
                    "recomputed_min_sic_change": comp.min_sic_change,
                    "reported_cumulative_sic_loss": float(row["cumulative_sic_loss"]),
                    "recomputed_cumulative_sic_loss": comp.cumulative_sic_loss,
                    "reported_centroid_lon": float(row["centroid_lon"]),
                    "recomputed_centroid_lon": comp.weighted_centroid_lon,
                    "reported_centroid_lat": float(row["centroid_lat"]),
                    "recomputed_centroid_lat": comp.weighted_centroid_lat,
                    "reported_max_loss_lon": float(row["max_loss_lon"]),
                    "recomputed_max_loss_lon": comp.max_loss_lon,
                    "reported_max_loss_lat": float(row["max_loss_lat"]),
                    "recomputed_max_loss_lat": comp.max_loss_lat,
                    "reported_legacy_nominal_area_km2": float(row["object_area_km2"]),
                    "recomputed_legacy_nominal_area_km2": recomputed_legacy_area,
                    "recomputed_nsidc0771_cell_area_km2": area_sum,
                    "status": "PASS" if metric_pass else "FAIL",
                }
            )
            index_rows.append(
                {
                    "object_index": object_index,
                    "object_id": row["object_id"],
                    "patch_key": patch_key(row),
                    "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                    "start_date": pd.Timestamp(row["start_date"]).strftime("%Y-%m-%d"),
                    "window_days": int(row["window_days"]),
                    "threshold": float(row["threshold"]),
                    "year": year,
                    "npz_file": f"{year}.npz",
                    "start_offset": int(start_offset),
                    "cell_count": int(len(y_idx)),
                    "cell_area_km2": area_sum,
                }
            )
        if group_idx % 250 == 0:
            print(f"reconstructed footprint groups {group_idx}/{len(grouped)} patches={len(index_rows)}", flush=True)

    for year, arrays in year_arrays.items():
        np.savez_compressed(
            out_dir / f"{year}.npz",
            object_index=np.concatenate(arrays["object_index"]) if arrays["object_index"] else np.asarray([], dtype=np.int32),
            y_idx=np.concatenate(arrays["y_idx"]) if arrays["y_idx"] else np.asarray([], dtype=np.int32),
            x_idx=np.concatenate(arrays["x_idx"]) if arrays["x_idx"] else np.asarray([], dtype=np.int32),
            delta_sic=np.concatenate(arrays["delta_sic"]) if arrays["delta_sic"] else np.asarray([], dtype=np.float32),
        )
    index = pd.DataFrame(index_rows)
    recon = pd.DataFrame(recon_rows)
    index.to_csv(out_dir / "patch_cell_index.csv", index=False)
    recon.to_csv(reconciliation_path, index=False)
    if not recon["status"].eq("PASS").all():
        raise ValueError("footprint_detector_reconciliation.csv contains FAIL rows")
    return index, recon


def restore_unique_membership(
    filtered: pd.DataFrame,
    canonical_unique: pd.DataFrame,
    out_dir: Path,
    reconciliation_path: Path,
    *,
    merge_days: int = 2,
    merge_distance_km: float = 300.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique, membership = merge_local_patches(filtered, merge_days, merge_distance_km, return_membership=True)
    membership.to_csv(out_dir / "unique_event_patch_membership.csv", index=False)
    compare_cols = ["unique_local_event_id", "event_start", "event_end", "cumulative_loss", "max_area_km2"]
    left = canonical_unique[compare_cols].sort_values("unique_local_event_id").reset_index(drop=True)
    right = unique[compare_cols].sort_values("unique_local_event_id").reset_index(drop=True)
    rows = []
    if len(left) != len(right):
        rows.append({"check": "unique_event_count", "status": "FAIL", "reported": len(left), "recomputed": len(right)})
    else:
        rows.append({"check": "unique_event_count", "status": "PASS", "reported": len(left), "recomputed": len(right)})
    for col in compare_cols:
        if col in {"cumulative_loss", "max_area_km2"}:
            ok = bool(np.allclose(pd.to_numeric(left[col]), pd.to_numeric(right[col]), rtol=1e-6, atol=1e-3))
        else:
            ok = bool(left[col].astype(str).equals(right[col].astype(str)))
        rows.append({"check": col, "status": "PASS" if ok else "FAIL", "reported": "", "recomputed": ""})
    rec = pd.DataFrame(rows)
    rec.to_csv(reconciliation_path, index=False)
    if not rec["status"].eq("PASS").all():
        raise ValueError("unique_event_membership_reconciliation.csv contains FAIL rows")
    return membership, rec


def load_patch_cells(index: pd.DataFrame, base_dir: Path, object_ids: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    sub = index[index["object_id"].isin(object_ids)]
    for npz_name, grp in sub.groupby("npz_file"):
        data = np.load(base_dir / npz_name)
        object_index = data["object_index"]
        y_all = data["y_idx"]
        x_all = data["x_idx"]
        d_all = data["delta_sic"]
        for row in grp.itertuples(index=False):
            mask = object_index == int(row.object_index)
            out[str(row.object_id)] = (y_all[mask], x_all[mask], d_all[mask])
    return out
