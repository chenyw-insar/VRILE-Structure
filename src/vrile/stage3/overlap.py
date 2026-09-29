"""Raster overlap between pan-Arctic fields and local footprints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from vrile.stage3.footprints import load_patch_cells
from vrile.stage3.grid import load_cell_area_km2


def event_level_lookup(unique_ids: pd.Series, severe_ids: set[str], major_ids: set[str]) -> dict[str, str]:
    out = {}
    for uid in unique_ids.astype(str):
        if uid in major_ids:
            out[uid] = "major_severe"
        elif uid in severe_ids:
            out[uid] = "severe"
        else:
            out[uid] = "broad"
    return out


def union_cells(
    patch_index: pd.DataFrame,
    local_cells_dir: Path,
    object_ids: set[str],
    shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    cells = load_patch_cells(patch_index, local_cells_dir, object_ids)
    union = np.zeros(shape, dtype=bool)
    for y, x, _ in cells.values():
        union[y, x] = True
    return union, cells


def run_overlap(
    pan_events: pd.DataFrame,
    membership: pd.DataFrame,
    patch_index: pd.DataFrame,
    unique_events: pd.DataFrame,
    severe: pd.DataFrame,
    major: pd.DataFrame,
    local_cells_dir: Path,
    cdr_dir: Path,
    detector_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    area_km2, _, _ = load_cell_area_km2()
    severe_ids = set(severe["unique_local_event_id"].astype(str)) if not severe.empty else set()
    major_ids = set(major["unique_local_event_id"].astype(str)) if not major.empty else set()
    levels = event_level_lookup(unique_events["unique_local_event_id"], severe_ids, major_ids)
    unique = unique_events.copy()
    unique["event_start"] = pd.to_datetime(unique["event_start"]).dt.normalize()
    unique["event_end"] = pd.to_datetime(unique["event_end"]).dt.normalize()
    membership = membership.copy()
    membership["unique_local_event_id"] = membership["unique_local_event_id"].astype(str)
    membership["object_id"] = membership["object_id"].astype(str)
    membership["date"] = pd.to_datetime(membership["date"]).dt.normalize()
    membership["start_date"] = pd.to_datetime(membership["start_date"]).dt.normalize()

    event_rows: list[dict] = []
    relation_rows: list[dict] = []
    pair_rows: list[dict] = []
    alignment_rows: list[dict] = []
    for pan in pan_events.itertuples(index=False):
        pan_id = str(pan.pan_event_id)
        event_date = pd.Timestamp(pan.date).normalize()
        win_start = event_date - pd.Timedelta(days=5)

        old_candidates = unique[(unique["event_start"] <= event_date) & (unique["event_end"] >= win_start)].copy()
        old_member_sub = membership[membership["unique_local_event_id"].isin(old_candidates["unique_local_event_id"].astype(str))]
        interval_member_sub = membership[(membership["start_date"] <= event_date) & (membership["date"] >= win_start)]
        member_sub = interval_member_sub[interval_member_sub["date"] <= event_date].copy()
        if (member_sub["date"] > event_date).any():
            raise ValueError(f"No-look-ahead gate failed for {pan_id}")
        candidate_ids = set(member_sub["unique_local_event_id"].astype(str))
        candidates = unique[unique["unique_local_event_id"].astype(str).isin(candidate_ids)].copy()

        cdr = xr.open_dataset(cdr_dir / f"{pan_id}.nc")
        det = xr.open_dataset(detector_dir / f"{pan_id}.nc")
        cdr_loss = np.asarray(cdr["cdr_sic_loss"].values, dtype=float)
        cdr_gain = np.asarray(cdr["cdr_sic_gain"].values, dtype=float)
        comp = np.asarray(cdr["loss_component_id"].values, dtype=np.int32)
        det_loss = np.asarray(det["detector_extent_loss_mask"].values, dtype=bool)
        det_gain = np.asarray(det["detector_extent_gain_mask"].values, dtype=bool)

        old_union, _ = union_cells(patch_index, local_cells_dir, set(old_member_sub["object_id"].astype(str)), cdr_loss.shape)
        union = np.zeros_like(cdr_loss, dtype=bool)
        cells = load_patch_cells(patch_index, local_cells_dir, set(member_sub["object_id"].astype(str)))

        level_counts = {"broad": 0, "severe": 0, "major_severe": 0}
        pan_relation_ids: set[str] = set()
        for uid, grp in member_sub.groupby("unique_local_event_id"):
            uid = str(uid)
            pan_relation_ids.add(uid)
            level = levels.get(uid, "broad")
            level_counts[level] += 1
            local_mask = np.zeros_like(cdr_loss, dtype=bool)
            time_aligned = 0
            for obj in grp["object_id"].astype(str):
                if obj not in cells:
                    continue
                y, x, _ = cells[obj]
                local_mask[y, x] = True
                time_aligned += 1
            union |= local_mask
            local_area = float(np.nansum(np.where(local_mask, area_km2, 0.0)))
            relation_rows.append(
                {
                    "pan_event_id": pan_id,
                    "unique_local_event_id": uid,
                    "event_level": level,
                    "time_aligned_patch_count": int(time_aligned),
                    "local_footprint_cells": int(local_mask.sum()),
                    "local_footprint_area_km2": local_area,
                    "cdr_sic_loss_in_local_footprint_km2eq": float(np.nansum(np.where(local_mask, cdr_loss * area_km2, 0.0))),
                    "cdr_sic_gain_in_local_footprint_km2eq": float(np.nansum(np.where(local_mask, cdr_gain * area_km2, 0.0))),
                    "detector_gross_extent_loss_in_local_footprint_km2": float(np.nansum(np.where(local_mask & det_loss, area_km2, 0.0))),
                    "detector_gross_extent_gain_in_local_footprint_km2": float(np.nansum(np.where(local_mask & det_gain, area_km2, 0.0))),
                }
            )
            for cid in sorted(int(v) for v in np.unique(comp[local_mask]) if int(v) > 0):
                comp_mask = comp == cid
                inter = local_mask & comp_mask
                if not inter.any():
                    continue
                inter_area = float(np.nansum(np.where(inter, area_km2, 0.0)))
                local_area_den = float(np.nansum(np.where(local_mask, area_km2, 0.0)))
                comp_area = float(np.nansum(np.where(comp_mask, area_km2, 0.0)))
                inter_loss = float(np.nansum(np.where(inter, cdr_loss * area_km2, 0.0)))
                comp_loss = float(np.nansum(np.where(comp_mask, cdr_loss * area_km2, 0.0)))
                union_area = float(np.nansum(np.where(local_mask | comp_mask, area_km2, 0.0)))
                pair_rows.append(
                    {
                        "pan_event_id": pan_id,
                        "unique_local_event_id": uid,
                        "event_level": level,
                        "component_id": cid,
                        "intersection_cells": int(inter.sum()),
                        "intersection_area_km2": inter_area,
                        "IoU": inter_area / union_area if union_area > 0 else np.nan,
                        "local_capture_fraction": inter_area / local_area_den if local_area_den > 0 else np.nan,
                        "component_capture_fraction": inter_area / comp_area if comp_area > 0 else np.nan,
                        "cdr_sic_loss_in_intersection_km2eq": inter_loss,
                        "cdr_loss_weighted_overlap_fraction": inter_loss / comp_loss if comp_loss > 0 else np.nan,
                    }
                )
        if pan_relation_ids != candidate_ids:
            raise ValueError(f"Aligned relation ULE set mismatch for {pan_id}")
        if len(candidates) != sum(level_counts.values()):
            raise ValueError(f"Candidate hierarchy count mismatch for {pan_id}")

        old_cells = int(old_union.sum())
        primary_cells = int(union.sum())
        future_excluded = int((interval_member_sub["date"] > event_date).sum())
        reduction = old_cells - primary_cells
        if primary_cells > old_cells or reduction < 0 or future_excluded < 0:
            raise ValueError(f"Temporal alignment audit gate failed for {pan_id}")
        alignment_rows.append(
            {
                "pan_event_id": pan_id,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "pan_window_start": win_start.strftime("%Y-%m-%d"),
                "pan_window_end": event_date.strftime("%Y-%m-%d"),
                "old_ule_lifespan_candidate_count": int(len(old_candidates)),
                "patch_interval_candidate_count": int(interval_member_sub["unique_local_event_id"].nunique()),
                "primary_no_lookahead_candidate_count": int(len(candidates)),
                "old_lifetime_member_patch_count": int(len(old_member_sub)),
                "interval_intersecting_patch_count": int(len(interval_member_sub)),
                "primary_no_lookahead_patch_count": int(len(member_sub)),
                "future_patch_excluded_count": future_excluded,
                "old_union_footprint_cells": old_cells,
                "primary_union_footprint_cells": primary_cells,
                "union_footprint_cell_reduction": reduction,
                "union_footprint_cell_reduction_fraction": reduction / old_cells if old_cells > 0 else 0.0,
            }
        )
        event_rows.append(
            {
                "pan_event_id": pan_id,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "n_temporal_candidates": int(len(candidates)),
                "no_temporal_candidate": bool(len(candidates) == 0),
                "n_broad_candidates": level_counts["broad"],
                "n_severe_candidates": level_counts["severe"],
                "n_major_severe_candidates": level_counts["major_severe"],
                "union_local_footprint_cells": primary_cells,
                "union_local_footprint_area_km2": float(np.nansum(np.where(union, area_km2, 0.0))),
                "cdr_sic_loss_in_union_local_footprint_km2eq": float(np.nansum(np.where(union, cdr_loss * area_km2, 0.0))),
                "cdr_sic_gain_in_union_local_footprint_km2eq": float(np.nansum(np.where(union, cdr_gain * area_km2, 0.0))),
                "detector_gross_extent_loss_in_union_local_footprint_km2": float(np.nansum(np.where(union & det_loss, area_km2, 0.0))),
                "detector_gross_extent_gain_in_union_local_footprint_km2": float(np.nansum(np.where(union & det_gain, area_km2, 0.0))),
            }
        )
        cdr.close()
        det.close()
    return pd.DataFrame(event_rows), pd.DataFrame(relation_rows), pd.DataFrame(pair_rows), pd.DataFrame(alignment_rows)

