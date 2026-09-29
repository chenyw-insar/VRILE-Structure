#!/usr/bin/env python3
"""Rebuild the local-event catalog from deterministic frozen patch cells.

This production entrypoint starts after patch detection. It preserves object
membership, loss weights, dates, regions, thresholds, and patch attributes;
only the canonical longitude geometry and its dependent catalog chain are
regenerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vrile.io import projected_grid_geometry
from vrile.local_objects import merge_local_patches, weighted_patch_centroid

import stage2_experiment_core as stage2_producer
import stage1_filter_severe_local_events as severe_producer
import stage1_consolidate_local_objects as broad_producer


FIG3_METRICS = [
    "pan_to_severe_pm3d",
    "pan_to_severe_pm3d_le500km",
    "pan_to_severe_pm3d_le300km",
    "pan_to_major_severe_pm3d",
    "pan_to_major_severe_pm3d_le500km",
    "pan_to_major_severe_pm3d_le300km",
    "major_severe_to_pan_pm3d",
    "major_severe_to_pan_pm3d_le500km",
    "major_severe_to_pan_pm3d_le300km",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-index", default="data/processed/local_event_cells/patch_cell_index.csv")
    parser.add_argument('--cells_dir', default="data/processed/local_event_cells")
    parser.add_argument("--patch-attributes", default="outputs/local_vrile_enhanced/local_objects_filtered.csv")
    parser.add_argument("--grid", default="data/raw/nsidc_ancillary/NSIDC0771_CellArea_PS_N25km_v1.1.nc")
    parser.add_argument("--panarctic-events", default="outputs/reproduce_sie/vrile_events_unique_both_jja_5p.csv")
    parser.add_argument("--panarctic-locations", default="outputs/reproduce_sie/vrile_locations.csv")
    parser.add_argument('--out_dir', required=True)
    parser.add_argument("--merge-days", type=int, default=2)
    parser.add_argument("--merge-distance-km", type=float, default=300.0)
    parser.add_argument(
        "--promotion-layout",
        action="store_true",
        help="Write a complete mirrored outputs/data tree suitable for transactional promotion.",
    )
    return parser.parse_args()


def root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_grid(path: Path):
    with xr.open_dataset(path, decode_cf=False, mask_and_scale=False) as dataset:
        native_x2, native_y2, inverse_transformer, source_crs = projected_grid_geometry(dataset)
        lon2, lat2 = inverse_transformer.transform(native_x2, native_y2)
    return (
        native_x2,
        native_y2,
        np.asarray(lon2, dtype=float),
        np.asarray(lat2, dtype=float),
        inverse_transformer,
        source_crs,
    )


def rebuild_patch_geometry(index: pd.DataFrame, cells_dir: Path, attributes: pd.DataFrame, grid) -> pd.DataFrame:
    native_x2, native_y2, lon2, lat2, inverse_transformer, _ = grid
    lookup = attributes.set_index("object_id", drop=False)
    rows = []
    for npz_name, group in index.groupby("npz_file", sort=True):
        with np.load(cells_dir / str(npz_name), allow_pickle=False) as data:
            object_index = data["object_index"]
            y_idx = data["y_idx"]
            x_idx = data["x_idx"]
            delta = data["delta_sic"].astype(float)
            for record in group.sort_values("start_offset").itertuples(index=False):
                start = int(record.start_offset)
                count = int(record.cell_count)
                stop = start + count
                if not np.all(object_index[start:stop] == int(record.object_index)):
                    raise RuntimeError(f"sparse object-index mismatch: {record.object_id}")
                source = lookup.loc[str(record.object_id)].to_dict()
                yi = y_idx[start:stop]
                xi = x_idx[start:stop]
                weights = -np.minimum(delta[start:stop], 0.0)
                centroid_lon, centroid_lat, geometry = weighted_patch_centroid(
                    lon2[yi, xi],
                    lat2[yi, xi],
                    weights,
                    native_x=native_x2[yi, xi],
                    native_y=native_y2[yi, xi],
                    inverse_transform=inverse_transformer.transform,
                    frozen_arithmetic_reference=(
                        float(source["centroid_lon"]),
                        float(source["centroid_lat"]),
                    ),
                )
                source.update(
                    {
                        "centroid_lon": centroid_lon,
                        "centroid_lat": centroid_lat,
                        "longitude_geometry_class": geometry.geometry_class,
                        "raw_signed_longitude_span_deg": geometry.raw_signed_span_deg,
                        "minimum_circular_covering_arc_deg": geometry.minimum_circular_covering_arc_deg,
                        "largest_longitude_gap_deg": geometry.largest_gap_deg,
                        "unwrap_arc_start_deg": geometry.unwrap_arc_start_deg,
                        "largest_gap_tie_count": geometry.largest_gap_tie_count,
                    }
                )
                rows.append(source)
    result = pd.DataFrame(rows)
    if len(result) != len(index) or result.object_id.nunique() != len(index):
        raise RuntimeError("rebuilt patch geometry does not cover the deterministic patch index")
    return result.sort_values(["region", "date", "cumulative_sic_loss"], ascending=[True, True, False]).reset_index(drop=True)


def local_frame(events: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": events["unique_local_event_id"].astype(str),
            "definition": "local",
            "region": events["dominant_region"],
            "event_date": pd.to_datetime(events["event_start"]),
            "start_date": pd.to_datetime(events["event_start"]),
            "end_date": pd.to_datetime(events["event_end"]),
            "centroid_lon": events["track_centroid_lon"].astype(float),
            "centroid_lat": events["track_centroid_lat"].astype(float),
        }
    )


def fig3_tables(
    pan: pd.DataFrame,
    severe: pd.DataFrame,
    major: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    severe_frame = local_frame(severe)
    major_frame = local_frame(major)
    specs = [
        ("pan_to_severe", pan, severe_frame),
        ("pan_to_major_severe", pan, major_frame),
        ("major_severe_to_pan", major_frame, pan),
    ]
    rows = []
    source_rows = []
    for direction, source, target in specs:
        for criterion, distance in [("pm3d", None), ("pm3d_le500km", 500.0), ("pm3d_le300km", 300.0)]:
            count, _ = stage2_producer.match_one(source, target, 3, False, distance)
            source_name, target_name = {
                "pan_to_severe": ("panarctic", "severe"),
                "pan_to_major_severe": ("panarctic", "major_severe"),
                "major_severe_to_pan": ("major_severe", "panarctic"),
            }[direction]
            rule = {
                "pm3d": "time-only_pm3d",
                "pm3d_le500km": "time+centroid_500km_pm3d",
                "pm3d_le300km": "time+centroid_300km_pm3d",
            }[criterion]
            rows.append(
                {
                    "metric": f"{direction}_{criterion}",
                    "numerator": int(count),
                    "denominator": int(len(source)),
                }
            )
            source_rows.append(
                {
                    "source_definition": source_name,
                    "target_definition": target_name,
                    "matching_rule": rule,
                    "numerator": int(count),
                    "denominator": int(len(source)),
                    "ratio": float(count / len(source)) if len(source) else np.nan,
                    "median_distance_km": np.nan,
                    "p90_distance_km": np.nan,
                    "region": "all",
                    "notes": "corrected local-catalog transition; cumulative matching criterion",
                    "analysis_role": "event_definition_comparison",
                    "scored_in_stage2_evidence": False,
                }
            )
    result = pd.DataFrame(rows)
    if result.metric.tolist() != FIG3_METRICS:
        raise RuntimeError("Fig.3 metric order changed unexpectedly")
    missed = []
    for event in major_frame.to_dict("records"):
        candidates = pan[(pd.to_datetime(pan["event_date"]) - pd.Timestamp(event["event_date"])).abs().dt.days <= 3]
        if candidates.empty:
            missed.append(event)
    return result, pd.DataFrame(source_rows), pd.DataFrame(missed)


def major_region_summary(severe_summary: pd.DataFrame, major: pd.DataFrame) -> pd.DataFrame:
    counts = (
        major.groupby("dominant_region")
        .agg(
            major_event_count=("unique_local_event_id", "nunique"),
            median_cumulative_loss=("cumulative_loss", "median"),
            median_max_area_km2=("max_area_km2", "median"),
            median_duration_days=("duration_days", "median"),
        )
        .reset_index()
        .rename(columns={"dominant_region": "region"})
    )
    result = severe_summary.merge(counts, on="region", how="left").fillna(0)
    result["major_fraction_of_severe"] = result.major_event_count / result.severe_event_count.replace(0, np.nan)
    return result


def output_directories(out: Path, promotion_layout: bool) -> dict[str, Path]:
    if not promotion_layout:
        return {"flat": out}
    directories = {
        "enhanced": out / "outputs/local_vrile_enhanced",
        "severe": out / "outputs/local_vrile_severe",
        "major": out / "outputs/local_vrile_major_severe",
        "matching": out / "outputs/spatiotemporal_matching",
        "cells": out / "data/processed/local_event_cells",
        "metadata": out,
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def main() -> None:
    args = parse_args()
    out = root_path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_path = root_path(args.patch_index)
    cells_dir = root_path(args.cells_dir)
    attributes_path = root_path(args.patch_attributes)
    grid_path = root_path(args.grid)
    index = pd.read_csv(index_path, parse_dates=["date", "start_date"])
    attributes = pd.read_csv(attributes_path, parse_dates=["date", "start_date"])
    grid = load_grid(grid_path)
    patches = rebuild_patch_geometry(index, cells_dir, attributes, grid)
    unique, membership = merge_local_patches(
        patches,
        args.merge_days,
        args.merge_distance_km,
        return_membership=True,
    )
    severe_all = severe_producer.mark_severe(unique, 20, 100, 2, -0.10)
    severe = severe_all[severe_all["is_severe_primary"]].copy()
    severe["severe_definition"] = "primary_union_top20_min100_duration2_or_high_intensity_sic-0.10"
    selected = stage2_producer.select_major_severe_events(severe)
    major = selected["union"].copy()
    pan = stage2_producer.load_panarctic(root_path(args.panarctic_events), root_path(args.panarctic_locations))
    fig3, fig3_source, missed_major = fig3_tables(pan, severe, major)
    broad_summary = broad_producer.region_summary(patches, unique)
    severe_summary = severe_producer.summarize_regions(unique, severe)
    major_summary = major_region_summary(severe_summary, major)
    directories = output_directories(out, args.promotion_layout)

    if args.promotion_layout:
        patches.to_csv(directories["enhanced"] / "local_objects_filtered.csv", index=False)
        unique.to_csv(directories["enhanced"] / "unique_local_events.csv", index=False)
        broad_summary.to_csv(directories["enhanced"] / "unique_local_event_region_summary.csv", index=False)
        membership.to_csv(directories["cells"] / "unique_event_patch_membership.csv", index=False)
        unique.to_csv(directories["severe"] / "broad_unique_local_events.csv", index=False)
        severe.to_csv(directories["severe"] / "severe_unique_local_events.csv", index=False)
        severe_summary.to_csv(directories["severe"] / "severe_local_event_region_summary.csv", index=False)
        selected["global"].to_csv(directories["major"] / "major_severe_events_global.csv", index=False)
        selected["region_normalized"].to_csv(directories["major"] / "major_severe_events_region_normalized.csv", index=False)
        selected["severity_index"].to_csv(directories["major"] / "major_severe_events_severity_index.csv", index=False)
        major.to_csv(directories["major"] / "major_severe_events_union.csv", index=False)
        selected["sensitivity"].to_csv(directories["major"] / "major_severe_sensitivity.csv", index=False)
        major_summary.to_csv(directories["major"] / "major_severe_event_region_summary.csv", index=False)
        fig3.to_csv(directories["matching"] / "fig3_count_vector.csv", index=False)
        fig3_source.to_csv(directories["matching"] / "matching_rule_sensitivity.csv", index=False)
        missed_major.to_csv(directories["matching"] / "missed_major_severe_events.csv", index=False)
    else:
        patches.to_csv(out / "local_objects_repaired_geometry.csv", index=False)
        unique.to_csv(out / "unique_local_events.csv", index=False)
        membership.to_csv(out / "unique_event_patch_membership.csv", index=False)
        severe.to_csv(out / "severe_unique_local_events.csv", index=False)
        major.to_csv(out / "major_severe_events_union.csv", index=False)
        fig3.to_csv(out / "fig3_count_vector.csv", index=False)
    patches.groupby("longitude_geometry_class", sort=True).size().rename("patch_count").reset_index().to_csv(
        out / "patch_geometry_classification_summary.csv", index=False
    )
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "merge_days": args.merge_days,
        "merge_distance_km": args.merge_distance_km,
        "source_crs_wkt": grid[-1].to_wkt(),
        "inputs": {
            str(index_path): sha256_file(index_path),
            str(attributes_path): sha256_file(attributes_path),
            str(grid_path): sha256_file(grid_path),
        },
        "production_sources": {
            "src/vrile/local_objects.py": sha256_file(ROOT / "src/vrile/local_objects.py"),
            "src/vrile/io.py": sha256_file(ROOT / "src/vrile/io.py"),
            "scripts/stage1_rebuild_local_catalog_from_frozen_cells.py": sha256_file(Path(__file__)),
            "scripts/stage1_filter_severe_local_events.py": sha256_file(ROOT / "scripts/stage1_filter_severe_local_events.py"),
            "scripts/stage2_experiment_core.py": sha256_file(ROOT / "scripts/stage2_experiment_core.py"),
        },
        "counts": {
            "patches": len(patches),
            "broad": len(unique),
            "severe": len(severe),
            "major_severe": len(major),
        },
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(out), **metadata["counts"], "fig3": fig3.numerator.tolist()}, indent=2))


if __name__ == "__main__":
    main()
