#!/usr/bin/env python3
"""Prepare hash-pinned frozen VRILE products as GMT input tables.

This program is deliberately non-rendering.  It verifies the authoritative
scientific inputs, performs only deterministic tabular/projection operations,
and writes plain GMT-ready tables plus a hash manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer


EXPECTED = {
    'F1/F1_R3_annual_mean_context.csv': "55be2f9abbdccc546e6a5b0b9db12dcc221ceb31cb5a7108cf3e3b72aadad31c",
    'F1/F1_R3_observed_window.csv': "93e6a0ddfe2c3d478e31d58d9829633480425de9a74900740f2cb975f1b00449",
    'F1/F1_R3_scale_payload.json': "47650824a9dbc90903ed6efd53abba664ab0dab97b80a176738adc0499f03a8a",
    'F2/F2_R3_event_display.csv': "45c707820d164aec8a78c3c51d5db14b204cbd5c33b26317a6551ee047facdf3",
    'F2/F2_R3_year_halfmonth_counts.csv': "3956e49484055b9f0a873b3293137b627bcb33930e2d59d90e4ba328b2dd8f6a",
    'raw/nsidc_region_masks/NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc': "5cc42dba4e57e7f55abe448e010100367fb679435024c7fcf37a29d55c29ca92",
    'F3_F4/F3_revision_round3_formal_evidence_matrix.csv': "37ec9086f8002006a07a253021a04af3014772ab625ad8f39f51c3eed0312c78",
    'F3_F4/F3_revision_round3_interpretation_effect_rows.csv': "ca0e4c0e42a14660faa425cfff73057c76bc76d65a76b0e69d195b4f1512f6f9",
    'F3_F4/F3_revision_round3_kara_registered_changes.csv': "915b2a67ee1ce987f8b0cf8e6d0847ad9dd99c034e7da0b835066f8e851fd192",
    'F3_F4/F4A_REPRESENTATIVE_COMPONENT_CELLS.csv': "bd518846f8c7381559804d341cdddbb469dee65ce178224a3dc702b30bdab6a7",
    'F4/F4B_POPULATION_SOURCE_DATA.csv': "a5f0d664a38fa2baf87b192bb1c10b8c598dac440f9d01eb72b20d16dac7e96b",
    'F3_F4/F4C_CORRESPONDENCE_DEFINITION_DATA.csv': "835ae78a70a2b3d8c72ec9c1a3f2446f88ced702a82ed20275581387adab50f8",
    'F4/F4D_NORMALIZED_SOURCE_DATA.csv': "7a7f8bfcbb20135284c18b09a9b142ccc57ac094f8d1553e2439881a398c0272",
    'F3_F4/F4S_MULTIREGION_SOURCE_DATA.csv': "dd91a91573c8d510bd8e55e46a2de2965c1aad3816c6c73d472bddbe19b70f1c",
    'F3_F4/F4S_ABSOLUTE_DOMAIN_CONTRAST_SOURCE_DATA.csv': "2ac4774ebb2ef89ca13694e96573dfc6b837dda4ead8480c43863e371fb16d98",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, header=False, float_format="%.15g", lineterminator="\n")


def stable_strip(values: pd.Series, ids: pd.Series, y0: float = 1.0) -> pd.DataFrame:
    frame = pd.DataFrame({"value": pd.to_numeric(values), "event_id": ids.astype(str)}).sort_values(
        ["value", "event_id"], kind="mergesort"
    )
    n = len(frame)
    offsets = ((np.arange(n) % 9) - 4) * 0.035
    frame["y"] = y0 + offsets
    return frame[["value", "y"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, dest='root')
    parser.add_argument('--output_dir', required=True, dest='output')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    source_rows = []
    for rel, expected in EXPECTED.items():
        path = root / rel
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"HOLD_FROZEN_FIGURE_INPUT_HASH_MISMATCH:{rel}:{actual}")
        source_rows.append((rel, actual, path.stat().st_size))

    # F1: direct frozen tables.
    annual = pd.read_csv(root / next(k for k in EXPECTED if k.endswith("F1_R3_annual_mean_context.csv")))
    assert len(annual) == 37 and annual["selected_year"].sum() == 1
    write_tsv(annual[["year", "annual_mean_extent", "selected_year"]].assign(selected_year=annual.selected_year.astype(int)), out / "f1_annual.tsv")
    window = pd.read_csv(root / next(k for k in EXPECTED if k.endswith("F1_R3_observed_window.csv")))
    assert len(window) == 31 and np.isclose(window.loc[window.is_delta_center, "delta_sie"].iloc[0], -0.598)
    window_plot = pd.DataFrame({"day": np.arange(len(window)), "extent": window.extent})
    write_tsv(window_plot, out / "f1_window.tsv")
    write_tsv(window_plot.loc[window.is_delta_start | window.is_delta_end], out / "f1_window_highlight.tsv")

    # F2: one row per accepted unique event.
    events = pd.read_csv(root / next(k for k in EXPECTED if k.endswith("F2_R3_event_display.csv")))
    assert len(events) == 9530 and events.unique_local_event_id.nunique() == 9530
    class_order = ["broad_only", "severe_nonmajor", "major_severe"]
    expected_counts = {"broad_only": 7103, "severe_nonmajor": 1873, "major_severe": 554}
    assert events.exclusive_class.value_counts().to_dict() == expected_counts
    for cls in class_order:
        subset = events.loc[events.exclusive_class == cls]
        write_tsv(subset[["track_centroid_lon", "track_centroid_lat"]], out / f"f2_map_{cls}.tsv")
        scatter = pd.DataFrame({"duration": subset.duration_days, "log10_area": np.log10(subset.max_area_km2)})
        write_tsv(scatter, out / f"f2_scatter_{cls}.tsv")
    counts = pd.DataFrame(
        {"x": [1, 2, 3], "count": [9530, 2427, 554], "label": ["Broad", "Severe", "Major"]}
    )
    write_tsv(counts, out / "f2_counts.tsv")
    heat = pd.read_csv(root / next(k for k in EXPECTED if k.endswith("F2_R3_year_halfmonth_counts.csv")))
    assert len(heat) == 222 and heat.total_events.sum() == 9530
    heat_out = pd.DataFrame({"year": heat.year, "halfmonth": heat.half_month_index, "count": heat.total_events})
    write_tsv(heat_out, out / "f2_heatmap.tsv")

    # Native NSIDC mask projected to lon/lat only for GMT map display.
    mask_path = root / next(k for k in EXPECTED if k.endswith("NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc"))
    transformer = Transformer.from_crs(3413, 4326, always_xy=True)
    with xr.open_dataset(mask_path, decode_cf=False) as dataset:
        xs = dataset.x.values.astype(float)
        ys = dataset.y.values.astype(float)
        mask = dataset["sea_ice_region_surface_mask"].values
    xx, yy = np.meshgrid(xs, ys)
    region_codes = [(2, "Beaufort Sea"), (5, "Laptev Sea"), (6, "Kara Sea"), (7, "Barents Sea"), (1, "Central Arctic")]
    region_labels = []
    for code, label in region_codes:
        use = mask == code
        lon, lat = transformer.transform(xx[use], yy[use])
        write_tsv(pd.DataFrame({"lon": lon, "lat": lat}), out / f"f3_region_{code}.tsv")
        cx, cy = float(np.mean(xx[use])), float(np.mean(yy[use]))
        clon, clat = transformer.transform(cx, cy)
        region_labels.append((clon, clat, label))
    write_tsv(pd.DataFrame(region_labels, columns=["lon", "lat", "label"]), out / "f3_region_labels.tsv")

    # F3 matrix and interpretation layer.
    matrix = pd.read_csv(root / next(k for k in EXPECTED if k.endswith("F3_revision_round3_formal_evidence_matrix.csv")))
    status_code = {"NOT_SUPPORTED": 0, "SUPPORTED_PRIMARY_ONLY": 1, "SUPPORTED_WITH_ROBUSTNESS": 2}
    status_letter = {0: "N", 1: "P", 2: "R"}
    matrix_out = pd.DataFrame(
        {
            "x": matrix.region_order + 0.5,
            "y": 7.5 - matrix.mechanism_order,
            "status": matrix.manuscript_status.map(status_code),
        }
    )
    matrix_out["letter"] = matrix_out.status.map(status_letter)
    assert matrix_out.status.value_counts().to_dict() == {0: 27, 1: 11, 2: 2}
    write_tsv(matrix_out, out / "f3_matrix.tsv")
    effects = pd.read_csv(root / next(k for k in EXPECTED if k.endswith("F3_revision_round3_interpretation_effect_rows.csv")))
    effects["row"] = effects.groupby("case", sort=False).cumcount().rsub(6)
    for prefix, est, low, high in [
        ("primary", "primary_difference", "primary_ci_low", "primary_ci_high"),
        ("floor015", "floor015_difference", "floor015_ci_low", "floor015_ci_high"),
    ]:
        tab = effects[["case", "display_label", "row", est, low, high]].rename(columns={est: "estimate", low: "low", high: "high"})
        for idx, (_, group) in enumerate(tab.groupby("case", sort=False), start=1):
            write_tsv(group[["estimate", "row", "low", "high", "display_label"]], out / f"f3_c{idx}_{prefix}.tsv")
    kara = pd.read_csv(root / next(k for k in EXPECTED if k.endswith("F3_revision_round3_kara_registered_changes.csv")))
    assert len(kara) == 3
    kara.to_csv(out / "f3_c3_registered_changes.tsv", sep="\t", index=False, lineterminator="\n")

    # F4 representative field in lon/lat and exact native-cell perimeter.
    cells = pd.read_csv(root / next(k for k in EXPECTED if k.endswith("F4A_REPRESENTATIVE_COMPONENT_CELLS.csv")))
    assert len(cells) == 2777 and cells.loss_component_id.nunique() == 38
    lon, lat = transformer.transform(cells.x_m.to_numpy(), cells.y_m.to_numpy())
    component_out = pd.DataFrame(
        {"lon": lon, "lat": lat, "rank": cells.component_loss_rank, "largest": cells.is_largest_component.astype(int), "loss": cells.cdr_sic_loss}
    )
    write_tsv(component_out, out / "f4_components.tsv")
    largest_xy = {(int(round(x)), int(round(y))) for x, y in cells.loc[cells.is_largest_component, ["x_m", "y_m"]].itertuples(index=False)}
    segments = []
    half = 12500
    for x, y in sorted(largest_xy):
        edges = [
            ((x - half, y - half), (x + half, y - half), (0, -25000)),
            ((x + half, y - half), (x + half, y + half), (25000, 0)),
            ((x + half, y + half), (x - half, y + half), (0, 25000)),
            ((x - half, y + half), (x - half, y - half), (-25000, 0)),
        ]
        for a, b, neighbor in edges:
            if (x + neighbor[0], y + neighbor[1]) not in largest_xy:
                alon, alat = transformer.transform(*a)
                blon, blat = transformer.transform(*b)
                segments.extend([(">", np.nan), (alon, alat), (blon, blat)])
    with (out / "f4_largest_perimeter.gmt").open("w", encoding="utf-8", newline="\n") as handle:
        for a, b in segments:
            if a == ">":
                handle.write(">\n")
            else:
                handle.write(f"{a:.15g}\t{b:.15g}\n")

    population = pd.read_csv(root / next(k for k in EXPECTED if k.endswith("F4B_POPULATION_SOURCE_DATA.csv")))
    assert len(population) == 99
    metrics = [
        ("neff", "effective_component_number_sic", 1.0),
        ("largest", "largest_component_resolved_sic_loss_share", 100.0),
        ("top3", "top3_component_resolved_sic_loss_share", 100.0),
        ("spread", "weighted_component_spatial_spread_km", 1.0),
    ]
    summaries = []
    for name, column, scale in metrics:
        values = population[column] * scale
        write_tsv(stable_strip(values, population.pan_event_id), out / f"f4b_{name}_points.tsv")
        summaries.append((name, float(values.median()), float(values.min()), float(values.max())))
    write_tsv(pd.DataFrame(summaries, columns=["metric", "median", "min", "max"]), out / "f4b_summary.tsv")

    # F5 correspondence and normalized contrasts.
    cross = pd.read_csv(root / next(k for k in EXPECTED if k.endswith("F4C_CORRESPONDENCE_DEFINITION_DATA.csv")), dtype=str)
    bool_rows = cross.loc[cross.event_start_temporal_correspondence_3d.isin(["True", "False"])].copy()
    assert bool_rows.n_events.astype(int).sum() == 554
    coords = {("True", "True"): (0.5, 1.5), ("True", "False"): (1.5, 1.5), ("False", "True"): (0.5, 0.5), ("False", "False"): (1.5, 0.5)}
    f5a = []
    for row in bool_rows.itertuples(index=False):
        x, y = coords[(row.event_start_temporal_correspondence_3d, row.track_s_footprint_linked)]
        f5a.append((x, y, int(row.n_events), 100.0 * int(row.n_events) / 554.0))
    write_tsv(pd.DataFrame(f5a, columns=["x", "y", "count", "percent"]), out / "f5a_matrix.tsv")
    norm = pd.read_csv(root / next(k for k in EXPECTED if k.endswith("F4D_NORMALIZED_SOURCE_DATA.csv")))
    ymap = {"focal": 3.0, "residual": 2.0, "other_region": 1.0}
    norm["y"] = norm.domain.map(ymap)
    write_tsv(norm[["estimate", "y", "ci_low", "ci_high", "series", "domain"]], out / "f5b_contrasts.tsv")

    # Supporting figures.
    regions = pd.read_csv(root / next(k for k in EXPECTED if k.endswith("F4S_MULTIREGION_SOURCE_DATA.csv")))
    assert len(regions) == 99 and set(regions.stage3_region_framework) == {"18_named_regions"}
    s1_metrics = [
        ("neff", "effective_named_region_number_sic_loss", 1.0),
        ("largest", "largest_named_region_sic_loss_share", 100.0),
        ("top3", "top3_named_region_sic_loss_share", 100.0),
    ]
    s1_summary = []
    for name, column, scale in s1_metrics:
        values = regions[column] * scale
        write_tsv(stable_strip(values, regions.pan_event_id), out / f"s1_{name}_points.tsv")
        s1_summary.append((name, float(values.median()), float(values.min()), float(values.max())))
    write_tsv(pd.DataFrame(s1_summary, columns=["metric", "median", "min", "max"]), out / "s1_summary.tsv")
    absolute = pd.read_csv(root / next(k for k in EXPECTED if k.endswith("F4S_ABSOLUTE_DOMAIN_CONTRAST_SOURCE_DATA.csv")))
    absolute["y"] = absolute.metric.map(
        {"local_sic_loss_km2eq": 3.0, "residual_sic_loss_km2eq": 2.0, "other_region_sic_loss_km2eq": 1.0}
    )
    write_tsv(absolute[["mean_difference", "y", "ci_low", "ci_high", "metric", "median_initial_ice_equivalent_opportunity_km2eq"]], out / "s2_contrasts.tsv")

    # Hash every generated table after all writes.
    prepared_rows = []
    for path in sorted(out.iterdir()):
        if path.is_file():
            prepared_rows.append((path.name, sha256(path), path.stat().st_size))
    with (out / "GMT_INPUT_TABLE_MANIFEST.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["kind", "path", "sha256", "bytes"])
        for rel, digest, size in source_rows:
            writer.writerow(["frozen_source", rel, digest, size])
        for name, digest, size in prepared_rows:
            writer.writerow(["prepared_input", name, digest, size])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
