#!/usr/bin/env python3
"""Assemble the six completed floor015 Stage-2 families without rerunning them."""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("VRILE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve() / '.'
sys.path.insert(0, str(ROOT / "src"))
from vrile.robustness.downstream_floor015 import (  # noqa: E402
    STAGE2_TABLE_SPECS,
    _pair_evidence_table,
    _stage2_region_summary,
    _wrong_region_four_of_four_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor-run", required=True)
    parser.add_argument('--primary_output_dir', required=True, dest='primary_output_root')
    args = parser.parse_args()
    run = Path(args.floor_run)
    stage = run / "stage2_work"
    primary_root = Path(args.primary_output_root)
    paired = []
    counts = {}
    for diagnostic, (relative, keys, expected) in STAGE2_TABLE_SPECS.items():
        primary = pd.read_csv(primary_root / relative)
        floor = pd.read_csv(stage / relative)
        if diagnostic in {"cyclone", "ice_edge"}:
            status_name = "cyclone_proximity_evidence_status.csv" if diagnostic == "cyclone" else "ice_edge_relative_wind_evidence_status.csv"
            primary_status = pd.read_csv((primary_root / relative).parent / status_name).set_index("region")["evidence_status"]
            floor_status = pd.read_csv((stage / relative).parent / status_name).set_index("region")["evidence_status"]
            primary["formal_evidence_status"] = primary["region"].map(primary_status)
            floor["formal_evidence_status"] = floor["region"].map(floor_status)
        counts[diagnostic] = {"primary": len(primary), "floor015": len(floor), "expected": expected}
        if len(primary) != expected or len(floor) != expected:
            raise RuntimeError(f"formal floor015 scope mismatch: {diagnostic} {counts[diagnostic]}")
        paired.append(_pair_evidence_table(diagnostic, primary, floor, keys))
    raw = pd.concat(paired, ignore_index=True, sort=False)
    if len(raw) != 770:
        raise RuntimeError(f"formal floor015 total={len(raw)}, expected=770")
    summary = pd.concat(
        [_stage2_region_summary(raw), _wrong_region_four_of_four_summary(primary_root, stage)],
        ignore_index=True,
    )
    raw.to_csv(run / "floor015_stage2_raw_evidence.csv", index=False)
    summary.to_csv(run / "floor015_stage2_region_summary.csv", index=False)
    metadata = {
        "scope": "CORRECTED_FLOOR015_FORMAL_STAGE2_ONLY",
        "canonical_longitude_geometry_reached": True,
        "primary_corrected_catalog_hash_lock": "stage2_primary_input_hash_lock.csv",
        "floor_none_raw_redetection": False,
        "stage3_executed": False,
        "ice_edge_backend": "process",
        "raw_rows": len(raw),
        "region_summary_rows": len(summary),
        "family_counts": counts,
        "primary_output_root": str(primary_root.resolve()),
        "output_dir": str(run.resolve()),
    }
    (run / "stage2_floor015_scope_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
