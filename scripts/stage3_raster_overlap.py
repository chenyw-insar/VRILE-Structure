#!/usr/bin/env python
"""Compute Batch 2 pan-Arctic/local raster overlaps."""

from __future__ import annotations

import sys
import os
import json
import platform
from pathlib import Path

import pandas as pd

_THIS = Path(__file__).resolve()
_SRC = _THIS.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vrile.stage3.grid import LOCAL_MAJOR, LOCAL_SEVERE, LOCAL_UNIQUE, OUT_DIR, ROOT, load_pan_events, sha256_file
from vrile.stage3.overlap import run_overlap


def read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> int:
    pan = load_pan_events()
    cells_dir = ROOT / "data/processed/local_event_cells"
    patch_index = pd.read_csv(cells_dir / "patch_cell_index.csv")
    membership = pd.read_csv(cells_dir / "unique_event_patch_membership.csv")
    unique = pd.read_csv(LOCAL_UNIQUE)
    severe = read_optional(LOCAL_SEVERE)
    major = read_optional(LOCAL_MAJOR)
    event_df, relation_df, pair_df, alignment_df = run_overlap(
        pan,
        membership,
        patch_index,
        unique,
        severe,
        major,
        cells_dir,
        ROOT / "data/processed/panarctic_loss_fields",
        ROOT / "data/processed/panarctic_detector_extent_fields",
    )
    event_df.to_csv(OUT_DIR / "panarctic_event_overlap_summary.csv", index=False)
    relation_df.to_csv(OUT_DIR / "panarctic_local_event_overlap.csv", index=False)
    pair_df.to_csv(OUT_DIR / "panarctic_local_component_overlap_pairs.csv", index=False)
    alignment_df.to_csv(OUT_DIR / "time_alignment_audit.csv", index=False)
    if len(event_df) != 99 or event_df["pan_event_id"].nunique() != 99:
        raise SystemExit("Pan-event overlap summary must have exactly 99 unique pan_event_id")
    if len(alignment_df) != 99 or alignment_df["pan_event_id"].nunique() != 99:
        raise SystemExit("Time-alignment audit must have exactly 99 unique pan_event_id")
    if not (event_df["n_temporal_candidates"] == event_df["n_broad_candidates"] + event_df["n_severe_candidates"] + event_df["n_major_severe_candidates"]).all():
        raise SystemExit("Temporal candidate hierarchy count gate failed")
    route_path = OUT_DIR / "stage3_route_decision.json"
    route = json.loads(route_path.read_text()).get("route", "unknown")
    files_for_hash = [
        "scripts/stage3_panarctic_local_contribution.py",
        "scripts/stage3_detector_extent_fields.py",
        "scripts/stage3_local_footprints.py",
        "scripts/stage3_cdr_loss_fields.py",
        "scripts/stage3_raster_overlap.py",
        "src/vrile/local_objects.py",
        "src/vrile/stage3/grid.py",
        "src/vrile/stage3/fields.py",
        "src/vrile/stage3/footprints.py",
        "src/vrile/stage3/overlap.py",
    ]
    meta = {
        "active_conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": platform.python_version(),
        "route": route,
        "plan_sha256": sha256_file(ROOT / 'method_static/provenance/VRILE_Stage3_PanArctic_Local_Contribution_Experiment_Plan.md'),
        "batch2_prompt_status": "found" if (ROOT / "VRILE_Stage3_Batch2_Codex_Prompt.md").exists() else "not_present",
        "batch2_prompt_sha256": sha256_file(ROOT / "VRILE_Stage3_Batch2_Codex_Prompt.md") if (ROOT / "VRILE_Stage3_Batch2_Codex_Prompt.md").exists() else "",
        "foundation_code_sha256": sha256_file(ROOT / "scripts/stage3_panarctic_local_contribution.py"),
        "batch2_script_sha256s": {p: sha256_file(ROOT / p) for p in files_for_hash if p.startswith("scripts/")},
        "stage3_package_sha256s": {p: sha256_file(ROOT / p) for p in files_for_hash if p.startswith("src/")},
        "footprint_strategy": json.loads((OUT_DIR / "footprint_strategy_audit.json").read_text()).get("selected_plan", ""),
        "event_count": int(len(pan)),
        "patch_count": int(len(patch_index)),
        "required_geotiff_count": int(len(pd.read_csv(OUT_DIR / "g02135_geotiff_grid_signature_audit.csv"))),
    }
    (OUT_DIR / "batch2_run_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    batch2r_prompt = ROOT / "VRILE_Stage3_Batch2R_Codex_Prompt.md"
    modified_files = files_for_hash + [
        "run_stage3_panarctic_local_contribution.sh",
        "scripts/stage1_detect_local_sic_loss_objects.py",
        "scripts/stage1_consolidate_local_objects.py",
    ]
    meta2r = {
        "active_conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": platform.python_version(),
        "route": route,
        "plan_sha256": sha256_file(ROOT / 'method_static/provenance/VRILE_Stage3_PanArctic_Local_Contribution_Experiment_Plan.md'),
        "batch2r_prompt_status": "found" if batch2r_prompt.exists() else "not_present",
        "batch2r_prompt_sha256": sha256_file(batch2r_prompt) if batch2r_prompt.exists() else "",
        "modified_code_sha256s": {p: sha256_file(ROOT / p) for p in modified_files if (ROOT / p).exists()},
        "baseline_pan_local_rows": 2806,
        "baseline_major_severe_rows": 382,
        "baseline_component_pairs": 5437,
        "new_pan_local_rows": int(len(relation_df)),
        "new_major_severe_rows": int((relation_df["event_level"] == "major_severe").sum()) if not relation_df.empty else 0,
        "new_component_pairs": int(len(pair_df)),
        "events_affected_by_alignment_repair": int((alignment_df["union_footprint_cell_reduction"] > 0).sum()),
        "future_patches_excluded": int(alignment_df["future_patch_excluded_count"].sum()),
    }
    (OUT_DIR / "batch2r_run_metadata.json").write_text(json.dumps(meta2r, indent=2, sort_keys=True), encoding="utf-8")
    print(f"WROTE {OUT_DIR}")
    print(f"pan_events={len(event_df)} pan_local_rows={len(relation_df)} component_pairs={len(pair_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
