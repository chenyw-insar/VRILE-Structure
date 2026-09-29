#!/usr/bin/env python
"""Build Track D detector-aligned extent fields and budget closure audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_THIS = Path(__file__).resolve()
_SRC = _THIS.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vrile.stage3.fields import detector_extent_field
from vrile.stage3.grid import OUT_DIR, ROOT, load_pan_events, write_g02135_signature_audit
from common_utils_parallel import run_with_fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--backend", choices=["process", "thread", "serial"], default="process")
    parser.add_argument("--disable-parallel", action="store_true")
    return parser


def _detector_field_task(task: dict) -> dict:
    out_dir = Path(task["out_dir"]) if task.get("out_dir") else None
    return detector_extent_field(task["event"], out_dir)


def main() -> int:
    args = build_parser().parse_args()
    events = load_pan_events()
    write_g02135_signature_audit(events, OUT_DIR / "g02135_geotiff_grid_signature_audit.csv")
    field_dir = ROOT / "data/processed/panarctic_detector_extent_fields"
    ranked = events.sort_values("raw_delta_sie").reset_index(drop=True)
    idx = sorted(set([0, 1, 2, len(ranked) // 4, len(ranked) // 2, (3 * len(ranked)) // 4, len(ranked) - 3, len(ranked) - 2, len(ranked) - 1]))
    smoke = ranked.iloc[idx].copy()
    smoke = pd.concat([smoke, events[pd.to_datetime(events["date"]).dt.year.eq(2025)].head(1)], ignore_index=True).drop_duplicates("pan_event_id").head(10)
    smoke_rows = [detector_extent_field(row, None) for _, row in smoke.iterrows()]
    if not pd.DataFrame(smoke_rows)["status"].eq("PASS").all():
        raise SystemExit("Track D smoke budget closure failed")
    tasks = [{"event": row, "out_dir": str(field_dir)} for row in events.to_dict("records")]
    use_parallel = args.parallel and not args.disable_parallel
    print(f"parallel={use_parallel} backend={args.backend} workers={args.workers} events={len(tasks)}")
    rows = run_with_fallback(
        _detector_field_task,
        tasks,
        workers=args.workers,
        parallel=use_parallel,
        backend=args.backend,
    )
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT_DIR / "detector_extent_budget_closure_audit.csv", index=False)
    if len(audit) != 99 or not audit["status"].eq("PASS").all():
        raise SystemExit("Detector extent budget closure gate failed")
    d1a_path = OUT_DIR / "panarctic_detector_source_closure_audit.csv"
    if d1a_path.exists():
        d1a = pd.read_csv(d1a_path)
        joined = audit[["pan_event_id", "detector_geotiff_delta_sie_km2"]].merge(
            d1a[["pan_event_id", "geotiff_reconstructed_delta_sie"]],
            on="pan_event_id",
            how="inner",
        )
        ok = (
            len(joined) == 99
            and (joined["detector_geotiff_delta_sie_km2"] / 1_000_000.0).sub(joined["geotiff_reconstructed_delta_sie"]).abs().le(1e-12).all()
        )
        if not ok:
            raise SystemExit("Track D detector field reconstruction does not match D1a geotiff_reconstructed_delta_sie")
    print(f"WROTE {field_dir}")
    print(audit["status"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
