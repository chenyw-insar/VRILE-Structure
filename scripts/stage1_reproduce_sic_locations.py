#!/usr/bin/env python
"""Locate SIC-loss objects for reproduced VRILE events using local daily files."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

_this_dir = Path(__file__).resolve()
_src_dir = _this_dir.parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from vrile.io import locate_event_object
from common_utils_parallel import run_with_fallback


def find_sic_file(root: Path, date: pd.Timestamp) -> Path:
    pattern = root / f"{date.year}" / f"sic_psn25_{date:%Y%m%d}_*.nc"
    matches = sorted(glob.glob(str(pattern)))
    if not matches:
        raise FileNotFoundError(f"No SIC file matched {pattern}")
    return Path(matches[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        default="outputs/reproduce_sie/vrile_events_both_jja_5p.csv",
        help="VRILE event table produced by stage1_reproduce_sie_detection.py.",
    )
    parser.add_argument(
        '--sic_dir',
        default="data/raw/nsidc_sic",
        help="Root directory containing year subdirectories of SIC NetCDF files.",
    dest='sic_root')
    parser.add_argument(
        '--output',
        default="outputs/reproduce_sie/vrile_locations.csv",
        help="Output CSV for SIC-loss object locations.",
    dest='output')
    parser.add_argument("--region", default="pan_arctic")
    parser.add_argument("--window-days", type=int, default=5)
    parser.add_argument("--loss-threshold", type=float, default=-0.1)
    parser.add_argument("--min-object-cells", type=int, default=4)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--backend", choices=["process", "thread", "serial"], default="process")
    parser.add_argument("--disable-parallel", action="store_true")
    return parser


def _locate_event_task(task: dict) -> dict:
    row = task["row"]
    event_date = pd.Timestamp(row["date"])
    start_date = event_date - pd.Timedelta(days=task["window_days"])
    try:
        sic_root = Path(task["sic_root"])
        files = [find_sic_file(sic_root, start_date), find_sic_file(sic_root, event_date)]
        location = locate_event_object(
            files,
            event_date,
            region=task["region"],
            sic_var="cdr_seaice_conc",
            window_days=task["window_days"],
            loss_threshold=task["loss_threshold"],
            min_object_cells=task["min_object_cells"],
        )
    except Exception as exc:
        # Missing or invalid event inputs are represented in the output table,
        # matching the established serial behavior.
        location = {
            "date": event_date.strftime("%Y-%m-%d"),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "region": task["region"],
            "found": False,
            "message": str(exc),
        }
    location["method"] = row.get("method")
    location["raw_delta_sie"] = row.get("raw_delta_sie")
    location["vrile_value"] = row.get("vrile_value")
    return location


def main() -> int:
    args = build_parser().parse_args()
    events = pd.read_csv(args.events, parse_dates=["date"])
    tasks = [
        {
            "row": row,
            "sic_root": args.sic_root,
            "region": args.region,
            "window_days": args.window_days,
            "loss_threshold": args.loss_threshold,
            "min_object_cells": args.min_object_cells,
        }
        for row in events.to_dict("records")
    ]
    use_parallel = args.parallel and not args.disable_parallel
    print(f"parallel={use_parallel} backend={args.backend} workers={args.workers} tasks={len(tasks)}")
    rows = run_with_fallback(
        _locate_event_task,
        tasks,
        workers=args.workers,
        parallel=use_parallel,
        backend=args.backend,
    )
    for i, location in enumerate(rows, start=1):
        print(f"{i}/{len(rows)} {location.get('date')} found={location.get('found')}")

    out = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"WROTE {output_path}")
    print(out["found"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
