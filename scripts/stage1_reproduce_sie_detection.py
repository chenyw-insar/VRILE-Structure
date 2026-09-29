#!/usr/bin/env python
"""Reproduce the SIE-based VRILE detection experiment on local data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_this_dir = Path(__file__).resolve()
_src_dir = _this_dir.parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from vrile import DetectionConfig, detect_vriles, write_event_table
from vrile.months import months_label, parse_months


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/data/raw/nsidc_sie/N_seaice_extent_daily_v4.0.csv",
        help="Daily Northern Hemisphere Sea Ice Index CSV.",
    )
    parser.add_argument(
        '--out_dir',
        default="outputs/reproduce_sie",
        help="Directory for event tables and diagnostics.",
    )
    parser.add_argument("--months", default="jja", help="'jja', 'annual', or comma-separated month numbers such as 6,7,8.")
    parser.add_argument("--method", default="both", choices=["mean_removed", "butterworth", "both"])
    parser.add_argument("--unique-gap-days", type=int, default=1)
    parser.add_argument("--start-year", type=int, default=1989)
    parser.add_argument(
        "--end-year",
        type=int,
        default=2023,
        help="Default keeps the original reproduced experiment period. Use 2025 for updated runs.",
    )
    return parser


def make_config(months: tuple[int, ...], unique_gap_days: int, start_year: int, end_year: int) -> DetectionConfig:
    return DetectionConfig(
        start_year=start_year,
        end_year=end_year,
        climatology_start_year=1990,
        climatology_end_year=2018,
        delta_days=3,
        percentile=5.0,
        butterworth_cutoff_days=18.0,
        butterworth_order=12,
        unique_gap_days=unique_gap_days,
        months=months,
        exclude_month_boundary=True,
    )


def main() -> int:
    args = build_parser().parse_args()
    months = parse_months(args.months)
    label = months_label(months)
    all_config = make_config(months, unique_gap_days=0, start_year=args.start_year, end_year=args.end_year)
    unique_config = make_config(months, unique_gap_days=args.unique_gap_days, start_year=args.start_year, end_year=args.end_year)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events_all, diagnostics = detect_vriles(args.input, all_config, method=args.method)
    events_unique, _ = detect_vriles(args.input, unique_config, method=args.method)
    all_events_path = out_dir / f"vrile_events_all_{args.method}_{label}_5p.csv"
    unique_events_path = out_dir / f"vrile_events_unique_{args.method}_{label}_5p.csv"
    write_event_table(events_all, all_events_path)
    write_event_table(events_unique, unique_events_path)
    write_event_table(events_unique, out_dir / f"vrile_events_{args.method}_{label}_5p.csv")
    diagnostics["daily"].to_csv(out_dir / "daily_sie_processed.csv", index=False)
    diagnostics["daily_climatology"].to_csv(out_dir / "daily_delta_climatology.csv", index=False)
    diagnostics["monthly_climatology"].to_csv(out_dir / "monthly_delta_climatology.csv", index=False)

    metadata = {
        "input": args.input,
        "months": list(months),
        "months_label": label,
        "all_events_config": all_config.__dict__,
        "unique_events_config": unique_config.__dict__,
        "method": args.method,
        "n_events_all": int(len(events_all)),
        "counts_by_method_all": events_all["method"].value_counts().to_dict() if not events_all.empty else {},
        "n_events_unique": int(len(events_unique)),
        "counts_by_method_unique": events_unique["method"].value_counts().to_dict() if not events_unique.empty else {},
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(events_unique.head(20).to_string(index=False))
    print(f"N_EVENTS_ALL {len(events_all)}")
    if not events_all.empty:
        print(events_all["method"].value_counts().to_string())
    print(f"N_EVENTS_UNIQUE {len(events_unique)}")
    if not events_unique.empty:
        print(events_unique["method"].value_counts().to_string())
    print(f"WROTE {all_events_path}")
    print(f"WROTE {unique_events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
