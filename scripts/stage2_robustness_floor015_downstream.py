#!/usr/bin/env python3
"""Run the isolated start-SIC floor=0.15 downstream robustness closure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vrile.robustness.downstream_floor015 import (
    run_downstream_floor015,
    run_stage2_floor015_only,
)


def parse_args() -> argparse.Namespace:
    """Parse the isolated robustness runner arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--stage2-only", action="store_true")
    parser.add_argument('--out_dir', type=Path)
    parser.add_argument('--primary_output_dir', type=Path, dest='primary_output_root')
    return parser.parse_args()


def main() -> int:
    """Execute the transaction and return a shell-compatible status code."""

    args = parse_args()
    if args.stage2_only:
        if args.out_dir is None or args.primary_output_root is None:
            raise SystemExit('--stage2-only requires --out_dir and --primary_output_dir')
        run_stage2_floor015_only(
            args.project_root.resolve(),
            output_dir=args.out_dir,
            primary_output_root=args.primary_output_root,
            workers=args.workers,
        )
    else:
        run_downstream_floor015(args.project_root.resolve(), workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
