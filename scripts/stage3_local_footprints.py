#!/usr/bin/env python
"""Backfill sparse local patch footprints and unique-event membership."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_THIS = Path(__file__).resolve()
_SRC = _THIS.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vrile.stage3.footprints import filtered_primary_objects, reconstruct_patch_cells, restore_unique_membership
from vrile.stage3.grid import LOCAL_FILTERED, LOCAL_UNIQUE, OUT_DIR, PROCESSED_DATA_ROOT


def main() -> int:
    filtered = filtered_primary_objects(LOCAL_FILTERED)
    canonical = pd.read_csv(LOCAL_UNIQUE, parse_dates=["event_start", "event_end"])
    out_dir = PROCESSED_DATA_ROOT / "local_event_cells"
    index, recon = reconstruct_patch_cells(filtered, out_dir, OUT_DIR / "footprint_detector_reconciliation.csv")
    if len(recon) != len(filtered) or len(index) != len(filtered):
        raise SystemExit(f"Footprint reconciliation coverage failed: filtered={len(filtered)} recon={len(recon)} index={len(index)}")
    if set(index["object_id"].astype(str)) != set(filtered["object_id"].astype(str)):
        raise SystemExit("Footprint reconstruction object_id set does not match authoritative filtered objects")
    if not recon["status"].eq("PASS").all():
        raise SystemExit("Footprint strict reconciliation has FAIL rows")
    membership, membership_rec = restore_unique_membership(
        filtered,
        canonical,
        out_dir,
        OUT_DIR / "unique_event_membership_reconciliation.csv",
    )
    print(f"WROTE {out_dir}")
    print(f"patches={len(index)} unique_events={membership['unique_local_event_id'].nunique()} recon={recon['status'].value_counts().to_dict()}")
    print(membership_rec.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
