#!/usr/bin/env python
"""Build Track S 5-day CDR SIC loss fields and loss components."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import hashlib
import os

import numpy as np
import pandas as pd
import xarray as xr

_THIS = Path(__file__).resolve()
_SRC = _THIS.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vrile.stage3.fields import cdr_loss_field
from vrile.stage3.grid import OUT_DIR, ROOT, load_pan_events
from common_utils_parallel import run_with_fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--backend", choices=["process", "thread", "serial"], default="process")
    parser.add_argument("--disable-parallel", action="store_true")
    return parser


def _cdr_field_task(task: dict) -> tuple[dict, list[dict]]:
    out_dir = Path(task["out_dir"]) if task.get("out_dir") else None
    return cdr_loss_field(task["event"], out_dir)


def _validation_root() -> Path | None:
    value = os.environ.get("STAGE3_VALIDATION_ROOT", "").strip()
    return ROOT / value if value and not Path(value).is_absolute() else (Path(value) if value else None)


def _array_sha256_int32(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype="<i4").tobytes()).hexdigest()


def write_field_validation_audits(field_dir: Path, summary_df: pd.DataFrame) -> None:
    root = _validation_root()
    if root is None:
        return
    root.mkdir(parents=True, exist_ok=True)
    dtype_rows = []
    hash_rows = []
    for pan_id in summary_df["pan_event_id"].astype(str):
        with xr.open_dataset(field_dir / f"{pan_id}.nc") as ds:
            dtype_rows.append(
                {
                    "pan_event_id": pan_id,
                    "cdr_signed_sic_change_dtype": str(ds["cdr_signed_sic_change"].dtype),
                    "cdr_sic_loss_dtype": str(ds["cdr_sic_loss"].dtype),
                    "cdr_sic_gain_dtype": str(ds["cdr_sic_gain"].dtype),
                    "loss_component_id_dtype": str(ds["loss_component_id"].dtype),
                }
            )
            hash_rows.append(
                {
                    "pan_event_id": pan_id,
                    "new_sha256": _array_sha256_int32(np.asarray(ds["loss_component_id"].values, dtype=np.int32)),
                }
            )
    pd.DataFrame(dtype_rows).to_csv(root / "cdr_field_dtype_audit.csv", index=False)
    baseline = root / "component_membership_baseline.csv"
    if baseline.exists():
        base = pd.read_csv(baseline)
        cmp = base.merge(pd.DataFrame(hash_rows), on="pan_event_id", how="outer")
        cmp["hash_match"] = cmp["sha256"].astype(str).eq(cmp["new_sha256"].astype(str))
        cmp.to_csv(root / "component_membership_hash_comparison.csv", index=False)
        if not cmp["hash_match"].all():
            raise SystemExit("Track S component membership hash changed after field repair")


def main() -> int:
    args = build_parser().parse_args()
    events = load_pan_events()
    field_dir = ROOT / "data/processed/panarctic_loss_fields"
    tasks = [{"event": row, "out_dir": str(field_dir)} for row in events.to_dict("records")]
    use_parallel = args.parallel and not args.disable_parallel
    print(f"parallel={use_parallel} backend={args.backend} workers={args.workers} events={len(tasks)}")
    results = run_with_fallback(
        _cdr_field_task,
        tasks,
        workers=args.workers,
        parallel=use_parallel,
        backend=args.backend,
    )
    summaries = []
    components = []
    for summary, comp_rows in results:
        summaries.append(summary)
        components.extend(comp_rows)
    summary_df = pd.DataFrame(summaries)
    comp_df = pd.DataFrame(components)
    summary_df.to_csv(OUT_DIR / "panarctic_loss_field_event_summary.csv", index=False)
    comp_df.to_csv(OUT_DIR / "panarctic_loss_components.csv", index=False)
    if len(summary_df) != 99 or summary_df["pan_event_id"].nunique() != 99:
        raise SystemExit("CDR loss field event summary must have exactly 99 unique pan_event_id")
    if not summary_df["cdr_budget_closure_status"].eq("PASS").all():
        raise SystemExit("Track S SIC budget closure failed")
    if not summary_df["component_loss_closure_status"].eq("PENDING_INDEPENDENT_RECONCILIATION").all():
        raise SystemExit("Track S component-loss closure must remain pending until independent raster/table reconciliation")
    if not summary_df["component_resolved_sic_loss_fraction"].between(0.0, 1.0).all():
        raise SystemExit("Track S component_resolved_sic_loss_fraction outside [0, 1]")
    if len(comp_df) != int(summary_df["n_loss_components"].sum()):
        raise SystemExit("Track S component row count does not match n_loss_components")
    write_field_validation_audits(field_dir, summary_df)
    print(f"WROTE {field_dir}")
    print(f"events={len(summary_df)} components={len(comp_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
