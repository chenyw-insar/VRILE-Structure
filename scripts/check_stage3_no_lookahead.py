#!/usr/bin/env python
"""Independent event-specific no-look-ahead audit for accepted corrected Stage3."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("VRILE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
PIPE = ROOT / '.'
STAGE3 = Path(
    os.environ.get(
        "VRILE_STAGE3_PRODUCTS_ROOT",
        ROOT / 'outputs/m021_corrected_stage3/products',
    )
).expanduser().resolve()
OUT = Path(
    os.environ.get(
        "VRILE_STAGE3_NO_LOOKAHEAD_OUT",
        ROOT / 'outputs/m022_no_lookahead',
    )
).expanduser().resolve()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


if Path.cwd().resolve() != ROOT.resolve():
    raise SystemExit("HOLD_CANONICAL_PROJECT_ROOT")
OUT.mkdir(parents=True, exist_ok=True)

paths = {
    "pan_events": PIPE / "outputs/reproduce_sie/vrile_events_unique_both_jja_5p.csv",
    "membership": PIPE / "data/processed/local_event_cells/unique_event_patch_membership.csv",
    "relations": STAGE3 / "panarctic_local_event_overlap.csv",
    "time_alignment": STAGE3 / "time_alignment_audit.csv",
    "event_summary": STAGE3 / "panarctic_event_overlap_summary.csv",
}
pan = pd.read_csv(paths["pan_events"], parse_dates=["date"])
pan.insert(0, "pan_event_id", [f"PAN{i:06d}" for i in range(1, len(pan) + 1)])
membership = pd.read_csv(paths["membership"])
membership["unique_local_event_id"] = membership["unique_local_event_id"].astype(str)
membership["object_id"] = membership["object_id"].astype(str)
membership["date"] = pd.to_datetime(membership["date"]).dt.normalize()
membership["start_date"] = pd.to_datetime(membership["start_date"]).dt.normalize()
relations = pd.read_csv(paths["relations"])
relations["unique_local_event_id"] = relations["unique_local_event_id"].astype(str)
alignment = pd.read_csv(paths["time_alignment"])
summary = pd.read_csv(paths["event_summary"])

rows = []
relation_detail = []
for event in pan.itertuples(index=False):
    pan_id = str(event.pan_event_id)
    t = pd.Timestamp(event.date).normalize()
    start = t - pd.Timedelta(days=5)
    rel = relations[relations["pan_event_id"].eq(pan_id)].copy()
    rel_ids = set(rel["unique_local_event_id"])

    # Reconstruct the exact patch rows consumed by production run_overlap from
    # the relation IDs and locked primary interval/no-look-ahead semantics.
    candidates = membership[
        membership["unique_local_event_id"].isin(rel_ids)
        & (membership["start_date"] <= t)
        & (membership["date"] >= start)
    ].copy()
    included = candidates[candidates["date"] <= t].copy()
    future_candidates = candidates[candidates["date"] > t].copy()

    relation_count_by_uid = rel.set_index("unique_local_event_id")["time_aligned_patch_count"].astype(int).to_dict()
    included_count_by_uid = included.groupby("unique_local_event_id")["object_id"].nunique().to_dict()
    per_uid_exact = all(included_count_by_uid.get(uid, 0) == relation_count_by_uid.get(uid, -1) for uid in rel_ids)
    uid_set_exact = set(included["unique_local_event_id"]) == rel_ids
    n_included = int(included["object_id"].nunique())
    n_relation_rows = int(sum(relation_count_by_uid.values()))
    align_row = alignment[alignment["pan_event_id"].eq(pan_id)].iloc[0]
    summary_row = summary[summary["pan_event_id"].eq(pan_id)].iloc[0]
    count_closure = (
        n_included == n_relation_rows == int(align_row["primary_no_lookahead_patch_count"])
        and len(rel_ids) == int(align_row["primary_no_lookahead_candidate_count"])
        and len(rel_ids) == int(summary_row["n_temporal_candidates"])
    )
    max_date = included["date"].max() if not included.empty else pd.NaT
    min_date = included["date"].min() if not included.empty else pd.NaT
    max_after = int((max_date - t).days) if pd.notna(max_date) else 0
    n_future_included = int((included["date"] > t).sum())
    passed = n_future_included == 0 and max_after <= 0 and per_uid_exact and uid_set_exact and count_closure
    rows.append({
        "pan_event_id": pan_id,
        "T": t.strftime("%Y-%m-%d"),
        "n_included_patches": n_included,
        "min_included_patch_date": min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else "",
        "max_included_patch_date": max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else "",
        "n_future_included_patches": n_future_included,
        "max_days_after_T": max_after,
        "relation_patch_count_closure": bool(count_closure),
        "per_ULE_patch_count_closure": bool(per_uid_exact),
        "relation_ULE_set_closure": bool(uid_set_exact),
        "recorded_future_patch_excluded_count": int(align_row["future_patch_excluded_count"]),
        "independently_reconstructed_future_candidate_count": int(len(future_candidates)),
        "pass_no_look_ahead": bool(passed),
    })
    for patch in included.itertuples(index=False):
        relation_detail.append({
            "pan_event_id": pan_id, "T": t.strftime("%Y-%m-%d"),
            "unique_local_event_id": str(patch.unique_local_event_id),
            "object_id": str(patch.object_id),
            "patch_date": pd.Timestamp(patch.date).strftime("%Y-%m-%d"),
            "patch_start_date": pd.Timestamp(patch.start_date).strftime("%Y-%m-%d"),
            "days_after_T": int((pd.Timestamp(patch.date) - t).days),
        })

audit = pd.DataFrame(rows)
detail = pd.DataFrame(relation_detail)
audit.to_csv(OUT / "STAGE3_NO_LOOKAHEAD_EVENT_AUDIT.csv", index=False)
detail.to_csv(OUT / "STAGE3_NO_LOOKAHEAD_INCLUDED_PATCH_RELATIONS.csv", index=False)

global_future = int(audit["n_future_included_patches"].sum())
global_max = int(audit["max_days_after_T"].max())
included_total = int(audit["n_included_patches"].sum())
recorded_exclusions = int(audit["recorded_future_patch_excluded_count"].sum())
reconstructed_future_candidates = int(audit["independently_reconstructed_future_candidate_count"].sum())
status = "PASS_NO_LOOKAHEAD_QA_CLOSED" if len(audit) == 99 and audit["pass_no_look_ahead"].all() and global_future == 0 and global_max <= 0 else "FAIL_FUTURE_PATCH_INCLUDED"
payload = {
    "status": status, "events_audited": int(len(audit)),
    "included_patch_event_relations": included_total,
    "future_included_relations": global_future,
    "maximum_included_days_after_T": global_max,
    "recorded_future_exclusions": recorded_exclusions,
    "independently_reconstructed_future_candidates": reconstructed_future_candidates,
    "all_events_have_exclusion_record": bool(audit["recorded_future_patch_excluded_count"].notna().all()),
    "input_sha256": {name: sha(path) for name, path in paths.items()},
}
(OUT / "no_lookahead_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
if status != "PASS_NO_LOOKAHEAD_QA_CLOSED":
    raise SystemExit(status)
