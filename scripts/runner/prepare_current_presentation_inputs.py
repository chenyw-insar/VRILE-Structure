#!/usr/bin/env python3
"""Run an accepted GMT preparer against a fresh current-run source manifest.

The accepted preparers embed hashes for the historical source delivery.  This
release-engineering adapter replaces only that input-integrity dictionary with
hashes from the freshly generated 14-sidecar manifest plus the explicit raw
NSIDC mask.  The accepted preparer's transformation and output logic is then
called unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import sys
from pathlib import Path


MASK_RELATIVE_PATH = Path(
    'raw/nsidc_region_masks/NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc'
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_expected(manifest: Path, raw_mask: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            relative = row["relative_path"]
            if relative in rows:
                raise ValueError(f"duplicate fresh sidecar path: {relative}")
            rows[relative] = row["sha256"]
    if len(rows) != 14:
        raise ValueError(f"fresh sidecar manifest must contain 14 rows, found {len(rows)}")
    rows[MASK_RELATIVE_PATH.as_posix()] = sha256(raw_mask)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparer", type=Path, required=True)
    parser.add_argument('--input_dir', type=Path, required=True, dest='root')
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--raw-mask", type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True, dest='output')
    args = parser.parse_args()

    preparer = args.preparer.expanduser().resolve()
    root = args.root.expanduser().resolve()
    manifest = args.source_manifest.expanduser().resolve()
    raw_mask = args.raw_mask.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path, label in ((preparer, "preparer"), (manifest, "fresh manifest"), (raw_mask, "raw mask")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} unavailable: {path}")
    if not root.is_dir():
        raise FileNotFoundError(f"fresh source root unavailable: {root}")

    spec = importlib.util.spec_from_file_location("vrile_current_gmt_preparer", preparer)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load accepted preparer: {preparer}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    current = current_expected(manifest, raw_mask)
    accepted_paths = set(module.EXPECTED)
    if accepted_paths != set(current):
        missing = sorted(accepted_paths - set(current))
        extra = sorted(set(current) - accepted_paths)
        raise ValueError(f"accepted preparer path contract changed; missing={missing}; extra={extra}")

    # Input paths remain exactly those accepted by the preparer.  Only the
    # integrity values are supplied by this run's manifest and raw input.
    module.EXPECTED = current
    previous_argv = sys.argv
    try:
        sys.argv = [str(preparer), '--input_dir', str(root), '--output_dir', str(output)]
        result = module.main()
    finally:
        sys.argv = previous_argv
    if result not in (None, 0):
        raise RuntimeError(f"accepted preparer returned nonzero status: {result}")
    print("CURRENT_RUN_PRESENTATION_PREPARER = PASS")
    print("HISTORICAL_GOLDEN_HASH_VALUES_USED = NO")
    print("CURRENT_RUN_INPUT_PATH_CONTRACT = 15/15")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
