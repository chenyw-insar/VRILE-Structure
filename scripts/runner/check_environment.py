#!/usr/bin/env python3
"""Check the scientific environment's exact pip overlay; no scientific imports.

The comparison is extracted unchanged from the tested post-generation tool.
This check is a runtime installation prerequisite, not a paper-result test.
"""
import argparse
import json
import re
from pathlib import Path


def pip_overlay(args):
    expected = {}
    for raw in args.requirements.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        expected[re.sub(r"[-_.]+", "-", name).lower()] = version
    observed = {
        re.sub(r"[-_.]+", "-", row["name"]).lower(): row["version"]
        for row in json.loads(args.conda_list.read_text(encoding="utf-8"))
        if row.get("channel") == "pypi"
    }
    if observed != expected:
        raise SystemExit(f"HOLD_SCIENCE_PIP_OVERLAY_MISMATCH:expected={sorted(expected.items())}:observed={sorted(observed.items())}")
    print("SCIENCE_PIP_OVERLAY=PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--conda-list", type=Path, required=True)
    pip_overlay(parser.parse_args())
