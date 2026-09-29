"""Frozen Stage 3 output guard and candidate-root selection."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path


FROZEN_WRITER_SCRIPTS = frozenset(
    {
        "stage3_panarctic_local_contribution.py",
        "stage3_detector_extent_fields.py",
        "stage3_local_footprints.py",
        "stage3_local_cells_preflight.py",
        "stage3_cdr_loss_fields.py",
        "stage3_batch3_preflight.py",
        "stage3_raster_overlap.py",
        "stage3_contribution_budget.py",
        "stage3_spatial_composition_metrics.py",
        "stage3_coloss_controls.py",
        "stage3_reverse_analysis.py",
        "stage3_ice_motion_coverage.py",
        "stage3_finalize.py",
    }
)
ALLOW_FLAG = "--allow-regenerate-frozen"
OUTPUT_ENV = "VRILE_STAGE3_OUTPUT_ROOT"
TRANSACTION_ENV = "STAGE3_TRANSACTION_ROOT"


def _frozen_status(root: Path) -> tuple[str, str, str]:
    path = root / "outputs/panarctic_local_contribution/stage3_final_status.json"
    if not path.is_file():
        return "UNKNOWN", "UNKNOWN", "UNKNOWN"
    value = json.loads(path.read_text(encoding="utf-8"))
    return (
        str(value.get("stage3_core_status", "UNKNOWN")),
        str(value.get("stage3_freeze_status", "UNKNOWN")),
        str(value.get("stage3_lifecycle_status", "UNKNOWN")),
    )


def frozen_output_root(root: Path) -> Path:
    """Return production output root or activate an isolated candidate root."""

    project_root = root.resolve()
    script = Path(sys.argv[0]).name
    configured = os.environ.get(OUTPUT_ENV, "").strip()
    if script not in FROZEN_WRITER_SCRIPTS:
        if configured:
            return Path(configured).expanduser().resolve()
        output_base = Path(os.environ.get("VRILE_OUTPUT_ROOT", project_root / "outputs")).expanduser().resolve()
        return output_base / "panarctic_local_contribution"
    flag_present = ALLOW_FLAG in sys.argv
    while ALLOW_FLAG in sys.argv:
        sys.argv.remove(ALLOW_FLAG)
    if not flag_present:
        core, freeze, lifecycle = _frozen_status(project_root)
        raise SystemExit(
            "Stage 3 scientific outputs are frozen and read-only "
            f"(core={core}, freeze={freeze}, lifecycle={lifecycle}). "
            f"Use {ALLOW_FLAG} to generate an isolated transaction candidate."
        )

    transaction = os.environ.get(TRANSACTION_ENV, "").strip()
    if transaction:
        transaction_path = Path(transaction)
        if not transaction_path.is_absolute():
            transaction_path = project_root / transaction_path
        transaction_root = transaction_path.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transaction_root = project_root / f"_stage3_transaction_{stamp}_{os.getpid()}"

    if transaction_root.parent != project_root or not transaction_root.name.startswith("_stage3_transaction_"):
        raise SystemExit("HOLD_INVALID_TRANSACTION_ROOT: transaction root must be an _stage3_transaction_* directory directly under the project root")

    lexical_output = transaction_root / "candidate/outputs/panarctic_local_contribution"
    output = lexical_output.resolve()
    if output != Path(os.path.abspath(lexical_output)):
        raise SystemExit("HOLD_INVALID_CANDIDATE_ROOT: candidate path contains a symlink escape")
    if configured:
        configured_path = Path(configured)
        if not configured_path.is_absolute():
            configured_path = project_root / configured_path
        if configured_path.resolve() != output:
            raise SystemExit("HOLD_INVALID_CANDIDATE_ROOT: VRILE_STAGE3_OUTPUT_ROOT does not match the guarded transaction candidate")

    output_base = Path(os.environ.get("VRILE_OUTPUT_ROOT", project_root / "outputs")).expanduser().resolve()
    production = (output_base / "panarctic_local_contribution").resolve()
    if output == production or production in output.parents:
        raise SystemExit("HOLD_INVALID_CANDIDATE_ROOT: candidate output must not be inside production outputs")
    os.environ[TRANSACTION_ENV] = str(transaction_root)
    os.environ[OUTPUT_ENV] = str(output)
    output.mkdir(parents=True, exist_ok=True)
    return output
