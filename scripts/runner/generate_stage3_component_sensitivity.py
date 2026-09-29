#!/usr/bin/env python3
"""Generate the Stage-3 3x3 component-sensitivity tables from current-run inputs.

This release-engineering entrypoint deliberately imports the bundled scientific
implementation instead of reimplementing it.  It consumes only an explicit
current-run Stage-3 product directory and an explicit external raw-data root.
Frozen evidence is neither located nor read by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


PRODUCT_NAMES = (
    "panarctic_loss_components.csv",
    "panarctic_loss_field_event_summary.csv",
    "panarctic_spatial_composition_metrics.csv",
)
EXPECTED_FLOORS = {"none", "0.15", "0.3"}
EXPECTED_MIN_CELLS = {4, 20, 50}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--stage3_products_dir',
        type=Path,
        required=True,
        help="Current-run M021 products directory.",
    )
    parser.add_argument(
        '--raw_dir',
        type=Path,
        required=True,
        help="External read-only raw-data root containing nsidc_sic and NSIDC ancillary files.",
    dest='raw_data_root')
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help="New current-run output directory; an existing path is refused.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"HOLD_STAGE3_SENSITIVITY_{message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    package_root = Path(__file__).resolve().parents[2]
    source_root = package_root / '.' / "src"
    products = args.stage3_products_dir.expanduser().resolve()
    raw = args.raw_data_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()

    require(source_root.is_dir(), f"BUNDLED_SOURCE_ROOT_MISSING:{source_root}")
    require(products.is_dir(), f"M021_PRODUCTS_DIR_MISSING:{products}")
    require(raw.is_dir(), f"RAW_DATA_ROOT_MISSING:{raw}")
    require(not output.exists(), f"OUTPUT_ALREADY_EXISTS:{output}")
    require(output != products and products not in output.parents, "OUTPUT_INSIDE_M021_PRODUCTS")
    require(output != raw and raw not in output.parents, "OUTPUT_INSIDE_RAW_DATA")
    for name in PRODUCT_NAMES:
        require((products / name).is_file(), f"M021_PRODUCT_MISSING:{name}")
    require((raw / "nsidc_sic").is_dir(), "NSIDC_SIC_ROOT_MISSING")
    require(
        (raw / "nsidc_ancillary" / "NSIDC0771_CellArea_PS_N25km_v1.1.nc").is_file(),
        "NSIDC0771_CELL_AREA_MISSING",
    )
    require(
        (raw / "nsidc_region_masks" / "NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc").is_file(),
        "NSIDC0780_REGION_MASK_MISSING",
    )

    # stage3.grid binds its raw-data root at import time.  Set the explicit
    # current invocation's external root before importing the bundled modules.
    os.environ["VRILE_RAW_DATA_ROOT"] = str(raw)
    sys.path.insert(0, str(source_root))
    from vrile.robustness import lowsic_core as core  # noqa: PLC0415

    expected_core = source_root / "vrile" / "robustness" / "lowsic_core.py"
    require(Path(core.__file__).resolve() == expected_core.resolve(), "UNBUNDLED_LOWSIC_CORE_IMPORT")

    output.parent.mkdir(parents=True, exist_ok=True)
    candidate = output.with_name(f".{output.name}_candidate_{os.getpid()}")
    require(not candidate.exists(), f"CANDIDATE_ALREADY_EXISTS:{candidate}")
    candidate.mkdir()
    try:
        # lowsic_core accepts one baseline root.  This temporary view maps only
        # its two required namespaces to explicit current-run/external inputs.
        with tempfile.TemporaryDirectory(prefix="baseline_view_", dir=candidate) as temp_name:
            view = Path(temp_name)
            (view / "data").mkdir()
            (view / "outputs").mkdir()
            (view / "data" / "raw").symlink_to(raw, target_is_directory=True)
            (view / "outputs" / "panarctic_local_contribution").symlink_to(
                products, target_is_directory=True
            )
            _, event_level, _, primary_events = core.run_stage3(view, quick=False)

        import pandas as pd  # noqa: PLC0415

        primary_metrics = pd.read_csv(products / "panarctic_spatial_composition_metrics.csv")
        require(len(primary_events) == 99, f"PAN_EVENT_ROWS:{len(primary_events)}")
        require(primary_events["pan_event_id"].astype(str).nunique() == 99, "PAN_EVENT_IDS")
        require(len(primary_metrics) == 99, f"PRIMARY_METRIC_ROWS:{len(primary_metrics)}")
        require(
            primary_metrics["pan_event_id"].astype(str).nunique() == 99,
            "PRIMARY_METRIC_EVENT_IDS",
        )
        primary_median = float(primary_metrics["effective_component_number_sic"].median())
        matrix = core.summarize_stage3(event_level, primary_median)

        event_floors = set(event_level["sic_floor"].astype(str))
        event_min_cells = set(event_level["strict_min_cells"].astype(int))
        require(len(event_level) == 891, f"EVENT_LEVEL_ROWS:{len(event_level)}")
        require(event_floors == EXPECTED_FLOORS, f"EVENT_LEVEL_FLOORS:{sorted(event_floors)}")
        require(
            event_min_cells == EXPECTED_MIN_CELLS,
            f"EVENT_LEVEL_MIN_CELLS:{sorted(event_min_cells)}",
        )
        require(
            not event_level.duplicated(
                ["pan_event_id", "sic_floor", "strict_min_cells"]
            ).any(),
            "DUPLICATE_EVENT_SCENARIO_ROWS",
        )
        scenario_counts = event_level.groupby(
            ["sic_floor", "strict_min_cells"], dropna=False
        ).size()
        require(len(scenario_counts) == 9, f"EVENT_LEVEL_SCENARIOS:{len(scenario_counts)}")
        require(scenario_counts.eq(99).all(), "EVENT_LEVEL_SCENARIO_DENOMINATORS")
        require(event_level["n_components"].astype(int).ge(3).all(), "EVENTS_WITH_LT3_COMPONENTS")

        matrix_floors = set(matrix["sic_floor"].astype(str))
        matrix_min_cells = set(matrix["strict_min_cells"].astype(int))
        require(len(matrix) == 9, f"MATRIX_ROWS:{len(matrix)}")
        require(matrix_floors == EXPECTED_FLOORS, f"MATRIX_FLOORS:{sorted(matrix_floors)}")
        require(matrix_min_cells == EXPECTED_MIN_CELLS, f"MATRIX_MIN_CELLS:{sorted(matrix_min_cells)}")
        require(
            not matrix.duplicated(["sic_floor", "strict_min_cells"]).any(),
            "DUPLICATE_MATRIX_SCENARIOS",
        )
        require(matrix["n_events_total"].astype(int).eq(99).all(), "MATRIX_EVENT_DENOMINATORS")
        require(
            matrix["n_events_with_components"].astype(int).eq(99).all(),
            "MATRIX_COMPONENT_EVENT_COUNTS",
        )
        require(
            matrix["fraction_events_with_ge3_components"].astype(float).eq(1.0).all(),
            "MATRIX_GE3_COMPONENT_FRACTION",
        )
        require(matrix["qualitative_multicenter_supported"].astype(bool).all(), "MULTICENTER_GATE")
        require(
            matrix["scenario_class"].value_counts().to_dict()
            == {"CORE": 6, "STRICT_STRESS": 3},
            "SCENARIO_CLASS_COUNTS",
        )

        event_path = candidate / "stage3_component_sensitivity_event_level.csv"
        matrix_path = candidate / "stage3_component_sensitivity_matrix.csv"
        event_level.to_csv(event_path, index=False)
        matrix.to_csv(matrix_path, index=False)
        metadata = {
            "status": "PASS",
            "input_mode": "EXTERNAL_RAW_PLUS_CURRENT_RUN_M021",
            "frozen_or_accepted_generated_inputs_read": False,
            "bundled_lowsic_core": str(expected_core.relative_to(package_root)),
            "bundled_lowsic_core_sha256": sha256(expected_core),
            "stage3_products_dir": str(products),
            "stage3_product_sha256": {
                name: sha256(products / name) for name in PRODUCT_NAMES
            },
            "raw_data_root": str(raw),
            "event_level_rows": int(len(event_level)),
            "matrix_rows": int(len(matrix)),
            "event_level_sha256": sha256(event_path),
            "matrix_sha256": sha256(matrix_path),
        }
        (candidate / "stage3_component_sensitivity_generation.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        candidate.rename(output)
    except BaseException:
        if candidate.exists():
            shutil.rmtree(candidate)
        raise

    print("STAGE3_COMPONENT_SENSITIVITY_GENERATION=PASS")
    print("STAGE3_COMPONENT_SENSITIVITY_INPUT_MODE=EXTERNAL_RAW_PLUS_CURRENT_RUN_M021")
    print("STAGE3_COMPONENT_SENSITIVITY_EVENT_LEVEL_ROWS=891")
    print("STAGE3_COMPONENT_SENSITIVITY_MATRIX_ROWS=9")
    print(f"STAGE3_COMPONENT_SENSITIVITY_OUTPUT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
