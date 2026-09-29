"""Verify current M007 output routing/ownership; never compute scientific values."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
from release_identity import manifest_name

PRODUCTS = {
    "broad": "results/local_vrile_enhanced/unique_local_events.csv",
    "patches": "results/local_vrile_enhanced/local_objects_filtered.csv",
    "broad_summary": "results/local_vrile_enhanced/unique_local_event_region_summary.csv",
    "severe": "results/local_vrile_severe/severe_unique_local_events.csv",
    "severe_summary": "results/local_vrile_severe/severe_local_event_region_summary.csv",
    "major_severe": "results/local_vrile_major_severe/major_severe_events_union.csv",
    "major_summary": "results/local_vrile_major_severe/major_severe_event_region_summary.csv",
    "matching": "results/spatiotemporal_matching/fig3_count_vector.csv",
    "matching_rules": "results/spatiotemporal_matching/matching_rule_sensitivity.csv",
    "matching_missed": "results/spatiotemporal_matching/missed_major_severe_events.csv",
    "membership": "data/processed/local_event_cells/unique_event_patch_membership.csv",
}

def require(condition, label):
    if not condition:
        raise ValueError("HOLD_M007_HANDOFF:" + label)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()

def json_read(path):
    return json.loads(path.read_text())

def verify(run):
    run = run.resolve()
    identity = json_read(run / "state/execution_identity.json")
    project = Path(identity["project_root"])
    require(Path(identity["run_root"]) == run, "IDENTITY")
    require(project.resolve().is_relative_to(run / "work"), "PROJECT_ISOLATION")
    require((project / "outputs").resolve() == run / "results", "OUTPUT_LINK")
    require((project / "data").resolve() == run / "data", "DATA_LINK")
    require(sha(project / manifest_name(project)) == identity["candidate_manifest_sha256"], "CANDIDATE_IDENTITY")
    start_path = run / "evidence/commands/15_m007_corrected_rebuild.start.json"
    end_path = run / "evidence/commands/15_m007_corrected_rebuild.end.json"
    start, end = json_read(start_path), json_read(end_path)
    require(start["identity"] == end["identity"] == identity, "COMMAND_IDENTITY")
    require(end["command_exit"] == end["tee_exit"] == 0 and end["status"] == "PASS", "COMMAND_EXIT")
    require(end["start_sha256"] == sha(start_path) and end["log_sha256"] == sha(end["log"]), "COMMAND_HASH_CHAIN")
    argv = start["argv"]
    producer = project / "scripts/stage1_rebuild_local_catalog_from_frozen_cells.py"
    require(str(producer) in argv and argv.count("--out_dir") == 1, "PRODUCER_ARGV")
    require(Path(argv[argv.index("--out_dir") + 1]).resolve() == project.resolve(), "OUT_DIR_NOT_WORKING_PROJECT")
    require(argv.count("--promotion-layout") == 1, "PROMOTION_LAYOUT")
    metadata_path = project / "run_metadata.json"
    metadata = json_read(metadata_path)
    require(metadata["command"] == " ".join(argv[argv.index(str(producer)):]), "METADATA_COMMAND")
    require(metadata["production_sources"]["scripts/stage1_rebuild_local_catalog_from_frozen_cells.py"] == sha(producer), "PRODUCER_SOURCE")
    for relative, expected in metadata["production_sources"].items():
        require(sha(project / relative) == expected, "METADATA_SOURCE:" + relative)
    paths = {role: run / rel for role, rel in PRODUCTS.items()}
    paths["metadata"] = metadata_path
    for role, path in paths.items():
        require(path.is_file() and path.stat().st_size > 0, "MISSING_OR_EMPTY:" + role)
        require(path.resolve().is_relative_to(run), "OUTSIDE_CURRENT_RUN:" + role)
    # Audit events establish current-command writes, not filesystem mtime authority.
    start_ns = int(datetime.datetime.fromisoformat(start["started_utc"]).timestamp() * 1e9)
    end_ns = int(datetime.datetime.fromisoformat(end["ended_utc"]).timestamp() * 1e9)
    writes = set()
    traces = []
    for trace in sorted((run / "evidence/python_reads").glob("*.jsonl")):
        selected = False
        with trace.open() as stream:
            for line in stream:
                event = json.loads(line)
                if event["event"] == "trace_enabled":
                    selected = start_ns <= event["details"]["time_ns"] <= end_ns
                    if not selected:
                        break
                    traces.append({"path": str(trace), "sha256": sha(trace)})
                if selected and event["event"] == "open":
                    details = event["details"]
                    if (details.get("flags") or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                        writes.add(Path(details["path"]).resolve())
    require(bool(traces), "NO_CURRENT_COMMAND_TRACE")
    for role, path in paths.items():
        require(path.resolve() in writes, "NO_CURRENT_M007_WRITE:" + role)
    products = []
    for role, path in paths.items():
        count = None
        if path.suffix == ".csv":
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                require(bool(reader.fieldnames), "NO_SCHEMA:" + role)
                count = sum(1 for _ in reader)
            if role in metadata["counts"]:
                require(count == metadata["counts"][role], "METADATA_COUNT:" + role)
        products.append({"role": role, "relative_path": path.relative_to(run).as_posix(), "sha256": sha(path), "rows": count, "current_m007_write_observed": True})
    return {"status": "PASS", "scope": "CURRENT_RUN_M007_HANDOFF_NOT_ACCEPTED_SCIENCE_PARITY", "identity": identity, "m007_start_sha256": sha(start_path), "m007_end_sha256": sha(end_path), "metadata_path": str(metadata_path), "metadata_sha256": sha(metadata_path), "products": products, "write_traces": traces, "centroid_parity": "REQUIRED_POST_GENERATION_NOT_WAIVED"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.run_dir)
    target = args.run_dir / "validation/m007_handoff.json"
    with target.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("M007_CURRENT_RUN_HANDOFF=PASS")

if __name__ == "__main__":
    main()
