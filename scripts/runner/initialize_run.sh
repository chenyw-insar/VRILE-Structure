#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  initialize_run.sh --source_dir DIR --run_dir DIR --raw_dir DIR [options]

Options:
  --science-env NAME               Default: vrile
  --gmt-env NAME                   Default: gmt
  --workers N                      Engineering-only parallelism; default: 12
  --require-readonly-raw           Stop unless RAW is on a read-only mount.
  --allow-writable-raw             Warning-only RAW policy (the manual default).
EOF
}

SOURCE_ROOT=""
RUN_ROOT=""
RAW_ROOT=""
SCIENCE_ENV=vrile
GMT_ENV=gmt
WORKERS=12
ALLOW_WRITABLE_RAW=YES

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source_dir) SOURCE_ROOT="$2"; shift 2 ;;
    --run_dir) RUN_ROOT="$2"; shift 2 ;;
    --raw_dir) RAW_ROOT="$2"; shift 2 ;;
    --science-env|--test-env) SCIENCE_ENV="$2"; shift 2 ;;
    --gmt-env) GMT_ENV="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --allow-writable-raw) ALLOW_WRITABLE_RAW=YES; shift ;;
    --require-readonly-raw) ALLOW_WRITABLE_RAW=NO; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "HOLD_UNKNOWN_INIT_OPTION:$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$SOURCE_ROOT" && -n "$RUN_ROOT" && -n "$RAW_ROOT" ]] || {
  usage >&2
  exit 2
}
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "HOLD_INVALID_WORKERS" >&2; exit 2; }
SOURCE_ROOT="$(realpath -m -- "$SOURCE_ROOT")"
RUN_ROOT="$(realpath -m -- "$RUN_ROOT")"
RAW_ROOT="$(realpath -m -- "$RAW_ROOT")"
[[ -d "$SOURCE_ROOT" && -r "$SOURCE_ROOT/sha256.txt" ]] || { echo "HOLD_INVALID_SOURCE_PACKAGE" >&2; exit 2; }
[[ -d "$RAW_ROOT" ]] || { echo "HOLD_RAW_DATA_ROOT_MISSING:$RAW_ROOT" >&2; exit 2; }
python3 -B "$SOURCE_ROOT/scripts/runner/run_policy.py" layout \
  --source_dir "$SOURCE_ROOT" --run_dir "$RUN_ROOT" --raw_dir "$RAW_ROOT" || exit 2

(
  cd "$SOURCE_ROOT"
  sha256sum -c --strict sha256.txt
)

python3 -B "$SOURCE_ROOT/scripts/runner/copy_release.py" --source_dir "$SOURCE_ROOT"
mkdir -p "$RUN_ROOT"/{authority,work,data/processed,results,logs,validation,figures,environment,state}
PROJECT_ROOT="$RUN_ROOT/work/$(basename "$SOURCE_ROOT")"
python3 -B "$SOURCE_ROOT/scripts/runner/copy_release.py" \
  --source_dir "$SOURCE_ROOT" --destination_dir "$PROJECT_ROOT" --initializing_run_dir "$RUN_ROOT"
(
  cd "$PROJECT_ROOT"
  sha256sum -c --strict sha256.txt
) > "$RUN_ROOT/validation/working_copy_integrity_immediately_after_copy.txt"
(
  cd "$PROJECT_ROOT"
  { awk '{sub(/^[^ ]+  /, ""); print} END {print "sha256.txt"}' sha256.txt; } | LC_ALL=C sort
) > "$RUN_ROOT/environment/expected_release_files_after_copy.txt"
(
  cd "$PROJECT_ROOT"
  find . -type f -printf '%P\n' | LC_ALL=C sort
) > "$RUN_ROOT/environment/observed_release_files_after_copy.txt"
cmp -s "$RUN_ROOT/environment/expected_release_files_after_copy.txt" \
  "$RUN_ROOT/environment/observed_release_files_after_copy.txt" || {
  echo "HOLD_WORKING_COPY_FILE_SET_MISMATCH_IMMEDIATELY_AFTER_COPY" >&2
  exit 2
}
ln -s "$RAW_ROOT" "$RUN_ROOT/data/raw"
ln -s "$RUN_ROOT/data" "$PROJECT_ROOT/data"
ln -s "$RUN_ROOT/results" "$PROJECT_ROOT/outputs"

CONFIG="$RUN_ROOT/state/runner.env"
{
  printf 'VRILE_RUN_ROOT=%q\n' "$RUN_ROOT"
  printf 'VRILE_PROJECT_ROOT=%q\n' "$PROJECT_ROOT"
  printf 'VRILE_RAW_DATA_ROOT=%q\n' "$RAW_ROOT"
  printf 'VRILE_ACCEPTED_COMPARISON_ROOT=%q\n' ""
  printf 'VRILE_SCIENCE_ENV=%q\n' "$SCIENCE_ENV"
  printf 'VRILE_GMT_ENV=%q\n' "$GMT_ENV"
  printf 'VRILE_WORKERS=%q\n' "$WORKERS"
  printf 'VRILE_ALLOW_WRITABLE_RAW=%q\n' "$ALLOW_WRITABLE_RAW"
} > "$CONFIG"
chmod 0444 "$CONFIG"
printf 'status=PASS\ninitialized_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_ROOT/state/00_initialized.pass"
python3 -B "$PROJECT_ROOT/scripts/runner/run_evidence.py" init --run_dir "$RUN_ROOT" --project-root "$PROJECT_ROOT"

echo "RESEARCHER_RUN_INITIALIZED=PASS"
echo "WORKING_COPY_INTEGRITY_IMMEDIATELY_AFTER_COPY=PASS"
echo "VRILE_RUN_ROOT=$RUN_ROOT"
echo "VRILE_PROJECT_ROOT=$PROJECT_ROOT"
echo "NEXT=bash $SOURCE_ROOT/run_stage1.sh --run_dir $RUN_ROOT"
