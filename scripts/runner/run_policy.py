"""Run-directory and pre-science RAW access policy. No scientific computation."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid
from release_identity import manifest_name


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contains(parent, child):
    return parent == child or parent in child.parents


def check_location(source, run, raw=None):
    if contains(run, source):
        raise ValueError('RUN_DIR_OVERLAPS_SOURCE:choose_a_separate_run_directory')
    if raw is not None and (contains(raw, run) or contains(run, raw)):
        raise ValueError('RUN_DIR_OVERLAPS_RAW:choose_a_directory_outside_RAW')
    if contains(source, run):
        reserved = {'raw', 'data', '.git'}
        with (source / manifest_name(source)).open() as stream:
            reserved.update(Path(r['relative_path']).parts[0] for r in csv.DictReader(stream))
        if run.relative_to(source).parts[0] in reserved:
            raise ValueError('RUN_DIR_OVERLAPS_PACKAGE_FILES:choose_a_new_output_directory')
    # Never nest a fresh run in an already initialized run's results or work.
    for parent in run.parents:
        if (parent / 'state/execution_identity.json').is_file():
            raise ValueError('RUN_DIR_INSIDE_EXISTING_RUN:choose_a_separate_directory')


def initialized_run(path):
    """Recognize run trees for pruning; never recurse into their scientific data."""
    identity = path / 'state/execution_identity.json'
    if not identity.is_file():
        return False
    try:
        record = json.loads(identity.read_text())
        project = Path(record['project_root'])
        return (Path(record['run_root']) == path.resolve()
                and project.parent == path.resolve() / 'work'
                and digest(project / manifest_name(project)) == record['candidate_manifest_sha256']
                and digest(project / 'sha256.txt') == record['candidate_sha256_ledger'])
    except (OSError, ValueError, KeyError):
        return False


def prepare_existing(source, run, policy):
    record = json.loads((run / 'state/execution_identity.json').read_text())
    project = Path(record['project_root'])
    if not initialized_run(run):
        raise ValueError('RUN_IDENTITY_MISMATCH')
    for filename, key in [(manifest_name(source), 'candidate_manifest_sha256'),
                          ('sha256.txt', 'candidate_sha256_ledger')]:
        if digest(source / filename) != record[key]:
            raise ValueError('RUN_BELONGS_TO_DIFFERENT_RELEASE:use_its_original_runner_or_a_new_run_dir')
    if policy is None:
        return
    config = run / 'state/runner.env'
    before = config.read_bytes()
    lines = before.decode().splitlines(True)
    found = [i for i, line in enumerate(lines) if line.startswith('VRILE_ALLOW_WRITABLE_RAW=')]
    if len(found) != 1:
        raise ValueError('RAW_POLICY_CONFIG_INVALID')
    index = found[0]
    previous = lines[index].strip().split('=', 1)[1]
    if previous not in {'YES', 'NO'}:
        raise ValueError('RAW_POLICY_CONFIG_INVALID')
    if previous == policy:
        return
    for directory in ('results', 'data/processed', 'figures', 'logs', 'evidence/commands'):
        path = run / directory
        if path.exists() and any(path.iterdir()):
            raise ValueError('RAW_POLICY_LOCKED_AFTER_SCIENCE:keep_this_run_or_choose_a_new_run_dir')
    if any(p.name[:1].isdigit() and int(p.name.split('_', 1)[0]) >= 10
           for p in (run / 'state').iterdir() if p.suffix in {'.pass', '.failed'}):
        raise ValueError('RAW_POLICY_LOCKED_AFTER_SCIENCE:stage_already_started')
    lines[index] = f'VRILE_ALLOW_WRITABLE_RAW={policy}\n'
    after = ''.join(lines).encode()
    # Preserve the actual initialization snapshot; append a separate change record.
    event = {'scope': 'PRE_SCIENCE_RAW_ACCESS_POLICY_ONLY', 'execution_id': record['execution_id'],
             'utc': datetime.now(timezone.utc).isoformat(), 'previous': previous, 'requested': policy,
             'before_sha256': hashlib.sha256(before).hexdigest(),
             'after_sha256': hashlib.sha256(after).hexdigest(),
             'before_config': before.decode(), 'after_config': after.decode()}
    log = run / 'evidence' / ('raw_policy_change_' + uuid.uuid4().hex + '.json')
    with log.open('x') as stream:
        json.dump(event, stream, indent=2); stream.write('\n')
    mode = config.stat().st_mode & 0o777
    config.chmod(mode | 0o200)
    try:
        config.write_bytes(after)
    finally:
        config.chmod(mode)
    print('PRE_SCIENCE_RAW_POLICY_UPDATED=' + policy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['layout', 'existing'])
    parser.add_argument('--source_dir', type=Path, required=True)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--raw_dir', type=Path)
    parser.add_argument('--allow-writable', choices=['YES', 'NO'])
    args = parser.parse_args()
    try:
        source = args.source_dir.resolve(strict=True)
        run = args.run_dir.resolve()
        if args.action == 'layout':
            check_location(source, run, args.raw_dir.resolve(strict=True))
            if run.exists() and (not run.is_dir() or any(run.iterdir())):
                raise ValueError('REFUSE_NONEMPTY_RUN_DIR:use_an_empty_or_new_directory')
        else:
            prepare_existing(source, run, args.allow_writable)
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit('HOLD_' + str(exc))


if __name__ == '__main__':
    main()
