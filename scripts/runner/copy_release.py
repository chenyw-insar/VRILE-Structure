"""Verify/copy sealed source files only; never traverse raw or run directories."""
import argparse
import csv
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
from run_policy import check_location, initialized_run
from release_identity import manifest_name

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(source, initializing_run=None):
    rows = list(csv.DictReader((source / manifest_name(source)).open()))
    names = [row['relative_path'] for row in rows]
    if len(names) != len(set(n.casefold() for n in names)):
        raise ValueError('duplicate manifest paths')
    for row in rows:
        name = row['relative_path']
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError('unsafe manifest path: ' + name)
        if relative.parts[0] in {'raw', 'outputs', 'manuscript', 'manuscript_new', 'REVIEW_PACKAGES'}:
            raise ValueError('non-source namespace in manifest: ' + name)
        path = source / name
        if path.is_symlink() or not path.is_file() or source not in path.resolve().parents:
            raise ValueError('non-regular source: ' + name)
        if row['sha256'] != 'CONTROL_SELF_REFERENCE' and digest(path) != row['sha256']:
            raise ValueError('source hash mismatch: ' + name)
    ledger = {}
    for line in (source / 'sha256.txt').read_text().splitlines():
        value, name = line.split('  ', 1)
        if name in ledger or name not in names:
            raise ValueError('invalid ledger member: ' + name)
        if digest(source / name) != value:
            raise ValueError('ledger mismatch: ' + name)
        ledger[name] = value
    if set(ledger) != set(names) - {'sha256.txt'}:
        raise ValueError('incomplete SHA ledger')
    observed = set()
    for directory, dirs, files in os.walk(source, followlinks=False):
        base = Path(directory)
        if base != source and (base == initializing_run or initialized_run(base)):
            if any(base == (source / name).parent or base in (source / name).parents for name in names):
                raise ValueError('run overlaps sealed source paths: ' + str(base))
            dirs[:] = []
            continue
        if base == source:
            dirs[:] = [d for d in dirs if d not in {'raw', 'outputs'}]
            files = [f for f in files if f != 'raw']
        for name in dirs + files:
            if (base / name).is_symlink():
                raise ValueError('unexpected source symlink: ' + str(base / name))
        observed.update((base / name).relative_to(source).as_posix() for name in files)
    if observed != set(names):
        raise ValueError('unsealed/missing files: ' + repr(sorted(observed ^ set(names))))
    return names

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_dir', type=Path, required=True)
    parser.add_argument('--destination_dir', type=Path)
    parser.add_argument('--initializing_run_dir', type=Path)
    args = parser.parse_args()
    source = args.source_dir.resolve(strict=True)
    initializing_run = args.initializing_run_dir.resolve() if args.initializing_run_dir else None
    if initializing_run is not None:
        check_location(source, initializing_run)
        if args.destination_dir is None or args.destination_dir.resolve().parent != initializing_run / 'work':
            raise ValueError('initializing run exclusion requires its exact working-copy destination')
    names = verify(source, initializing_run)
    if args.destination_dir is not None:
        destination = args.destination_dir
        if destination.exists() or destination.is_symlink():
            raise ValueError('refusing existing destination')
        destination.mkdir(parents=True)
        for name in names:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
        if verify(destination.resolve()) != names:
            raise ValueError('working-copy manifest differs')
    print('SEALED_SOURCE_ONLY_COPY=PASS;FILES=' + str(len(names)))

if __name__ == '__main__':
    main()
