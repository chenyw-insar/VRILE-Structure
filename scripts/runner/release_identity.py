"""Read the one current release identity; never infer it from a directory name."""
import argparse
import json
from pathlib import Path, PurePosixPath


def read_identity(project):
    record = json.loads((Path(project) / 'config/release_identity.json').read_text())
    for key in ('release_id', 'version', 'manifest', 'scientific_core_lineage'):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError('HOLD_RELEASE_IDENTITY_FIELD:' + key)
    path = PurePosixPath(record['manifest'])
    if len(path.parts) != 1 or path.is_absolute() or '\\' in record['manifest']:
        raise ValueError('HOLD_RELEASE_MANIFEST_NAME')
    return record


def manifest_name(project):
    return read_identity(project)['manifest']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--field')
    args = parser.parse_args()
    data = read_identity(args.project_root)
    print(data[args.field] if args.field else json.dumps(data, sort_keys=True))
