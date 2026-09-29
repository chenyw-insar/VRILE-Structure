#!/usr/bin/env python3
"""Resolve/check the one exact plotting contract without importing science.

The default interpreter is an external minimal venv. An advanced
VRILE_PLOT_PYTHON override must pass the same strict check. This program never
installs dependencies or renders; --default-python only prints the setup path.
"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys


def specification():
    root = Path(__file__).resolve().parents[2]
    path = root / 'config/plotting-runtime.json'
    spec = json.loads(path.read_text())
    requirements = root / 'config/plotting-requirements.txt'
    effective = [s.strip() for s in requirements.read_text().splitlines()
                 if s.strip() and not s.lstrip().startswith('#')]
    if effective != ['PyMuPDF==' + spec['pymupdf_version']]:
        raise RuntimeError('HOLD_PLOTTING_SPECIFICATION_REQUIREMENT_MISMATCH')
    return spec, hashlib.sha256(path.read_bytes()).hexdigest()


def default_python(spec):
    data = Path(os.environ.get('XDG_DATA_HOME', str(Path.home() / '.local/share')))
    return data.expanduser().absolute() / 'vrile' / ('plotting-pymupdf-' + spec['pymupdf_version']) / 'bin/python'


PROBE = '''import json,sys,platform
try:
 import pymupdf
except ImportError as e:
 print("HOLD_PYMUPDF_MISSING:" + str(e), file=sys.stderr); raise SystemExit(2)
print(json.dumps({"python_executable":sys.executable,"python_version":platform.python_version(),
 "pymupdf_version":getattr(pymupdf,"__version__",None),"pymupdf_module":pymupdf.__file__,
 "mupdf_version":getattr(pymupdf,"mupdf_version",None),
 "pymupdf_version_tuple":getattr(pymupdf,"version",None)}))
'''


def check_runtime(requested=None):
    spec, spec_sha = specification()
    requested = requested or os.environ.get('VRILE_PLOT_PYTHON')
    selection = 'VRILE_PLOT_PYTHON_OR_EXPLICIT_OVERRIDE' if requested else 'DEFAULT_EXTERNAL_VENV'
    executable = str(Path(requested).expanduser()) if requested else str(default_python(spec))
    if os.sep not in executable:
        executable = shutil.which(executable) or executable
    # Do not resolve interpreter symlinks: doing so would bypass the venv.
    executable = str(Path(executable).absolute())
    if not Path(executable).is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError('HOLD_PLOTTING_PYTHON_INVALID:' + executable + ':see setup_plotting_runtime.sh --help')
    try:
        completed = subprocess.run([executable, '-I', '-B', '-c', PROBE],
                                   text=True, capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError('HOLD_PLOTTING_PYTHON_PROBE:' + str(exc)) from exc
    if completed.returncode:
        raise RuntimeError('HOLD_PLOTTING_RUNTIME_IMPORT:' + completed.stderr.strip())
    try:
        observed = json.loads(completed.stdout)
    except (ValueError, TypeError) as exc:
        raise RuntimeError('HOLD_PLOTTING_RUNTIME_INVALID_PROBE') from exc
    return validate_observed(observed, spec, spec_sha, executable, selection)


def validate_observed(observed, spec, spec_sha, executable, selection):
    if observed.get('pymupdf_version') != spec['pymupdf_version']:
        raise RuntimeError('HOLD_PYMUPDF_VERSION_MISMATCH:expected=' + spec['pymupdf_version']
                           + ':observed=' + str(observed.get('pymupdf_version')))
    observed.update(status='PASS', selection=selection, selected_python=executable,
                    required_pymupdf_version=spec['pymupdf_version'],
                    contract_sha256=spec_sha, contract=spec['contract'])
    return observed


def current_runtime():
    """Direct figure APIs gate the actual importing process, not an override."""
    spec, spec_sha = specification()
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError('HOLD_PYMUPDF_MISSING:' + str(exc)) from exc
    observed = dict(python_executable=sys.executable, python_version=platform.python_version(),
                    pymupdf_version=getattr(pymupdf, '__version__', None),
                    pymupdf_module=pymupdf.__file__,
                    mupdf_version=getattr(pymupdf, 'mupdf_version', None),
                    pymupdf_version_tuple=getattr(pymupdf, 'version', None))
    return validate_observed(observed, spec, spec_sha, sys.executable, 'CURRENT_PROCESS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', help='Advanced interpreter override; exact same PyMuPDF version gate')
    parser.add_argument('--default-python', action='store_true', help='Print deterministic setup destination without import/install')
    parser.add_argument('--print-python', action='store_true', help='After verification, print only the selected interpreter')
    parser.add_argument('--report', type=Path, help='Write actual interpreter/PyMuPDF/MuPDF metadata after PASS')
    args = parser.parse_args()
    if args.default_python:
        print(default_python(specification()[0])); return
    observed = check_runtime(args.python)
    encoded = json.dumps(observed, sort_keys=True, indent=2) + '\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded)
    print(observed['selected_python'] if args.print_python else encoded, end='\n' if args.print_python else '')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr); raise SystemExit(2)
