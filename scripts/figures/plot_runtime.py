"""Direct-API bridge to the exact same contract used by public shell runners."""
import importlib.util
from pathlib import Path


def require_current_runtime():
    path = Path(__file__).resolve().parent.parent / 'runner/plotting_runtime.py'
    spec = importlib.util.spec_from_file_location('vrile_plotting_runtime_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.current_runtime()
