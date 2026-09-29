#!/usr/bin/env python3
"""Small parallel/reproducibility helpers for VRILE submission-stage scripts.

The helpers are intentionally conservative: serial execution remains the
fallback, BLAS/OpenMP worker oversubscription is limited, and parallel workers
never write the same output file directly.
"""

from __future__ import annotations

import os
import pickle
import random
import shutil
import tempfile
import warnings
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def configure_blas_threads(n_threads: int = 1) -> None:
    """Keep numeric libraries from oversubscribing CPU cores."""

    value = str(max(1, int(n_threads)))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, value)


def get_default_workers(max_default: int = 12) -> int:
    """Return a conservative worker count for WSL/desktop runs."""

    cpu = os.cpu_count() or 2
    if cpu <= 4:
        return max(1, cpu - 1)
    return max(1, min(max_default, cpu - 2))


def recommend_workers_for_wsl(max_memory_gb: float | None = None) -> int:
    """Recommend workers using CPU count and, when available, memory."""

    workers = get_default_workers()
    if max_memory_gb is not None and max_memory_gb > 0:
        # Most submission-stage tasks are table/statistics based; keep at least
        # ~1 GB per worker to avoid WSL swapping on Windows-mounted drives.
        workers = min(workers, max(1, int(max_memory_gb)))
    return workers


def set_reproducible_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def log_parallel_config(logger, workers: int, parallel: bool, backend: str, quick: bool = False) -> None:
    logger.info(
        "parallel config: parallel=%s backend=%s workers=%s cpu_count=%s quick=%s",
        parallel,
        backend,
        workers,
        os.cpu_count(),
        quick,
    )


def parallel_map(
    func: Callable[[T], R],
    items: Sequence[T] | Iterable[T],
    *,
    workers: int | None = None,
    parallel: bool = False,
    backend: str = "process",
) -> list[R]:
    """Map with a serial fallback and deterministic result order."""

    items_list = list(items)
    if backend == "serial" or not parallel or len(items_list) <= 1:
        return [func(item) for item in items_list]

    workers = workers or get_default_workers()
    workers = max(1, min(int(workers), len(items_list)))
    if workers == 1:
        return [func(item) for item in items_list]

    if backend == "thread":
        executor_cls = ThreadPoolExecutor
    elif backend == "process":
        executor_cls = ProcessPoolExecutor
    elif backend == "dask":
        try:
            from dask.distributed import Client, LocalCluster

            cluster = LocalCluster(n_workers=workers, threads_per_worker=1, dashboard_address=None)
            client = Client(cluster)
            try:
                futures = client.map(func, items_list)
                return list(client.gather(futures))
            finally:
                client.close()
                cluster.close()
        except Exception:
            executor_cls = ProcessPoolExecutor
    else:
        raise ValueError(f"unsupported backend: {backend}")

    out: list[R | None] = [None] * len(items_list)
    with executor_cls(max_workers=workers) as ex:
        future_to_idx = {ex.submit(func, item): idx for idx, item in enumerate(items_list)}
        for fut in as_completed(future_to_idx):
            out[future_to_idx[fut]] = fut.result()
    return list(out)  # type: ignore[arg-type]


def is_executor_fallback_error(exc: Exception) -> bool:
    """Return True only for executor infrastructure or pickling failures."""

    if isinstance(exc, (BrokenExecutor, pickle.PicklingError)):
        return True
    msg = str(exc).lower()
    return "can't pickle" in msg or "cannot pickle" in msg


def run_with_fallback(
    func: Callable[[T], R],
    items: Sequence[T] | Iterable[T],
    *,
    workers: int | None = None,
    parallel: bool = False,
    backend: str = "process",
    logger=None,
) -> list[R]:
    """Run in parallel, falling back to serial on pickling/backend failures."""

    try:
        return parallel_map(func, items, workers=workers, parallel=parallel, backend=backend)
    except Exception as exc:
        if not is_executor_fallback_error(exc):
            raise
        msg = f"parallel executor failed; falling back to serial execution: {exc}"
        if logger is not None:
            logger.warning(msg)
        else:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return parallel_map(func, items, workers=1, parallel=False, backend="thread")


def safe_write_csv(df, path: str | Path, **kwargs) -> None:
    """Write a CSV through a temporary file and atomic replace."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.to_csv(tmp, index=False, **kwargs)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def safe_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def copy_if_exists(src: str | Path, dst: str | Path) -> bool:
    src, dst = Path(src), Path(dst)
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True
