"""Utilities for parsing and labelling month selections."""

from __future__ import annotations

from collections.abc import Iterable


def parse_months(value: str | Iterable[int] | None) -> tuple[int, ...]:
    """Return a validated tuple of month numbers from a label or list."""
    if value is None:
        return tuple(range(1, 13))
    if isinstance(value, str):
        key = value.strip().lower()
        aliases = {
            "all": tuple(range(1, 13)),
            "annual": tuple(range(1, 13)),
            "year": tuple(range(1, 13)),
            "djf": (12, 1, 2),
            "mam": (3, 4, 5),
            "jja": (6, 7, 8),
            "summer": (6, 7, 8),
            "son": (9, 10, 11),
        }
        if key in aliases:
            return aliases[key]
        months = tuple(int(part.strip()) for part in key.split(",") if part.strip())
    else:
        months = tuple(int(month) for month in value)
    if not months:
        raise ValueError("At least one month must be provided.")
    bad = [month for month in months if month < 1 or month > 12]
    if bad:
        raise ValueError(f"Invalid month numbers: {bad}")
    seen: set[int] = set()
    unique = []
    for month in months:
        if month not in seen:
            unique.append(month)
            seen.add(month)
    return tuple(unique)


def months_label(months: str | Iterable[int] | None) -> str:
    """Return a stable lowercase label for filenames and metadata."""
    parsed = parse_months(months)
    key = tuple(sorted(parsed))
    canonical = {
        tuple(range(1, 13)): "annual",
        (1, 2, 12): "djf",
        (3, 4, 5): "mam",
        (6, 7, 8): "jja",
        (9, 10, 11): "son",
    }
    if key in canonical:
        return canonical[key]
    return "m" + "-".join(f"{month:02d}" for month in key)
