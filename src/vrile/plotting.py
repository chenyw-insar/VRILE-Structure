"""Basic plotting helpers for VRILE diagnostics."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_daily_delta(daily_csv: str | Path, output_png: str | Path) -> None:
    """Plot the ΔSIE (or filtered ΔSIE) time series and save as PNG."""
    df = pd.read_csv(daily_csv, parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(11, 4))
    y = "delta_sie_butterworth" if "delta_sie_butterworth" in df.columns else "delta_sie"
    ax.plot(df["date"], df[y], lw=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("ΔSIE (10$^6$ km$^2$)")
    ax.set_title(y)
    ax.grid(True, alpha=0.3)
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def plot_event_counts(events_csv: str | Path, output_png: str | Path) -> None:
    """Plot the number of VRILEs per year and method."""
    ev = pd.read_csv(events_csv, parse_dates=["date"])
    if ev.empty:
        raise ValueError("Event file is empty; cannot plot counts.")
    ev["year"] = ev["date"].dt.year
    counts = ev.groupby(["year", "method"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    counts.plot(ax=ax, lw=1.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of VRILEs")
    ax.grid(True, alpha=0.3)
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)