from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def plot_time_series_with_labels(
    df: pd.DataFrame,
    channel: str = "Pc",
    title: Optional[str] = None,
    filename: str = "time_series_Pc.png",
) -> None:
    """
    Plot time series of a given telemetry channel with color-coded labels.
    Assumes df has columns: 'time', channel, 'label'.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = df["label"].unique()
    for lbl in labels:
        mask = df["label"] == lbl
        ax.plot(
            df.loc[mask, "time"],
            df.loc[mask, channel],
            label=lbl,
        )

    ax.set_xlabel("Time, s")
    ax.set_ylabel(channel)
    ax.set_title(title or f"{channel} time series by regime")
    ax.grid(True)
    ax.legend()

    out_path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation_matrix(
    df: pd.DataFrame,
    channels: Optional[List[str]] = None,
    filename: str = "correlation_matrix.png",
) -> None:
    """
    Plot a simple correlation matrix for selected telemetry channels.
    """
    if channels is None:
        channels = ["Pc", "N_pump", "T_in", "Vib"]

    corr = df[channels].corr().values

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(corr, interpolation="nearest")

    ax.set_xticks(range(len(channels)))
    ax.set_yticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha="right")
    ax.set_yticklabels(channels)

    fig.colorbar(cax, ax=ax)
    ax.set_title("Correlation matrix of telemetry channels")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)


def plot_feature_importance(
    model: RandomForestClassifier,
    feature_names: List[str],
    top_n: int = 10,
    filename: str = "feature_importance.png",
) -> None:
    """
    Plot top_n most important features for a trained RandomForest model.
    """
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(idx)), importances[idx])
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right")

    ax.set_ylabel("Importance")
    ax.set_title("Top feature importances (RandomForest)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
