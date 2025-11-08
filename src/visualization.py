from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


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
  
