import os
from typing import List

import numpy as np
import pandas as pd


def generate_time_series(run_duration: int = 300, fs: float = 1.0) -> np.ndarray:
    """
    Generate time axis, e.g. 300 seconds with 1 Hz sampling rate.
    """
    t = np.arange(0, run_duration, 1 / fs)
    return t


def normal_behavior(t: np.ndarray) -> pd.DataFrame:
    """
    Generate normal engine behavior:

    - Pc: chamber pressure, ramp-up at start, stable in steady state, ramp-down at shutdown
    - N_pump: turbopump RPM, fast rise and stabilization
    - T_in: injector/inlet temperature
    - Vib: low, relatively stable vibrations
    """
    # Chamber pressure: smoothed ramp up and down using tanh
    Pc = 50 + 30 * (np.tanh((t - 30) / 10)) - 30 * (np.tanh((t - 270) / 10))
    Pc += np.random.normal(0, 0.5, size=len(t))

    # Turbopump speed
    N_pump = 3000 + 1500 * (np.tanh((t - 30) / 15)) - 1500 * (np.tanh((t - 270) / 15))
    N_pump += np.random.normal(0, 20, size=len(t))

    # Injector inlet temperature
    T_in = 300 + 20 * np.sin(t / 50) + np.random.normal(0, 1, size=len(t))

    # Vibrations
    Vib = 0.3 + 0.05 * np.random.normal(0, 1, size=len(t))

    return pd.DataFrame(
        {
            "time": t,
            "Pc": Pc,
            "N_pump": N_pump,
            "T_in": T_in,
            "Vib": Vib,
            "label": "normal",
        }
    )


def faulty_behavior(t: np.ndarray, fault_type: str) -> pd.DataFrame:
    """
    Generate faulty engine behavior based on the normal profile with injected anomalies.

    fault_type:
      - "pressure_decay"
      - "turbopump_overspeed"
      - "temp_rise"
      - "vibration_increase"
    """
    df = normal_behavior(t)

    if fault_type == "pressure_decay":
        # Slow pressure decay over time
        decay = -0.1 * (t / t.max())  # normalized decay factor
        df["Pc"] += decay * 30

    elif fault_type == "turbopump_overspeed":
        # Turbopump overspeed after mid-run
        overspeed = 3000 + 2000 * (np.tanh((t - 150) / 10))
        df["N_pump"] = np.where(t > 150, overspeed, df["N_pump"])

    elif fault_type == "temp_rise":
        # Abnormal injector temperature rise
        rise = 100 * (np.tanh((t - 200) / 10))
        df["T_in"] += rise

    elif fault_type == "vibration_increase":
        # Increasing vibration towards the end of the run
        increase = 1.5 * (np.tanh((t - 180) / 10))
        df["Vib"] += increase + 0.1 * np.random.normal(0, 1, size=len(t))

    df["label"] = fault_type
    return df


def generate_and_save_runs(
    output_dir: str = "data",
    n_normal: int = 5,
    n_faulty_per_type: int = 3,
    run_duration: int = 300,
    fs: float = 1.0,
    random_seed: int = 42,
) -> List[str]:
    """
    Generate multiple normal and faulty runs and save them as CSV files.
    Returns a list of generated filenames.
    """
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    t = generate_time_series(run_duration=run_duration, fs=fs)
    generated_files: List[str] = []

    # Normal runs
    for i in range(n_normal):
        df = normal_behavior(t)
        fname = os.path.join(output_dir, f"normal_run_{i}.csv")
        df.to_csv(fname, index=False)
        generated_files.append(fname)

    # Faulty runs
    faults = [
        "pressure_decay",
        "turbopump_overspeed",
        "temp_rise",
        "vibration_increase",
    ]

    for fault in faults:
        for i in range(n_faulty_per_type):
            df = faulty_behavior(t, fault)
            fname = os.path.join(output_dir, f"{fault}_run_{i}.csv")
            df.to_csv(fname, index=False)
            generated_files.append(fname)

    return generated_files


if __name__ == "__main__":
    files = generate_and_save_runs()
    print(f"Generated {len(files)} CSV files in ./data")
