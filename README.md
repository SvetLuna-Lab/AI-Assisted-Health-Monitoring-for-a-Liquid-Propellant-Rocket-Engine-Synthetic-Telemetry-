# AI-Assisted Health Monitoring for a Liquid-Propellant Rocket Engine (Synthetic Telemetry)

This project demonstrates how AI/ML can be applied to monitor the health of a liquid-propellant rocket engine using **synthetic telemetry data**.

The engine telemetry includes:

- `Pc` – chamber pressure
- `N_pump` – turbopump speed (RPM)
- `T_in` – injector / inlet temperature
- `Vib` – vibration level

We simulate normal start–steady–shutdown behavior and several types of faults:
- slow chamber pressure decay,
- turbopump overspeed,
- abnormal temperature rise,
- increased vibrations.

A simple ML pipeline (Random Forest classifier) is used to distinguish between normal and faulty regimes based on rolling statistics of the telemetry channels.

## Project structure

- `data/` – generated CSV files with synthetic telemetry runs  
- `src/`
  - `generate_telemetry.py` – synthetic data generation
  - `preprocessing.py` – feature engineering (rolling statistics)
  - `train_classifier.py` – training and evaluation script
- `notebooks/` – Jupyter notebooks for exploratory analysis and plots
- `figures/` – saved figures for documentation
- `requirements.txt` – Python dependencies

## How to run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Generate data and train the model
python -m src.train_classifier


The script will:

Generate synthetic telemetry CSV files in data/ (if none exist).

Compute rolling features.

Train a Random Forest classifier and print a classification report.
