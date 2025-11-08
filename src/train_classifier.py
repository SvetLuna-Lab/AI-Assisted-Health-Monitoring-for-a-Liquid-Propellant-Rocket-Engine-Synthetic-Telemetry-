from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from .generate_telemetry import generate_and_save_runs
from .preprocessing import load_data, feature_engineering


def train_and_evaluate(df_feat: pd.DataFrame) -> RandomForestClassifier:
    """
    Train a simple supervised classifier and print evaluation metrics.
    """
    X = df_feat.drop(columns=["label"])
    y = df_feat["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred))

    return clf


def main() -> None:
    # 1. Generate data (or skip if already generated)
    data_dir = Path("data")
    if not data_dir.exists() or not any(data_dir.glob("*.csv")):
        print("No data found, generating synthetic telemetry...")
        generated_files = generate_and_save_runs(output_dir=str(data_dir))
    else:
        generated_files = [str(p) for p in data_dir.glob("*.csv")]

    # 2. Load and preprocess
    df_raw = load_data(generated_files)
    df_feat = feature_engineering(df_raw, window=10)

    # 3. Train classifier
    clf = train_and_evaluate(df_feat)
    print("Training finished.")


if __name__ == "__main__":
    main()
