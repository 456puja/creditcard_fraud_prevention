"""
feature_engineering.py

Purpose:
--------
Create meaningful, stable, and model-ready features from preprocessed
credit card transaction data to improve fraud detection performance.

This module is used in:
- Offline training (model building)
- Online inference (FastAPI)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = Path("../data/preprocessed")
MODEL_DIR = Path("../models")

MODEL_DIR.mkdir(exist_ok=True)


# -----------------------------
# Helper Functions
# -----------------------------

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between highly fraud-correlated PCA features.

    Why:
    ----
    Fraud patterns are often non-linear. Feature interactions help
    tree-based models capture complex behavior.

    Achieved:
    ---------
    Added combined risk representation.
    """
    df["V14_V17_interaction"] = df["V14"] * df["V17"]
    return df


def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer amount-based risk features.

    Why:
    ----
    Transaction amount is highly skewed and contains outliers.
    Transformations stabilize learning.

    Achieved:
    ---------
    - Log-scaled amount
    - Risk bucketed amount
    """
    df["Amount_log"] = np.log1p(df["Amount"])

    df["Amount_risk_bucket"] = pd.cut(
        df["Amount"],
        bins=[-1, 10, 100, 1000, df["Amount"].max()],
        labels=[0, 1, 2, 3]
    )

    df["Amount_risk_bucket"] = df["Amount_risk_bucket"].astype(int)
    return df


def create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create statistical summary features from PCA components.

    Why:
    ----
    Instead of relying on individual PCA values, statistics
    capture overall transaction abnormality.

    Achieved:
    ---------
    Compact anomaly representation.
    """
    pca_cols = [f"V{i}" for i in range(1, 29)]

    df["pca_mean"] = df[pca_cols].mean(axis=1)
    df["pca_std"] = df[pca_cols].std(axis=1)

    return df


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale features using StandardScaler.

    Why:
    ----
    Ensures consistent feature magnitude for ML models
    and prevents bias toward large-value features.

    Achieved:
    ---------
    Scaled and deployment-safe features.
    """
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    return X_train_scaled, X_test_scaled


# -----------------------------
# Main Feature Engineering Flow
# -----------------------------

def run_feature_engineering():
    """
    Execute full feature engineering pipeline.
    """

    # Load preprocessed data
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    # Feature creation
    X_train = create_interaction_features(X_train)
    X_test = create_interaction_features(X_test)

    X_train = create_amount_features(X_train)
    X_test = create_amount_features(X_test)

    X_train = create_statistical_features(X_train)
    X_test = create_statistical_features(X_test)

    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Save engineered data
    X_train_scaled.to_csv(DATA_DIR / "X_train_fe.csv", index=False)
    X_test_scaled.to_csv(DATA_DIR / "X_test_fe.csv", index=False)

    print("✅ Feature engineering completed successfully.")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    run_feature_engineering()
