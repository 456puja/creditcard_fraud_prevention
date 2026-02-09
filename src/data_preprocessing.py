"""
data_preprocessing.py

Purpose:
--------
Prepare raw transaction data for feature engineering and model training
by cleaning, splitting, and balancing the dataset.

This file ensures:
- No data leakage
- Proper class imbalance handling
- Reproducibility across environments
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from pathlib import Path


# -----------------------------
# Configuration
# -----------------------------
RAW_DATA_PATH = Path("../data/raw/creditcard.csv")
PROCESSED_DATA_DIR = Path("../data/preprocessed")

PROCESSED_DATA_DIR.mkdir(exist_ok=True)


# -----------------------------
# Helper Functions
# -----------------------------

def load_raw_data() -> pd.DataFrame:
    """
    Load raw credit card transaction dataset.

    Why:
    ----
    Centralizes data loading logic for maintainability.

    Achieved:
    ---------
    Consistent data source across pipeline.
    """
    return pd.read_csv(RAW_DATA_PATH)


def split_features_target(df: pd.DataFrame):
    """
    Separate features and target variable.

    Why:
    ----
    Prevents accidental transformation of target variable.

    Achieved:
    ---------
    Clean feature-target separation.
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


def train_test_data_split(X, y):
    """
    Split dataset into training and testing sets.

    Why:
    ----
    Ensures unbiased model evaluation on unseen data.

    Achieved:
    ---------
    Reliable generalization measurement.
    """
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def apply_smote(X_train, y_train):
    """
    Balance training data using SMOTE.

    Why:
    ----
    Fraud datasets are extremely imbalanced.
    SMOTE prevents model bias toward majority class.

    Achieved:
    ---------
    Balanced training data without leaking test information.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


# -----------------------------
# Main Preprocessing Flow
# -----------------------------

def run_data_preprocessing():
    """
    Execute full preprocessing pipeline.
    """

    # Load raw data
    df = load_raw_data()

    # Split features and target
    X, y = split_features_target(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_data_split(X, y)

    # Handle class imbalance
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # Save outputs
    X_train_balanced.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)

    y_train_balanced.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    print("✅ Data preprocessing completed successfully.")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    run_data_preprocessing()