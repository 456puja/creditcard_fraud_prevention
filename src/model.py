"""
model.py

Purpose:
--------
Train, evaluate, and save machine learning models for fraud detection.

This file handles:
- Model training
- Model comparison
- Final model selection
- Model persistence for deployment

XGBoost is used as the final production model.
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = Path("../data/preprocessed")
MODEL_DIR = Path("../models")
MODEL_DIR.mkdir(exist_ok=True)

# -----------------------------
# Helper Functions
# -----------------------------

def load_data():
    """
    Load preprocessed and feature-engineered datasets.

    Achieved:
    ---------
    X_train, X_test, y_train, y_test loaded for model training.
    """
    X_train = pd.read_csv(DATA_DIR / "X_train_fe.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test_fe.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """
    Train multiple models:
    - Logistic Regression (baseline)
    - Random Forest (ensemble)
    - XGBoost (final model)

    Achieved:
    ---------
    Multiple candidate models for comparison.
    """

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=50,  # handles extreme imbalance
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)

    return {"LogisticRegression": lr, "RandomForest": rf, "XGBoost": xgb}


def evaluate_models(models: dict, X_test, y_test):
    """
    Evaluate trained models using ROC-AUC and select the best.

    Achieved:
    ---------
    Objective model selection for deployment.
    """
    scores = {}
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:,1]
        roc_auc = roc_auc_score(y_test, y_proba)
        scores[name] = roc_auc
        print(f"{name} ROC-AUC: {roc_auc:.4f}")

    best_model_name = max(scores, key=scores.get)
    print(f"\n✅ Selected Best Model: {best_model_name} (ROC-AUC = {scores[best_model_name]:.4f})")

    return best_model_name, models[best_model_name]


def save_model(model, filename="fraud_model.pkl"):
    """
    Persist the trained model to disk for deployment.

    Achieved:
    ---------
    Deployment-ready, reproducible model artifact.
    """
    joblib.dump(model, MODEL_DIR / filename)
    print(f"✅ Model saved at {MODEL_DIR / filename}")


# -----------------------------
# Main Training Flow
# -----------------------------
def run_model_training():
    """
    Execute full training and model selection workflow.
    """
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate and select the best
    best_name, best_model = evaluate_models(models, X_test, y_test)

    # Save the best model
    save_model(best_model, filename="fraud_model.pkl")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    run_model_training()