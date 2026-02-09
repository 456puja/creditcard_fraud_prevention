"""
pipeline.py

Purpose:
--------
End-to-end orchestration pipeline for Credit Card Fraud Detection project.

Pipeline Steps:
---------------
1. Data Preprocessing (raw → cleaned & balanced)
2. Feature Engineering (preprocessed → model-ready features)
3. Model Training (train, evaluate, select best model)
4. Save trained model & scaler for deployment

This file enables a **single-command execution** for the full ML workflow.
"""

from data_preprocessing import run_data_preprocessing
from feature_engineering import run_feature_engineering
from model import run_model_training


# -----------------------------
# Pipeline Runner
# -----------------------------
def run_full_pipeline():
    """
    Execute the full end-to-end fraud detection ML pipeline.
    """
    print("🔹 Starting Data Preprocessing...")
    run_data_preprocessing()

    print("\n🔹 Starting Feature Engineering...")
    run_feature_engineering()

    print("\n🔹 Starting Model Training...")
    run_model_training()

    print("\n✅ End-to-End Pipeline Completed Successfully!")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    run_full_pipeline()