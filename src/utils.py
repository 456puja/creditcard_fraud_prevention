import pickle
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def load_model():
    with open(os.path.join(MODEL_DIR, "fraud_model.pkl"), "rb") as f:
        return pickle.load(f)

def load_scaler():
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        return pickle.load(f)

def load_pipeline():
    with open(os.path.join(MODEL_DIR, "pipeline.pkl"), "rb") as f:
        return pickle.load(f)

def predict_fraud(input_df, model, scaler, threshold=0.5):
    X_scaled = scaler.transform(input_df)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities
