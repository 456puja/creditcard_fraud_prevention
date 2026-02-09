import pandas as pd
from src.utils import load_model, load_scaler, predict_fraud

# Load artifacts once
MODEL = load_model()
SCALER = load_scaler()

def run_inference(input_df: pd.DataFrame, threshold: float = 0.5):
    predictions, probabilities = predict_fraud(
        input_df=input_df,
        model=MODEL,
        scaler=SCALER,
        threshold=threshold
    )

    return {
        "prediction": predictions.tolist(),
        "probability": probabilities.tolist()
    }

