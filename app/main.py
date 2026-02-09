from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import TransactionInput, TransactionResponse
from app.model_inference import run_inference

app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthcheck")
def healthcheck():
    return {"status": "API is running"}

@app.post("/predict", response_model=TransactionResponse)
def predict(input_data: TransactionInput):

    input_df = input_data.to_dataframe()

    result = model_inference.run_inference(input_df)

    return result





