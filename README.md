# 🛡️ Credit Card Fraud Detection System

## 🚀 End-to-End Machine Learning Project (FastAPI | Docker)

Welcome to this project!
In this repository, we build a Fraud Detection Model for a financial and e-commerce platforms using machine learning.

📌 Project Goal

Develop a production-ready system that detects fraudulent credit card transactions in real-time, minimizes false positives, and provides a scalable API for integration with financial and e-commerce platforms.


🛑 Problem Statement

Credit card fraud causes significant financial losses and damages customer trust. Traditional rule-based detection systems are often slow, rigid, and prone to false positives, making them inefficient for real-time transactions. There is a need for a machine learning-based system that can accurately identify fraudulent transactions in real-time, reduce false positives, and support scalable deployment for financial and e-commerce platforms.


### 📌 Project Overview

This project implements a production-ready Credit Card Fraud Detection system using machine learning best practices. It covers the complete ML lifecycle—from feature engineering and model training to scalable inference and deployment—designed for real-world payment fraud prevention systems.

This project uses the Credit Card Fraud Detection dataset, which contains anonymized transaction features and highly imbalanced fraud labels.

* Source: Kaggle
* Dataset Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
* Records: 284,807 transactions
* Fraud Rate: ~0.17%
* Features: PCA-transformed features (V1–V28), Amount, Time


### 🎯 Business Objective

* Detect fraudulent credit card transactions with high recall
* Minimize false positives to reduce customer friction
* Provide a scalable and deployable ML inference service
* Ensure consistency between training and inference using pipelines


### 🧠 Machine Learning Solution Highlights

* Advanced feature engineering on transaction data
* Explicit handling of severe class imbalance
* Baseline and ensemble model comparison
* End-to-end Scikit-learn pipeline for inference consistency
* FastAPI-based REST inference service
* Fully Dockerized deployment setup


### 🗂️ Project Structure

<pre>
CreditCard_Fraud_Prevention/
├── app/
│   ├── main.py                       # FastAPI application entrypoint
│   ├── schemas.py                    # Request and response schemas
│   └── utils.py                      # Utility and helper functions
├── data/
│   ├── raw/                          # Original raw datasets (ignored via .gitignore)
│   └── processed/                    # Preprocessed data used for modeling (ignored via .gitignore)
├── models/
│   ├── fraud_model.pkl               # Final trained Random Forest model
│   ├── scaler.pkl                    # Feature scaler
│   └── pipeline.pkl                  # End-to-end inference pipeline
├── src/
│   ├── data_preprocessing.py         # Data cleaning and preprocessing logic
│   ├── feature_engineering.py        # Feature engineering transformations
│   ├── model.py                      # Model training and selection logic
│   ├── pipeline.py                   # Training and inference pipeline creation
│   └── utils.py                      # Shared utility functions
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_data_preprocessing.ipynb   # Data cleaning and preprocessing
│   ├── 03_feature_engineering.ipynb  # Feature engineering and transformations
│   ├── 04_model_training.ipynb       # Model training and pipeline creation
│   └── 05_model_evaluation.ipynb     # Model evaluation and threshold analysis
├── requirements.txt                  # Project dependencies
├── Dockerfile                        # Docker configuration
├── .gitignore                        # Git ignore rules
└── README.md                         # Project documentation
</pre>


### ⚙️ Tech Stack & Purpose

* Python 3.10 – Core language for ML workflows and API development
* pandas, NumPy – Data preprocessing, feature engineering and numerical operations
* scikit-learn – Model training, evaluation, pipelines (Random Forest)
* imbalanced-learn – Handling extreme class imbalance
* Logistic Regression – Baseline model
* Random Forest, XGBoost – Candidate models
* Random Forest (Final Model) – Selected based on evaluation metrics
* Matplotlib, Seaborn - Visualization of model performance including ROC curves, Precision–Recall curves, and evaluation plots.
* FastAPI – Real-time inference API
* Uvicorn – ASGI server for serving the FastAPI application
* Pydantic – Request and response data validation for API endpoints
* Joblib – Model and pipeline serialization
* Docker – Containerized deployment
* Jupyter Notebook / PyCharm – Experimentation and development


### 🔬 Machine Learning Workflow

* Feature-engineered data loading and preprocessing
* Handling class imbalance using class-weighted learning
* Baseline model training (Logistic Regression)
* Advanced model training (Random Forest, XGBoost)
* Model evaluation and selection using imbalance-aware metrics
* Pipeline creation (preprocessing + model)
* Model and pipeline serialization
* API-based inference using FastAPI
* Containerized deployment using Docker


### 📦 Model Artifacts

* fraud_model.pkl → Final trained classification model

* scaler.pkl → Feature scaling object

* pipeline.pkl → End-to-end inference pipeline (recommended for production)


### 🚀 Running the API Locally

1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Activate virtual environment
venv\Scripts\activate

3️⃣ Run the FastAPI application
uvicorn app.main:app --reload

4️⃣ Open Swagger UI
http://127.0.0.1:8000/docs


### 🐳 Deployment Using Docker

This project uses Docker to containerize the FastAPI-based Machine Learning inference service, ensuring environment consistency, portability, and production readiness.

#### Prerequisites
- Docker installed and running on the system

#### Build the Docker Image
Navigate to the project root directory and run:
docker build -t credit-card-fraud-api .

#### Run the Docker Container
docker run -d -p 8000:8000 credit-card-fraud-api

#### Access the Application
* Swagger UI:http://127.0.0.1:8000/docs
* API Base URL:http://127.0.0.1:8000


### 🧩 Key Design Considerations

* ROC-AUC and Precision-Recall metrics prioritized over accuracy
* Threshold tuning aligned with fraud business objectives
* Pipeline-based inference to avoid training-serving skew
* Modular architecture for easy extensibility

### 📈 Future Improvements

* Real-time streaming integration (Kafka / PubSub)
* Model monitoring and drift detection
* Advanced threshold optimization using cost-based metrics
* CI/CD pipeline for automated retraining and deployment


### 🏁 Conclusion

This project demonstrates a complete, industry-aligned ML system for fraud detection, combining robust modeling practices with production-ready deployment. It reflects real-world ML engineering workflows and is suitable as a portfolio-grade project for machine learning and data engineering roles.


### ⭐ Thank you for visiting this project!