from fastapi import FastAPI

from app.model import predict
from app.schemas import CustomerData, PredictionResponse


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    result = predict(customer.model_dump())
    return result