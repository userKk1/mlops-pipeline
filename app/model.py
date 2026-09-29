from pathlib import Path

import joblib
import pandas as pd


# Path to the trained model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "artifacts" / "churn_model.pkl"


# Load the model once when the application starts
model = joblib.load(MODEL_PATH)


def predict(customer_data: dict) -> dict:
    """
    Make a churn prediction for one customer.
    """

    # Convert input dictionary into a DataFrame
    data = pd.DataFrame([customer_data])

    # Make prediction
    prediction = model.predict(data)[0]

    # Get probability of churn
    probability = model.predict_proba(data)[0][1]

    return {
        "churn": bool(prediction),
        "churn_probability": float(probability)
    }