from pathlib import Path
import time

import joblib
import pandas as pd

from app.metrics import (
    predictions_total,
    churn_predictions_total,
    prediction_errors_total,
    prediction_latency,
)

# Path to the trained model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "artifacts" / "churn_model.pkl"


# Load the model once when the application starts
model = joblib.load(MODEL_PATH)


def predict(customer_data: dict) -> dict:
    """
    Make a churn prediction for one customer.
    """
    start_time = time.time()

    try:
        data = pd.DataFrame([customer_data])

        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]

        predictions_total.inc()

        if prediction == 1:
            churn_predictions_total.inc()

        return {
            "churn": bool(prediction),
            "churn_probability": float(probability)
        }

    except Exception:
        prediction_errors_total.inc()
        raise

    finally:
        prediction_latency.observe(time.time() - start_time)