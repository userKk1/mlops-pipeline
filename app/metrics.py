from prometheus_client import Counter, Histogram


predictions_total = Counter(
    "predictions_total",
    "Total number of churn predictions"
)

churn_predictions_total = Counter(
    "churn_predictions_total",
    "Total number of customers predicted to churn"
)

prediction_errors_total = Counter(
    "prediction_errors_total",
    "Total number of prediction errors"
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Time spent making a churn prediction"
)