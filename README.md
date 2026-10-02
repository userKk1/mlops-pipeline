# Customer Churn Prediction — MLOps Pipeline

An end-to-end **Machine Learning and MLOps project** for predicting customer churn from the IBM Telco Customer Churn dataset.

The project goes beyond model training by integrating **API serving, automated testing, Docker, CI/CD, and application monitoring** with Prometheus and Grafana.

---

## Project Overview

Customer churn prediction is a classification problem where the objective is to predict whether a customer is likely to leave a service.

This project implements a complete workflow:

```text
Raw Data
   ↓
Data Preprocessing
   ↓
Machine Learning Model
   ↓
Model Artifact
   ↓
FastAPI REST API
   ↓
Docker
   ↓
GitHub Actions CI/CD
   ↓
Prometheus + Grafana Monitoring
```

The main objective is not only to build a predictive model, but to understand how a machine learning model can be integrated into a reproducible and monitored application.

---

## Machine Learning

The project uses the **IBM Telco Customer Churn** dataset.

### Preprocessing

The training pipeline includes:

- Handling missing values
- Converting `TotalCharges` to numerical values
- Separating numerical and categorical features
- One-hot encoding categorical features
- Stratified train/test split
- Integrated preprocessing using Scikit-learn `Pipeline` and `ColumnTransformer`

The `customerID` field is excluded because it does not provide predictive information.

### Model

A `RandomForestClassifier` is used as the baseline classification model with:

```text
class_weight = "balanced"
random_state = 42
```

The complete preprocessing and model pipeline is saved as:

```text
model/artifacts/churn_model.pkl
```

---

## 🌐 FastAPI

The trained model is served through a REST API using **FastAPI**.

### Endpoints

#### Health / Root

```http
GET /
```

Example response:

```json
{
  "message": "Customer Churn Prediction API is running"
}
```

#### Prediction

```http
POST /predict
```

Example request:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 50.5,
  "TotalCharges": 606.0
}
```

Example response:

```json
{
  "churn": true,
  "churn_probability": 0.72
}
```

Interactive API documentation is available through FastAPI:

```text
http://localhost:8000/docs
```

---

## 🐳 Docker

The API is containerized using Docker.

Build the image:

```bash
docker build -t churn-api .
```

Run the container:

```bash
docker run -p 8000:8000 churn-api
```

The API is then available at:

```text
http://localhost:8000
```

---

## Testing

Automated tests are implemented using **pytest**.

Run the tests locally:

```bash
python -m pytest
```

The test suite covers:

- API availability
- `/predict` endpoint
- Prediction output
- Model prediction behavior
- Probability output

---

## 🔄 CI/CD

GitHub Actions is used to automate the development workflow.

The current pipeline performs:

```text
Push / Pull Request
        ↓
    Run tests
        ↓
   Build Docker image
        ↓
       CD stage
```

The CI stage automatically:

1. Checks out the repository
2. Installs Python dependencies
3. Runs the test suite
4. Builds the Docker image

The CD stage is structured as the next deployment step, allowing the pipeline to be connected to a cloud deployment platform.

---

## 📊 Monitoring

The API exposes application metrics using:

- **Prometheus**
- **Grafana**
- `prometheus-fastapi-instrumentator`

The application tracks metrics such as:

- Total predictions
- Churn predictions
- Prediction errors
- Prediction latency

Example metrics endpoint:

```text
http://localhost:8000/metrics
```

### Monitoring Architecture

```text
FastAPI
   │
   │ metrics
   ▼
Prometheus
   │
   │ queries
   ▼
Grafana
```

A Grafana dashboard is included in:

```text
monitoring/grafana/dashboard.json
```

---

## ⚙️ Running the Complete Stack

The project includes a Docker Compose configuration for running the API, Prometheus, and Grafana together.

Start the stack:

```bash
docker compose up --build
```

Services:

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| FastAPI Docs | http://localhost:8000/docs |
| Metrics | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Stop the stack:

```bash
docker compose down
```

---

## Project Structure

```text
mlops-pipeline/
│
├── app/
│   ├── main.py
│   ├── model.py
│   ├── metrics.py
│   ├── schemas.py
│   └── requirements.txt
│
├── model/
│   ├── train.py
│   ├── artifacts/
│   │   └── churn_model.pkl
│   └── requirements.txt
│
├── tests/
│   ├── test_api.py
│   └── test_model.py
│
├── scripts/
│   └── generate_predictions.py
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       └── dashboard.json
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
└── README.md
```

---

## Technologies

### Machine Learning
- Python
- Scikit-learn
- Pandas
- NumPy
- Joblib

### API
- FastAPI
- Pydantic
- Uvicorn

### MLOps / DevOps
- Docker
- Docker Compose
- Git
- GitHub
- GitHub Actions

### Monitoring
- Prometheus
- Grafana
- prometheus-fastapi-instrumentator

### Testing
- Pytest

---

## Future Improvements

Possible extensions include:

- Cloud deployment
- Infrastructure as Code with Terraform
- Model registry and experiment tracking with MLflow
- Data and model drift detection
- Automated model retraining
- Container image registry
- Production-grade authentication
- Advanced model comparison and hyperparameter optimization

---

## Learning Objectives

This project was developed to gain practical experience with:

- Building and serving machine learning models
- Designing REST APIs for ML applications
- Containerizing ML applications
- Automated testing
- CI/CD pipelines
- Application monitoring
- Prometheus metrics
- Grafana dashboards
- MLOps architecture
