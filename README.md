# First ML Model — End-to-End ML Project

A beginner-friendly Machine Learning project that uses **Linear Regression** to predict scores based on study hours. This project demonstrates model training, evaluation, artifact management, and serving predictions through a REST API using FastAPI.

## Project Overview

This project covers the basic workflow of deploying a machine learning model:

1. Prepare the training dataset.
2. Train a Linear Regression model.
3. Evaluate the model using regression metrics.
4. Save the trained model and evaluation artifacts.
5. Serve predictions through a FastAPI application.
6. Test the API using a browser or Swagger UI.

## Tech Stack

- **Python** — Programming language
- **Scikit-learn** — Model training and evaluation
- **NumPy** — Numerical operations
- **Pandas** — Prediction results and CSV handling
- **Joblib** — Model serialization
- **FastAPI** — REST API development
- **Uvicorn** — ASGI server for running the API
- **Git and GitHub** — Source code version control

## Project Structure

```text
First-ml-model/
├── artifacts/
│   ├── model.pkl
│   ├── metrics.json
│   ├── metadata.json
│   └── predictions.csv
├── .venv/
├── app.py
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

### File Description

| File | Description |
|---|---|
| `train.py` | Trains and evaluates the Linear Regression model |
| `app.py` | Exposes model predictions through a REST API |
| `requirements.txt` | Lists Python dependencies |
| `artifacts/model.pkl` | Serialized trained model |
| `artifacts/metrics.json` | Model evaluation metrics |
| `artifacts/metadata.json` | Model configuration and training information |
| `artifacts/predictions.csv` | Actual and predicted values for the test dataset |
| `.gitignore` | Excludes files that should not be tracked by Git |

The `.venv/` directory is created locally and should not be committed to GitHub.

## Prerequisites

Install the following before running the project:

- Python 3
- pip
- Git
- Ubuntu on WSL or another compatible Linux environment

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/AmittAshok/First-ml-model.git
cd First-ml-model
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

If Python reports that the `venv` module is unavailable on Ubuntu, install the appropriate package for your Python version. For example:

```bash
sudo apt update
sudo apt install python3-venv
```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

If you have not yet added all required dependencies to `requirements.txt`, use:

```text
scikit-learn
joblib
numpy
pandas
fastapi
uvicorn[standard]
```

## Train the Machine Learning Model

Run the training script:

```bash
python train.py
```

The script prepares the dataset, splits it into training and testing sets, trains the model, evaluates its performance, and saves the generated artifacts.

The `artifacts/` directory is created automatically if it does not exist.

### Generated Artifacts

- **model.pkl:** Trained Linear Regression model.
- **metrics.json:** Evaluation metrics such as MAE, MSE, RMSE, and R².
- **metadata.json:** Model configuration, feature information, and training timestamp.
- **predictions.csv:** Actual values, predicted values, and prediction errors for the test dataset.

To inspect the metrics:

```bash
cat artifacts/metrics.json
```

To list the artifacts:

```bash
ls -lh artifacts/
```

## Run the FastAPI Application

Start the API server:

```bash
uvicorn app:app --reload
```

The application will be available at:

```text
http://127.0.0.1:8000
```

The `--reload` option automatically restarts the development server when the application code changes. It is intended for development, not production.

## API Endpoints

### 1. Health Check

**Endpoint:** `GET /`

URL:

```text
http://127.0.0.1:8000/
```

Example response:

```json
{
  "message": "ML Model API is running",
  "model": "LinearRegression",
  "status": "healthy"
}
```

### 2. Make a Prediction

**Endpoint:** `GET /predict?hours=12`

URL:

```text
http://127.0.0.1:8000/predict?hours=12
```

This sends `12` as the number of study hours to the trained model.

Example response:

```json
{
  "hours_studied": 12,
  "predicted_score": 155.0
}
```

The predicted value is illustrative and depends on the model and training data used.

The current API validates study hours to be greater than zero and no greater than 24.

### 3. Retrieve Model Metrics

**Endpoint:** `GET /metrics`

URL:

```text
http://127.0.0.1:8000/metrics
```

Returns the evaluation metrics saved by the training script.

### 4. Interactive API Documentation

**Endpoint:** `GET /docs`

URL:

```text
http://127.0.0.1:8000/docs
```

Swagger UI allows you to explore endpoints, provide input values, and execute requests interactively.

## Model Evaluation Metrics

The training script calculates the following regression metrics:

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error; lower is generally better |
| MSE | Mean Squared Error; penalizes larger errors more heavily |
| RMSE | Root Mean Squared Error, expressed in the target's units |
| R² | Coefficient of determination; measures how well predictions explain variation in the target |

Metrics should be evaluated on data that was not used to train the model.

**Note:** This project uses a small synthetic dataset for learning purposes. Its results should not be interpreted as evidence of real-world predictive performance.

## Development Workflow

```text
Dataset
   |
   v
Data Preparation
   |
   v
Train/Test Split
   |
   v
Model Training
   |
   v
Model Evaluation
   |
   v
Save Model and Metrics
   |
   v
FastAPI Application
   |
   v
Prediction API
```

## Future Enhancements

Planned improvements as the project evolves:

- [ ] Add data validation and preprocessing.
- [ ] Introduce experiment tracking with MLflow.
- [ ] Implement data and model versioning with DVC.
- [ ] Add automated unit and API tests.
- [ ] Containerize the application using Docker.
- [ ] Implement CI/CD using GitHub Actions.
- [ ] Deploy the application to AWS.
- [ ] Add monitoring for model performance and API health.
- [ ] Explore model retraining and version management.

## Learning Objectives

Through this project, I am learning how to:

- Build and evaluate a basic machine learning model.
- Manage model artifacts and evaluation metrics.
- Expose an ML model through a REST API.
- Use virtual environments for dependency isolation.
- Apply software engineering and MLOps practices to ML projects.

## Author

**Amitt Ashok**

GitHub: [AmittAshok](https://github.com/AmittAshok)

Repository: [First-ml-model](https://github.com/AmittAshok/First-ml-model)

---

*This project is part of my hands-on journey into Machine Learning and MLOps.*
