from fastapi import FastAPI
import joblib

app = FastAPI()

# Load the trained model
model = joblib.load("model.pkl")


@app.get("/")
def home():
    return {"message": "ML model API is running"}


@app.get("/predict")
def predict(hours: float):
    prediction = model.predict([[hours]])

    return {
        "hours_studied": hours,
        "predicted_score": float(prediction[0])
    }
