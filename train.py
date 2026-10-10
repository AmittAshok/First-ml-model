



from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ARTIFACTS_DIR = Path("artifacts")

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Create artifacts directory if it doesn't exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Prepare training data
# --------------------------------------------------

# Example: predict a score based on study hours
X = np.array([[i] for i in range(1, 11)])
y = np.array([35, 45, 55, 65, 75, 85, 95, 105, 115, 125])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# --------------------------------------------------
# 3. Train the model
# --------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training completed.")


# --------------------------------------------------
# 4. Evaluate the model
# --------------------------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = float(np.sqrt(mse))
r2 = r2_score(y_test, y_pred)

metrics = {
    "mae": float(mae),
    "mse": float(mse),
    "rmse": float(rmse),
    "r2_score": float(r2),
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
}

# Save evaluation metrics
with METRICS_PATH.open("w") as file:
    json.dump(metrics, file, indent=4)

print("\nEvaluation metrics:")
print(json.dumps(metrics, indent=4))


# --------------------------------------------------
# 5. Save the trained model
# --------------------------------------------------

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")


# --------------------------------------------------
# 6. Save test predictions
# --------------------------------------------------

predictions_df = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred,
    "error": y_test - y_pred,
})

predictions_df.to_csv(PREDICTIONS_PATH, index=False)

print(f"Predictions saved to: {PREDICTIONS_PATH}")


# --------------------------------------------------
# 7. Save model metadata
# --------------------------------------------------

metadata = {
    "model_name": "LinearRegression",
    "target": "score",
    "features": ["study_hours"],
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "intercept": float(model.intercept_),
    "coefficient": float(model.coef_[0]),
    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    "model_file": str(MODEL_PATH),
}

with METADATA_PATH.open("w") as file:
    json.dump(metadata, file, indent=4)

print(f"Metadata saved to: {METADATA_PATH}")


# --------------------------------------------------
# 8. Make new predictions
# --------------------------------------------------

new_hours = np.array([[11], [12], [13]])
new_predictions = model.predict(new_hours)

print("\nNew predictions:")

for hours, prediction in zip(
    new_hours.flatten(),
    new_predictions,
):
    print(f"Study hours: {hours}, Predicted score: {prediction:.2f}")

print("\nAll artifacts saved successfully.")

