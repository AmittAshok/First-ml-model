import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

test_df = pd.read_csv("data/test.csv")
model = joblib.load("artifacts/model.pkl")

X_test = test_df[["hours_studied"]]
y_test = test_df["score"]

predictions = model.predict(X_test)

metrics = {
    "mae": float(mean_absolute_error(y_test, predictions)),
    "mse": float(mean_squared_error(y_test, predictions)),
    "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
    "r2_score": float(r2_score(y_test, predictions)),
    "testing_samples": int(len(test_df)),
}

artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(parents=True, exist_ok=True)

with open(artifacts_dir / "metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

predictions_df = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": predictions,
    "error": y_test.to_numpy() - predictions,
})

predictions_df.to_csv(
    artifacts_dir / "predictions.csv",
    index=False,
)

print("Evaluation completed.")
print(json.dumps(metrics, indent=4))
