from pathlib import Path

import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

train_df = pd.read_csv("data/train.csv")

X_train = train_df[["hours_studied"]]
y_train = train_df["score"]

model = LinearRegression()
model.fit(X_train, y_train)

model_path = Path("artifacts/model.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_path)

print(f"Model saved to {model_path}")
print("Model training completed.")
